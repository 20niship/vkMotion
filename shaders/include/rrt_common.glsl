// GPU RRT-Connect(rrt_advance.comp / rrt_gpu.comp)の共通部: バッファ・push constant・ノードの詰め方。
// 木はデバイスメモリ上の大きな配列で、同じ問題を担当する全ワープが共有する(ラウンド=dispatchごとに、他ワープの追加分はスナップショット越しにしか見えない)。
// 複数の問題(それぞれ別の世界・start・goal)を1回のdispatchでまとめて解く(バッチ)。問題番号はdispatchのworkgroup IDで渡す。
// 問題ごとに params / ctl / 木 / 結果 / 世界(箱・球) を別の領域に持つ。

layout(std430, binding = 0) readonly buffer Params { float prm[]; };  // 問題ごとに PRM_STRIDE 個: lower[8], upper[8], reach[8], start[8], goals[8*8], 以降は整数のビット列(PRM_NGOALS..)
layout(std430, binding = 11) buffer Result { uint res[]; };           // 問題ごとの勝者の経路: [0]=点数 [1]=ラウンド, [4+k*8+j]=点k関節j
layout(std430, binding = 12) buffer Ctl { uint ctl[]; };              // 問題ごとの制御ブロック(CTL_STRIDE個、下のCTL_*)
layout(std430, binding = 13) buffer Tree { uvec4 nd[]; };             // 問題pの木0(start側)が[(2p)CAP,(2p+1)CAP)、木1(goal側)がその次のCAP個

layout(push_constant) uniform PushConstants {
  uint num_configs_unused;
  uint dof;
  uint num_steps;
  uint num_spheres;
  uint num_clusters;
  uint num_groups;
  uint num_pairs;
  uint num_boxes;
  uint num_balls;
  uint num_links;
  float margin;
  float range;
  float motion_step;
  uint num_goals;
  uint seed;
  uint gen;    // dispatchの世代(planごとに増える。ctl[CTL_SOLVED]>=genなら解けた)
  uint g0;     // 最初のラウンドのワープ数
  uint gmax;   // ワープ数の上限
  uint grow;   // 何ラウンドごとにワープ数を倍にするか
  uint iters;  // 1ラウンドで各ワープが行うRRT-Connect反復の数
  uint tree_cap;  // 1本の木のノード数上限(問題数が多いときは小さくしてメモリに収める)
}
pc;

const uint MAXP       = 128;   // 返す経路の最大点数
const uint CTL_CNT0   = 0;     // 木0のノード数(append用アトミックカウンタ)
const uint CTL_CNT1   = 1;
const uint CTL_SNAP0  = 2;     // ラウンド開始時点のノード数(これ未満だけ読んでよい)
const uint CTL_SNAP1  = 3;
const uint CTL_SOLVED = 4;     // 最後に解けた世代
const uint CTL_WIN    = 5;     // 勝ったワープ
const uint CTL_ROUND  = 6;     // 次のラウンド番号
const uint CTL_MARGIN = 7;     // 現在の余裕(floatのビット列)。木が伸びないとき 1/4 -> 0 と縮める
const uint CTL_STAGE  = 12;    // 余裕を最後に縮めたラウンド
const uint CTL_ARGS   = 8;     // 次のラウンドでこの問題に使うワープ数
const uint CTL_STRIDE = 16;
const uint PRM_STRIDE = 104;
const uint PRM_NGOALS = 96;    // 以降は整数のビット列
const uint PRM_NBOXES = 97;
const uint PRM_NBALLS = 98;
const uint PRM_SEED   = 99;
const uint MAXB       = 32;    // 問題あたりの箱の上限(boxes[]は6 vec4/箱)
const uint MAXBALL    = 8;     // 問題あたりの球の上限(balls[]は2 vec4/球)
const uint RES_STRIDE = 4u + MAXP * 8u;

uint g_pid, g_pb, g_cb;  // 担当する問題の番号 / params / ctl の先頭
void init_problem(uint pid) {
  g_pid = pid;
  g_pb  = pid * PRM_STRIDE;
  g_cb  = pid * CTL_STRIDE;
}
#define PRM(i) prm[g_pb + (i)]
#define CTL(i) ctl[g_cb + (i)]
#define CAP pc.tree_cap
#define CAP_SOFT (pc.tree_cap - pc.tree_cap / 8u)  // これを超えたらラウンド境界で作り直す(根だけ残す)
#define NODE(t, i) nd[(g_pid * 2u + (t)) * CAP + (i)]

float qlo[8], qstep[8], qinv[8];
void init_quant() {
  for(uint j = 0; j < 8u; j++) {
    float lo = PRM(j), hi = PRM(8u + j);
    qlo[j]   = lo;
    qstep[j] = (hi - lo) / 65535.0;
    qinv[j]  = hi > lo ? 1.0 / (hi - lo) : 0.0;
  }
}
uint enc(float v, uint j) { return uint(clamp((v - qlo[j]) * qinv[j] * 65535.0 + 0.5, 0.0, 65535.0)); }
float dec(uint u, uint j) { return qlo[j] + float(u) * qstep[j]; }
// 量子化して戻す(木に入れる値と辺の検査に使う値を一致させる)
void quantize(inout float q[8]) {
  for(uint j = 0; j < 7u; j++) q[j] = dec(enc(q[j], j), j);
}

// 16B/ノード: 関節16bit x7 + (親index+2)<<16。親フィールド0=未使用
uvec4 pack_node(float q[8], int parent) {
  uvec4 w;
  w.x = enc(q[0], 0u) | (enc(q[1], 1u) << 16);
  w.y = enc(q[2], 2u) | (enc(q[3], 3u) << 16);
  w.z = enc(q[4], 4u) | (enc(q[5], 5u) << 16);
  w.w = enc(q[6], 6u) | (uint(parent + 2) << 16);
  return w;
}
void unpack_node(uvec4 w, out float q[8]) {
  q[0] = dec(w.x & 0xFFFFu, 0u);
  q[1] = dec(w.x >> 16, 1u);
  q[2] = dec(w.y & 0xFFFFu, 2u);
  q[3] = dec(w.y >> 16, 3u);
  q[4] = dec(w.z & 0xFFFFu, 4u);
  q[5] = dec(w.z >> 16, 5u);
  q[6] = dec(w.w & 0xFFFFu, 6u);
  q[7] = 0.0;
}
