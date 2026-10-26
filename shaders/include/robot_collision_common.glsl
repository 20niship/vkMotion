// 衝突判定の共通部。robot_validate.comp / rrt_gpu.comp から #include する。
// 呼び出し側が先に push_constant ブロック `pc` (num_links, num_steps, num_spheres, num_clusters, num_groups, num_pairs, num_boxes, num_balls, margin を含む) と
// binding 0 / 11以降を宣言しておくこと。関節角はグローバルの Q[] に入れて config_free() を呼ぶ。
// 判定は vkmotion/fast_collision.cpp の free_scalar / mask8_impl と同じ(球半径+margin <= 距離 で衝突)。変更時は両方そろえること。
// 直方体の距離は「外側距離^2 <= (r+margin)^2」で判定する(内部はmax(q,0)=0なので常に衝突になる)。

// 上限はshared memoryに写す都合(FastCollision::gpu_compatible / pack_gpu_robot と同じ値にそろえること)
const int MAX_LINKS = 32;
const int MAX_SPHERES = 96;
const int MAX_CLUSTERS = 40;
const int MAX_GROUPS = 16;
const int MAX_DOF = 16;

layout(std430, binding = 1) readonly buffer StepMeta { ivec4 step_meta[]; };    // parent, link, type, active
layout(std430, binding = 2) readonly buffer StepData { vec4 step_data[]; };      // 4 vec4 / step: axis, ro row0+to.x, row1+to.y, row2+to.z
layout(std430, binding = 3) readonly buffer Spheres { vec4 spheres[]; };         // local x,y,z,r
layout(std430, binding = 4) readonly buffer ClusterMeta { ivec4 cluster_meta[]; };  // link, sphere_begin, count, group
layout(std430, binding = 5) readonly buffer ClusterBs { vec4 cluster_bs[]; };
layout(std430, binding = 6) readonly buffer GroupMeta { ivec4 group_meta[]; };   // first_cluster, count, link, 0
layout(std430, binding = 7) readonly buffer GroupBs { vec4 group_bs[]; };
layout(std430, binding = 8) readonly buffer Pairs { int pairs[]; };               // グループ(リンク)ごとのビットマスク: bit h = グループgとh(>g)の自己衝突を見る
layout(std430, binding = 9) readonly buffer Boxes { vec4 boxes[]; };             // 6 vec4 / box
layout(std430, binding = 10) readonly buffer Balls { vec4 balls[]; };            // 2 vec4 / ball

// 余裕(margin)は MARGIN で参照する。呼び出し側が定義しなければ push constant の margin。GPU探索では木の伸び具合に応じてラウンド途中で縮めるので変数にしている。
#ifndef MARGIN
#define MARGIN pc.margin
#endif

// ロボット/世界のデータの置き場所。
//   ROBOT_SHARED を定義した場合: ワークグループ開始時にshared memoryへ写して使う(load_robot_shared を全レーンから呼び、続けて barrier() すること。早期returnより前)。
//     上限(MAX_*)を超える場合はホスト側で弾く(FastCollision::gpu_compatible)。
//   定義しない場合: ストレージバッファを直接読む(同じアドレスを全レーンが読むのでキャッシュに乗る。shared memoryを木などに使いたいときや、占有率を優先するとき)。
#ifdef ROBOT_SHARED
const int MAX_STEPS = 24;
const int MAX_BOXES = 32;
const int MAX_BALLS = 8;
shared ivec4 s_step_meta[MAX_STEPS];
shared vec4 s_step_data[MAX_STEPS * 4];
shared vec4 s_spheres[MAX_SPHERES];
shared ivec4 s_cluster_meta[MAX_CLUSTERS];
shared vec4 s_cluster_bs[MAX_CLUSTERS];
shared ivec4 s_group_meta[MAX_GROUPS];
shared vec4 s_group_bs[MAX_GROUPS];
shared int s_pairs[MAX_GROUPS];
shared vec4 s_boxes[MAX_BOXES * 6];
shared vec4 s_balls[MAX_BALLS * 2];

void load_robot_shared(uint lane, uint lanes) {
  for(uint i = lane; i < pc.num_steps; i += lanes) {
    s_step_meta[i] = step_meta[i];
    for(uint k = 0; k < 4u; k++) s_step_data[i * 4u + k] = step_data[i * 4u + k];
  }
  for(uint i = lane; i < pc.num_clusters; i += lanes) {
    s_cluster_meta[i] = cluster_meta[i];
    s_cluster_bs[i] = cluster_bs[i];
  }
  for(uint i = lane; i < pc.num_spheres; i += lanes) s_spheres[i] = spheres[i];
  for(uint i = lane; i < pc.num_groups; i += lanes) {
    s_group_meta[i] = group_meta[i];
    s_group_bs[i] = group_bs[i];
  }
  for(uint i = lane; i < pc.num_groups; i += lanes) s_pairs[i] = pairs[i];
  for(uint i = lane; i < pc.num_boxes * 6u; i += lanes) s_boxes[i] = boxes[i];
  for(uint i = lane; i < pc.num_balls * 2u; i += lanes) s_balls[i] = balls[i];
}

#else
#define s_step_meta step_meta
#define s_step_data step_data
#define s_spheres spheres
#define s_cluster_meta cluster_meta
#define s_cluster_bs cluster_bs
#define s_group_meta group_meta
#define s_group_bs group_bs
#define s_pairs pairs
#define s_boxes boxes
#define s_balls balls
void load_robot_shared(uint lane, uint lanes) {}
#endif

float Q[MAX_DOF];
float R[MAX_LINKS * 9];     // リンクの姿勢(分岐する親を引くための保存用。直列に連なる子は直前の値をレジスタで受け取る)
vec3 P[MAX_LINKS];
vec4 sph[MAX_SPHERES];     // 球の世界座標 x,y,z と半径
vec3 cbs[MAX_CLUSTERS];    // クラスタ境界球の中心(世界)
vec3 gbs[MAX_GROUPS];      // リンク(グループ)境界球の中心(世界)

// 精度はCPU側のsincos8(cephes)と同じ。GLSLの組み込みsin/cosは引数が大きい場合の縮約まで含むので1回が数千サイクルかかる。
void sincos_fast(float x, out float sn, out float cs) {
  float ax = abs(x);
  int j    = (int(ax * 1.27323954473516) + 1) & ~1;
  float y  = float(j);
  float r  = ((ax - y * 0.78515625) - y * 2.4187564849853515625e-4) - y * 3.77489497744594108e-8;
  float z  = r * r;
  float sp = ((-1.9515295891E-4 * z + 8.3321608736E-3) * z - 1.6666654611E-1) * z * r + r;
  float cp = ((2.443315711809948E-005 * z - 1.388731625493765E-003) * z + 4.166664568298827E-002) * z * z - 0.5 * z + 1.0;
  bool swp = (j & 2) == 0;
  float sv = swp ? sp : cp;
  float cv = swp ? cp : sp;
  bool sflip = ((j & 4) != 0) != (x < 0.0);
  bool cflip = ((j & 4) != 0) != ((j & 2) != 0);
  sn = sflip ? -sv : sv;
  cs = cflip ? -cv : cv;
}

vec3 rot_apply(float r[9], vec3 v) {
  return vec3(r[0] * v.x + r[1] * v.y + r[2] * v.z, r[3] * v.x + r[4] * v.y + r[5] * v.z, r[6] * v.x + r[7] * v.y + r[8] * v.z);
}

// リンクの姿勢(r,p)が確定した直後に、そのリンクに属する境界球・衝突球の世界座標を計算する(姿勢がレジスタにあるうちに)。
// group = そのリンクのグループ番号(無ければ負)。ステップデータのaxis.wに入っている。
void place_group(int g, float r[9], vec3 p) {
  if(g < 0) return;
  ivec4 gm = s_group_meta[g];
  gbs[g]   = rot_apply(r, s_group_bs[g].xyz) + p;
  for(int c = gm.x; c < gm.x + gm.y; c++) {
    cbs[c]    = rot_apply(r, s_cluster_bs[c].xyz) + p;
    ivec4 cm = s_cluster_meta[c];
    for(int i = cm.y; i < cm.y + cm.z; i++) {
      vec4 sp = s_spheres[i];
      sph[i]  = vec4(rot_apply(r, sp.xyz) + p, sp.w);
    }
  }
}

float box_sumsq(uint b, vec3 p) {
  vec3 d = p - s_boxes[b].xyz;
  vec3 l = vec3(dot(s_boxes[b + 1].xyz, d), dot(s_boxes[b + 2].xyz, d), dot(s_boxes[b + 3].xyz, d));
  vec3 q = max(abs(l) - s_boxes[b + 4].xyz, vec3(0.0));
  return dot(q, q);
}

// ---- ブロードフェーズで候補が出たときだけ呼ぶ詳細判定(動的ループ)。true なら衝突 ----

// グループgと直方体boの衝突(クラスタ境界球 -> 球)
bool box_group_hit(uint bo, int g) {
  ivec4 gm = s_group_meta[g];
  for(int c = gm.x; c < gm.x + gm.y; c++) {
    float rc = s_cluster_bs[c].w + MARGIN;
    if(box_sumsq(bo, cbs[c]) > rc * rc) continue;
    ivec4 cm = s_cluster_meta[c];
    for(int i = cm.y; i < cm.y + cm.z; i++) {
      float r = sph[i].w + MARGIN;
      if(box_sumsq(bo, sph[i].xyz) <= r * r) return true;
    }
  }
  return false;
}

// クラスタcと直方体boの球単位の衝突(クラスタ境界球を通過した後に呼ぶ)
bool box_cluster_hit(uint bo, int c) {
  ivec4 cm = s_cluster_meta[c];
  for(int i = cm.y; i < cm.y + cm.z; i++) {
    float r = sph[i].w + MARGIN;
    if(box_sumsq(bo, sph[i].xyz) <= r * r) return true;
  }
  return false;
}

// クラスタa,bの球単位の自己衝突(クラスタ境界球を通過した後に呼ぶ)
bool cluster_pair_hit(int ca, int cb) {
  ivec4 cma = s_cluster_meta[ca], cmb = s_cluster_meta[cb];
  for(int i = cma.y; i < cma.y + cma.z; i++) {
    for(int j = cmb.y; j < cmb.y + cmb.z; j++) {
      vec3 e   = sph[i].xyz - sph[j].xyz;
      float rs = sph[i].w + sph[j].w + 2.0 * MARGIN;
      if(dot(e, e) <= rs * rs) return true;
    }
  }
  return false;
}

// グループgとh(別リンク)の自己衝突(クラスタ境界球 -> 球)
bool group_pair_hit(int g, int h) {
  ivec4 ga = s_group_meta[g], gh = s_group_meta[h];
  for(int ca = ga.x; ca < ga.x + ga.y; ca++) {
    for(int cb = gh.x; cb < gh.x + gh.y; cb++) {
      vec3 d   = cbs[ca] - cbs[cb];
      float rr = s_cluster_bs[ca].w + s_cluster_bs[cb].w + 2.0 * MARGIN;
      if(dot(d, d) > rr * rr) continue;
      ivec4 cma = s_cluster_meta[ca], cmb = s_cluster_meta[cb];
      for(int i = cma.y; i < cma.y + cma.z; i++) {
        for(int j = cmb.y; j < cmb.y + cmb.z; j++) {
          vec3 e   = sph[i].xyz - sph[j].xyz;
          float rs = sph[i].w + sph[j].w + 2.0 * MARGIN;
          if(dot(e, e) <= rs * rs) return true;
        }
      }
    }
  }
  return false;
}

// 球障害物(数が少ないのでブロードフェーズ無し)
bool balls_hit() {
  for(uint b = 0; b < pc.num_balls; b++) {
    vec4 ball = s_balls[b * 2];
    uint allow = floatBitsToUint(s_balls[b * 2 + 1].x);
    for(uint c = 0; c < pc.num_clusters; c++) {
      ivec4 cm = s_cluster_meta[c];
      if(((allow >> uint(cm.w)) & 1u) != 0u) continue;
      for(int i = cm.y; i < cm.y + cm.z; i++) {
        vec3 d = sph[i].xyz - ball.xyz;
        float r = sph[i].w + ball.w + MARGIN;
        if(dot(d, d) <= r * r) return true;
      }
    }
  }
  return false;
}

#ifdef ROBOT_GENERATED
// ロボット専用に生成したFK+ブロードフェーズ(vkmotion/gpu_codegen.cpp)。ロボットの定数をリテラルに展開し、リンク姿勢を静的インデックスのレジスタに置く。
#include "robot_generated.glsl"
#else
// ステップは親が先に来る順。type: 0=固定, 1=回転, 2=直動, 3=ルート(単位姿勢で開始)
void fk() {
  float cr[9] = float[9](1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
  vec3 cp     = vec3(0.0);
  int cur     = -1;  // cr,cp が表すリンク
  for(uint s = 0; s < pc.num_steps; s++) {
    ivec4 m  = s_step_meta[s];
    vec4 axv = s_step_data[s * 4 + 0];
    int grp  = floatBitsToInt(axv.w);
    if(m.z == 3) {
      for(int k = 0; k < 9; k++) R[m.y * 9 + k] = (k % 4 == 0) ? 1.0 : 0.0;
      P[m.y] = vec3(0.0);
      for(int k = 0; k < 9; k++) cr[k] = (k % 4 == 0) ? 1.0 : 0.0;
      cp  = vec3(0.0);
      cur = m.y;
      place_group(grp, cr, cp);
      continue;
    }
    vec4 r0 = s_step_data[s * 4 + 1];
    vec4 r1 = s_step_data[s * 4 + 2];
    vec4 r2 = s_step_data[s * 4 + 3];
    float ro[9] = float[9](r0.x, r0.y, r0.z, r1.x, r1.y, r1.z, r2.x, r2.y, r2.z);
    vec3 t = vec3(r0.w, r1.w, r2.w);

    float mm[9];
    if(m.z == 1) {
      float sn, cs;
      sincos_fast(Q[m.w], sn, cs);
      vec3 a = axv.xyz;
      float tt = 1.0 - cs;
      float rj[9] = float[9](cs + a.x * a.x * tt, a.x * a.y * tt - a.z * sn, a.x * a.z * tt + a.y * sn,
                             a.y * a.x * tt + a.z * sn, cs + a.y * a.y * tt, a.y * a.z * tt - a.x * sn,
                             a.z * a.x * tt - a.y * sn, a.z * a.y * tt + a.x * sn, cs + a.z * a.z * tt);
      for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++) mm[i * 3 + j] = ro[i * 3] * rj[j] + ro[i * 3 + 1] * rj[3 + j] + ro[i * 3 + 2] * rj[6 + j];
    } else {
      for(int i = 0; i < 9; i++) mm[i] = ro[i];
      if(m.z == 2) t += axv.xyz * Q[m.w];
    }
    float pr[9];
    vec3 pp;
    if(m.x == cur) {
      pr = cr;
      pp = cp;
    } else {
      for(int k = 0; k < 9; k++) pr[k] = R[m.x * 9 + k];
      pp = P[m.x];
    }
    for(int i = 0; i < 3; i++)
      for(int j = 0; j < 3; j++) cr[i * 3 + j] = pr[i * 3] * mm[j] + pr[i * 3 + 1] * mm[3 + j] + pr[i * 3 + 2] * mm[6 + j];
    cp  = pp + rot_apply(pr, t);
    cur = m.y;
    for(int k = 0; k < 9; k++) R[m.y * 9 + k] = cr[k];
    P[m.y] = cp;
    place_group(grp, cr, cp);
  }
}

// ブロードフェーズは「グループ数の上限(MAX_GROUPS)まで展開した静的インデックスのループ」で、候補ビットマスクだけを作る。
// 候補が出たグループだけ、上の詳細判定(動的ループ)に進む。
bool config_free() {
  fk();
  const uint ng = pc.num_groups;
  vec3 gb[MAX_GROUPS];
  float gr[MAX_GROUPS];
  [[unroll]] for(int g = 0; g < MAX_GROUPS; g++) {
    gb[g] = gbs[g];
    gr[g] = s_group_bs[g].w;
  }

  for(uint b = 0; b < pc.num_boxes; b++) {
    const uint bo = b * 6;
    const uint allow = floatBitsToUint(s_boxes[bo].w);
    const vec3 center = s_boxes[bo].xyz;
    const vec3 aabb_half = s_boxes[bo + 5].xyz;
    uint cand = 0u;
    [[unroll]] for(int g = 0; g < MAX_GROUPS; g++) {
      vec3 dd = abs(gb[g] - center) - aabb_half - vec3(gr[g] + MARGIN);
      bool hit = uint(g) < ng && ((allow >> uint(g)) & 1u) == 0u && all(lessThanEqual(dd, vec3(0.0)));
      cand |= hit ? (1u << uint(g)) : 0u;
    }
    while(cand != 0u) {
      const int g = findLSB(cand);
      cand &= cand - 1u;
      if(box_group_hit(bo, g)) return false;
    }
  }

  if(balls_hit()) return false;

  // 自己衝突(グループ=リンク単位のペアから候補を絞る。クラスタ・球のペアは別リンクの全組み合わせ)
  [[unroll]] for(int g = 0; g < MAX_GROUPS; g++) {
    uint m = uint(s_pairs[g]);
    uint c = 0u;
    [[unroll]] for(int h = g + 1; h < MAX_GROUPS; h++) {
      vec3 d   = gb[g] - gb[h];
      float rr = gr[g] + gr[h] + 2.0 * MARGIN;
      bool hit = uint(h) < ng && uint(g) < ng && ((m >> uint(h)) & 1u) != 0u && dot(d, d) <= rr * rr;
      c |= hit ? (1u << uint(h)) : 0u;
    }
    while(c != 0u) {
      const int h = findLSB(c);
      c &= c - 1u;
      if(group_pair_hit(g, h)) return false;
    }
  }
  return true;
}
#endif
