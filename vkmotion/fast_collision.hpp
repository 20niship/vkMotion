#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "allowed_collision_matrix.hpp"
#include "fast_kinematics.hpp"
#include "simd8.hpp"
#include "robot_config.hpp"
#include "world_config.hpp"

namespace vkmotion {

// ロボットの衝突球・自己衝突ペア・世界障害物をロード時にコンパイルした、アロケーション無し/文字列比較無しの衝突判定。
// RobotWorldの参照実装(sphere_collision.cpp)と同じ判定(符号付き距離<=0で衝突)を、計画ループ内で数µsで返すためのもの。
//   - リンクごとに「リンク内の全球を覆う境界球」を持ち、障害物/他リンクと離れていれば球単位の判定を丸ごと省く
//   - motion_free()は関節間のLipschitz上界から検査間隔を決め、二分法の順(粗→細)で検査して早期に棄却する
// GPU(shaders/robot_validate.comp)へ渡すフラット化済みデータ。ロボットは構築時に1回、世界はversionが変わったときだけ書き出す。
struct GpuRobotData {
  int dof = 0, num_links = 0, num_steps = 0, num_clusters = 0, num_groups = 0, num_pairs = 0, num_spheres = 0;
  std::vector<int32_t> step_meta, roots;
  std::vector<float> step_data;
  std::vector<float> spheres;        // vec4 (x,y,z,r) リンクローカル
  std::vector<int32_t> cluster_meta;  // 4 int: link, sphere_begin, count, group
  std::vector<float> cluster_bs;      // vec4 (xyz, radius)
  std::vector<int32_t> group_meta;    // 4 int: first_cluster, count, link, 0
  std::vector<float> group_bs;        // vec4
  std::vector<int32_t> pairs;         // 2 int: cluster a, cluster b
};
struct GpuWorldData {
  int num_boxes = 0, num_balls = 0;
  std::vector<float> boxes;  // 6 vec4/box: center(xyz)+allow_mask(bit pattern), rt row0, row1, row2 (w=0), half(xyz,0), aabb_half(xyz,0)
  std::vector<float> balls;  // 1 vec4/ball: center(xyz), radius ; + 1 vec4: allow_mask in x
};

class FastCollision {
 public:
  FastCollision(const RobotConfig& robot, const AllowedCollisionMatrix& acm);

  // 障害物をコンパイルする。convex/SDF/attached objectは未対応なのでfalseを返す(呼び出し側は参照実装へフォールバック)。
  bool set_world(const WorldConfig& world);

  const CompiledKinematics& kinematics() const { return kin_; }
  size_t dof() const { return kin_.dof(); }

  // 自己衝突+世界障害物が無ければtrue。スレッドセーフ(スレッドごとのscratchを使う)。
  bool is_free(const float* q) const;
  // 球を margin だけ膨らませて判定する(計画用の安全余裕)。margin=0 なら is_free と同じ。
  // 計画で作る経路は障害物に接して最短化されるため、検査点の取り方が違う別の検証器で見ると微小に食い込む。planner が経路の内部点にだけ余裕を持たせてそれを避ける。
  bool is_free_margin(const float* q, float margin) const { return free_impl(q, margin); }
  // 障害物とだけ判定(自己衝突は見ない)
  bool is_free_world_only(const float* q) const;

  // 検査間隔: 隣り合う検査点の間で球中心が動く距離の上界[m]。既定0.05m(Lipschitz上界なので実移動はこの半分程度。障害物厚み+球径をまたがない目安)。
  void set_motion_step(float meters) { motion_step_ = meters; }
  // a->bの直線補間が衝突しないか(関節ごとのLipschitz上界から検査点数を決め、粗い分割から検査する)
  // margin>0なら内部点を膨らませた球で検査する(端点は呼び出し側の責任)。
  // include_end=trueなら終点bも同じSIMDブロックの1レーンとして検査する(新規ノード検査を別呼び出しにしないため)。
  // step>0なら検査間隔[m]をこの呼び出しだけ置き換える(粗い探索用と、細かい検証用を使い分けるため)。
  bool motion_free(const float* a, const float* b, float margin = 0.0f, bool include_end = false, float step = 0.0f) const;
  // a->bを検査するのに必要な分割数(>=1)
  int motion_steps(const float* a, const float* b, float step = 0.0f) const;

  static uint64_t debug_block_count();
  static uint64_t debug_edge_count();
  static uint64_t debug_check_count();  // このスレッドで実行したis_free系の累計(性能解析用)
  float min_sphere_radius() const { return r_min_; }
  const std::vector<float>& joint_reach() const { return joint_reach_; }

  // 8コンフィグ(q[lane*dof + j])を同時に判定し、衝突なしのレーンのビットを立てて返す。SIMDで単発の数分の1のコスト。
  uint32_t free_mask8(const float* q, float margin = 0.0f) const;

  // GPU用のフラットデータ。ロボットがGPU上限(dof<=16, link<=32, group<=16, sphere<=96, cluster<=40, step<=24, pair<=192, box<=32, ball<=8)を超える、または世界がSIMD非対応ならfalse。
  bool pack_gpu_robot(GpuRobotData& out) const;
  void pack_gpu_world(GpuWorldData& out) const;
  // ロボット専用CPUカーネルを使っているか
  bool jit_active() const { return jit_ != nullptr; }
  bool gpu_compatible() const { return simd_supported() && boxes_.size() <= 32 && balls_.size() <= 8; }  // shaders/include/robot_collision_common.glsl のshared memory上限

  // 並列バッチ判定。out[i]=1:衝突なし。threads<=1なら呼び出しスレッドのみ。
  void is_free_batch(const float* qs, size_t n, uint8_t* out, int threads = 1) const;

 private:
  struct LinkSpheres {
    int link;
    int begin;
    int count;
    float bs_c[3];
    float bs_r;
  };
  // リンク単位の粗い境界球(その下にクラスタ links_[first .. first+count) がぶら下がる)
  struct Group {
    int first;
    int count;
    int link;
    float bs_c[3];
    float bs_r;
  };
  struct Box { float c[3]; float rot[9]; float half[3]; float aabb_half[3]; };
  struct Ball { float c[3]; float r; };
  struct Cyl { float c[3]; float rot[9]; float r; float half_h; };
  struct Cap { float c[3]; float rot[9]; float base[3]; float tip[3]; float r; };
  struct Pln { float c[3]; float n[3]; };

  struct Scratch;
  struct Scratch8;
  static Scratch& scratch(size_t num_links, size_t num_spheres);
  static Scratch8& scratch8(size_t num_links, size_t num_spheres, size_t num_link_spheres, size_t dof);
  bool simd_supported() const { return cyls_.empty() && caps_.empty() && planes_.empty(); }

  bool free_impl(const float* q, float margin) const;
  bool free_scalar(const float* q, float margin) const;
  uint32_t mask8_impl(const float* q, float margin, bool broadcast, uint32_t required) const;
  bool world_free(const Scratch& sc, float margin) const;
  bool self_free(const Scratch& sc, float margin) const;
  void world_positions(const float* q, Scratch& sc) const;

  CompiledKinematics kin_;
  std::vector<LinkSpheres> links_;  // 空間クラスタ(最大4球)。リンク名はlink_names_[i]
  std::vector<Group> groups_;
  std::vector<float> lx_, ly_, lz_, lr_;  // リンクローカルの球
  std::vector<std::pair<int, int>> pairs_;  // links_のindexペア(自己衝突を見る組)
  std::vector<std::string> link_names_;     // links_[i]のリンク名(ACM引き当て用)
  AllowedCollisionMatrix acm_;

  std::vector<Box> boxes_;
  std::vector<Ball> balls_;
  std::vector<Cyl> cyls_;
  std::vector<Cap> caps_;
  std::vector<Pln> planes_;
  // [obstacle kind順に連結した通し番号][links_ index] -> ACMで許可(=判定スキップ)
  std::vector<std::vector<uint8_t>> allowed_;
  size_t n_spheres_ = 0;
  float r_min_      = 0.0f;
  std::vector<float> joint_reach_;
  float motion_step_ = 0.05f;

  // ロボット専用に生成・コンパイルしたカーネル(gpu_codegen.cpp)。コンパイラが無い等でnullptrなら汎用のmask8_implを使う。
  uint32_t (*jit_)(const float*, float, uint32_t, const float*, int, const float*, int) = nullptr;
  std::vector<float> jit_boxes_, jit_balls_;
  int jit_num_boxes_ = 0, jit_num_balls_ = 0;
};

}  // namespace vkmotion
