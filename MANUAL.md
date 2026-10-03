# vkMotion 動作マニュアル (仮)

> この文書は暫定版。クラス名・関数名は [cuRobo](https://github.com/NVlabs/curobo) / cuMotion の Python API にできるだけ合わせてある(MoveItのように目標 Pose を渡すとそこへの衝突回避軌道を返す、という使い方を踏襲)。
> 型は [cutil](https://github.com/20niship/cutil) の `Vec3f` / `Quatf` / `Mat4f` / `Ref<T>` を使用する。
> ここに書いたシグネチャが、そのまま [PLAN.md](./PLAN.md) の各フェーズの実装対象になる。

## 0. 全体の使い方(MoveIt的な「目標Poseへ移動」のフルフロー)

```cpp
#include <vkmotion/robot_config.hpp>
#include <vkmotion/world_config.hpp>
#include <vkmotion/motion_gen.hpp>

using namespace vkmotion;
using namespace cutil;

int main() {
  // 1. ロボット構成を読み込む。URDFの<collision>プリミティブから衝突球を自動生成する
  //    (第2引数省略時)。手動チューニング済みの球を使うなら第2引数にYAMLパスを渡す。
  Ref<RobotConfig> robot_cfg = RobotConfig::from_file("test_assets/urdf/ur5e.urdf");

  // 2. 障害物を含むワールドを構築する
  Ref<WorldConfig> world_cfg = WorldConfig::from_file("test_assets/world/table_scene.yml");
  world_cfg->add_obstacle(Cuboid{
      .name = "table",
      .pose = Pose{Vec3f(0.5f, 0.0f, -0.05f), Quatf::identity()},
      .dims = Vec3f(1.0f, 1.0f, 0.1f),
  });

  // 3. MotionGenを構築(内部でVulkan compute初期化。use_vulkan=falseでCPUのみ動作)
  MotionGenConfig motion_gen_cfg = MotionGenConfig::load_from_robot_config(robot_cfg, world_cfg);
  MotionGen motion_gen(motion_gen_cfg);
  motion_gen.warmup();

  // 4. 開始状態と目標Poseを指定してプランニング(= MoveItのmove_group.plan()相当)
  JointState start_state{.position = {0.0f, -1.57f, 1.57f, -1.57f, -1.57f, 0.0f}};
  Pose goal_pose{Vec3f(0.4f, 0.3f, 0.5f), Quatf::identity()};

  MotionGenResult result = motion_gen.plan_single(start_state, goal_pose, MotionGenPlanConfig{});

  if (result.success) {
    for (const auto& js : result.trajectory) {
      // js.position を実機/シミュレータへ送る
    }
  }
  return 0;
}
```

以下、各ステップで使うクラスの詳細。

## 1. `RobotConfig` — ロボット構成の読み込み

`vkmotion/robot_config.hpp`

```cpp
namespace vkmotion {

struct JointState {
  std::vector<float> position;
  std::vector<float> velocity;  // optional, 未使用時は空
};

struct Pose {
  cutil::Vec3f position;
  cutil::Quatf quaternion;  // [qw, qx, qy, qz]
  static Pose from_list(const std::vector<float>& xyzwxyz);
  cutil::Mat4f to_matrix() const;
};

class RobotConfig {
 public:
  // URDFを読み込む。collision_spheres_yaml_pathが空なら、URDFの各リンク<collision>
  // プリミティブ(box/cylinder/sphere)から衝突球を自動生成する。
  // 非空ならそのYAMLの内容で球定義を上書き(手動チューニング用)。
  static cutil::Ref<RobotConfig> from_file(const std::string& urdf_path,
                                            const std::string& collision_spheres_yaml_path = "");

  const std::vector<std::string>& joint_names() const;
  size_t dof() const;
  const std::vector<float>& joint_position_limits_lower() const;
  const std::vector<float>& joint_position_limits_upper() const;
  const std::vector<float>& joint_velocity_limits() const;  // URDFの<limit velocity>、無指定なら実質無制限
};

}  // namespace vkmotion
```

- `from_file` は cuRobo の `RobotConfig.from_dict()` に相当(cuRoboはYAML dictで球モデルを必須指定するが、vkMotionはメッシュ無し純粋URDF前提のため URDF の `<collision>` プリミティブから自動生成するのがデフォルト)。
- 自動生成ロジック(`vkmotion/urdf_loader.hpp` 内、`generate_collision_spheres_from_urdf`):
  - `<sphere radius="r">` → 中心に1球、半径そのまま
  - `<cylinder radius="r" length="l">` → 軸方向に `ceil(l / (2r))` 個の球を等間隔でスタック
  - `<box size="x y z">` → OBBを包含するよう軸ごとに球をグリッド配置(球半径は最短辺の半分程度)
  - `<capsule>`(URDF拡張) → 両端+軸上に等間隔で球配置
- 手動調整用の球モデルYAMLフォーマット(案、cuRoboの`collision_spheres`キーに準拠。上書き時のみ使用):

```yaml
collision_spheres:
  shoulder_link:
    - center: [0.0, 0.0, 0.0]
      radius: 0.06
  upper_arm_link:
    - center: [0.0, 0.0, 0.1]
      radius: 0.05
    - center: [0.0, 0.0, 0.3]
      radius: 0.05
```

## 2. `WorldConfig` / `Obstacle` — 障害物構成

`vkmotion/types.hpp`, `vkmotion/world_config.hpp`

```cpp
namespace vkmotion {

struct Obstacle {
  std::string name;
  Pose pose;
};

struct Cuboid : Obstacle { cutil::Vec3f dims; };
struct Sphere : Obstacle { float radius; };
struct Cylinder : Obstacle { float radius; float height; };
struct Capsule : Obstacle { float radius; cutil::Vec3f base; cutil::Vec3f tip; };

class WorldConfig {
 public:
  static cutil::Ref<WorldConfig> from_file(const std::string& yaml_path);

  void add_obstacle(const Cuboid& obstacle);
  void add_obstacle(const Sphere& obstacle);
  void add_obstacle(const Cylinder& obstacle);
  void add_obstacle(const Capsule& obstacle);

  const std::vector<Cuboid>& cuboid() const;
  const std::vector<Sphere>& sphere() const;
  const std::vector<Cylinder>& cylinder() const;
  const std::vector<Capsule>& capsule() const;
};

}  // namespace vkmotion
```

- クラス名・フィールド名は cuRobo の `curobo.geom.types.{Cuboid,Sphere,Cylinder,Capsule}` に合わせる(PLAN.md 1.2節参照)。
- `Mesh` / `VoxelGrid`(SDF)は型のみ将来追加、今回はスタブなし。

## 3. `RobotWorld` — 衝突判定

`vkmotion/robot_world.hpp`

```cpp
namespace vkmotion {

struct RobotWorldConfig {
  cutil::Ref<RobotConfig> robot_config;
  cutil::Ref<WorldConfig> world_config;
  bool use_vulkan = true;  // falseならCPUリファレンス実装のみで計算
};

class RobotWorld {
 public:
  explicit RobotWorld(const RobotWorldConfig& config);

  // 各衝突球中心から最近傍障害物表面までの符号付き距離(負値=衝突)。cuRoboのget_world_self_collision_distance_from_joints相当。
  std::vector<float> get_collision_distance(const JointState& joint_state) const;

  // 自己衝突(リンク間)の符号付き距離
  std::vector<float> get_self_collision_distance(const JointState& joint_state) const;

  bool is_collision_free(const JointState& joint_state) const;

  // バッチ版(多数の関節角度を一括でVulkan computeに投げる)
  std::vector<bool> is_collision_free_batch(const std::vector<JointState>& joint_states) const;
};

}  // namespace vkmotion
```

対応テスト: `tests/test_collision_cpu.cpp`(`use_vulkan=false`)、`tests/test_collision_vulkan.cpp`(`use_vulkan=true`、CPU版と出力一致を確認)、`tests/test_collision_gt.cpp`(Python cuRoboで生成したハードコードGTと比較)。

## 3.5 `AllowedCollisionMatrix` / `AttachedObject` — MoveIt2のACM/attached object相当

`vkmotion/allowed_collision_matrix.hpp`, `vkmotion/types.hpp`

```cpp
namespace vkmotion {

class AllowedCollisionMatrix {
 public:
  void disable_collision(const std::string& a, const std::string& b);  // 対称、名前は link_name か Obstacle::name
  bool is_allowed(const std::string& a, const std::string& b) const;
};

// link_nameを持つ普通のCollisionSphereとして追加される「掴んだ物体」。center/radiusはlinkローカル座標。
struct AttachedObject {
  std::string name;
  std::string link_name;
  cutil::Vec3f center;
  float radius;
};

}  // namespace vkmotion
```

- `RobotWorldConfig::acm`に設定したペアは、自己衝突(`get_self_collision_distance`)とworld障害物衝突(`get_collision_distance`、CPU経路)で追加の除外ルールとして使われる。
- `WorldConfig::add_attached_object()`で登録したAttachedObjectは、毎クエリで追従先リンクのFK結果から現在のワールド座標を再計算し、通常のCollisionSphereと同様にself-collision/world-collisionチェックに加わる(自分自身が属するリンク・隣接リンクとは`is_adjacent`で既に除外される)。
- `get_collision_distance`/`get_self_collision_distance`/`get_min_world_collision_distance_batch`/`is_collision_free_batch`はCPU・GPUどちらの経路でもAttachedObjectに対応する(普通のCollisionSphereとして扱われるだけで、シェーダ変更が不要なため)。**GPU常駐の`RobotWorld::repair_waypoints`(`use_vulkan=true`)のみAttachedObject未対応**(ロボット構造を構築時に1回だけアップロードする設計のため。`use_vulkan=false`のCPU repairは対応済み)。
- ACMはCPU経路のworld障害物チェックにのみ適用される(GPU経路の`get_collision_distance`等はACMを見ない、全ロボット球に対して全障害物をチェックする)。

## 4. `IKSolver` — 逆運動学

`vkmotion/ik_solver.hpp`

```cpp
namespace vkmotion {

struct IKSolverConfig {
  cutil::Ref<RobotConfig> robot_config;
  cutil::Ref<WorldConfig> world_config;  // 衝突回避IKをやる場合に使用、nullptrなら衝突無視
  int num_seeds = 32;  // 並列に試す初期値の数
};

class IKSolver {
 public:
  explicit IKSolver(const IKSolverConfig& config);

  // 単一ゴール。収束しなければstd::nullopt
  std::optional<JointState> solve_single(const Pose& goal_pose,
                                          const std::optional<JointState>& seed = std::nullopt);

  // 複数ゴールを並列に解く(Vulkan compute: num_seeds x batch_size を1 dispatch)
  std::vector<std::optional<JointState>> solve_batch(const std::vector<Pose>& goal_poses);
};

}  // namespace vkmotion
```

- CPUリファレンス実装(`solve_single`の内部)はヤコビアンの減衰最小二乗法(DLS)で解く。ヤコビアンは`6×dof`の可変サイズ行列なので cutil の固定長 `Mat4f` ではなく **Eigen** (`Eigen::MatrixXd` + `Eigen::JacobiSVD`等) を使う。Vulkan compute版は同じDLS更新式をGLSLで並列化(シード毎に1スレッド)したもの。

## 5. `MotionGen` — Pose指定プランニング(本題)

`vkmotion/motion_gen.hpp`

これが MoveIt の `move_group.setPoseTarget()` + `plan()` + `execute()` に相当する、本プロジェクトの中心機能。cuRoboでは `MotionGen.plan_single()` が同じ役割。

```cpp
namespace vkmotion {

struct MotionGenConfig {
  cutil::Ref<RobotConfig> robot_config;
  cutil::Ref<WorldConfig> world_config;
  int num_trajopt_seeds = 12;  // 並列に試す初期シード数(IK解候補+乱数)
  bool use_vulkan = true;

  static MotionGenConfig load_from_robot_config(cutil::Ref<RobotConfig> robot_config,
                                                 cutil::Ref<WorldConfig> world_config);
};

struct MotionGenPlanConfig {
  int max_attempts = 10;
  bool enable_graph_planner = true;   // IK+TrajOptが失敗した場合に幾何プランナーへフォールバック(stretch, 未実装時は無視)
  float timeout_seconds = 5.0f;
};

struct MotionGenResult {
  bool success = false;
  std::vector<JointState> trajectory;      // 補間済み軌道(関節角度列)
  std::vector<float> interpolated_times;   // 各ステップの時刻[s]
  std::string status;                      // 失敗理由など
};

class MotionGen {
 public:
  explicit MotionGen(const MotionGenConfig& config);

  // GPU/Vulkanパイプラインの事前ウォームアップ(初回dispatchのレイテンシ削減。cuRoboのwarmup()相当)
  void warmup();

  // 開始関節角度 -> 目標Pose への衝突回避軌道を生成
  MotionGenResult plan_single(const JointState& start_state,
                               const Pose& goal_pose,
                               const MotionGenPlanConfig& plan_config = {});

  // 開始関節角度 -> 目標関節角度(Poseでなく直接関節空間ゴール)
  MotionGenResult plan_single_js(const JointState& start_state,
                                  const JointState& goal_state,
                                  const MotionGenPlanConfig& plan_config = {});
};

}  // namespace vkmotion
```

内部処理フロー(cuRobo同様):

1. `goal_pose` を `IKSolver` で複数解に展開(`num_trajopt_seeds`個の初期値)
2. 各初期値から軌道最適化(勾配降下 / L-BFGS相当 + `RobotWorld`の衝突距離をコスト項として使用)を並列実行
3. 衝突なし・関節limit内で最初に収束した軌道を採用、`MotionGenResult`に詰めて返す
4. 全シード失敗なら `enable_graph_planner` が真なら幾何プランナーにフォールバック(未実装の間は `success=false`)

## 5.5 `CartesianPathPlanner` — 直線Cartesianパス(MoveIt2: computeCartesianPath)

`vkmotion/cartesian_path.hpp`

エンドエフェクタをCartesian空間(位置+姿勢)で直線的に動かしたい場合に使う。`MotionGen`は関節空間で直線補間するため、生成される経路はCartesian空間では一般に曲線になる。これに対し`CartesianPathPlanner`は目標Poseの列を位置lerp+姿勢slerpで細かく分割し、各点をIKチェイニング(直前の解を次のシードにする)で解くことで、エンドエフェクタの軌跡そのものを直線に保つ。

```cpp
struct CartesianWaypoint { Pose pose; };

struct CartesianPathConfig {
  float max_step         = 0.01f;  // 隣接補間点間の最大直線距離[m]
  float max_angular_step = 0.05f;  // 隣接補間点間の最大回転角[rad]
  float min_fraction      = 1.0f;  // この到達率未満なら失敗扱い
};

struct CartesianPathResult {
  bool success;
  float fraction_achieved;  // MoveIt2のcomputeCartesianPathと同様、0.0〜1.0の到達率
  std::vector<JointState> trajectory;
  std::string status;
};

class CartesianPathPlanner {
 public:
  explicit CartesianPathPlanner(cutil::Ref<RobotConfig> robot_config, cutil::Ref<WorldConfig> world_config, bool use_vulkan = true);
  CartesianPathResult compute_cartesian_path(const JointState& start_state, const std::vector<CartesianWaypoint>& waypoints,
                                              const CartesianPathConfig& config = {});
};
```

- IKで解けない、または衝突する中間点に到達した時点で経路を打ち切り、そこまでの到達率を`fraction_achieved`に入れる(`min_fraction`以上なら成功扱い)
- IKチェイニング自体は逐次処理だが、各ステップのIK自体は既存の`IKSolver`(複数シード並列)をそのまま使う

## 5.6 `Constraint` / `ConstraintSet` — path constraints(MoveIt2: position/orientation/joint constraints)

`vkmotion/constraints.hpp`

C++20 conceptで「制約」を定義し、具象型(`JointConstraint`/`PositionConstraint`/`OrientationConstraint`)がそれを満たす。ヘテロな制約集合は`std::variant`+`std::visit`で型消去した`ConstraintSet`に保持する。

```cpp
template <typename T>
concept Constraint = requires(const T& c, const JointState& js, const RobotConfig& robot) {
  { c.is_satisfied(js, robot) } -> std::convertible_to<bool>;
  { c.violation(js, robot) } -> std::convertible_to<float>;  // 0以下なら満たす、正なら違反量
};

struct JointConstraint { int joint_index; float lower, upper; /* URDFのlimitより厳しい一時範囲 */ };
struct PositionConstraint { cutil::Vec3f target, tolerance; /* エンドエフェクタ位置のボックス制約 */ };
struct OrientationConstraint { cutil::Quatf target; float tolerance_rad; /* エンドエフェクタ姿勢の角度制約 */ };

class ConstraintSet {
 public:
  template <Constraint T> void add(T c);
  bool is_satisfied(const JointState&, const RobotConfig&) const;   // 全制約のAND
  float total_violation(const JointState&, const RobotConfig&) const; // repair用の目的関数
};
```

`MotionGenConfig::constraints`(`std::optional<ConstraintSet>`、既定`nullopt`)に設定すると、`MotionGen::plan_single_js`は衝突repair・shortcut後に制約違反waypointを数値勾配降下で修正し、修正後も衝突フリーであることを再確認してから成功を返す(未設定時は従来どおり制約判定を一切行わない)。制約チェック自体は軽量なCPU計算なのでGPU化していない。

## 6. 実装フェーズとの対応

| マニュアル節 | PLAN.mdのフェーズ |
|---|---|
| 1. RobotConfig | フェーズ1 |
| 2. WorldConfig/Obstacle | フェーズ2 |
| 3. RobotWorld(CPU) | フェーズ2 |
| 3. RobotWorld(Vulkan) | フェーズ3 |
| 4. IKSolver | フェーズ4 |
| 5. MotionGen | フェーズ5 |

MotionGenは本マニュアルの主目的(Pose指定プランニング)だが工数が大きいため、PLAN.md通りフェーズ4・5として段階実装する。シグネチャ自体は本マニュアルで先に確定させ、各フェーズの実装対象とする。
