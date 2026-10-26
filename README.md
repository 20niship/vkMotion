# vkMotion

[cuMotion](https://nvidia-isaac.github.io/cumotion/) ([cuRobo](https://github.com/NVlabs/curobo)) の Vulkan 版。CUDA/PyTorchの代わりにVulkan computeで動かす、クラス名・関数名をcuRobo互換にしたモーションプランニング・衝突判定ライブラリ。

調査・設計の詳細は [PLAN.md](./PLAN.md)、API仕様・使用例は [MANUAL.md](./MANUAL.md) を参照。

## 特徴

- ロボットを衝突球(collision spheres)の集合として表現し、障害物(Cuboid/Sphere/Cylinder/Capsule)との符号付き距離を計算して衝突判定する(cuRoboと同じ方式)
- 衝突球はURDFの`<collision>`プリミティブ(box/cylinder/sphere)から自動生成(メッシュ不要、手動YAMLでの上書きも可能)
- 衝突判定はCPUリファレンス実装とVulkan compute実装の両方を持ち、相互に結果が一致することをテストで検証
- IK(解析ヤコビアン+Levenberg-Marquardt)、MoveIt的な「目標Poseへの衝突回避軌道生成」(`MotionGen`)を実装
- 衝突判定・FK・RRT-ConnectはロードタイムにコンパイルしたSIMD実装。VAMP・cuRoboと同じ衝突球・同じ問題(MotionBenchMaker)で比較すると、FK/衝突判定/IK/経路の精度は同等、計画時間の中央値はVAMPと同程度(関節ゴール約0.7ms)でcuRoboより2桁速い。計画時間の裾(p95/p99)はVAMPより長く、軌道の滑らかさ(jerk制限)はcuRoboに及ばない。数値と条件は[bench/RESULTS.md](./bench/RESULTS.md)、再現手順は[bench/README.md](./bench/README.md)
- OpenArm(7軸)・SO-101(6軸)・UR5e(6軸)のテスト用URDF(メッシュなし、primitive collisionのみ)を同梱

## 必要なもの

- CMake 3.20以上
- C++17コンパイラ
- Vulkan SDK(loader + `glslc` + validation layer)。Ubuntuなら`sudo apt install vulkan-sdk`または`mesa-vulkan-drivers libvulkan-dev vulkan-validationlayers glslc`
- GPUが無い環境でも[Mesa lavapipe](https://docs.mesa3d.org/drivers/llvmpipe.html)(software Vulkan ICD、`mesa-vulkan-drivers`に含まれる)があれば動作する

以下は`git submodule`([tinyxml2](https://github.com/leethomason/tinyxml2))とCMakeの`FetchContent`([cutil](https://github.com/20niship/cutil) / [Eigen](https://eigen.tuxfamily.org/) / [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) / テスト有効時のみ[doctest](https://github.com/doctest/doctest))で自動取得されるので、個別インストールは不要。

## ビルド

CMakeオプション: `VKMOTION_BUILD_TESTS` / `VKMOTION_BUILD_EXAMPLES` / `VKMOTION_BUILD_BENCH`(比較用ベンチ) / `VKMOTION_NATIVE_ARCH`(既定ON: `-march=native`でSIMDを最大限使う。別マシンで動かすバイナリを作るときはOFF)。

```sh
git clone --recursive https://github.com/20niship/vkMotion.git
cd vkMotion
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

`--recursive`を付け忘れた場合は `git submodule update --init` を実行する。

## テスト

```sh
cmake -S . -B build -DVKMOTION_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

`VKMOTION_BUILD_TESTS=ON`のときだけdoctestがFetchContentされ、`tests/`以下のテストがビルド・ctest登録される。CIでは[Mesa lavapipe](https://docs.mesa3d.org/drivers/llvmpipe.html)を使い、GPUなしのランナーでもVulkan compute実装のテストまで実行する(`.github/workflows/ci.yml`)。

## 使い方(最小例)

目標Poseを指定して衝突回避軌道を生成する、MoveItの`move_group.plan()`に相当する処理:

```cpp
#include <vkmotion/robot_config.hpp>
#include <vkmotion/world_config.hpp>
#include <vkmotion/motion_gen.hpp>

using namespace vkmotion;

int main() {
  auto robot_cfg = RobotConfig::from_file("test_assets/urdf/ur5e.urdf");  // 衝突球はURDFから自動生成
  auto world_cfg = WorldConfig::from_file("test_assets/world/sample_world.yml");

  MotionGenConfig cfg = MotionGenConfig::load_from_robot_config(robot_cfg, world_cfg);
  MotionGen motion_gen(cfg);

  JointState start{.position = {0, -1.57f, 1.57f, -1.57f, -1.57f, 0}};
  Pose goal{cutil::Vec3f(0.4f, 0.3f, 0.5f), cutil::Quatf::identity()};

  MotionGenResult result = motion_gen.plan_single(start, goal);
  if (result.success) {
    for (const auto& js : result.trajectory) { /* 実機/シミュレータへ送る */ }
  }
}
```

各クラス(`RobotConfig`, `WorldConfig`, `RobotWorld`, `IKSolver`, `MotionGen`)の詳しいAPIは[MANUAL.md](./MANUAL.md)を参照。

## cuRoboとの対応

| vkMotion | cuRobo | 備考 |
|---|---|---|
| `RobotConfig` | `RobotConfig` | URDFの`<collision>`プリミティブから衝突球を自動生成(cuRoboは事前スフィアフィッティング済みYAML必須) |
| `WorldConfig` / `Cuboid`/`Sphere`/`Cylinder`/`Capsule` | `WorldConfig` / `curobo.geom.types.*` | クラス名・フィールド名をほぼそのまま踏襲 |
| `RobotWorld` | `RobotWorld` | CPU実装とVulkan compute実装(`use_vulkan`)の両方 |
| `IKSolver` | `IKSolver` | 解析ヤコビアン+Levenberg-Marquardt(CPU)の多シードIK。衝突は解の棄却で扱う(cuRoboは衝突コストを最適化に入れる) |
| `MotionGen` | `MotionGen` | 複数IK解をゴールにした双方向RRT-Connect(CPU SIMD)+shortcut+加速度制限つき時間パラメタライズ。cuRoboのL-BFGS/MPPI軌道最適化は未実装なので軌道は幾何経路(jerk制限なし) |

## ディレクトリ構成

```
vkMotion/
├── vkmotion/          # ライブラリ本体(.hpp/.cppを同階層に配置)
├── shaders/           # Vulkan compute shader (GLSL, ビルド時にSPIR-Vへコンパイル)
├── bench/             # VAMP / cuRobo との速度・精度比較 (RESULTS.mdが結果)
├── third_party/       # git submodule (tinyxml2)
├── test_assets/urdf/  # OpenArm/SO-101/UR5eのテスト用URDF(メッシュなし)
├── test_assets/gt/    # GT比較テスト用の独立計算スクリプト
├── tests/             # doctestベースの単体テスト
└── PLAN.md / MANUAL.md  # 設計資料・API仕様
```

## 既知の制約・今後の課題

- `IKSolver`/`MotionGen`の計画ループはCPU実装(SIMD+任意でマルチスレッド)。Vulkanは大きいバッチの衝突判定(4096件以上)に使う
- `MotionGen`はcuRoboのL-BFGS/MPPI軌道最適化ではなく幾何経路。速度・加速度制限は守るがjerk制限は未対応で、軌道はcuRoboより少し長い(`bench/RESULTS.md`の経路長)
- 高速パスの対象外(障害物にconvex/SDF/attached objectを含む)は従来の参照実装(遅い)にフォールバックする
- 障害物の`Mesh`/`VoxelGrid`(SDF)表現は未対応(プリミティブ形状のみ)
