# vkMotion

[cuMotion](https://nvidia-isaac.github.io/cumotion/) ([cuRobo](https://github.com/NVlabs/curobo)) の Vulkan 版。CUDA/PyTorchの代わりにVulkan computeで動かす、クラス名・関数名をcuRobo互換にしたモーションプランニング・衝突判定ライブラリ。

調査・設計の詳細は [PLAN.md](./PLAN.md)、API仕様・使用例は [MANUAL.md](./MANUAL.md) を参照。

## 特徴

- ロボットを衝突球(collision spheres)の集合として表現し、障害物(Cuboid/Sphere/Cylinder/Capsule)との符号付き距離を計算して衝突判定する(cuRoboと同じ方式)
- 衝突球はURDFの`<collision>`プリミティブ(box/cylinder/sphere)から自動生成(メッシュ不要、手動YAMLでの上書きも可能)
- 衝突判定はCPUリファレンス実装とVulkan compute実装の両方を持ち、相互に結果が一致することをテストで検証
- IK(解析ヤコビアン+Levenberg-Marquardt)、MoveIt的な「目標Poseへの衝突回避軌道生成」(`MotionGen`)を実装
- 衝突判定・FK・RRT-Connectは、ロボットごとに専用のSIMD/GPUカーネルを実行時に生成(`vkmotion/gpu_codegen.cpp`)。経路探索は、複数のCPUスレッドが1組の木を共有して並列に伸ばす(既定)。単スレッドのときはVulkan compute(GPU)の探索がCPUと競争する(複数ワープが1本の木を共有して並列に伸ばす)。
  VAMP・cuRoboと同じ衝突球・同じ問題(MotionBenchMaker)で比較すると、FK/衝突判定/IK/経路の精度は同等。
  関節ゴールの計画時間は、中央値がVAMPの約1.8倍速く(約0.2ms)、裾(p95/p99)はVAMPと同程度(複数CPUスレッドが共有木を伸ばす既定の構成。VAMPは1スレッド)、cuRoboより3桁速い。軌道の滑らかさ(jerk制限)はcuRoboに及ばない。数値と条件は[bench/RESULTS.md](./bench/RESULTS.md)、測り方は[「ベンチマーク」](#ベンチマークvamp--curobo-との比較)
- OpenArm(7軸)・SO-101(6軸)・UR5e(6軸)のテスト用URDF(メッシュなし、primitive collisionのみ)を同梱

## 必要なもの

- CMake 3.20以上
- C++17コンパイラ
- Vulkan SDK(loader + `glslc` + validation layer)。Ubuntuなら`sudo apt install vulkan-sdk`または`mesa-vulkan-drivers libvulkan-dev vulkan-validationlayers glslc`
- (任意)実行時にロボット専用カーネルを生成してコンパイルするため、`glslc`(Vulkan SDKに同梱)とC++コンパイラがCMake時に見つかるとより速くなる。見つからない・失敗した場合は汎用実装に自動で戻る。生成物は`~/.cache/vkmotion`(環境変数`VKMOTION_SHADER_CACHE`で変更、`VKMOTION_NO_CODEGEN=1`で無効化)にキャッシュされ、初回だけ数秒かかる
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

## 使い方

### まず動かす(サンプル実行)

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVKMOTION_BUILD_EXAMPLES=ON
cmake --build build -j$(nproc)
./build/examples/run_stress_test      # ランダムな問題での耐久テスト(examples/run_stress_test.cpp)
./build/examples/run_batch_benchmark  # バッチ衝突判定のベンチ(examples/run_batch_benchmark.cpp)
```

### 目標Poseへの衝突回避軌道(最小例)

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
  motion_gen.warmup();  // Vulkanパイプライン・専用カーネルの準備。最初の1回の遅延を計画の外へ出す(初回は数秒かかることがある)

  JointState start{.position = {0, -1.57f, 1.57f, -1.57f, -1.57f, 0}};
  Pose goal{cutil::Vec3f(0.4f, 0.3f, 0.5f), cutil::Quatf::identity()};

  MotionGenResult result = motion_gen.plan_single(start, goal);
  if (result.success) {
    for (const auto& js : result.trajectory) { /* 実機/シミュレータへ送る */ }
  }
}
```

### 関節角ゴール・設定

```cpp
MotionGenResult r = motion_gen.plan_single_js(start, JointState{.position = {1.5f, -1.0f, 1.0f, -1.0f, -1.0f, 0.5f}});
```

主な設定(`MotionGenConfig`。全項目は[MANUAL.md](./MANUAL.md)):

```cpp
cfg.num_threads      = 0;       // 大域探索(RRT-Connect)で1組の木を共有して伸ばすCPUスレッド数。0=自動(物理コア数相当、1〜8)。難しい問題の時間が大きく縮む(下のベンチ参照)。1にすると単スレッド
cfg.gpu_search       = true;    // num_threads=1のとき、大域探索をGPUでも行う(CPUと競争し、先に解けた方を採る)。GPUが無ければ自動でCPUのみ。複数スレッドのときはGPUを使わない(共有木の方が速い)
cfg.gpu_handoff_s    = 0.0001f; // 最初にCPU単スレッドだけでこの秒数試す(簡単な問題はここで終わる)。GPU探索はその後に走らせ始める
cfg.collision_margin = 0.005f;  // 経路の内部点に持たせる安全余裕[m]
cfg.seed             = 1234;    // 0なら非決定的、それ以外なら再現可能
```

### 複数の問題をまとめて解く(バッチ)

問題ごとに別の世界(障害物)を持てる。大域探索は全問題を1回のGPU submit列で並列に行い、経路の整形はCPUスレッドで並列に行う:

```cpp
std::vector<MotionGen::BatchProblem> problems;
for (auto& p : my_problems) problems.push_back({p.start, p.goal, p.world});  // JointState, JointState, Ref<WorldConfig>
std::vector<MotionGenResult> results = motion_gen.plan_batch_js(problems);
```

単発より1問あたりの時間が短くなるとは限らない(`bench/RESULTS.md`のGPUバッチの行を参照)。多数の問題をまとめて処理したいときに使う。

各クラス(`RobotConfig`, `WorldConfig`, `RobotWorld`, `IKSolver`, `MotionGen`)の詳しいAPIは[MANUAL.md](./MANUAL.md)を参照。

## ベンチマーク(VAMP / cuRobo との比較)

同じロボット(Franka Panda)・同じ衝突球(VAMPの`panda_spherized.urdf`)・同じ自己衝突除外・同じ問題(VAMP同梱のMotionBenchMaker、障害物7〜21個)で、vkMotion・[VAMP](https://github.com/KavrakiLab/vamp)・[cuRobo](https://github.com/NVlabs/curobo)の速度と精度を比べる。実装は`bench/`、結果は[bench/RESULTS.md](./bench/RESULTS.md)。

```sh
bench/setup.sh               # 初回のみ。bench/.venv に vamp/cuRobo、bench/.deps に問題セットとcuRoboを用意 (uv, git, NVIDIA GPU が必要)
bench/run_all.sh --limit 30  # 各シーン先頭30問(計210問)で全実装を実行し、bench/RESULTS.md を再生成 (約3分)
bench/run_all.sh --limit 0   # 699問すべて
bench/run_all.sh --libs vkmotion         # vkMotionだけ再計測 (VAMP/cuRoboの結果は bench/results/ のものを使う)
bench/run_all.sh --libs vamp,vkmotion    # cuRoboを除く (GPUをcuRoboが占有しない)
```

`run_all.sh`は、実装を**直列に**実行する(計測の干渉を避けるため)。実行前に、負荷平均・CPUを食っているプロセス・GPUを使っているプロセスを`bench/results/env.txt`に記録し、`RESULTS.md`の冒頭に載せる。

手動で1つだけ測るとき:

```sh
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DVKMOTION_BUILD_BENCH=ON -DVKMOTION_ENABLE_VALIDATION=OFF
cmake --build build-bench -j$(nproc) --target bench_vkmotion
build-bench/bench/bench_vkmotion bench/data out.json --backend vulkan --limit 30   # GPU探索あり(既定)
build-bench/bench/bench_vkmotion bench/data out.json --backend cpu    --limit 30   # GPUを使わない
build-bench/bench/bench_vkmotion bench/data out.json --backend cpu --threads 8 --limit 30   # CPUで8本の独立探索を競わせる
build-bench/bench/bench_vkmotion bench/data out.json --backend vulkan --batch 32 --limit 30  # 32問ずつバッチで解く
bench/.venv/bin/python bench/run_vamp.py --limit 30    # VAMPだけ
bench/.venv/bin/python bench/scene_stats.py out.json   # シーンごとの中央値/p95/p99/最大
```

測るもの: FK・衝突判定の精度とスループット、関節ゴール/姿勢ゴールの計画時間(中央値・p95・p99)・成功率・経路長、計画された経路を**VAMPの検証器**で見たときの衝突なし率(精度の基準はVAMP)、姿勢ゴールの末端誤差。

測るときの注意:

- **他のプロセスの負荷で数値が大きく変わる**。CPUを大量に使うプロセス(例: `ollama serve`が暴走していると、VAMPの中央値が0.35msから1.5msに見えた)やGPUを使うジョブが動いていると、CPU側・GPU側とも2〜3倍ぶれる。測定前に`uptime`と`nvidia-smi`を確認する。長いベンチは`run_in_background`などで裏で回し、その間は他の重い作業をしない。
- GPUはアイドルが続くとクロックが落ち、間隔を空けた呼び出しは遅くなる。初回はロボット専用シェーダ/カーネルのコンパイル(数秒)が走るので、`bench_vkmotion`は`warmup()`でそれを計測から外している。
- VAMPの時間は、`planner(...)`+`simplify`のwall clock(pythonバインディング経由)。1回目の実行の時間を使う(3回の中央値は`time_ms_median3`に別に保存)。
- cuRoboは姿勢ゴール対応でGPU上の軌道最適化のため1回数百msかかる。`max_attempts=10`で実行している。
- テストケースはMotionBenchMaker(VAMP同梱の問題セット)。障害物は直方体のみ(円柱は外接直方体に正規化)。

## cuRoboとの対応

| vkMotion | cuRobo | 備考 |
|---|---|---|
| `RobotConfig` | `RobotConfig` | URDFの`<collision>`プリミティブから衝突球を自動生成(cuRoboは事前スフィアフィッティング済みYAML必須) |
| `WorldConfig` / `Cuboid`/`Sphere`/`Cylinder`/`Capsule` | `WorldConfig` / `curobo.geom.types.*` | クラス名・フィールド名をほぼそのまま踏襲 |
| `RobotWorld` | `RobotWorld` | CPU実装とVulkan compute実装(`use_vulkan`)の両方 |
| `IKSolver` | `IKSolver` | 解析ヤコビアン+Levenberg-Marquardt(CPU)の多シードIK。衝突は解の棄却で扱う(cuRoboは衝突コストを最適化に入れる) |
| `MotionGen` | `MotionGen` | 複数IK解をゴールにした双方向RRT-Connect(CPU SIMD + GPU)+shortcut+加速度制限つき時間パラメタライズ。cuRoboのL-BFGS/MPPI軌道最適化は未実装なので軌道は幾何経路(jerk制限なし) |

## ディレクトリ構成

```
vkMotion/
├── vkmotion/          # ライブラリ本体(.hpp/.cppを同階層に配置)
├── shaders/           # Vulkan compute shader (GLSL, ビルド時にSPIR-Vへコンパイル)
├── bench/             # VAMP / cuRobo との速度・精度比較 (RESULTS.mdが結果、README.mdが再現手順)
├── third_party/       # git submodule (tinyxml2)
├── test_assets/urdf/  # OpenArm/SO-101/UR5eのテスト用URDF(メッシュなし)
├── test_assets/gt/    # GT比較テスト用の独立計算スクリプト
├── tests/             # doctestベースの単体テスト
└── PLAN.md / MANUAL.md  # 設計資料・API仕様
```

## 既知の制約・今後の課題

- 大域探索(木の伸長・衝突判定)はGPU(Vulkan compute、サブグループ幅32のGPU、dof<=7)でも動く。GPUが使えない・ロボットが対応外(dof>7、衝突球・リンク数の上限超過)なら自動でCPU(SIMD)に戻る。経路の整形・検証・時間パラメタライズ、IKはCPU
- 計画時間の裾(p95/p99)はVAMPより長い。主因は衝突判定1回の重さ(8コンフィグで約0.5〜0.8µs、VAMPは約0.2µs)。`num_threads=1`のときGPU探索はCPUの探索と競争させている(先にCPUで0.1ms試し、解けなければGPUを裏で走らせたままCPUも続け、先に解けた方を採る)。GPUはアイドルが続くとクロックが落ちるので、間隔を空けた呼び出しでは遅くなる。既定(複数CPUスレッドが共有木を伸ばす)ではGPUを使わない。難しい問題の時間はこちらの方が短かった
- `MotionGen`はcuRoboのL-BFGS/MPPI軌道最適化ではなく幾何経路。速度・加速度制限は守るがjerk制限は未対応で、軌道はcuRoboより少し長い(`bench/RESULTS.md`の経路長)
- 高速パスの対象外(障害物にconvex/SDF/attached objectを含む)は従来の参照実装(遅い)にフォールバックする。バッチ計画は高速パスの世界(箱・球・円柱・カプセル・平面。GPUは箱32個・球8個まで)のみ
- 障害物の`Mesh`/`VoxelGrid`(SDF)表現は未対応(プリミティブ形状のみ)
