# vkMotion 実装計画

cuMotion (cuRobo) の Vulkan 版を作る。目標は API 互換(関数名・クラス名をできる限り合わせる)と、CPU 実装との GT (Ground Truth) 比較による正当性検証。API の具体的なシグネチャ・使用例は [MANUAL.md](./MANUAL.md) (仮) を参照。

## 1. cuMotion / cuRobo の技術調査まとめ

調査元: [cuRobo論文(ICRA2023)](https://curobo.org/reports/curobo_report.pdf), [cuRobo v2 GitHub](https://github.com/NVlabs/curobo), [cuMotion公式](https://nvidia-isaac.github.io/cumotion/), [Isaac Sim documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/manipulators/manipulators_curobo.html)

### 1.1 ロボット表現

- ロボットは **衝突球 (collision spheres)** の集合で近似する。メッシュを使わず、各リンクに対して複数の球 `{center: [x,y,z], radius: r}` を割り当てる。
- 球中心はリンクのローカル座標系で定義され、FK で得たリンク姿勢を使ってワールド座標に変換する。
- 球同士・球と障害物の距離計算だけで全身衝突判定が完結するため、GPU で大量並列に投げても軽い。
- cuRobo本家は、リンクの衝突メッシュに対してスフィアフィッティング(`generate_collision_spheres`系のツール)を事前実行し、結果を球モデルYAMLとして保存・手動調整して使う(実行時に毎回メッシュから計算はしない)。

**vkMotionでの方針**: REQUIREMENTS.md により URDF はメッシュ無しの純粋な`.urdf`で、各リンクの`<collision>`は`<box>`/`<cylinder>`/`<sphere>`等のプリミティブ形状になる。よって cuRobo のように外部メッシュに対する事前スフィアフィッティングは不要で、**URDFの`<collision>`プリミティブから衝突球をロード時に自動生成**できる(円柱なら軸方向にスタックした複数球、ボックスならOBBを覆う複数球、球ならそのまま1球)。これにより別YAMLが必須ではなくなる。ただし自動生成した球は粗いため、**手動調整用に球モデルYAMLで明示的に上書きできるオプションも残す**(球数・半径のチューニング用、cuRoboの手動調整ワークフローに相当)。

### 1.2 障害物表現 (`WorldConfig` / `Obstacle`)

cuRobo の `curobo/_src/geom/types.py` 相当の構成 (`Obstacle` 基底クラスを継承):

| クラス | フィールド | 備考 |
|---|---|---|
| `Cuboid` | `dims: [x,y,z]` | pose(位置+クォータニオン)必須 |
| `Capsule` | `radius`, `base: [x,y,z]`, `tip: [x,y,z]` | ローカル軸上の始点・終点 |
| `Cylinder` | `radius`, `height` | |
| `Sphere` | `radius` | `position`は非推奨、`pose`推奨 |
| `Mesh` | `vertices`, `faces` | 本プロジェクトではスコープ外(REQUIREMENTS.md によりメッシュはcommitしない方針) |
| `PointCloud` | 点群 | スコープ外 |
| `VoxelGrid` / SDF | occupancy / ESDF | 将来対応。cuRoboはBlox(ESDF)でSDFベース衝突も対応 |

共通フィールド: `name`, `pose: [x,y,z,qw,qx,qy,qz]`, `color`, `scale`。

→ vkMotionでは `Box` (Cuboidではなくcutilの命名慣習に寄せて検討、または互換性重視でCuboidのままにするか要検討)、`Cylinder`、`Capsule`、`Sphere` を初期実装、`SDF`/`Mesh`は型だけ用意してスタブにする。

### 1.3 衝突判定アルゴリズム

- 球-障害物間は「符号付き距離関数 (SDF) の点クエリ + 最近傍点」のみで表現できる形に統一している。これにより障害物の種類(プリミティブ/メッシュ/占有格子)によらず同じ衝突コスト計算コードを使い回せる。
- 連続衝突判定 (continuous collision / swept sphere) も、始点・終点の球を結ぶ線分と障害物SDFの最近傍点計算で近似する。
- `RobotWorld` クラスが「ロボット-ワールド間の符号付き距離」「自己衝突距離」を計算するエントリポイント。

### 1.4 最適化・軌道生成

- 多数の初期シード(乱数関節角度)を**並列**に用意し、それぞれについて軌道最適化を実行、最初に収束したものを採用する。
- 最適化手法: 勾配降下法、L-BFGS、MPPI (particle-based) を組み合わせる。L-BFGSのステップ方向 + ノイズ付き並列line searchでロバスト性を確保。
- 幾何プランナー(RRT-Connect相当だが並列化)をフォールバックとして用意。
- IK は数値ソルバー(多数シード並列)。

### 1.5 API構成 (cuRobo Python)

```
RobotConfig.from_dict()      → ロボット構成(URDF + 球モデル + joint limits)
WorldConfig.from_dict()      → 障害物構成
RobotWorld / RobotWorldConfig → ロボット-ワールド間 衝突距離計算
IKSolver / IKSolverConfig     → 逆運動学
MotionGenConfig.load_from_robot_config()
MotionGen().plan_single()     → 軌道生成
```

## 2. Vulkan での代替方針

| cuRobo (CUDA/PyTorch) | vkMotion (Vulkan) |
|---|---|
| CUDA kernel | Vulkan compute shader (GLSL → SPIR-V, `glslc`でビルド時コンパイル) |
| PyTorchテンソル(バッチ次元=並列シード) | SSBO (Storage Buffer) に シード数 × 関節数 を敷き詰め、`vkCmdDispatch` で並列化 |
| 自動微分(勾配) | 解析ヤコビアン(FKは閉形式で微分可能) or 有限差分をshader内で計算 |
| cudaMalloc / メモリ管理 | VMA (Vulkan Memory Allocator) |
| cuRoboのsphere-primitive距離関数群 | GLSL関数として移植(sphere-sphere, sphere-cuboid(OBB), sphere-cylinder, sphere-capsule) |
| Validation (cuda-memcheck等) | Vulkan Validation Layer を Debugビルドで常時有効化 |

- CPU実装はcutilの`Vec3f`/`Mat4f`等を使い、**Vulkanに依存しないリファレンス実装**として先に作る。Vulkan compute版はCPU版と同じ入出力になるようにし、結果を突き合わせるテストを用意する。
- GPUが無いCI環境でもVulkan computeを動かせるよう、software Vulkan実装 (Mesa lavapipe等) をCIで使う。

## 3. スコープ方針 (MVP優先)

REQUIREMENTS.md が要求する CI は「CPUでのmotion planning/衝突判定とハードコードしたGT比較」なので、**まずCPU版の衝突判定を完成させ、Vulkan化は衝突判定から着手する**。軌道最適化(MotionGen本体)はフェーズを分けて後回しにする。

### フェーズ0: プロジェクト基盤
- CMakeLists.txt (トップレベル)
- cutilをFetchContentまたはsubmoduleで取得、`Vec3f`/`Mat4f`/`Ref`等を使用
- tinyxml2を`third_party/tinyxml2`にgit submoduleとして追加、CMakeから`add_subdirectory`
- Eigenをheader-onlyでFetchContent
- doctestをCMakeオプション(`VKMOTION_BUILD_TESTS`)でFetchContent取得、有効時のみctest登録
- Vulkan SDK前提のfind_package、validation layer有効化オプション
- GitHub Actions CIの土台(ビルドのみ、`submodules: recursive`でcheckout)

### フェーズ1: URDF読み込み + FK (CPU)
- 軽量URDFパーサ(tinyxml2ベース、joint tree・origin・axis・limit・`<collision>`プリミティブのみ対応。メッシュ参照は無視)
- 順運動学 (FK): ジョイント角度配列 → 各リンクのワールド姿勢
- URDFの`<collision>`プリミティブ(box/cylinder/sphere)から衝突球を自動生成するロジック
- 球モデルYAMLローダー(自動生成結果を上書きする手動調整用、任意)
- doctestでFK単体テスト、自動球生成の単体テスト

### フェーズ2: 障害物表現 + CPU衝突判定
- `Obstacle`基底 + `Cuboid`/`Cylinder`/`Capsule`/`Sphere` (cuRobo命名互換)
- `WorldConfig`相当のコンテナ
- 球-障害物間距離関数(各プリミティブごと)
- `RobotWorld`相当: ロボット球 × 障害物 の衝突判定・自己衝突判定
- Python版cuRoboで生成したGT結果をテストにハードコードし比較するテストを追加
- OpenArm, SO101, URシリーズの`.urdf`(メッシュなし、純粋URDFのみ)をtest fixtureとして用意

### フェーズ3: Vulkan化(衝突判定)
- `vk::Context` (instance/device/queue/VMA初期化、validation layer)
- compute shaderでのバッチ衝突判定(シード×姿勢を一括処理)
- CPU版と同一入力でVulkan版の出力一致を検証するテスト
- CIにsoftware Vulkan(lavapipe)導入、ヘッドレスでcompute shaderを実行

### フェーズ4(stretch): IKSolver
- 数値IK、複数初期値を並列(Vulkan compute)で解く

### フェーズ5(stretch): MotionGen
- 軌道最適化(簡略化したgradient descent / L-BFGS)をVulkan computeへ移植
- 幾何プランナーは優先度低、必要なら別途検討

### フェーズ6: ドキュメント
- README.md を日本語で作成、ビルド方法・API対応表・使用例を記載

## 4. ディレクトリ構成(案)

include/とsrc/は分けず、`vkmotion/`直下に `.hpp`/`.cpp`を同じ階層で置く。

```
vkMotion/
├── CMakeLists.txt
├── cmake/                      # FetchContent用のcmakeモジュール
├── third_party/
│   └── tinyxml2/                # git submodule
├── vkmotion/
│   ├── types.hpp / types.cpp               # Obstacle, Cuboid, Cylinder, Capsule, Sphere, WorldConfig
│   ├── robot_config.hpp / robot_config.cpp
│   ├── urdf_loader.hpp / urdf_loader.cpp
│   ├── kinematics.hpp / kinematics.cpp     # CudaRobotModel相当のFK
│   ├── sphere_collision.hpp / sphere_collision.cpp  # CPUリファレンス実装
│   ├── robot_world.hpp / robot_world.cpp            # RobotWorld
│   ├── ik_solver.hpp / ik_solver.cpp
│   ├── motion_gen.hpp / motion_gen.cpp
│   ├── vk_context.hpp / vk_context.cpp     # instance/device/VMAラッパ
│   ├── vk_buffer.hpp / vk_buffer.cpp
│   ├── vk_pipeline.hpp / vk_pipeline.cpp
│   └── vk_collision.hpp / vk_collision.cpp # Vulkan dispatcher
├── shaders/                    # *.comp (GLSL), ビルド時にglslcでspvへ
├── test_assets/urdf/           # OpenArm, SO101, URシリーズの .urdf のみ(mesh不要)
├── tests/
│   ├── test_fk.cpp
│   ├── test_collision_cpu.cpp
│   ├── test_collision_gt.cpp   # cuRobo生成のハードコードGTと比較
│   ├── test_collision_vulkan.cpp
│   ├── test_ik_solver.cpp
│   ├── test_motion_gen.cpp
│   └── CMakeLists.txt
├── .github/workflows/ci.yml
├── README.md
├── PLAN.md
├── MANUAL.md                   # 使い方(仮)。クラス/関数シグネチャの暫定実装計画を兼ねる
└── REQUIREMENTS.md
```

## 5. 依存関係

- [cutil](https://github.com/20niship/cutil): `Vec3f`/`Vec4f`/`Mat4f`/`Ref<T>` などに使用。ヘッダオンリーなのでFetchContentまたはsubmoduleで取得
- [doctest](https://github.com/doctest/doctest): テスト用、CMakeオプションで有効時のみFetchContent
- Vulkan SDK (loader + validation layer)
- [VMA (Vulkan Memory Allocator)](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator): FetchContent
- [tinyxml2](https://github.com/leethomason/tinyxml2): URDFパース用。**git submodule**として`third_party/tinyxml2`に配置(フル機能のurdfdomは依存が重いため見送り、URDF読み込みは自前実装)
- [Eigen](https://eigen.tuxfamily.org/): IKSolverのヤコビアン擬似逆行列計算(SVD/減衰最小二乗法)に使用。ヘッダオンリーなのでFetchContent。cutilのVec/Matは固定サイズ用途、可変サイズの線形代数(Jacobian: 6×DOF)はEigenに任せる

## 6. CI/CD方針

- ビルドジョブ: Ubuntu runner + Vulkan SDK(apt or lunarg repo) + CMake
- GPUなしランナーでもVulkan computeを通すため、Mesa lavapipe (software Vulkan ICD) を導入
- `VKMOTION_BUILD_TESTS=ON`でdoctestビルド、ctest実行
- テスト内容: FK単体テスト、CPU衝突判定 vs ハードコードGT、Vulkan版 vs CPU版の一致確認
- Validation layerはDebugビルドで有効化し、CIでもエラーが出ないことを確認

## 7. オープン課題 / 要相談事項

- 障害物クラス名は cuRobo 互換を優先し `Cuboid` を正式名として採用(決定)。詳細シグネチャは [MANUAL.md](./MANUAL.md) 参照
- URDFパーサは tinyxml2(git submodule)を使い自前実装(決定)
- 衝突球は URDF の `<collision>` プリミティブから自動生成し、YAMLは手動調整の上書き用オプションとする(決定)
- SDFベース障害物(VoxelGrid/ESDF)は初期スコープ外、将来対応
- MotionGen(軌道最適化本体)のフル実装は工数が大きいため、まずは衝突判定とIKまでを優先し、進捗を見てスコープ調整する
