# vkMotion ベンチマーク結果 (vs VAMP / cuRobo)

- 生成日: 2026-10-27 / vkMotion commit: `7b43117`(作業ツリー)
- CPU: AMD Ryzen 5 5600X 6-Core Processor / GPU: NVIDIA GeForce RTX 3060, 565.57.01
- ロボット: Franka Panda(7DoF)。衝突球はVAMPの`panda_spherized.urdf`(59球)を3実装で共有、ACMもVAMPのSRDFと同一。
- 問題: MotionBenchMaker (VAMP同梱, `bookshelf_*`/`cage`/`box`/`table_*`) 210件。障害物は直方体(円柱は外接直方体に正規化)。
- 時間はプランナー呼び出し1回のwall clock(IK・時間パラメタライズ込み、シーン更新は除く)。cuRoboはCUDA graph warmup済み。
- 計測開始時のマシン負荷: load average: 2.14 2.00 3.72 (CPU 12 threads) / cc1plus(101%)(負荷が高いとCPU側の時間が悪化する。特にスレッド並列の行)
- 再現手順: [bench/README.md](./README.md)

## 0. vkMotionの改善前後 (同一問題・同一マシン)

| 実装 | 項目 | 問題数 | 成功率(前) | median(前) [ms] | 成功率(後) | median(後) [ms] | 高速化 |
|---|---|---|---|---|---|---|---|
| vkMotion (CPU, 1 thread) | 関節ゴール計画 | 70 | 100.0% | 384.32 | 100.0% | 0.29 | 1327x |
| vkMotion (CPU, 1 thread) | 姿勢ゴール計画 | 70 | 81.4% | 426.05 | 100.0% | 0.80 | 530x |
| vkMotion (CPU, 1 thread) | IK | 70 | 0.0% | - | 100.0% | 1.40 | - |
| vkMotion (CPU, 1 thread) | 衝突判定(バッチ µs/check) | - | - | 308.935 | - | 0.040 | 7707x |
| vkMotion (Vulkan + CPU) | 衝突判定(バッチ µs/check) | - | - | 241.210 | - | 0.033 | 7361x |

## 1. FK (精度・速度)

| 実装 | FK [µs/config] | 平均誤差 [mm] | 最大誤差 [mm] |
|---|---|---|---|
| VAMP (CPU SIMD, 1 thread) | 2.695 | 0.0000 | 0.0000 |
| cuRobo v2 (GPU) | 0.099 | 0.0001 | 0.0003 |
| vkMotion (Vulkan + CPU) | 0.199 | 0.0001 | 0.0002 |
| vkMotion (CPU, 1 thread) | 0.203 | 0.0001 | 0.0002 |

## 2. 衝突判定 (精度・スループット)

基準はVAMP。バッチ=10万configを1回で判定(GPU実装はdispatch1回、vkMotion(CPU)はSIMD+8スレッド)。VAMPはpythonバインディング経由の1config呼び出しなので呼び出しオーバーヘッド込み(ネイティブはこれより速い)。

| 実装 | バッチ µs/check (中央値) | checks/s | 1スレッド逐次 µs/check | VAMPとの一致率 | 見逃し(衝突を無衝突と判定) | 過剰検出 |
|---|---|---|---|---|---|---|
| VAMP (CPU SIMD, 1 thread) | 2.464 | 405,845 | - | 100.00% | 0 | 0 |
| cuRobo v2 (GPU) | 0.289 | 3,463,761 | - | 100.00% | 0 | 0 |
| vkMotion (Vulkan + CPU) | 0.033 | 30,516,340 | 0.670 | 100.00% | 0 | 0 |
| vkMotion (CPU, 1 thread) | 0.040 | 24,946,925 | 0.691 | 100.00% | 0 | 0 |

## 3. 関節空間ゴールの軌道生成 (start/goal=関節角)

| 実装 | 問題数 | 成功率 | 時間 median [ms] | mean | p95 | p99 | 時間 median(共通集合) [ms] |
|---|---|---|---|---|---|---|---|
| VAMP (CPU SIMD, 1 thread) | 210 | 100.0% | 0.34 | 0.43 | 0.95 | 1.58 | 0.34 |
| cuRobo v2 (GPU) | 210 | 93.8% | 201.68 | 259.51 | 633.57 | 1,174.57 | 201.68 |
| vkMotion (CPU, 1 thread) | 210 | 100.0% | 0.28 | 1.18 | 6.62 | 11.58 | 0.28 |
| vkMotion (CPU, 8 threads) | 210 | 100.0% | 0.31 | 1.00 | 4.99 | 11.06 | 0.30 |
| vkMotion (GPU探索) | 210 | 100.0% | 0.32 | 1.00 | 4.79 | 7.26 | 0.32 |

| 実装 | 経路長 median [rad] | 経路長(共通集合) | VAMP検証で衝突なし | 動作時間 median [s] | 最大関節加速度 median [rad/s²] |
|---|---|---|---|---|---|
| VAMP (CPU SIMD, 1 thread) | 4.905 | 4.830 | 99.5% | - | - |
| cuRobo v2 (GPU) | 4.329 | 4.329 | 99.5% | 2.00 | 8.8 |
| vkMotion (CPU, 1 thread) | 4.890 | 4.834 | 98.6% | 1.88 | 11.1 |
| vkMotion (CPU, 8 threads) | 4.936 | 4.857 | 99.0% | 1.93 | 11.1 |
| vkMotion (GPU探索) | 4.939 | 4.864 | 99.0% | 1.96 | 11.1 |

共通集合 = 全実装が成功した問題 197 件。動作時間・加速度は時刻を持つ実装(vkMotion, cuRobo)のみ。

## 4. 姿勢ゴールの軌道生成 (IK+軌道生成)

VAMPは関節空間プランナーのため対象外。

| 実装 | 問題数 | 成功率 | 時間 median [ms] | mean | p95 | p99 | 時間 median(共通集合) [ms] |
|---|---|---|---|---|---|---|---|
| cuRobo v2 (GPU) | 210 | 99.5% | 197.97 | 238.55 | 254.56 | 292.59 | 197.97 |
| vkMotion (CPU, 1 thread) | 210 | 100.0% | 0.88 | 1.75 | 6.63 | 9.98 | 0.88 |
| vkMotion (CPU, 8 threads) | 210 | 100.0% | 0.95 | 1.67 | 5.35 | 8.81 | 0.93 |
| vkMotion (GPU探索) | 210 | 100.0% | 0.85 | 1.63 | 5.68 | 8.98 | 0.85 |

| 実装 | 経路長 median [rad] | 経路長(共通集合) | VAMP検証で衝突なし | 動作時間 median [s] | 最大関節加速度 median [rad/s²] | 末端位置誤差 median [mm] | 末端姿勢誤差 median [deg] |
|---|---|---|---|---|---|---|---|
| cuRobo v2 (GPU) | 3.830 | 3.830 | 99.5% | 2.00 | 9.1 | 0.000 | 0.005 |
| vkMotion (CPU, 1 thread) | 4.430 | 4.421 | 99.5% | 1.71 | 10.4 | 0.005 | 0.013 |
| vkMotion (CPU, 8 threads) | 4.547 | 4.537 | 100.0% | 1.75 | 10.6 | 0.007 | 0.012 |
| vkMotion (GPU探索) | 4.563 | 4.559 | 100.0% | 1.75 | 10.9 | 0.011 | 0.010 |

共通集合 = 全実装が成功した問題 209 件。動作時間・加速度は時刻を持つ実装(vkMotion, cuRobo)のみ。

## 5. IK単体 (衝突回避あり)

| 実装 | 問題数 | 成功率 | 時間 median [ms] | p95 | 位置誤差 median [mm] | 姿勢誤差 median [deg] |
|---|---|---|---|---|---|---|
| cuRobo v2 (GPU) | 210 | 99.5% | 3.28 | 8.93 | 0.0002 | 0.0000 |
| vkMotion (CPU, 1 thread) | 210 | 100.0% | 1.21 | 2.03 | 0.0046 | 0.0227 |

## 6. 読み方と、まだ差が残っている点


- **VAMPとの比較(正直なところ)**: 中央値はVAMPと同程度(vkMotion CPUが0.3ms前後、VAMPが0.35ms)で、「VAMPの数倍速い」には届いていない。
  裾はVAMPが明確に良い(VAMP p95≈1ms/p99≈1.6ms に対して、vkMotion CPUはp95≈6ms/p99≈12ms、GPU探索併用でp99≈7ms)。難しいケース(`cage`)の1問あたりの上限は
  vkMotion CPU 約15〜20ms / GPU探索 約7〜9ms / VAMP 約3ms。VAMPの時間は`time_ms`=1回目の実行(plan+simplify、`time_ms_median3`は3回の中央値)。
- **なぜ裾が長いか**: VAMPの`cage`は約1000反復を約0.6msで回す(1反復0.6µs)。vkMotionは1反復あたり約3µsで、ロボット専用に展開したカーネル(`gpu_codegen`)を使っても
  8コンフィグの衝突判定が約0.6〜0.9µsかかる(VAMPは約0.2µs)。主因は箱×リンクグループのブロードフェーズで、VAMPのように障害物を距離順に並べて早期に打ち切っても改善しなかった。
- **GPU探索**(`MotionGenConfig::gpu_search`): 1本の木を多数のワープで共有して伸ばすVulkan compute版RRT-Connect。ラウンドをdispatchに分けて1回のsubmitに並べる(CUDA graphに相当)。
  submitの固定費が約0.5msあるので、まずCPUで`gpu_handoff_s`(1ms)試し、解けなければGPUに渡す構成が既定。GPUは裾(難しい問題)を短くする用途で、中央値は縮めない。
  GPUはアイドルが続くとクロックが落ち、疎な呼び出しでは遅くなる(固定クロックにすると改善する)。
- **精度**: FK・衝突判定(2000サンプル)・姿勢ゴールの末端誤差は各実装で同等(サブマイクロメートル〜0.01mm台、衝突判定はVAMPと100%一致)。`VAMP検証で衝突なし`はVAMP自身・cuRoboも100%ではない。
  サンプリングで検査するプランナーは、障害物に接した経路を作ると別の検証器の検査点では微小に食い込むため。vkMotionは経路の内部点に`collision_margin`(5mm)の余裕を持たせてこれを抑えるが、
  端点が障害物すれすれで木が伸びないときだけ余裕を縮めるので、その問題では食い込む経路が出うる。
- **衝突判定スループット**: 10万config一括ではVulkanバッチ・CPU SIMD(8スレッド)ともcuRoboを上回る。VAMPの値はpythonバインディングの呼び出しオーバーヘッドを含む。
- **cuRoboとの関係**: cuRoboは姿勢ゴールをGPU上の多シードIK + 多シード軌道最適化(CUDA graph)で解き、PRMグラフは再試行時のシード用。スループット重視で1回数百ms。
  vkMotionは幾何経路(RRT-Connect)なので2桁速いが、経路は折れ線でjerk制限を持たず、経路長はcuRoboより長い。
- **IK**: vkMotionは運動学的IKの解を衝突判定で棄却する方式(解析ヤコビアン+LM、多シード)。cuRoboは衝突コストを最適化に含める。
- **計測の注意**: 他プロセスの負荷は制御していない(冒頭の「計測開始時のマシン負荷」)。以前の版はollamaがCPUを約1000%使っていた状態の数値で、VAMPが約1.5msに見えていたが、
  負荷が無い状態では0.35msだった。GPUも他セッションと共有しており、GPU探索の数値は±数十%ぶれる。cuRoboはmax_attempts=10・graph seed 2回目以降で実行(既定は5)。
