# vkMotion ベンチマーク結果 (vs VAMP / cuRobo)

- 生成日: 2026-10-27 / vkMotion commit: `030a6d6`(作業ツリー)
- CPU: AMD Ryzen 5 5600X 6-Core Processor / GPU: NVIDIA GeForce RTX 3060, 565.57.01
- ロボット: Franka Panda(7DoF)。衝突球はVAMPの`panda_spherized.urdf`(59球)を3実装で共有、ACMもVAMPのSRDFと同一。
- 問題: MotionBenchMaker (VAMP同梱, `bookshelf_*`/`cage`/`box`/`table_*`) 210件。障害物は直方体(円柱は外接直方体に正規化)。
- 時間はプランナー呼び出し1回のwall clock(IK・時間パラメタライズ込み、シーン更新は除く)。cuRoboはCUDA graph warmup済み。
- 計測開始時のマシン負荷: load average: 17.35 15.87 14.90 (CPU 12 threads) / ollama(1056%) python(64.4%)(負荷が高いとCPU側の時間が悪化する。特にスレッド並列の行)
- 再現手順: [bench/README.md](./README.md)

## 0. vkMotionの改善前後 (同一問題・同一マシン)

| 実装 | 項目 | 問題数 | 成功率(前) | median(前) [ms] | 成功率(後) | median(後) [ms] | 高速化 |
|---|---|---|---|---|---|---|---|
| vkMotion (CPU, 1 thread) | 関節ゴール計画 | 70 | 100.0% | 384.32 | 100.0% | 0.70 | 553x |
| vkMotion (CPU, 1 thread) | 姿勢ゴール計画 | 70 | 81.4% | 426.05 | 100.0% | 1.94 | 220x |
| vkMotion (CPU, 1 thread) | IK | 70 | 0.0% | - | 100.0% | 0.16 | - |
| vkMotion (CPU, 1 thread) | 衝突判定(バッチ µs/check) | - | - | 308.935 | - | 0.143 | 2157x |
| vkMotion (Vulkan + CPU) | 衝突判定(バッチ µs/check) | - | - | 241.210 | - | 0.052 | 4645x |

## 1. FK (精度・速度)

| 実装 | FK [µs/config] | 平均誤差 [mm] | 最大誤差 [mm] |
|---|---|---|---|
| VAMP (CPU SIMD, 1 thread) | 6.070 | 0.0000 | 0.0000 |
| cuRobo v2 (GPU) | 0.151 | 0.0001 | 0.0003 |
| vkMotion (Vulkan + CPU) | 0.241 | 0.0001 | 0.0002 |
| vkMotion (CPU, 1 thread) | 0.258 | 0.0001 | 0.0002 |

## 2. 衝突判定 (精度・スループット)

基準はVAMP。バッチ=10万configを1回で判定(GPU実装はdispatch1回、vkMotion(CPU)はSIMD+8スレッド)。VAMPはpythonバインディング経由の1config呼び出しなので呼び出しオーバーヘッド込み(ネイティブはこれより速い)。

| 実装 | バッチ µs/check (中央値) | checks/s | 1スレッド逐次 µs/check | VAMPとの一致率 | 見逃し(衝突を無衝突と判定) | 過剰検出 |
|---|---|---|---|---|---|---|
| VAMP (CPU SIMD, 1 thread) | 5.014 | 199,459 | - | 100.00% | 0 | 0 |
| cuRobo v2 (GPU) | 0.293 | 3,409,313 | - | 100.00% | 0 | 0 |
| vkMotion (Vulkan + CPU) | 0.052 | 19,255,679 | 2.148 | 100.00% | 0 | 0 |
| vkMotion (CPU, 1 thread) | 0.143 | 6,983,365 | 1.649 | 100.00% | 0 | 0 |

## 3. 関節空間ゴールの軌道生成 (start/goal=関節角)

| 実装 | 問題数 | 成功率 | 時間 median [ms] | mean | p95 | p99 | 時間 median(共通集合) [ms] |
|---|---|---|---|---|---|---|---|
| VAMP (CPU SIMD, 1 thread) | 210 | 100.0% | 1.52 | 2.21 | 4.68 | 5.06 | 1.48 |
| cuRobo v2 (GPU) | 210 | 93.8% | 218.32 | 271.96 | 665.75 | 1,248.73 | 218.32 |
| vkMotion (CPU, 1 thread) | 210 | 100.0% | 0.66 | 2.59 | 13.70 | 23.06 | 0.64 |
| vkMotion (CPU, 8 threads) | 210 | 100.0% | 0.66 | 2.12 | 7.82 | 17.29 | 0.65 |
| vkMotion (GPU探索) | 210 | 100.0% | 1.71 | 2.13 | 5.09 | 7.50 | 1.66 |

| 実装 | 経路長 median [rad] | 経路長(共通集合) | VAMP検証で衝突なし | 動作時間 median [s] | 最大関節加速度 median [rad/s²] |
|---|---|---|---|---|---|
| VAMP (CPU SIMD, 1 thread) | 4.905 | 4.830 | 99.5% | - | - |
| cuRobo v2 (GPU) | 4.329 | 4.329 | 99.5% | 2.00 | 8.8 |
| vkMotion (CPU, 1 thread) | 4.890 | 4.832 | 98.6% | 1.88 | 11.0 |
| vkMotion (CPU, 8 threads) | 4.936 | 4.857 | 97.6% | 1.93 | 11.2 |
| vkMotion (GPU探索) | 4.616 | 4.564 | 98.1% | 1.79 | 10.7 |

共通集合 = 全実装が成功した問題 197 件。動作時間・加速度は時刻を持つ実装(vkMotion, cuRobo)のみ。

## 4. 姿勢ゴールの軌道生成 (IK+軌道生成)

VAMPは関節空間プランナーのため対象外。

| 実装 | 問題数 | 成功率 | 時間 median [ms] | mean | p95 | p99 | 時間 median(共通集合) [ms] |
|---|---|---|---|---|---|---|---|
| cuRobo v2 (GPU) | 210 | 99.5% | 228.08 | 257.56 | 268.11 | 384.65 | 228.08 |
| vkMotion (CPU, 1 thread) | 210 | 100.0% | 1.92 | 3.68 | 14.63 | 23.15 | 1.91 |
| vkMotion (CPU, 8 threads) | 210 | 100.0% | 1.48 | 3.04 | 11.37 | 15.02 | 1.47 |
| vkMotion (GPU探索) | 210 | 100.0% | 3.01 | 3.69 | 8.26 | 10.91 | 3.00 |

| 実装 | 経路長 median [rad] | 経路長(共通集合) | VAMP検証で衝突なし | 動作時間 median [s] | 最大関節加速度 median [rad/s²] | 末端位置誤差 median [mm] | 末端姿勢誤差 median [deg] |
|---|---|---|---|---|---|---|---|
| cuRobo v2 (GPU) | 3.830 | 3.830 | 99.5% | 2.00 | 9.1 | 0.000 | 0.005 |
| vkMotion (CPU, 1 thread) | 4.467 | 4.439 | 99.0% | 1.71 | 10.4 | 0.005 | 0.013 |
| vkMotion (CPU, 8 threads) | 4.517 | 4.513 | 99.0% | 1.74 | 10.5 | 0.005 | 0.011 |
| vkMotion (GPU探索) | 4.223 | 4.213 | 100.0% | 1.64 | 10.4 | 0.035 | 0.008 |

共通集合 = 全実装が成功した問題 209 件。動作時間・加速度は時刻を持つ実装(vkMotion, cuRobo)のみ。

## 5. IK単体 (衝突回避あり)

| 実装 | 問題数 | 成功率 | 時間 median [ms] | p95 | 位置誤差 median [mm] | 姿勢誤差 median [deg] |
|---|---|---|---|---|---|---|
| cuRobo v2 (GPU) | 210 | 99.5% | 5.15 | 18.96 | 0.0002 | 0.0000 |
| vkMotion (CPU, 1 thread) | 210 | 100.0% | 0.13 | 1.21 | 0.0046 | 0.0227 |

## 6. 読み方と、まだ差が残っている点

- **精度**: FK・衝突判定(2000サンプル)・姿勢ゴールの末端誤差は3実装で同等(サブマイクロメートル〜0.01mm台、衝突判定はVAMPと100%一致)。`VAMP検証で衝突なし`はVAMP自身・cuRoboも100%ではない。
  サンプリングで検査するプランナーは、障害物に接した経路を作ると別の検証器の検査点では微小に食い込むため。vkMotionは経路の内部点に`collision_margin`(5mm)の余裕を持たせてこれを抑えている。
- **速度**: 関節ゴール・姿勢ゴールとも、中央値はVAMP(関節ゴール)と同程度以上、cuRoboより2桁速い(cuRoboはGPU上の軌道最適化で、1回の呼び出しに数百ms)。
  一方、裾(p95/p99)はVAMPより長い。RRT-Connectの探索時間のばらつきによるもので、`MotionGenConfig::num_threads`を増やすと一部の問題で緩和される。
- **衝突判定スループット**: 10万config一括ではVulkanバッチ・CPU SIMD(8スレッド)ともcuRoboを上回る。1configずつの逐次判定は1〜2µs(計画ループはこのSIMD版を8点まとめて呼ぶ)。
  VAMPの値はpythonバインディングの呼び出しオーバーヘッドを含むため、ネイティブ呼び出しのVAMPはもっと速い。
- **経路の質**: 経路長はVAMPと同程度、cuRoboより長い(cuRoboは軌道最適化、vkMotionは幾何経路+shortcut)。vkMotionの軌道は折れ線で、速度・加速度の制限は守るが(上の表の最大関節加速度を参照)jerk制限は持たない。
  cuRoboはjerk制限つきの滑らかなB-spline軌道を返す。
- **IK**: vkMotionは運動学的IKの解を衝突判定で棄却する方式(解析ヤコビアン+LM、多シード)。cuRoboは衝突コストを最適化に含める。狭所のゴールではvkMotionは多くのシードが必要になる(時間はms単位で済んでいる)。
- **計測の注意**: VAMP・cuRobo・vkMotionは同じマシンで直列に計測しているが、他プロセスの負荷は制御していない(冒頭の「計測開始時のマシン負荷」)。負荷があるとCPU側(VAMP・vkMotion)の時間は悪化する。
  cuRoboはmax_attempts=10・graph seed 2回目以降で実行(既定は5)。成功率は問題数が少ないとき±数%の幅がある。