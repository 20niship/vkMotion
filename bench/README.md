# bench — vkMotion vs VAMP vs cuRobo

同じロボット・同じ衝突球・同じ障害物・同じ問題で3つのプランナーを動かし、速度と精度を比較する。結果は [RESULTS.md](./RESULTS.md)。

| 実装 | 何を測るか |
|---|---|
| [VAMP](https://github.com/KavrakiLab/vamp) 0.6.4 (pip) | FK / 衝突判定 / 関節空間RRT-Connect+simplify (CPU SIMD, 1スレッド) |
| [cuRobo](https://github.com/NVlabs/curobo) v0.8.0 (v2) | FK / 衝突判定 / IK / 関節空間・姿勢ゴールの軌道最適化 (GPU) |
| vkMotion (このリポジトリ) | 同上 (CPU / Vulkan) |

## 比較条件(公平にするためにそろえたもの)

- ロボット: Franka Panda 7DoF。衝突球は VAMP の `panda_spherized.urdf`(59球)を3実装に同一で渡す(cuRoboにはYAMLに変換、vkMotionにはメッシュ無しURDFの`<collision><sphere>`として渡す)。
- 自己衝突の除外ペア(ACM)も VAMP の `panda.srdf` と同一。
- 問題: VAMP 同梱の MotionBenchMaker (`bookshelf_small/tall/thin`, `box`, `cage`, `table_pick`, `table_under_pick`) のうち VAMP で start/goal が有効な699問。障害物は直方体のみ(VAMPが円柱をカプセルで過大近似する都合で、円柱は外接直方体に正規化。cuRoboのシーン衝突はOBB/mesh/voxelのみ)。
- 精度の基準は VAMP: FK 末端位置、2000サンプルのコンフィグの衝突有無、計画された経路が VAMP の `validate_motion` で衝突なしか、姿勢ゴールの末端誤差を VAMP の FK で測る。
- 時間はプランナー呼び出し1回のwall clock。cuRoboは warmup 済み(CUDA graph)。VAMPは python バインディング経由(`simplify`込み)。

## 使い方

```sh
bench/setup.sh               # bench/.venv に vamp/cuRobo、bench/.deps に問題セット(VAMP)とcuRoboを用意 (uv, git, NVIDIA GPU が必要)
bench/run_all.sh             # 各シーン先頭20問で全実装を実行して RESULTS.md を再生成
bench/run_all.sh --limit 0   # 699問すべて
bench/run_all.sh --libs vkmotion      # vkMotionだけ再計測 (1スレッド版と8本並列版の両方)
```

注意:

- GPUを他のプロセス(ollama, 学習ジョブなど)が使っているとcuRoboの時間が汚れる。`run_all.sh`は使用中プロセスがあると警告する。
- ドライバがCUDA 12系だと、`warp-lang`の最新版(CUDA 13同梱)が動かないため `setup.sh` は 1.10.1 に固定している。
- `bench/data/`(問題・URDF・サンプル)、`bench/results/`(生JSON)、`.venv`/`.deps` はgit管理外。再生成できる。

## ファイル

- `prepare.py` — VAMPの問題tar・球モデルから、共通入力(`data/problems.json`, `data/panda.urdf`, `data/panda_curobo.yml`, `data/acm.json`, `data/samples.json`)を作る
- `run_vamp.py` / `run_curobo.py` / `bench_vkmotion.cpp` — 実装ごとの計測。結果を同じスキーマの `results/<lib>.json` に書く
- `report.py` — JSONを突き合わせて `RESULTS.md` を作る(経路のVAMP検証・末端誤差・加速度の有限差分はここで計算)
- `bench_vkmotion --check` — デバッグ用。`[{scene,index,q}]` を渡すと参照実装と高速パスの衝突判定を並べて返す
