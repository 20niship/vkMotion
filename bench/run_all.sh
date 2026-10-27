#!/usr/bin/env bash
# vamp / cuRobo / vkMotion を同じ問題で順番に(計測干渉を避けるため直列で)実行し、bench/RESULTS.md を再生成する。
#   bench/run_all.sh [--limit N(シーンごとの問題数, 0=全部)] [--libs vamp,curobo,vkmotion]
set -euo pipefail
cd "$(dirname "$0")/.."

LIMIT=20
LIBS=vamp,curobo,vkmotion
while [[ $# -gt 0 ]]; do
  case $1 in
    --limit) LIMIT=$2; shift 2 ;;
    --libs) LIBS=$2; shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done
PY=bench/.venv/bin/python
[[ -x $PY ]] || { echo "bench/setup.sh を先に実行してください"; exit 1; }
[[ -f bench/data/problems.json ]] || $PY bench/prepare.py

# 計測前の負荷(他プロセスがCPU/GPUを使っていると時間が汚れる)を記録する。report.pyがRESULTS.mdに載せる
mkdir -p bench/results
{
  echo "load average: $(cut -d' ' -f1-3 /proc/loadavg) (CPU ${NPROC:=$(nproc)} threads)"
  ps -eo pcpu,comm --sort=-pcpu | awk 'NR>1 && $1>=50 {printf "%s(%s%%) ", $2, $1}' | sed 's/ $//'
} > bench/results/env.txt
cat bench/results/env.txt; echo

# GPUを他プロセスが使っていると計測値が汚れるので警告する
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader 2>/dev/null | sed 's/^/[warn] GPU使用中: /' || true

if [[ $LIBS == *vamp* ]]; then $PY bench/run_vamp.py --limit "$LIMIT"; fi
if [[ $LIBS == *curobo* ]]; then $PY bench/run_curobo.py --limit "$LIMIT"; fi
if [[ $LIBS == *vkmotion* ]]; then
  cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DVKMOTION_BUILD_BENCH=ON -DVKMOTION_ENABLE_VALIDATION=OFF -DVKMOTION_NATIVE_ARCH=ON >/dev/null
  cmake --build build-bench -j"$(nproc)" --target bench_vkmotion
  for backend in cpu vulkan; do
    ./build-bench/bench/bench_vkmotion bench/data "bench/results/vkmotion-$backend.json" --backend $backend --limit "$LIMIT"
  done
  # 複数スレッドで1組の木を共有して伸ばす構成(--threads 0 = 自動 = 物理コア数相当)。FK/衝突のスループットは上の-cpuと同じなので飛ばす
  ./build-bench/bench/bench_vkmotion bench/data bench/results/vkmotion-cpu-mt.json --backend cpu --limit "$LIMIT" --threads 0 --skip fkcol
  # 大域探索をGPU(rrt_gpu.comp)で行う構成。初回はロボット専用シェーダのコンパイル(glslc)が走る(以降はキャッシュ)
  # GPUは他のプロセスと共有していると数十msの停止が入ってぶれるので、3回測ってp99が中央の回を採る(採用した回は標準出力に出る。ぶれ幅はnotesの注意を参照)
  for k in 1 2 3; do
    ./build-bench/bench/bench_vkmotion bench/data "bench/results/vkmotion-gpu.run$k.json" --backend vulkan --limit "$LIMIT" --skip fkcol,ik
  done
  $PY - <<'PYEOF'
import json, numpy as np, shutil
runs = []
for k in (1, 2, 3):
    f = f"bench/results/vkmotion-gpu.run{k}.json"
    d = json.load(open(f))
    t = [r["time_ms"] for r in d["plan_js"] if r.get("success")]
    runs.append((float(np.percentile(t, 99)), f))
runs.sort()
print("GPU探索 3回のp99[ms]:", [round(r[0], 2) for r in runs], "-> 採用:", runs[1][1])
shutil.copy(runs[1][1], "bench/results/vkmotion-gpu.json")
PYEOF
  # 同じ問題をバッチ(32件ずつ、問題ごとに別の世界)でまとめてGPU探索。time_msは「バッチ時間/件数」
  ./build-bench/bench/bench_vkmotion bench/data bench/results/vkmotion-gpu-batch.json --backend vulkan --limit "$LIMIT" --batch 32 --skip fkcol,ik,pose
fi
(cd bench && ../$PY report.py > /dev/null)
echo "=> bench/RESULTS.md"
