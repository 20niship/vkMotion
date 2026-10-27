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
  # RRT-Connectを独立シードで並列起動する構成(レイテンシの裾を削る)。FK/衝突のスループットは上の-cpuと同じなので飛ばす
  ./build-bench/bench/bench_vkmotion bench/data bench/results/vkmotion-cpu-mt.json --backend cpu --limit "$LIMIT" --threads 8 --skip fkcol
  # 大域探索をGPU(rrt_gpu.comp)で行う構成。初回はロボット専用シェーダのコンパイル(glslc)が走る(以降はキャッシュ)
  ./build-bench/bench/bench_vkmotion bench/data bench/results/vkmotion-gpu.json --backend vulkan --limit "$LIMIT" --gpu-search --skip fkcol,ik
  # 同じ問題をバッチ(32件ずつ、問題ごとに別の世界)でまとめてGPU探索。time_msは「バッチ時間/件数」
  ./build-bench/bench/bench_vkmotion bench/data bench/results/vkmotion-gpu-batch.json --backend vulkan --limit "$LIMIT" --batch 32 --skip fkcol,ik,pose
fi
(cd bench && ../$PY report.py > /dev/null)
echo "=> bench/RESULTS.md"
