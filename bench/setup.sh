#!/usr/bin/env bash
# vamp / cuRobo の比較環境を bench/.venv と bench/.deps に作る(どちらもgit管理外)。
#   前提: uv, git, CUDA対応GPU+ドライバ。cuRobo v2はwarp(CUDAカーネルJIT)を使うのでnvccは不要。
#   ドライバが古い(CUDA 12.x)場合に備え、warp-langはCUDA 13同梱の新しい版ではなく1.10.1に固定している。
set -euo pipefail
cd "$(dirname "$0")"

VAMP_REF=27cb9b66ebd8d101ce88d330f3b028398afc1cfa   # 問題セット(MotionBenchMaker)と球モデルの取得元
CUROBO_TAG=v0.8.0
VAMP_PLANNER_VERSION=0.6.4

mkdir -p .deps
[[ -d .deps/vamp ]] || { git clone https://github.com/KavrakiLab/vamp .deps/vamp; git -C .deps/vamp checkout "$VAMP_REF"; }
[[ -d .deps/curobo ]] || git clone --depth 1 --branch "$CUROBO_TAG" https://github.com/NVlabs/curobo .deps/curobo

[[ -d .venv ]] || uv venv .venv --python 3.12
export VIRTUAL_ENV="$PWD/.venv"
uv pip install "vamp-planner==$VAMP_PLANNER_VERSION" numpy pyyaml
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".deps/curobo[cu12-torch]"
uv pip install "warp-lang==1.10.1"

.venv/bin/python prepare.py
echo "setup done. run: bench/run_all.sh"
