#!/usr/bin/env bash
# Builds async/rust with cuda-oxide (no GPU needed to compile). Output:
# async/rust/target/release/async_test and async/rust/async_test.ptx.
set -euxo pipefail
. "$HOME/.cargo/env"
C=${VENV:-$HOME/venvs/gpu}/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$C CUDA_PATH=$C CUDA_TOOLKIT_PATH=$C PATH=$C/bin:$PATH
export CUDA_OXIDE_LLC=/usr/bin/llc-21 CUDA_OXIDE_DEVICE_ARCH=sm_80
export CARGO_BUILD_JOBS=3 CARGO_INCREMENTAL=0
cd "$(dirname "$0")/../async/rust"
cargo +1.99.1 clean --release -p async_test  # cargo oxide sometimes misses source edits
cargo +1.99.1 oxide build -- --release
