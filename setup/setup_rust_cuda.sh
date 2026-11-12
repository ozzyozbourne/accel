#!/usr/bin/env bash
# Sets up a GPU-less Linux machine (tested on Debian 13) to compile the CUDA C++,
# Mojo and Rust (cuda-oxide) code in this repo. Running needs an NVIDIA GPU.
# - CUDA 13 nvcc/headers and Mojo come from pip wheels in a venv ($VENV).
# - cuda-oxide needs clang/llvm 21 (apt.llvm.org) and Rust 1.99.1.
set -euxo pipefail
VENV=${VENV:-$HOME/venvs/gpu}
[ -d "$VENV" ] || uv venv -q -p 3.12 "$VENV"
VIRTUAL_ENV=$VENV uv pip install -q "nvidia-cuda-nvcc>=13,<14" "nvidia-cuda-runtime>=13,<14" \
  "nvidia-cuda-cuobjdump>=13,<14" "nvidia-cuda-cccl>=13,<14" nvidia-curand modular==26.7.0
if ! command -v clang-21 >/dev/null; then
  # Debian 13 has no software-properties-common, so add apt.llvm.org by hand.
  codename=$(. /etc/os-release; echo "$VERSION_CODENAME")
  curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | sudo gpg --dearmor --yes -o /usr/share/keyrings/llvm.gpg
  echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/$codename/ llvm-toolchain-$codename-21 main" \
    | sudo tee /etc/apt/sources.list.d/llvm-21.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq clang-21 libclang-common-21-dev libclang-21-dev llvm-21 >/dev/null
fi
if ! command -v rustup >/dev/null && [ ! -x "$HOME/.cargo/bin/rustup" ]; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
fi
. "$HOME/.cargo/env"
rustup toolchain install 1.99.1 --profile minimal
rustup component add rust-src rustc-dev llvm-tools --toolchain 1.99.1
CARGO_BUILD_JOBS=3 CARGO_INCREMENTAL=0 cargo +1.99.1 install --git https://github.com/NVIDIA/cuda-rust.git cargo-oxide
echo SETUP-DONE
