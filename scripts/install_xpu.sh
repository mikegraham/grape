#!/usr/bin/env bash
# Install grape (editable) into a venv with Intel XPU torch.
#
#   scripts/install_xpu.sh [VENV_DIR]     # default: ~/venvs/grape
#
#   WHEELS=dir   install torch/torchvision only from wheel files in this folder
#                (no index, no download). Default: ~/wheels if it holds a torch
#                XPU wheel, else the XPU index. WHEELS= (empty) forces the index.
#   REPO=path    checkout to install; default: the one this script is in. It
#                must contain the GPU code (grape.model.pick_device).
#   DEV=1        also install the dev tools (pytest, ruff, mypy)
#
# grape is installed against a constraints file pinning the exact
# torch/torchvision just installed, so if anything tries to swap in PyPI's
# CUDA torch (nvidia-* wheels), pip fails naming the package instead of
# downloading gigabytes.
set -euo pipefail

venv="${1:-$HOME/venvs/grape}"
repo="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
xpu_index="https://download.pytorch.org/whl/xpu"

if [ -z "${WHEELS+set}" ] && compgen -G "$HOME/wheels/torch-*+xpu-*.whl" >/dev/null; then
  WHEELS="$HOME/wheels"
fi

if ! grep -q 'def pick_device' "$repo/grape/model.py" 2>/dev/null; then
  echo "ERROR: $repo has no GPU support (grape/model.py lacks pick_device)." >&2
  echo "       Point REPO= at a checkout of the xpu-acceleration branch." >&2
  exit 1
fi

echo "== python: $(command -v python3) ($(python3 --version))"
echo "== repo: $repo"
python3 -m venv "$venv"
py="$venv/bin/python"
"$py" -m pip install --quiet --upgrade pip

if [ -n "${WHEELS:-}" ]; then
  echo "== torch + torchvision from $WHEELS (offline)"
  "$py" -m pip install --no-index --find-links "$WHEELS" torch torchvision
else
  echo "== torch + torchvision from $xpu_index"
  "$py" -m pip install torch torchvision --index-url "$xpu_index"
fi

constraints="$venv/torch-constraints.txt"
"$py" - "$constraints" <<'EOF'
import sys
from importlib.metadata import version
pins = [f"{p}=={version(p)}" for p in ("torch", "torchvision")]
open(sys.argv[1], "w").write("\n".join(pins) + "\n")
print("== pinned:", ", ".join(pins))
if "+xpu" not in version("torch"):
    sys.exit("ERROR: got a torch without +xpu; stopping")
EOF

target="$repo"
[ -n "${DEV:-}" ] && target="$repo[dev]"
echo "== grape (editable) from $target"
"$py" -m pip install -e "$target" -c "$constraints"

echo "== check"
"$py" - <<'EOF'
import torch
print("torch", torch.__version__, "| xpu available:", torch.xpu.is_available())
if torch.xpu.is_available():
    print("device:", torch.xpu.get_device_name(0))
from grape.model import pick_device
print("grape picks:", pick_device())
EOF
echo "== done: $venv/bin/grape"
