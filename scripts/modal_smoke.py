"""Install grape on a fresh machine (Modal) and check its main flows.

    modal run scripts/modal_smoke.py                # CPU
    MODAL_GPU=L4 modal run scripts/modal_smoke.py   # an NVIDIA GPU

Installs this checkout with a plain `pip install`, as a user would, then:
indexes tests/fixtures (loads torch, downloads the default model), and
searches for a new keyword, which must not import torch. Exits nonzero
if a step fails.
"""

import os
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
GPU = os.environ.get("MODAL_GPU")  # None: CPU only

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir(REPO / "grape", "/src/grape", copy=True, ignore=["**/__pycache__"])
    .add_local_dir(REPO / "tests" / "fixtures", "/src/tests/fixtures", copy=True)
    .add_local_file(REPO / "pyproject.toml", "/src/pyproject.toml", copy=True)
    .add_local_file(REPO / "README.md", "/src/README.md", copy=True)
    .add_local_file(REPO / "LICENSE", "/src/LICENSE", copy=True)
    .run_commands("pip install /src")
)
app = modal.App("grape-smoke", image=image)


def _run(cmd: str, show: bool = True) -> str:
    print(f"\n$ {cmd}", flush=True)
    result = subprocess.run(
        cmd, shell=True, cwd="/src", capture_output=True, text=True,
    )
    if show:
        print(result.stdout + result.stderr, flush=True)
    if result.returncode != 0:
        raise SystemExit(f"failed ({result.returncode}): {cmd}")
    return result.stdout + result.stderr


@app.function(gpu=GPU, timeout=1800)
def smoke() -> None:
    search = "grape --cache /tmp/grape.db -v -s -n 3 -R tests/fixtures --keywords"
    _run("python -c 'import grape; print(\"grape\", grape.__version__)'")
    _run(f"{search} dog")  # first run: indexes, loads torch
    out = _run(f"{search} wolf")  # new keyword over indexed photos
    if "encoded without torch" not in out:
        raise SystemExit("the new keyword was encoded with torch")
    importtime = _run(f"python -X importtime -m {search} sunset", show=False)
    if any(line.split("|")[-1].strip() == "torch" for line in importtime.splitlines()):
        raise SystemExit("a new-keyword search imported torch")
    print("\nsmoke test passed", flush=True)


@app.local_entrypoint()
def main() -> None:
    smoke.remote()
