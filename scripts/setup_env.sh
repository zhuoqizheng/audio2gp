#!/usr/bin/env bash
# Build the project-local environment without touching the system Python.
#
# Torch/torchaudio are reused from the user site. The user-site .pth is added
# only after every pip install, otherwise pip sees the user's packages as
# satisfying the pinned requirements and skips installing them into the venv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV="$ROOT/.venv"
VENV_PY="$VENV/bin/python"

command -v ffmpeg >/dev/null || { echo "ffmpeg is required" >&2; exit 1; }
"$PYTHON" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        "This build is tested on Python 3.10 (the tflite-runtime wheel is "
        f"Py3.10/Linux); current: {sys.version.split()[0]}"
    )
PY

if [ ! -x "$VENV_PY" ]; then
    "$PYTHON" -m venv "$VENV"
fi

PURELIB="$("$VENV_PY" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
USER_SITE="$("$VENV_PY" -c 'import site; print(site.getusersitepackages())')"
PTH="$PURELIB/_audio2gp_user_site.pth"

# A stale .pth must not influence this pip run.
rm -f "$PTH"

"$VENV_PY" -m pip install --upgrade pip setuptools wheel
"$VENV_PY" -m pip install -r requirements.txt -c constraints.txt

# Demucs and its runtime deps without dependency resolution. In particular,
# julius declares torch>=1.13, so a normal install would download a second
# (CUDA 13) torch before the user-site .pth exists. Versions come from
# constraints.txt; huggingface-hub is explicit so Demucs does not rely on
# Gradio's dependency tree.
"$VENV_PY" -m pip install --no-deps 'demucs==4.1.0'
"$VENV_PY" -m pip install --no-deps -c constraints.txt \
    einops huggingface-hub 'julius>=0.2.3' 'lameenc>=1.2' pyyaml safetensors \
    'sphn>=0.1.12' tqdm

"$VENV_PY" -m pip install -e . --no-deps

printf '%s\n' "$USER_SITE" > "$PTH"

"$VENV_PY" - <<'PY'
import importlib.metadata as md
import pathlib
import sys

import torch

print("python:", sys.version.split()[0])
print("torch:", torch.__version__, torch.__file__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
if "site-packages" not in str(pathlib.Path(torch.__file__).parent) or ".local" not in str(torch.__file__):
    raise SystemExit(
        "torch was not loaded from the user site. Remove .venv and install "
        "CUDA torch 2.6.0+cu124 into the user site before rerunning."
    )
if not torch.cuda.is_available():
    print("WARNING: CUDA is not available; the app will fall back to CPU and be very slow.", file=sys.stderr)

import basic_pitch  # noqa: E402
import demucs  # noqa: E402
import gradio  # noqa: E402
import guitarpro  # noqa: E402
import madmom_infer  # noqa: E402
import resampy  # noqa: E402  (asserts pkg_resources / setuptools<81 is intact)

for dist, module in (
    ("demucs", demucs),
    ("basic-pitch", basic_pitch),
    ("PyGuitarPro", guitarpro),
    ("madmom-infer", madmom_infer),
    ("gradio", gradio),
):
    print(f"{dist}: {md.version(dist)}  ({pathlib.Path(module.__file__).parent})")

# Basic Pitch 0.4.0 on Linux/Py3.10 uses the bundled TFLite model.
try:
    import tflite_runtime  # noqa: F401
    print("basic-pitch backend: tflite-runtime (bundled model)")
except ImportError as exc:
    raise SystemExit(f"tflite-runtime is missing: {exc}")
PY

echo
echo "Downloading Demucs weights (first run only)..."
"$VENV_PY" - <<'PY'
from demucs.pretrained import get_model

for name in ("htdemucs_ft", "htdemucs"):
    try:
        model = get_model(name)
        print(f"  {name}: ok ({type(model).__name__})")
    except Exception as exc:  # network/HF failures are actionable
        raise SystemExit(f"Could not download {name}: {exc}") from exc
PY

echo
echo "Downloading downbeat model weights (first run only)..."
"$VENV_PY" - <<'PY'
try:
    from madmom_infer.features.downbeats import RNNDownBeatProcessor

    RNNDownBeatProcessor()
    print("  madmom downbeat BLSTM: ok")
except Exception as exc:
    # The app falls back to librosa accent-phase alignment, so this is a
    # warning rather than a setup failure.
    print(f"  WARNING: downbeat model unavailable ({exc}); librosa fallback will be used")
PY

echo
echo "Environment ready."
echo "  CLI:  $VENV/bin/python -m audio2gp.cli --help"
echo "  Web:  $VENV/bin/audio2gp-web   (http://127.0.0.1:7860)"
