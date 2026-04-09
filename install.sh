#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v python >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
  echo "Error: Python is not installed."
  exit 1
fi

PYTHON_CMD="python"
if ! command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python3"
fi

echo "Installing UniversalVTG release in: $ROOT_DIR"
echo "Using CUDA_HOME=${CUDA_HOME:-<unset>}"

if ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
  echo "pip not found; bootstrapping with ensurepip..."
  $PYTHON_CMD -m ensurepip --upgrade
fi

$PYTHON_CMD -m pip install --upgrade pip setuptools wheel

echo "Installing pinned PyTorch stack (cu124 wheels)..."
$PYTHON_CMD -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers \
  --index-url https://download.pytorch.org/whl/cu124
$PYTHON_CMD -m pip install torchcodec==0.1 --index-url https://download.pytorch.org/whl/cu124

echo "Installing release requirements..."
$PYTHON_CMD -m pip install -r requirements.txt

if [ -d .git ]; then
  echo "Updating submodules..."
  git submodule update --init --recursive perception_models
fi

if [ ! -d perception_models ]; then
  echo "Error: perception_models/ is missing. Clone with --recurse-submodules or run git submodule update --init --recursive."
  exit 1
fi

PY_MM="$($PYTHON_CMD - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
echo "Detected Python version: ${PY_MM}"

if $PYTHON_CMD - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
then
  echo "Installing perception_models in editable mode..."
  $PYTHON_CMD -m pip install -e perception_models
else
  echo "Python < 3.11 detected; skipping editable install of perception_models."
  echo "The release scripts will fall back to the checked-out submodule path for imports."
fi

echo "Building 1D NMS extension..."
(
  cd libs/nms
  $PYTHON_CMD setup_nms.py build_ext --inplace
)

echo "Running import/config sanity checks..."
$PYTHON_CMD - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str((Path.cwd() / 'perception_models').resolve()))

import yaml
from libs.core import load_opt

with open('opts/eval/multidata_evaluation.yaml', 'r') as f:
    eval_cfg = yaml.load(f, Loader=yaml.FullLoader)
assert 'eval' in eval_cfg and 'datasets' in eval_cfg['eval']

load_opt('opts/pretrain/multinode_multidataset_pretraining.yaml', is_training=True)
load_opt('opts/finetune/multidataset_finetuning.yaml', is_training=True)

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
import feature_extraction.extract_visual_features
import feature_extraction.extract_text_features
print('Config load and feature extraction imports OK')
PY

echo "Installation complete."
