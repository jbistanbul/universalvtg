# Installation

## Prerequisites

- Linux with CUDA GPU
- Conda or Miniconda
- CUDA toolkit (12.x recommended)

## CUDA bootstrap

This release expects a CUDA toolchain. Make sure `CUDA_HOME` is set in your shell (e.g., `~/.bashrc`):

```bash
export CUDA_HOME=/usr/local/cuda   # adjust to your CUDA install path
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Setup

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/jbistanbul/universalvtg.git
cd universal_vtg
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate universal_vtg_release
```

### 3. Run the install script

```bash
bash install.sh
```

`install.sh` performs the following:

1. Installs the pinned PyTorch / torchvision / torchaudio wheel set
2. Installs release Python dependencies from `requirements.txt`
3. Initializes and installs `perception_models` submodule (editable mode, requires Python ≥ 3.11)
4. Builds the 1D NMS C extension
5. Runs basic config/import sanity checks


## Notes

- The `perception_models` submodule is subject to the upstream [Meta/Fair license](https://github.com/facebookresearch/perception_models).
- The 1D NMS extension should be rebuilt whenever the Python / PyTorch build environment changes.
- Python ≥ 3.11 is required for editable install of `perception_models`. On older Python, the install script falls back to using the checked-out submodule path directly.
