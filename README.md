# ellipsoids_public

Code accompanying:

> Hong, F., Bouhassira, R., Chow, J., Sanders, C., Shvartsman, M., Guan, P., Williams, A. H., & Brainard, D. H. (2026). **Comprehensive characterization of human color discrimination thresholds.** *eLife*, 14:RP108943. https://doi.org/10.7554/eLife.108943.2

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/fh862/ellipsoids_public.git
cd ellipsoids_public
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 3. Install the package and all dependencies

```bash
pip install -e ellipsoids/
```

JAX (CPU build) is pulled in automatically.  For GPU acceleration see below.

This installs the package in editable mode so that `from core import ...`, `from analysis import ...`, etc. work from any script or notebook without manually adjusting `sys.path`.

### GPU acceleration (optional)

**NVIDIA GPU (CUDA 12):**
```bash
pip install "jax[cuda12]"
```
Run this after `pip install -e ellipsoids/` to replace the CPU JAX build.

**Apple Silicon (M1/M2/M3/M4):** GPU acceleration is not available for this
codebase.  The code requires 64-bit floating point
(`jax_enable_x64 = True`), which the Apple Metal JAX plugin (`jax-metal`)
does not support.  CPU-only performance on Apple Silicon is still very good.

### Future sessions

Each time you open a new terminal, activate the environment before running code:

```bash
cd ellipsoids_public
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

---

## Dataset

The experimental data are hosted on OSF at <https://osf.io/k27js>.

To download the data and work through example analyses, see
**[BrainardLab/wppmpy_public](https://github.com/BrainardLab/wppmpy_public)**,
which provides a targeted download script and example notebooks that illustrate
how to load and use the dataset alongside this code.

---

## Path configuration

Some analysis and experiment scripts need local machine-specific roots for Dropbox data, network disks, and calibration files. These paths are stored in JSON files under `config/` instead of being hardcoded in each script.

On macOS, edit:

```text
config/hardcoded_paths_mac.json
```

Example:

```json
{
  "dropbox_root_mac": "/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/",
  "dropbox_root_mac_elps": "/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_analysis/",
  "network_disk_mac": "/Volumes/BrainardLabShares/brainardlab/",
  "psychtoolbox_files": "/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/FilesFromPsychtoolbox/"
}
```

Scripts read these values through `analysis.utils_load`:

```python
from analysis.utils_load import get_path

base_dir = get_path("dropbox_root_mac")
```

By default, macOS loads `hardcoded_paths_mac.json` and Windows loads `hardcoded_paths_windows.json`. To use a custom file, set:

```bash
export ELLIPSOIDS_PATH_CONFIG=/path/to/your_paths.json
```

---

## Running tests

```bash
cd ellipsoids
python -m pytest --tb=short -v
```

---

## Using this code from another project

After installing with `pip install -e /path/to/ellipsoids_public/ellipsoids`, the packages (`core`, `analysis`, `plotting`, …) are available in that environment:

```python
from core.wishart_process import WishartProcessModel
from analysis.color_thres import color_thresholds
```

---

## Related repositories

Code that builds directly on this repository:

- **[BrainardLab/wppmpy_public](https://github.com/BrainardLab/wppmpy_public)** — companion toolbox for the WPPM.  Contains a script to download the Hong et al. (2025) OSF dataset, example notebooks reproducing key figures from the paper, and AEPsych-based experiment runner code for collecting new color-discrimination data.

---

## Repository structure

| Directory | Contents |
|-----------|----------|
| `core/` | Wishart Process model, optimisation, Chebyshev basis |
| `analysis/` | Data loading, cross-validation, threshold estimation |
| `plotting/` | Visualisation utilities |
| `cieLab/` | CIE Lab colour space utilities |
| `eval/` | Model evaluation and comparison scripts |
| `export/` | Data export to CSV and other formats |
| `sim/` | Simulation scripts |
| `model_demo/` | Stand-alone model demonstrations |
| `scripts/` | Utility scripts (e.g. data download) |
| `dconfig/` | Experiment configuration files |
| `fit/` | Model fitting scripts |
| `tests/` | Automated test suite |
| `data/` | Downloaded dataset — **not tracked by git** |

*(All paths above are relative to the `ellipsoids/` subdirectory.)*

The following directories live at the **repository root** (not inside `ellipsoids/`):

| Directory | Contents |
|-----------|----------|
| `aepsy/expt/` | Experiment scripts (require AEPsych — see below) |
| `aepsy/sim/` | Adaptive simulation scripts (require AEPsych — see below) |

---

## AEPsych-dependent code

The scripts under `aepsy/expt/` and `aepsy/sim/` require `aepsych==0.7.3`.  AEPsych pins `numpy<2.0`, which is incompatible with JAX 0.5+, so a **separate virtual environment** is needed.

### 1. Create a new virtual environment (separate from the JAX one)

```bash
python -m venv .venv-aepsych
source .venv-aepsych/bin/activate
```

### 2. Install AEPsych and compatible JAX

```bash
pip install -e ./aepsy
```

This installs `aepsych==0.7.3`, its dependencies (torch, botorch, numpy 1.x, …), and `jax<0.5` (the latest JAX compatible with numpy 1.x).  For NVIDIA GPU support replace the JAX install afterwards:

```bash
pip install "jax[cuda12]<0.5"
```

### 3. Make the analysis and plotting modules importable

The `aepsy/` scripts import from `analysis.*`, `plotting.*`, etc.  Install the package source without reinstalling its JAX dependencies (which would conflict with numpy 1.x):

```bash
pip install --no-deps -e ./ellipsoids
```

After this, `from analysis.color_thres import color_thresholds` and similar imports work in the AEPsych environment.

---

## Pre-commit hooks

This repository uses [pre-commit](https://pre-commit.com) for basic file hygiene.
Install once per clone:

```bash
pip install pre-commit
pre-commit install
```
