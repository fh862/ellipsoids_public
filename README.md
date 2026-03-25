# ellipsoids_eLife2025 (dev)

Code accompanying:

> Hong, F., Kawakita, G., Yerxa, T., Garg, A., Grüner, M., Wool, L. E., Solomon, S. G., Brainard, D. H., & Ma, W. J. (2025). **Comprehensive characterization of human color discrimination thresholds.** *eLife.* https://elifesciences.org/reviewed-preprints/108943v1

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/fh862/ellipsoids_eLife2025.git
cd ellipsoids_eLife2025
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

### 3. Install JAX

JAX installation depends on your hardware. Choose **one**:

```bash
# CPU only
pip install jax jaxlib

# NVIDIA GPU (CUDA 12)
pip install "jax[cuda12]"
```

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for other platforms.

### 4. Install this package and remaining dependencies

```bash
pip install -e ellipsoids/
```

This installs the package in editable mode so that `from core import ...`, `from analysis import ...`, etc. work from any script or notebook without manually adjusting `sys.path`.

---

## Dataset

The experimental data are hosted on OSF at <https://osf.io/k27js>.
Download everything into `data/` with:

```bash
python ellipsoids/scripts/download_osf_data.py
```

Pass `--data-dir /your/preferred/path` to save elsewhere. The script uses only Python stdlib (no extra packages needed).

---

## Using this code from another project

After installing with `pip install -e /path/to/ellipsoids_eLife2025/ellipsoids`, the packages (`core`, `analysis`, `plotting`, …) are available in that environment:

```python
from core.wishart_process import WishartProcessModel
from analysis.color_thres import color_thresholds
```

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
| `tests/` | Automated test suite |
| `data/` | Downloaded dataset — **not tracked by git** |
| `fit_4d_human.py` | Main fitting script (4-D stimulus space) |
| `fit_6d_human.py` | Main fitting script (6-D stimulus space) |
| `fit_MOCS.py` | Fitting script for MOCS paradigm |

---

## Pre-commit hooks

This repository uses [pre-commit](https://pre-commit.com) for basic file hygiene.
Install once per clone:

```bash
pip install pre-commit
pre-commit install
```
