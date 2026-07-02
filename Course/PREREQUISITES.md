# Prerequisites

## Required Before Starting

### Programming

- Python 3.10 or higher
- Variables, loops, conditionals, functions, classes
- File I/O (reading/writing files)
- Basic command line (`cd`, `pip install`, `python`)

You already meet this requirement.

### Software

| Tool | Purpose | Install |
|------|---------|---------|
| Python 3.10+ | Runtime | System or pyenv |
| Jupyter Lab | Notebooks | `pip install jupyterlab` |
| Git | Version control | System package |

### Hardware

| Phase | Requirement |
|-------|-------------|
| Modules 00–04 | CPU only, 8 GB RAM minimum |
| Modules 05–06 | CUDA GPU strongly recommended (8+ GB VRAM) |
| Modules 07–12 | CUDA GPU + 16 GB RAM; GDAL for geospatial |

Your existing ML_Setup environment (`../ML_Setup/requirements.txt`) satisfies all requirements for the full course.

---

## NOT Required (We Teach From Scratch)

- Linear algebra
- Calculus
- Probability and statistics
- Machine learning theory
- PyTorch or TensorFlow
- scikit-learn internals
- GIS libraries (introduced when needed)

---

## Environment Setup

### Option A: Minimal (Modules 00–03)

```bash
cd Machine-Learning
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-course.txt
jupyter lab
```

### Option B: Full Stack (Modules 05+)

Use your existing ML_Setup environment which includes PyTorch, GDAL, rasterio, geopandas, albumentations, etc.

```bash
# Activate your ML_Setup venv, then:
cd Machine-Learning
jupyter lab
```

### Verify Setup

Run the setup cell in `00_Course_Introduction/00_Welcome_and_Learning_Contract.ipynb`. You should see:

```
Python: 3.10+
NumPy: 1.26+
Pandas: 2.2+
Matplotlib: 3.8+
Jupyter: OK
```

---

## Datasets

Bundled CSV files in the repo root (used from Module 01 onward):

| File | Used In |
|------|---------|
| `houseprice.csv` | Module 01, 03 (Linear Regression) |
| `TitanicSurvival.csv` | Module 01, 03 (Classification) |
| `heart_Disease.csv` | Module 01 assignment, 03 (Logistic Regression) |
| `salary.csv` | Module 03 (Polynomial Regression) |
| `breat_cancer.csv` | Module 03 (Multi-algorithm comparison) |

Notebooks use relative paths like `../../houseprice.csv` from module folders.

---

## Optional Background (Helpful but Not Required)

- GIS experience (rasterio, geopandas) — you have this
- Deep learning project experience — you have this via water-bodies-detection
- Linear algebra from university — will be retaught in Module 02

---

## Before Module 01

- [ ] Python 3.10+ installed
- [ ] Virtual environment created and activated
- [ ] `pip install -r requirements-course.txt` succeeded
- [ ] `jupyter lab` opens in browser
- [ ] Setup verification notebook runs without errors

---

**Next:** [00_Course_Introduction/](00_Course_Introduction/)
