# Machine Learning

A personal Machine Learning learning repository containing classical ML tutorial scripts and a comprehensive university-level course.

## Repository Structure

```
Machine-Learning/
├── Course/                  # Complete ML course (start here)
├── Day - *.py               # Legacy tutorial scripts (2019–2022)
├── *.csv                    # Datasets bundled with tutorials
└── requirements-course.txt  # Dependencies for the course
```

## Getting Started

**New learners:** Start with [Course/README.md](Course/README.md).

**Legacy scripts:** The `Day - N` Python files are original Colab-era tutorials. They are preserved as historical reference. The course reimplements and extends them with full theory, mathematics, and exercises in Jupyter notebooks.

## Course Overview

The course covers:

- Mathematics for ML (linear algebra, calculus, probability, statistics, optimization)
- Classical ML (every algorithm from scratch + scikit-learn)
- Deep Learning (MLP, CNN, from-scratch implementations)
- Computer Vision (segmentation, detection, transformers)
- Production ML (MLOps, deployment, optimization)
- Capstone: line-by-line walkthrough of the [water-bodies-detection](../water-bodies-detection/) project

**Estimated duration:** 18–24 months at 5–8 hours/week.

## Setup

```bash
cd Machine-Learning
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-course.txt
jupyter lab
```

Open `Course/00_Course_Introduction/00_Welcome_and_Learning_Contract.ipynb` to begin.

## Related Repositories

| Repository | Role in Course |
|------------|----------------|
| [water-bodies-detection](../water-bodies-detection/) | Capstone project (Module 12) — dual-head UNet++ aquaculture segmentation |
| [ML_Setup](../ML_Setup/) | Full environment with PyTorch, GDAL, geospatial stack |
