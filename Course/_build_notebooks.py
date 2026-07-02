#!/usr/bin/env python3
"""Generate Jupyter notebooks for the ML Course."""
import json
from pathlib import Path

COURSE = Path(__file__).resolve().parent


def nb(cells, kernel="python3"):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": kernel},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source if isinstance(source, list) else [source]}


def code(source, outputs=None):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source if isinstance(source, list) else [source],
        "outputs": outputs or [],
        "execution_count": None,
    }


def save(path, cells):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb(cells), indent=1))
    print(f"Created {path}")


def build_module_00():
    cells = [
        md("# Module 00: Welcome and Learning Contract\n\n**Course:** Machine Learning — From First Principles to Production\n\n**Your mentor role:** ML Professor, Research Mentor, Senior ML Engineer\n\n---\n\n## What This Course Is\n\nThis is not a collection of tutorials. It is a **structured transformation** from someone who builds ML with AI assistance to someone who:\n\n- Understands the mathematics behind every algorithm\n- Can read and implement research papers\n- Debugs training runs independently\n- Designs production ML systems\n- Explains every line of code they write\n\n## What This Course Is NOT\n\n- Shortcut code snippets without explanation\n- Skipped mathematical derivations\n- Passive video watching\n- Moving forward without mastering prerequisites"),

        md("## Your Existing Work\n\nYou already have valuable assets this course builds on:\n\n| Repository | Role |\n|------------|------|\n| **Machine-Learning** | ~35 Day scripts → reimplemented with full theory in Module 03 |\n| **water-bodies-detection** | Capstone walkthrough in Module 12 — dual-head UNet++ pipeline |\n\nWe will **not rewrite** your projects. We will **explain every line** until you own them completely."),

        md("## The Learning Contract\n\nBy starting this course, you agree to:\n\n1. **Complete modules in order** — no skipping ahead\n2. **Attempt exercises before solutions** — minimum 20 minutes of effort\n3. **Derive mathematics by hand** — not just read derivations\n4. **Pass module gates** — quiz ≥80% + mentor questions\n5. **Use AI as mentor, not crutch** — ask for hints, not full solutions\n6. **Track progress** in `Course/PROGRESS.md`"),

        md("## Environment Setup Verification\n\nRun the cell below. All imports must succeed before proceeding to Module 01."),

        code(
            "import sys\nimport numpy as np\nimport pandas as pd\nimport matplotlib\nimport matplotlib.pyplot as plt\n\nprint('=' * 50)\nprint('ENVIRONMENT VERIFICATION')\nprint('=' * 50)\nprint(f'Python:     {sys.version.split()[0]}')\nprint(f'NumPy:      {np.__version__}')\nprint(f'Pandas:     {pd.__version__}')\nprint(f'Matplotlib: {matplotlib.__version__}')\nprint('=' * 50)\n\n# Quick sanity checks\nassert np.array([1, 2, 3]).sum() == 6\nassert pd.DataFrame({'a': [1]}).shape == (1, 1)\nprint('All checks passed. Ready for Module 01.')"
        ),

        md("## Course Roadmap Preview\n\n```\nPhase 1 (Months 1-4):   00 Intro → 01 Python → 02 Math\nPhase 2 (Months 5-8):   03 Classical ML → 04 Paradigms\nPhase 3 (Months 9-11):  05 Deep Learning → 06 CNN\nPhase 4 (Months 12-16): 07 Segmentation → 08 Detection → 09 Instance Seg\nPhase 5 (Months 17-20): 10 Transformers → 11 Production ML\nPhase 6 (Months 21-24): 12 Capstone (water-bodies-detection)\n```\n\nFull details: [ROADMAP.md](../ROADMAP.md)"),

        md("## How Modules Connect to Your Code\n\n**Example — Linear Regression (Module 03):**\n\nYour legacy script `Day - 11 House price Prediction Using Linear Regression.py` loads a CSV, fits sklearn's `LinearRegression`, and plots a scatter. The course notebook will:\n\n1. Derive the normal equation $\\hat{\\beta} = (X^T X)^{-1} X^T y$ from scratch\n2. Implement gradient descent for the same problem\n3. Walk through your Day script line by line\n4. Compare sklearn vs manual implementation\n\n**Example — Capstone (Module 12):**\n\nEvery file in `water-bodies-detection/` — from `tile_and_mask.py` to `post_process_aqua_boundary.py` — explained as if you wrote none of it."),

        md("## First Conceptual Question (Gate)\n\nBefore moving to Module 01, answer this in your own words (in chat with your mentor):\n\n> **What is the difference between a Python list and a NumPy ndarray, and why does Machine Learning almost always use NumPy arrays instead of lists?**\n\nHints to think about (don't peek at Module 01 yet):\n- Memory layout\n- Data types (dtype)\n- Speed of numerical operations\n- Shape and dimensionality\n\n---\n\n## Module 00 Complete\n\nWhen setup passes and you've answered the gate question, update [PROGRESS.md](../PROGRESS.md) and begin [Module 01](../01_Python_Revision/)."),
    ]
    save(COURSE / "00_Course_Introduction" / "00_Welcome_and_Learning_Contract.ipynb", cells)


def build_numpy_notebook():
    cells = [
        md("# Notebook 01: NumPy Foundations for Machine Learning\n\n**Module:** 01 Python Revision  \n**Duration:** ~2 hours\n\n---\n\n## Learning Objectives\n\n1. Understand scalars, vectors, and matrices as NumPy arrays\n2. Master array creation, indexing, slicing, and broadcasting\n3. Perform dot products and matrix multiplication\n4. Use axis-based aggregations (sum, mean along rows/columns)\n5. Connect arrays to ML data representation"),

        md("## 1. Intuition: Why NumPy?\n\nMachine Learning is **numerical computation on large arrays**:\n\n- A dataset of 10,000 houses × 8 features = matrix of shape `(10000, 8)`\n- A grayscale image 512×512 = matrix of shape `(512, 512)`\n- A satellite tile with 6 bands = tensor of shape `(6, 512, 512)`\n\nPython lists are flexible but **slow** for this. NumPy stores data in contiguous memory blocks and runs operations in compiled C code.\n\n**Why ML uses NumPy instead of lists:**\n1. **Homogeneous dtype** — all elements same type (float64, int32)\n2. **Contiguous memory** — cache-friendly, vectorizable\n3. **Broadcasting** — operate on different shapes without loops\n4. **Linear algebra** — dot product, matrix inverse built-in\n5. **GPU bridge** — PyTorch/TensorFlow tensors extend ndarrays"),

        md("## 2. Scalars, Vectors, Matrices\n\n| Math | NumPy | Shape | Example |\n|------|-------|-------|--------|\n| Scalar $c$ | 0-d array | `()` | `np.array(3.14)` |\n| Vector $\\mathbf{x}$ | 1-d array | `(n,)` | `np.array([1,2,3])` |\n| Matrix $X$ | 2-d array | `(m, n)` | `np.array([[1,2],[3,4]])` |\n| Tensor | n-d array | `(d1,...,dk)` | image batch `(32, 3, 224, 224)` |"),

        code("import numpy as np\n\n# Scalar\nscalar = np.array(42)\nprint(f'Scalar: {scalar}, shape: {scalar.shape}, ndim: {scalar.ndim}')\n\n# Vector\nvector = np.array([1.0, 2.0, 3.0, 4.0])\nprint(f'Vector: {vector}, shape: {vector.shape}')\n\n# Matrix\nmatrix = np.array([[1, 2, 3],\n                   [4, 5, 6]])\nprint(f'Matrix shape: {matrix.shape}')  # (2, 3) = 2 rows, 3 columns\nprint(matrix)"),

        md("## 3. Array Creation and dtype\n\nEvery element in an ndarray has the **same data type** (`dtype`). ML typically uses `float32` (GPU) or `float64` (CPU training)."),

        code("# Common creation methods\nzeros = np.zeros((3, 4))       # 3x4 matrix of zeros\nones = np.ones((2, 3))         # 2x3 matrix of ones\nidentity = np.eye(3)           # 3x3 identity matrix\narange = np.arange(0, 10, 2)   # [0, 2, 4, 6, 8]\nlinspace = np.linspace(0, 1, 5)  # 5 evenly spaced values [0, 0.25, ..., 1]\n\nrng = np.random.default_rng(42)\nrandom_normal = rng.normal(0, 1, size=(1000,))  # mean=0, std=1\n\nprint(f'dtype of zeros: {zeros.dtype}')\nprint(f'Random sample mean: {random_normal.mean():.3f} (expect ~0)')\nprint(f'Random sample std:  {random_normal.std():.3f} (expect ~1)')"),

        md("## 4. Indexing and Slicing\n\nUnderstanding indexing is critical — bugs from wrong indexing cause silent errors in ML pipelines.\n\n- `arr[i]` — element or row\n- `arr[i, j]` — element at row i, column j\n- `arr[:, j]` — entire column j\n- `arr[i, :]` — entire row i\n- `arr[arr > 0]` — boolean (fancy) indexing"),

        code("data = np.array([[10, 20, 30],\n                 [40, 50, 60],\n                 [70, 80, 90]])\n\nprint('Element [1,2]:', data[1, 2])      # 60\nprint('Row 0:', data[0])                  # [10 20 30]\nprint('Column 1:', data[:, 1])            # [20 50 80]\nprint('Submatrix top-left 2x2:\\n', data[:2, :2])\n\n# Boolean indexing: all elements > 50\nprint('Elements > 50:', data[data > 50])"),

        md("## 5. Broadcasting\n\n**Broadcasting** applies operations between arrays of different shapes without explicit loops.\n\nRules (simplified):\n1. Compare shapes from right to left\n2. Dimensions are compatible if equal or one is 1\n3. Size-1 dimension is stretched (broadcast) to match\n\nExample: subtract mean from each column of a matrix."),

        code("X = np.array([[1, 2, 3],\n              [4, 5, 6],\n              [7, 8, 9]], dtype=float)\n\ncolumn_means = X.mean(axis=0)  # shape (3,)\nprint('Column means:', column_means)\n\n# Broadcasting: (3,3) - (3,) → (3,3)\nX_centered = X - column_means\nprint('Centered matrix:\\n', X_centered)\nprint('Column means after centering:', X_centered.mean(axis=0))  # ~0"),

        md("## 6. Dot Product and Matrix Multiplication\n\nThese are the **most important operations in ML**.\n\n**Dot product** (vectors): $\\mathbf{a} \\cdot \\mathbf{b} = \\sum_{i} a_i b_i$\n\n**Matrix multiplication**: $(AB)_{ij} = \\sum_k A_{ik} B_{kj}$\n\nIn NumPy:\n- `np.dot(a, b)` or `a @ b`\n- For 2D arrays, `@` is matrix multiply\n- For 1D arrays, `@` is dot product"),

        code("a = np.array([1, 2, 3])\nb = np.array([4, 5, 6])\n\n# Dot product\nprint('a · b =', np.dot(a, b))   # 1*4 + 2*5 + 3*6 = 32\nprint('a @ b =', a @ b)\n\nA = np.array([[1, 2], [3, 4]])\nB = np.array([[5, 6], [7, 8]])\n\nprint('A @ B =\\n', A @ B)\n# [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]\n\n# Connection to Linear Regression (Module 03):\n# y_pred = X @ weights + bias"),

        md("## 7. Axis Operations\n\nThe `axis` parameter is a common source of confusion.\n\nFor matrix shape `(m, n)`:\n- `axis=0` → collapse rows → result shape `(n,)` (column-wise operation)\n- `axis=1` → collapse columns → result shape `(m,)` (row-wise operation)\n\n**Memory trick:** `axis=0` goes **down** the rows (operates on each column)."),

        code("X = np.array([[1, 2, 3],\n              [4, 5, 6]])\n\nprint('Shape:', X.shape)\nprint('Sum all:       ', X.sum())\nprint('Sum axis=0 (col):', X.sum(axis=0))  # [5, 7, 9]\nprint('Sum axis=1 (row):', X.sum(axis=1))  # [6, 15]\nprint('Mean axis=0:   ', X.mean(axis=0))\nprint('Std axis=0:    ', X.std(axis=0))"),

        md("## 8. Reshape, Transpose, and Memory\n\nML constantly reshapes data:\n- Flatten image: `(512, 512)` → `(262144,)`\n- Batch images: `(32, 3, 224, 224)`\n- Feature matrix: `(n_samples, n_features)`"),

        code("img = np.arange(12).reshape(3, 4)  # simulate 3x4 image\nprint('Original:\\n', img)\nprint('Flatten:', img.flatten())\nprint('Transpose:\\n', img.T)\n\n# ML convention: samples as ROWS, features as COLUMNS\nn_samples, n_features = 100, 8\nX = rng.normal(0, 1, (n_samples, n_features))\nprint(f'Dataset shape: {X.shape} = ({n_samples} samples, {n_features} features)')"),

        md("## 9. Connection to GeoSpatial ML\n\nIn your **water-bodies-detection** project:\n\n```python\n# A Planet tile: 6 spectral bands, 512x512 pixels\ntile = np.zeros((6, 512, 512), dtype=np.float32)\n\n# After normalization (percentile scaling per band):\nfor b in range(6):\n    band = tile[b]\n    p2, p98 = np.percentile(band[band > 0], [2, 98])\n    tile[b] = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)\n```\n\nEvery raster band is a 2D NumPy array. The full tile is a 3D array. A training batch is a 4D array `(batch, channels, height, width)`."),

        md("## 10. Common Mistakes\n\n| Mistake | Problem | Fix |\n|---------|---------|-----|\n| `a * b` vs `a @ b` | Element-wise vs matrix multiply | Use `@` for linear algebra |\n| Wrong axis in mean | Normalizing wrong dimension | Print `.shape` before and after |\n| Modifying a view | Unexpected side effects | Use `.copy()` when unsure |\n| Integer division | `3/4` in int array truncates | Cast to float: `arr.astype(float)` |\n| `(n,) vs (n,1)` shape | sklearn expects 2D targets | Use `.reshape(-1, 1)` |"),

        md("## Exercise 1: Cosine Similarity\n\nImplement cosine similarity between two vectors **without sklearn**:\n\n$$\\text{cosine\\_sim}(\\mathbf{a}, \\mathbf{b}) = \\frac{\\mathbf{a} \\cdot \\mathbf{b}}{\\|\\mathbf{a}\\| \\|\\mathbf{b}\\|}$$\n\nwhere $\\|\\mathbf{a}\\| = \\sqrt{\\sum_i a_i^2}$"),

        code("# YOUR CODE HERE\ndef cosine_similarity(a, b):\n    \"\"\"Return cosine similarity between vectors a and b.\"\"\"\n    pass\n\n# Test\nu = np.array([1, 2, 3])\nv = np.array([4, 5, 6])\n# Expected: ~0.9746\nprint('Cosine similarity:', cosine_similarity(u, v))"),

        md("## Exercise 2: Manual Standardization\n\nStandardize each column of matrix X to zero mean and unit variance:\n\n$$X_{\\text{std}} = \\frac{X - \\mu}{\\sigma}$$\n\nDo NOT use sklearn. Use axis operations only."),

        code("# YOUR CODE HERE\nX = rng.normal(5, 2, (100, 4))  # 100 samples, 4 features\n\n# Standardize X → X_std\n# X_std should have mean ≈ 0 and std ≈ 1 per column\n"),

        md("## Interview Questions\n\n1. What is the difference between a Python list and a NumPy array?\n2. Explain broadcasting with an example.\n3. What does `axis=0` mean for a 2D array operation?\n4. Why do we use `float32` instead of `float64` on GPUs?\n5. What is the shape of `X @ W` if X is `(100, 10)` and W is `(10, 5)`?\n\n---\n\n## Summary\n\n- NumPy ndarrays are homogeneous, contiguous, fast numerical arrays\n- Shape `(m, n)` = m rows, n columns; samples are typically rows\n- `@` is matrix multiply; `*` is element-wise\n- Broadcasting eliminates loops; axis controls aggregation direction\n- Every ML pipeline starts with NumPy arrays\n\n**Next:** [02_Pandas_for_ML.ipynb](02_Pandas_for_ML.ipynb)"),
    ]
    save(COURSE / "01_Python_Revision" / "01_NumPy_Foundations.ipynb", cells)


def build_pandas_notebook():
    cells = [
        md("# Notebook 02: Pandas for Machine Learning\n\n**Module:** 01 Python Revision  \n**Duration:** ~2 hours\n\n---\n\n## Learning Objectives\n\n1. Load and inspect CSV datasets with Pandas\n2. Select, filter, and transform DataFrames\n3. Handle missing values and encode categoricals\n4. Perform groupby aggregations and merges\n5. Reproduce data loading from your legacy Day scripts (Colab-free)"),

        md("## 1. Intuition: Why Pandas?\n\nNumPy handles **homogeneous numerical arrays**. Real datasets have:\n- Column names (`price`, `area`, `survived`)\n- Mixed types (float, int, string, datetime)\n- Missing values (NaN)\n- Row indices\n\nPandas `DataFrame` = labeled table = what every ML dataset looks like before it becomes a NumPy matrix."),

        code("import pandas as pd\nimport numpy as np\nfrom pathlib import Path\n\n# Path to repo root CSV files (two levels up from this notebook)\nREPO_ROOT = Path('../../').resolve()\nprint('Repo root:', REPO_ROOT)"),

        md("## 2. Loading Your Datasets\n\nYour Machine-Learning repo includes CSV files used in the original Day tutorials."),

        code("# Load house price dataset (used in Day - 11)\nhouse = pd.read_csv(REPO_ROOT / 'houseprice.csv')\nprint('=== House Price Dataset ===')\nprint(f'Shape: {house.shape}')\nprint(house.head())\nprint('\\nColumns:', house.columns.tolist())\nprint('\\nInfo:')\nhouse.info()"),

        code("# Load Titanic dataset (used in Day - 6)\ntitanic = pd.read_csv(REPO_ROOT / 'TitanicSurvival.csv')\nprint('=== Titanic Dataset ===')\nprint(titanic.head())\nprint(f'\\nShape: {titanic.shape}')\nprint(f'\\nSurvival rate: {titanic[\"Survived\"].mean():.2%}')"),

        md("## 3. Selection: loc vs iloc\n\n- **`loc`** — label-based: `df.loc[row_label, col_name]`\n- **`iloc`** — integer-based: `df.iloc[0, 1]`\n\nFor ML, you typically extract feature matrix X and target y:"),

        code("# Feature-target split (Day - 11 pattern, modernized)\n# Original: x = dataset.drop('price', axis='columns'); y = dataset.price\n\nX = house.drop('price', axis='columns')\ny = house['price']\n\nprint('Features X:')\nprint(X.head())\nprint(f'\\nTarget y shape: {y.shape}')\nprint(y.head())"),

        md("## 4. Exploratory Data Analysis (EDA)\n\nBefore any ML model, always explore:\n1. Shape, dtypes, missing values\n2. Distributions of features and target\n3. Correlations\n4. Outliers"),

        code("print('=== EDA: House Prices ===')\nprint(house.describe())\nprint('\\nMissing values:')\nprint(house.isnull().sum())\nprint('\\nCorrelation with price:')\nprint(house.corr(numeric_only=True)['price'].sort_values(ascending=False))"),

        md("## 5. Handling Missing Values\n\nStrategies:\n- **Drop** rows/columns with too many NaNs\n- **Impute** with mean, median, mode\n- **Model-based** imputation (Module 03)\n\nRule: fit imputation on **training set only**, apply to test set (prevent data leakage)."),

        code("# Example with heart disease dataset\nheart = pd.read_csv(REPO_ROOT / 'heart_Disease.csv')\nprint('Heart disease shape:', heart.shape)\nprint('\\nMissing values per column:')\nmissing = heart.isnull().sum()\nprint(missing[missing > 0] if missing.sum() > 0 else 'No missing values')\n\n# Imputation example (demonstration only — proper split in Module 03)\nheart_filled = heart.fillna(heart.median(numeric_only=True))\nprint('\\nAfter median imputation, missing:', heart_filled.isnull().sum().sum())"),

        md("## 6. Encoding Categorical Variables\n\nML models need numbers, not strings.\n\n| Method | When to use |\n|--------|-------------|\n| Label encoding | Ordinal categories (low < medium < high) |\n| One-hot encoding | Nominal categories (red, blue, green) |\n\nModule 03 covers this in depth with sklearn `ColumnTransformer`."),

        code("# Titanic: encode Sex column\nprint('Original Sex values:', titanic['Sex'].unique())\n\n# Manual label encoding\nsex_map = {'male': 0, 'female': 1}\ntitanic_encoded = titanic.copy()\ntitanic_encoded['Sex_encoded'] = titanic_encoded['Sex'].map(sex_map)\n\n# One-hot encoding with pandas\nembarked_dummies = pd.get_dummies(titanic['Embarked'], prefix='embarked')\nprint(embarked_dummies.head())"),

        md("## 7. Groupby Aggregations\n\nEssential for understanding data before modeling."),

        code("# Survival rate by passenger class\nsurvival_by_class = titanic.groupby('Pclass')['Survived'].agg(['mean', 'count'])\nsurvival_by_class.columns = ['survival_rate', 'count']\nprint('Survival by class:')\nprint(survival_by_class)\n\n# Survival by sex and class\nprint('\\nSurvival by sex and class:')\nprint(titanic.groupby(['Sex', 'Pclass'])['Survived'].mean().unstack())"),

        md("## 8. Train/Test Split Pattern\n\nModule 03 uses sklearn's `train_test_split`. Here is the manual concept:"),

        code("def manual_train_test_split(X, y, test_ratio=0.2, seed=42):\n    \"\"\"Split data into train and test sets.\"\"\"\n    rng = np.random.default_rng(seed)\n    n = len(X)\n    indices = rng.permutation(n)\n    test_size = int(n * test_ratio)\n    test_idx = indices[:test_size]\n    train_idx = indices[test_size:]\n    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]\n\nX_train, X_test, y_train, y_test = manual_train_test_split(X, y)\nprint(f'Train: {len(X_train)}, Test: {len(X_test)}')"),

        md("## 9. Reimplementing Day - 11 (Colab-Free)\n\nYour original script used Google Colab file upload. Here is the same workflow locally:"),

        code("# Day - 11 reimplemented locally\nfrom sklearn.linear_model import LinearRegression\n\n# Load (already done above)\ndataset = pd.read_csv(REPO_ROOT / 'houseprice.csv')\n\n# Visualize\nimport matplotlib.pyplot as plt\nplt.figure(figsize=(8, 5))\nplt.scatter(dataset['area'], dataset['price'], color='red', marker='*')\nplt.xlabel('Area (sq ft)')\nplt.ylabel('Price')\nplt.title('House Price vs Area (Day - 11 dataset)')\nplt.show()\n\n# Train\nx = dataset.drop('price', axis='columns')\ny = dataset['price']\nmodel = LinearRegression()\nmodel.fit(x, y)\n\n# Predict for 4685 sq ft (same as Day - 11)\npred = model.predict([[4685]])\nprint(f'Predicted price for 4685 sq ft: {pred[0]:,.2f}')\nprint(f'Coefficient (slope): {model.coef_[0]:.2f}')\nprint(f'Intercept: {model.intercept_:,.2f}')"),

        md("## 10. Common Mistakes\n\n| Mistake | Consequence | Fix |\n|---------|-------------|-----|\n| Chained indexing `df[col][row]` | Unpredictable with `.loc` | Use `df.loc[row, col]` |\n| SettingWithCopyWarning | Modifying a view | Use `.copy()` explicitly |\n| Leaking test data into EDA | Overoptimistic metrics | Split first, then explore train only |\n| Not checking dtypes | Model errors on strings | Run `.info()` and `.dtypes` |\n| Ignoring missing values | NaN propagates silently | Always check `.isnull().sum()` |"),

        md("## Exercise 3: Titanic EDA\n\nUsing `TitanicSurvival.csv`, compute and print:\n1. Overall survival rate\n2. Survival rate by passenger class\n3. Average age of survivors vs non-survivors\n4. Count of missing values per column"),

        code("# YOUR CODE HERE\n"),

        md("## Exercise 4: Feature Matrix\n\nFrom the Titanic dataset, create:\n- X: columns `Pclass`, `Sex_encoded`, `Age`, `Fare` (drop rows with NaN Age)\n- y: `Survived`\n\nPrint shapes of X and y."),

        code("# YOUR CODE HERE\n"),

        md("## Mini Project: EDA Report\n\nCreate a complete EDA for `heart_Disease.csv` (this is the Module 01 assignment):\n- Summary statistics\n- Missing value analysis\n- Correlation heatmap\n- Distribution plots for key features\n- Written observations (3–5 bullet points)\n\nSee `exercises/README.md` for full assignment spec.\n\n---\n\n## Summary\n\n- Pandas DataFrame = labeled table for ML datasets\n- Always run EDA before modeling\n- Split train/test before any preprocessing that learns from data\n- Your Day scripts work locally — just replace Colab upload with `pd.read_csv()`\n\n**Next:** [03_Matplotlib_and_Visualization.ipynb](03_Matplotlib_and_Visualization.ipynb)"),
    ]
    save(COURSE / "01_Python_Revision" / "02_Pandas_for_ML.ipynb", cells)


def build_matplotlib_notebook():
    cells = [
        md("# Notebook 03: Matplotlib and Visualization for ML\n\n**Module:** 01 Python Revision  \n**Duration:** ~1.5 hours\n\n---\n\n## Learning Objectives\n\n1. Use the Figure/Axes API correctly\n2. Create scatter, histogram, boxplot, and heatmap plots\n3. Recreate visualizations from your Day - 11 script\n4. Preview confusion matrix heatmaps for Module 03"),

        code("import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom pathlib import Path\n\nREPO_ROOT = Path('../../').resolve()\n\n# Style for readable plots\nplt.rcParams['figure.figsize'] = (8, 5)\nplt.rcParams['font.size'] = 12"),

        md("## 1. Figure and Axes API\n\nAlways use explicit Figure/Axes — not implicit pyplot state machine — for production-quality plots."),

        code("fig, ax = plt.subplots(1, 1, figsize=(8, 5))\nx = np.linspace(0, 10, 100)\nax.plot(x, np.sin(x), label='sin(x)')\nax.plot(x, np.cos(x), label='cos(x)')\nax.set_xlabel('x')\nax.set_ylabel('y')\nax.set_title('Sine and Cosine')\nax.legend()\nax.grid(True, alpha=0.3)\nplt.tight_layout()\nplt.show()"),

        md("## 2. Subplots\n\nCompare multiple views of data side by side."),

        code("fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n\nrng = np.random.default_rng(42)\nfor i, ax in enumerate(axes):\n    data = rng.normal(i, 1, 500)\n    ax.hist(data, bins=30, edgecolor='black', alpha=0.7)\n    ax.set_title(f'Distribution {i+1} (μ={i})')\n    ax.set_xlabel('Value')\n\nplt.tight_layout()\nplt.show()"),

        md("## 3. Scatter Plot — Day - 11 Recreation\n\nYour original Day - 11 script:\n```python\nplt.xlabel('Area')\nplt.ylabel('Price')\nplt.scatter(dataset.area, dataset.price, color='red', marker='*')\n```"),

        code("house = pd.read_csv(REPO_ROOT / 'houseprice.csv')\n\nfig, ax = plt.subplots(figsize=(8, 5))\nax.scatter(house['area'], house['price'], color='red', marker='*', alpha=0.7)\nax.set_xlabel('Area (sq ft)')\nax.set_ylabel('Price ($)')\nax.set_title('House Price vs Area')\nplt.tight_layout()\nplt.show()"),

        md("## 4. Scatter with Regression Line\n\nVisualize the linear relationship (preview of Module 03 Linear Regression)."),

        code("from sklearn.linear_model import LinearRegression\n\nX = house[['area']].values\ny = house['price'].values\nmodel = LinearRegression().fit(X, y)\n\nx_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)\ny_line = model.predict(x_line)\n\nfig, ax = plt.subplots(figsize=(8, 5))\nax.scatter(house['area'], house['price'], color='red', marker='*', alpha=0.7, label='Data')\nax.plot(x_line, y_line, color='blue', linewidth=2, label=f'Linear fit (slope={model.coef_[0]:.2f})')\nax.set_xlabel('Area (sq ft)')\nax.set_ylabel('Price ($)')\nax.set_title('House Price vs Area with Linear Regression')\nax.legend()\nplt.tight_layout()\nplt.show()"),

        md("## 5. Histograms and Distributions\n\nUnderstand feature distributions before modeling."),

        code("titanic = pd.read_csv(REPO_ROOT / 'TitanicSurvival.csv')\n\nfig, axes = plt.subplots(1, 2, figsize=(12, 4))\n\n# Age distribution\naxes[0].hist(titanic['Age'].dropna(), bins=30, edgecolor='black', alpha=0.7)\naxes[0].set_xlabel('Age')\naxes[0].set_ylabel('Count')\naxes[0].set_title('Age Distribution')\n\n# Fare distribution (log scale often better for skewed data)\naxes[1].hist(titanic['Fare'].dropna(), bins=30, edgecolor='black', alpha=0.7)\naxes[1].set_xlabel('Fare')\naxes[1].set_ylabel('Count')\naxes[1].set_title('Fare Distribution')\n\nplt.tight_layout()\nplt.show()"),

        md("## 6. Boxplots — Detecting Outliers\n\nBoxplots show median, quartiles, and outliers. Critical before training."),

        code("fig, ax = plt.subplots(figsize=(8, 5))\ntitanic.boxplot(column='Fare', by='Pclass', ax=ax)\nax.set_xlabel('Passenger Class')\nax.set_ylabel('Fare')\nax.set_title('Fare by Passenger Class')\nplt.suptitle('')  # remove default boxplot title\nplt.tight_layout()\nplt.show()"),

        md("## 7. Heatmap — Correlation Matrix\n\nPreview of confusion matrix visualization in Module 03."),

        code("heart = pd.read_csv(REPO_ROOT / 'heart_Disease.csv')\ncorr = heart.corr(numeric_only=True)\n\nfig, ax = plt.subplots(figsize=(10, 8))\nim = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)\nax.set_xticks(range(len(corr.columns)))\nax.set_yticks(range(len(corr.columns)))\nax.set_xticklabels(corr.columns, rotation=45, ha='right')\nax.set_yticklabels(corr.columns)\nax.set_title('Feature Correlation Matrix (Heart Disease)')\nplt.colorbar(im, ax=ax, label='Correlation')\nplt.tight_layout()\nplt.show()"),

        md("## 8. Confusion Matrix Preview (Module 03)\n\nClassification models are evaluated with confusion matrices. Here is how to plot one:"),

        code("from sklearn.metrics import confusion_matrix\n\n# Simulated predictions for demonstration\ny_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])\ny_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 0])\n\ncm = confusion_matrix(y_true, y_pred)\n\nfig, ax = plt.subplots(figsize=(6, 5))\nim = ax.imshow(cm, cmap='Blues')\nax.set_xticks([0, 1])\nax.set_yticks([0, 1])\nax.set_xticklabels(['Pred 0', 'Pred 1'])\nax.set_yticklabels(['True 0', 'True 1'])\nax.set_xlabel('Predicted')\nax.set_ylabel('Actual')\nax.set_title('Confusion Matrix')\n\nfor i in range(2):\n    for j in range(2):\n        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=20, color='white' if cm[i,j] > cm.max()/2 else 'black')\n\nplt.colorbar(im)\nplt.tight_layout()\nplt.show()"),

        md("## 9. Saving Figures\n\nAlways save figures for reports and papers."),

        code("# fig, ax = plt.subplots()\n# ... create plot ...\n# fig.savefig('output/plot.png', dpi=150, bbox_inches='tight')\n# For vector graphics: fig.savefig('output/plot.pdf')"),

        md("## Exercise 5: Multi-panel EDA Plot\n\nCreate a 2×2 subplot figure for the house price dataset:\n1. Scatter: area vs price\n2. Histogram of price\n3. Histogram of area\n4. Boxplot of price\n\nAdd titles and labels to every subplot."),

        code("# YOUR CODE HERE\n"),

        md("## Interview Questions\n\n1. When would you use a histogram vs a boxplot?\n2. Why is the Figure/Axes API preferred over implicit pyplot?\n3. How do you visualize a confusion matrix?\n4. What does a correlation heatmap tell you before modeling?\n\n---\n\n## Summary\n\n- Use `fig, ax = plt.subplots()` for all plots\n- Scatter for relationships, histogram for distributions, boxplot for outliers\n- Heatmaps for correlation and confusion matrices\n- Always label axes, add titles, use `tight_layout()`\n\n**Next:** [04_Vectorization_and_Performance.ipynb](04_Vectorization_and_Performance.ipynb)"),
    ]
    save(COURSE / "01_Python_Revision" / "03_Matplotlib_and_Visualization.ipynb", cells)


def build_vectorization_notebook():
    cells = [
        md("# Notebook 04: Vectorization and Performance\n\n**Module:** 01 Python Revision  \n**Duration:** ~1.5 hours\n\n---\n\n## Learning Objectives\n\n1. Understand why vectorization matters for ML\n2. Benchmark Python loops vs NumPy operations\n3. Implement vectorized distance computations (KNN preview)\n4. Use random seeds for reproducibility\n5. Connect performance to production inference scale"),

        code("import numpy as np\nimport time\nimport matplotlib.pyplot as plt\n\nrng = np.random.default_rng(42)"),

        md("## 1. Intuition: Why Vectorization?\n\nML trains on millions of numbers. Python loops interpret each operation. NumPy pushes work to compiled C/Fortran.\n\nAt inference scale (e.g., sliding window over a 10000×10000 GeoTIFF in water-bodies-detection), vectorization is the difference between seconds and hours."),

        md("## 2. Loop vs Vectorized: Element-wise Sum"),

        code("n = 1_000_000\na = rng.random(n)\nb = rng.random(n)\n\n# Python loop\nstart = time.perf_counter()\nresult_loop = sum(a[i] + b[i] for i in range(n))\nt_loop = time.perf_counter() - start\n\n# NumPy vectorized\nstart = time.perf_counter()\nresult_vec = (a + b).sum()\nt_vec = time.perf_counter() - start\n\nprint(f'Loop result:       {result_loop:.4f}')\nprint(f'Vectorized result: {result_vec:.4f}')\nprint(f'Loop time:         {t_loop:.4f}s')\nprint(f'Vectorized time:   {t_vec:.6f}s')\nprint(f'Speedup:           {t_loop/t_vec:.0f}x')"),

        md("## 3. Loop vs Vectorized: Euclidean Distance\n\nK-Nearest Neighbors (Module 03, your Day - 4) requires computing distances between points. Vectorization is essential."),

        code("def euclidean_loop(a, b):\n    \"\"\"Distance between two vectors using Python loop.\"\"\"\n    return sum((a[i] - b[i]) ** 2 for i in range(len(a))) ** 0.5\n\ndef euclidean_vec(a, b):\n    \"\"\"Distance using NumPy.\"\"\"\n    return np.sqrt(np.sum((a - b) ** 2))\n\np = rng.random(1000)\nq = rng.random(1000)\n\nassert abs(euclidean_loop(p, q) - euclidean_vec(p, q)) < 1e-10\nprint(f'Distance (loop): {euclidean_loop(p, q):.6f}')\nprint(f'Distance (vec):  {euclidean_vec(p, q):.6f}')"),

        md("## 4. Vectorized Distance Matrix (KNN Preview)\n\nGiven query point q and training matrix X (n_samples × n_features), compute distances to ALL training points at once.\n\n$$d_i = \\|\\mathbf{x}_i - \\mathbf{q}\\|_2 = \\sqrt{\\sum_j (x_{ij} - q_j)^2}$$\n\nVectorized using broadcasting:"),

        code("def distance_matrix(X, q):\n    \"\"\"\n    Compute Euclidean distance from query q to all rows of X.\n    X: (n_samples, n_features)\n    q: (n_features,)\n    Returns: (n_samples,) distances\n    \"\"\"\n    diff = X - q  # broadcasting: (n, d) - (d,) → (n, d)\n    return np.sqrt((diff ** 2).sum(axis=1))\n\n# Test\nX_train = rng.normal(0, 1, (1000, 5))\nq = rng.normal(0, 1, 5)\n\ndists = distance_matrix(X_train, q)\nprint(f'Distances shape: {dists.shape}')\nprint(f'Min distance: {dists.min():.4f}')\nprint(f'Max distance: {dists.max():.4f}')\n\n# Find 5 nearest neighbors\nk = 5\nnearest_idx = np.argsort(dists)[:k]\nprint(f'5 nearest indices: {nearest_idx}')\nprint(f'Their distances: {dists[nearest_idx]}')"),

        md("## 5. Pairwise Distance Matrix\n\nCompute distances between ALL pairs — used in clustering (Module 03)."),

        code("def pairwise_distances(X):\n    \"\"\"\n    Compute pairwise Euclidean distances.\n    X: (n, d) → returns (n, n) distance matrix\n    \n    Uses: ||a - b||² = ||a||² + ||b||² - 2(a·b)\n    \"\"\"\n    sq_norms = (X ** 2).sum(axis=1)\n    dists_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * (X @ X.T)\n    dists_sq = np.maximum(dists_sq, 0)  # numerical stability\n    return np.sqrt(dists_sq)\n\nX_small = rng.normal(0, 1, (50, 3))\nD = pairwise_distances(X_small)\nprint(f'Pairwise distance matrix shape: {D.shape}')\nprint(f'Diagonal (should be 0): {D.diagonal()[:5]}')"),

        md("## 6. Reproducibility with Random Seeds\n\nML experiments must be reproducible. Always set seeds:"),

        code("# NumPy random generator with seed\nrng1 = np.random.default_rng(42)\nprint('Run 1:', rng1.random(3))\n\nrng2 = np.random.default_rng(42)\nprint('Run 2:', rng2.random(3))  # identical\n\n# Different seed → different results\nrng3 = np.random.default_rng(0)\nprint('Run 3:', rng3.random(3))"),

        md("## 7. In-place Operations and Memory\n\nFor large arrays, avoid unnecessary copies:"),

        code("a = np.ones(1_000_000)\nb = a  # b is a VIEW, not a copy\nb += 1  # modifies a too!\nprint(f'a[0] after b+=1: {a[0]}')  # 2.0\n\na = np.ones(1_000_000)\nb = a.copy()  # explicit copy\nb += 1\nprint(f'a[0] after copy b+=1: {a[0]}')  # 1.0"),

        md("## 8. Benchmark Visualization"),

        code("sizes = [100, 1000, 10000, 100000, 500000]\nloop_times = []\nvec_times = []\n\nfor n in sizes:\n    a = rng.random(n)\n    b = rng.random(n)\n    \n    start = time.perf_counter()\n    _ = sum(a[i] + b[i] for i in range(n))\n    loop_times.append(time.perf_counter() - start)\n    \n    start = time.perf_counter()\n    _ = (a + b).sum()\n    vec_times.append(time.perf_counter() - start)\n\nfig, ax = plt.subplots(figsize=(8, 5))\nax.plot(sizes, loop_times, 'o-', label='Python loop', color='red')\nax.plot(sizes, vec_times, 's-', label='NumPy vectorized', color='blue')\nax.set_xlabel('Array size')\nax.set_ylabel('Time (seconds)')\nax.set_title('Loop vs Vectorized Performance')\nax.legend()\nax.set_xscale('log')\nax.set_yscale('log')\nax.grid(True, alpha=0.3)\nplt.tight_layout()\nplt.show()"),

        md("## 9. Production Consideration\n\nIn **water-bodies-detection** `predict.py`, sliding-window inference processes thousands of 512×512 tiles. Each tile operation uses vectorized NumPy/PyTorch ops. A Python loop over pixels would make inference unusable.\n\nRule: **If you're writing a for-loop over data points in ML, you're probably doing it wrong.**"),

        md("## Exercise 6: Vectorized Softmax\n\nImplement softmax vectorized (preview of Module 05):\n\n$$\\text{softmax}(z_i) = \\frac{e^{z_i}}{\\sum_j e^{z_j}}$$\n\nInput: 1D array of logits. Output: probabilities summing to 1."),

        code("# YOUR CODE HERE\ndef softmax(z):\n    pass\n\nz = np.array([2.0, 1.0, 0.1])\nprobs = softmax(z)\nprint('Softmax:', probs)\nprint('Sum:', probs.sum())  # should be 1.0"),

        md("## Exercise 7: Vectorized Accuracy\n\nGiven arrays `y_true` and `y_pred`, compute accuracy without loops:\n\n$$\\text{accuracy} = \\frac{1}{n} \\sum_{i=1}^{n} \\mathbb{1}[y_{\\text{true},i} = y_{\\text{pred},i}]$$"),

        code("# YOUR CODE HERE\ny_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])\ny_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1, 0, 0])\n\n# accuracy = ?\n"),

        md("## Interview Questions\n\n1. Why is vectorization faster than Python loops?\n2. How would you compute distances from one point to 1 million training points efficiently?\n3. Why set random seeds in ML experiments?\n4. What is the difference between a view and a copy in NumPy?\n\n---\n\n## Module 01 Complete\n\nYou now have the Python/numerical foundation for all subsequent modules.\n\n**Before Module 02:** Complete exercises, assignment, quiz, and gate questions.\n\nSee: [CHEATSHEET.md](CHEATSHEET.md) | [quiz/module_01_quiz.md](quiz/module_01_quiz.md) | [exercises/README.md](exercises/README.md)"),
    ]
    save(COURSE / "01_Python_Revision" / "04_Vectorization_and_Performance.ipynb", cells)


if __name__ == "__main__":
    build_module_00()
    build_numpy_notebook()
    build_pandas_notebook()
    build_matplotlib_notebook()
    build_vectorization_notebook()
    print("Done.")
