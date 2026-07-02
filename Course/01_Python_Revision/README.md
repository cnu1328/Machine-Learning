# Module 01 — Python Revision (ML-Focused)

**Duration:** 1 week (~5–8 hours)  
**Prerequisites:** Module 00 complete, environment verified

---

## Learning Objectives

By the end of this module you will:

- Manipulate NumPy arrays for ML data representation
- Load, explore, and preprocess datasets with Pandas
- Create publication-quality visualizations with Matplotlib
- Write vectorized code that runs 10–1000× faster than Python loops
- Reimplement data loading from your legacy Day scripts locally

---

## Notebooks

| # | Notebook | Time | Topics |
|---|----------|------|--------|
| 1 | [01_NumPy_Foundations.ipynb](01_NumPy_Foundations.ipynb) | 2 hrs | ndarrays, broadcasting, dot product, axis ops |
| 2 | [02_Pandas_for_ML.ipynb](02_Pandas_for_ML.ipynb) | 2 hrs | DataFrames, EDA, missing values, Day - 11 reimplementation |
| 3 | [03_Matplotlib_and_Visualization.ipynb](03_Matplotlib_and_Visualization.ipynb) | 1.5 hrs | scatter, histogram, boxplot, heatmap, confusion matrix |
| 4 | [04_Vectorization_and_Performance.ipynb](04_Vectorization_and_Performance.ipynb) | 1.5 hrs | loops vs NumPy, KNN distances, softmax, reproducibility |

---

## Legacy Code References

| Your Script | Course Notebook |
|-------------|-----------------|
| `Day - 11 House price Prediction Using Linear Regression.py` | Notebook 02 (Colab-free reimplementation) |
| `Day - 6 Titanic Survival prediction.py` | Notebook 02 (Titanic EDA) |
| `Day - 4 Salary_Estimatiom_by_KNerst.py` | Notebook 04 (KNN distance preview) |

---

## Assignment

**EDA Report on Heart Disease Dataset**

Using `heart_Disease.csv`, create a notebook or script that includes:

1. Dataset shape, dtypes, and missing value summary
2. Summary statistics (`describe()`)
3. Correlation heatmap
4. Distribution plots for at least 3 features
5. Target variable analysis (class balance)
6. Written observations (5 bullet points minimum)

Submit your work to your mentor for review before advancing.

---

## Exercises

See [exercises/README.md](exercises/README.md) — 8 exercises total.

---

## Quiz

[quiz/module_01_quiz.md](quiz/module_01_quiz.md) — 15 questions, need ≥12/15 (80%) to pass.

---

## Interview Questions

1. What is the difference between a Python list and a NumPy ndarray?
2. Explain broadcasting with an example.
3. What does `axis=0` mean for a 2D array?
4. When would you use `loc` vs `iloc`?
5. Why is vectorization critical for ML?
6. How do you handle missing values without data leakage?
7. What is the shape convention for ML datasets (samples vs features)?
8. Explain the difference between `@` and `*` for NumPy arrays.

---

## Common Mistakes

- Confusing `axis=0` and `axis=1`
- Using `a * b` when you mean `a @ b`
- Chained Pandas indexing causing SettingWithCopyWarning
- Exploring test data before train/test split
- Forgetting to set random seeds
- Using implicit pyplot instead of Figure/Axes API

---

## Real-World Applications

- **GIS:** Satellite rasters as `(bands, height, width)` NumPy arrays
- **Tabular ML:** Customer data as Pandas DataFrames → NumPy feature matrices
- **EDA:** Every Kaggle competition and industry project starts here
- **Production:** Vectorized preprocessing pipelines in scikit-learn and PyTorch

---

## Module Gate

Before Module 02, complete:

- [ ] All 4 notebooks
- [ ] All exercises (attempt before solutions)
- [ ] Assignment: heart disease EDA
- [ ] Quiz ≥80%
- [ ] Answer 3 mentor gate questions in chat

---

## Summary

Module 01 builds the numerical Python foundation. Every subsequent module — from gradient descent (Module 02) to UNet++ training (Module 12) — depends on fluent NumPy/Pandas/Matplotlib usage.

---

## Further Reading

- [NumPy Quickstart Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [Pandas Getting Started](https://pandas.pydata.org/docs/getting_started/index.html)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)

---

**Next Module:** [02_Mathematics/](../02_Mathematics/)
