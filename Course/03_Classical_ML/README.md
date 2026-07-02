# Module 03 — Classical Machine Learning

**Duration:** 8–10 weeks  
**Prerequisites:** Module 02 complete  
**Status:** Ready

---

## Overview

Every algorithm from your legacy Day scripts, reimplemented with full mathematical derivation, plus algorithms not yet in your repo (XGBoost, DBSCAN, t-SNE, UMAP, etc.).

---

## Study Plan (8–10 weeks at 5–8 hrs/week)

| Week | Section | Notebooks |
|------|---------|-----------|
| 1–2 | Regression | 01–06 |
| 3–5 | Classification | 07–14 |
| 6 | Ensemble | 15–19 |
| 7 | Clustering & Dim Reduction | 20–27 |
| 8 | Model Selection | 28–32 |
| 9 | Anomaly & Recommendation | 33–35 |
| 10 | Review | Exercises, assignment, quiz |

---

## Part 1: Regression (Notebooks 01–06)

| # | Notebook | Legacy Script |
|---|----------|---------------|
| 01 | [01_Linear_Regression.ipynb](01_Linear_Regression.ipynb) | Day - 11 |
| 02 | [02_Polynomial_Regression.ipynb](02_Polynomial_Regression.ipynb) | Day - 12 |
| 03 | [03_Support_Vector_Regression.ipynb](03_Support_Vector_Regression.ipynb) | Day - 13 |
| 04 | [04_Decision_Tree_Regression.ipynb](04_Decision_Tree_Regression.ipynb) | Day - 15 |
| 05 | [05_Random_Forest_Regression.ipynb](05_Random_Forest_Regression.ipynb) | Day - 16 |
| 06 | [06_Regression_Model_Evaluation.ipynb](06_Regression_Model_Evaluation.ipynb) | Day - 17 & 18 |

---

## Part 2: Classification (Notebooks 07–14)

| # | Notebook | Legacy Script |
|---|----------|---------------|
| 07 | [07_Logistic_Regression.ipynb](07_Logistic_Regression.ipynb) | Day - 3 |
| 08 | [08_Naive_Bayes.ipynb](08_Naive_Bayes.ipynb) | Day - 6 |
| 09 | [09_K_Nearest_Neighbors.ipynb](09_K_Nearest_Neighbors.ipynb) | Day - 4 |
| 10 | [10_Support_Vector_Machines.ipynb](10_Support_Vector_Machines.ipynb) | Day - 5 |
| 11 | [11_Decision_Tree_Classifier.ipynb](11_Decision_Tree_Classifier.ipynb) | Day - 7 |
| 12 | [12_Random_Forest_Classifier.ipynb](12_Random_Forest_Classifier.ipynb) | Digit recognition randomforest.py |
| 13 | [13_Classification_Model_Evaluation.ipynb](13_Classification_Model_Evaluation.ipynb) | Day - 9 |
| 14 | [14_Multi_Algorithm_Comparison.ipynb](14_Multi_Algorithm_Comparison.ipynb) | Day - 10 |

---

## Part 3: Ensemble Methods (Notebooks 15–19)

| # | Notebook |
|---|----------|
| 15 | [15_AdaBoost.ipynb](15_AdaBoost.ipynb) |
| 16 | [16_Gradient_Boosting.ipynb](16_Gradient_Boosting.ipynb) |
| 17 | [17_XGBoost.ipynb](17_XGBoost.ipynb) |
| 18 | [18_LightGBM.ipynb](18_LightGBM.ipynb) |
| 19 | [19_CatBoost.ipynb](19_CatBoost.ipynb) |

---

## Part 4: Clustering & Dimensionality Reduction (Notebooks 20–27)

| # | Notebook | Legacy Script |
|---|----------|---------------|
| 20 | [20_K_Means_Clustering.ipynb](20_K_Means_Clustering.ipynb) | New |
| 21 | [21_Hierarchical_Clustering.ipynb](21_Hierarchical_Clustering.ipynb) | Day - 20 |
| 22 | [22_DBSCAN.ipynb](22_DBSCAN.ipynb) | New |
| 23 | [23_Gaussian_Mixture_Models.ipynb](23_Gaussian_Mixture_Models.ipynb) | New |
| 24 | [24_PCA.ipynb](24_PCA.ipynb) | Day - 21 |
| 25 | [25_LDA.ipynb](25_LDA.ipynb) | New |
| 26 | [26_tSNE.ipynb](26_tSNE.ipynb) | New |
| 27 | [27_UMAP.ipynb](27_UMAP.ipynb) | New |

---

## Part 5: Feature Engineering & Model Selection (Notebooks 28–32)

| # | Notebook |
|---|----------|
| 28 | [28_Feature_Engineering.ipynb](28_Feature_Engineering.ipynb) |
| 29 | [29_Feature_Selection.ipynb](29_Feature_Selection.ipynb) |
| 30 | [30_Cross_Validation.ipynb](30_Cross_Validation.ipynb) |
| 31 | [31_Bias_Variance_Tradeoff.ipynb](31_Bias_Variance_Tradeoff.ipynb) |
| 32 | [32_Hyperparameter_Tuning.ipynb](32_Hyperparameter_Tuning.ipynb) |

---

## Part 6: Anomaly Detection & Recommendation (Notebooks 33–35)

| # | Notebook | Legacy Script |
|---|----------|---------------|
| 33 | [33_Isolation_Forest.ipynb](33_Isolation_Forest.ipynb) | New |
| 34 | [34_Autoencoders_Classical.ipynb](34_Autoencoders_Classical.ipynb) | New |
| 35 | [35_SVD_Recommendation.ipynb](35_SVD_Recommendation.ipynb) | Day - 22 |

---

## Per-Algorithm Template

Every notebook includes:

1. Why it exists
2. Mathematical formulation
3. scikit-learn implementation
4. Legacy Day script connection (where applicable)
5. Hyperparameters
6. Advantages and limitations
7. Exercises
8. Real-world applications

---

## Module Deliverables

- [ ] All 35 notebooks completed
- [ ] 14 exercises attempted
- [ ] Assignment: Titanic end-to-end ML pipeline
- [ ] Quiz ≥20/25 (80%)
- [ ] Gate questions answered in chat

---

## Assignment

**Titanic End-to-End Pipeline** — EDA, feature engineering, 5+ model comparison, hyperparameter tuning, >80% test accuracy.

See [exercises/README.md](exercises/README.md).

---

## Interview Questions

1. Derive the normal equation for linear regression.
2. Explain bias-variance tradeoff with an example.
3. When would you use Random Forest over Logistic Regression?
4. Compare K-Means vs DBSCAN.
5. Why scale features before KNN but not before Decision Trees?
6. Explain precision vs recall with a medical diagnosis example.
7. What is the kernel trick in SVM?
8. How does gradient boosting differ from random forest?

---

## Common Mistakes

- Not scaling features for distance-based models
- Using accuracy on imbalanced datasets
- Data leakage in preprocessing
- Overfitting with high-degree polynomials
- Using t-SNE output as ML features

---

## Resources

- [CHEATSHEET.md](CHEATSHEET.md)
- [quiz/module_03_quiz.md](quiz/module_03_quiz.md)
- [exercises/README.md](exercises/README.md)

---

**Previous:** [02_Mathematics/](../02_Mathematics/)  
**Next:** [04_ML_Paradigms/](../04_ML_Paradigms/)
