# Module 03 Exercises

Attempt before checking [solutions/](solutions/).

---

## Regression (Notebooks 01–06)

### Exercise 1 — Linear Regression
Implement gradient descent for multivariate linear regression on `houseprice.csv` with multiple features. Compare with normal equation.

### Exercise 2 — Polynomial Regression
Find optimal polynomial degree (1–10) for `salary.csv` using 5-fold cross-validation.

### Exercise 3 — Model Comparison
Compare Linear Regression, SVR, Decision Tree, and Random Forest on California housing. Report R², RMSE, MAE.

---

## Classification (Notebooks 07–14)

### Exercise 4 — Logistic Regression from Scratch
Train logistic regression on `heart_Disease.csv` without sklearn. Achieve >75% accuracy.

### Exercise 5 — KNN from Scratch
Implement KNN classifier on Iris. Plot accuracy vs k (1–20).

### Exercise 6 — Confusion Matrix
Train SVM on breast cancer data. Compute precision, recall, F1 manually from confusion matrix.

### Exercise 7 — Multi-Model Benchmark
Replicate Day - 10: compare 5+ algorithms on `breat_cancer.csv` with stratified 5-fold CV.

---

## Ensemble (Notebooks 15–19)

### Exercise 8 — Boosting Comparison
Compare AdaBoost, GradientBoosting, and XGBoost on breast cancer. Plot learning curves.

---

## Clustering (Notebooks 20–27)

### Exercise 9 — K-Means from Scratch
Implement K-Means (already in notebook). Find optimal k using elbow method and silhouette score on Iris.

### Exercise 10 — PCA Variance
Determine minimum number of PCA components to retain 95% variance on Iris.

---

## Model Selection (Notebooks 28–32)

### Exercise 11 — Feature Engineering
Engineer 5 new features for Titanic. Measure accuracy improvement over baseline.

### Exercise 12 — Grid Search
Tune Random Forest on breast cancer with GridSearchCV. Report best params and CV score.

---

## Anomaly & Recommendation (Notebooks 33–35)

### Exercise 13 — Anomaly Detection
Inject 5% outliers into Gaussian data. Detect with Isolation Forest. Report precision/recall.

### Exercise 14 — SVD Recommender
Extend Day - 22: build user-user collaborative filtering with SVD on a 10×10 rating matrix.

---

## Module Assignment: End-to-End ML Pipeline

**Deliverable:** `exercises/assignment_titanic_pipeline.ipynb`

Build a complete ML pipeline on `TitanicSurvival.csv`:

1. **EDA** — missing values, distributions, correlations
2. **Feature engineering** — at least 3 new features
3. **Preprocessing** — imputation, encoding, scaling
4. **Model comparison** — at least 5 algorithms with stratified CV
5. **Hyperparameter tuning** — GridSearchCV on best model
6. **Final evaluation** — confusion matrix, classification report on held-out test set
7. **Legacy comparison** — compare with your `Day - 6 Titanic Survival prediction.py` approach

Target: >80% test accuracy.

---

## Submission

> Module 03 exercises complete. Assignment attached. Quiz score: X/25.
