# Module 03 Quiz

**Passing score:** 20/25 (80%)

---

## Section A: Regression (5 questions)

**Q1.** Normal equation for linear regression:
- (a) $w = Xy$
- (b) $w = (X^TX)^{-1}X^Ty$
- (c) $w = X^Ty$
- (d) $w = (XX^T)^{-1}y$

**Q2.** R² = 0.85 means:
- (a) 85% of predictions are correct
- (b) Model explains 85% of variance in target
- (c) 85% of features are important
- (d) RMSE is 0.85

**Q3.** Polynomial degree 10 on 20 data points likely:
- (a) Underfits
- (b) Overfits
- (c) Is optimal
- (d) Cannot be determined

**Q4.** Random Forest reduces:
- (a) Bias
- (b) Variance
- (c) Both equally
- (d) Neither

**Q5.** Adjusted R² differs from R² by:
- (a) Penalizing extra features
- (b) Using MAE instead of MSE
- (c) Cross-validation
- (d) Log transform

---

## Section B: Classification (6 questions)

**Q6.** Logistic regression outputs:
- (a) Class labels directly
- (b) Probabilities via sigmoid
- (c) Distances to hyperplane only
- (d) Cluster assignments

**Q7.** Naive Bayes "naive" assumption:
- (a) Features are independent given class
- (b) Classes are equally likely
- (c) Data is normally distributed
- (d) No missing values

**Q8.** KNN with k=1:
- (a) Always overfits
- (b) Has lowest bias, highest variance
- (c) Cannot classify
- (d) Requires scaling

**Q9.** SVM with RBF kernel is useful when:
- (a) Data is linearly separable
- (b) Decision boundary is non-linear
- (c) Dataset is very large
- (d) Features are uncorrelated

**Q10.** High precision, low recall means:
- (a) Many false positives
- (b) Many false negatives
- (c) Model finds most positives but with errors
- (d) Model is conservative about predicting positive

**Q11.** For imbalanced classes (1% positive), best metric:
- (a) Accuracy
- (b) F1 or AUC-ROC
- (c) R²
- (d) Silhouette score

---

## Section C: Clustering & Dim Reduction (5 questions)

**Q12.** K-Means requires specifying:
- (a) eps and min_samples
- (b) Number of clusters k
- (c) Linkage method
- (d) Number of components

**Q13.** DBSCAN can:
- (a) Only find spherical clusters
- (b) Find arbitrary shapes and label outliers
- (c) Do supervised learning
- (d) Replace PCA

**Q14.** PCA finds directions of:
- (a) Maximum class separation
- (b) Maximum variance
- (c) Minimum correlation
- (d) Maximum entropy

**Q15.** t-SNE should be used for:
- (a) Feature preprocessing before training
- (b) Visualization only
- (c) Anomaly detection
- (d) Hyperparameter tuning

**Q16.** LDA differs from PCA because LDA:
- (a) Is unsupervised
- (b) Uses class labels
- (c) Is faster
- (d) Works on text data

---

## Section D: Model Selection & Ensemble (5 questions)

**Q17.** 5-fold CV on 1000 samples uses for validation per fold:
- (a) 200 samples
- (b) 800 samples
- (c) 1000 samples
- (d) 5000 samples

**Q18.** High bias model:
- (a) Overfits training data
- (b) Underfits (too simple)
- (c) Has low variance
- (d) Both b and c

**Q19.** Gradient Boosting fits each new tree to:
- (a) Random subset of data
- (b) Residuals of current ensemble
- (c) Original labels only
- (d) PCA-transformed data

**Q20.** GridSearchCV with 3 params (3 values each), 5-fold CV trains:
- (a) 9 models
- (b) 27 models
- (c) 135 models
- (d) 5 models

**Q21.** Feature scaling is critical for:
- (a) Decision Trees
- (b) KNN and SVM
- (c) Naive Bayes
- (d) Random Forest

---

## Section E: Applied (4 questions)

**Q22.** Your Day - 11 script uses:
- (a) Logistic Regression
- (b) Linear Regression
- (c) KNN
- (d) SVM

**Q23.** Isolation Forest detects anomalies by:
- (a) Distance to centroid
- (b) Random partitioning (short path = anomaly)
- (c) Density estimation
- (d) Reconstruction error

**Q24.** SVD recommendation decomposes:
- (a) User-item rating matrix
- (b) Confusion matrix
- (c) Covariance matrix
- (d) Kernel matrix

**Q25.** Before comparing models, you should:
- (a) Use different test sets for each
- (b) Use same CV splits and preprocessing
- (c) Always pick the most complex model
- (d) Skip cross-validation

---

See [solutions/quiz_answers.md](solutions/quiz_answers.md) after submitting.
