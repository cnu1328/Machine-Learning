# Module 03 Cheat Sheet — Classical Machine Learning

## Regression

| Algorithm | Key Idea | Loss/Criterion | When to Use |
|-----------|----------|----------------|-------------|
| Linear Regression | $\hat{y} = w^Tx + b$ | MSE | Linear relationships, interpretability |
| Polynomial Regression | Linear in expanded features | MSE | Non-linear trends |
| SVR | Epsilon tube + margin | Epsilon-insensitive | Non-linear regression, outliers |
| Decision Tree | Axis-aligned splits | MSE / Gini | Non-linear, interpretable |
| Random Forest | Bagging + random features | Ensemble avg | Strong default, tabular |

**Normal equation:** $\hat{w} = (X^TX)^{-1}X^Ty$

## Classification

| Algorithm | Key Idea | When to Use |
|-----------|----------|-------------|
| Logistic Regression | $P(y=1) = \sigma(w^Tx)$ | Linear boundary, probabilities |
| Naive Bayes | $P(y|x) \propto P(y)\prod P(x_j|y)$ | Text, small data, fast |
| KNN | Majority vote of k neighbors | Small datasets, no training |
| SVM | Max margin + kernel trick | High-dim, clear margin |
| Decision Tree | Recursive splits (Gini/entropy) | Interpretable rules |
| Random Forest | Bagged trees | Strong default |

**Sigmoid:** $\sigma(z) = 1/(1+e^{-z})$

**Cross-entropy:** $L = -[y\log\hat{p} + (1-y)\log(1-\hat{p})]$

## Ensemble

| Algorithm | Strategy |
|-----------|----------|
| AdaBoost | Reweight misclassified samples |
| Gradient Boosting | Fit trees to residuals |
| XGBoost | Regularized GB + parallel |
| LightGBM | Leaf-wise growth, fast |
| CatBoost | Native categorical handling |

## Clustering

| Algorithm | Needs k? | Shape | Outliers |
|-----------|----------|-------|----------|
| K-Means | Yes | Spherical | No |
| Hierarchical | No (dendrogram) | Any | No |
| DBSCAN | No (eps) | Any | Yes |
| GMM | Yes | Elliptical | Soft assignment |

## Dimensionality Reduction

| Method | Supervised? | Use |
|--------|-------------|-----|
| PCA | No | Preprocessing, visualization |
| LDA | Yes | Classification preprocessing |
| t-SNE | No | Visualization only |
| UMAP | No | Visualization + preprocessing |

## Evaluation

### Regression
- **R²:** $1 - SS_{res}/SS_{tot}$
- **RMSE:** $\sqrt{mean((y-\hat{y})^2)}$
- **MAE:** $mean(|y-\hat{y}|)$

### Classification
- **Precision:** $TP/(TP+FP)$
- **Recall:** $TP/(TP+FN)$
- **F1:** $2PR/(P+R)$
- **Confusion matrix:** TP, FP, TN, FN

## Model Selection

- **Train/Val/Test split:** 60/20/20 or 70/15/15
- **k-Fold CV:** Average k validation scores
- **GridSearchCV:** Exhaustive hyperparameter search
- **Bias-Variance:** Error = Bias² + Variance + Noise

## sklearn Quick Reference

```python
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score
```

## Legacy Script Map

| Day Script | Notebook |
|------------|----------|
| Day - 11 | 01 Linear Regression |
| Day - 12 | 02 Polynomial Regression |
| Day - 13 | 03 SVR |
| Day - 15 | 04 Decision Tree Reg |
| Day - 16 | 05 Random Forest Reg |
| Day - 17 & 18 | 06 Regression Evaluation |
| Day - 3 | 07 Logistic Regression |
| Day - 6 | 08 Naive Bayes |
| Day - 4 | 09 KNN |
| Day - 5 | 10 SVM |
| Day - 7 | 11 Decision Tree Clf |
| Day - 9 | 13 Classification Evaluation |
| Day - 10 | 14 Multi-Algorithm |
| Day - 20 | 21 Hierarchical |
| Day - 21 | 24 PCA |
| Day - 22 | 35 SVD Recommendation |

## Common Mistakes

- Not scaling features for KNN/SVM/PCA
- Using accuracy on imbalanced data
- Data leakage (fit scaler on full dataset)
- Overfitting polynomial regression (high degree)
- Using t-SNE features for ML training
