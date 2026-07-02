#!/usr/bin/env python3
"""Generate Module 03 Classical ML notebooks (35 total)."""
import json
from pathlib import Path

M03 = Path(__file__).resolve().parent / "03_Classical_ML"
REPO = "REPO = Path('../../').resolve()"


def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": [s]}


def code(s):
    return {"cell_type": "code", "metadata": {}, "source": [s], "outputs": [], "execution_count": None}


def save(name, cells):
    M03.mkdir(parents=True, exist_ok=True)
    (M03 / name).write_text(json.dumps(nb(cells), indent=1))
    print(f"  {name}")


def hdr(num, title, part, dur, objs, legacy=""):
    leg = f"\n\n**Legacy script:** `{legacy}`" if legacy else ""
    return md(
        f"# {num}: {title}\n\n**Module:** 03 Classical ML  \n**Part:** {part}  \n**Duration:** ~{dur}{leg}\n\n---\n\n## Learning Objectives\n\n{objs}"
    )


def footer(summary, nxt=None):
    nxt_line = f"\n\n**Next:** [{nxt}]({nxt})" if nxt else ""
    return md(f"---\n\n## Summary\n\n{summary}{nxt_line}")


SETUP = code(
    "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom pathlib import Path\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\n\nREPO = Path('../../').resolve()\nplt.rcParams['figure.figsize'] = (8, 5)\nrng = np.random.default_rng(42)"
)


def algo_sections(why, math, hyper, pros, cons, ml_use, exercise, interview):
    cells = []
    cells.append(md(f"## 1. Why This Algorithm Exists\n\n{why}"))
    cells.append(md(f"## 2. Mathematical Formulation\n\n{math}"))
    cells.append(md(f"## 3. Hyperparameters\n\n{hyper}"))
    cells.append(md(f"## 4. Advantages & Limitations\n\n**Advantages:** {pros}\n\n**Limitations:** {cons}"))
    cells.append(md(f"## 5. Real-World Applications\n\n{ml_use}"))
    cells.append(md(f"## Exercise\n\n{exercise}"))
    cells.append(code("# YOUR CODE HERE\n"))
    cells.append(md(f"## Interview Questions\n\n{interview}"))
    return cells


NOTEBOOKS = []  # filled by register()


def register(name, cells):
    NOTEBOOKS.append((name, cells))


# ── REGRESSION ──────────────────────────────────────────────────────────

register("01_Linear_Regression.ipynb", [
    hdr("Notebook 01", "Linear Regression", "Regression", "2.5 hrs",
        "1. Derive the normal equation and MSE loss\n2. Implement linear regression with gradient descent\n3. Implement closed-form solution\n4. Walk through Day - 11 house price script",
        "Day - 11 House price Prediction Using Linear Regression.py"),
    md("## 1. Why Linear Regression Exists\n\nPredict a **continuous target** as a linear combination of features:\n$$\\hat{y} = w_0 + w_1 x_1 + \\cdots + w_d x_d = \\mathbf{w}^T \\mathbf{x} + b$$\n\nOldest and most interpretable ML model. Baseline for every regression task."),
    md("## 2. Mathematical Derivation\n\n**Loss (MSE):** $L(\\mathbf{w}) = \\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\mathbf{w}^T\\mathbf{x}_i)^2$\n\n**Normal equation** (Module 02): $\\hat{\\mathbf{w}} = (X^T X)^{-1} X^T \\mathbf{y}$\n\n**Gradient descent:** $w_j \\leftarrow w_j - \\eta \\frac{\\partial L}{\\partial w_j}$"),
    SETUP,
    code("# Load Day - 11 dataset\nhouse = pd.read_csv(REPO / 'houseprice.csv')\nX = house[['area']].values\ny = house['price'].values\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n# Closed-form (with bias)\nX_b = np.column_stack([np.ones(len(X_train)), X_train])\nw = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_train\nprint(f'Intercept: {w[0]:,.0f}, Slope: {w[1]:.2f}')"),
    code("# Gradient descent from scratch\nw_gd = np.zeros(2)\nlr = 1e-7\nfor _ in range(1000):\n    preds = X_b @ w_gd\n    grad = (2/len(X_b)) * X_b.T @ (preds - y_train)\n    w_gd -= lr * grad\nprint(f'GD — Intercept: {w_gd[0]:,.0f}, Slope: {w_gd[1]:.2f}')"),
    code("from sklearn.linear_model import LinearRegression\nm = LinearRegression().fit(X_train, y_train)\nprint(f'sklearn R² train: {m.score(X_train, y_train):.4f}')\nprint(f'sklearn R² test:  {m.score(X_test, y_test):.4f}')\n\nplt.scatter(X_test, y_test, alpha=0.6, label='Actual')\nplt.plot(X_test, m.predict(X_test), 'r-', linewidth=2, label='Predicted')\nplt.xlabel('Area'); plt.ylabel('Price'); plt.legend(); plt.title('Linear Regression — House Prices'); plt.show()"),
    md("## Hyperparameters\n\n| Param | Role |\n|-------|------|\n| None (OLS) | No hyperparameters for basic linear regression |\n| `fit_intercept` | Whether to learn bias term |\n| Regularization (Ridge/Lasso) | Added in Feature Selection notebook |"),
    md("## Exercise\n\nPredict price for 4685 sq ft (same as Day - 11). Compare all three methods."),
    code("# YOUR CODE HERE\n"),
    footer("Linear regression = MSE minimization. Normal equation for small data; GD for large.", "02_Polynomial_Regression.ipynb"),
])

register("02_Polynomial_Regression.ipynb", [
    hdr("Notebook 02", "Polynomial Regression", "Regression", "2 hrs",
        "1. Extend linear regression to non-linear relationships\n2. Understand feature expansion and overfitting\n3. Implement polynomial features\n4. Walk through Day - 12 salary script",
        "Day - 12 Salary prediction using polynomial Regression.py"),
    md("## 1. Why Polynomial Regression\n\nLinear model in **expanded features**, not raw inputs:\n$$\\hat{y} = w_0 + w_1 x + w_2 x^2 + w_3 x^3 + \\cdots$$\n\nCaptures curvature without changing the linear regression solver."),
    SETUP,
    code("salary = pd.read_csv(REPO / 'salary.csv')\nprint(salary.head())\n\nfrom sklearn.preprocessing import PolynomialFeatures\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.pipeline import Pipeline\n\nX = salary[['Level']].values\ny = salary['Salary'].values\n\nfor deg in [1, 2, 3, 10]:\n    pipe = Pipeline([('poly', PolynomialFeatures(deg)), ('lr', LinearRegression())])\n    pipe.fit(X, y)\n    print(f'Degree {deg:2d}: R² = {pipe.score(X, y):.4f}')"),
    code("fig, ax = plt.subplots()\nax.scatter(X, y, label='Data')\nX_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)\nfor deg, c in [(1,'blue'),(3,'green'),(10,'red')]:\n    pipe = Pipeline([('poly', PolynomialFeatures(deg)), ('lr', LinearRegression())])\n    pipe.fit(X, y); ax.plot(X_line, pipe.predict(X_line), c=c, label=f'deg={deg}')\nax.legend(); ax.set_title('Polynomial Regression — Overfitting at degree 10'); plt.show()"),
    md("## Exercise\n\nFind optimal polynomial degree using cross-validation (preview Notebook 30)."),
    code("# YOUR CODE HERE\n"),
    footer("Polynomial regression models non-linearity via feature expansion. Watch for overfitting.", "03_Support_Vector_Regression.ipynb"),
])

register("03_Support_Vector_Regression.ipynb", [
    hdr("Notebook 03", "Support Vector Regression", "Regression", "2 hrs",
        "1. Understand epsilon-insensitive loss\n2. Apply kernel trick for non-linear SVR\n3. Train SVR on time-series-like data",
        "Day - 13 Stock Prediction Using Support Vector Regression.py"),
    md("## 1. Mathematical Formulation\n\nSVR finds a function with at most $\\epsilon$ deviation from targets while keeping $\\|w\\|$ small:\n\nMinimize: $\\frac{1}{2}\\|w\\|^2 + C\\sum_i \\xi_i$\n\nSubject to: $|y_i - w^T\\phi(x_i)| \\leq \\epsilon + \\xi_i$\n\n**Kernel trick:** $\\phi(x)$ never computed explicitly — use $K(x_i, x_j) = \\phi(x_i)^T\\phi(x_j)$"),
    SETUP,
    code("from sklearn.svm import SVR\nfrom sklearn.datasets import make_regression\n\nX, y = make_regression(n_samples=200, n_features=1, noise=15, random_state=42)\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\nsvr = SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')\nsvr.fit(X_train, y_train)\nprint(f'SVR R² test: {svr.score(X_test, y_test):.4f}')"),
    md("## Hyperparameters\n\n| Param | Role |\n|-------|------|\n| `C` | Penalty for violations |\n| `epsilon` | Tube width (no loss inside) |\n| `kernel` | linear, rbf, poly |\n| `gamma` | RBF kernel width |"),
    footer("SVR uses kernel trick for non-linear regression with sparse support vectors.", "04_Decision_Tree_Regression.ipynb"),
])

register("04_Decision_Tree_Regression.ipynb", [
    hdr("Notebook 04", "Decision Tree Regression", "Regression", "2 hrs",
        "1. Understand recursive partitioning\n2. Learn split criteria (MSE reduction)\n3. Train decision tree regressor",
        "Day - 15 Height prediction using Decision Tree.py"),
    md("## 1. How Decision Trees Work\n\nRecursively split data to minimize **impurity**:\n\n**Regression criterion (MSE):** $H(S) = \\frac{1}{|S|}\\sum_{i \\in S}(y_i - \\bar{y}_S)^2$\n\nChoose split that maximizes **information gain** = parent impurity − weighted child impurity."),
    SETUP,
    code("from sklearn.tree import DecisionTreeRegressor, plot_tree\nfrom sklearn.datasets import fetch_california_housing\n\ndata = fetch_california_housing()\nX, y = data.data[:500], data.target[:500]\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\ndt = DecisionTreeRegressor(max_depth=4, random_state=42)\ndt.fit(X_train, y_train)\nprint(f'DT R² test: {dt.score(X_test, y_test):.4f}')"),
    md("## Hyperparameters\n\n| Param | Effect |\n|-------|--------|\n| `max_depth` | Limit tree depth (prevent overfitting) |\n| `min_samples_split` | Min samples to split node |\n| `min_samples_leaf` | Min samples in leaf |"),
    footer("Decision trees partition feature space into axis-aligned rectangles.", "05_Random_Forest_Regression.ipynb"),
])

register("05_Random_Forest_Regression.ipynb", [
    hdr("Notebook 05", "Random Forest Regression", "Regression", "2 hrs",
        "1. Understand bagging and ensemble averaging\n2. Train random forest regressor\n3. Interpret feature importance",
        "Day - 16 Car price Prediction using Random Forest REgression.py"),
    md("## 1. Random Forest = Bagging + Random Subspaces\n\n1. Train $B$ trees on bootstrap samples\n2. At each split, consider random subset of features\n3. Predict by averaging all tree outputs\n\n$$\\hat{f}(x) = \\frac{1}{B}\\sum_{b=1}^{B} T_b(x)$$"),
    SETUP,
    code("from sklearn.ensemble import RandomForestRegressor\n\nrf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)\nrf.fit(X_train, y_train)\nprint(f'RF R² test: {rf.score(X_test, y_test):.4f}')\n\nimportances = rf.feature_importances_\nplt.barh(data.feature_names, importances)\nplt.xlabel('Importance'); plt.title('Random Forest Feature Importance'); plt.show()"),
    footer("Random Forest reduces variance through ensemble averaging.", "06_Regression_Model_Evaluation.ipynb"),
])

register("06_Regression_Model_Evaluation.ipynb", [
    hdr("Notebook 06", "Regression Model Evaluation", "Regression", "2 hrs",
        "1. Compute R², Adjusted R², MAE, RMSE\n2. Compare multiple regression models\n3. Walk through Day - 17 & 18 evaluation script",
        "Day - 17 & 18 Evaluating Regression Model Using Rsquared Adusted Rsquared & Model Selection.py"),
    md("## Metrics\n\n**R²:** $R^2 = 1 - \\frac{SS_{res}}{SS_{tot}} = 1 - \\frac{\\sum(y_i - \\hat{y}_i)^2}{\\sum(y_i - \\bar{y})^2}$\n\n**Adjusted R²:** $R^2_{adj} = 1 - (1-R^2)\\frac{n-1}{n-p-1}$ (penalizes extra features)\n\n**RMSE:** $\\sqrt{\\frac{1}{n}\\sum(y_i - \\hat{y}_i)^2}$\n\n**MAE:** $\\frac{1}{n}\\sum|y_i - \\hat{y}_i|$"),
    SETUP,
    code("from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\nfrom sklearn.linear_model import LinearRegression\n\ny_pred = rf.predict(X_test)\nprint(f'R²:   {r2_score(y_test, y_pred):.4f}')\nprint(f'MAE:  {mean_absolute_error(y_test, y_pred):.4f}')\nprint(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')\n\nn, p = len(y_test), X_test.shape[1]\nr2 = r2_score(y_test, y_pred)\nr2_adj = 1 - (1-r2)*(n-1)/(n-p-1)\nprint(f'Adjusted R²: {r2_adj:.4f}')"),
    footer("Always use multiple metrics. Adjusted R² prevents overfitting via feature stuffing.", "07_Logistic_Regression.ipynb"),
])

# ── CLASSIFICATION ──────────────────────────────────────────────────────

register("07_Logistic_Regression.ipynb", [
    hdr("Notebook 07", "Logistic Regression", "Classification", "2.5 hrs",
        "1. Derive sigmoid and cross-entropy loss\n2. Implement logistic regression with gradient descent\n3. Walk through Day - 3 heart disease script",
        "Day - 3 Logistic_regression_Heart_Diseases.py"),
    md("## 1. Mathematical Derivation\n\n**Sigmoid:** $\\sigma(z) = \\frac{1}{1+e^{-z}}$\n\n**Model:** $P(y=1|x) = \\sigma(w^T x + b)$\n\n**Loss (cross-entropy):** $L = -\\frac{1}{n}\\sum[y\\log\\hat{p} + (1-y)\\log(1-\\hat{p})]$\n\n**Gradient:** $\\frac{\\partial L}{\\partial w} = \\frac{1}{n}X^T(\\hat{p} - y)$ (Module 02 assignment)"),
    SETUP,
    code("heart = pd.read_csv(REPO / 'heart_Disease.csv')\nprint(heart.shape, heart.columns.tolist()[:8])\n\n# Use sklearn for baseline\ntarget_col = heart.columns[-1]\nX = heart.drop(columns=[target_col]).select_dtypes(include=[np.number]).fillna(heart.median(numeric_only=True))\ny = heart[target_col]\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n\nscaler = StandardScaler()\nX_train_s = scaler.fit_transform(X_train)\nX_test_s = scaler.transform(X_test)"),
    code("def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))\n\ndef train_logistic(X, y, lr=0.1, epochs=500):\n    w = np.zeros(X.shape[1])\n    n = len(y)\n    for _ in range(epochs):\n        p = sigmoid(X @ w)\n        grad = X.T @ (p - y) / n\n        w -= lr * grad\n    return w\n\nw = train_logistic(X_train_s, y_train.values)\npred = (sigmoid(X_test_s @ w) > 0.5).astype(int)\nprint(f'Accuracy: {(pred == y_test.values).mean():.4f}')"),
    code("from sklearn.linear_model import LogisticRegression\nlr_model = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)\nprint(f'sklearn accuracy: {lr_model.score(X_test_s, y_test):.4f}')"),
    footer("Logistic regression = linear classifier with sigmoid + cross-entropy.", "08_Naive_Bayes.ipynb"),
])

register("08_Naive_Bayes.ipynb", [
    hdr("Notebook 08", "Naive Bayes", "Classification", "2 hrs",
        "1. Derive Naive Bayes from Bayes theorem\n2. Train Gaussian and Bernoulli variants\n3. Walk through Day - 6 Titanic script",
        "Day - 6 Titanic Survival prediction.py"),
    md("## 1. Derivation\n\n$$P(y|\\mathbf{x}) = \\frac{P(y)P(\\mathbf{x}|y)}{P(\\mathbf{x})} \\propto P(y)\\prod_{j=1}^{d} P(x_j|y)$$\n\n**Naive assumption:** features conditionally independent given class.\n\n**Gaussian NB:** $P(x_j|y) = \\mathcal{N}(\\mu_{yj}, \\sigma_{yj}^2)$"),
    SETUP,
    code("titanic = pd.read_csv(REPO / 'TitanicSurvival.csv')\nprint(titanic.head())\n\nfrom sklearn.naive_bayes import GaussianNB\n\n# Prepare features\ndf = titanic[['Pclass','Age','SibSp','Parch','Fare','Survived']].dropna()\nX = df.drop('Survived', axis=1).values\ny = df['Survived'].values\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\ngnb = GaussianNB().fit(X_train, y_train)\nprint(f'GaussianNB accuracy: {gnb.score(X_test, y_test):.4f}')"),
    footer("Naive Bayes is fast, works well with small data, assumes feature independence.", "09_K_Nearest_Neighbors.ipynb"),
])

register("09_K_Nearest_Neighbors.ipynb", [
    hdr("Notebook 09", "K-Nearest Neighbors", "Classification", "2 hrs",
        "1. Understand instance-based learning\n2. Implement KNN from scratch (Module 01 distance matrix)\n3. Walk through Day - 4 KNN salary script",
        "Day - 4 Salary_Estimatiom_by_KNerst.py"),
    md("## 1. Algorithm\n\n1. Store all training samples\n2. For new point, find $k$ nearest neighbors (by distance)\n3. **Classification:** majority vote\n4. **Regression:** average of neighbor targets\n\nNo training phase — lazy learning."),
    SETUP,
    code("def knn_predict(X_train, y_train, X_test, k=5):\n    preds = []\n    for x in X_test:\n        dists = np.sqrt(((X_train - x)**2).sum(axis=1))\n        nearest = np.argsort(dists)[:k]\n        preds.append(np.round(np.mean(y_train[nearest])))\n    return np.array(preds)\n\nfrom sklearn.datasets import load_iris\niris = load_iris()\nX_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)\n\npred = knn_predict(X_train, y_train, X_test, k=5)\nprint(f'KNN scratch accuracy: {(pred == y_test).mean():.4f}')"),
    code("from sklearn.neighbors import KNeighborsClassifier\nknn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)\nprint(f'sklearn KNN: {knn.score(X_test, y_test):.4f}')"),
    footer("KNN: simple, no assumptions, but slow at inference and sensitive to scale.", "10_Support_Vector_Machines.ipynb"),
])

register("10_Support_Vector_Machines.ipynb", [
    hdr("Notebook 10", "Support Vector Machines", "Classification", "2.5 hrs",
        "1. Understand maximum margin classifier\n2. Derive hinge loss\n3. Train SVM on digits (Day - 5)",
        "Day - 5 Handwritten_Digit_Recognition.py"),
    md("## 1. Mathematical Formulation\n\nFind hyperplane $w^T x + b = 0$ maximizing margin:\n\n$$\\min_{w,b} \\frac{1}{2}\\|w\\|^2 \\quad \\text{s.t.} \\quad y_i(w^T x_i + b) \\geq 1$$\n\n**Soft margin:** add slack variables $\\xi_i$, penalized by $C$.\n\n**Hinge loss:** $L = \\max(0, 1 - y_i \\hat{y}_i)$"),
    SETUP,
    code("from sklearn.svm import SVC\nfrom sklearn.datasets import load_digits\n\ndigits = load_digits()\nX_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)\n\nsvm = SVC(kernel='rbf', C=1.0, gamma='scale')\nsvm.fit(X_train, y_train)\nprint(f'SVM accuracy: {svm.score(X_test, y_test):.4f}')"),
    footer("SVM finds maximum-margin boundary. Kernel trick handles non-linearity.", "11_Decision_Tree_Classifier.ipynb"),
])

register("11_Decision_Tree_Classifier.ipynb", [
    hdr("Notebook 11", "Decision Tree Classifier", "Classification", "2 hrs",
        "1. Understand Gini impurity and entropy\n2. Train decision tree classifier\n3. Walk through Day - 7 leaf species script",
        "Day - 7 Leaf_species_Detection_DescisionTree.py"),
    md("## Split Criteria\n\n**Gini:** $G = 1 - \\sum_k p_k^2$\n\n**Entropy:** $H = -\\sum_k p_k \\log p_k$\n\nChoose split maximizing impurity reduction."),
    SETUP,
    code("from sklearn.tree import DecisionTreeClassifier, plot_tree\n\ndt = DecisionTreeClassifier(max_depth=4, random_state=42)\ndt.fit(X_train, y_train)\nprint(f'DT accuracy: {dt.score(X_test, y_test):.4f}')\n\nplt.figure(figsize=(16,8))\nplot_tree(dt, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, fontsize=8)\nplt.title('Decision Tree — Iris'); plt.show()"),
    footer("Decision trees are interpretable but prone to overfitting.", "12_Random_Forest_Classifier.ipynb"),
])

register("12_Random_Forest_Classifier.ipynb", [
    hdr("Notebook 12", "Random Forest Classifier", "Classification", "2 hrs",
        "1. Apply bagging to classification\n2. Train RF on digits\n3. Walk through Digit recognition randomforest.py",
        "Digit recognition randomforest.py"),
    SETUP,
    code("from sklearn.ensemble import RandomForestClassifier\n\nrf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)\nrf.fit(X_train, y_train)\nprint(f'RF accuracy: {rf.score(X_test, y_test):.4f}')"),
    footer("Random Forest is a strong default classifier for tabular data.", "13_Classification_Model_Evaluation.ipynb"),
])

register("13_Classification_Model_Evaluation.ipynb", [
    hdr("Notebook 13", "Classification Model Evaluation", "Classification", "2 hrs",
        "1. Compute accuracy, precision, recall, F1\n2. Build and interpret confusion matrix\n3. Walk through Day - 9 evaluation script",
        "Day - 9 Evaluatiing Classification model performance.py"),
    md("## Metrics\n\n**Precision:** $P = \\frac{TP}{TP+FP}$ — of predicted positives, how many correct?\n\n**Recall:** $R = \\frac{TP}{TP+FN}$ — of actual positives, how many found?\n\n**F1:** $F1 = 2\\frac{PR}{P+R}$\n\n**Accuracy misleading** when classes imbalanced."),
    SETUP,
    code("from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay\n\ny_pred = rf.predict(X_test)\nprint(classification_report(y_test, y_pred))\n\nConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()\nplt.title('Confusion Matrix — Random Forest Digits'); plt.show()"),
    footer("Use precision/recall/F1 for imbalanced data, not accuracy alone.", "14_Multi_Algorithm_Comparison.ipynb"),
])

register("14_Multi_Algorithm_Comparison.ipynb", [
    hdr("Notebook 14", "Multi-Algorithm Comparison", "Classification", "2.5 hrs",
        "1. Compare LR, KNN, SVM, DT, RF on same dataset\n2. Use cross-validation for fair comparison\n3. Walk through Day - 10 breast cancer script",
        "Day - 10 Breat cancer Detection_VariousMLAlgorithm.py"),
    SETUP,
    code("cancer = pd.read_csv(REPO / 'breat_cancer.csv')\nprint(cancer.shape)\n\ntarget = 'diagnosis' if 'diagnosis' in cancer.columns else cancer.columns[-1]\nX = cancer.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(0)\ny = (cancer[target] == 'M').astype(int) if cancer[target].dtype == object else cancer[target]\n\nfrom sklearn.model_selection import cross_val_score\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.svm import SVC\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.ensemble import RandomForestClassifier\n\nmodels = {\n    'LogisticRegression': LogisticRegression(max_iter=1000),\n    'KNN': KNeighborsClassifier(),\n    'SVM': SVC(),\n    'DecisionTree': DecisionTreeClassifier(random_state=42),\n    'RandomForest': RandomForestClassifier(random_state=42),\n}\n\nfor name, model in models.items():\n    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')\n    print(f'{name:22s}: {scores.mean():.4f} ± {scores.std():.4f}')"),
    footer("Always compare models with cross-validation on the same data splits.", "15_AdaBoost.ipynb"),
])

# ── ENSEMBLE ────────────────────────────────────────────────────────────

for nb_name, title, desc, code_block in [
    ("15_AdaBoost.ipynb", "AdaBoost", "Sequential ensemble weighting misclassified samples.",
     "from sklearn.ensemble import AdaBoostClassifier\nfrom sklearn.tree import DecisionTreeClassifier\n\nada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42)\nada.fit(X_train, y_train)\nprint(f'AdaBoost accuracy: {ada.score(X_test, y_test):.4f}')"),
    ("16_Gradient_Boosting.ipynb", "Gradient Boosting", "Fit each tree to residuals of previous ensemble.",
     "from sklearn.ensemble import GradientBoostingClassifier\n\ngb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)\ngb.fit(X_train, y_train)\nprint(f'GradientBoosting accuracy: {gb.score(X_test, y_test):.4f}')"),
    ("17_XGBoost.ipynb", "XGBoost", "Optimized gradient boosting with regularization.",
     "try:\n    import xgboost as xgb\n    xg = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='logloss')\n    xg.fit(X_train, y_train)\n    print(f'XGBoost accuracy: {xg.score(X_test, y_test):.4f}')\nexcept ImportError:\n    print('Install xgboost: pip install xgboost')"),
    ("18_LightGBM.ipynb", "LightGBM", "Leaf-wise tree growth for speed.",
     "try:\n    import lightgbm as lgb\n    lgbm = lgb.LGBMClassifier(n_estimators=100, max_depth=4, random_state=42, verbose=-1)\n    lgbm.fit(X_train, y_train)\n    print(f'LightGBM accuracy: {lgbm.score(X_test, y_test):.4f}')\nexcept ImportError:\n    print('Install lightgbm: pip install lightgbm')"),
    ("19_CatBoost.ipynb", "CatBoost", "Handles categorical features natively.",
     "try:\n    import catboost as cb\n    cat = cb.CatBoostClassifier(iterations=100, depth=4, random_state=42, verbose=0)\n    cat.fit(X_train, y_train)\n    print(f'CatBoost accuracy: {cat.score(X_test, y_test):.4f}')\nexcept ImportError:\n    print('Install catboost: pip install catboost')"),
]:
    nxt = f"{int(nb_name.split('_')[0])+1:02d}_{nb_name.split('_',1)[1]}"
    register(nb_name, [
        hdr(f"Notebook {nb_name[:2]}", title, "Ensemble Methods", "2 hrs", f"1. Understand {title}\n2. Train and evaluate\n3. Know key hyperparameters"),
        md(f"## 1. Overview\n\n{desc}"),
        SETUP,
        code(code_block),
        md("## Hyperparameters\n\nSee sklearn/xgboost documentation. Key: `n_estimators`, `learning_rate`, `max_depth`."),
        footer(f"{title} is a powerful ensemble method for tabular data competitions.", nxt),
    ])

# ── CLUSTERING & DIM REDUCTION ──────────────────────────────────────────

register("20_K_Means_Clustering.ipynb", [
    hdr("Notebook 20", "K-Means Clustering", "Clustering", "2 hrs", "1. Derive K-means objective\n2. Implement K-means from scratch\n3. Choose optimal k"),
    md("## 1. Objective\n\n$$\\min_{C} \\sum_{k=1}^{K}\\sum_{x_i \\in C_k} \\|x_i - \\mu_k\\|^2$$\n\nAlternate: assign points to nearest centroid, update centroids as cluster means."),
    SETUP,
    code("def kmeans(X, k, max_iters=100):\n    centroids = X[rng.choice(len(X), k, replace=False)]\n    for _ in range(max_iters):\n        dists = np.array([[np.linalg.norm(x - c) for c in centroids] for x in X])\n        labels = dists.argmin(axis=1)\n        new_centroids = np.array([X[labels==i].mean(axis=0) if (labels==i).any() else centroids[i] for i in range(k)])\n        if np.allclose(centroids, new_centroids): break\n        centroids = new_centroids\n    return labels, centroids\n\nfrom sklearn.datasets import load_iris\nX = load_iris().data\nlabels, _ = kmeans(X, 3)\nprint('Cluster sizes:', np.bincount(labels))"),
    code("from sklearn.cluster import KMeans\nfrom sklearn.metrics import silhouette_score\n\nfor k in range(2, 8):\n    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)\n    print(f'k={k}: silhouette={silhouette_score(X, km.labels_):.3f}')"),
    footer("K-Means: fast, scalable, but requires k and assumes spherical clusters.", "21_Hierarchical_Clustering.ipynb"),
])

register("21_Hierarchical_Clustering.ipynb", [
    hdr("Notebook 21", "Hierarchical Clustering", "Clustering", "2 hrs", "1. Understand agglomerative clustering\n2. Read dendrograms\n3. Walk through Day - 20 script",
        "Day - 20 Clustering income spent using Hierarchial clustering.py"),
    md("## Linkage Methods\n\n- **Single:** min distance between clusters\n- **Complete:** max distance\n- **Ward:** minimize variance increase"),
    SETUP,
    code("from scipy.cluster.hierarchy import dendrogram, linkage\nfrom sklearn.datasets import load_iris\n\nX = load_iris().data[:50]\nZ = linkage(X, method='ward')\n\nplt.figure(figsize=(10, 5))\ndendrogram(Z)\nplt.title('Hierarchical Clustering Dendrogram'); plt.xlabel('Sample'); plt.ylabel('Distance'); plt.show()"),
    footer("Hierarchical clustering needs no k upfront; dendrogram shows cluster structure.", "22_DBSCAN.ipynb"),
])

register("22_DBSCAN.ipynb", [
    hdr("Notebook 22", "DBSCAN", "Clustering", "2 hrs", "1. Understand density-based clustering\n2. Handle arbitrary cluster shapes\n3. Identify outliers"),
    md("## Algorithm\n\n- **eps:** neighborhood radius\n- **min_samples:** min points to form dense region\n- Core points → clusters; border points → assigned; rest → noise"),
    SETUP,
    code("from sklearn.cluster import DBSCAN\nfrom sklearn.datasets import make_moons\n\nX, _ = make_moons(n_samples=300, noise=0.05, random_state=42)\ndb = DBSCAN(eps=0.2, min_samples=5).fit(X)\n\nplt.scatter(X[:,0], X[:,1], c=db.labels_, cmap='viridis', s=20)\nplt.title(f'DBSCAN — {len(set(db.labels_))-1} clusters'); plt.show()"),
    footer("DBSCAN finds arbitrary shapes and labels outliers. Sensitive to eps.", "23_Gaussian_Mixture_Models.ipynb"),
])

register("23_Gaussian_Mixture_Models.ipynb", [
    hdr("Notebook 23", "Gaussian Mixture Models", "Clustering", "2 hrs", "1. Understand soft clustering with EM\n2. Compare GMM vs K-Means"),
    md("## Model\n\n$$p(x) = \\sum_{k=1}^{K} \\pi_k \\mathcal{N}(x | \\mu_k, \\Sigma_k)$$\n\nEM algorithm alternates E-step (responsibilities) and M-step (update parameters)."),
    SETUP,
    code("from sklearn.mixture import GaussianMixture\n\ngmm = GaussianMixture(n_components=3, random_state=42).fit(X)\nprint('Converged:', gmm.converged_)\nprint('BIC:', gmm.bic(X))"),
    footer("GMM provides soft cluster assignments and probabilistic framework.", "24_PCA.ipynb"),
])

register("24_PCA.ipynb", [
    hdr("Notebook 24", "Principal Component Analysis", "Dimensionality Reduction", "2 hrs",
        "1. Derive PCA from eigendecomposition (Module 02)\n2. Apply PCA for visualization\n3. Walk through Day - 21 iris script",
        "Day - 21 Plant Iris Clustering Uing Principal Component Analysis.py"),
    md("## Derivation\n\n1. Center data: $X_c = X - \\bar{X}$\n2. Covariance: $C = \\frac{1}{n}X_c^T X_c$\n3. Eigendecompose: $C = V\\Lambda V^T$\n4. Project: $X_{PCA} = X_c V_k$"),
    SETUP,
    code("from sklearn.decomposition import PCA\n\npca = PCA(n_components=2)\nX_pca = pca.fit_transform(X)\nprint('Explained variance:', pca.explained_variance_ratio_)\n\nplt.scatter(X_pca[:,0], X_pca[:,1], c=load_iris().target, cmap='viridis', alpha=0.7)\nplt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('PCA — Iris'); plt.show()"),
    footer("PCA reduces dimensions while preserving maximum variance.", "25_LDA.ipynb"),
])

register("25_LDA.ipynb", [
    hdr("Notebook 25", "Linear Discriminant Analysis", "Dimensionality Reduction", "2 hrs", "1. Understand supervised dimensionality reduction\n2. Compare LDA vs PCA"),
    md("## Objective\n\nMaximize between-class variance / within-class variance. Supervised — uses labels."),
    SETUP,
    code("from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\n\nlda = LinearDiscriminantAnalysis(n_components=2)\nX_lda = lda.fit_transform(X, load_iris().target)\nplt.scatter(X_lda[:,0], X_lda[:,1], c=load_iris().target, cmap='viridis'); plt.title('LDA — Iris'); plt.show()"),
    footer("LDA uses class labels; PCA does not.", "26_tSNE.ipynb"),
])

register("26_tSNE.ipynb", [
    hdr("Notebook 26", "t-SNE", "Dimensionality Reduction", "2 hrs", "1. Understand manifold learning\n2. Visualize high-dimensional data"),
    md("## Idea\n\nPreserve local neighborhood structure in 2D/3D. Non-linear, slow, for visualization only."),
    SETUP,
    code("from sklearn.manifold import TSNE\n\nX_sub = load_iris().data\ny_sub = load_iris().target\nX_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_sub)\nplt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_sub, cmap='viridis', alpha=0.7)\nplt.title('t-SNE — Iris'); plt.show()"),
    footer("t-SNE for visualization only — not for feature preprocessing.", "27_UMAP.ipynb"),
])

register("27_UMAP.ipynb", [
    hdr("Notebook 27", "UMAP", "Dimensionality Reduction", "2 hrs", "1. Compare UMAP vs t-SNE\n2. Use for visualization and preprocessing"),
    SETUP,
    code("try:\n    import umap\n    X_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(X_sub)\n    plt.scatter(X_umap[:,0], X_umap[:,1], c=y_sub, cmap='viridis', alpha=0.7)\n    plt.title('UMAP — Iris'); plt.show()\nexcept ImportError:\n    print('Install umap-learn: pip install umap-learn')"),
    footer("UMAP is faster than t-SNE and preserves more global structure.", "28_Feature_Engineering.ipynb"),
])

# ── FEATURE ENGINEERING & MODEL SELECTION ───────────────────────────────

register("28_Feature_Engineering.ipynb", [
    hdr("Notebook 28", "Feature Engineering", "Model Selection", "2 hrs", "1. Create polynomial, interaction, binning features\n2. Encode categoricals\n3. Handle datetime and text basics"),
    md("## Techniques\n\n- Polynomial / interaction features\n- Log / sqrt transforms for skewed data\n- One-hot / target encoding\n- Domain-specific features (GIS: area, perimeter, shape index)"),
    SETUP,
    code("titanic = pd.read_csv(REPO / 'TitanicSurvival.csv')\n\ndf = titanic.copy()\ndf['FamilySize'] = df['SibSp'] + df['Parch'] + 1\ndf['IsAlone'] = (df['FamilySize'] == 1).astype(int)\ndf = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)\nprint(df[['FamilySize','IsAlone']].head())\nprint('\\nNew columns:', [c for c in df.columns if c not in titanic.columns])"),
    footer("Feature engineering often beats algorithm tuning.", "29_Feature_Selection.ipynb"),
])

register("29_Feature_Selection.ipynb", [
    hdr("Notebook 29", "Feature Selection", "Model Selection", "2 hrs", "1. Filter, wrapper, embedded methods\n2. Apply RFE and L1 regularization"),
    SETUP,
    code("from sklearn.feature_selection import RFE, SelectFromModel\nfrom sklearn.linear_model import LogisticRegression\n\nX_f = df.select_dtypes(include=[np.number]).drop(columns=['Survived'], errors='ignore').fillna(0)\ny_f = df['Survived']\n\nrfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=5)\nrfe.fit(X_f, y_f)\nprint('Selected:', list(X_f.columns[rfe.support_]))"),
    footer("Remove redundant features to reduce overfitting and training time.", "30_Cross_Validation.ipynb"),
])

register("30_Cross_Validation.ipynb", [
    hdr("Notebook 30", "Cross-Validation", "Model Selection", "2 hrs", "1. Understand k-fold CV\n2. Apply stratified CV\n3. Avoid data leakage"),
    md("## k-Fold CV\n\nSplit data into k folds. Train on k-1, validate on 1. Repeat k times. Average score.\n\n**Stratified:** preserve class proportions in each fold."),
    SETUP,
    code("from sklearn.model_selection import cross_val_score, StratifiedKFold\n\nX_c = cancer.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(0)\ny_c = (cancer[target] == 'M').astype(int) if cancer[target].dtype == object else cancer[target]\n\ncv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\nscores = cross_val_score(RandomForestClassifier(random_state=42), X_c, y_c, cv=cv)\nprint(f'5-fold CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}')"),
    footer("Cross-validation gives reliable performance estimates.", "31_Bias_Variance_Tradeoff.ipynb"),
])

register("31_Bias_Variance_Tradeoff.ipynb", [
    hdr("Notebook 31", "Bias-Variance Tradeoff", "Model Selection", "2 hrs", "1. Decompose prediction error\n2. Visualize under/overfitting"),
    md("## Decomposition\n\n$$\\text{Error} = \\text{Bias}^2 + \\text{Variance} + \\text{Irreducible Noise}$$\n\n- **High bias:** underfitting (too simple)\n- **High variance:** overfitting (too complex)"),
    SETUP,
    code("from sklearn.model_selection import learning_curve\n\ntrain_sizes, train_scores, val_scores = learning_curve(\n    DecisionTreeClassifier(max_depth=6, random_state=42), X_c, y_c,\n    train_sizes=np.linspace(0.1, 1.0, 8), cv=5)\n\nplt.plot(train_sizes, train_scores.mean(axis=1), label='Train')\nplt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')\nplt.xlabel('Training size'); plt.ylabel('Score'); plt.legend(); plt.title('Learning Curve'); plt.show()"),
    footer("Balance model complexity: too simple = bias, too complex = variance.", "32_Hyperparameter_Tuning.ipynb"),
])

register("32_Hyperparameter_Tuning.ipynb", [
    hdr("Notebook 32", "Hyperparameter Tuning", "Model Selection", "2 hrs", "1. Grid search and random search\n2. Tune RF hyperparameters"),
    SETUP,
    code("from sklearn.model_selection import GridSearchCV\n\nparam_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5, 10], 'min_samples_split': [2, 5]}\ngrid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)\ngrid.fit(X_c, y_c)\nprint('Best params:', grid.best_params_)\nprint('Best CV score:', grid.best_score_:.4f)"),
    footer("Systematic hyperparameter search beats manual guessing.", "33_Isolation_Forest.ipynb"),
])

# ── ANOMALY & RECOMMENDATION ────────────────────────────────────────────

register("33_Isolation_Forest.ipynb", [
    hdr("Notebook 33", "Isolation Forest", "Anomaly Detection", "2 hrs", "1. Understand anomaly detection\n2. Apply Isolation Forest"),
    SETUP,
    code("from sklearn.ensemble import IsolationForest\n\nX_normal = rng.normal(0, 1, (200, 2))\nX_outliers = rng.uniform(-4, 4, (20, 2))\nX_all = np.vstack([X_normal, X_outliers])\n\niso = IsolationForest(contamination=0.1, random_state=42).fit(X_all)\nlabels = iso.predict(X_all)\n\nplt.scatter(X_all[:,0], X_all[:,1], c=labels, cmap='coolwarm', alpha=0.7)\nplt.title('Isolation Forest — Anomaly Detection'); plt.show()"),
    footer("Isolation Forest detects anomalies by random partitioning.", "34_Autoencoders_Classical.ipynb"),
])

register("34_Autoencoders_Classical.ipynb", [
    hdr("Notebook 34", "Autoencoders for Anomaly Detection", "Anomaly Detection", "2 hrs", "1. Understand encoder-decoder for reconstruction\n2. Detect anomalies via reconstruction error"),
    md("## Idea\n\nTrain network to reconstruct normal data. Anomalies have high reconstruction error.\n\n(Full autoencoder implementation in Module 05 Deep Learning.)"),
    SETUP,
    code("# PCA as linear autoencoder baseline\nfrom sklearn.decomposition import PCA\n\nX_n = rng.normal(0, 1, (500, 10))\nX_n[0] = 10  # inject anomaly\n\npca = PCA(n_components=5).fit(X_n[1:])\nreconstructed = pca.inverse_transform(pca.transform(X_n))\nerrors = ((X_n - reconstructed)**2).mean(axis=1)\nprint('Reconstruction error (normal):', errors[1:6])\nprint('Reconstruction error (anomaly):', errors[0])"),
    footer("Reconstruction-based anomaly detection previewed here; full NN version in Module 05.", "35_SVD_Recommendation.ipynb"),
])

register("35_SVD_Recommendation.ipynb", [
    hdr("Notebook 35", "SVD Recommendation System", "Recommendation", "2.5 hrs",
        "1. Understand matrix factorization\n2. Build movie recommender with SVD\n3. Walk through Day - 22 script",
        "Day - 22 Movie Recommendation system using SVD.py"),
    md("## Matrix Factorization\n\nRating matrix $R \\approx U \\Sigma V^T$\n\n- $U$ = user factors\n- $V$ = item factors\n- Low-rank approximation captures latent preferences"),
    SETUP,
    code("# Simple user-item matrix\nR = np.array([[5, 3, 0, 1], [4, 0, 0, 1], [1, 1, 0, 5], [0, 0, 5, 4], [0, 1, 5, 4]], dtype=float)\n\nU, S, Vt = np.linalg.svd(R, full_matrices=False)\n\nk = 2  # latent factors\nR_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]\nprint('Original:\\n', R)\nprint('\\nApprox (k=2):\\n', np.round(R_approx, 1))"),
    md("## Exercise\n\nPredict rating for user 0 on item 2 using the k=2 approximation."),
    code("# YOUR CODE HERE\n"),
    footer("SVD matrix factorization powers collaborative filtering recommenders.", None),
])


def main():
    print("Building Module 03 notebooks...")
    for name, cells in NOTEBOOKS:
        save(name, cells)
    print(f"Done: {len(NOTEBOOKS)} notebooks created.")


if __name__ == "__main__":
    main()
