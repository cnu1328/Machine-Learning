# Module 01 Exercises

Attempt each exercise before checking [solutions/](solutions/).

---

## Exercise 1: Cosine Similarity (Easy)

**Notebook:** `01_NumPy_Foundations.ipynb`

Implement `cosine_similarity(a, b)` without sklearn:

$$\text{cosine\_sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$$

Test: `cosine_similarity([1,2,3], [4,5,6])` ≈ 0.9746

---

## Exercise 2: Column Standardization (Easy)

**Notebook:** `01_NumPy_Foundations.ipynb`

Standardize each column of a matrix to zero mean and unit variance using only NumPy axis operations.

---

## Exercise 3: Titanic EDA (Medium)

**Notebook:** `02_Pandas_for_ML.ipynb`

Using `TitanicSurvival.csv`:
1. Overall survival rate
2. Survival rate by passenger class
3. Average age of survivors vs non-survivors
4. Missing value count per column

---

## Exercise 4: Feature Matrix (Medium)

**Notebook:** `02_Pandas_for_ML.ipynb`

Create X (Pclass, Sex_encoded, Age, Fare) and y (Survived) from Titanic. Drop rows with NaN Age. Print shapes.

---

## Exercise 5: Multi-panel EDA Plot (Medium)

**Notebook:** `03_Matplotlib_and_Visualization.ipynb`

Create a 2×2 subplot for house price data: scatter, price histogram, area histogram, price boxplot.

---

## Exercise 6: Vectorized Softmax (Medium)

**Notebook:** `04_Vectorization_and_Performance.ipynb`

Implement vectorized softmax. Handle numerical stability (subtract max before exp).

---

## Exercise 7: Vectorized Accuracy (Easy)

**Notebook:** `04_Vectorization_and_Performance.ipynb`

Compute classification accuracy from `y_true` and `y_pred` arrays without loops.

---

## Exercise 8: Pairwise Distance for KNN (Hard)

**Notebook:** `04_Vectorization_and_Performance.ipynb`

Given training matrix X (n×d) and a query q (d,), find the k nearest neighbors using your vectorized distance function. Return indices and distances.

---

## Assignment: Heart Disease EDA Report

**Dataset:** `../../heart_Disease.csv`

Create a complete exploratory data analysis:

1. Shape, dtypes, missing values
2. Summary statistics
3. Correlation heatmap
4. Distribution plots for ≥3 features
5. Target class balance analysis
6. 5 written observations about the data

Save as a notebook or Python script in `exercises/assignment_heart_eda.ipynb`.

---

## Submission

Tell your mentor:

> Module 01 exercises complete. Assignment attached. Ready for review.

Do not open solutions until you have attempted each exercise.
