# Module 01 Quiz — Answer Key

**Do not read until you have completed the quiz.**

---

| Q | Answer | Explanation |
|---|--------|-------------|
| 1 | (b) | 2 rows, 3 columns |
| 2 | (b) | axis=0 collapses rows → one mean per column |
| 3 | (c) | (50,10) @ (10,3) → (50,3) |
| 4 | (a) | (3,4)+(4,) broadcasts; (3,4)+(3,) incompatible; @ is not broadcasting |
| 5 | (d) | All reasons apply |
| 6 | (a) | loc is label-based indexing |
| 7 | (b) | Test set stats must not influence training preprocessing |
| 8 | (b) | Creates binary columns per category |
| 9 | (a) | GroupBy aggregation |
| 10 | (a) | Modifying a view triggers the warning |
| 11 | (a) | Boxplots compare groups and show outliers |
| 12 | (b) | NumPy uses compiled C; loops use Python interpreter |
| 13 | (a) | exp(large number) overflows; subtracting max preserves result |
| 14 | (b) | Rows = actual, columns = predicted |
| 15 | (a) | Same seed → same random sequence |

---

## Exercise Solutions

### Exercise 1: Cosine Similarity

```python
def cosine_similarity(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### Exercise 2: Standardization

```python
X_std = (X - X.mean(axis=0)) / X.std(axis=0)
```

### Exercise 6: Softmax

```python
def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()
```

### Exercise 7: Accuracy

```python
accuracy = (y_true == y_pred).mean()
```

### Exercise 8: KNN

```python
def knn(X, q, k=5):
    dists = np.sqrt(((X - q) ** 2).sum(axis=1))
    idx = np.argsort(dists)[:k]
    return idx, dists[idx]
```
