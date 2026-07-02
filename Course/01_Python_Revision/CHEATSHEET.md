# Module 01 Cheat Sheet — Python for ML

## NumPy

```python
import numpy as np

# Creation
np.array([1,2,3])           # vector
np.zeros((3, 4))            # 3×4 zeros
np.ones((2, 3))             # 2×3 ones
np.eye(3)                   # 3×3 identity
np.arange(0, 10, 2)         # [0,2,4,6,8]
np.linspace(0, 1, 5)        # 5 evenly spaced
rng = np.random.default_rng(42)
rng.normal(0, 1, (100,))    # random normal

# Shape & dtype
arr.shape                   # dimensions
arr.ndim                    # number of dimensions
arr.dtype                   # data type
arr.reshape(3, 4)           # change shape
arr.flatten()               # to 1D

# Indexing
arr[i, j]                   # element
arr[i, :]                   # row i
arr[:, j]                   # column j
arr[arr > 0]                # boolean indexing

# Operations
a @ b                       # matrix multiply / dot product
a * b                       # element-wise multiply
a + b, a - b               # element-wise add/subtract
arr.sum(), arr.mean(), arr.std()
arr.sum(axis=0)             # column-wise (down rows)
arr.sum(axis=1)             # row-wise (across columns)

# Broadcasting
X - X.mean(axis=0)          # center each column

# Linear algebra
np.linalg.inv(A)            # matrix inverse
np.linalg.det(A)            # determinant
np.linalg.eig(A)            # eigenvalues & eigenvectors
```

## Pandas

```python
import pandas as pd

# Load
df = pd.read_csv('file.csv')

# Inspect
df.head(), df.tail(), df.info(), df.describe()
df.shape, df.columns, df.dtypes
df.isnull().sum()           # missing values

# Select
df['col']                   # Series
df[['a', 'b']]              # DataFrame subset
df.loc[row, 'col']          # label-based
df.iloc[0, 1]               # integer-based

# Filter
df[df['age'] > 30]
df.query('age > 30')

# Transform
df.drop('col', axis=1)
df.fillna(df.median())
df['new'] = df['a'] + df['b']
pd.get_dummies(df['cat'])   # one-hot encode

# Group
df.groupby('class')['score'].mean()
df.groupby(['a', 'b']).agg(['mean', 'count'])

# Merge
pd.merge(df1, df2, on='key')
```

## Matplotlib

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x, y, alpha=0.7)
ax.plot(x, y, label='line')
ax.hist(data, bins=30)
ax.boxplot(data)
ax.imshow(matrix, cmap='Blues')
ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_title('Title')
ax.legend()
plt.tight_layout()
plt.savefig('plot.png', dpi=150)
plt.show()
```

## Vectorization Patterns

```python
# Distance: point to all rows
diff = X - q                    # (n, d) - (d,) → (n, d)
dists = np.sqrt((diff**2).sum(axis=1))

# Pairwise distances (trick)
sq = (X**2).sum(axis=1)
D = np.sqrt(np.maximum(sq[:,None] + sq[None,:] - 2*X@X.T, 0))

# Softmax
e = np.exp(z - z.max())         # subtract max for stability
softmax = e / e.sum()

# Accuracy
acc = (y_true == y_pred).mean()

# Standardize
X_std = (X - X.mean(axis=0)) / X.std(axis=0)
```

## Key Conventions

| Convention | Meaning |
|------------|---------|
| X shape `(n, d)` | n samples, d features |
| y shape `(n,)` | n targets |
| Image `(C, H, W)` | channels, height, width |
| Batch `(N, C, H, W)` | batch of images |
| `float32` | GPU training |
| `float64` | CPU / numerical analysis |
| `seed=42` | reproducibility |

## Common Errors

| Error | Fix |
|-------|-----|
| `(3,4) + (3,)` | Shapes incompatible for broadcasting |
| SettingWithCopyWarning | Use `.copy()` or `.loc[]` |
| `NaN` in model | Check `.isnull().sum()` first |
| Slow loop | Vectorize with NumPy |
