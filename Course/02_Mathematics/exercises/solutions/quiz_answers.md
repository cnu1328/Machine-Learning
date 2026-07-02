# Module 02 Quiz — Answer Key

| Q | Answer | Explanation |
|---|--------|-------------|
| 1 | (c) | (3×4)(4×2) → (3×2) |
| 2 | (b) | Normal equation |
| 3 | (b) | λ=3 is eigenvalue, v is eigenvector |
| 4 | (b) | 5 - 2 = 3 redundant columns |
| 5 | (a) | (A^T A)^T = A^T A → symmetric |
| 6 | (b) | Gradient = steepest ascent direction |
| 7 | (b) | Backprop = chain rule applied to computation graph |
| 8 | (a) | MSE is convex quadratic in w |
| 9 | (b) | Standard MSE gradient |
| 10 | (b) | Sample mean is MLE for μ |
| 11 | (b) | Need P(B) > 0 to divide |
| 12 | (b) | Negated Bernoulli log-likelihood |
| 13 | (b) | High correlation, not necessarily causation |
| 14 | (b) | Overshooting minimum |
| 15 | (a) | 1st moment (momentum) + 2nd moment (RMSProp) |
| 16 | (b) | Batch of 32 samples per gradient step |
| 17 | (b) | Dice handles small objects / imbalance |
| 18 | (b) | Intersection over union of regions |
| 19 | (a) | min(-log L) = max(L) = MLE |
| 20 | (b) | Linear for large errors → less outlier sensitivity |

---

## Assignment Solution Sketch: Logistic Regression Gradient

**Derivation:**

Let $\hat{p} = \sigma(z)$ where $z = w^T x$.

$$L = -[y \log \hat{p} + (1-y) \log(1-\hat{p})]$$

Using $\sigma'(z) = \sigma(z)(1-\sigma(z))$:

$$\frac{\partial L}{\partial z} = \hat{p} - y$$

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w} = (\hat{p} - y) \cdot x$$

**Implementation:**

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def logistic_loss_grad(X, y, w):
    p = sigmoid(X @ w)
    return X.T @ (p - y) / len(y)
```

---

## Exercise Solutions

### Exercise 1
```python
a, b = np.array([1,2,2]), np.array([2,1,-2])
cos_theta = np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))
angle = np.degrees(np.arccos(cos_theta))  # ~66.42°
```

### Exercise 11 (Bayes)
```python
P_spam = 0.3
P_free_spam = 0.9
P_free_ham = 0.1
P_free = P_free_spam * P_spam + P_free_ham * (1 - P_spam)
P_spam_given_free = P_free_spam * P_spam / P_free  # 0.79
```

### Exercise 13 (GD)
```python
w = 10.0
for _ in range(50):
    w -= 0.1 * (2*w + 4)  # f'(w) = 2w + 4
# w → -2.0
```
