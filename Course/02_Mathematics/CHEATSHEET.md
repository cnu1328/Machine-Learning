# Module 02 Cheat Sheet — Mathematics for ML

## Linear Algebra

| Concept | Formula |
|---------|---------|
| Vector magnitude | $\|\mathbf{v}\| = \sqrt{\mathbf{v}^T \mathbf{v}}$ |
| Dot product | $\mathbf{a} \cdot \mathbf{b} = \sum a_i b_i = \mathbf{a}^T \mathbf{b}$ |
| Matrix multiply | $(AB)_{ij} = \sum_k a_{ik} b_{kj}$ |
| Transpose product | $(AB)^T = B^T A^T$ |
| 2×2 determinant | $\det = ad - bc$ |
| 2×2 inverse | $A^{-1} = \frac{1}{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$ |
| Normal equation | $\hat{\mathbf{w}} = (X^T X)^{-1} X^T \mathbf{y}$ |
| Eigenvalue | $A\mathbf{v} = \lambda \mathbf{v}$ |
| Projection | $\text{proj}_{\mathbf{a}}\mathbf{b} = \frac{\mathbf{a}^T\mathbf{b}}{\mathbf{a}^T\mathbf{a}}\mathbf{a}$ |
| Covariance | $C = \frac{1}{n} X_c^T X_c$ |

## Calculus

| Concept | Formula |
|---------|---------|
| Derivative | $f'(x) = \lim_{h\to0}\frac{f(x+h)-f(x)}{h}$ |
| Chain rule | $\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$ |
| Gradient | $\nabla f = [\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}]^T$ |
| MSE gradient | $\nabla_{\mathbf{w}} L = \frac{2}{n} X^T(X\mathbf{w} - \mathbf{y})$ |
| Hessian | $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$ |

## Probability & Statistics

| Concept | Formula |
|---------|---------|
| Bayes theorem | $P(A\|B) = \frac{P(B\|A)P(A)}{P(B)}$ |
| Expected value | $E[X] = \sum x P(X=x)$ or $\int x f(x) dx$ |
| Variance | $\text{Var}(X) = E[X^2] - (E[X])^2$ |
| Gaussian PDF | $\mathcal{N}(\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ |
| MLE (Gaussian μ) | $\hat{\mu} = \frac{1}{n}\sum x_i$ |
| Correlation | $\rho = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$ |

## Optimization

| Algorithm | Update Rule |
|-----------|-------------|
| Gradient Descent | $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla L$ |
| Momentum | $\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla L$; $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{v}_{t+1}$ |
| Adam | $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$; $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$; $\mathbf{w} -= \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ |

## Loss Functions

| Loss | Formula | Use |
|------|---------|-----|
| MSE | $\frac{1}{n}\sum(y - \hat{y})^2$ | Regression |
| MAE | $\frac{1}{n}\sum\|y - \hat{y}\|$ | Robust regression |
| Binary CE | $-[y\log\hat{p} + (1-y)\log(1-\hat{p})]$ | Classification |
| Dice | $1 - \frac{2\sum p_i g_i}{\sum p_i + \sum g_i}$ | Segmentation (imbalanced) |
| IoU | $1 - \frac{\sum p_i g_i}{\sum p_i + \sum g_i - \sum p_i g_i}$ | Segmentation |

## NumPy Quick Reference

```python
np.dot(a, b)          # dot product
A @ B                 # matrix multiply
np.linalg.inv(A)      # inverse
np.linalg.det(A)      # determinant
np.linalg.eig(A)      # eigenvalues, eigenvectors
np.linalg.norm(v)     # L2 norm
np.linalg.matrix_rank(A)  # rank
```

## Key Connections to ML

| Math | Algorithm (Module 03+) |
|------|------------------------|
| Normal equation | Linear Regression |
| Gradient descent | All neural networks |
| Bayes theorem | Naive Bayes |
| Eigenvalues | PCA |
| Cross-entropy | Logistic Regression |
| Dice/IoU | Segmentation (Module 07) |
| Adam | water-bodies-detection training |
