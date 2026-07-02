# Module 02 Exercises

Attempt every exercise before checking [solutions/](solutions/).

---

## Part A: Linear Algebra

### Exercise 1 — Angle Between Vectors (Notebook 01)
Compute the angle (degrees) between $\mathbf{a} = [1, 2, 2]$ and $\mathbf{b} = [2, 1, -2]$.

### Exercise 2 — Matrix-Vector Product (Notebook 02)
Compute $X\mathbf{w}$ for $X = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}$, $\mathbf{w} = [1, 1]^T$ by hand and verify.

### Exercise 3 — Transpose Property (Notebook 03)
Verify $(AB)^T = B^T A^T$ for random matrices.

### Exercise 4 — Matrix Inverse (Notebook 04)
Invert $\begin{bmatrix} 2 & 1 \\ 5 & 3 \end{bmatrix}$ by hand using the 2×2 formula.

### Exercise 5 — Rank (Notebook 05)
Identify redundant column in $\begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 1 & 1 \end{bmatrix}$.

### Exercise 6 — Eigenvalues (Notebook 06)
Solve $\det(A - \lambda I) = 0$ for $A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$ by hand.

### Exercise 7 — Projection (Notebook 07)
Project $\mathbf{b} = [3, 4]$ onto $\mathbf{a} = [1, 0]$. Compute residual.

---

## Part B: Calculus

### Exercise 8 — Product Rule (Notebook 08)
Differentiate $(x^2 + 3x)(\sin x)$. Verify at $x = 1$.

### Exercise 9 — MSE Gradient (Notebook 09)
Derive $\nabla_{\mathbf{w}} \|X\mathbf{w} - \mathbf{y}\|^2$ on paper. Verify numerically.

### Exercise 10 — Hessian (Notebook 10)
Compute Hessian of $f(x,y) = x^3 + xy^2$ at $(1, 2)$.

---

## Part C: Probability

### Exercise 11 — Bayes Theorem (Notebook 12)
Spam filter: P(spam)=0.3, P("free"|spam)=0.9, P("free"|ham)=0.1. Find P(spam|"free").

### Exercise 12 — MLE Variance (Notebook 15)
Derive MLE for Gaussian variance. Verify with simulation.

---

## Part D: Optimization

### Exercise 13 — Gradient Descent (Notebook 18)
Minimize $f(w) = w^2 + 4w + 6$ with GD from $w=10$. Analytical min at $w=-2$.

### Exercise 14 — RMSProp (Notebook 20)
Implement RMSProp from scratch. Compare with SGD.

### Exercise 15 — Cross-Entropy (Notebook 22)
Implement multi-class cross-entropy with softmax. Verify gradient.

---

## Module Assignment: Logistic Regression Loss Gradient

**Deliverable:** Notebook or script demonstrating:

1. **Derive on paper:** For binary cross-entropy loss
   $$L(w) = -[y \log \sigma(w^T x) + (1-y) \log(1 - \sigma(w^T x))]$$
   show that $\frac{\partial L}{\partial w} = (\sigma(w^T x) - y) \cdot x$

2. **Implement in NumPy:** Logistic regression training using this gradient

3. **Verify:** Compare analytical gradient with numerical gradient

4. **Train:** On `heart_Disease.csv` (binary target), achieve >75% accuracy

5. **Compare:** With sklearn `LogisticRegression` — coefficients should match

Save as `exercises/assignment_logistic_regression.ipynb`.

---

## Submission

> Module 02 exercises complete. Assignment attached. Quiz score: X/20. Ready for gate questions.
