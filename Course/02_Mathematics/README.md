# Module 02 — Mathematics for Machine Learning

**Duration:** 6–8 weeks  
**Prerequisites:** Module 01 complete  
**Status:** Ready

---

## Overview

Every ML algorithm is mathematics. This module teaches all math from first principles — no skipped derivations. By the end, you will read equations in papers and implement them in code.

---

## Study Plan (6–8 weeks at 5–8 hrs/week)

| Week | Part | Notebooks | Focus |
|------|------|-----------|-------|
| 1 | A | 01–03 | Vectors, matrices, matrix multiply, normal equation |
| 2 | A | 04–07 | Inverse, rank, eigenvalues, PCA, projection |
| 3 | B | 08–11 | Derivatives, gradient, Hessian, loss landscapes |
| 4 | C | 12–15 | Probability, distributions, MLE |
| 5 | C | 16–17 | Statistics, hypothesis testing |
| 6 | D | 18–20 | Gradient descent, SGD, Adam |
| 7 | E | 21–23 | Loss functions (regression, classification, segmentation) |
| 8 | Review | All | Exercises, assignment, quiz, gate questions |

---

## Part A: Linear Algebra (Notebooks 01–07)

| # | Notebook | Topics |
|---|----------|--------|
| 01 | [01_Scalars_and_Vectors.ipynb](01_Scalars_and_Vectors.ipynb) | Magnitude, dot product, unit vectors |
| 02 | [02_Matrices.ipynb](02_Matrices.ipynb) | Transpose, special matrices, matrix-vector multiply |
| 03 | [03_Matrix_Multiplication.ipynb](03_Matrix_Multiplication.ipynb) | Matrix product, **normal equation derivation** |
| 04 | [04_Matrix_Inverse_and_Determinant.ipynb](04_Matrix_Inverse_and_Determinant.ipynb) | Determinant, inverse, solving Ax=b |
| 05 | [05_Rank_and_Linear_Independence.ipynb](05_Rank_and_Linear_Independence.ipynb) | Rank, feature redundancy |
| 06 | [06_Eigenvalues_and_Eigenvectors.ipynb](06_Eigenvalues_and_Eigenvectors.ipynb) | Eigendecomposition, **PCA derivation** |
| 07 | [07_Orthogonality_and_Projection.ipynb](07_Orthogonality_and_Projection.ipynb) | Projection, Gram-Schmidt |

---

## Part B: Calculus (Notebooks 08–11)

| # | Notebook | Topics |
|---|----------|--------|
| 08 | [08_Functions_and_Derivatives.ipynb](08_Functions_and_Derivatives.ipynb) | Power rule, chain rule |
| 09 | [09_Partial_Derivatives_and_Gradient.ipynb](09_Partial_Derivatives_and_Gradient.ipynb) | Gradient, **MSE gradient derivation** |
| 10 | [10_Jacobian_and_Hessian.ipynb](10_Jacobian_and_Hessian.ipynb) | Jacobian, Hessian, curvature |
| 11 | [11_Optimization_Landscapes.ipynb](11_Optimization_Landscapes.ipynb) | Convexity, saddle points |

---

## Part C: Probability & Statistics (Notebooks 12–17)

| # | Notebook | Topics |
|---|----------|--------|
| 12 | [12_Probability_Foundations.ipynb](12_Probability_Foundations.ipynb) | Bayes theorem |
| 13 | [13_Random_Variables.ipynb](13_Random_Variables.ipynb) | PMF, PDF, CDF, expectation |
| 14 | [14_Distributions.ipynb](14_Distributions.ipynb) | Gaussian, Bernoulli, Binomial, Poisson |
| 15 | [15_Maximum_Likelihood_Estimation.ipynb](15_Maximum_Likelihood_Estimation.ipynb) | MLE derivations |
| 16 | [16_Descriptive_Statistics.ipynb](16_Descriptive_Statistics.ipynb) | Covariance, correlation |
| 17 | [17_Hypothesis_Testing.ipynb](17_Hypothesis_Testing.ipynb) | p-values, t-test |

---

## Part D: Optimization (Notebooks 18–20)

| # | Notebook | Topics |
|---|----------|--------|
| 18 | [18_Gradient_Descent.ipynb](18_Gradient_Descent.ipynb) | GD from scratch on house prices |
| 19 | [19_SGD_and_Mini_batch.ipynb](19_SGD_and_Mini_batch.ipynb) | Batch size effects |
| 20 | [20_Momentum_and_Adaptive_Methods.ipynb](20_Momentum_and_Adaptive_Methods.ipynb) | Momentum, Adam implementation |

---

## Part E: Loss Functions (Notebooks 21–23)

| # | Notebook | Topics |
|---|----------|--------|
| 21 | [21_Regression_Losses.ipynb](21_Regression_Losses.ipynb) | MSE, MAE, Huber |
| 22 | [22_Classification_Losses.ipynb](22_Classification_Losses.ipynb) | Cross-entropy, focal loss |
| 23 | [23_Segmentation_Losses.ipynb](23_Segmentation_Losses.ipynb) | Dice, IoU, water-bodies loss |

---

## Module Deliverables

- [ ] All 23 notebooks completed
- [ ] 15 exercises attempted
- [ ] Assignment: logistic regression gradient derivation + implementation
- [ ] Quiz ≥16/20 (80%)
- [ ] Gate questions answered in chat

---

## Assignment

**Logistic Regression Loss Gradient** — derive $\frac{\partial L}{\partial w} = (\sigma(w^T x) - y) x$, implement training on `heart_Disease.csv`, compare with sklearn.

See [exercises/README.md](exercises/README.md).

---

## Interview Questions

1. Derive the normal equation for linear regression.
2. What is the gradient of MSE loss?
3. Explain Bayes theorem with a real example.
4. Why is MLE for Gaussian mean equal to the sample mean?
5. Derive binary cross-entropy from Bernoulli MLE.
6. Compare SGD, Momentum, and Adam.
7. When would you use Dice loss instead of cross-entropy?
8. What happens when $X^T X$ is singular?

---

## Common Mistakes

- Confusing $(AB)^T = B^T A^T$ with $(AB)^{-1} = B^{-1} A^{-1}$ (order reverses for both!)
- Forgetting to normalize features before gradient descent
- Using population variance (n) vs sample variance (n-1)
- Assuming high correlation implies causation
- Setting learning rate too high for ill-conditioned problems

---

## Real-World Applications

| Math | Your Projects |
|------|---------------|
| Normal equation | Day - 11 house price regression |
| PCA | Day - 21 iris clustering |
| Gradient descent | All deep learning training |
| Bayes theorem | Day - 6 Titanic Naive Bayes |
| Adam optimizer | water-bodies-detection train.py |
| Dice/IoU loss | water-bodies-detection losses.py |

---

## Module Gate

Before Module 03, complete all deliverables and answer 3 mentor questions in chat.

---

## Resources

- [CHEATSHEET.md](CHEATSHEET.md)
- [quiz/module_02_quiz.md](quiz/module_02_quiz.md)
- [exercises/README.md](exercises/README.md)
- [Further reading](../references/FURTHER_READING.md)

---

**Previous:** [01_Python_Revision/](../01_Python_Revision/)  
**Next:** [03_Classical_ML/](../03_Classical_ML/)
