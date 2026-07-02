# Module 02 Quiz

**Passing score:** 16/20 (80%)

---

## Section A: Linear Algebra (5 questions)

**Q1.** What is the shape of $(3 \times 4)(4 \times 2)$?

- (a) (3, 4)  (b) (4, 2)  (c) (3, 2)  (d) Error

**Q2.** The normal equation for linear regression is:

- (a) $w = Xy$
- (b) $w = (X^T X)^{-1} X^T y$
- (c) $w = X^T y$
- (d) $w = (XX^T)^{-1} y$

**Q3.** If $Av = 3v$ for non-zero $v$, then 3 is an ___ and $v$ is an ___.

- (a) eigenvector, eigenvalue
- (b) eigenvalue, eigenvector
- (c) singular value, singular vector
- (d) rank, null vector

**Q4.** A matrix with rank 2 but 5 columns has:

- (a) 5 independent features
- (b) 3 redundant features
- (c) Full rank
- (d) Zero determinant always

**Q5.** $A^T A$ is always:

- (a) Symmetric
- (b) Orthogonal
- (c) Diagonal
- (d) Singular

---

## Section B: Calculus (4 questions)

**Q6.** The gradient points in the direction of:

- (a) Steepest descent
- (b) Steepest ascent
- (c) Zero change
- (d) Maximum curvature

**Q7.** Chain rule is essential for:

- (a) Matrix multiplication
- (b) Backpropagation
- (c) Eigenvalue computation
- (d) Data normalization

**Q8.** MSE loss for linear regression is:

- (a) Convex
- (b) Non-convex
- (c) Discontinuous
- (d) Always negative

**Q9.** $\nabla_w \frac{1}{n}\|Xw - y\|^2 = ?

- (a) $2X(Xw - y)$
- (b) $\frac{2}{n} X^T(Xw - y)$
- (c) $X^T y$
- (d) $(X^T X)^{-1} X^T y$

---

## Section C: Probability (4 questions)

**Q10.** MLE for Gaussian mean equals:

- (a) Sample median
- (b) Sample mean
- (c) Sample mode
- (d) Zero

**Q11.** Bayes theorem requires:

- (a) Independent events
- (b) $P(B) > 0$
- (c) Normal distribution
- (d) Equal priors

**Q12.** Binary cross-entropy is derived from:

- (a) Gaussian MLE
- (b) Bernoulli MLE
- (c) Poisson MLE
- (d) Uniform MLE

**Q13.** Correlation of 0.95 between two features implies:

- (a) One causes the other
- (b) Strong linear relationship
- (c) They are independent
- (d) One should always be removed

---

## Section D: Optimization & Loss (7 questions)

**Q14.** Learning rate too large causes:

- (a) Slow convergence
- (b) Divergence
- (c) Better generalization
- (d) No effect

**Q15.** Adam combines:

- (a) Momentum and RMSProp
- (b) SGD and Newton's method
- (c) L1 and L2 regularization
- (d) Batch and stochastic GD only

**Q16.** Mini-batch size 32 means:

- (a) 32 epochs per update
- (b) Gradient computed on 32 samples per update
- (c) 32 models trained
- (d) Learning rate is 32

**Q17.** Dice loss is preferred over CE when:

- (a) Classes are balanced
- (b) Objects are small relative to image (class imbalance)
- (c) Regression task
- (d) Multi-class with many classes

**Q18.** IoU = 0.8 means:

- (a) 80% of pixels correct
- (b) 80% overlap between prediction and ground truth regions
- (c) 80% precision
- (d) Loss equals 0.8

**Q19.** Negative log-likelihood minimization equals:

- (a) MLE maximization
- (b) MAP estimation always
- (c) Minimizing variance
- (d) Maximizing entropy

**Q20.** Huber loss is robust because:

- (a) Uses squared error for all errors
- (b) Uses linear penalty for large errors
- (c) Is always convex
- (d) Ignores outliers completely

---

## Answer Key

See [solutions/quiz_answers.md](solutions/quiz_answers.md) after submitting.
