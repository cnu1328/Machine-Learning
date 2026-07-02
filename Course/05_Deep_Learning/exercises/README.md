# Module 05 Exercises

Attempt before checking [solutions/](solutions/).

---

## NumPy From Scratch (Notebooks 01–04)

### Exercise 1 — Step Activation AND Gate (Notebook 01)
Implement step activation neuron. Verify AND gate truth table.

### Exercise 2 — XOR with MLP (Notebook 02–03)
Build 2→4→1 MLP in NumPy. Train on XOR until 100% accuracy.

### Exercise 3 — Backprop by Hand (Notebook 04)
For 2-layer network (2→3→1), compute $\partial L/\partial W^{[1]}$ manually. Verify with PyTorch autograd.

---

## PyTorch (Notebooks 05–12)

### Exercise 4 — Activation Comparison (Notebook 05)
Train same MLP with sigmoid vs ReLU hidden layers on MNIST subset. Compare convergence speed.

### Exercise 5 — Optimizer Comparison (Notebook 07)
Train MNIST MLP with SGD, SGD+Momentum, Adam. Plot loss curves.

### Exercise 6 — Initialization Effect (Notebook 08)
Compare random small init vs Kaiming init. Plot activation histograms after first forward pass.

### Exercise 7 — Dropout Effect (Notebook 09)
Train with dropout=0, 0.3, 0.5. Plot train vs validation accuracy gap.

### Exercise 8 — LR Scheduler (Notebook 11)
Train with fixed LR vs CosineAnnealingLR. Compare final accuracy.

---

## Module Assignment: Day - 28 Reimplementation

**Deliverable:** `exercises/assignment_pytorch_mlp.ipynb`

Reimplement your **Day - 28** Keras diabetes model in PyTorch:

1. **Architecture:** 8 → 12 (ReLU) → 8 (ReLU) → 1 (sigmoid) — same as Day - 28
2. **Data:** Use sklearn diabetes dataset (Day - 28 uses pima-indians — use `load_diabetes` or fetch Pima online)
3. **Training:** Proper train/val/test split, Adam optimizer, BCE loss
4. **Compare:** Keras-style epochs vs PyTorch training loop
5. **Save/load:** `torch.save` / `torch.load` model weights
6. **Report:** Side-by-side comparison table (Keras Day - 28 vs your PyTorch)

Target: ≥75% accuracy on test set.

---

## Mini Projects

| Project | Notebook | Goal |
|---------|----------|------|
| MNIST NumPy MLP | 03–04 | Train without PyTorch |
| MNIST PyTorch MLP | 12 | ≥95% on 10K subset |
| Spiral classification | 12 | Visualize decision boundary |

---

## Submission

> Module 05 exercises complete. Assignment attached. Quiz score: X/20.
