# Module 05 — Deep Learning from Scratch

**Duration:** 4–5 weeks  
**Prerequisites:** Module 02 (calculus, optimization) and Module 04  
**Status:** Ready

---

## Overview

Build neural networks from first principles: **NumPy first** (understand every gradient), then **PyTorch** (production training). Extends your `Day - 28` and `Day - 29` Keras scripts into modern PyTorch implementations.

**Framework:** NumPy (notebooks 01–04) → PyTorch (notebooks 05–12). No Keras/TensorFlow taught.

---

## Study Plan (4–5 weeks at 5–8 hrs/week)

| Week | Notebooks | Focus |
|------|-----------|-------|
| 1 | 01–04 | Neuron, perceptron, MLP, backprop in NumPy |
| 2 | 05–07 | Activations, losses, optimizers in PyTorch |
| 3 | 08–10 | Init, dropout, batch norm |
| 4 | 11–12 | LR scheduling, residuals, MNIST project |
| 5 | Review | Exercises, Day - 28 assignment, quiz |

---

## Notebooks

| # | Notebook | Framework | Key Topic |
|---|----------|-----------|-----------|
| 01 | [01_Artificial_Neuron.ipynb](01_Artificial_Neuron.ipynb) | NumPy | Weighted sum + activation |
| 02 | [02_Perceptron.ipynb](02_Perceptron.ipynb) | NumPy | Perceptron rule, XOR problem |
| 03 | [03_Multi_Layer_Perceptron.ipynb](03_Multi_Layer_Perceptron.ipynb) | NumPy | Forward propagation |
| 04 | [04_Backpropagation.ipynb](04_Backpropagation.ipynb) | NumPy + PyTorch | Chain rule, autograd preview |
| 05 | [05_Activation_Functions.ipynb](05_Activation_Functions.ipynb) | Both | ReLU, sigmoid, softmax, GELU |
| 06 | [06_Loss_Functions_DL.ipynb](06_Loss_Functions_DL.ipynb) | PyTorch | MSE, BCE, cross-entropy |
| 07 | [07_Optimizers_DL.ipynb](07_Optimizers_DL.ipynb) | PyTorch | SGD, Adam, AdamW, training loop |
| 08 | [08_Weight_Initialization.ipynb](08_Weight_Initialization.ipynb) | PyTorch | Xavier, He/Kaiming |
| 09 | [09_Regularization.ipynb](09_Regularization.ipynb) | PyTorch | Dropout, weight decay |
| 10 | [10_Batch_Normalization.ipynb](10_Batch_Normalization.ipynb) | PyTorch | BatchNorm, train/eval modes |
| 11 | [11_Learning_Rate_Scheduling.ipynb](11_Learning_Rate_Scheduling.ipynb) | PyTorch | Step, cosine schedulers |
| 12 | [12_Residual_Learning.ipynb](12_Residual_Learning.ipynb) | PyTorch | Skip connections, MNIST MLP |

---

## Legacy Code Mapping

| Your Keras Script | Module 05 Equivalent |
|-------------------|---------------------|
| `Day - 28` Dense layers + Adam + BCE | Notebooks 03, 06, 07, 12 |
| `Day - 29` Dropout + BatchNorm + CNN | Notebooks 09, 10 + Module 06 |

### Keras → PyTorch Quick Map

| Keras | PyTorch |
|-------|---------|
| `Dense(n, activation='relu')` | `nn.Linear(in, n)` + `nn.ReLU()` |
| `model.compile(optimizer='adam')` | `torch.optim.Adam(...)` |
| `model.fit(x, y, epochs=5)` | Training loop + DataLoader |
| `model.save_weights('model.h5')` | `torch.save(model.state_dict())` |

---

## Module Deliverables

- [ ] All 12 notebooks completed
- [ ] XOR MLP trained in NumPy (Exercise 2)
- [ ] MNIST MLP trained in PyTorch (Notebook 12)
- [ ] Assignment: Day - 28 reimplemented in PyTorch
- [ ] Quiz ≥16/20 (80%)

---

## Assignment

**Day - 28 Reimplementation in PyTorch** — same 8→12→8→1 architecture, proper train/val/test split, ≥75% accuracy.

See [exercises/README.md](exercises/README.md).

---

## Training Loop Template (memorize)

```python
model.train()
for X_batch, y_batch in train_loader:
    optimizer.zero_grad()
    loss = criterion(model(X_batch), y_batch)
    loss.backward()
    optimizer.step()
```

---

## Debug Checklist

- [ ] Loss decreasing?
- [ ] `param.grad` non-zero after backward?
- [ ] Learning rate appropriate?
- [ ] Data normalized?
- [ ] `model.train()` / `model.eval()` correct?

---

## Resources

- [CHEATSHEET.md](CHEATSHEET.md)
- [quiz/module_05_quiz.md](quiz/module_05_quiz.md)
- [exercises/README.md](exercises/README.md)

---

**Previous:** [04_ML_Paradigms/](../04_ML_Paradigms/)  
**Next:** [06_CNN/](../06_CNN/)
