# Module 05 Cheat Sheet — Deep Learning

## Artificial Neuron

$$z = \mathbf{w}^T \mathbf{x} + b, \quad a = \sigma(z)$$

## Forward Propagation

$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}, \quad A^{[l]} = \sigma(Z^{[l]})$$

## Backpropagation (key gradients)

| Layer | Gradient |
|-------|----------|
| Output (sigmoid + BCE) | $\partial L / \partial Z^{[L]} = \hat{Y} - Y$ |
| Hidden | $\partial L / \partial Z^{[l]} = (W^{[l+1]T} \delta^{[l+1]}) \odot \sigma'(Z^{[l]})$ |
| Weights | $\partial L / \partial W^{[l]} = \frac{1}{m} \delta^{[l]} A^{[l-1]T}$ |

## Activation Functions

| Name | Formula | Derivative | Use |
|------|---------|------------|-----|
| ReLU | $\max(0,z)$ | $1$ if $z>0$ else $0$ | Hidden (default) |
| Sigmoid | $1/(1+e^{-z})$ | $\sigma(1-\sigma)$ | Binary output |
| Tanh | $(e^z-e^{-z})/(e^z+e^{-z})$ | $1-\tanh^2$ | Hidden (legacy) |
| Softmax | $e^{z_i}/\sum e^{z_j}$ | Jacobian matrix | Multi-class output |
| GELU | $z\Phi(z)$ | — | Transformers |

## Loss Functions (PyTorch)

```python
F.mse_loss(y_pred, y_true)                          # Regression
F.binary_cross_entropy_with_logits(logits, targets) # Binary classification
F.cross_entropy(logits, class_indices)              # Multi-class
```

## Weight Initialization

| Method | Formula | For |
|--------|---------|-----|
| Xavier | $\mathcal{U}[-\sqrt{6/(n_{in}+n_{out})}, \sqrt{6/(n_{in}+n_{out})}]$ | Sigmoid/Tanh |
| He/Kaiming | $\mathcal{N}(0, 2/n_{in})$ | ReLU |

## Training Loop Template (memorize)

```python
model.train()
for X_batch, y_batch in train_loader:
    optimizer.zero_grad()
    y_pred = model(X_batch)
    loss = criterion(y_pred, y_batch)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    y_pred = model(X_test)
```

## Regularization

- **L2 / weight decay:** `AdamW(..., weight_decay=0.01)`
- **Dropout:** `nn.Dropout(0.5)` — ON in train, OFF in eval
- **Early stopping:** stop when val loss stops improving

## Batch Normalization

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

Always: `model.train()` for training, `model.eval()` for inference.

## Keras → PyTorch Mapping

| Keras | PyTorch |
|-------|---------|
| `Dense(n, activation='relu')` | `nn.Linear(in, n)` + `nn.ReLU()` |
| `Dropout(0.5)` | `nn.Dropout(0.5)` |
| `BatchNormalization()` | `nn.BatchNorm1d/2d` |
| `model.compile(optimizer='adam')` | `torch.optim.Adam(...)` |
| `model.fit()` | Manual training loop |
| `model.predict()` | `model.eval(); model(x)` |

## Residual Connection

$$y = F(x) + x \quad \text{(skip connection)}$$

## Debug Checklist

- [ ] Loss decreasing?
- [ ] `param.grad` non-zero after `loss.backward()`?
- [ ] Learning rate appropriate?
- [ ] Data normalized (0-1 or standardized)?
- [ ] Shapes printed at each layer?
- [ ] `model.train()` / `model.eval()` correct?
- [ ] Random seeds set?

## Your Legacy Scripts

| Script | Module 05 Equivalent |
|--------|---------------------|
| Day - 28 (Keras MLP) | Notebook 12 MNIST MLP |
| Day - 29 (Keras CNN+Dropout+BN) | Notebooks 09, 10 + Module 06 |
