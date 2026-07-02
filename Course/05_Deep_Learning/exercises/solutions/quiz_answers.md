# Module 05 Quiz — Answer Key

| Q | Answer | Explanation |
|---|--------|-------------|
| 1 | (b) | No activation = linear output |
| 2 | (b) | XOR needs non-linear boundary |
| 3 | (b) | (out_features, in_features) |
| 4 | (b) | Standard BCE+sigmoid gradient |
| 5 | (a) | ReLU grad = 1 for z>0 |
| 6 | (a) | Gradients accumulate without zero_grad |
| 7 | (b) | He init for ReLU |
| 8 | (b) | eval() disables dropout |
| 9 | (b) | Batch statistics during training |
| 10 | (b) | y = F(x) + x |
| 11 | (b) | Linear + ReLU |
| 12 | (c) | Autograd backprop |
| 13 | (b) | Softmax + cross-entropy |
| 14 | (a) | Momentum + RMSProp |
| 15 | (b) | Sigmoid deriv max 0.25, shrinks through layers |
| 16 | (b) | Dropout/BN behave differently |
| 17 | (b) | binary_crossentropy |
| 18 | (b) | L2 penalty on weights |
| 19 | (c) | Multi-class probabilities |
| 20 | (c) | Clear grads before forward pass |

## Training Loop Order

```python
optimizer.zero_grad()  # 1
loss = criterion(model(X), y)  # 2 forward + loss
loss.backward()        # 3
optimizer.step()       # 4
```
