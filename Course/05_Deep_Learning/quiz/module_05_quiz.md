# Module 05 Quiz

**Passing score:** 16/20 (80%)

---

**Q1.** A single neuron without activation is equivalent to:
- (a) Logistic regression
- (b) Linear regression
- (c) KNN
- (d) Decision tree

**Q2.** XOR cannot be solved by a single perceptron because:
- (a) XOR has too many features
- (b) XOR is not linearly separable
- (c) Perceptron is too slow
- (d) XOR needs regression

**Q3.** In forward propagation, $W^{[l]}$ has shape:
- (a) $(n^{[l-1]}, n^{[l]})$
- (b) $(n^{[l]}, n^{[l-1]})$
- (c) $(n^{[l]}, n^{[l]})$
- (d) $(1, n^{[l]})$

**Q4.** For sigmoid output + BCE loss, $\partial L / \partial z$ at output layer equals:
- (a) $y - \hat{y}$
- (b) $\hat{y} - y$
- (c) $y^2 - \hat{y}^2$
- (d) $1 - \hat{y}$

**Q5.** ReLU is preferred over sigmoid for hidden layers because:
- (a) ReLU avoids vanishing gradients
- (b) ReLU outputs probabilities
- (c) ReLU is always differentiable everywhere
- (d) Sigmoid is faster to compute

**Q6.** `optimizer.zero_grad()` is needed because:
- (a) PyTorch accumulates gradients by default
- (b) It resets the model weights
- (c) It clears the loss
- (d) It enables dropout

**Q7.** He/Kaiming initialization is designed for:
- (a) Sigmoid activations
- (b) ReLU activations
- (c) Softmax output
- (d) Linear output only

**Q8.** Dropout during inference should be:
- (a) ON (same as training)
- (b) OFF (model.eval())
- (c) Set to 1.0
- (d) Random

**Q9.** Batch normalization during training uses:
- (a) Global dataset statistics
- (b) Mini-batch mean and variance
- (c) Fixed precomputed values only
- (d) No normalization

**Q10.** A residual connection computes:
- (a) $y = F(x) - x$
- (b) $y = F(x) + x$
- (c) $y = F(x) \times x$
- (d) $y = F(x)$ only

**Q11.** PyTorch equivalent of Keras `Dense(10, activation='relu')`:
- (a) `nn.Conv2d(10, relu)`
- (b) `nn.Linear(in, 10)` + `nn.ReLU()`
- (c) `nn.Dropout(10)`
- (d) `F.softmax(10)`

**Q12.** `loss.backward()` triggers:
- (a) Weight update
- (b) Forward pass
- (c) Gradient computation via autograd
- (d) Data loading

**Q13.** Cross-entropy loss for multi-class uses:
- (a) Sigmoid + BCE
- (b) Softmax + NLL
- (c) MSE
- (d) Hinge loss

**Q14.** Adam optimizer combines:
- (a) Momentum and RMSProp
- (b) SGD and BatchNorm
- (c) Dropout and L2
- (d) Xavier and He init

**Q15.** Vanishing gradient is mainly caused by:
- (a) ReLU in deep networks
- (b) Sigmoid/tanh derivatives < 1 multiplied through many layers
- (c) Large learning rate
- (d) Too much training data

**Q16.** `model.train()` vs `model.eval()` affects:
- (a) Only the loss function
- (b) Dropout and BatchNorm behavior
- (c) Optimizer choice
- (d) Data augmentation only

**Q17.** Day - 28 uses which loss function?
- (a) MSE
- (b) Binary cross-entropy
- (c) Hinge
- (d) Dice

**Q18.** Weight decay in AdamW is equivalent to:
- (a) Dropout
- (b) L2 regularization
- (c) Batch normalization
- (d) Data augmentation

**Q19.** Softmax output layer is used for:
- (a) Regression
- (b) Binary classification only
- (c) Multi-class classification
- (d) Clustering

**Q20.** The first step in a PyTorch training iteration is:
- (a) optimizer.step()
- (b) loss.backward()
- (c) optimizer.zero_grad()
- (d) model.eval()

---

See [solutions/quiz_answers.md](solutions/quiz_answers.md) after submitting.
