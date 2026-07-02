#!/usr/bin/env python3
"""Generate Module 05 Deep Learning notebooks (12 total)."""
import json
from pathlib import Path

M05 = Path(__file__).resolve().parent / "05_Deep_Learning"


def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": [s]}


def code(s):
    return {"cell_type": "code", "metadata": {}, "source": [s], "outputs": [], "execution_count": None}


def save(name, cells):
    M05.mkdir(parents=True, exist_ok=True)
    (M05 / name).write_text(json.dumps(nb(cells), indent=1))
    print(f"  {name}")


def hdr(num, title, dur, objs, legacy=""):
    leg = f"\n\n**Legacy script:** `{legacy}`" if legacy else ""
    return md(
        f"# Notebook {num}: {title}\n\n**Module:** 05 Deep Learning  \n**Duration:** ~{dur}{leg}\n\n---\n\n## Learning Objectives\n\n{objs}"
    )


def footer(summary, nxt=None):
    nxt = f"\n\n**Next:** [{nxt}]({nxt})" if nxt else ""
    return md(f"---\n\n## Summary\n\n{summary}{nxt}")


SETUP_NP = code(
    "import numpy as np\nimport matplotlib.pyplot as plt\nfrom pathlib import Path\n\nREPO = Path('../../').resolve()\nplt.rcParams['figure.figsize'] = (8, 5)\nrng = np.random.default_rng(42)"
)

SETUP_TORCH = code(
    "import numpy as np\nimport matplotlib.pyplot as plt\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch.utils.data import DataLoader, TensorDataset\n\nplt.rcParams['figure.figsize'] = (8, 5)\ntorch.manual_seed(42)\nrng = np.random.default_rng(42)\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nprint(f'Device: {device}')"
)

NOTEBOOKS = []


def register(name, cells):
    NOTEBOOKS.append((name, cells))


# ── 01 Artificial Neuron ────────────────────────────────────────────────

register("01_Artificial_Neuron.ipynb", [
    hdr("01", "The Artificial Neuron", "2 hrs",
        "1. Define the biological vs artificial neuron\n2. Implement weighted sum + activation\n3. Understand weights, bias, and activation role\n4. Connect to linear regression (Module 03)"),
    md("## 1. Intuition\n\nA **biological neuron** receives signals through dendrites, processes them in the cell body, and fires through the axon if the signal exceeds a threshold.\n\nAn **artificial neuron** does the same mathematically:\n\n$$z = w_1 x_1 + w_2 x_2 + \\cdots + w_n x_n + b = \\mathbf{w}^T \\mathbf{x} + b$$\n$$a = \\sigma(z)$$\n\n- $\\mathbf{x}$ = inputs (features)\n- $\\mathbf{w}$ = weights (learned parameters)\n- $b$ = bias (shift activation threshold)\n- $\\sigma$ = activation function (non-linearity)"),
    md("## 2. Single Neuron = Linear Regression + Activation\n\nWithout activation ($\\sigma$ = identity): $\\hat{y} = w^T x + b$ → **linear regression**.\n\nWith sigmoid: $\\hat{y} = \\sigma(w^T x + b)$ → **logistic regression**.\n\nWith ReLU: $\\hat{y} = \\max(0, w^T x + b)$ → building block of deep networks."),
    SETUP_NP,
    code("def neuron(x, w, b, activation='relu'):\n    z = np.dot(w, x) + b\n    if activation == 'relu':\n        return max(0, z), z\n    elif activation == 'sigmoid':\n        return 1 / (1 + np.exp(-z)), z\n    return z, z\n\nx = np.array([2.0, 3.0, 1.0])\nw = np.array([0.5, -0.3, 0.8])\nb = 0.1\n\na, z = neuron(x, w, b, 'relu')\nprint(f'Input: {x}')\nprint(f'Weights: {w}, Bias: {b}')\nprint(f'z = w·x + b = {z:.4f}')\nprint(f'a = ReLU(z) = {a:.4f}')"),
    code("# Visualize neuron response\nw1_range = np.linspace(-2, 2, 100)\noutputs = [neuron(np.array([w1, 1.0]), np.array([1.0, 0.5]), 0, 'relu')[0] for w1 in w1_range]\nplt.plot(w1_range, outputs)\nplt.xlabel('x1'); plt.ylabel('Output a')\nplt.title('Single Neuron Response (ReLU)'); plt.grid(True, alpha=0.3); plt.show()"),
    md("## 3. Your Day - 28 Keras Model\n\n```python\nmodel.add(Dense(12, input_dim=8, activation='relu'))  # 12 neurons, each does w·x+b → ReLU\n```\n\nEach `Dense` layer = matrix of neurons operating in parallel."),
    md("## Exercise\n\nImplement a neuron with **step activation** (0 if z<0, else 1). Test on AND gate: (0,0)→0, (0,1)→0, (1,0)→0, (1,1)→1."),
    code("# YOUR CODE HERE\n"),
    footer("The artificial neuron is the atomic unit of all neural networks.", "02_Perceptron.ipynb"),
])

# ── 02 Perceptron ───────────────────────────────────────────────────────

register("02_Perceptron.ipynb", [
    hdr("02", "The Perceptron", "2 hrs",
        "1. Understand the perceptron learning rule\n2. Implement perceptron training\n3. Prove XOR requires hidden layers\n4. See why depth matters"),
    md("## 1. Perceptron Learning Rule (Rosenblatt, 1958)\n\nFor misclassified sample $(x, y)$:\n\n$$w_j \\leftarrow w_j + \\eta (y - \\hat{y}) x_j$$\n$$b \\leftarrow b + \\eta (y - \\hat{y})$$\n\nOnly updates when $\\hat{y} \\neq y$."),
    SETUP_NP,
    code("class Perceptron:\n    def __init__(self, n_features, lr=0.1):\n        self.w = np.zeros(n_features)\n        self.b = 0.0\n        self.lr = lr\n\n    def predict(self, x):\n        return 1 if np.dot(self.w, x) + self.b >= 0 else 0\n\n    def fit(self, X, y, epochs=100):\n        for _ in range(epochs):\n            for xi, yi in zip(X, y):\n                y_hat = self.predict(xi)\n                update = self.lr * (yi - y_hat)\n                self.w += update * xi\n                self.b += update\n\n# AND gate\nX_and = np.array([[0,0],[0,1],[1,0],[1,1]])\ny_and = np.array([0,0,0,1])\n\np = Perceptron(2)\np.fit(X_and, y_and)\nprint('AND predictions:', [p.predict(x) for x in X_and])"),
    code("# XOR fails with single perceptron\nX_xor = np.array([[0,0],[0,1],[1,0],[1,1]])\ny_xor = np.array([0,1,1,0])\n\np_xor = Perceptron(2, lr=0.1)\np_xor.fit(X_xor, y_xor, epochs=100)\nprint('XOR predictions (single layer):', [p_xor.predict(x) for x in X_xor])\nprint('XOR is NOT linearly separable → need hidden layer (MLP)')"),
    code("# Visualize decision boundary\nxx, yy = np.meshgrid(np.linspace(-0.5,1.5,100), np.linspace(-0.5,1.5,100))\nZ = np.array([p.predict(np.array([x,y])) for x,y in zip(xx.ravel(), yy.ravel())]).reshape(xx.shape)\nplt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')\nplt.scatter(X_and[:,0], X_and[:,1], c=y_and, s=100, edgecolors='k')\nplt.title('Perceptron Decision Boundary — AND gate'); plt.show()"),
    footer("Single perceptron = linear classifier. XOR proves we need multiple layers.", "03_Multi_Layer_Perceptron.ipynb"),
])

# ── 03 MLP Forward Propagation ──────────────────────────────────────────

register("03_Multi_Layer_Perceptron.ipynb", [
    hdr("03", "Multi-Layer Perceptron — Forward Propagation", "2.5 hrs",
        "1. Build MLP architecture with matrix operations\n2. Implement forward pass layer by layer\n3. Understand weight matrix dimensions\n4. Map to Day - 28 architecture",
        "Day - 28 Introduction to DeepLearning.py"),
    md("## 1. Architecture\n\n**Input layer** → **Hidden layer(s)** → **Output layer**\n\nFor layer $l$:\n$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$$\n$$A^{[l]} = \\sigma(Z^{[l]})$$\n\nWhere $A^{[0]} = X$ (input)."),
    md("## 2. Shape Convention\n\n$W^{[l]}$ shape: $(n^{[l]}, n^{[l-1]})$ — rows = neurons in layer l, cols = inputs from layer l-1\n\nDay - 28: `Dense(12, input_dim=8)` → $W$ shape $(12, 8)$"),
    SETUP_NP,
    code("def relu(z): return np.maximum(0, z)\ndef sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))\n\ndef forward_pass(X, weights, biases, activation='relu'):\n    \"\"\"X: (n_samples, n_features), weights: list of (out, in) matrices\"\"\"\n    A = X\n    activations = [A]\n    zs = []\n    act_fn = relu if activation == 'relu' else sigmoid\n    for W, b in zip(weights, biases):\n        Z = A @ W.T + b  # (n, in) @ (in, out) + (out,) → (n, out)\n        A = act_fn(Z)\n        zs.append(Z)\n        activations.append(A)\n    return activations, zs\n\n# Day - 28 architecture: 8 → 12 → 8 → 1\nn_samples = 5\nX = rng.normal(0, 1, (n_samples, 8))\nW1 = rng.normal(0, 0.1, (12, 8)); b1 = np.zeros(12)\nW2 = rng.normal(0, 0.1, (8, 12));  b2 = np.zeros(8)\nW3 = rng.normal(0, 0.1, (1, 8));   b3 = np.zeros(1)\n\nacts, zs = forward_pass(X, [W1, W2, W3], [b1, b2, b3], activation='relu')\nfor i, a in enumerate(acts):\n    print(f'Layer {i} activation shape: {a.shape}')"),
    md("## Exercise\n\nImplement forward pass for XOR with architecture 2 → 4 → 1. Use ReLU hidden, sigmoid output."),
    code("# YOUR CODE HERE\n"),
    footer("Forward propagation = chain of matrix multiplies + activations.", "04_Backpropagation.ipynb"),
])

# ── 04 Backpropagation ──────────────────────────────────────────────────

register("04_Backpropagation.ipynb", [
    hdr("04", "Backpropagation", "3 hrs",
        "1. Derive backpropagation using chain rule\n2. Implement backward pass in NumPy\n3. Compute gradients for each layer\n4. Understand why autograd exists"),
    md("## 1. Derivation (Binary Cross-Entropy + Sigmoid Output)\n\n**Loss:** $L = -[y\\log\\hat{y} + (1-y)\\log(1-\\hat{y})]$\n\n**Output layer gradient:**\n$$\\frac{\\partial L}{\\partial Z^{[L]}} = \\hat{Y} - Y = A^{[L]} - Y$$\n\n**Hidden layer (chain rule):**\n$$\\frac{\\partial L}{\\partial Z^{[l]}} = (W^{[l+1]T} \\frac{\\partial L}{\\partial Z^{[l+1]}}) \\odot \\sigma'(Z^{[l]})$$\n\n**Weight gradient:**\n$$\\frac{\\partial L}{\\partial W^{[l]}} = \\frac{1}{m} \\frac{\\partial L}{\\partial Z^{[l]}} A^{[l-1]T}$$"),
    md("## 2. ReLU derivative\n\n$$\\text{ReLU}'(z) = \\begin{cases} 1 & z > 0 \\\\ 0 & z \\leq 0 \\end{cases}$$"),
    SETUP_NP,
    code("def relu_derivative(z): return (z > 0).astype(float)\n\ndef backward_pass(y_true, activations, zs, weights):\n    \"\"\"Backprop for BCE + sigmoid output or MSE + linear output.\"\"\"\n    m = y_true.shape[0]\n    grads_W, grads_b = [], []\n    # Output layer (sigmoid + BCE → dZ = A - Y)\n    dZ = activations[-1] - y_true.reshape(-1, 1)\n    for l in reversed(range(len(weights))):\n        grads_W.insert(0, (dZ.T @ activations[l]) / m)\n        grads_b.insert(0, dZ.mean(axis=0))\n        if l > 0:\n            dA = dZ @ weights[l]\n            dZ = dA * relu_derivative(zs[l-1])\n    return grads_W, grads_b\n\nprint('Backpropagation computes gradients layer by layer via chain rule.')"),
    md("## 3. PyTorch Autograd Preview\n\nPyTorch computes all of this automatically:\n```python\nloss.backward()  # runs backprop\nw.grad           # gradients appear here\n```\n\nYou must understand the math to debug when training fails."),
    SETUP_TORCH,
    code("x = torch.tensor([[1.0, 2.0]], requires_grad=True)\nw = torch.tensor([[0.5, -0.3]], requires_grad=True)\nb = torch.tensor([0.1], requires_grad=True)\n\ny = x @ w.T + b\nloss = y.sum()\nloss.backward()\n\nprint(f'dL/dw = {w.grad}')\nprint(f'dL/db = {b.grad}')"),
    md("## Exercise\n\nFor a 2-layer network, compute $\\partial L / \\partial W^{[1]}$ by hand for one sample. Verify with PyTorch autograd."),
    code("# YOUR CODE HERE\n"),
    footer("Backpropagation = chain rule applied systematically through the network.", "05_Activation_Functions.ipynb"),
])

# ── 05 Activation Functions ─────────────────────────────────────────────

register("05_Activation_Functions.ipynb", [
    hdr("05", "Activation Functions", "2 hrs",
        "1. Know all major activation functions and derivatives\n2. Understand vanishing gradient problem\n3. Choose activation for each layer type"),
    md("## Activation Functions\n\n| Function | Formula | Range | Use |\n|----------|---------|-------|-----|\n| Sigmoid | $1/(1+e^{-z})$ | (0,1) | Binary output, old hidden |\n| Tanh | $(e^z-e^{-z})/(e^z+e^{-z})$ | (-1,1) | Hidden (zero-centered) |\n| ReLU | $\\max(0,z)$ | [0,∞) | Default hidden layer |\n| Leaky ReLU | $\\max(0.01z, z)$ | (-∞,∞) | Avoid dead neurons |\n| GELU | $z\\Phi(z)$ | (-∞,∞) | Transformers, modern CNNs |\n| Softmax | $e^{z_i}/\\sum e^{z_j}$ | (0,1), sum=1 | Multi-class output |"),
    SETUP_NP,
    code("z = np.linspace(-5, 5, 200)\n\nactivations = {\n    'Sigmoid': 1 / (1 + np.exp(-z)),\n    'Tanh': np.tanh(z),\n    'ReLU': np.maximum(0, z),\n    'Leaky ReLU': np.where(z > 0, z, 0.01 * z),\n}\n\nfig, axes = plt.subplots(1, 2, figsize=(12, 4))\nfor name, a in activations.items():\n    axes[0].plot(z, a, label=name)\naxes[0].set_title('Activation Functions'); axes[0].legend(); axes[0].grid(True, alpha=0.3)\n\n# Derivatives\nsigmoid = activations['Sigmoid']\naxes[1].plot(z, sigmoid * (1 - sigmoid), label=\"Sigmoid'\")\naxes[1].plot(z, 1 - np.tanh(z)**2, label=\"Tanh'\")\naxes[1].plot(z, (z > 0).astype(float), label=\"ReLU'\")\naxes[1].set_title('Derivatives'); axes[1].legend(); axes[1].grid(True, alpha=0.3)\nplt.tight_layout(); plt.show()"),
    md("## Vanishing Gradient\n\nSigmoid derivative max = 0.25. Through L layers: $0.25^L \\rightarrow 0$.\n\n**ReLU** derivative = 1 for z > 0 → gradients flow without shrinking."),
    SETUP_TORCH,
    code("print('PyTorch activations:')\nprint(f'  F.relu:    {F.relu(torch.tensor(-1.0))}')\nprint(f'  F.gelu:    {F.gelu(torch.tensor(1.0)):.4f}')\nprint(f'  F.softmax: {F.softmax(torch.tensor([1.,2.,3.]), dim=0)}')"),
    footer("ReLU is default for hidden layers; sigmoid/softmax for outputs.", "06_Loss_Functions_DL.ipynb"),
])

# ── 06 Loss Functions ───────────────────────────────────────────────────

register("06_Loss_Functions_DL.ipynb", [
    hdr("06", "Loss Functions for Deep Learning", "2 hrs",
        "1. Implement MSE and cross-entropy loss in NumPy and PyTorch\n2. Derive gradients for backprop\n3. Match Day - 28 binary cross-entropy",
        "Day - 28 Introduction to DeepLearning.py"),
    md("## Loss Functions\n\n**MSE (regression):** $L = \\frac{1}{n}\\sum(y - \\hat{y})^2$, $\\frac{\\partial L}{\\partial \\hat{y}} = \\frac{2}{n}(\\hat{y} - y)$\n\n**Binary CE:** $L = -[y\\log\\hat{p} + (1-y)\\log(1-\\hat{p})]$, with sigmoid: $\\frac{\\partial L}{\\partial z} = \\hat{p} - y$\n\n**Categorical CE:** $L = -\\sum_k y_k \\log \\hat{p}_k$, with softmax: $\\frac{\\partial L}{\\partial z} = \\hat{p} - y$"),
    SETUP_TORCH,
    code("# PyTorch losses\ny_true = torch.tensor([1.0, 0.0, 1.0])\ny_pred_logits = torch.tensor([2.0, -1.0, 0.5])\n\nbce = F.binary_cross_entropy_with_logits(y_pred_logits, y_true)\nce = F.cross_entropy(y_pred_logits.unsqueeze(0).repeat(3,1), torch.tensor([0,1,0]))\nmse = F.mse_loss(torch.tensor([1.,2.,3.]), torch.tensor([1.1, 1.9, 3.2]))\n\nprint(f'BCE with logits: {bce:.4f}')\nprint(f'MSE: {mse:.4f}')\nprint('\\nDay - 28 uses: loss=\"binary_crossentropy\" → BCE + sigmoid')"),
    footer("Choose loss to match task: MSE for regression, CE for classification.", "07_Optimizers_DL.ipynb"),
])

# ── 07 Optimizers ───────────────────────────────────────────────────────

register("07_Optimizers_DL.ipynb", [
    hdr("07", "Optimizers in PyTorch", "2.5 hrs",
        "1. Implement SGD and Adam update rules\n2. Use torch.optim in training loops\n3. Connect to Module 02 derivations\n4. Match Day - 28 Adam optimizer",
        "Day - 28 Introduction to DeepLearning.py"),
    md("## Update Rules (Module 02 recap)\n\n**SGD:** $\\theta \\leftarrow \\theta - \\eta \\nabla L$\n\n**Adam:** Uses running averages $m_t$ (momentum) and $v_t$ (RMSprop) — see Module 02 Notebook 20."),
    SETUP_TORCH,
    code("model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))\n\n# Different optimizers\noptimizers = {\n    'SGD': torch.optim.SGD(model.parameters(), lr=0.01),\n    'SGD+Momentum': torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),\n    'Adam': torch.optim.Adam(model.parameters(), lr=0.001),\n    'AdamW': torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01),\n}\n\nfor name, opt in optimizers.items():\n    print(f'{name}: {opt}')\n\nprint('\\nDay - 28: optimizer=\"adam\"')\nprint('water-bodies-detection: AdamW with weight decay')"),
    code("# Training step pattern (memorize this)\ndef train_step(model, optimizer, X_batch, y_batch):\n    optimizer.zero_grad()       # 1. Clear old gradients\n    y_pred = model(X_batch)     # 2. Forward pass\n    loss = F.mse_loss(y_pred, y_batch)  # 3. Compute loss\n    loss.backward()             # 4. Backprop (autograd)\n    optimizer.step()            # 5. Update weights\n    return loss.item()\n\nX_b = torch.randn(32, 10)\ny_b = torch.randn(32, 1)\nmodel = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))\nopt = torch.optim.Adam(model.parameters(), lr=0.01)\nloss = train_step(model, opt, X_b, y_b)\nprint(f'Loss after one step: {loss:.4f}')"),
    footer("Adam/AdamW is default. Understand SGD for debugging and papers.", "08_Weight_Initialization.ipynb"),
])

# ── 08 Weight Initialization ────────────────────────────────────────────

register("08_Weight_Initialization.ipynb", [
    hdr("08", "Weight Initialization", "2 hrs",
        "1. Understand why initialization matters\n2. Apply Xavier and He initialization\n3. See effect on training convergence"),
    md("## Why Initialization Matters\n\n- **All zeros:** all neurons learn same thing (symmetry problem)\n- **Too large:** activations explode, gradients explode\n- **Too small:** activations vanish, gradients vanish\n\n**Xavier (Glorot):** $W \\sim \\mathcal{U}\\left[-\\sqrt{6/(n_{in}+n_{out})}, \\sqrt{6/(n_{in}+n_{out})}\\right]$ — for sigmoid/tanh\n\n**He (Kaiming):** $W \\sim \\mathcal{N}(0, 2/n_{in})$ — for ReLU"),
    SETUP_TORCH,
    code("def xavier_init(fan_in, fan_out):\n    limit = np.sqrt(6 / (fan_in + fan_out))\n    return torch.empty(fan_out, fan_in).uniform_(-limit, limit)\n\ndef he_init(fan_in, fan_out):\n    return torch.randn(fan_out, fan_in) * np.sqrt(2 / fan_in)\n\nfor name, W in [('Xavier', xavier_init(784, 256)), ('He', he_init(784, 256)), ('Default', torch.randn(256, 784)*0.01)]:\n    x = torch.randn(256, 784)\n    out = F.relu(x @ W.T)\n    print(f'{name}: output mean={out.mean():.4f}, std={out.std():.4f}')"),
    code("# PyTorch built-in\nlayer = nn.Linear(784, 256)\nnn.init.kaiming_normal_(layer.weight, nonlinearity='relu')\nprint(f'Kaiming init std: {layer.weight.std():.4f}')"),
    footer("Use He/Kaiming for ReLU networks; Xavier for sigmoid/tanh.", "09_Regularization.ipynb"),
])

# ── 09 Regularization ───────────────────────────────────────────────────

register("09_Regularization.ipynb", [
    hdr("09", "Regularization — L1, L2, Dropout", "2.5 hrs",
        "1. Implement L2 weight decay in PyTorch\n2. Apply dropout during training\n3. Understand early stopping\n4. Match Day - 29 dropout layers",
        "Day - 29 covid-19 Detction using Deep Learning.py"),
    md("## L2 Regularization (Weight Decay)\n\nAdd penalty: $L_{total} = L_{data} + \\lambda \\sum w_i^2$\n\nIn PyTorch: `weight_decay=0.01` in AdamW.\n\n## Dropout (Srivastava et al., 2014)\n\nDuring training: randomly zero each neuron with probability $p$.\n\nScale remaining activations by $1/(1-p)$ to preserve expected value.\n\n**Inference:** dropout OFF (use all neurons)."),
    SETUP_TORCH,
    code("class MLPWithDropout(nn.Module):\n    def __init__(self, input_dim, hidden, output_dim, dropout=0.5):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Linear(input_dim, hidden),\n            nn.ReLU(),\n            nn.Dropout(dropout),\n            nn.Linear(hidden, hidden),\n            nn.ReLU(),\n            nn.Dropout(dropout),\n            nn.Linear(hidden, output_dim),\n        )\n    def forward(self, x):\n        return self.net(x)\n\nmodel = MLPWithDropout(784, 256, 10, dropout=0.5)\nmodel.train();  out_train = model(torch.randn(32, 784))\nmodel.eval();   out_eval  = model(torch.randn(32, 784))\nprint(f'Training mode (dropout ON):  output std={out_train.std():.4f}')\nprint(f'Eval mode (dropout OFF):     output std={out_eval.std():.4f}')"),
    md("## Day - 29 uses Dropout + BatchNormalization — same pattern for COVID CNN."),
    footer("Dropout and weight decay prevent overfitting in deep networks.", "10_Batch_Normalization.ipynb"),
])

# ── 10 Batch Normalization ──────────────────────────────────────────────

register("10_Batch_Normalization.ipynb", [
    hdr("10", "Batch Normalization", "2.5 hrs",
        "1. Derive batch norm forward pass\n2. Understand train vs eval mode\n3. Use nn.BatchNorm1d/2d in PyTorch\n4. Connect to Day - 29 BatchNormalization",
        "Day - 29 covid-19 Detction using Deep Learning.py"),
    md("## Batch Normalization (Ioffe & Szegedy, 2015)\n\nNormalize each mini-batch:\n\n$$\\hat{x}_i = \\frac{x_i - \\mu_B}{\\sqrt{\\sigma_B^2 + \\epsilon}}$$\n$$y_i = \\gamma \\hat{x}_i + \\beta$$\n\nLearnable scale $\\gamma$ and shift $\\beta$.\n\n**Benefits:** Faster training, higher learning rates, regularization effect."),
    SETUP_TORCH,
    code("bn = nn.BatchNorm1d(64)\nx = torch.randn(32, 64) * 5 + 10  # shifted, scaled data\n\nbn.train()\ny_train = bn(x)\nprint(f'Before BN: mean={x.mean():.2f}, std={x.std():.2f}')\nprint(f'After BN (train): mean={y_train.mean():.4f}, std={y_train.std():.4f}')\n\nbn.eval()\ny_eval = bn(x)\nprint(f'After BN (eval):  mean={y_eval.mean():.4f}, std={y_eval.std():.4f}')"),
    md("## Critical: model.train() vs model.eval()\n\n- **train():** dropout ON, batch norm uses batch statistics\n- **eval():** dropout OFF, batch norm uses running statistics\n\nForgetting `model.eval()` before inference is a common bug."),
    footer("BatchNorm stabilizes training. Always switch modes correctly.", "11_Learning_Rate_Scheduling.ipynb"),
])

# ── 11 Learning Rate Scheduling ───────────────────────────────────────────

register("11_Learning_Rate_Scheduling.ipynb", [
    hdr("11", "Learning Rate Scheduling", "2 hrs",
        "1. Implement step, exponential, and cosine schedulers\n2. Use torch.optim.lr_scheduler\n3. Understand warmup"),
    md("## Schedulers\n\n| Scheduler | Formula | Use |\n|-----------|---------|-----|\n| StepLR | $\\eta \\leftarrow \\eta \\cdot \\gamma$ every step_size epochs | Simple decay |\n| ExponentialLR | $\\eta \\leftarrow \\eta \\cdot \\gamma$ every epoch | Smooth decay |\n| CosineAnnealingLR | $\\eta_t = \\eta_{min} + \\frac{1}{2}(\\eta_{max}-\\eta_{min})(1+\\cos(t\\pi/T))$ | Modern default |\n| Warmup | Linear increase for first W steps | Transformers, large batch |"),
    SETUP_TORCH,
    code("model = nn.Linear(10, 1)\noptimizer = torch.optim.SGD(model.parameters(), lr=0.1)\n\nschedulers = {\n    'StepLR': torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5),\n    'CosineAnnealing': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50),\n    'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),\n}\n\nfor name, scheduler in schedulers.items():\n    opt = torch.optim.SGD(model.parameters(), lr=0.1)\n    if name == 'StepLR': s = torch.optim.lr_scheduler.StepLR(opt, 10, 0.5)\n    elif name == 'CosineAnnealing': s = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 50)\n    else: s = torch.optim.lr_scheduler.ExponentialLR(opt, 0.95)\n    lrs = []\n    for _ in range(50):\n        lrs.append(opt.param_groups[0]['lr'])\n        s.step()\n    plt.plot(lrs, label=name)\n\nplt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.legend(); plt.title('LR Schedulers'); plt.show()"),
    footer("Cosine annealing + warmup is standard for modern training.", "12_Residual_Learning.ipynb"),
])

# ── 12 Residual Learning + Full MLP Project ─────────────────────────────

register("12_Residual_Learning.ipynb", [
    hdr("12", "Residual Learning & MNIST MLP Project", "3 hrs",
        "1. Understand skip connections and gradient flow\n2. Build complete PyTorch MLP on MNIST\n3. Reimplement Day - 28 architecture in PyTorch\n4. Complete Module 05 capstone"),
    md("## 1. Residual Connections (He et al., 2016)\n\n$$y = F(x) + x$$\n\nInstead of learning $H(x)$, learn residual $F(x) = H(x) - x$.\n\n**Why:** Gradients can flow directly through skip connection → train very deep networks.\n\nPreview of Module 06 (ResNet) and water-bodies UNet++ skip connections."),
    SETUP_TORCH,
    code("class ResidualBlock(nn.Module):\n    def __init__(self, dim):\n        super().__init__()\n        self.fc = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))\n    def forward(self, x):\n        return F.relu(self.fc(x) + x)  # skip connection\n\nblock = ResidualBlock(64)\nx = torch.randn(16, 64)\nprint(f'Input shape: {x.shape}, Output shape: {block(x).shape}')"),
    md("## 2. Complete MNIST MLP (PyTorch)\n\nFull training loop — the template for all future modules."),
    code("# Load MNIST\nfrom sklearn.datasets import fetch_openml\nmnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')\nX, y = mnist.data.astype(np.float32) / 255.0, mnist.target.astype(int)\n\n# Subset for speed\nX, y = X[:10000], y[:10000]\nX_train, X_test = X[:8000], X[8000:]\ny_train, y_test = y[:8000], y[8000:]\n\nX_train_t = torch.tensor(X_train)\ny_train_t = torch.tensor(y_train)\nX_test_t = torch.tensor(X_test)\ny_test_t = torch.tensor(y_test)\n\ntrain_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)"),
    code("class MNISTMLP(nn.Module):\n    \"\"\"Day - 28 equivalent in PyTorch: 784 → 256 → 128 → 10\"\"\"\n    def __init__(self):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),\n            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),\n            nn.Linear(128, 10),\n        )\n    def forward(self, x):\n        return self.net(x)\n\nmodel = MNISTMLP().to(device)\noptimizer = torch.optim.Adam(model.parameters(), lr=0.001)\n\nfor epoch in range(5):\n    model.train()\n    total_loss = 0\n    for X_b, y_b in train_loader:\n        X_b, y_b = X_b.to(device), y_b.to(device)\n        optimizer.zero_grad()\n        loss = F.cross_entropy(model(X_b), y_b)\n        loss.backward()\n        optimizer.step()\n        total_loss += loss.item()\n    model.eval()\n    with torch.no_grad():\n        acc = (model(X_test_t.to(device)).argmax(1) == y_test_t.to(device)).float().mean()\n    print(f'Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, test_acc={acc:.4f}')"),
    md("## 3. Day - 28 Keras → PyTorch Mapping\n\n| Keras | PyTorch |\n|-------|--------|\n| `Sequential()` | `nn.Sequential()` or `nn.Module` |\n| `Dense(12, activation='relu')` | `nn.Linear(8, 12)` + `nn.ReLU()` |\n| `model.compile(loss='binary_crossentropy')` | `F.binary_cross_entropy_with_logits` |\n| `model.fit(x, y, epochs=5)` | Training loop with DataLoader |\n| `model.save_weights('model.h5')` | `torch.save(model.state_dict(), 'model.pt')` |"),
    md("## Module 05 Complete\n\nYou can now build, train, and debug neural networks in PyTorch.\n\n**Next:** Module 06 CNN — convolutions, LeNet, ResNet."),
    md("## Exercise\n\nAdd a residual block to the MNIST MLP hidden layer. Does accuracy improve?"),
    code("# YOUR CODE HERE\n"),
    footer("Residual learning enables deep networks. MNIST MLP = your first complete PyTorch training pipeline.", None),
])


def main():
    print("Building Module 05 notebooks...")
    for name, cells in NOTEBOOKS:
        save(name, cells)
    print(f"Done: {len(NOTEBOOKS)} notebooks created.")


if __name__ == "__main__":
    main()
