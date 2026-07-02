#!/usr/bin/env python3
"""Generate Module 06 CNN notebooks (14 total)."""
import json
from pathlib import Path

M06 = Path(__file__).resolve().parent / "06_CNN"


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
    M06.mkdir(parents=True, exist_ok=True)
    (M06 / name).write_text(json.dumps(nb(cells), indent=1))
    print(f"  {name}")


def hdr(num, title, dur, objs, legacy=""):
    leg = f"\n\n**Legacy script:** `{legacy}`" if legacy else ""
    return md(
        f"# Notebook {num}: {title}\n\n**Module:** 06 CNN  \n**Duration:** ~{dur}{leg}\n\n---\n\n## Learning Objectives\n\n{objs}"
    )


def footer(summary, nxt=None):
    nxt = f"\n\n**Next:** [{nxt}]({nxt})" if nxt else ""
    return md(f"---\n\n## Summary\n\n{summary}{nxt}")


SETUP_NP = code(
    "import numpy as np\nimport matplotlib.pyplot as plt\nfrom pathlib import Path\n\nplt.rcParams['figure.figsize'] = (8, 5)\nrng = np.random.default_rng(42)"
)

SETUP_TORCH = code(
    "import numpy as np\nimport matplotlib.pyplot as plt\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch.utils.data import DataLoader\nimport torchvision\nimport torchvision.transforms as T\n\nplt.rcParams['figure.figsize'] = (8, 5)\ntorch.manual_seed(42)\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nprint(f'Device: {device}')"
)

CIFAR_LOAD = code(
    "transform = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])\n\ntrain_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)\ntest_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)\ntrain_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)\ntest_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)\nclasses = train_set.classes\nprint(f'CIFAR-10: {len(train_set)} train, {len(test_set)} test, {len(classes)} classes')"
)

TRAIN_LOOP = code(
    "def train_epoch(model, loader, optimizer, criterion):\n    model.train()\n    total, correct, loss_sum = 0, 0, 0\n    for X, y in loader:\n        X, y = X.to(device), y.to(device)\n        optimizer.zero_grad()\n        out = model(X)\n        loss = criterion(out, y)\n        loss.backward()\n        optimizer.step()\n        loss_sum += loss.item() * len(y)\n        correct += (out.argmax(1) == y).sum().item()\n        total += len(y)\n    return loss_sum/total, correct/total\n\ndef evaluate(model, loader, criterion):\n    model.eval()\n    total, correct, loss_sum = 0, 0, 0\n    with torch.no_grad():\n        for X, y in loader:\n            X, y = X.to(device), y.to(device)\n            out = model(X)\n            loss = criterion(out, y)\n            loss_sum += loss.item() * len(y)\n            correct += (out.argmax(1) == y).sum().item()\n            total += len(y)\n    return loss_sum/total, correct/total"
)

NOTEBOOKS = []


def register(name, cells):
    NOTEBOOKS.append((name, cells))


# ── FOUNDATIONS ─────────────────────────────────────────────────────────

register("01_Image_Representation.ipynb", [
    hdr("01", "Image Representation", "2 hrs",
        "1. Understand pixels, channels, and tensors\n2. Load and visualize images with PyTorch\n3. Know HWC vs CHW conventions\n4. Connect to satellite multi-band imagery"),
    md("## 1. Digital Images as Tensors\n\n**Grayscale:** shape $(H, W)$ — height × width pixels, values 0–255 or 0.0–1.0\n\n**RGB:** shape $(H, W, 3)$ or $(3, H, W)$ — 3 color channels\n\n**Batch:** $(N, C, H, W)$ — N images, C channels, H×W spatial\n\n**PyTorch convention:** $(N, C, H, W)$ — channels first\n\n**Your water-bodies project:** $(N, 6, 512, 512)$ — 6 Planet spectral bands"),
    SETUP_TORCH,
    code("# Load sample images from CIFAR-10\ndataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)\nimg, label = dataset[0]\nprint(f'PIL image size: {img.size}')  # (W, H) = (32, 32)\n\nimg_t = T.ToTensor()(img)  # (C, H, W)\nprint(f'Tensor shape: {img_t.shape}')  # (3, 32, 32)\nprint(f'Value range: [{img_t.min():.2f}, {img_t.max():.2f}]')"),
    code("fig, axes = plt.subplots(1, 3, figsize=(10, 3))\nimg_np = np.array(img)\naxes[0].imshow(img_np); axes[0].set_title(f'RGB — {classes[0] if \"classes\" in dir() else label}'); axes[0].axis('off')\nfor i, (c, ax) in enumerate(zip(['R','G','B'], axes[1:])):\n    ax.imshow(img_t[i], cmap='gray'); ax.set_title(f'{c} channel'); ax.axis('off')\nplt.tight_layout(); plt.show()"),
    md("## GeoSpatial Note\n\nSatellite GeoTIFF: `(bands, height, width)` in rasterio. Same as PyTorch $(C, H, W)$ per tile."),
    footer("Images are 3D/4D tensors. PyTorch uses NCHW format.", "02_Convolution_Operation.ipynb"),
])

register("02_Convolution_Operation.ipynb", [
    hdr("02", "Convolution Operation", "2.5 hrs",
        "1. Derive 2D convolution mathematically\n2. Implement convolution from scratch in NumPy\n3. Compare with PyTorch nn.Conv2d\n4. Understand kernels/filters"),
    md("## 1. Mathematical Definition\n\n2D convolution (cross-correlation in DL):\n\n$$(I * K)_{ij} = \\sum_m \\sum_n I_{i+m, j+n} \\cdot K_{m,n}$$\n\n- $I$ = input image\n- $K$ = kernel/filter (learnable weights)\n- Output = feature map\n\n**Key insight:** Convolution detects local patterns (edges, textures) by sliding a small kernel across the image."),
    SETUP_NP,
    code("def conv2d_manual(image, kernel):\n    \"\"\"Naive 2D convolution (valid, no padding, stride=1).\"\"\"\n    h, w = image.shape\n    kh, kw = kernel.shape\n    out_h, out_w = h - kh + 1, w - kw + 1\n    output = np.zeros((out_h, out_w))\n    for i in range(out_h):\n        for j in range(out_w):\n            output[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)\n    return output\n\n# Edge detection kernel (Sobel X)\nimage = np.random.rand(8, 8)\nsobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])\nresult = conv2d_manual(image, sobel_x)\nprint(f'Input: {image.shape}, Kernel: {sobel_x.shape}, Output: {result.shape}')"),
    SETUP_TORCH,
    code("# PyTorch equivalent\nimg = torch.randn(1, 1, 8, 8)  # (N, C, H, W)\nkernel = torch.tensor(sobel_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)\nout = F.conv2d(img, kernel)\nprint(f'PyTorch conv2d output shape: {out.shape}')"),
    md("## Exercise\n\nImplement vertical Sobel filter. Apply to a grayscale MNIST digit."),
    code("# YOUR CODE HERE\n"),
    footer("Convolution slides learnable filters across the image to detect local features.", "03_Padding_and_Stride.ipynb"),
])

register("03_Padding_and_Stride.ipynb", [
    hdr("03", "Padding and Stride", "2 hrs",
        "1. Derive output size formula\n2. Understand valid, same, and full padding\n3. Use stride to downsample"),
    md("## Output Size Formula\n\n$$O = \\left\\lfloor \\frac{W - K + 2P}{S} \\right\\rfloor + 1$$\n\n- $W$ = input width/height\n- $K$ = kernel size\n- $P$ = padding\n- $S$ = stride\n\n**Same padding:** $P = (K-1)/2$ for $S=1$ → output size = input size\n\n**Stride > 1:** downsamples spatial dimensions (replaces some pooling)."),
    SETUP_TORCH,
    code("def output_size(W, K, P, S):\n    return (W - K + 2*P) // S + 1\n\nfor W, K, P, S in [(32,3,1,1),(32,3,0,1),(32,3,1,2),(224,7,3,2)]:\n    print(f'W={W}, K={K}, P={P}, S={S} → O={output_size(W,K,P,S)}')"),
    code("x = torch.randn(1, 3, 32, 32)\nfor padding, stride in [(1,1),(0,1),(1,2)]:\n    out = F.conv2d(x, torch.randn(16, 3, 3, 3), padding=padding, stride=stride)\n    print(f'padding={padding}, stride={stride} → {tuple(out.shape)}')"),
    footer("Padding preserves spatial size; stride downsamples.", "04_Pooling.ipynb"),
])

register("04_Pooling.ipynb", [
    hdr("04", "Pooling Layers", "1.5 hrs",
        "1. Understand max pooling and average pooling\n2. Know why pooling provides translation invariance\n3. Compute output dimensions"),
    md("## Pooling\n\n**Max Pool:** $y_{ij} = \\max_{(m,n) \\in \\text{window}} x_{i+m, j+n}$\n\n**Avg Pool:** $y_{ij} = \\text{mean}_{(m,n) \\in \\text{window}} x_{i+m, j+n}$\n\nReduces spatial dimensions, provides local translation invariance, increases receptive field."),
    SETUP_TORCH,
    code("x = torch.randn(1, 1, 8, 8)\nmax_out = F.max_pool2d(x, kernel_size=2, stride=2)\navg_out = F.avg_pool2d(x, kernel_size=2, stride=2)\nprint(f'Input: {x.shape}, MaxPool: {max_out.shape}, AvgPool: {avg_out.shape}')"),
    code("# Visualize max pooling\nsample = torch.arange(16, dtype=torch.float32).reshape(1,1,4,4)\nprint('Input:\\n', sample[0,0].numpy())\nprint('MaxPool 2x2:\\n', F.max_pool2d(sample, 2)[0,0].numpy())"),
    footer("Pooling downsamples feature maps and adds translation invariance.", "05_Feature_Maps.ipynb"),
])

register("05_Feature_Maps.ipynb", [
    hdr("05", "Feature Maps and Filter Visualization", "2 hrs",
        "1. Understand what CNN filters learn\n2. Visualize first-layer filters\n3. Visualize activation maps"),
    md("## What Filters Learn\n\n- **Layer 1:** edges, colors, simple textures\n- **Layer 2-3:** corners, patterns\n- **Deeper layers:** object parts, complex textures\n\n**Feature map:** output of one filter applied to input — highlights where that pattern appears."),
    SETUP_TORCH,
    code("from torchvision import models\n\n# Load pretrained AlexNet, visualize first conv filters\nmodel = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)\nfirst_conv = model.features[0].weight.data  # (64, 3, 11, 11)\nprint(f'First conv weights shape: {first_conv.shape}')\n\nfig, axes = plt.subplots(4, 8, figsize=(12, 6))\nfor i, ax in enumerate(axes.flat):\n    if i < first_conv.shape[0]:\n        f = first_conv[i].permute(1, 2, 0)\n        f = (f - f.min()) / (f.max() - f.min() + 1e-8)\n        ax.imshow(f.numpy()); ax.axis('off')\nplt.suptitle('AlexNet First Layer Filters (learned edge detectors)'); plt.tight_layout(); plt.show()"),
    footer("Early filters detect edges; deeper layers detect complex patterns.", "06_Receptive_Field.ipynb"),
])

register("06_Receptive_Field.ipynb", [
    hdr("06", "Receptive Field", "1.5 hrs",
        "1. Compute receptive field size layer by layer\n2. Understand why depth increases RF\n3. Connect to segmentation context"),
    md("## Receptive Field (RF)\n\nThe RF of a neuron = region of input image that influences its value.\n\n**Recursive formula:**\n$$RF_l = RF_{l-1} + (K_l - 1) \\times \\prod_{i=1}^{l-1} S_i$$\n\nDeep networks see larger portions of the input — critical for context in segmentation."),
    SETUP_NP,
    code("def compute_rf(layers):\n    \"\"\"layers: list of (kernel_size, stride) per layer\"\"\"\n    rf, jump = 1, 1\n    for K, S in layers:\n        rf = rf + (K - 1) * jump\n        jump = jump * S\n    return rf\n\n# AlexNet-like stack\nlayers = [(11,4),(3,1),(3,1),(3,1),(3,1),(3,1)]\nprint(f'Receptive field after {len(layers)} layers: {compute_rf(layers)} pixels')\n\n# water-bodies SE-ResNet50: RF covers entire 512x512 tile after encoder\nprint('SE-ResNet50 encoder RF >> 512 → each output pixel sees full tile context')"),
    footer("Receptive field grows with depth — why deep encoders capture global context.", "07_LeNet.ipynb"),
])

# ── ARCHITECTURES ───────────────────────────────────────────────────────

def arch_notebook(num, name, year, innovation, arch_code, nxt):
    return [
        hdr(num, name, "2 hrs", f"1. Understand {name} architecture ({year})\n2. Key innovation: {innovation}\n3. Implement in PyTorch\n4. Train/evaluate on CIFAR-10 subset"),
        md(f"## {name} ({year})\n\n**Key innovation:** {innovation}"),
        SETUP_TORCH, CIFAR_LOAD, TRAIN_LOOP,
        code(arch_code),
        md(f"## Training (3 epochs demo on subset — run full training for benchmark)"),
        code("model = model.to(device)\noptimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\ncriterion = nn.CrossEntropyLoss()\n\n# Subset for demo speed\nfrom torch.utils.data import Subset\nsubset = Subset(train_set, range(2000))\nsub_loader = DataLoader(subset, batch_size=64, shuffle=True)\n\nfor epoch in range(3):\n    tr_loss, tr_acc = train_epoch(model, sub_loader, optimizer, criterion)\n    te_loss, te_acc = evaluate(model, test_loader, criterion)\n    print(f'Epoch {epoch+1}: train_acc={tr_acc:.3f}, test_acc={te_acc:.3f}')"),
        footer(f"{name}: {innovation}", nxt),
    ]

register("07_LeNet.ipynb", arch_notebook("07", "LeNet-5", "1998", "First successful CNN for digit recognition",
    "class LeNet5(nn.Module):\n    def __init__(self, num_classes=10):\n        super().__init__()\n        self.features = nn.Sequential(\n            nn.Conv2d(3, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),\n            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),\n        )\n        self.classifier = nn.Sequential(\n            nn.Flatten(), nn.Linear(16*6*6, 120), nn.ReLU(),\n            nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, num_classes),\n        )\n    def forward(self, x): return self.classifier(self.features(x))\n\nmodel = LeNet5()\nprint(model)\nprint(f'Params: {sum(p.numel() for p in model.parameters()):,}')",
    "08_AlexNet.ipynb"))

register("08_AlexNet.ipynb", arch_notebook("08", "AlexNet", "2012", "ReLU + dropout + GPU training sparked deep learning revolution",
    "class AlexNetCIFAR(nn.Module):\n    def __init__(self, num_classes=10):\n        super().__init__()\n        self.features = nn.Sequential(\n            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),\n            nn.Conv2d(64, 192, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),\n            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(),\n            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),\n            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),\n        )\n        self.classifier = nn.Sequential(\n            nn.Flatten(), nn.Linear(256*4*4, 1024), nn.ReLU(), nn.Dropout(0.5),\n            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes),\n        )\n    def forward(self, x): return self.classifier(self.features(x))\n\nmodel = AlexNetCIFAR()\nprint(f'Params: {sum(p.numel() for p in model.parameters()):,}')",
    "09_VGG.ipynb"))

register("09_VGG.ipynb", [
    hdr("09", "VGG", "2 hrs", "1. Understand VGG depth with 3×3 filters\n2. Build VGG-style blocks\n3. See parameter count explosion"),
    md("## VGG (Simonyan & Zisserman, 2014)\n\n**Key idea:** Only 3×3 conv layers, stacked deeply. Smaller filters + more layers = more non-linearity with fewer parameters than large filters.\n\nVGG-16: 13 conv + 3 FC layers. ~138M parameters."),
    SETUP_TORCH,
    code("def vgg_block(in_ch, out_ch, n_convs):\n    layers = []\n    for i in range(n_convs):\n        layers += [nn.Conv2d(in_ch if i==0 else out_ch, out_ch, 3, padding=1), nn.ReLU()]\n    layers.append(nn.MaxPool2d(2))\n    return nn.Sequential(*layers)\n\nclass VGGSmall(nn.Module):\n    def __init__(self, num_classes=10):\n        super().__init__()\n        self.features = nn.Sequential(\n            vgg_block(3, 64, 2), vgg_block(64, 128, 2), vgg_block(128, 256, 2),\n        )\n        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256*4*4, 256), nn.ReLU(), nn.Linear(256, num_classes))\n    def forward(self, x): return self.classifier(self.features(x))\n\nmodel = VGGSmall()\nprint(f'VGG-Small params: {sum(p.numel() for p in model.parameters()):,}')"),
    footer("VGG: stack 3×3 convs for depth. Simple but many parameters.", "10_GoogLeNet.ipynb"),
])

register("10_GoogLeNet.ipynb", [
    hdr("10", "GoogLeNet (Inception)", "2 hrs", "1. Understand inception modules\n2. Multi-scale feature extraction in parallel\n3. 1×1 conv for dimension reduction"),
    md("## Inception Module\n\nApply 1×1, 3×3, 5×5 conv and max pool **in parallel**, concatenate outputs.\n\n**1×1 conv bottleneck:** reduces channels before expensive 3×3/5×5 convs."),
    SETUP_TORCH,
    code("class InceptionModule(nn.Module):\n    def __init__(self, in_ch, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):\n        super().__init__()\n        self.branch1 = nn.Conv2d(in_ch, ch1x1, 1)\n        self.branch2 = nn.Sequential(nn.Conv2d(in_ch, ch3x3red, 1), nn.ReLU(), nn.Conv2d(ch3x3red, ch3x3, 3, padding=1))\n        self.branch3 = nn.Sequential(nn.Conv2d(in_ch, ch5x5red, 1), nn.ReLU(), nn.Conv2d(ch5x5red, ch5x5, 5, padding=2))\n        self.branch4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(in_ch, pool_proj, 1))\n    def forward(self, x):\n        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)\n\ninc = InceptionModule(192, 64, 96, 128, 16, 32, 32)\nprint(f'Inception output channels: {inc(torch.randn(1,192,28,28)).shape}')"),
    footer("Inception: multi-scale parallel convolutions with 1×1 bottlenecks.", "11_ResNet.ipynb"),
])

register("11_ResNet.ipynb", [
    hdr("11", "ResNet", "2.5 hrs",
        "1. Understand skip/residual connections\n2. Implement ResNet block\n3. Connect to water-bodies SE-ResNet50 encoder\n4. Train ResNet on CIFAR-10"),
    md("## ResNet (He et al., 2016)\n\n**Problem:** Very deep networks degrade (vanishing gradients).\n\n**Solution:** Skip connection: $y = F(x) + x$\n\nLearn residual $F(x)$ instead of desired mapping $H(x)$.\n\n**Your water-bodies-detection** uses **SE-ResNet50** encoder — ResNet50 + Squeeze-and-Excitation channel attention."),
    SETUP_TORCH, CIFAR_LOAD, TRAIN_LOOP,
    code("class ResidualBlock(nn.Module):\n    def __init__(self, in_ch, out_ch, stride=1):\n        super().__init__()\n        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)\n        self.bn1 = nn.BatchNorm2d(out_ch)\n        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)\n        self.bn2 = nn.BatchNorm2d(out_ch)\n        self.shortcut = nn.Sequential()\n        if stride != 1 or in_ch != out_ch:\n            self.shortcut = nn.Sequential(\n                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))\n    def forward(self, x):\n        out = F.relu(self.bn1(self.conv1(x)))\n        out = self.bn2(self.conv2(out))\n        return F.relu(out + self.shortcut(x))\n\nclass ResNetSmall(nn.Module):\n    def __init__(self, num_classes=10):\n        super().__init__()\n        self.stem = nn.Sequential(nn.Conv2d(3,64,3,padding=1,bias=False), nn.BatchNorm2d(64), nn.ReLU())\n        self.layer1 = ResidualBlock(64, 64)\n        self.layer2 = ResidualBlock(64, 128, stride=2)\n        self.layer3 = ResidualBlock(128, 256, stride=2)\n        self.pool = nn.AdaptiveAvgPool2d(1)\n        self.fc = nn.Linear(256, num_classes)\n    def forward(self, x):\n        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)\n        return self.fc(self.pool(x).flatten(1))\n\nmodel = ResNetSmall()\nprint(f'ResNet-Small params: {sum(p.numel() for p in model.parameters()):,}')"),
    code("# Pretrained ResNet50 — same family as water-bodies encoder\nfrom torchvision.models import resnet50, ResNet50_Weights\npretrained = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)\nprint('ResNet50 layer names (first 10):', [n for n,_ in list(pretrained.named_children())[:10]])\nprint('\\nwater-bodies uses se_resnet50 — ResNet50 + Squeeze-Excitation')"),
    md("## Exercise\n\nTrain ResNetSmall for 10 epochs on full CIFAR-10. Target: >70% test accuracy."),
    code("# YOUR CODE HERE\n"),
    footer("ResNet skip connections enable training of very deep networks. Your segmentation encoder is ResNet-based.", "12_DenseNet.ipynb"),
])

register("12_DenseNet.ipynb", [
    hdr("12", "DenseNet", "2 hrs", "1. Understand dense connections\n2. Compare DenseNet vs ResNet\n3. Feature reuse advantage"),
    md("## DenseNet (Huang et al., 2017)\n\nEach layer receives **all previous feature maps** as input:\n$$x_l = H_l([x_0, x_1, \\ldots, x_{l-1}])$$\n\n**Benefit:** Feature reuse, fewer parameters, strong gradient flow."),
    SETUP_TORCH,
    code("class DenseBlock(nn.Module):\n    def __init__(self, in_ch, growth_rate, n_layers):\n        super().__init__()\n        layers = []\n        ch = in_ch\n        for _ in range(n_layers):\n            layers.append(nn.Sequential(\n                nn.BatchNorm2d(ch), nn.ReLU(), nn.Conv2d(ch, growth_rate, 3, padding=1)))\n            ch += growth_rate\n        self.layers = nn.ModuleList(layers)\n        self.out_ch = ch\n    def forward(self, x):\n        features = [x]\n        for layer in self.layers:\n            out = layer(torch.cat(features, dim=1))\n            features.append(out)\n        return torch.cat(features, dim=1)\n\nblock = DenseBlock(64, growth_rate=32, n_layers=4)\nprint(f'DenseBlock output: {block(torch.randn(1,64,16,16)).shape}')"),
    footer("DenseNet connects every layer to every subsequent layer — maximum feature reuse.", "13_EfficientNet.ipynb"),
])

register("13_EfficientNet.ipynb", [
    hdr("13", "EfficientNet", "2 hrs", "1. Understand compound scaling\n2. Use torchvision EfficientNet\n3. Balance depth, width, resolution"),
    md("## EfficientNet (Tan & Le, 2019)\n\n**Compound scaling:** uniformly scale depth ($\\alpha$), width ($\\beta$), and resolution ($\\gamma$):\n\n$$\\text{depth} = \\alpha^\\phi, \\quad \\text{width} = \\beta^\\phi, \\quad \\text{resolution} = \\gamma^\\phi$$\n\nSubject to $\\alpha \\beta^2 \\gamma^2 \\approx 2$ and baseline FLOPS constraint."),
    SETUP_TORCH,
    code("from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights\n\nmodel = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)\nprint(f'EfficientNet-B0 params: {sum(p.numel() for p in model.parameters()):,}')\n\n# Compound scaling variants\nfor name in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']:\n    m = getattr(__import__('torchvision.models', fromlist=[name]), name)(weights=None)\n    print(f'{name}: {sum(p.numel() for p in m.parameters()):,} params')"),
    footer("EfficientNet scales depth, width, and resolution together for optimal accuracy/FLOPS.", "14_ConvNeXt.ipynb"),
])

register("14_ConvNeXt.ipynb", [
    hdr("14", "ConvNeXt & Module 06 Capstone", "2.5 hrs",
        "1. Understand modernized CNN design\n2. Compare ConvNeXt with ViT preview\n3. Complete MNIST CNN and CIFAR-10 projects\n4. Connect to water-bodies encoder"),
    md("## ConvNeXt (Liu et al., 2022)\n\nModernizes ResNet with ViT-inspired design choices:\n- Larger kernels (7×7)\n- LayerNorm instead of BatchNorm\n- GELU instead of ReLU\n- Inverted bottleneck (like Transformer FFN)\n\n**Bridge to Module 10:** ConvNeXt borrows from Transformers; ViT borrows from CNNs."),
    SETUP_TORCH, CIFAR_LOAD, TRAIN_LOOP,
    code("class ConvNeXtBlock(nn.Module):\n    def __init__(self, dim):\n        super().__init__()\n        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)  # depthwise\n        self.norm = nn.LayerNorm(dim, eps=1e-6)\n        self.pwconv1 = nn.Linear(dim, 4*dim)  # pointwise\n        self.act = nn.GELU()\n        self.pwconv2 = nn.Linear(4*dim, dim)\n    def forward(self, x):\n        residual = x\n        x = self.dwconv(x)\n        x = x.permute(0,2,3,1); x = self.norm(x)\n        x = self.pwconv2(self.act(self.pwconv1(x)))\n        x = x.permute(0,3,1,2)\n        return residual + x\n\nblock = ConvNeXtBlock(64)\nprint(f'ConvNeXt block: {block(torch.randn(1,64,16,16)).shape}')"),
    md("## Module 06 Capstone: MNIST CNN\n\nBeat your Module 05 MLP on MNIST with a simple CNN."),
    code("class MNISTCNN(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),\n            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),\n            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU(), nn.Linear(128, 10),\n        )\n    def forward(self, x): return self.net(x)\n\nprint('MNIST CNN — train this to beat Module 05 MLP (~98% target)')"),
    md("## water-bodies-detection Connection\n\n```\nYour pipeline:\n  Input: (6, 512, 512) Planet bands\n  Encoder: SE-ResNet50 (Module 06 ResNet + channel attention)\n  Decoder: UNet++ (Module 07)\n  Output: (2, 512, 512) aqua + boundary masks\n```\n\nModule 06 teaches what happens inside that ResNet50 encoder."),
    md("## Module 06 Complete\n\n**Next:** Module 07 Segmentation — UNet, UNet++, and your water-bodies project."),
    footer("ConvNeXt modernizes CNNs. You now understand every major architecture from LeNet to ConvNeXt.", None),
])


def main():
    print("Building Module 06 notebooks...")
    for name, cells in NOTEBOOKS:
        save(name, cells)
    print(f"Done: {len(NOTEBOOKS)} notebooks created.")


if __name__ == "__main__":
    main()
