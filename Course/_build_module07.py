#!/usr/bin/env python3
"""Generate Module 07 Segmentation notebooks (20 total)."""
import json
from pathlib import Path

M07 = Path(__file__).resolve().parent / "07_Segmentation"


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
    M07.mkdir(parents=True, exist_ok=True)
    (M07 / name).write_text(json.dumps(nb(cells), indent=1))
    print(f"  {name}")


def hdr(num, title, dur, objs):
    return md(
        f"# Notebook {num}: {title}\n\n**Module:** 07 Segmentation  \n**Duration:** ~{dur}\n\n---\n\n## Learning Objectives\n\n{objs}"
    )


def footer(summary, nxt=None):
    nxt = f"\n\n**Next:** [{nxt}]({nxt})" if nxt else ""
    return md(f"---\n\n## Summary\n\n{summary}{nxt}")


SETUP = code(
    "import numpy as np\nimport matplotlib.pyplot as plt\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch.utils.data import DataLoader, TensorDataset\n\nplt.rcParams['figure.figsize'] = (8, 5)\ntorch.manual_seed(42)\nrng = np.random.default_rng(42)\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nprint(f'Device: {device}')"
)

METRICS_CODE = code(
    "def pixel_accuracy(pred, target):\n    return (pred == target).float().mean().item()\n\ndef iou_score(pred, target, n_classes=None, smooth=1e-7):\n    \"\"\"pred, target: (N,H,W) long tensors\"\"\"\n    if n_classes is None:\n        n_classes = int(max(pred.max(), target.max())) + 1\n    ious = []\n    for c in range(n_classes):\n        p = (pred == c)\n        t = (target == c)\n        inter = (p & t).sum().float()\n        union = (p | t).sum().float()\n        if union > 0:\n            ious.append(((inter + smooth) / (union + smooth)).item())\n    return float(np.mean(ious)) if ious else 0.0\n\ndef dice_score(pred, target, smooth=1e-7):\n    \"\"\"Binary pred, target: (N,1,H,W) float\"\"\"\n    inter = (pred * target).sum(dim=(2,3))\n    return ((2*inter + smooth) / (pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + smooth)).mean().item()"
)

NOTEBOOKS = []


def register(name, cells):
    NOTEBOOKS.append((name, cells))


# ── SEGMENTATION TYPES (01-05) ──────────────────────────────────────────

register("01_Binary_Segmentation.ipynb", [
    hdr("01", "Binary Segmentation", "2 hrs",
        "1. Define binary segmentation\n2. Understand pixel-wise classification\n3. Implement simple binary UNet\n4. Connect to water vs land / aqua detection"),
    md("## 1. Binary Segmentation\n\n**Task:** Classify each pixel as foreground (1) or background (0).\n\n**Output:** Single channel mask of shape $(H, W)$ with values in $\\{0, 1\\}$ or probabilities $[0, 1]$.\n\n**Examples:** Water body detection, building footprint, road extraction, medical tumor segmentation.\n\n**Loss:** Binary cross-entropy + Dice (your water-bodies project)."),
    SETUP,
    code("# Synthetic binary segmentation data\nN, H, W = 32, 64, 64\nimages = torch.randn(N, 3, H, W)\nmasks = (torch.sin(torch.linspace(0, 3.14, W)) > 0).float()\nmasks = masks.unsqueeze(0).unsqueeze(0).expand(N, 1, H, W)\n\nfig, axes = plt.subplots(1, 3, figsize=(10, 3))\naxes[0].imshow(images[0].permute(1,2,0).numpy().clip(0,1)); axes[0].set_title('Image'); axes[0].axis('off')\naxes[1].imshow(masks[0,0], cmap='Blues'); axes[1].set_title('Binary Mask'); axes[1].axis('off')\naxes[2].imshow((images[0].permute(1,2,0).numpy()*0.5 + masks[0,0].numpy()[:,:,None]*0.5).clip(0,1))\naxes[2].set_title('Overlay'); axes[2].axis('off')\nplt.tight_layout(); plt.show()"),
    md("## GeoSpatial: Your water-bodies project predicts binary aqua interior mask (channel 0) per pixel."),
    footer("Binary segmentation = per-pixel foreground/background classification.", "02_Multi_Class_Segmentation.ipynb"),
])

register("02_Multi_Class_Segmentation.ipynb", [
    hdr("02", "Multi-Class Segmentation", "2 hrs",
        "1. Distinguish multi-class from binary\n2. Use softmax output per pixel\n3. Compute mIoU across classes"),
    md("## Multi-Class Segmentation\n\nEach pixel belongs to **exactly one** of $K$ classes (mutually exclusive).\n\n**Output:** $(K, H, W)$ logits → softmax → argmax → $(H, W)$ label map.\n\n**Loss:** Cross-entropy per pixel.\n\n**Metric:** mIoU (mean IoU across classes)."),
    SETUP, METRICS_CODE,
    code("# 3-class synthetic: background, road, building\nN, H, W, K = 16, 32, 32, 3\ntarget = torch.randint(0, K, (N, H, W))\nlogits = torch.randn(N, K, H, W)\npred = logits.argmax(dim=1)\n\nprint(f'Pixel accuracy: {pixel_accuracy(pred, target):.4f}')\nprint(f'mIoU: {iou_score(pred, target, K):.4f}')"),
    md("## GeoSpatial: Multi-class land cover (water, vegetation, urban, bare soil) from satellite imagery."),
    footer("Multi-class: one label per pixel, softmax + cross-entropy.", "03_Multi_Label_Segmentation.ipynb"),
])

register("03_Multi_Label_Segmentation.ipynb", [
    hdr("03", "Multi-Label Segmentation", "2 hrs",
        "1. Understand independent labels per pixel\n2. Use sigmoid (not softmax) per channel\n3. Distinguish from multi-class"),
    md("## Multi-Label vs Multi-Class\n\n| | Multi-Class | Multi-Label |\n|---|-------------|-------------|\n| Labels per pixel | Exactly 1 | 0 or more |\n| Output activation | Softmax | Sigmoid |\n| Loss | Cross-entropy | BCE per channel |\n| Example | Land cover class | Cloud AND shadow |\n\n**Your water-bodies:** Two binary heads (aqua + boundary) = multi-label style with 2 channels."),
    SETUP,
    code("# Multi-label: pixel can have aqua=1 AND boundary=1 (bund pixel)\ntarget = torch.zeros(4, 2, 32, 32)\ntarget[:, 0, 10:22, 10:22] = 1.0  # aqua interior\ntarget[:, 1, 9:23, 9:23] = 1.0   # boundary ring\n\nfig, axes = plt.subplots(1, 2, figsize=(8, 3))\naxes[0].imshow(target[0,0], cmap='Blues'); axes[0].set_title('Aqua channel'); axes[0].axis('off')\naxes[1].imshow(target[0,1], cmap='Reds'); axes[1].set_title('Boundary channel'); axes[1].axis('off')\nplt.tight_layout(); plt.show()"),
    footer("Multi-label uses sigmoid per channel — your dual-head design is multi-label.", "04_Segmentation_Types_Compared.ipynb"),
])

register("04_Segmentation_Types_Compared.ipynb", [
    hdr("04", "Semantic vs Instance vs Panoptic", "2 hrs",
        "1. Compare semantic, instance, and panoptic segmentation\n2. Know when to use each\n3. Preview Module 09"),
    md("## Three Segmentation Types\n\n**Semantic:** Classify every pixel (all ponds = same class \"water\").\n\n**Instance:** Separate individual objects (pond 1, pond 2, pond 3).\n\n**Panoptic:** Semantic + instance (stuff + things).\n\n| Type | Separates instances? | Module |\n|------|---------------------|--------|\n| Semantic | No | 07 (this module) |\n| Instance | Yes | 09 |\n| Panoptic | Both | 09 |"),
    md("## Your Adjacent Pond Problem\n\n**Semantic only:** Adjacent ponds merge into one blob.\n\n**Your solution (water-bodies):** Dual-head boundary detection separates ponds without full instance segmentation.\n\n**Instance seg alternative:** Mask R-CNN assigns unique ID per pond (Module 09)."),
    footer("Choose segmentation type based on whether you need individual object IDs.", "05_Boundary_Detection.ipynb"),
])

register("05_Boundary_Detection.ipynb", [
    hdr("05", "Boundary Detection", "2 hrs",
        "1. Understand boundary/bund prediction\n2. See how boundaries separate adjacent regions\n3. Connect to water-bodies boundary head"),
    md("## Boundary Detection\n\nPredict a thin band at object edges. Used to:\n- Separate touching instances (your aquaculture ponds)\n- Improve edge quality in medical imaging\n- Post-process semantic masks\n\n**Your project:** `masks_boundary/` — dilated polygon edges at configurable width (meters per pixel)."),
    SETUP,
    code("# Simulate aqua mask + boundary mask\naqua = torch.zeros(1, 1, 64, 64)\naqua[0, 0, 15:50, 15:50] = 1.0\n# Boundary = morphological gradient (dilate - erode approximation)\npad = aqua.clone()\npad[:,:,1:,:] = torch.maximum(pad[:,:,1:,:], aqua[:,:,:-1,:])\npad[:,:,:-1,:] = torch.maximum(pad[:,:,:-1,:], aqua[:,:,1:,:])\nboundary = (pad - aqua).clamp(0, 1)\n\nfig, axes = plt.subplots(1, 3, figsize=(10, 3))\nfor ax, m, t in zip(axes, [aqua[0,0], boundary[0,0], aqua[0,0]+boundary[0,0]*0.5], ['Aqua','Boundary','Combined']):\n    ax.imshow(m.numpy(), cmap='Blues'); ax.set_title(t); ax.axis('off')\nplt.tight_layout(); plt.show()"),
    footer("Boundary head prevents adjacent pond merging — core design of water-bodies-detection.", "06_FCN.ipynb"),
])

# ── ARCHITECTURES (06-14) ───────────────────────────────────────────────

register("06_FCN.ipynb", [
    hdr("06", "Fully Convolutional Networks (FCN)", "2.5 hrs",
        "1. Understand how FCN replaces FC layers with conv\n2. Implement upsampling with transposed convolution\n3. Know skip connections in FCN-8s"),
    md("## FCN (Long et al., 2015)\n\n**Key idea:** Replace fully connected layers of classification CNN with 1×1 convolutions → dense prediction map.\n\n**Upsampling:** Transposed convolution (deconv) or bilinear upsampling to restore spatial resolution.\n\n**FCN-8s:** Combine predictions from pool3, pool4, pool5 (skip connections)."),
    SETUP,
    code("class SimpleFCN(nn.Module):\n    def __init__(self, n_classes=2):\n        super().__init__()\n        self.enc = nn.Sequential(\n            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),\n            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),\n        )\n        self.head = nn.Conv2d(64, n_classes, 1)\n        self.up = nn.ConvTranspose2d(n_classes, n_classes, 4, stride=2, padding=1)\n    def forward(self, x):\n        x = self.enc(x)\n        x = self.head(x)\n        x = self.up(self.up(x))\n        return x\n\nmodel = SimpleFCN(2)\nx = torch.randn(2, 3, 64, 64)\nprint(f'FCN output: {model(x).shape}')  # (2, 2, 64, 64)"),
    footer("FCN pioneered dense prediction — replaced FC with conv.", "07_UNet.ipynb"),
])

register("07_UNet.ipynb", [
    hdr("07", "UNet", "2.5 hrs",
        "1. Understand encoder-decoder with skip connections\n2. Implement UNet in PyTorch\n3. Train on synthetic binary segmentation"),
    md("## UNet (Ronneberger et al., 2015)\n\n```\nEncoder (contract)          Decoder (expand)\n  Conv → Pool    ────────→  UpConv → Concat → Conv\n  Conv → Pool    ──skip──→  UpConv → Concat → Conv\n  ...                       ...\n  Bottleneck\n```\n\n**Skip connections:** Preserve fine spatial details lost during downsampling."),
    SETUP,
    code("class DoubleConv(nn.Module):\n    def __init__(self, in_ch, out_ch):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),\n            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU())\n    def forward(self, x): return self.net(x)\n\nclass UNet(nn.Module):\n    def __init__(self, in_ch=3, out_ch=1, features=(64,128,256)):\n        super().__init__()\n        self.downs = nn.ModuleList()\n        self.ups = nn.ModuleList()\n        ch = in_ch\n        for f in features:\n            self.downs.append(DoubleConv(ch, f)); ch = f\n        self.pool = nn.MaxPool2d(2)\n        self.bottleneck = DoubleConv(features[-1], features[-1]*2)\n        for f in reversed(features):\n            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, stride=2))\n            self.ups.append(DoubleConv(f*2, f))\n        self.final = nn.Conv2d(features[0], out_ch, 1)\n    def forward(self, x):\n        skips = []\n        for down in self.downs:\n            x = down(x); skips.append(x); x = self.pool(x)\n        x = self.bottleneck(x)\n        skips = skips[::-1]\n        for i in range(0, len(self.ups), 2):\n            x = self.ups[i](x)\n            skip = skips[i//2]\n            if x.shape != skip.shape:\n                x = F.interpolate(x, size=skip.shape[2:])\n            x = torch.cat([skip, x], dim=1)\n            x = self.ups[i+1](x)\n        return self.final(x)\n\nmodel = UNet(in_ch=3, out_ch=1)\nprint(f'UNet params: {sum(p.numel() for p in model.parameters()):,}')\nprint(f'Output: {model(torch.randn(1,3,64,64)).shape}')"),
    footer("UNet = encoder-decoder + skip connections. Foundation of GeoSpatial segmentation.", "08_UNetPlusPlus.ipynb"),
])

register("08_UNetPlusPlus.ipynb", [
    hdr("08", "UNet++", "3 hrs",
        "1. Understand nested skip pathways\n2. Connect to water-bodies-detection model.py\n3. Use segmentation_models_pytorch"),
    md("## UNet++ (Zhou et al., 2018)\n\nDense skip connections between **all** encoder and decoder nodes — not just matching levels.\n\n**Your water-bodies-detection:**\n```python\nsmp.UnetPlusPlus(\n    encoder_name='se_resnet50',\n    in_channels=6,  # Planet bands\n    classes=2,      # aqua + boundary\n    decoder_attention_type='scse',\n)\n```\n\nSee: `water-bodies-detection/model.py`"),
    SETUP,
    code("try:\n    import segmentation_models_pytorch as smp\n    model = smp.UnetPlusPlus(\n        encoder_name='resnet34',\n        encoder_weights='imagenet',\n        in_channels=3,\n        classes=2,\n        activation=None,\n    )\n    x = torch.randn(1, 3, 256, 256)\n    out = model(x)\n    print(f'UNet++ output shape: {out.shape}')  # (1, 2, 256, 256)\n    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')\nexcept ImportError:\n    print('Install: pip install segmentation-models-pytorch')\n    print('Your water-bodies project uses this exact library')"),
    md("## Dual-Head Output\n\n- **Channel 0:** Aqua interior logits → sigmoid → aqua probability\n- **Channel 1:** Bund boundary logits → sigmoid → boundary probability\n\nPost-process uses boundary to split adjacent ponds into separate polygons."),
    footer("UNet++ with SE-ResNet50 is your production architecture.", "09_DeepLab.ipynb"),
])

register("09_DeepLab.ipynb", [
    hdr("09", "DeepLab v3+", "2 hrs",
        "1. Understand atrous/dilated convolution\n2. Know ASPP (Atrous Spatial Pyramid Pooling)\n3. Compare DeepLab vs UNet for GeoSpatial"),
    md("## DeepLab v3+ (Chen et al., 2018)\n\n**Atrous convolution:** Same kernel, larger receptive field without downsampling.\n\n**ASPP:** Parallel atrous conv at rates (6, 12, 18) + global average pool → multi-scale context.\n\n**Encoder-Decoder:** DeepLabv3 + UNet-style decoder for sharper boundaries."),
    SETUP,
    code("class ASPP(nn.Module):\n    def __init__(self, in_ch, out_ch, rates=(6, 12, 18)):\n        super().__init__()\n        self.convs = nn.ModuleList([\n            nn.Conv2d(in_ch, out_ch, 1),\n            *[nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r) for r in rates],\n            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1)),\n        ])\n        self.project = nn.Conv2d(out_ch * (len(rates) + 2), out_ch, 1)\n    def forward(self, x):\n        h, w = x.shape[2:]\n        feats = [c(x) if i < len(self.convs)-1 else F.interpolate(c(x), (h,w)) for i,c in enumerate(self.convs)]\n        return self.project(torch.cat(feats, dim=1))\n\naspp = ASPP(256, 256)\nprint(f'ASPP output: {aspp(torch.randn(1,256,32,32)).shape}')"),
    footer("DeepLab uses atrous conv + ASPP for multi-scale context — good for objects at varying scales.", "10_PSPNet.ipynb"),
])

register("10_PSPNet.ipynb", [
    hdr("10", "PSPNet", "2 hrs", "1. Understand Pyramid Pooling Module\n2. Compare with ASPP"),
    md("## PSPNet (Zhao et al., 2017)\n\n**Pyramid Pooling Module (PPM):** Apply adaptive average pooling at scales (1×1, 2×2, 3×3, 6×6), upsample, concatenate.\n\nSimilar goal to ASPP: capture multi-scale context."),
    SETUP,
    code("class PPM(nn.Module):\n    def __init__(self, in_ch, pool_sizes=(1,2,3,6)):\n        super().__init__()\n        self.stages = nn.ModuleList([\n            nn.Sequential(nn.AdaptiveAvgPool2d(ps), nn.Conv2d(in_ch, in_ch//len(pool_sizes), 1))\n            for ps in pool_sizes\n        ])\n    def forward(self, x):\n        h, w = x.shape[2:]\n        return torch.cat([x] + [F.interpolate(s(x), (h,w), mode='bilinear') for s in self.stages], dim=1)\n\nppm = PPM(512)\nprint(f'PPM output channels: {ppm(torch.randn(1,512,32,32)).shape[1]}')"),
    footer("PSPNet pyramid pooling captures global and local context.", "11_HRNet.ipynb"),
])

register("11_HRNet.ipynb", [
    hdr("11", "HRNet", "2 hrs", "1. Understand high-resolution parallel streams\n2. Know when HRNet beats UNet"),
    md("## HRNet (Wang et al., 2019)\n\nMaintains **high-resolution representation** throughout — parallel multi-resolution streams with repeated cross-resolution fusion.\n\nBetter for tasks needing precise localization (pose estimation, fine boundaries)."),
    SETUP,
    code("try:\n    import segmentation_models_pytorch as smp\n    hrnet = smp.HRNet(encoder_name='hrnet_w18', encoder_weights=None, in_channels=3, classes=1)\n    print(f'HRNet output: {hrnet(torch.randn(1,3,128,128)).shape}')\nexcept Exception as e:\n    print(f'HRNet via SMP: {e}')\n    print('Concept: maintains high-res stream parallel to low-res — sharper boundaries')"),
    footer("HRNet preserves high-resolution features — good for precise boundaries.", "12_SegFormer.ipynb"),
])

register("12_SegFormer.ipynb", [
    hdr("12", "SegFormer", "2 hrs", "1. Preview transformer-based segmentation\n2. Understand MiT encoder + MLP decoder"),
    md("## SegFormer (Xie et al., 2021)\n\n**Encoder:** Hierarchical Vision Transformer (MiT)\n**Decoder:** Lightweight MLP (no complex upsampling)\n\nEfficient, strong on ADE20K/Cityscapes. Bridge to Module 10 Transformers."),
    SETUP,
    code("print('SegFormer architecture:')\nprint('  Encoder: Mix Transformer (MiT-B0 to MiT-B5)')\nprint('  Decoder: MLP on each scale → fuse → classify')\nprint('  No positional encoding interpolation needed')\nprint('\\nFor GeoSpatial: growing use for land cover mapping at scale')"),
    footer("SegFormer = Transformer encoder + simple MLP decoder.", "13_Mask2Former.ipynb"),
])

register("13_Mask2Former.ipynb", [
    hdr("13", "Mask2Former", "2 hrs", "1. Understand mask classification paradigm\n2. Preview universal segmentation"),
    md("## Mask2Former (Cheng et al., 2022)\n\nPredicts **N mask proposals** + class labels — unifies semantic, instance, and panoptic segmentation.\n\nUses masked attention in transformer decoder."),
    footer("Mask2Former unifies all segmentation types via mask classification.", "14_SAM.ipynb"),
])

register("14_SAM.ipynb", [
    hdr("14", "Segment Anything Model (SAM)", "2 hrs",
        "1. Understand promptable segmentation\n2. Know zero-shot segmentation use cases\n3. Preview for GeoSpatial"),
    md("## SAM (Kirillov et al., 2023)\n\n**Foundation model** for segmentation:\n- **Prompts:** points, boxes, or coarse masks\n- **Zero-shot:** works on unseen categories without fine-tuning\n\n**GeoSpatial use:** Prompt with point on pond → get mask. Useful for rapid labeling and inference on new regions."),
    SETUP,
    code("print('SAM components:')\nprint('  1. Image encoder (ViT-H)')\nprint('  2. Prompt encoder (points/boxes/masks)')\nprint('  3. Lightweight mask decoder')\nprint('\\nGeoSpatial workflow: click on water body → SAM segments it')\nprint('Limitation: may need fine-tuning for satellite multispectral (6+ bands)')"),
    footer("SAM enables prompt-based zero-shot segmentation.", "15_Cross_Entropy_Loss.ipynb"),
])

# ── LOSSES (15-20) ──────────────────────────────────────────────────────

register("15_Cross_Entropy_Loss.ipynb", [
    hdr("15", "Cross-Entropy Loss for Segmentation", "2 hrs",
        "1. Derive pixel-wise cross-entropy\n2. Implement in PyTorch\n3. Handle class weights for imbalance"),
    md("## Pixel-wise Cross-Entropy\n\n$$L = -\\frac{1}{N}\\sum_{i} \\sum_{c} y_{i,c} \\log \\hat{p}_{i,c}$$\n\nFor multi-class segmentation: `F.cross_entropy(logits, target)` where logits $(N,C,H,W)$, target $(N,H,W)$ long."),
    SETUP,
    code("logits = torch.randn(4, 3, 32, 32)\ntarget = torch.randint(0, 3, (4, 32, 32))\nloss = F.cross_entropy(logits, target)\nprint(f'CE loss: {loss.item():.4f}')\n\n# Class weights for imbalance\nweights = torch.tensor([0.2, 0.3, 0.5])  # rare class gets higher weight\nloss_w = F.cross_entropy(logits, target, weight=weights)\nprint(f'Weighted CE: {loss_w.item():.4f}')"),
    footer("Cross-entropy is default for multi-class segmentation.", "16_Dice_Loss.ipynb"),
])

register("16_Dice_Loss.ipynb", [
    hdr("16", "Dice Loss", "2 hrs",
        "1. Derive Dice coefficient and loss\n2. Implement dice_with_logits (your losses.py)\n3. Understand why Dice handles imbalance"),
    md("## Dice Coefficient\n\n$$\\text{Dice} = \\frac{2|A \\cap B|}{|A| + |B|}$$\n\n$$L_{\\text{Dice}} = 1 - \\text{Dice}$$\n\n**Soft Dice (differentiable):** Use sigmoid probabilities instead of binary.\n\n**Your losses.py:**\n```python\ndef dice_with_logits(logits, targets):\n    probs = torch.sigmoid(logits)\n    inter = (probs * targets).sum(dim=(2,3))\n    union = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3))\n    return (2*inter + 1) / (union + 1)\n```"),
    SETUP, METRICS_CODE,
    code("# Reimplement your water-bodies dice_with_logits\ndef dice_with_logits(logits, targets, smooth=1.0):\n    probs = torch.sigmoid(logits)\n    t = targets.float()\n    inter = (probs * t).sum(dim=(2, 3))\n    union = probs.sum(dim=(2, 3)) + t.sum(dim=(2, 3))\n    return ((2.0 * inter + smooth) / (union + smooth)).mean()\n\nlogits = torch.randn(4, 1, 32, 32)\ntargets = (torch.rand(4, 1, 32, 32) > 0.7).float()\nprint(f'Dice score: {dice_with_logits(logits, targets):.4f}')\nprint(f'Dice loss: {1 - dice_with_logits(logits, targets):.4f}')"),
    footer("Dice loss handles class imbalance — essential for small water bodies in large tiles.", "17_IoU_Loss.ipynb"),
])

register("17_IoU_Loss.ipynb", [
    hdr("17", "IoU Loss", "2 hrs", "1. Derive IoU/Jaccard loss\n2. Compare IoU vs Dice\n3. Implement iou_with_logits"),
    md("## IoU (Jaccard) Loss\n\n$$\\text{IoU} = \\frac{|A \\cap B|}{|A \\cup B|}, \\quad L = 1 - \\text{IoU}$$\n\n**Your losses.py `iou_with_logits`:** Used for validation metric during training."),
    SETUP,
    code("def iou_with_logits(logits, targets, thresh=0.5):\n    probs = torch.sigmoid(logits)\n    pred = (probs > thresh).float()\n    t = targets.float()\n    inter = (pred * t).sum(dim=(2, 3))\n    union = pred.sum(dim=(2, 3)) + t.sum(dim=(2, 3)) - inter\n    return (inter / (union + 1e-7)).mean()\n\nprint(f'IoU: {iou_with_logits(logits, targets):.4f}')"),
    md("## Dice vs IoU\n\nDice gives higher score when both sets small. IoU penalizes false positives more. Both used in GeoSpatial segmentation benchmarks."),
    footer("IoU is the standard segmentation metric; IoU loss directly optimizes it.", "18_Focal_Loss.ipynb"),
])

register("18_Focal_Loss.ipynb", [
    hdr("18", "Focal Loss", "2 hrs", "1. Derive focal loss for hard examples\n2. Apply to imbalanced segmentation"),
    md("## Focal Loss (Lin et al., 2017)\n\n$$L = -(1 - p_t)^\\gamma \\log(p_t)$$\n\nDown-weights easy pixels, focuses on hard ones. $\\gamma=2$ typical.\n\nUseful when background dominates (small objects in satellite tiles)."),
    SETUP,
    code("def focal_loss(logits, targets, gamma=2.0, alpha=0.25):\n    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')\n    probs = torch.sigmoid(logits)\n    p_t = probs * targets + (1 - probs) * (1 - targets)\n    return (alpha * (1 - p_t) ** gamma * bce).mean()\n\nprint(f'Focal loss: {focal_loss(logits, targets):.4f}')"),
    footer("Focal loss focuses training on hard, misclassified pixels.", "19_Boundary_Loss.ipynb"),
])

register("19_Boundary_Loss.ipynb", [
    hdr("19", "Boundary Loss", "2 hrs", "1. Understand distance-weighted loss at boundaries\n2. Connect to your bund boundary head"),
    md("## Boundary Loss\n\nWeight pixels near object boundaries higher in loss function — improves edge quality.\n\n**Your project:** Separate boundary head (channel 1) explicitly predicts bund pixels — stronger than post-hoc boundary loss."),
    SETUP,
    code("# Distance transform weight map (concept)\nmask = torch.zeros(1, 1, 64, 64)\nmask[0, 0, 20:44, 20:44] = 1.0\n# Weight higher near edges\nfrom scipy.ndimage import distance_transform_edt\nm = mask[0,0].numpy()\ndist_in = distance_transform_edt(m)\ndist_out = distance_transform_edt(1 - m)\ndist = np.minimum(dist_in, dist_out)\nweight = np.exp(-dist**2 / (2*2.0**2))  # sigma=2\nplt.imshow(weight, cmap='hot'); plt.title('Boundary weight map'); plt.colorbar(); plt.show()"),
    footer("Boundary-aware loss improves edge precision — your dual-head design is even stronger.", "20_AquaBoundaryLoss_and_Capstone.ipynb"),
])

register("20_AquaBoundaryLoss_and_Capstone.ipynb", [
    hdr("20", "AquaBoundaryLoss & water-bodies-detection Capstone", "3 hrs",
        "1. Walk through complete loss from losses.py\n2. Understand training metrics\n3. Preview full pipeline\n4. Complete Module 07"),
    md("## Your Complete Loss (losses.py)\n\n```python\nclass AquaBoundaryLoss(nn.Module):\n    def forward(self, logits, targets):\n        la = BCEDiceLoss()(logits[:, 0:1], targets[:, 0:1])  # aqua\n        lb = BCEDiceLoss()(logits[:, 1:2], targets[:, 1:2])  # boundary\n        return w_aqua * la + w_boundary * lb\n```\n\n**BCEDiceLoss:** 0.35 × BCE + 0.65 × Dice per channel\n\n**Why BCE + Dice?** BCE for pixel-wise accuracy, Dice for region overlap (handles imbalance)."),
    SETUP,
    code("# Full reimplementation matching your losses.py\nclass BCEDiceLoss(nn.Module):\n    def __init__(self, bce_weight=0.35, dice_weight=0.65):\n        super().__init__()\n        self.bw, self.dw = bce_weight, dice_weight\n    def forward(self, logits, targets):\n        bce = F.binary_cross_entropy_with_logits(logits, targets.float())\n        dice = 1 - dice_with_logits(logits, targets)\n        return self.bw * bce + self.dw * dice\n\nclass AquaBoundaryLoss(nn.Module):\n    def __init__(self, w_aqua=1.0, w_boundary=0.35):\n        super().__init__()\n        self.core = BCEDiceLoss()\n        self.w_aqua, self.w_boundary = w_aqua, w_boundary\n    def forward(self, logits, targets):\n        la = self.core(logits[:, 0:1], targets[:, 0:1])\n        lb = self.core(logits[:, 1:2], targets[:, 1:2])\n        return self.w_aqua * la + self.w_boundary * lb\n\nloss_fn = AquaBoundaryLoss()\nlogits = torch.randn(2, 2, 64, 64)\ntargets = torch.zeros(2, 2, 64, 64)\ntargets[:, 0, 10:50, 10:50] = 1.0\ntargets[:, 1, 9:51, 9:51] = 1.0\nprint(f'AquaBoundaryLoss: {loss_fn(logits, targets):.4f}')"),
    md("## water-bodies-detection Pipeline Preview\n\n```\n1. tile_and_mask.py   → Planet GeoTIFF + shapefile → tiles + dual masks\n2. dataset.py         → Albumentations augmentation\n3. model.py           → UNet++ SE-ResNet50 (Module 06+08)\n4. losses.py          → AquaBoundaryLoss (this notebook)\n5. train.py           → Two-stage training, early stopping on val IoU\n6. predict.py         → Sliding window + TTA + Hann blending\n7. post_process/      → Threshold → polygons → shapefile\n```\n\n**Full walkthrough:** Module 12 Capstone"),
    md("## Module 07 Metrics Summary\n\n| Metric | Formula | Your project |\n|--------|---------|-------------|\n| IoU | intersection/union | Early stopping metric |\n| Dice | 2×inter/(sum) | Part of loss |\n| Pixel Acc | correct/total | Monitoring |\n| mIoU | mean IoU classes | Multi-class extension |"),
    md("## Exercise\n\nTrain UNet from Notebook 07 on synthetic data with BCEDiceLoss. Plot IoU vs epoch."),
    code("# YOUR CODE HERE\n"),
    md("## Module 07 Complete\n\n**Next:** Module 08 Object Detection — R-CNN, YOLO, DETR."),
    footer("AquaBoundaryLoss combines BCE+Dice on dual heads — the heart of your production pipeline.", None),
])


def main():
    print("Building Module 07 notebooks...")
    for name, cells in NOTEBOOKS:
        save(name, cells)
    print(f"Done: {len(NOTEBOOKS)} notebooks created.")


if __name__ == "__main__":
    main()
