# Module 07 Cheat Sheet — Segmentation

## Segmentation Types

| Type | Output | Activation | Loss |
|------|--------|------------|------|
| Binary | (1, H, W) | Sigmoid | BCE + Dice |
| Multi-class | (K, H, W) | Softmax | Cross-entropy |
| Multi-label | (C, H, W) | Sigmoid per ch | BCE per channel |
| Instance | N masks + IDs | Mask R-CNN | Module 09 |
| Panoptic | Semantic + instance | Combined | Module 09 |

## Architectures

| Model | Key Idea | Your Use |
|-------|----------|----------|
| FCN | Conv replaces FC | Historical |
| UNet | Encoder-decoder + skips | Road/building projects |
| UNet++ | Nested dense skips | **water-bodies** |
| DeepLab | Atrous + ASPP | Multi-scale objects |
| PSPNet | Pyramid pooling | Global context |
| HRNet | High-res parallel streams | Fine boundaries |
| SegFormer | Transformer + MLP | Modern land cover |
| SAM | Promptable zero-shot | Rapid labeling |

## Loss Functions

| Loss | Formula | When |
|------|---------|------|
| BCE | $-[y\log p + (1-y)\log(1-p)]$ | Pixel-wise binary |
| Dice | $1 - 2|A∩B|/(|A|+|B|)$ | Imbalanced regions |
| IoU | $1 - |A∩B|/|A∪B|$ | Direct metric opt |
| Focal | $(1-p_t)^γ CE$ | Hard examples |
| **AquaBoundary** | $w_a·BCEDice_{aqua} + w_b·BCEDice_{bund}$ | **Your project** |

## Metrics

```python
# Binary IoU
inter = (pred & target).sum()
iou = inter / (pred.sum() + target.sum() - inter)

# Dice
dice = 2*inter / (pred.sum() + target.sum())
```

## Your water-bodies Pipeline

```
GeoTIFF + shapefile
  → tile_and_mask.py (6 bands, dual masks)
  → UNet++ SE-ResNet50 (model.py)
  → AquaBoundaryLoss (losses.py)
  → train.py (2-stage, early stop on IoU)
  → predict.py (sliding window + TTA)
  → post_process (polygons)
```

## PyTorch Segmentation Template

```python
import segmentation_models_pytorch as smp

model = smp.UnetPlusPlus(
    encoder_name='se_resnet50',
    encoder_weights='imagenet',
    in_channels=6,
    classes=2,
    activation=None,
)
logits = model(image)  # (N, 2, H, W)
probs = torch.sigmoid(logits)
```

## Train vs Inference Thresholds

| Stage | Aqua threshold | Notes |
|-------|---------------|-------|
| Training loss | logits (no thresh) | BCE+Dice on logits |
| Validation IoU | 0.5 | Model selection |
| Inference raster | 0.5 | Probability GeoTIFF |
| Post-process GIS | 0.8 | High-precision polygons |

## Common Mistakes

- Using softmax for multi-label (use sigmoid)
- Wrong loss for imbalanced tiles (add Dice)
- Forgetting `model.eval()` at inference
- Merging adjacent ponds without boundary head
- Evaluating with wrong threshold
