# Module 07 — Semantic Segmentation

**Duration:** 5–6 weeks  
**Prerequisites:** Module 06 complete  
**Status:** Ready

---

## Overview

The core of your GeoSpatial AI work. Covers every segmentation type, architecture, loss function, and metric — culminating in an introduction to your **water-bodies-detection** project.

**Framework:** PyTorch + `segmentation-models-pytorch` (optional for UNet++ notebook)

---

## Study Plan (5–6 weeks at 5–8 hrs/week)

| Week | Notebooks | Focus |
|------|-----------|-------|
| 1 | 01–05 | Segmentation types, boundary detection |
| 2–3 | 06–10 | FCN, UNet, UNet++, DeepLab, PSPNet |
| 4 | 11–14 | HRNet, SegFormer, Mask2Former, SAM |
| 5–6 | 15–20 | Losses, AquaBoundaryLoss capstone bridge |

---

## Part 1: Segmentation Types (Notebooks 01–05)

| # | Notebook | Topic |
|---|----------|-------|
| 01 | [01_Binary_Segmentation.ipynb](01_Binary_Segmentation.ipynb) | One class vs background |
| 02 | [02_Multi_Class_Segmentation.ipynb](02_Multi_Class_Segmentation.ipynb) | Mutually exclusive classes |
| 03 | [03_Multi_Label_Segmentation.ipynb](03_Multi_Label_Segmentation.ipynb) | Independent labels per pixel |
| 04 | [04_Segmentation_Types_Compared.ipynb](04_Segmentation_Types_Compared.ipynb) | Semantic vs instance vs panoptic |
| 05 | [05_Boundary_Detection.ipynb](05_Boundary_Detection.ipynb) | Edge/boundary prediction |

---

## Part 2: Architectures (Notebooks 06–14)

| # | Notebook | Architecture | Your Use |
|---|----------|--------------|----------|
| 06 | [06_FCN.ipynb](06_FCN.ipynb) | Fully Convolutional Networks | Historical baseline |
| 07 | [07_UNet.ipynb](07_UNet.ipynb) | Encoder-decoder + skips | Road/building projects |
| 08 | [08_UNetPlusPlus.ipynb](08_UNetPlusPlus.ipynb) | Nested skip pathways | **water-bodies-detection** |
| 09 | [09_DeepLab.ipynb](09_DeepLab.ipynb) | Atrous conv + ASPP | Multi-scale objects |
| 10 | [10_PSPNet.ipynb](10_PSPNet.ipynb) | Pyramid pooling | Global context |
| 11 | [11_HRNet.ipynb](11_HRNet.ipynb) | High-resolution streams | Fine boundaries |
| 12 | [12_SegFormer.ipynb](12_SegFormer.ipynb) | Transformer encoder | Modern land cover |
| 13 | [13_Mask2Former.ipynb](13_Mask2Former.ipynb) | Mask classification | COCO-style |
| 14 | [14_SAM.ipynb](14_SAM.ipynb) | Segment Anything | Zero-shot labeling |

---

## Part 3: Loss Functions (Notebooks 15–20)

| # | Notebook | Loss | When to Use |
|---|----------|------|-------------|
| 15 | [15_Cross_Entropy_Loss.ipynb](15_Cross_Entropy_Loss.ipynb) | Cross-entropy | Multi-class default |
| 16 | [16_Dice_Loss.ipynb](16_Dice_Loss.ipynb) | Dice | Class imbalance |
| 17 | [17_IoU_Loss.ipynb](17_IoU_Loss.ipynb) | IoU | Direct metric optimization |
| 18 | [18_Focal_Loss.ipynb](18_Focal_Loss.ipynb) | Focal | Hard examples |
| 19 | [19_Boundary_Loss.ipynb](19_Boundary_Loss.ipynb) | Boundary-weighted | Edge accuracy |
| 20 | [20_AquaBoundaryLoss_and_Capstone.ipynb](20_AquaBoundaryLoss_and_Capstone.ipynb) | **AquaBoundaryLoss** | **Your project** |

---

## Connection to Your Work

```
water-bodies-detection:
  GeoTIFF (6 bands) + shapefile labels
    → tile_and_mask.py (dual masks: aqua + bund)
    → UNet++ SE-ResNet50 (model.py)     ← Module 07 Notebook 08
    → AquaBoundaryLoss (losses.py)      ← Module 07 Notebook 20
    → train.py → predict.py → post_process
```

Dual-head design (aqua interior + bund boundary) prevents adjacent pond merging — covered in Notebooks 05 and 20.

Full walkthrough in Module 12.

---

## Module Deliverables

- [ ] All 20 notebooks completed
- [ ] mIoU implemented from scratch (Exercise 2)
- [ ] UNet trained on synthetic data (Exercise 4)
- [ ] AquaBoundaryLoss reimplemented (Exercise 8)
- [ ] Assignment: binary segmentation pipeline with dual heads
- [ ] Quiz ≥16/20 (80%)

---

## Assignment

**Binary Segmentation Pipeline** — mini water-bodies pipeline with dual-head UNet++, BCEDice/AquaBoundary loss, sliding-window inference, and pond counting.

See [exercises/README.md](exercises/README.md).

---

## Resources

- [CHEATSHEET.md](CHEATSHEET.md)
- [quiz/module_07_quiz.md](quiz/module_07_quiz.md)
- [exercises/README.md](exercises/README.md)

---

**Previous:** [06_CNN/](../06_CNN/)  
**Next:** [08_Object_Detection/](../08_Object_Detection/)
