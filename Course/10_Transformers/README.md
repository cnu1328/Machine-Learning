# Module 10 — Transformers for Vision

**Duration:** 4 weeks  
**Prerequisites:** Module 05 (deep learning) and Module 06  
**Status:** Ready

---

## Overview

Transformers revolutionized NLP and now dominate computer vision. Full mathematical treatment of attention, then every major vision transformer architecture.

**Framework:** NumPy (attention from scratch) + PyTorch (MHA, ViT, encoder blocks)

---

## Study Plan (4 weeks at 5–8 hrs/week)

| Week | Notebooks | Focus |
|------|-----------|-------|
| 1 | 01–05 | Attention math, MHA, positional encoding |
| 2 | 06–08 | ViT, Swin, SegFormer |
| 3 | 09–11 | Mask2Former, DINO, CLIP |
| 4 | 12–13 | SAM, Grounding DINO, capstone |

---

## Part 1: Attention Foundations (Notebooks 01–05)

| # | Notebook | Topic |
|---|----------|-------|
| 01 | [01_Attention_Intuition.ipynb](01_Attention_Intuition.ipynb) | Q, K, V analogy |
| 02 | [02_Scaled_Dot_Product_Attention.ipynb](02_Scaled_Dot_Product_Attention.ipynb) | Full derivation + NumPy |
| 03 | [03_Self_Attention.ipynb](03_Self_Attention.ipynb) | Matrix dimensions |
| 04 | [04_Multi_Head_Attention.ipynb](04_Multi_Head_Attention.ipynb) | MHA + encoder block |
| 05 | [05_Positional_Encoding.ipynb](05_Positional_Encoding.ipynb) | Sinusoidal and learned |

---

## Part 2: Vision Transformers (Notebooks 06–13)

| # | Notebook | Architecture | Key Innovation |
|---|----------|--------------|----------------|
| 06 | [06_Vision_Transformer.ipynb](06_Vision_Transformer.ipynb) | ViT | Image patches as tokens |
| 07 | [07_Swin_Transformer.ipynb](07_Swin_Transformer.ipynb) | Swin | Hierarchical + shifted windows |
| 08 | [08_SegFormer.ipynb](08_SegFormer.ipynb) | SegFormer | Efficient transformer segmentation |
| 09 | [09_Mask2Former.ipynb](09_Mask2Former.ipynb) | Mask2Former | Mask classification transformer |
| 10 | [10_DINO.ipynb](10_DINO.ipynb) | DINO | Self-supervised ViT |
| 11 | [11_CLIP.ipynb](11_CLIP.ipynb) | CLIP | Vision-language pretraining |
| 12 | [12_SAM.ipynb](12_SAM.ipynb) | SAM | Promptable segmentation |
| 13 | [13_GroundingDINO.ipynb](13_GroundingDINO.ipynb) | Grounding DINO | Open-vocabulary detection |

---

## Implementations (from scratch)

| Component | Notebook |
|-----------|----------|
| Scaled dot-product attention | 02 (NumPy) |
| Multi-head attention | 04 (PyTorch) |
| Transformer encoder block | 04 (PyTorch) |
| ViT patch embedding | 06 (PyTorch) |

---

## Connection to Your Work

| Model | GeoSpatial Use |
|-------|----------------|
| **SegFormer** | Modern segmentation alternative to UNet++ |
| **SAM** | Zero-shot segmentation for rapid labeling |
| **CLIP** | Text-prompted exploration ("find aquaculture ponds") |
| **DINO** | Self-supervised pretraining on unlabeled satellite imagery |
| **Grounding DINO** | Open-vocabulary detection without retraining |

**Production recommendation:** Keep UNet++ for aquaculture inference; use transformers for labeling (SAM), pretraining (DINO), and exploration (CLIP/Grounding DINO).

---

## Module Deliverables

- [ ] All 13 notebooks completed
- [ ] Attention + MHA implemented (Exercises 1–4)
- [ ] ViT patch embedding (Exercise 6)
- [ ] Assignment: transformer components notebook
- [ ] Quiz ≥16/20 (80%)

---

## Assignment

**Transformer Components** — implement attention, MHA, encoder block, and patch embedding; optional CLIP/SAM demo.

See [exercises/README.md](exercises/README.md).

---

## Optional Dependencies

```bash
pip install timm                    # pretrained ViT
pip install git+https://github.com/openai/CLIP.git
pip install segment-anything        # SAM
```

Core module runs with NumPy + PyTorch only.

---

## Resources

- [CHEATSHEET.md](CHEATSHEET.md)
- [quiz/module_10_quiz.md](quiz/module_10_quiz.md)
- [exercises/README.md](exercises/README.md)

---

**Previous:** [09_Instance_Segmentation/](../09_Instance_Segmentation/)  
**Next:** [11_Production_ML/](../11_Production_ML/)
