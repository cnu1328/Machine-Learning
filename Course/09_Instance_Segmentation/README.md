# Module 09 — Instance Segmentation

**Duration:** 3 weeks  
**Prerequisites:** Module 07 and 08  
**Status:** Ready

---

## Overview

Combine detection and segmentation: identify each object instance with a pixel-level mask. Bridges semantic segmentation (Module 07) and object detection (Module 08).

**Framework:** PyTorch + torchvision detection (Mask R-CNN)

---

## Study Plan (3 weeks at 5–8 hrs/week)

| Week | Notebooks | Focus |
|------|-----------|-------|
| 1 | 01–02 | Paradigms, Mask R-CNN, mask IoU |
| 2 | 03–05 | YOLACT, SOLO, panoptic + PQ metric |
| 3 | 06–07 | Panoptic FPN, Mask2Former, capstone |

---

## Notebooks

| # | Notebook | Topic |
|---|----------|-------|
| 01 | [01_Instance_vs_Semantic_vs_Panoptic.ipynb](01_Instance_vs_Semantic_vs_Panoptic.ipynb) | Conceptual comparison |
| 02 | [02_Mask_R_CNN.ipynb](02_Mask_R_CNN.ipynb) | Faster R-CNN + mask head |
| 03 | [03_YOLACT.ipynb](03_YOLACT.ipynb) | Real-time instance segmentation |
| 04 | [04_SOLO_SOLOv2.ipynb](04_SOLO_SOLOv2.ipynb) | Direct instance segmentation |
| 05 | [05_Panoptic_Segmentation.ipynb](05_Panoptic_Segmentation.ipynb) | Semantic + instance unified |
| 06 | [06_Panoptic_FPN.ipynb](06_Panoptic_FPN.ipynb) | Multi-task heads + merge |
| 07 | [07_Mask2Former.ipynb](07_Mask2Former.ipynb) | Universal segmentation + capstone |

---

## Key Concepts

- Instance masks vs semantic masks
- Mask IoU evaluation
- Panoptic Quality (PQ) metric
- Handling overlapping instances
- Class-agnostic vs class-specific masks

---

## Connection to water-bodies-detection

Your project's **dual-head boundary design** is an alternative approach to the adjacent-pond problem that instance segmentation also solves. Module 09 compares both strategies:

| Approach | How it separates adjacent ponds |
|----------|--------------------------------|
| **Your approach (Mod 07)** | Learn bund boundaries explicitly (boundary head) |
| **Instance seg (Mod 09)** | Separate instances with Mask R-CNN |
| **Hybrid** | UNet++ boundaries + connected components / watershed |

Understanding both makes you a stronger architect. Notebook 07 includes the full decision guide.

---

## Module Deliverables

- [ ] All 7 notebooks completed
- [ ] Mask IoU implemented (Exercise 3)
- [ ] Mask R-CNN inference demo (Exercise 4)
- [ ] Assignment: instance labeling pipeline (watershed vs CC vs Mask R-CNN)
- [ ] Quiz ≥12/15 (80%)

---

## Assignment

**Instance Labeling Pipeline** — separate adjacent synthetic objects using watershed on dual-head masks; compare to Mask R-CNN and connected components.

See [exercises/README.md](exercises/README.md).

---

## Resources

- [CHEATSHEET.md](CHEATSHEET.md)
- [quiz/module_09_quiz.md](quiz/module_09_quiz.md)
- [exercises/README.md](exercises/README.md)

---

**Previous:** [08_Object_Detection/](../08_Object_Detection/)  
**Next:** [10_Transformers/](../10_Transformers/)
