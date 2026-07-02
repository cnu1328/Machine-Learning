# Module 08 — Object Detection

**Duration:** 4 weeks  
**Prerequisites:** Module 06 complete (Module 07 recommended)  
**Status:** Ready

---

## Overview

Detect and localize objects with bounding boxes. From two-stage R-CNN family to single-stage YOLO and transformer-based DETR.

**Framework:** PyTorch + torchvision detection + Ultralytics YOLO (optional)

---

## Study Plan (4 weeks at 5–8 hrs/week)

| Week | Notebooks | Focus |
|------|-----------|-------|
| 1 | 01–03 | IoU, NMS, mAP, detection losses |
| 2 | 04–06 | R-CNN → Fast R-CNN → Faster R-CNN |
| 3 | 07–10 | SSD, YOLO, RetinaNet, YOLOv8/v11 |
| 4 | 11–12 | DETR, RT-DETR, paradigm comparison |

---

## Part 1: Foundations (Notebooks 01–03)

| # | Notebook | Topic |
|---|----------|-------|
| 01 | [01_Detection_Fundamentals.ipynb](01_Detection_Fundamentals.ipynb) | Box formats, IoU |
| 02 | [02_NMS_and_mAP.ipynb](02_NMS_and_mAP.ipynb) | NMS, precision-recall, mAP |
| 03 | [03_Detection_Loss_Functions.ipynb](03_Detection_Loss_Functions.ipynb) | Smooth L1, focal, anchor matching |

---

## Part 2: Two-Stage Detectors (Notebooks 04–06)

| # | Notebook | Architecture | Key Innovation |
|---|----------|--------------|----------------|
| 04 | [04_R_CNN.ipynb](04_R_CNN.ipynb) | R-CNN | Region proposals + CNN classifier |
| 05 | [05_Fast_R_CNN.ipynb](05_Fast_R_CNN.ipynb) | Fast R-CNN | Shared convolution + RoI Pooling |
| 06 | [06_Faster_R_CNN.ipynb](06_Faster_R_CNN.ipynb) | Faster R-CNN | Region Proposal Network (RPN) |

---

## Part 3: Single-Stage Detectors (Notebooks 07–10)

| # | Notebook | Architecture | Key Innovation |
|---|----------|--------------|----------------|
| 07 | [07_SSD.ipynb](07_SSD.ipynb) | SSD | Multi-scale default boxes |
| 08 | [08_YOLO_v1_v3.ipynb](08_YOLO_v1_v3.ipynb) | YOLO v1–v3 | Single pass, grid-based |
| 09 | [09_RetinaNet.ipynb](09_RetinaNet.ipynb) | RetinaNet | Focal loss + FPN |
| 10 | [10_YOLOv5_v8_v11.ipynb](10_YOLOv5_v8_v11.ipynb) | YOLOv5/v8/v11 | Ultralytics production stack |

---

## Part 4: Transformer Detectors (Notebooks 11–12)

| # | Notebook | Architecture | Key Innovation |
|---|----------|--------------|----------------|
| 11 | [11_DETR.ipynb](11_DETR.ipynb) | DETR | Set prediction with transformers |
| 12 | [12_RT_DETR.ipynb](12_RT_DETR.ipynb) | RT-DETR | Real-time DETR + module capstone |

---

## Connection to GeoSpatial

Object detection complements segmentation in your pipeline:

| Approach | Module | Output | Best For |
|----------|--------|--------|----------|
| Segmentation | 07 | Pixel masks | Pond boundaries, land cover |
| Detection | 08 | Bounding boxes | Building/ship/vehicle counts |

Combined → panoptic understanding (Module 09).

---

## Module Deliverables

- [ ] All 12 notebooks completed
- [ ] IoU and NMS implemented from scratch (Exercises 1–2)
- [ ] Faster R-CNN or YOLOv8 inference demo (Exercises 5, 10)
- [ ] Assignment: custom detector with mAP@0.5 > 0.5
- [ ] Quiz ≥16/20 (80%)

---

## Assignment

**Custom Object Detector** — train YOLOv8 or Faster R-CNN on ≥50 annotated objects, report mAP, speed, and failure cases.

See [exercises/README.md](exercises/README.md).

---

## Resources

- [CHEATSHEET.md](CHEATSHEET.md)
- [quiz/module_08_quiz.md](quiz/module_08_quiz.md)
- [exercises/README.md](exercises/README.md)

---

**Previous:** [07_Segmentation/](../07_Segmentation/)  
**Next:** [09_Instance_Segmentation/](../09_Instance_Segmentation/)
