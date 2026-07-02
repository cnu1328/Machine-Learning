# Module 12 — Capstone: water-bodies-detection Complete Walkthrough

**Duration:** 4 weeks  
**Prerequisites:** Modules 01–11 complete  
**Status:** Ready

---

## Overview

Line-by-line explanation of every file in your [water-bodies-detection](../../water-bodies-detection/) repository. Assumes you wrote none of it. By the end, you can maintain, extend, and rebuild this pipeline independently.

**We do NOT rewrite the project.** We explain it until you own every line.

---

## Study Plan (4 weeks at 5–8 hrs/week)

| Week | Notebooks | Focus |
|------|-----------|-------|
| 1 | 01–04 | Overview, config, tiling, GIS |
| 2 | 05–08 | Dataset, model, losses, dual-head design |
| 3 | 09–12 | Training loop, optimizer, metrics, artifacts |
| 4 | 13–18 | Inference, post-process, automate, Docker, E2E |

---

## Week 1: Data Preparation & Configuration

| # | Notebook | File(s) | Topics |
|---|----------|---------|--------|
| 01 | [01_Project_Overview.ipynb](01_Project_Overview.ipynb) | README.md | Problem, dual-head rationale, architecture |
| 02 | [02_Configuration_Deep_Dive.ipynb](02_Configuration_Deep_Dive.ipynb) | `config/default.yaml` | Every parameter explained |
| 03 | [03_Tiling_Pipeline.ipynb](03_Tiling_Pipeline.ipynb) | `tile_and_mask.py` | Bands, normalization, dual masks |
| 04 | [04_GIS_Fundamentals.ipynb](04_GIS_Fundamentals.ipynb) | rasterio, geopandas | CRS, GSD, rasterize |

---

## Week 2: Model & Loss

| # | Notebook | File(s) | Topics |
|---|----------|---------|--------|
| 05 | [05_Dataset_Class.ipynb](05_Dataset_Class.ipynb) | `dataset.py` | WaterBodyTileDataset, Albumentations |
| 06 | [06_Model_Architecture.ipynb](06_Model_Architecture.ipynb) | `model.py` | UNet++, SE-ResNet50, SCSE |
| 07 | [07_Loss_Functions.ipynb](07_Loss_Functions.ipynb) | `losses.py` | BCEDiceLoss, AquaBoundaryLoss |
| 08 | [08_Why_Dual_Heads.ipynb](08_Why_Dual_Heads.ipynb) | design docs | Adjacent pond problem |

---

## Week 3: Training Pipeline

| # | Notebook | File(s) | Topics |
|---|----------|---------|--------|
| 09 | [09_Training_Loop.ipynb](09_Training_Loop.ipynb) | `train.py` | Two-stage, AMP, early stopping |
| 10 | [10_Optimizer_and_Scheduler.ipynb](10_Optimizer_and_Scheduler.ipynb) | `train.py` | AdamW, WarmupCosine |
| 11 | [11_Metrics_and_Validation.ipynb](11_Metrics_and_Validation.ipynb) | `train.py` | IoU, checkpoint selection |
| 12 | [12_Checkpoints_and_Artifacts.ipynb](12_Checkpoints_and_Artifacts.ipynb) | `train.py` | best.pt, model_meta.json |

---

## Week 4: Inference, Post-Processing & Deployment

| # | Notebook | File(s) | Topics |
|---|----------|---------|--------|
| 13 | [13_Sliding_Window_Inference.ipynb](13_Sliding_Window_Inference.ipynb) | `predict.py` | Hann blend, TTA d4 |
| 14 | [14_Probability_Rasters.ipynb](14_Probability_Rasters.ipynb) | `predict.py` | Thresholds, GeoTIFF output |
| 15 | [15_GIS_Post_Processing.ipynb](15_GIS_Post_Processing.ipynb) | `post_process_aqua_boundary.py` | Polygons, morphology |
| 16 | [16_Batch_Automation.ipynb](16_Batch_Automation.ipynb) | `automate/` | Batch predict + post-process |
| 17 | [17_Docker_Deployment.ipynb](17_Docker_Deployment.ipynb) | `Dockerfile` | GDAL + PyTorch container |
| 18 | [18_End_to_End_Pipeline.ipynb](18_End_to_End_Pipeline.ipynb) | all files | Full pipeline + graduation |

---

## Repository Architecture

```
water-bodies-detection/
├── config/default.yaml
├── tile_and_mask.py
├── dataset.py
├── model.py
├── losses.py
├── train.py
├── predict.py
├── post_process/
├── automate/
├── Dockerfile
└── requirements.txt
```

---

## Module Deliverables

- [ ] All 18 notebooks completed (with repo open side-by-side)
- [ ] Capstone pipeline report (full or smoke-test documented)
- [ ] One extension project attempted
- [ ] Quiz ≥20/25 (80%)
- [ ] Graduation checklist passed

---

## Final Assessment

You are ready to graduate when you can:

- [ ] Explain the dual-head design without looking at notes
- [ ] Walk through `train.py` line by line from memory
- [ ] Debug a training run that stalls at IoU = 0
- [ ] Modify `tile_and_mask.py` for a new band configuration
- [ ] Deploy inference in Docker and process a batch of GeoTIFFs
- [ ] Design a similar pipeline for a new GeoSpatial task from scratch

---

## Resources

- [CHEATSHEET.md](CHEATSHEET.md)
- [quiz/module_12_quiz.md](quiz/module_12_quiz.md)
- [exercises/README.md](exercises/README.md)
- [water-bodies-detection README](../../water-bodies-detection/README.md)

---

**Previous:** [11_Production_ML/](../11_Production_ML/)  
**Course Complete!** → [Course README](../README.md)
