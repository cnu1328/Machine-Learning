# Module 11 — Production Machine Learning

**Duration:** 4 weeks  
**Prerequisites:** Module 05+ (any trained model)  
**Status:** Ready

---

## Overview

Building models is 20% of the job. Production ML — pipelines, deployment, monitoring, optimization — is the other 80%. This module prepares you to ship models like your **water-bodies-detection** pipeline.

---

## Study Plan (4 weeks at 5–8 hrs/week)

| Week | Notebooks | Focus |
|------|-----------|-------|
| 1 | 01–05 | Data pipelines, config, tracking, DVC, TensorBoard |
| 2 | 06–12 | AMP, DDP, ONNX, TensorRT, quant, prune, distill |
| 3 | 13–17 | Docker, FastAPI, Triton, versioning, batch inference |
| 4 | 18–22 | Monitoring, drift, CI/CD, GPU optimization |

---

## Part 1: Data & Training Pipelines (Notebooks 01–05)

| # | Notebook | Topic |
|---|----------|-------|
| 01 | [01_Data_Pipeline_Design.ipynb](01_Data_Pipeline_Design.ipynb) | ETL, GeoSpatial tiling |
| 02 | [02_Training_Pipeline_Architecture.ipynb](02_Training_Pipeline_Architecture.ipynb) | YAML config-driven training |
| 03 | [03_Experiment_Tracking.ipynb](03_Experiment_Tracking.ipynb) | MLflow, W&B |
| 04 | [04_Data_Version_Control.ipynb](04_Data_Version_Control.ipynb) | DVC pipelines |
| 05 | [05_TensorBoard.ipynb](05_TensorBoard.ipynb) | Loss/metric visualization |

---

## Part 2: Model Optimization (Notebooks 06–12)

| # | Notebook | Topic |
|---|----------|-------|
| 06 | [06_Mixed_Precision_Training.ipynb](06_Mixed_Precision_Training.ipynb) | PyTorch AMP |
| 07 | [07_Distributed_Training.ipynb](07_Distributed_Training.ipynb) | DDP, multi-GPU |
| 08 | [08_ONNX_Export.ipynb](08_ONNX_Export.ipynb) | Cross-framework deploy |
| 09 | [09_TensorRT.ipynb](09_TensorRT.ipynb) | GPU inference optimization |
| 10 | [10_Quantization.ipynb](10_Quantization.ipynb) | INT8 quantization |
| 11 | [11_Pruning.ipynb](11_Pruning.ipynb) | Structured/unstructured |
| 12 | [12_Knowledge_Distillation.ipynb](12_Knowledge_Distillation.ipynb) | Teacher-student |

---

## Part 3: Deployment (Notebooks 13–17)

| # | Notebook | Topic |
|---|----------|-------|
| 13 | [13_Docker_for_ML.ipynb](13_Docker_for_ML.ipynb) | water-bodies Dockerfile |
| 14 | [14_REST_API_FastAPI.ipynb](14_REST_API_FastAPI.ipynb) | REST inference API |
| 15 | [15_Triton_Inference_Server.ipynb](15_Triton_Inference_Server.ipynb) | NVIDIA Triton |
| 16 | [16_Model_Versioning.ipynb](16_Model_Versioning.ipynb) | Registry patterns |
| 17 | [17_Batch_Inference.ipynb](17_Batch_Inference.ipynb) | automate/ scripts |

---

## Part 4: Monitoring & CI/CD (Notebooks 18–22)

| # | Notebook | Topic |
|---|----------|-------|
| 18 | [18_Logging_and_Monitoring.ipynb](18_Logging_and_Monitoring.ipynb) | Structured logging |
| 19 | [19_Data_Drift_Detection.ipynb](19_Data_Drift_Detection.ipynb) | PSI, drift triggers |
| 20 | [20_Model_Performance_Monitoring.ipynb](20_Model_Performance_Monitoring.ipynb) | Production QA |
| 21 | [21_CICD_for_ML.ipynb](21_CICD_for_ML.ipynb) | CI/CD pipelines |
| 22 | [22_GPU_Optimization_and_Capstone.ipynb](22_GPU_Optimization_and_Capstone.ipynb) | Full stack review |

---

## Case Study: water-bodies-detection

| Component | File | Production Pattern |
|-----------|------|-------------------|
| Config management | `config/default.yaml` | Single source of truth |
| Reproducibility | `model_meta.json`, `train_config.yaml` | Frozen experiment artifacts |
| Batch inference | `automate/automate_water_predictions.py` | Production batch processing |
| Containerization | `Dockerfile` | GDAL + PyTorch deployment |
| Post-processing | `post_process/post_process_aqua_boundary.py` | Domain-specific output |

---

## Module Deliverables

- [ ] All 22 notebooks completed
- [ ] MLflow or TensorBoard logging (Exercise 3–4)
- [ ] ONNX export + benchmark (Exercise 6)
- [ ] FastAPI stub (Exercise 9)
- [ ] Assignment: production readiness pack
- [ ] Quiz ≥16/20 (80%)

---

## Assignment

**Production Readiness Pack** — config audit, tracking, ONNX, FastAPI, Docker, batch, drift, CI/CD checklist for water-bodies.

See [exercises/README.md](exercises/README.md).

---

## Optional Dependencies

```bash
pip install mlflow tensorboard onnx onnxruntime fastapi uvicorn
```

Core module runs with PyTorch + standard library.

---

## Resources

- [CHEATSHEET.md](CHEATSHEET.md)
- [quiz/module_11_quiz.md](quiz/module_11_quiz.md)
- [exercises/README.md](exercises/README.md)

---

**Previous:** [10_Transformers/](../10_Transformers/)  
**Next:** [12_Capstone_water_bodies_detection/](../12_Capstone_water_bodies_detection/)
