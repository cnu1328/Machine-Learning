# Module 11 Exercises

Attempt before checking [solutions/](solutions/).

---

## Data & Training Pipelines (Notebooks 01–05)

### Exercise 1 — Pipeline Diagram (Notebook 01)
Draw the water-bodies data pipeline from GeoTIFF to DataLoader. Mark validation checkpoints at each stage.

### Exercise 2 — Config Parser (Notebook 02)
Write a function that loads `config/default.yaml` (or mock YAML) and validates required keys exist in `data`, `model`, `training` sections.

### Exercise 3 — MLflow Integration (Notebook 03)
Add MLflow logging to a 5-epoch training loop. Log lr, batch_size, and val_iou per epoch.

### Exercise 4 — TensorBoard (Notebook 05)
Log train loss and val IoU for 20 fake epochs. Open TensorBoard and screenshot the curves.

---

## Model Optimization (Notebooks 06–12)

### Exercise 5 — AMP Training (Notebook 06)
Convert a simple training loop to use `autocast` + `GradScaler`. Compare step time with/without AMP on GPU.

### Exercise 6 — ONNX Benchmark (Notebook 08)
Export a small conv model to ONNX. Compare PyTorch vs ONNX Runtime latency for 100 inferences.

### Exercise 7 — Quantization Check (Notebook 10)
Apply dynamic quantization to a linear model. Measure size reduction and output difference.

---

## Deployment (Notebooks 13–17)

### Exercise 8 — Docker Build (Notebook 13)
Write a minimal Dockerfile for a Python ML app with rasterio. List required apt packages.

### Exercise 9 — FastAPI Stub (Notebook 14)
Create FastAPI app with `/health` and `/predict` (returns mock JSON). Test with curl.

### Exercise 10 — Batch Job Tracker (Notebook 17)
Implement batch processor that logs success/failure per file, supports resume (skip completed).

---

## Monitoring & CI/CD (Notebooks 18–22)

### Exercise 11 — PSI Drift (Notebook 19)
Compute PSI between training and "production" band distributions. Flag if PSI > 0.25.

### Exercise 12 — CI Workflow (Notebook 21)
Write GitHub Actions YAML with: install deps, pytest, smoke `predict.py --help`.

---

## Module Assignment: Production Readiness Pack

**Deliverable:** `exercises/assignment_production_ml.ipynb`

Prepare a production readiness pack for water-bodies-detection (or any Module 05+ model):

1. **Config:** Document all sections of `config/default.yaml` with purpose of each key
2. **Tracking:** MLflow or TensorBoard integration in mini training loop
3. **Export:** ONNX export + latency benchmark (PyTorch vs ONNX)
4. **API:** FastAPI `/health` + `/predict` stub (file upload optional)
5. **Docker:** Dockerfile snippet with GDAL deps documented
6. **Batch:** Pseudocode or mini script for folder batch inference with logging
7. **Monitoring:** PSI drift check on synthetic band shift
8. **CI/CD:** GitHub Actions workflow sketch with metric gate
9. **Checklist:** Production readiness checklist (≥15 items) with pass/fail for your current water-bodies repo

**Targets:**
- ONNX max diff < 1e-4 vs PyTorch on dummy input
- FastAPI health endpoint returns 200
- Clear rollback strategy documented

---

## GeoSpatial Extension

Audit `water-bodies-detection/` against the Module 11 production stack table. List what exists, what's missing, and priority order to add.

---

## Submission

> Module 11 complete. Assignment attached. Quiz score: X/20.
