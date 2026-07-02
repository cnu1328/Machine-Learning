# Module 12 Exercises — Capstone

Study alongside `water-bodies-detection/` repo. Open each source file while working through notebooks.

---

## Week 1: Data (Notebooks 01–04)

### Exercise 1 — Pipeline Diagram
Draw the full pipeline from GeoTIFF to shapefile. Label every file involved.

### Exercise 2 — Config Audit
Load `config/default.yaml`. For each of the 6 sections, write one sentence explaining its purpose. Change `boundary_width_meters` to 2.0 — predict the effect on masks.

### Exercise 3 — Normalization
Implement `robust_normalize_tile` logic on a synthetic 6-band array with outliers. Plot before/after.

### Exercise 4 — GIS Concepts
Explain in 100 words: CRS, GSD, and why `meters_per_pixel` matters for boundary dilation.

---

## Week 2: Model (Notebooks 05–08)

### Exercise 5 — Dataset Trace
List every step in `WaterBodyTileDataset.__getitem__`. What shapes enter and exit?

### Exercise 6 — Model Params
Count parameters in `build_water_model(in_channels=6, out_channels=2)`. Use `sum(p.numel() for p in model.parameters())`.

### Exercise 7 — Loss Reimplementation
Reimplement `AquaBoundaryLoss` from `losses.py` without looking. Unit test against original.

### Exercise 8 — Design Defense
Write 300 words defending dual-head vs single-head for adjacent aquaculture ponds. Include post-process logic.

---

## Week 3: Training (Notebooks 09–12)

### Exercise 9 — train.py Walkthrough
Annotate `train.py` with comments on every major block (config load, stage 1, stage 2, checkpoint).

### Exercise 10 — LR Schedule Plot
Plot WarmupCosine for both stages using config defaults.

### Exercise 11 — Metric Computation
Given synthetic pred/gt masks, compute IoU, Dice, precision, recall by hand and verify with code.

### Exercise 12 — Artifact Inspection
After a training run (or mock), explain every field in `model_meta.json`.

---

## Week 4: Inference & Deploy (Notebooks 13–18)

### Exercise 13 — Hann Blend
Implement and visualize 2D Hann blend weights. Explain why center-weighted blending reduces seams.

### Exercise 14 — Threshold Analysis
On a probability map, compare polygon count at thresholds 0.5, 0.7, 0.8, 0.9. Document precision/recall tradeoff.

### Exercise 15 — Post-Process Trace
Write pseudocode for `post_process_aqua_boundary.py` main logic from memory.

### Exercise 16 — Docker Build
Build the Docker image (or document steps). List all apt packages and why each is needed.

---

## Capstone Assignment: Full Pipeline Execution

**Deliverable:** `exercises/capstone_pipeline_report.md`

Execute (or thoroughly simulate with documented commands) the full pipeline:

1. **Data prep:** Document tile_and_mask.py invocation with your data paths
2. **Training:** Run train.py (≥1 epoch smoke test acceptable if no GPU data)
3. **Inference:** Run predict.py on one mosaic (or document expected outputs)
4. **Post-process:** Run post_process to shapefile
5. **Report sections:**
   - Architecture decision summary (dual-head rationale)
   - Config changes from defaults (if any) and why
   - Training curves / final val IoU
   - Sample prediction visualization description
   - Post-process parameter tuning notes
   - Production deployment plan (Docker + batch automate)
   - 3 extension ideas with implementation steps

---

## Extension Projects (Choose One)

| Project | Files to Modify |
|---------|----------------|
| Add NDVI band (band 10) | config, tile_and_mask, model in_channels |
| MLflow integration | train.py |
| FastAPI deploy | new api.py wrapping predict.py |
| DeepLab swap | model.py only |
| New region training | full pipeline rerun |

---

## Final Assessment (Self-Check)

- [ ] Explain dual-head design without notes
- [ ] Walk through train.py line by line from memory
- [ ] Debug training run stalled at IoU = 0
- [ ] Modify tile_and_mask.py for new band configuration
- [ ] Deploy inference in Docker and process batch GeoTIFFs
- [ ] Design similar pipeline for new GeoSpatial task

---

## Submission

> Module 12 complete. Capstone report attached. Quiz score: X/25. **Course complete.**
