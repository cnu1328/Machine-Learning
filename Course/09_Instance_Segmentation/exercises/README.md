# Module 09 Exercises

Attempt before checking [solutions/](solutions/).

---

## Concepts (Notebook 01)

### Exercise 1 — Paradigm Comparison
Create a diagram (matplotlib or paper) showing semantic vs instance vs panoptic output for an image with 2 adjacent ponds + a road.

### Exercise 2 — Adjacent Pond Analysis
Write 200 words comparing dual-head boundary (Module 07) vs Mask R-CNN for your aquaculture use case. Include labeling cost, inference speed, and GIS output quality.

---

## Mask R-CNN (Notebook 02)

### Exercise 3 — Mask IoU from Scratch
Implement `mask_iou` for (N,H,W) binary masks. Test on partial overlap, disjoint, and identical masks.

### Exercise 4 — Mask R-CNN Inference
Run torchvision `maskrcnn_resnet50_fpn` with pretrained weights on a sample image. Visualize top-3 instance masks with distinct colors.

---

## Real-Time Instance Seg (Notebooks 03–04)

### Exercise 5 — YOLACT Assembly
Given 4 prototype masks (32×32) and random coefficients, assemble 3 instance masks. Visualize side by side.

### Exercise 6 — SOLO Grid Assignment
Given 8×8 grid and 5 object centers, assign each object to responsible grid cell. Plot grid with object centers marked.

---

## Panoptic (Notebooks 05–06)

### Exercise 7 — PQ Calculator
Implement Panoptic Quality for one class given TP/FP/FN counts and matched IoUs. Verify against manual calculation.

### Exercise 8 — Panoptic Merge
Given semantic map + 3 instance masks with scores, implement confidence-ordered merge. Visualize final panoptic map.

---

## Capstone (Notebook 07)

### Exercise 9 — Connected Components vs Mask R-CNN
On synthetic adjacent circles:
1. Binary semantic mask → connected components
2. Compare instance count to ground truth
3. Add boundary ring → watershed split → recount
4. Document when CC matches Mask R-CNN quality

---

## Module Assignment: Instance Labeling Pipeline

**Deliverable:** `exercises/assignment_instance_segmentation.ipynb`

Build an instance labeling pipeline without full Mask R-CNN training:

1. **Synthetic data:** 5–10 overlapping/adjacent circles on 256×256 canvas
2. **Semantic mask:** all circles = foreground
3. **Boundary mask:** thin ring between adjacent circles
4. **Method A:** Connected components on semantic only (expect merges)
5. **Method B:** Watershed using boundary as marker/separator
6. **Method C (optional):** Run pretrained Mask R-CNN, count instances
7. **Metrics:** instance count accuracy, mean mask IoU per instance
8. **Report:** When does dual-head + watershed beat semantic-only? When would Mask R-CNN be required?

**Targets:**
- Method B correctly separates ≥90% of adjacent pairs
- Clear comparison table of three approaches
- 1-page architectural recommendation for water-bodies-detection

---

## GeoSpatial Extension

Design a panoptic labeling schema for an aquaculture tile: which classes are stuff vs things? How would you encode labels for training Panoptic FPN?

---

## Submission

> Module 09 complete. Assignment attached. Quiz score: X/15.
