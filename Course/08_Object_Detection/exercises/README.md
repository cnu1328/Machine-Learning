# Module 08 Exercises

Attempt before checking [solutions/](solutions/).

---

## Foundations (Notebooks 01–03)

### Exercise 1 — IoU from Scratch (Notebook 01)
Implement `box_iou` for xyxy boxes. Test on overlapping, disjoint, and identical boxes. Verify against torchvision.ops.box_iou if available.

### Exercise 2 — NMS Implementation (Notebook 02)
Complete greedy NMS. Visualize before/after on image with 10 overlapping predictions.

### Exercise 3 — Smooth L1 (Notebook 03)
Plot Smooth L1 vs L1 vs L2 for errors in [-3, 3]. Explain why Huber is used for bbox regression.

---

## Two-Stage Detectors (Notebooks 04–06)

### Exercise 4 — RoI Pooling (Notebook 05)
Given 32×32 feature map and 3 proposals, manually compute RoI pool 7×7 output for one region.

### Exercise 5 — Faster R-CNN Inference (Notebook 06)
Run torchvision `fasterrcnn_resnet50_fpn` on a sample image. Plot boxes with matplotlib. Count detections above conf=0.5.

---

## Single-Stage Detectors (Notebooks 07–10)

### Exercise 6 — Anchor Count (Notebook 07)
For SSD with 6 feature maps (38², 19², 10², 5², 3², 1²) and 4 anchors per location, compute total anchors per image.

### Exercise 7 — YOLO Grid Decode (Notebook 08)
Given 7×7 grid prediction tensor, decode all boxes with conf > 0.5 and plot on 448×448 canvas.

### Exercise 8 — Focal vs CE (Notebook 09)
On 1000:10 imbalanced anchor labels, compare CE vs focal loss values. Explain why focal helps.

---

## Transformers (Notebooks 11–12)

### Exercise 9 — DETR Query Selection (Notebook 11)
Given 100 query outputs, filter by score threshold and plot boxes in normalized coordinates.

---

## Module Assignment: Custom Object Detector

**Deliverable:** `exercises/assignment_object_detection.ipynb`

Build a detection pipeline on a small custom dataset (or COCO subset):

1. **Dataset:** ≥50 annotated objects (buildings, vehicles, or synthetic shapes)
2. **Format:** YOLO txt or COCO JSON
3. **Model:** YOLOv8n **or** Faster R-CNN (torchvision)
4. **Training:** ≥20 epochs, log loss and mAP@0.5
5. **Evaluation:** mAP@0.5, precision/recall per class, failure case analysis
6. **Inference:** Run on 3 held-out images; visualize predictions + NMS
7. **GeoSpatial extension:** Document sliding-window strategy for large satellite tile
8. **Report:** Compare inference speed (ms/image) and mAP vs a baseline (e.g. untrained model)

**Targets:**
- mAP@0.5 > 0.5 on validation set (synthetic OK if real data unavailable)
- Working NMS post-processing
- Clear visualization of at least 5 TP and 3 FP cases

---

## GeoSpatial Extension

Compare detection vs segmentation for pond counting:
- When is YOLO faster/better?
- When is UNet++ (Module 07) necessary?
- Write 1-page decision guide.

---

## Submission

> Module 08 complete. Assignment attached. Quiz score: X/20.
