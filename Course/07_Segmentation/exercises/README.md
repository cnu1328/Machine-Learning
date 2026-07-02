# Module 07 Exercises

---

## Segmentation Types (01–05)

### Exercise 1 — Binary Mask EDA
Create synthetic image with circular foreground. Visualize image, mask, overlay.

### Exercise 2 — mIoU Calculator
Implement mIoU for 5-class segmentation from scratch. Test on random predictions.

### Exercise 3 — Multi-label vs Multi-class
Show a pixel where multi-label allows (aqua=1, boundary=1) but multi-class cannot.

---

## Architectures (06–14)

### Exercise 4 — Train Simple UNet
Train Notebook 07 UNet on synthetic circles. Plot loss and IoU for 20 epochs.

### Exercise 5 — UNet++ with SMP
If `segmentation-models-pytorch` installed, train UNet++ resnet34 on synthetic data. Compare params with custom UNet.

### Exercise 6 — ASPP Visualization
Visualize ASPP multi-scale feature maps at different dilation rates.

---

## Losses (15–20)

### Exercise 7 — BCE vs Dice
On imbalanced synthetic mask (5% foreground), compare BCE-only vs Dice-only vs BCEDice training curves.

### Exercise 8 — Reimplement AquaBoundaryLoss
Match `water-bodies-detection/losses.py` exactly. Unit test with known logits/targets.

### Exercise 9 — Focal Loss
Train with CE vs Focal on heavily imbalanced tile. Compare foreground IoU.

---

## Module Assignment: Binary Segmentation Pipeline

**Deliverable:** `exercises/assignment_segmentation_pipeline.ipynb`

Build a mini version of water-bodies pipeline:

1. **Synthetic dataset:** Random shapes as "ponds" on satellite-like RGB tiles
2. **Dual masks:** interior + boundary ring (like your project)
3. **Model:** UNet or UNet++ with 2 output channels
4. **Loss:** BCEDiceLoss or AquaBoundaryLoss (your losses.py logic)
5. **Metrics:** IoU and Dice per channel, tracked per epoch
6. **Inference:** Sliding window on larger image
7. **Post-process:** Threshold → connected components → count ponds
8. **Report:** Compare single-head vs dual-head for adjacent pond separation

Target: Boundary head improves adjacent pond separation vs aqua-only model.

---

## GeoSpatial Extension

Read `water-bodies-detection/config/default.yaml` and document how each parameter maps to Module 07 concepts.

---

## Submission

> Module 07 complete. Assignment attached. Quiz score: X/20.
