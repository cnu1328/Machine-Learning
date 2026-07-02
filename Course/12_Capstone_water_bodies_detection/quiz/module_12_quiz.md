# Module 12 Quiz — Capstone

**Passing score:** 20/25 (80%)

---

**Q1.** The core business problem this project solves:
- (a) Image classification
- (b) Adjacent aquaculture pond merging in semantic segmentation
- (c) Object detection only
- (d) Time series forecasting

**Q2.** Dual-head output channels are:
- (a) RGB
- (b) Aqua interior (ch0) + bund boundary (ch1)
- (c) 80 COCO classes
- (d) Depth + normals

**Q3.** Planet bands used (1-based indices):
- (a) 1, 2, 3
- (b) 2, 3, 4, 6, 7, 8
- (c) All 13 bands always
- (d) RGB only

**Q4.** `tile_and_mask.py` produces:
- (a) Shapefiles only
- (b) tiles/, masks_aqua/, masks_boundary/
- (c) Model weights
- (d) REST API

**Q5.** Robust normalization uses percentiles:
- (a) 0 and 100
- (b) 2 and 98 (default)
- (c) 50 only
- (d) No normalization

**Q6.** Boundary mask is created by:
- (a) Random noise
- (b) Rasterizing polygon boundaries + dilating by meters/GSD
- (c) Canny edge detection only
- (d) Manual painting

**Q7.** Negative tiles have:
- (a) Both masks = 1
- (b) Both masks = 0 (background only)
- (c) Random values
- (d) No image data

**Q8.** Model architecture:
- (a) Linear regression
- (b) UNet++ with SE-ResNet50 encoder
- (c) GPT
- (d) YOLO only

**Q9.** `activation=None` in model means:
- (a) No output
- (b) Raw logits (sigmoid applied in loss/inference)
- (c) Softmax
- (d) ReLU output

**Q10.** AquaBoundaryLoss combines:
- (a) MSE only
- (b) Weighted BCEDice on aqua + boundary heads
- (c) Cross-entropy 10-class
- (d) GAN loss

**Q11.** Boundary loss weight default:
- (a) 1.0
- (b) 0.35
- (c) 0.0
- (d) 10.0

**Q12.** Stage 1 training:
- (a) Full model random init
- (b) Frozen encoder, train decoder
- (c) No training
- (d) Only boundary head

**Q13.** Early stopping monitors:
- (a) Train loss only
- (b) Validation aqua IoU
- (c) Boundary loss only
- (d) Learning rate

**Q14.** best.pt is saved when:
- (a) Every epoch
- (b) val_iou_aqua improves
- (c) Random
- (d) Never

**Q15.** predict.py uses:
- (a) Single forward on full image always
- (b) Sliding window + Hann blending + optional TTA
- (c) K-means
- (d) Manual labeling

**Q16.** TTA d4 means:
- (a) 4-fold cross validation
- (b) 8 dihedral transforms averaged
- (c) 4-band input
- (d) 4 GPUs

**Q17.** Post-process threshold_aqua default:
- (a) 0.5 (same as inference)
- (b) 0.8 (higher for GIS precision)
- (c) 0.1
- (d) 1.0

**Q18.** Interior mask formula:
- (a) aqua OR boundary
- (b) aqua AND NOT boundary
- (c) boundary only
- (d) aqua XOR boundary

**Q19.** automate_water_predictions.py:
- (a) Trains model
- (b) Batch runs predict.py over input folder
- (c) Creates Docker
- (d) Downloads satellite data

**Q20.** Dockerfile installs GDAL because:
- (a) Decoration
- (b) rasterio/geopandas need native geospatial libraries
- (c) PyTorch requirement
- (d) For matplotlib only

**Q21.** model_meta.json contains:
- (a) Raw pixels
- (b) in_channels, out_channels, input_size, best IoU
- (c) Git history
- (d) Learning rate schedule

**Q22.** Albumentations uses additional_targets for:
- (a) Speed only
- (b) Applying same geometric transform to boundary mask
- (c) GPU acceleration
- (d) Loss computation

**Q23.** grad_clip_norm default:
- (a) 0 (disabled)
- (b) 1.0
- (c) 100
- (d) -1

**Q24.** min_pond_area default filters:
- (a) Large ponds
- (b) Small polygons below 300 m²
- (c) All ponds
- (d) Boundaries only

**Q25.** Why two-stage training?
- (a) Random choice
- (b) Stable decoder learning before fine-tuning pretrained encoder
- (c) Required by PyTorch
- (d) Doubles training time for no reason

---

See [solutions/quiz_answers.md](solutions/quiz_answers.md) after submitting.
