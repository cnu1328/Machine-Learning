# Module 12 Cheat Sheet — water-bodies-detection Capstone

## Pipeline (One Page)

```
GeoTIFF + shapefile
  → tile_and_mask.py     → tiles/, masks_aqua/, masks_boundary/
  → train.py             → best.pt, model_meta.json
  → predict.py           → *_aqua.tif, *_boundary.tif
  → post_process_*.py    → aqua_ponds.shp
```

## File Map

| File | Role |
|------|------|
| `config/default.yaml` | All hyperparameters |
| `tile_and_mask.py` | Tiling + dual mask rasterization |
| `dataset.py` | WaterBodyTileDataset + Albumentations |
| `model.py` | UNet++ SE-ResNet50 builder |
| `losses.py` | AquaBoundaryLoss (BCE+Dice) |
| `train.py` | Two-stage training + checkpoints |
| `predict.py` | Sliding window + TTA + blend |
| `post_process/post_process_aqua_boundary.py` | GIS polygons |
| `automate/*.py` | Batch production wrappers |
| `Dockerfile` | GDAL + PyTorch container |

## Dual-Head Design

| ch | Name | Mask dir | Loss weight |
|----|------|----------|-------------|
| 0 | Aqua interior | masks_aqua/ | 1.0 |
| 1 | Bund boundary | masks_boundary/ | 0.35 |

**Post-process:** `interior = (aqua ≥ 0.8) & ~(boundary ≥ 0.5)`

## Key Config Values

```yaml
data.band_indices: [2, 3, 4, 6, 7, 8]   # 6 Planet bands
data.input_size: 512
data.batch_size: 2
tiling.overlap_fraction: 0.5
tiling.boundary_width_meters: 1.0
tiling.negative_tile_ratio: 0.2
training.stage1_epochs: 12    # frozen encoder
training.stage2_epochs: 120   # full fine-tune
training.early_stopping_patience: 18
prediction.threshold_aqua: 0.5
postprocess.threshold_aqua: 0.8   # higher for GIS
postprocess.min_pond_area: 300.0  # m²
```

## Model

```python
smp.UnetPlusPlus(
    encoder_name='se_resnet50',
    in_channels=6, classes=2,
    activation=None,
    decoder_attention_type='scse',
)
```

## Loss

$$L = 1.0 \cdot \text{BCEDice}(aqua) + 0.35 \cdot \text{BCEDice}(boundary)$$

BCEDice = 0.35×BCE + 0.65×(1−Dice)

## Training Stages

1. **Stage 1:** Freeze encoder, LR=2e-4, 12 epochs
2. **Stage 2:** Unfreeze all, LR=2e-5, early stop on val_iou_aqua

## Inference

- Sliding window 512×512, 50% overlap
- Hann blending at seams
- TTA d4 = 8 dihedral views
- Output: float32 probability GeoTIFFs

## Post-Process Steps

1. Gaussian smooth (σ=0.3)
2. Threshold aqua/boundary
3. Morphology close on boundary
4. interior = aqua & ~boundary
5. fill_holes → connected_components
6. min_area filter → vectorize → smooth → simplify

## Artifacts Per Run

```
run_YYYYMMDD_HHMMSS/
├── best.pt
├── final.pt
├── model_meta.json
└── train_config.yaml
```

## Commands

```bash
python tile_and_mask.py --input_tif X --input_shp Y --output_dir Z --config config/default.yaml
python train.py --tiles_dir Z/tiles --masks_aqua_dir Z/masks_aqua --masks_boundary_dir Z/masks_boundary --config config/default.yaml
python predict.py --input_tif mosaic.tif --model_dir models/run_XXX/ --tta d4
python post_process/post_process_aqua_boundary.py --config config/default.yaml --aqua_tif ... --boundary_tif ...
```

## Debug Quick Reference

| Issue | Check |
|-------|-------|
| IoU = 0 | Mask filename stems match tiles |
| Merged ponds | Boundary head + post-process thresholds |
| Seam lines | overlap_fraction, Hann blend |
| OOM | batch_size=1, AMP on |
| CRS error | SHP reproject to raster CRS |

## Performance (Validation)

| Metric | Aqua | Boundary |
|--------|------|----------|
| IoU | 0.85 | 0.71 |
| Dice | 0.92 | 0.83 |

## Graduation Criteria

- Explain dual-head without notes
- Walk train.py from memory
- Modify config for new bands
- Run Docker batch inference
- Design similar pipeline for new task
