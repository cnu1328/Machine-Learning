#!/usr/bin/env python3
"""Generate Module 12 Capstone notebooks (18 total)."""
import json
from pathlib import Path

M12 = Path(__file__).resolve().parent / "12_Capstone_water_bodies_detection"
REPO = "../../water-bodies-detection"


def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": [s]}


def code(s):
    return {"cell_type": "code", "metadata": {}, "source": [s], "outputs": [], "execution_count": None}


def save(name, cells):
    M12.mkdir(parents=True, exist_ok=True)
    (M12 / name).write_text(json.dumps(nb(cells), indent=1))
    print(f"  {name}")


def hdr(num, title, dur, objs):
    return md(
        f"# Notebook {num}: {title}\n\n**Module:** 12 Capstone — water-bodies-detection  \n**Duration:** ~{dur}\n\n---\n\n## Learning Objectives\n\n{objs}"
    )


def footer(summary, nxt=None):
    nxt = f"\n\n**Next:** [{nxt}]({nxt})" if nxt else ""
    return md(f"---\n\n## Summary\n\n{summary}{nxt}")


SETUP = code(
    "import os, json, yaml\nimport numpy as np\nimport matplotlib.pyplot as plt\n\nplt.rcParams['figure.figsize'] = (8, 5)\nREPO = '../../water-bodies-detection'\nprint('Capstone repo path:', os.path.abspath(REPO) if os.path.exists(REPO) else 'clone water-bodies-detection alongside Machine-Learning')"
)

NOTEBOOKS = []


def register(name, cells):
    NOTEBOOKS.append((name, cells))


# ── WEEK 1: DATA & CONFIG (01-04) ────────────────────────────────────────

register("01_Project_Overview.ipynb", [
    hdr("01", "Project Overview", "2 hrs",
        "1. State the aquaculture pond detection problem\n2. Understand dual-head design rationale\n3. Map the full pipeline end-to-end\n4. Connect Modules 06–11 to this capstone"),
    md("## Problem Statement\n\n**Task:** Detect and delineate **aquaculture ponds** from Planet multispectral imagery.\n\n**Not simple water detection:** Adjacent ponds share walls (bunds). Single-class semantic segmentation merges them into one blob.\n\n**Solution:** Dual-head UNet++ predicts:\n- **ch0 — Aqua interior:** filled pond polygon\n- **ch1 — Bund boundary:** earthen divider lines\n\nPost-process: `interior = aqua & ~boundary` → connected components → one polygon per pond."),
    SETUP,
    md("## System Architecture\n\n```\nGeoTIFF + shapefile\n  → tile_and_mask.py\n  → train.py (UNet++ + AquaBoundaryLoss)\n  → predict.py (sliding window + TTA)\n  → post_process_aqua_boundary.py\n  → aqua_ponds.shp\n```\n\n**Open in repo:** `water-bodies-detection/README.md` — full documentation with Mermaid diagrams."),
    md("## Course Connection\n\n| Module | This Project |\n|--------|-------------|\n| 06 CNN | SE-ResNet50 encoder |\n| 07 Segmentation | UNet++, dual-head, AquaBoundaryLoss |\n| 08 Detection | Optional pond counting via boxes |\n| 09 Instance Seg | Alternative to dual-head |\n| 10 Transformers | SAM for labeling, SegFormer alt |\n| 11 Production | YAML config, Docker, batch automate |"),
    md("## Performance (from README)\n\n| Metric | Aqua | Boundary |\n|--------|------|----------|\n| IoU | 0.85 | 0.71 |\n| Dice | 0.92 | 0.83 |\n\n**Your goal:** Explain every file — not rewrite it."),
    footer("Dual-head segmentation separates adjacent ponds via learned bund boundaries.", "02_Configuration_Deep_Dive.ipynb"),
])

register("02_Configuration_Deep_Dive.ipynb", [
    hdr("02", "Configuration Deep Dive", "2.5 hrs",
        "1. Walk through every section of config/default.yaml\n2. Explain why each default value was chosen\n3. Know which params affect train vs infer vs post\n4. Practice modifying config safely"),
    md("## Single Source of Truth\n\n**File:** `water-bodies-detection/config/default.yaml`\n\nAll scripts load this YAML — never hardcode hyperparameters in Python."),
    SETUP,
    code("cfg_path = os.path.join(REPO, 'config/default.yaml')\nif os.path.isfile(cfg_path):\n    with open(cfg_path) as f:\n        cfg = yaml.safe_load(f)\n    for section in ['data', 'tiling', 'model', 'training', 'prediction', 'postprocess']:\n        print(f'\\n=== {section} ===')\n        print(yaml.dump({section: cfg.get(section, {})}, default_flow_style=False)[:600])\nelse:\n    print('Open config/default.yaml in the repo')"),
    md("## Section Guide\n\n### `data`\n| Key | Default | Why |\n|-----|---------|-----|\n| `input_size` | 512 | Matches tile size, GPU memory |\n| `band_indices` | [2,3,4,6,7,8] | Blue→NIR multispectral (not RGB) |\n| `batch_size` | 2 | 512×512×6ch fits ~8GB VRAM |\n| `validation_split` | 0.15 | Tile-level holdout |\n\n### `tiling`\n| Key | Default | Why |\n|-----|---------|-----|\n| `overlap_fraction` | 0.5 | Training coverage + inference blending |\n| `boundary_width_meters` | 1.0 | Bund dilation width in meters |\n| `negative_tile_ratio` | 0.2 | Reduce false positives on bare soil |\n| `percentile_low/high` | 2/98 | Robust normalization across scenes |\n\n### `training`\n| Key | Default | Why |\n|-----|---------|-----|\n| `stage1_epochs` | 12 | Decoder-only with frozen encoder |\n| `learning_rate_stage1` | 2e-4 | Higher LR for random decoder |\n| `learning_rate_stage2` | 2e-5 | 10× lower when unfreezing encoder |\n| `loss_weight_boundary` | 0.35 | Sparse boundary — don't dominate gradients |\n\n### `prediction` vs `postprocess` thresholds\n| Stage | `threshold_aqua` | Purpose |\n|-------|-----------------|----------|\n| Inference | 0.5 | Recall-friendly probability maps |\n| Post-process | 0.8 | Precision-friendly GIS polygons |"),
    footer("default.yaml drives every stage — understand before changing any value.", "03_Tiling_Pipeline.ipynb"),
])

register("03_Tiling_Pipeline.ipynb", [
    hdr("03", "Tiling Pipeline (tile_and_mask.py)", "3 hrs",
        "1. Understand windowed GeoTIFF reading\n2. Trace band selection and robust normalization\n3. Follow dual mask rasterization\n4. Explain negative tile sampling"),
    md("## tile_and_mask.py Purpose\n\nConverts raw **GeoTIFF + aquaculture shapefile** into ML-ready triplets:\n\n```\noutput_dir/\n├── tiles/           # 6-band float32 normalized\n├── masks_aqua/      # binary interior\n├── masks_boundary/  # dilated bund lines\n└── tiling_meta.json # reproducibility\n```"),
    SETUP,
    md("## Key Functions\n\n### `robust_normalize_tile(tile, p_low, p_high)`\n- Per-band percentile scaling (2nd–98th)\n- Fallback: minmax → sigma → flat\n- NaN outside valid pixels\n- **Why:** Planet scenes vary in radiometry\n\n### `pixel_valid_mask`\n- Alpha band (index 9) ≥ min_alpha_value\n- Exclude all-bands-zero padding\n- Exclude nodata\n\n### Mask generation\n1. **Aqua:** `rasterio.features.rasterize` polygon fills\n2. **Boundary:** rasterize `polygon.boundary`, dilate by `boundary_width_meters / (2 × GSD)`\n\n### Negative tiles\n- Random windows with both masks = 0\n- `negative_tile_ratio: 0.2` — teaches background class\n- Optional `--hard_negative_shp` for rivers/lakes"),
    code("# Normalization concept (from tile_and_mask.py)\ndef demo_percentile_norm(band, p_low=2, p_high=98):\n    lo, hi = np.percentile(band, [p_low, p_high])\n    return np.clip((band - lo) / (hi - lo + 1e-8), 0, 1)\n\nband = np.random.exponential(500, (512, 512)) + 100\nnorm = demo_percentile_norm(band)\nfig, ax = plt.subplots(1, 2, figsize=(10, 4))\nax[0].imshow(band, cmap='gray'); ax[0].set_title('Raw DN'); ax[0].axis('off')\nax[1].imshow(norm, cmap='gray'); ax[1].set_title('Percentile normalized'); ax[1].axis('off')\nplt.tight_layout(); plt.show()"),
    md("## CLI\n\n```bash\npython tile_and_mask.py \\\n  --input_tif area.tif \\\n  --input_shp aquaculture.shp \\\n  --output_dir tiles_masks/ \\\n  --config config/default.yaml\n```\n\n## Common Bugs\n- CRS mismatch between SHP and TIF → reproject in script\n- `meters_per_pixel` null → boundary width wrong\n- Too few positive tiles → check `min_positive_valid_pixels`"),
    footer("tile_and_mask.py is the data foundation — garbage in, garbage out.", "04_GIS_Fundamentals.ipynb"),
])

register("04_GIS_Fundamentals.ipynb", [
    hdr("04", "GIS Fundamentals for ML", "2 hrs",
        "1. Understand CRS, transform, and GSD\n2. Use rasterio windows and rasterize\n3. Know geopandas/shapely role in pipeline\n4. Connect pixel coords to map units"),
    md("## Core GIS Concepts\n\n| Concept | Meaning | In Project |\n|---------|---------|------------|\n| **CRS** | Coordinate reference system | Shapefile reprojected to match raster |\n| **Transform** | Pixel → map coordinates | Preserved in output GeoTIFFs |\n| **GSD** | Ground sample distance (m/px) | `meters_per_pixel` for bund width |\n| **Rasterize** | Vector → pixel mask | Aqua + boundary mask generation |\n| **Vectorize** | Pixel labels → polygons | post_process output .shp |\n\n## Libraries\n\n- **rasterio:** GeoTIFF I/O, windows, rasterize, shapes\n- **geopandas:** Shapefile reading, CRS ops\n- **shapely:** Geometry validation, buffer, simplify"),
    SETUP,
    md("## Rasterio Window Pattern\n\n```python\nfrom rasterio.windows import Window\nwin = Window(col_off, row_off, width, height)\ntile = src.read(window=win)  # (bands, H, W)\n```\n\nUsed in `tile_and_mask.py` (training tiles) and `predict.py` (sliding window)."),
    md("## meters_per_pixel → boundary pixels\n\n```\nhalf_width_px = boundary_width_meters / (2 × meters_per_pixel)\n# boundary_width_meters=1.0, GSD=3m → ~0.17 px half-width → dilate ~1 px\n```\n\nAt 3m GSD, 1m bund ≈ 1 pixel — why dual-head learning is critical."),
    footer("GIS fundamentals explain why this pipeline is not 'just another segmentation task'.", "05_Dataset_Class.ipynb"),
])

# ── WEEK 2: MODEL & LOSS (05-08) ────────────────────────────────────────

register("05_Dataset_Class.ipynb", [
    hdr("05", "Dataset Class (dataset.py)", "2.5 hrs",
        "1. Walk through WaterBodyTileDataset\n2. Understand Albumentations dual-mask pipeline\n3. Know train vs val behavior\n4. Trace tensor shapes to model input"),
    md("## WaterBodyTileDataset\n\n**Pairs by filename stem:**\n- `tiles/foo.tif`\n- `masks_aqua/foo.tif`\n- `masks_boundary/foo.tif`\n\n**Returns:** `(image, mask)` where:\n- `image`: `(6, 512, 512)` float32 [0,1]\n- `mask`: `(2, 512, 512)` float32 — ch0 aqua, ch1 boundary"),
    SETUP,
    md("## Augmentation (`build_train_augmentation`)\n\n**Critical:** `additional_targets={'mask_boundary': 'mask'}` — both masks transform together.\n\n| Aug | p | Why |\n|-----|---|-----|\n| H/V flip | 0.5 | No preferred orientation |\n| ShiftScaleRotate | 0.55 | GSD/view variation |\n| ElasticTransform/GridDistortion | 0.25 | Irregular pond shapes |\n| GaussNoise/Blur | 0.35 | Sensor/atmosphere |\n| RandomGamma | 0.45 | Sun angle variation |\n\n**Validation:** `augment=False` — no random transforms."),
    code("# __getitem__ flow (conceptual)\n# 1. rasterio.read tile → (C,H,W)\n# 2. load masks → (H,W) each\n# 3. if augment: albumentations on image + both masks\n# 4. stack masks → (2,H,W) tensor\nprint('Output: image (6,512,512), mask (2,512,512)')"),
    footer("dataset.py ensures geometry-safe augmentation on both mask channels.", "06_Model_Architecture.ipynb"),
])

register("06_Model_Architecture.ipynb", [
    hdr("06", "Model Architecture (model.py)", "2.5 hrs",
        "1. Read build_water_model line by line\n2. Understand UNet++ + SE-ResNet50 + SCSE\n3. Know in_channels=6 multispectral handling\n4. Explain set_encoder_trainable for two-stage training"),
    md("## model.py — Complete File\n\n```python\ndef build_water_model(in_channels, out_channels=2, ...):\n    return smp.UnetPlusPlus(\n        encoder_name='se_resnet50',\n        encoder_weights='imagenet',\n        in_channels=6,\n        classes=2,\n        activation=None,  # raw logits\n        decoder_attention_type='scse',\n        decoder_channels=(256, 128, 64, 32, 16),\n    )\n\ndef set_encoder_trainable(model, trainable: bool):\n    for p in model.encoder.parameters():\n        p.requires_grad = trainable\n```\n\n**Only 37 lines** — architecture delegated to `segmentation_models_pytorch`."),
    SETUP,
    md("## Architecture Choices\n\n| Component | Choice | Rationale |\n|-----------|--------|----------|\n| UNet++ | Nested dense skips | Best edge quality for water/ponds |\n| SE-ResNet50 | ImageNet pretrain + SE blocks | Channel recalibration for 6-band input |\n| SCSE decoder | Spatial + channel attention | Focus on thin bund pixels |\n| activation=None | Raw logits | BCEWithLogitsLoss in losses.py |\n| out_channels=2 | Aqua + boundary | Dual-head design |\n\n## Multispectral Input\n\nSMP initializes first conv for `in_channels != 3` — copies RGB weights, random init for extra bands."),
    footer("model.py is thin — SMP does the heavy lifting; you own the config choices.", "07_Loss_Functions.ipynb"),
])

register("07_Loss_Functions.ipynb", [
    hdr("07", "Loss Functions (losses.py)", "2.5 hrs",
        "1. Derive dice_with_logits\n2. Walk BCEDiceLoss and AquaBoundaryLoss\n3. Understand iou_with_logits for validation\n4. Explain loss weight choices"),
    md("## losses.py — Every Function\n\n### `dice_with_logits(logits, targets)`\n$$\\text{Dice} = \\frac{2 \\sum p_i t_i + \\epsilon}{\\sum p_i + \\sum t_i + \\epsilon}$$\n\n`p_i = sigmoid(logits)` — differentiable soft Dice.\n\n### `BCEDiceLoss`\n$$L = 0.35 \\cdot \\text{BCEWithLogits} + 0.65 \\cdot (1 - \\text{Dice})$$\n\nBCE: pixel-wise gradients. Dice: region overlap, handles imbalance.\n\n### `AquaBoundaryLoss`\n$$L_{total} = 1.0 \\cdot L_{aqua} + 0.35 \\cdot L_{boundary}$$\n\nApplied independently per channel via `logits[:, 0:1]` and `logits[:, 1:2]`.\n\n### `iou_with_logits`\nThreshold at 0.5 → hard IoU — used in `validate()` for checkpoint selection."),
    SETUP,
    code("import torch\nimport torch.nn.functional as F\n\n# Reproduce losses.py logic\ndef dice_with_logits(logits, targets, smooth=1.0):\n    probs = torch.sigmoid(logits)\n    t = targets.float()\n    inter = (probs * t).sum(dim=(2, 3))\n    union = probs.sum(dim=(2, 3)) + t.sum(dim=(2, 3))\n    return ((2.0 * inter + smooth) / (union + smooth)).mean()\n\nlogits = torch.randn(2, 1, 64, 64)\ntargets = (torch.rand(2, 1, 64, 64) > 0.7).float()\nbce = F.binary_cross_entropy_with_logits(logits, targets)\ndice_loss = 1 - dice_with_logits(logits, targets)\nprint(f'BCE={bce:.4f}, Dice loss={dice_loss:.4f}, combined={0.35*bce + 0.65*dice_loss:.4f}')"),
    footer("AquaBoundaryLoss = weighted BCE+Dice per head — the training objective.", "08_Why_Dual_Heads.ipynb"),
])

register("08_Why_Dual_Heads.ipynb", [
    hdr("08", "Why Dual Heads?", "2 hrs",
        "1. Articulate the adjacent-pond merging problem\n2. Compare single-head vs dual-head vs instance seg\n3. Explain post-process interior subtraction\n4. Defend design in an architecture review"),
    md("## The Core Problem\n\n```\n[Pond A][bund][Pond B]  ← single semantic head → one water blob\n[Pond A] | [Pond B]     ← dual head → bund separates → 2 polygons\n```\n\n## Three Solutions Compared\n\n| Approach | Labels | Inference | Your Project |\n|----------|--------|-----------|-------------|\n| Single-head aqua | Interior only | CC splits (fragile) | ❌ |\n| **Dual-head aqua+boundary** | Interior + bund ring | `aqua & ~boundary` | ✅ Production |\n| Mask R-CNN instance | Instance polygons | Native IDs | Alternative (Mod 09) |\n\n## Why boundary weight = 0.35?\n\nBoundary pixels are **sparse** (~1–3 px wide). Without down-weighting, boundary gradients would dominate and destabilize aqua learning."),
    SETUP,
    md("## Post-Process Logic\n\n```python\ninterior = (aqua_prob >= 0.8) & ~(boundary_prob >= 0.5)\ninterior = fill_holes(interior)\nlabels = connected_components(interior)\n```\n\nEach connected component → one pond polygon with `pond_id`, `area_map`."),
    footer("Dual-head is an engineering choice — learned bunds beat heuristic splitting.", "09_Training_Loop.ipynb"),
])

# ── WEEK 3: TRAINING (09-12) ────────────────────────────────────────────

register("09_Training_Loop.ipynb", [
    hdr("09", "Training Loop (train.py)", "3 hrs",
        "1. Walk main() from config load to checkpoint save\n2. Understand train_epoch with AMP\n3. Trace two-stage freeze/unfreeze\n4. Debug a stalled training run"),
    md("## train.py Flow\n\n```\n1. Load config/default.yaml\n2. get_stem_triplets() — match tiles + masks by filename\n3. train_test_split (tile-level, 85/15)\n4. WaterBodyTileDataset + DataLoader\n5. build_water_model()\n6. Stage 1: freeze encoder, train decoder (12 epochs)\n7. Stage 2: unfreeze all, early stopping (120 max)\n8. Save best.pt, final.pt, model_meta.json\n```"),
    SETUP,
    md("## train_epoch() — Line by Line\n\n```python\nfor x, y in loader:\n    x, y = x.to(device), y.to(device)\n    with torch.amp.autocast('cuda'):      # FP16 forward\n        logits = model(x)                 # (B, 2, H, W)\n    logits = logits.float()               # FP32 loss\n    loss = criterion(logits, y)           # AquaBoundaryLoss\n    scaler.scale(loss).backward()\n    clip_grad_norm_(..., grad_clip_norm)  # max norm 1.0\n    scaler.step(optimizer); scaler.update()\n```\n\n**AMP:** Forward in FP16, loss in FP32 — prevents numerical issues."),
    md("## Two-Stage Training\n\n**Stage 1 (epochs 1–12):**\n- `set_encoder_trainable(model, False)`\n- LR = 2e-4, warmup 2 epochs\n- Decoder learns on frozen ImageNet features\n\n**Stage 2 (epochs 1–120):**\n- `set_encoder_trainable(model, True)`\n- LR = 2e-5 (10× lower)\n- Full fine-tune with early stopping (patience=18 on val aqua IoU)"),
    footer("train.py orchestrates everything — know it line by line.", "10_Optimizer_and_Scheduler.ipynb"),
])

register("10_Optimizer_and_Scheduler.ipynb", [
    hdr("10", "Optimizer & Scheduler", "2 hrs",
        "1. Understand AdamW choice\n2. Derive WarmupCosine schedule\n3. Explain weight decay and grad clipping\n4. Tune LR for new datasets"),
    md("## AdamW\n\n- Adaptive per-parameter learning rates\n- Decoupled weight decay (1e-5) — regularizes without fighting momentum\n- Better than SGD for transfer learning small datasets\n\n## WarmupCosine Class\n\n```python\n# Warmup: linear 0 → base_lr over warmup epochs\n# Cosine: base_lr → min_lr over remaining epochs\nlr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * t))\n```\n\n**Stage 1:** warmup=2, base=2e-4\n**Stage 2:** warmup=3, base=2e-5\n\n## Gradient Clipping\n\n`grad_clip_norm=1.0` — prevents explosion when boundary loss spikes on sparse pixels."),
    SETUP,
    code("import numpy as np\n\ndef warmup_cosine_lr(epoch, base_lr, epochs, warmup, min_lr):\n    if epoch < warmup:\n        return base_lr * (epoch + 1) / warmup\n    t = (epoch - warmup) / max(1, epochs - warmup)\n    t = min(1.0, max(0.0, t))\n    return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * t))\n\nlrs = [warmup_cosine_lr(e, 2e-4, 12, 2, 1e-8) for e in range(12)]\nplt.plot(lrs); plt.xlabel('epoch'); plt.ylabel('LR'); plt.title('Stage 1 schedule'); plt.show()"),
    footer("WarmupCosine + AdamW + grad clip = stable two-stage fine-tuning.", "11_Metrics_and_Validation.ipynb"),
])

register("11_Metrics_and_Validation.ipynb", [
    hdr("11", "Metrics & Validation", "2 hrs",
        "1. Understand validate() function\n2. Know why aqua IoU drives checkpointing\n3. Compute precision/recall from masks\n4. Design better validation splits"),
    md("## validate() in train.py\n\n```python\n@torch.no_grad()\ndef validate(model, loader, device, use_amp, criterion):\n    for x, y in loader:\n        logits = model(x)\n        loss = criterion(logits, y)\n        iou_aqua.append(iou_with_logits(logits[:,0:1], y[:,0:1]))\n        iou_bound.append(iou_with_logits(logits[:,1:2], y[:,1:2]))\n    return mean_loss, mean_iou_aqua, mean_iou_boundary\n```\n\n**Checkpoint criterion:** `val_iou_aqua` only — primary business metric.\n\n## Metric Definitions\n\n| Metric | Formula | Use |\n|--------|---------|-----|\n| IoU | \\|P∩T\\|/\\|P∪T\\| | Checkpoint selection |\n| Dice | 2\\|P∩T\\|/(\\|P\\|+\\|T\\|) | Loss component |\n| Precision | TP/(TP+FP) | False pond rate |\n| Recall | TP/(TP+FN) | Missed ponds |\n\n## Validation Caveat\n\nTile-level split can **leak spatial correlation** (tiles from same mosaic in train and val). For rigorous benchmarks, use raster-level splits."),
    footer("val_iou_aqua drives best.pt — boundary IoU is logged but not used for early stop.", "12_Checkpoints_and_Artifacts.ipynb"),
])

register("12_Checkpoints_and_Artifacts.ipynb", [
    hdr("12", "Checkpoints & Artifacts", "2 hrs",
        "1. Understand best.pt vs final.pt\n2. Read model_meta.json fields\n3. Use frozen train_config.yaml for reproduction\n4. Resume training from checkpoint"),
    md("## Training Output Directory\n\n```\nmodels/run_YYYYMMDD_HHMMSS/\n├── best.pt              # Highest val_iou_aqua\n├── final.pt             # Last epoch weights\n├── model_meta.json      # Architecture metadata\n└── train_config.yaml    # Frozen config copy\n```\n\n## model_meta.json\n\n```json\n{\n  \"in_channels\": 6,\n  \"out_channels\": 2,\n  \"input_size\": 512,\n  \"best_val_iou_aqua\": 0.87,\n  \"tiles_dir\": \"...\",\n  \"masks_aqua_dir\": \"...\",\n  \"masks_boundary_dir\": \"...\"\n}\n```\n\n**predict.py reads this** to rebuild model with correct channels.\n\n## Resume\n\n```bash\npython train.py ... --resume path/to/best.pt\n```\n\nLoads state_dict before stage 1 — useful for continuing training on new tiles."),
    SETUP,
    code("meta_example = {\n    'in_channels': 6, 'out_channels': 2, 'input_size': 512,\n    'best_val_iou_aqua': 0.85, 'band_indices': [2,3,4,6,7,8]\n}\nprint(json.dumps(meta_example, indent=2))"),
    footer("Artifacts make every run reproducible — never deploy without model_meta.json.", "13_Sliding_Window_Inference.ipynb"),
])

# ── WEEK 4: INFERENCE & DEPLOY (13-18) ──────────────────────────────────

register("13_Sliding_Window_Inference.ipynb", [
    hdr("13", "Sliding Window Inference (predict.py)", "3 hrs",
        "1. Understand sliding window over full mosaic\n2. Implement Hann window blending concept\n3. Apply TTA d4 (8 dihedral transforms)\n4. Match train normalization at inference"),
    md("## predict.py Overview\n\nFull GeoTIFF too large for GPU → **sliding window** inference.\n\n```\n1. Read mosaic with rasterio\n2. Generate overlapping 512×512 windows (50% overlap)\n3. For each window:\n   a. Select bands [2,3,4,6,7,8]\n   b. robust_normalize_tile (same as training)\n   c. Model forward → logits (2, H, W)\n   d. Optional TTA d4 (8 views averaged)\n   e. Sigmoid → probabilities\n4. Hann-weighted blend into full raster\n5. Write *_aqua.tif + *_boundary.tif (float32)\n```"),
    SETUP,
    code("def hann_blend_weights(h, w, floor=1e-3):\n    wy = np.hanning(h)\n    wx = np.hanning(w)\n    return np.outer(wy, wx) + floor\n\nw = hann_blend_weights(512, 512)\nplt.imshow(w, cmap='viridis'); plt.title('Hann blend weights (center weighted)'); plt.colorbar(); plt.show()"),
    md("## TTA d4\n\n8 transforms: 4 rotations × 2 (with/without flip).\n\nEach view: transform input → predict → inverse transform output → average.\n\n**Cost:** 8× forward passes — use for production quality, skip for speed.\n\n## blend_weights_2d\n\nHann window reduces seam artifacts at tile edges — center pixels weighted higher than edges."),
    footer("Sliding window + Hann blending + TTA = seamless full-mosaic predictions.", "14_Probability_Rasters.ipynb"),
])

register("14_Probability_Rasters.ipynb", [
    hdr("14", "Probability Rasters", "2 hrs",
        "1. Understand float32 probability GeoTIFF output\n2. Know inference vs post-process thresholds\n3. Preserve georeferencing in outputs\n4. QA probability maps before post-process"),
    md("## Output Files\n\n| File | Content | Dtype |\n|------|---------|-------|\n| `*_aqua.tif` | P(pond interior) per pixel | float32 [0,1] |\n| `*_boundary.tif` | P(bund) per pixel | float32 [0,1] |\n\n**Georeferencing:** Same CRS + transform as input mosaic (via rasterio profile).\n\n## Threshold Strategy\n\n| Stage | aqua | boundary | Rationale |\n|-------|------|----------|----------|\n| Inference default | 0.5 | 0.5 | Balanced probability maps |\n| Post-process | **0.8** | 0.5 | High-precision GIS polygons |\n\n**Key insight:** Keep inference recall-friendly; tighten at post-process for GIS delivery.\n\n## CLI\n\n```bash\npython predict.py \\\n  --input_tif mosaic.tif \\\n  --output_dir predictions/ \\\n  --model_dir models/run_YYYYMMDD/ \\\n  --weights models/run_YYYYMMDD/best.pt \\\n  --tta d4\n```"),
    footer("Probability rasters are intermediate artifacts — post-process converts to GIS.", "15_GIS_Post_Processing.ipynb"),
])

register("15_GIS_Post_Processing.ipynb", [
    hdr("15", "GIS Post-Processing", "3 hrs",
        "1. Walk post_process_aqua_boundary.py logic\n2. Understand interior = aqua & ~boundary\n3. Trace connected components → vectorize\n4. Tune post-process parameters"),
    md("## Post-Process Pipeline\n\n```\n1. Gaussian smooth probability maps (sigma=0.3)\n2. Threshold: aqua >= 0.8, boundary >= 0.5\n3. Morphology: close boundary gaps (boundary_close_px=1)\n4. interior = aqua_mask & ~boundary_mask\n5. fill_holes(interior)\n6. connected_components (connectivity=8)\n7. filter min_pond_area (300 m²)\n8. vectorize → shapely polygons\n9. smooth (buffer +/-) + simplify (Douglas-Peucker)\n10. write aqua_ponds.shp\n```"),
    SETUP,
    md("## Key Parameters\n\n| Param | Default | Fixes |\n|-------|---------|-------|\n| `threshold_aqua` | 0.8 | False positives |\n| `boundary_close_px` | 1 | Gaps in bund lines |\n| `min_pond_area` | 300 m² | Tiny noise blobs |\n| `smooth_distance` | 2.0 | Jagged raster edges |\n| `simplify_tolerance` | 1.5 | Vertex count |\n| `fill_holes` | true | Donut artifacts |\n\n## Output Attributes\n\nEach polygon: `pond_id`, `area_px`, `area_map`, `perimeter`"),
    md("## CLI\n\n```bash\npython post_process/post_process_aqua_boundary.py \\\n  --config config/default.yaml \\\n  --aqua_tif predictions/mosaic_aqua.tif \\\n  --boundary_tif predictions/mosaic_boundary.tif \\\n  --output_path shapefiles/\n```"),
    footer("Post-processing turns ML probabilities into GIS-ready bund-separated polygons.", "16_Batch_Automation.ipynb"),
])

register("16_Batch_Automation.ipynb", [
    hdr("16", "Batch Automation (automate/)", "2 hrs",
        "1. Walk automate_water_predictions.py\n2. Understand subprocess batch pattern\n3. Chain predict → post-process\n4. Design fault-tolerant production jobs"),
    md("## automate/automate_water_predictions.py\n\n```python\nfor each GeoTIFF in input_dir:\n    subprocess.run([python, predict.py, ...])\n    → writes stem_aqua.tif, stem_boundary.tif\n```\n\n**Features:**\n- Streams subprocess stdout to console\n- `--model_dir` resolves weights + model_meta.json\n- Idempotent if outputs exist (check script logic)\n\n## automate/automate_water_postprocess.py\n\nBatch wrapper for `post_process_aqua_boundary.py` over prediction pairs.\n\n## Production Pattern\n\n```bash\n# Step 1: Batch predict\npython automate/automate_water_predictions.py \\\n  --model_dir models/run_20260115/ \\\n  --input_dir /data/mosaics/ \\\n  --output_dir /data/predictions/\n\n# Step 2: Batch post-process\npython automate/automate_water_postprocess.py \\\n  --predictions_dir /data/predictions/ \\\n  --output_dir /data/shapefiles/\n```"),
    footer("automate/ scripts wrap single-file tools for farm-scale GeoSpatial production.", "17_Docker_Deployment.ipynb"),
])

register("17_Docker_Deployment.ipynb", [
    hdr("17", "Docker Deployment", "2 hrs",
        "1. Read water-bodies Dockerfile line by line\n2. Understand GDAL system dependencies\n3. Run containerized inference\n4. Mount volumes for data"),
    md("## Dockerfile\n\n```dockerfile\nFROM python:3.11-slim\nRUN apt-get install gdal-bin libgdal-dev libgeos-dev ...\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY . /app\nCMD [\"python\", \"predict.py\", \"--help\"]\n```\n\n**Why GDAL in apt?** rasterio/geopandas need native libs — pip alone insufficient.\n\n## Run\n\n```bash\ndocker build -t water-bodies-detection .\ndocker run --gpus all --shm-size=16g \\\n  -v /data:/data \\\n  water-bodies-detection \\\n  python predict.py --input_tif /data/mosaic.tif ...\n```\n\n**--shm-size=16g:** PyTorch DataLoader shared memory for multi-worker loading."),
    SETUP,
    md("## Production Checklist\n\n- [ ] Pin requirements.txt versions\n- [ ] Test Docker build in CI\n- [ ] GPU runtime (nvidia-container-toolkit)\n- [ ] Volume mounts for GeoTIFF I/O\n- [ ] Non-root user for security (optional hardening)"),
    footer("Docker packages GDAL + PyTorch for reproducible deployment anywhere.", "18_End_to_End_Pipeline.ipynb"),
])

register("18_End_to_End_Pipeline.ipynb", [
    hdr("18", "End-to-End Pipeline & Graduation", "3 hrs",
        "1. Execute full pipeline from raw data to shapefile\n2. Debug common failure points\n3. Plan extensions (new bands, regions, architectures)\n4. Complete the course"),
    md("## Full Command Sequence\n\n```bash\n# 0. Install\npip install -r requirements.txt\n\n# 1. Tile + masks\npython tile_and_mask.py \\\n  --input_tif area.tif --input_shp ponds.shp \\\n  --output_dir tiles_masks/ --config config/default.yaml\n\n# 2. Train\npython train.py \\\n  --tiles_dir tiles_masks/tiles \\\n  --masks_aqua_dir tiles_masks/masks_aqua \\\n  --masks_boundary_dir tiles_masks/masks_boundary \\\n  --config config/default.yaml --output_dir models/\n\n# 3. Predict\npython predict.py \\\n  --input_tif mosaic.tif --output_dir preds/ \\\n  --model_dir models/run_YYYYMMDD/ --tta d4\n\n# 4. Post-process\npython post_process/post_process_aqua_boundary.py \\\n  --config config/default.yaml \\\n  --aqua_tif preds/mosaic_aqua.tif \\\n  --boundary_tif preds/mosaic_boundary.tif \\\n  --output_path shp/\n```"),
    SETUP,
    md("## Debug Guide\n\n| Symptom | Likely Cause | Fix |\n|---------|-------------|-----|\n| IoU stuck at 0 | Wrong mask paths, empty masks | Check triplet filenames |\n| NaN loss | Bad normalization, all-nodata tiles | Check valid pixel fraction |\n| Merged ponds in output | Boundary head weak | More boundary labels, tune loss weight |\n| Seam artifacts | Overlap too low | Increase overlap_fraction |\n| CRS errors | SHP vs TIF mismatch | Reproject in tile_and_mask |\n| OOM on GPU | batch_size too large | Reduce to 1, enable AMP |\n\n## Extension Projects\n\n1. Add band 10 (NDVI) → change `band_indices`, retrain\n2. Swap UNet++ → DeepLab (model.py only)\n3. Add MLflow to train.py\n4. FastAPI wrapper around predict.py\n5. Train on new region — full pipeline rerun"),
    md("## Graduation Checklist\n\n- [ ] Explain dual-head design without notes\n- [ ] Walk through train.py from memory\n- [ ] Debug IoU = 0 scenario\n- [ ] Modify tile_and_mask for new bands\n- [ ] Docker batch inference on GeoTIFF folder\n- [ ] Design new GeoSpatial pipeline from scratch\n\n## Course Complete\n\n**Modules 00–12 finished.** You understand ML from first principles through production GeoSpatial AI.\n\nReturn to [Course README](../README.md) and update [PROGRESS.md](../PROGRESS.md)."),
    footer("You now own the water-bodies-detection pipeline — every file, every line.", None),
])


def main():
    print("Building Module 12 notebooks...")
    for name, cells in NOTEBOOKS:
        save(name, cells)
    print(f"Done: {len(NOTEBOOKS)} notebooks created.")


if __name__ == "__main__":
    main()
