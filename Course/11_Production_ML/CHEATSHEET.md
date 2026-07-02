# Module 11 Cheat Sheet — Production ML

## Production Stack Layers

```
Data → Train → Track → Optimize → Deploy → Monitor → CI/CD
```

## Data Pipelines

| Principle | Meaning |
|-----------|---------|
| Idempotent | Same input → same output |
| Validated | Schema checks at each stage |
| Lineage | Raw → tiles → model traceable |
| Reproducible | Seeds + frozen config |

**water-bodies:** `tile_and_mask.py` → `tiles/`, `masks_aqua/`, `masks_boundary/`

## Config-Driven Training

```yaml
# config/default.yaml sections
data:       # bands, batch_size, input_size
tiling:     # tile_size, overlap, boundary_width_m
model:      # unetplusplus, se_resnet50
training:   # two-stage LR, early stopping
inference:  # thresholds, TTA
```

**Artifacts:** `train_config.yaml`, `model_meta.json`, `best.pt`

## Experiment Tracking

| Tool | Use |
|------|-----|
| MLflow | Params, metrics, artifacts, registry |
| W&B | Rich dashboards, team collaboration |
| TensorBoard | Local scalars, images, histograms |

```python
mlflow.log_params(cfg)
mlflow.log_metric('val_iou', iou, step=epoch)
mlflow.log_artifact('best.pt')
```

## DVC Pipeline

```yaml
stages:
  tile:
    cmd: python tile_and_mask.py
    outs: [tiles/, masks_aqua/]
  train:
    cmd: python train.py
    outs: [runs/latest/]
```

## Optimization

| Technique | Benefit | Notebook |
|-----------|---------|----------|
| AMP (FP16) | 1.5–2× train speed | 06 |
| DDP | Multi-GPU scale | 07 |
| ONNX | Cross-platform deploy | 08 |
| TensorRT | 2–5× inference | 09 |
| INT8 quant | 4× smaller, faster | 10 |
| Pruning | Smaller model | 11 |
| Distillation | Small student model | 12 |

### AMP Pattern
```python
scaler = torch.amp.GradScaler('cuda')
with torch.amp.autocast('cuda'):
    loss = model(x, y)
scaler.scale(loss).backward()
scaler.step(opt); scaler.update()
```

### ONNX Export
```python
torch.onnx.export(model, dummy, 'model.onnx',
    input_names=['image'], output_names=['logits'])
```

## Deployment

| Method | When |
|--------|------|
| **Docker** | GDAL + PyTorch reproducible env |
| **FastAPI** | REST API, low-medium QPS |
| **Triton** | High GPU throughput, dynamic batch |
| **Batch scripts** | Farm-scale GeoTIFF processing |

**water-bodies Dockerfile:** python:3.11-slim + GDAL + requirements.txt

**Batch:** `automate/automate_water_predictions.py`

## Model Registry

Track per version:
- Weights, config, metrics, data hash, git SHA, deploy status

## Monitoring

| Signal | Method |
|--------|--------|
| Data drift | PSI on band stats, KS test |
| Model degrade | Weekly QA sample IoU |
| Inference | p95 latency, error rate |
| Alerts | IoU < threshold → rollback |

**PSI > 0.25** → investigate drift

## CI/CD Gates

```
PR → lint + pytest + smoke train (1 epoch) + inference test
Merge → Docker build → staging → metric gate → prod
```

## water-bodies Full Stack

| File | Role |
|------|------|
| `config/default.yaml` | Single source of truth |
| `tile_and_mask.py` | Data pipeline |
| `train.py` | Two-stage training |
| `predict.py` | Sliding window + TTA |
| `automate/*.py` | Batch production |
| `post_process/*.py` | GIS polygons |
| `Dockerfile` | Container deploy |
| `model_meta.json` | Version metadata |

## GPU Checklist

- [ ] pin_memory, num_workers
- [ ] AMP training
- [ ] inference_mode() at predict
- [ ] Profile with torch.profiler
- [ ] ONNX/TensorRT for prod

## Common Mistakes

- Hardcoded hyperparameters (no config)
- Unversioned training data
- No metric logging → can't reproduce best run
- Docker missing GDAL deps
- Deploying without ONNX validation
- No drift monitoring on new satellite collections
- No rollback plan for model versions

## Interview Highlights

1. CI/CD vs CD for ML
2. When AMP vs full FP32
3. ONNX vs TensorRT tradeoffs
4. Data drift detection methods
5. Model registry design
