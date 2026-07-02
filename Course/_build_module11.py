#!/usr/bin/env python3
"""Generate Module 11 Production ML notebooks (22 total)."""
import json
from pathlib import Path

M11 = Path(__file__).resolve().parent / "11_Production_ML"


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
    M11.mkdir(parents=True, exist_ok=True)
    (M11 / name).write_text(json.dumps(nb(cells), indent=1))
    print(f"  {name}")


def hdr(num, title, dur, objs):
    return md(
        f"# Notebook {num}: {title}\n\n**Module:** 11 Production ML  \n**Duration:** ~{dur}\n\n---\n\n## Learning Objectives\n\n{objs}"
    )


def footer(summary, nxt=None):
    nxt = f"\n\n**Next:** [{nxt}]({nxt})" if nxt else ""
    return md(f"---\n\n## Summary\n\n{summary}{nxt}")


SETUP = code(
    "import os\nimport json\nimport time\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport torch\nimport torch.nn as nn\n\nplt.rcParams['figure.figsize'] = (8, 5)\ntorch.manual_seed(42)\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nprint(f'Device: {device}')"
)

NOTEBOOKS = []


def register(name, cells):
    NOTEBOOKS.append((name, cells))


# ── DATA & TRAINING PIPELINES (01-05) ───────────────────────────────────

register("01_Data_Pipeline_Design.ipynb", [
    hdr("01", "Data Pipeline Design", "2.5 hrs",
        "1. Design reproducible ML data pipelines\n2. Understand ETL vs ELT for GeoSpatial data\n3. Map water-bodies tile_and_mask.py workflow\n4. Identify pipeline failure points"),
    md("## Production Data Pipeline\n\n```\nRaw data → Validate → Transform → Version → Train/Val split → Loader\n```\n\n**Principles:**\n- **Idempotent:** Re-run produces same output given same inputs\n- **Validated:** Schema checks at every stage\n- **Lineage:** Track which raw data produced which tiles\n- **Reproducible:** Fixed seeds, frozen configs"),
    SETUP,
    md("## water-bodies-detection Pipeline\n\n```\nPlanet GeoTIFF + shapefile\n  → tile_and_mask.py (GDAL/rasterio)\n     • Band selection [2,3,4,6,7,8]\n     • 512×512 tiles, 50% overlap\n     • Dual masks: aqua + boundary\n     • Negative tile sampling (20%)\n  → tiles/, masks_aqua/, masks_boundary/\n  → dataset.py (Albumentations)\n  → train.py\n```\n\nSee: `water-bodies-detection/tile_and_mask.py`"),
    code("# Pipeline stage checklist\ndef validate_pipeline_stages(stages):\n    for name, checks in stages.items():\n        print(f'{name}: {sum(checks)}/{len(checks)} checks passed')\n\nstages = {\n    'ingest': [True, True, False],  # example\n    'tile': [True, True, True],\n    'train': [True, False, True],\n}\nvalidate_pipeline_stages(stages)"),
    md("## GeoSpatial-Specific Concerns\n\n- CRS consistency (EPSG codes)\n- NoData handling in multispectral bands\n- Alpha band masking (cloud/invalid pixels)\n- meters_per_pixel for boundary width\n- Large raster memory — tile before load"),
    footer("Production data pipelines are idempotent, validated, and traceable.", "02_Training_Pipeline_Architecture.ipynb"),
])

register("02_Training_Pipeline_Architecture.ipynb", [
    hdr("02", "Config-Driven Training", "2.5 hrs",
        "1. Design YAML-driven training pipelines\n2. Separate code from hyperparameters\n3. Walk through water-bodies config/default.yaml\n4. Save frozen configs with each run"),
    md("## Config-Driven Design\n\n**Single source of truth:** All hyperparameters in YAML, not hardcoded.\n\n**Benefits:**\n- Reproduce any experiment from config file\n- Git-diff config changes between runs\n- Non-engineers can tune without code changes"),
    SETUP,
    code("import yaml\n\n# Mini version of water-bodies config structure\nconfig = {\n    'data': {'input_size': 512, 'batch_size': 2, 'band_indices': [2,3,4,6,7,8]},\n    'model': {'architecture': 'unetplusplus', 'encoder_name': 'se_resnet50'},\n    'training': {'stage1_epochs': 12, 'stage2_epochs': 120, 'early_stopping_patience': 18},\n}\nprint(yaml.dump(config, default_flow_style=False))\n\n# Load pattern from train.py\n# with open('config/default.yaml') as f:\n#     cfg = yaml.safe_load(f)"),
    md("## water-bodies config/default.yaml Sections\n\n| Section | Key params |\n|---------|------------|\n| `data` | input_size, band_indices, batch_size |\n| `tiling` | tile_size, overlap, boundary_width_meters |\n| `model` | unetplusplus, se_resnet50, out_channels=2 |\n| `training` | two-stage LR, early stopping, loss weights |\n| `inference` | thresholds, TTA, sliding window |\n\n**Frozen artifacts:** `train.py` saves `train_config.yaml` + `model_meta.json` per run."),
    footer("Config-driven training separates code from hyperparameters — essential for reproducibility.", "03_Experiment_Tracking.ipynb"),
])

register("03_Experiment_Tracking.ipynb", [
    hdr("03", "Experiment Tracking", "2 hrs",
        "1. Compare MLflow vs Weights & Biases\n2. Log params, metrics, artifacts\n3. Integrate tracking into training loop\n4. Design experiment naming conventions"),
    md("## Why Track Experiments?\n\nWithout tracking: \"Which run had IoU 0.82? What config? What weights?\"\n\n**Log per run:**\n- Hyperparameters (full YAML)\n- Metrics per epoch (loss, IoU, Dice)\n- Artifacts (best.pt, config, plots)\n- Environment (git commit, package versions)"),
    SETUP,
    code("# MLflow pattern (concept)\ntry:\n    import mlflow\n    mlflow.set_experiment('water-bodies')\n    with mlflow.start_run(run_name='unetpp-se_resnet50-v1'):\n        mlflow.log_params({'lr': 2e-4, 'batch_size': 2, 'encoder': 'se_resnet50'})\n        for epoch in range(3):\n            mlflow.log_metrics({'train_loss': 0.5 - epoch*0.1, 'val_iou': 0.6 + epoch*0.05}, step=epoch)\n        print('MLflow run logged')\nexcept ImportError:\n    print('Optional: pip install mlflow')\n\n# W&B: wandb.init(project='water-bodies', config=cfg)"),
    md("## water-bodies Integration\n\nAdd to `train.py`:\n```python\nmlflow.log_params(flatten_config(cfg))\nmlflow.log_metric('val_iou', val_iou, step=epoch)\nmlflow.log_artifact('best.pt')\n```\n\n**Naming:** `{project}-{encoder}-{date}-{hash}`"),
    footer("MLflow/W&B make experiments searchable and reproducible.", "04_Data_Version_Control.ipynb"),
])

register("04_Data_Version_Control.ipynb", [
    hdr("04", "Data Version Control (DVC)", "2 hrs",
        "1. Understand DVC for datasets and models\n2. Compare DVC vs git for large files\n3. Design data versioning for GeoSpatial tiles\n4. Know when DVC is worth the overhead"),
    md("## DVC (Data Version Control)\n\n**Problem:** Git cannot store 50GB of GeoTIFF tiles.\n\n**DVC solution:**\n- Store data in S3/GCS/local cache\n- Git tracks `.dvc` pointer files + `dvc.yaml` pipelines\n- `dvc repro` rebuilds pipeline from raw data\n\n```\nraw/ → dvc stage tile → tiles/ → dvc stage train → models/\n```"),
    SETUP,
    code("# dvc.yaml pipeline (conceptual)\npipeline = '''\nstages:\n  tile:\n    cmd: python tile_and_mask.py --config config/default.yaml\n    deps:\n      - tile_and_mask.py\n      - config/default.yaml\n    outs:\n      - tiles/\n      - masks_aqua/\n      - masks_boundary/\n  train:\n    cmd: python train.py --config config/default.yaml\n    deps:\n      - train.py\n      - tiles/\n    outs:\n      - runs/latest/\n'''\nprint(pipeline)"),
    md("## GeoSpatial Data Versioning\n\nVersion together:\n- Raw GeoTIFF + shapefile (DVC remote)\n- `config/default.yaml` (git)\n- Tile output hash (DVC outs)\n- Model weights (MLflow artifact or DVC)\n\n**Rule:** Never train on unversioned data."),
    footer("DVC versions datasets and pipelines; git versions code.", "05_TensorBoard.ipynb"),
])

register("05_TensorBoard.ipynb", [
    hdr("05", "TensorBoard Visualization", "1.5 hrs",
        "1. Log scalars, images, and histograms\n2. Compare multiple runs visually\n3. Add TensorBoard to PyTorch training\n4. Debug training with live curves"),
    md("## TensorBoard\n\nBuilt into PyTorch — no extra service needed for local dev.\n\n**Log:** loss curves, learning rate, IoU, sample predictions, weight histograms."),
    SETUP,
    code("from torch.utils.tensorboard import SummaryWriter\n\nwriter = SummaryWriter('runs/demo_water_bodies')\nfor step in range(50):\n    loss = 1.0 / (step + 1) + np.random.randn() * 0.02\n    iou = min(0.95, 0.3 + step * 0.012)\n    writer.add_scalar('Loss/train', loss, step)\n    writer.add_scalar('IoU/val', iou, step)\nwriter.close()\nprint('Run: tensorboard --logdir runs/demo_water_bodies')"),
    md("## water-bodies: Log These Metrics\n\n- `Loss/train`, `Loss/val` (AquaBoundaryLoss)\n- `IoU/aqua`, `IoU/boundary` per epoch\n- Sample tile predictions every N epochs\n- Learning rate schedule"),
    footer("TensorBoard gives live training visibility — use from day one.", "06_Mixed_Precision_Training.ipynb"),
])

# ── MODEL OPTIMIZATION (06-12) ──────────────────────────────────────────

register("06_Mixed_Precision_Training.ipynb", [
    hdr("06", "Mixed Precision Training (AMP)", "2 hrs",
        "1. Understand FP16 vs FP32 training\n2. Use torch.cuda.amp.autocast and GradScaler\n3. Know when AMP helps GeoSpatial training\n4. Avoid numerical instability"),
    md("## Automatic Mixed Precision\n\n**FP16:** Half precision — 2× faster, half memory on Tensor Cores.\n\n**GradScaler:** Prevents gradient underflow in FP16.\n\n```python\nscaler = torch.cuda.amp.GradScaler()\nwith torch.cuda.amp.autocast():\n    loss = model(x, y)\nscaler.scale(loss).backward()\nscaler.step(optimizer)\nscaler.update()\n```"),
    SETUP,
    code("scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')\nmodel = nn.Linear(64, 2).to(device)\nopt = torch.optim.Adam(model.parameters())\nx = torch.randn(8, 64, device=device)\ny = torch.randint(0, 2, (8,), device=device)\n\nfor _ in range(3):\n    opt.zero_grad()\n    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):\n        loss = nn.CrossEntropyLoss()(model(x), y)\n    scaler.scale(loss).backward()\n    scaler.step(opt)\n    scaler.update()\nprint(f'AMP training step OK, loss={loss.item():.4f}')"),
    footer("AMP speeds training ~1.5–2× on modern GPUs with minimal accuracy loss.", "07_Distributed_Training.ipynb"),
])

register("07_Distributed_Training.ipynb", [
    hdr("07", "Distributed Training (DDP)", "2 hrs",
        "1. Understand data parallel vs model parallel\n2. Know PyTorch DDP basics\n3. Scale water-bodies training to multi-GPU\n4. Identify when distribution is worth it"),
    md("## Distributed Data Parallel (DDP)\n\nEach GPU:\n- Gets a full model copy\n- Processes different batch shard\n- Gradients all-reduced across GPUs\n\n**Launch:**\n```bash\ntorchrun --nproc_per_node=2 train.py --config config/default.yaml\n```\n\n**When worth it:** Large dataset, batch size limited by GPU memory."),
    SETUP,
    code("# DDP pattern (concept — single process demo)\nprint('DDP requires torchrun with multiple processes')\nprint('Key changes: DistributedSampler, model = DDP(model), set_epoch(sampler)')\nprint('Effective batch = batch_size * num_gpus')"),
    md("## water-bodies: batch_size=2 is small\n\nMulti-GPU lets effective batch=8 with 4 GPUs — may stabilize AquaBoundaryLoss training on heterogeneous tiles."),
    footer("DDP scales training horizontally; use when single-GPU batch is too small.", "08_ONNX_Export.ipynb"),
])

register("08_ONNX_Export.ipynb", [
    hdr("08", "ONNX Export", "2 hrs",
        "1. Export PyTorch model to ONNX\n2. Validate ONNX vs PyTorch outputs\n3. Deploy ONNX with ONNX Runtime\n4. Plan water-bodies ONNX export"),
    md("## ONNX (Open Neural Network Exchange)\n\n**Why:** Deploy same model in PyTorch, C++, TensorRT, mobile, browser.\n\n**Export:**\n```python\ntorch.onnx.export(model, dummy_input, 'model.onnx',\n    input_names=['image'], output_names=['logits'],\n    dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}})\n```"),
    SETUP,
    code("class TinySeg(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.net = nn.Sequential(nn.Conv2d(6, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 2, 1))\n    def forward(self, x):\n        return self.net(x)\n\nmodel = TinySeg().eval()\ndummy = torch.randn(1, 6, 128, 128)\nout_pt = model(dummy)\n\ntry:\n    torch.onnx.export(model, dummy, '/tmp/tiny_seg.onnx', input_names=['image'], output_names=['logits'])\n    import onnxruntime as ort\n    sess = ort.InferenceSession('/tmp/tiny_seg.onnx')\n    out_onnx = sess.run(None, {'image': dummy.numpy()})[0]\n    print(f'Max diff PT vs ONNX: {np.abs(out_pt.detach().numpy() - out_onnx).max():.6f}')\nexcept ImportError:\n    print('Optional: pip install onnx onnxruntime')"),
    footer("ONNX enables cross-framework deployment and TensorRT conversion.", "09_TensorRT.ipynb"),
])

register("09_TensorRT.ipynb", [
    hdr("09", "TensorRT Optimization", "2 hrs",
        "1. Understand TensorRT graph optimization\n2. Convert ONNX → TensorRT engine\n3. Benchmark inference latency\n4. Evaluate for production GeoSpatial batch jobs"),
    md("## TensorRT (NVIDIA)\n\n**Optimizations:** Layer fusion, kernel auto-tuning, FP16/INT8 precision.\n\n**Pipeline:** PyTorch → ONNX → TensorRT engine → inference\n\n**Typical speedup:** 2–5× over PyTorch eager mode on GPU."),
    SETUP,
    code("# Benchmark pattern\nmodel = nn.Sequential(nn.Conv2d(6, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 2, 1)).eval().to(device)\nx = torch.randn(1, 6, 512, 512, device=device)\n\n# Warmup\nfor _ in range(10):\n    with torch.no_grad():\n        model(x)\n\nif device.type == 'cuda':\n    torch.cuda.synchronize()\n    t0 = time.perf_counter()\n    for _ in range(50):\n        with torch.no_grad():\n            model(x)\n    torch.cuda.synchronize()\n    ms = (time.perf_counter() - t0) / 50 * 1000\n    print(f'PyTorch FP32: {ms:.1f} ms/image (512×512)')\nelse:\n    print('TensorRT requires NVIDIA GPU — benchmark on deployment hardware')"),
    footer("TensorRT maximizes GPU inference throughput for production.", "10_Quantization.ipynb"),
])

register("10_Quantization.ipynb", [
    hdr("10", "Quantization (INT8)", "2 hrs",
        "1. Understand post-training vs quantization-aware training\n2. Apply dynamic quantization in PyTorch\n3. Measure accuracy vs speed tradeoff\n4. Know when INT8 is safe for segmentation"),
    md("## Quantization\n\n**FP32 → INT8:** 4× smaller model, faster inference, slight accuracy drop.\n\n**Dynamic quantization:** Weights INT8, activations FP32 (easiest).\n\n**QAT:** Train with fake quant nodes — best accuracy.\n\n**Segmentation caution:** Small IoU drops matter for GIS boundaries — always validate on holdout tiles."),
    SETUP,
    code("model = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 2))\nmodel_quant = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)\nprint('Dynamic quant applied to Linear layers')\nx = torch.randn(4, 256)\nwith torch.no_grad():\n    print('Output shape:', model_quant(x).shape)"),
    footer("Quantization trades small accuracy for major inference speed gains.", "11_Pruning.ipynb"),
])

register("11_Pruning.ipynb", [
    hdr("11", "Pruning", "1.5 hrs",
        "1. Distinguish structured vs unstructured pruning\n2. Apply magnitude pruning in PyTorch\n3. Fine-tune after pruning\n4. Evaluate size/speed gains"),
    md("## Pruning\n\n**Unstructured:** Zero individual weights — needs sparse hardware support.\n\n**Structured:** Remove entire channels/filters — direct speedup on standard GPUs.\n\n**Workflow:** Train → prune → fine-tune → export."),
    SETUP,
    code("import torch.nn.utils.prune as prune\nlayer = nn.Conv2d(32, 64, 3, padding=1)\nprune.l1_unstructured(layer, name='weight', amount=0.3)\nsparsity = (layer.weight == 0).float().mean().item()\nprint(f'Sparsity after 30% prune: {sparsity:.1%}')"),
    footer("Pruning reduces model size; structured pruning gives real speedups.", "12_Knowledge_Distillation.ipynb"),
])

register("12_Knowledge_Distillation.ipynb", [
    hdr("12", "Knowledge Distillation", "2 hrs",
        "1. Understand teacher-student training\n2. Derive distillation loss with temperature\n3. Compress UNet++ teacher to smaller student\n4. Connect to DINO self-distillation (Module 10)"),
    md("## Knowledge Distillation\n\n**Teacher:** Large accurate model (UNet++ SE-ResNet50)\n\n**Student:** Smaller fast model (UNet++ MobileNet encoder)\n\n$$L = \\alpha L_{hard} + (1-\\alpha) T^2 \\cdot KL(\\text{softmax}(z_s/T), \\text{softmax}(z_t/T))$$\n\n**Temperature T:** Softens probability distributions — transfers dark knowledge."),
    SETUP,
    code("import torch.nn.functional as F\n\ndef distillation_loss(student_logits, teacher_logits, targets, T=4.0, alpha=0.5):\n    hard = nn.CrossEntropyLoss()(student_logits, targets)\n    soft = nn.KLDivLoss(reduction='batchmean')(\n        F.log_softmax(student_logits / T, dim=1),\n        F.softmax(teacher_logits / T, dim=1),\n    ) * (T * T)\n    return alpha * hard + (1 - alpha) * soft\n\ns = torch.randn(4, 10)\nt = torch.randn(4, 10)\ny = torch.randint(0, 10, (4,))\nprint(f'Distillation loss: {distillation_loss(s, t, y):.4f}')"),
    footer("Distillation compresses teacher knowledge into deployable student models.", "13_Docker_for_ML.ipynb"),
])

# ── DEPLOYMENT (13-17) ──────────────────────────────────────────────────

register("13_Docker_for_ML.ipynb", [
    hdr("13", "Docker for ML", "2.5 hrs",
        "1. Containerize ML inference pipelines\n2. Walk through water-bodies Dockerfile\n3. Handle GDAL/rasterio native dependencies\n4. Build reproducible deployment images"),
    md("## Docker for ML\n\n**Why containers:** Same environment dev → staging → production.\n\n**water-bodies-detection Dockerfile:**\n```dockerfile\nFROM python:3.11-slim\nRUN apt-get install gdal-bin libgdal-dev libgeos-dev ...\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY . /app\nCMD [\"python\", \"predict.py\", \"--help\"]\n```\n\n**GeoSpatial deps:** GDAL, PROJ, GEOS — must match host or run fully containerized."),
    SETUP,
    code("dockerfile = open('../../water-bodies-detection/Dockerfile').read() if os.path.exists('../../water-bodies-detection/Dockerfile') else 'See water-bodies-detection/Dockerfile'\nprint(dockerfile[:800] if isinstance(dockerfile, str) else dockerfile)"),
    md("## Best Practices\n\n- Pin Python + package versions in requirements.txt\n- Multi-stage builds for smaller images\n- Mount data volumes, not COPY large GeoTIFFs\n- GPU: `nvidia/cuda` base + `--gpus all`"),
    footer("Docker packages code + deps + native libs for reproducible deployment.", "14_REST_API_FastAPI.ipynb"),
])

register("14_REST_API_FastAPI.ipynb", [
    hdr("14", "REST API with FastAPI", "2.5 hrs",
        "1. Build inference API with FastAPI\n2. Handle file upload and async inference\n3. Return GeoTIFF or GeoJSON responses\n4. Add health checks and model versioning"),
    md("## FastAPI for ML\n\n```python\nfrom fastapi import FastAPI, UploadFile\napp = FastAPI()\n\n@app.post('/predict')\nasync def predict(file: UploadFile):\n    # load model, run inference, return mask URL or base64\n    return {'status': 'ok', 'iou_estimate': 0.85}\n\n@app.get('/health')\ndef health():\n    return {'model_version': 'v1.2.0'}\n```\n\n**Production additions:** auth, rate limiting, request queue, GPU worker pool."),
    SETUP,
    code("try:\n    from fastapi import FastAPI\n    from pydantic import BaseModel\n\n    class PredictResponse(BaseModel):\n        status: str\n        num_detections: int\n\n    app = FastAPI(title='Water Bodies API')\n\n    @app.get('/health')\n    def health():\n        return {'status': 'healthy', 'model': 'unetpp-se_resnet50'}\n\n    print('FastAPI app defined — run: uvicorn notebook:app --reload')\nexcept ImportError:\n    print('Optional: pip install fastapi uvicorn')"),
    footer("FastAPI serves ML models over HTTP — standard for microservice deployment.", "15_Triton_Inference_Server.ipynb"),
])

register("15_Triton_Inference_Server.ipynb", [
    hdr("15", "Triton Inference Server", "2 hrs",
        "1. Understand NVIDIA Triton architecture\n2. Deploy ONNX/TensorRT models\n3. Dynamic batching for throughput\n4. Compare Triton vs FastAPI for batch GeoSpatial"),
    md("## Triton Inference Server\n\n**Features:**\n- Multi-model serving (PyTorch, ONNX, TensorRT)\n- Dynamic batching — groups requests for GPU efficiency\n- Model ensemble pipelines\n- Metrics endpoint for Prometheus\n\n**When to use:** High-throughput GPU serving, multiple models, SLA requirements.\n\n**When FastAPI suffices:** Low QPS, simple single-model API."),
    SETUP,
    md("## water-bodies on Triton\n\n```\nmodels/\n  water_seg/\n    config.pbtxt\n    1/model.plan  # TensorRT engine\n```\n\nBatch automate/ scripts can call Triton gRPC instead of local predict.py for farm-scale inference."),
    footer("Triton optimizes GPU throughput with dynamic batching and multi-model serving.", "16_Model_Versioning.ipynb"),
])

register("16_Model_Versioning.ipynb", [
    hdr("16", "Model Versioning & Registry", "2 hrs",
        "1. Design model registry patterns\n2. Use semantic versioning for models\n3. Walk through model_meta.json in water-bodies\n4. Implement rollback strategy"),
    md("## Model Registry\n\n**Track per version:**\n- Weights file (best.pt)\n- Training config (frozen YAML)\n- Metrics (val IoU, Dice)\n- Data hash / DVC revision\n- Git commit SHA\n- Deployment status (staging/prod)\n\n**water-bodies:** `model_meta.json` in each run folder documents architecture, bands, thresholds."),
    SETUP,
    code("model_meta = {\n    'architecture': 'unetplusplus',\n    'encoder': 'se_resnet50',\n    'in_channels': 6,\n    'band_indices': [2, 3, 4, 6, 7, 8],\n    'out_channels': 2,\n    'val_iou_aqua': 0.87,\n    'train_config': 'train_config.yaml',\n    'created_at': '2026-01-15T10:30:00Z',\n    'git_commit': 'abc1234',\n}\nprint(json.dumps(model_meta, indent=2))"),
    md("## Rollback\n\nIf production IoU drops → redeploy previous registry version → alert team → investigate drift."),
    footer("Model registry enables traceability, rollback, and staged deployment.", "17_Batch_Inference.ipynb"),
])

register("17_Batch_Inference.ipynb", [
    hdr("17", "Batch Inference Pipelines", "2.5 hrs",
        "1. Design batch inference for large GeoSpatial jobs\n2. Walk through automate_water_predictions.py\n3. Handle failures, retries, and logging\n4. Optimize throughput vs memory"),
    md("## Batch Inference\n\n**water-bodies automate/automate_water_predictions.py:**\n```\nFor each GeoTIFF in input_dir:\n  → subprocess predict.py\n  → write stem_aqua.tif, stem_boundary.tif\n  → log success/failure\n```\n\n**Patterns:**\n- Idempotent outputs (skip if exists)\n- Parallel workers (careful with GPU memory)\n- Checkpoint/resume on failure\n- Structured logs per file"),
    SETUP,
    code("# Batch job tracker (concept)\nfiles = ['tile_001.tif', 'tile_002.tif', 'tile_003.tif']\nstatus = {}\nfor f in files:\n    try:\n        # predict(f)\n        status[f] = 'ok'\n    except Exception as e:\n        status[f] = f'fail: {e}'\nok = sum(1 for v in status.values() if v == 'ok')\nprint(f'Batch complete: {ok}/{len(files)} succeeded')"),
    md("## Post-Process Chain\n\n```\nautomate_water_predictions.py → probability GeoTIFFs\nautomate_water_postprocess.py → polygons / shapefiles\npost_process_aqua_boundary.py   → GIS-ready outputs\n```"),
    footer("Batch pipelines process production volumes with logging and fault tolerance.", "18_Logging_and_Monitoring.ipynb"),
])

# ── MONITORING & CI/CD (18-22) ──────────────────────────────────────────

register("18_Logging_and_Monitoring.ipynb", [
    hdr("18", "Logging and Monitoring", "2 hrs",
        "1. Structured logging for ML pipelines\n2. Integrate Prometheus/Grafana basics\n3. Log inference latency and errors\n4. Design alert thresholds"),
    md("## Production Logging\n\n**Structured JSON logs:**\n```json\n{\"event\": \"inference\", \"model\": \"v1.2\", \"tile\": \"pond_042\", \"latency_ms\": 340, \"status\": \"ok\"}\n```\n\n**Monitor:** request rate, p95 latency, error rate, GPU utilization."),
    SETUP,
    code("import logging\nlogging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')\nlog = logging.getLogger('water_bodies')\nlog.info('inference_complete', extra={'tile': 'demo', 'iou': 0.85, 'ms': 120})"),
    footer("Structured logs enable search, alerts, and post-incident debugging.", "19_Data_Drift_Detection.ipynb"),
])

register("19_Data_Drift_Detection.ipynb", [
    hdr("19", "Data Drift Detection", "2 hrs",
        "1. Define covariate vs label drift\n2. Monitor input distribution shifts\n3. Detect seasonal/satellite drift in GeoSpatial data\n4. Trigger retraining workflows"),
    md("## Data Drift\n\n**Covariate drift:** Input distribution changes (new satellite sensor, season, region).\n\n**Label drift:** P(y) changes (new pond types, drought).\n\n**Detection methods:**\n- PSI (Population Stability Index) on band statistics\n- KS test on NDVI/NDWI distributions\n- Embedding distance (DINO features)\n\n**GeoSpatial triggers:** New Planet collection, different sun angle, new geography."),
    SETUP,
    code("def psi(expected, actual, bins=10):\n    \"\"\"Population Stability Index — >0.25 often flagged.\"\"\"\n    e_hist, _ = np.histogram(expected, bins=bins, density=True)\n    a_hist, _ = np.histogram(actual, bins=bins, density=True)\n    e_hist = np.clip(e_hist, 1e-6, None)\n    a_hist = np.clip(a_hist, 1e-6, None)\n    return float(np.sum((a_hist - e_hist) * np.log(a_hist / e_hist)))\n\ntrain_bands = np.random.normal(0.3, 0.1, 1000)\nnew_bands = np.random.normal(0.45, 0.15, 500)  # shifted distribution\nprint(f'PSI (shifted): {psi(train_bands, new_bands):.3f}')"),
    footer("Drift detection triggers retraining before production quality degrades.", "20_Model_Performance_Monitoring.ipynb"),
])

register("20_Model_Performance_Monitoring.ipynb", [
    hdr("20", "Model Performance Monitoring", "2 hrs",
        "1. Monitor production IoU/precision proxies\n2. Human-in-the-loop validation sampling\n3. Detect model degradation over time\n4. Design feedback loops for retraining"),
    md("## Performance Monitoring\n\n**Direct metrics (if labels available):** Sample tiles → manual QA → track IoU weekly.\n\n**Proxy metrics (no labels):**\n- Prediction confidence distribution\n- Fraction of empty predictions\n- Polygon area statistics vs historical\n- User correction rate in GIS tool\n\n**water-bodies:** Compare post-process polygon counts vs known aquaculture registers."),
    SETUP,
    code("# Weekly QA sample tracker\nqa_results = [{'week': 1, 'sample_iou': 0.86}, {'week': 2, 'sample_iou': 0.84}, {'week': 3, 'sample_iou': 0.79}]\nweeks = [r['week'] for r in qa_results]\nious = [r['sample_iou'] for r in qa_results]\nplt.plot(weeks, ious, 'o-'); plt.axhline(0.82, color='r', ls='--', label='Alert threshold')\nplt.xlabel('Week'); plt.ylabel('Sample IoU'); plt.legend(); plt.title('Production QA trend'); plt.show()"),
    footer("Monitor production metrics continuously; degrade triggers investigation.", "21_CICD_for_ML.ipynb"),
])

register("21_CICD_for_ML.ipynb", [
    hdr("21", "CI/CD for ML", "2.5 hrs",
        "1. Design ML CI/CD pipelines\n2. Separate CI (tests) from CD (deploy)\n3. Gate deployment on metric thresholds\n4. Automate Docker build and push"),
    md("## ML CI/CD Pipeline\n\n```\nPush code → CI:\n  - lint, unit tests\n  - smoke train (1 epoch)\n  - inference test on fixture tile\n  - compare IoU vs baseline\n\nMerge → CD (if metrics pass):\n  - build Docker image\n  - push to registry\n  - deploy to staging\n  - run integration tests\n  - promote to prod\n```\n\n**GitHub Actions / GitLab CI** — same as software, plus model metric gates."),
    SETUP,
    code("ci_pipeline = '''\nname: water-bodies-ci\non: [push, pull_request]\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - run: pip install -r requirements.txt\n      - run: pytest tests/\n      - run: python predict.py --help\n  smoke-train:\n    runs-on: [self-hosted, gpu]\n    steps:\n      - run: python train.py --epochs 1 --config config/smoke.yaml\n'''\nprint(ci_pipeline)"),
    footer("CI/CD automates testing and deployment with metric gates.", "22_GPU_Optimization_and_Capstone.ipynb"),
])

register("22_GPU_Optimization_and_Capstone.ipynb", [
    hdr("22", "GPU Optimization & Module Capstone", "2.5 hrs",
        "1. Profile GPU bottlenecks\n2. Optimize data loading and inference\n3. Review full water-bodies production stack\n4. Complete Module 11"),
    md("## GPU Optimization Checklist\n\n- [ ] `pin_memory=True`, `num_workers>0` in DataLoader\n- [ ] AMP training (Notebook 06)\n- [ ] `torch.inference_mode()` at predict time\n- [ ] Batch tiles when GPU memory allows\n- [ ] ONNX/TensorRT for production inference\n- [ ] Avoid CPU↔GPU copies in hot loop\n- [ ] Profile with `torch.profiler`"),
    SETUP,
    code("if device.type == 'cuda':\n    print(f'GPU: {torch.cuda.get_device_name(0)}')\n    print(f'Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB')\nelse:\n    print('Profile on target deployment GPU')"),
    md("## water-bodies Production Stack (Full)\n\n| Layer | Component |\n|-------|----------|\n| Config | `config/default.yaml` |\n| Data | `tile_and_mask.py` → DVC (optional) |\n| Train | `train.py` + TensorBoard + MLflow |\n| Track | `model_meta.json`, frozen config |\n| Infer | `predict.py` sliding window + TTA |\n| Batch | `automate/automate_water_predictions.py` |\n| Post | `post_process/post_process_aqua_boundary.py` |\n| Deploy | Docker + FastAPI or batch cron |\n| Monitor | Logs + drift PSI + weekly QA IoU |\n| CI/CD | pytest + smoke train + metric gate |"),
    md("## Module 11 Assignment\n\n1. Add MLflow logging to a training script\n2. Export a model to ONNX and benchmark latency\n3. Write a FastAPI `/predict` stub\n4. Document Docker build steps for water-bodies\n\nSee `exercises/README.md`.\n\n**Next:** Module 12 Capstone — full line-by-line water-bodies-detection walkthrough."),
    code("# YOUR CODE HERE — production readiness checklist for your deployment\n"),
    footer("Production ML = pipelines + optimization + deployment + monitoring. Module 12 applies it all.", None),
])


def main():
    print("Building Module 11 notebooks...")
    for name, cells in NOTEBOOKS:
        save(name, cells)
    print(f"Done: {len(NOTEBOOKS)} notebooks created.")


if __name__ == "__main__":
    main()
