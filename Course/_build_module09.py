#!/usr/bin/env python3
"""Generate Module 09 Instance Segmentation notebooks (7 total)."""
import json
from pathlib import Path

M09 = Path(__file__).resolve().parent / "09_Instance_Segmentation"


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
    M09.mkdir(parents=True, exist_ok=True)
    (M09 / name).write_text(json.dumps(nb(cells), indent=1))
    print(f"  {name}")


def hdr(num, title, dur, objs):
    return md(
        f"# Notebook {num}: {title}\n\n**Module:** 09 Instance Segmentation  \n**Duration:** ~{dur}\n\n---\n\n## Learning Objectives\n\n{objs}"
    )


def footer(summary, nxt=None):
    nxt = f"\n\n**Next:** [{nxt}]({nxt})" if nxt else ""
    return md(f"---\n\n## Summary\n\n{summary}{nxt}")


SETUP = code(
    "import numpy as np\nimport matplotlib.pyplot as plt\nimport matplotlib.patches as patches\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nplt.rcParams['figure.figsize'] = (8, 5)\ntorch.manual_seed(42)\nrng = np.random.default_rng(42)\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nprint(f'Device: {device}')"
)

MASK_IOU_CODE = code(
    "def mask_iou(pred, target):\n    \"\"\"Binary masks (H,W) or (N,H,W). Returns scalar or per-instance IoU.\"\"\"\n    pred = pred.bool()\n    target = target.bool()\n    if pred.dim() == 2:\n        pred, target = pred.unsqueeze(0), target.unsqueeze(0)\n    inter = (pred & target).float().sum(dim=(1, 2))\n    union = (pred | target).float().sum(dim=(1, 2))\n    return inter / (union + 1e-7)"
)

NOTEBOOKS = []


def register(name, cells):
    NOTEBOOKS.append((name, cells))


register("01_Instance_vs_Semantic_vs_Panoptic.ipynb", [
    hdr("01", "Instance vs Semantic vs Panoptic", "2.5 hrs",
        "1. Compare three segmentation paradigms\n2. Understand instance IDs vs class labels\n3. Analyze adjacent-pond problem\n4. Compare dual-head vs instance segmentation strategies"),
    md("## Three Segmentation Paradigms\n\n| Type | Output | Separates instances? | Example |\n|------|--------|---------------------|--------|\n| **Semantic** | Class per pixel | No | All water = class 1 |\n| **Instance** | Mask per object + ID | Yes | Pond 1, Pond 2, Pond 3 |\n| **Panoptic** | Semantic (stuff) + instance (things) | Yes for things | Road + building₁ + building₂ |\n\n**Stuff vs Things:**\n- **Stuff:** Amorphous regions (sky, road, water) — semantic only\n- **Things:** Countable objects (car, pond, building) — need instance IDs"),
    SETUP,
    code("# Simulate adjacent ponds: semantic merges, instance separates\nH, W = 64, 128\nsemantic = torch.zeros(H, W, dtype=torch.long)\nsemantic[:, 20:45] = 1  # one water blob (two ponds merged)\n\ninst1 = torch.zeros(H, W, dtype=torch.bool)\ninst1[:, 20:32] = True\ninst2 = torch.zeros(H, W, dtype=torch.bool)\ninst2[:, 33:45] = True\n\nfig, axes = plt.subplots(1, 3, figsize=(12, 3))\naxes[0].imshow(semantic, cmap='Blues'); axes[0].set_title('Semantic (merged)'); axes[0].axis('off')\naxes[1].imshow(inst1.numpy(), cmap='Blues'); axes[1].set_title('Instance 1'); axes[1].axis('off')\naxes[2].imshow(inst2.numpy(), cmap='Oranges'); axes[2].set_title('Instance 2'); axes[2].axis('off')\nplt.tight_layout(); plt.show()"),
    md("## Your water-bodies-detection vs Instance Seg\n\n**Instance seg (Mask R-CNN):** Predict separate mask per pond instance.\n\n**Your approach (Module 07):** Dual-head semantic — aqua interior + bund boundary — separates adjacent ponds without instance IDs.\n\n| Approach | Pros | Cons |\n|----------|------|------|\n| Dual-head boundary | Simpler labels, fast inference | No instance ID for tracking |\n| Mask R-CNN | Unique ID per pond | Needs instance annotations |\n| Panoptic | Full scene understanding | Heavy compute, complex labels |"),
    footer("Instance seg assigns unique masks; panoptic unifies stuff and things.", "02_Mask_R_CNN.ipynb"),
])

register("02_Mask_R_CNN.ipynb", [
    hdr("02", "Mask R-CNN", "3 hrs",
        "1. Understand Mask R-CNN architecture\n2. Know mask head vs box head\n3. Run torchvision Mask R-CNN\n4. Evaluate with mask IoU"),
    md("## Mask R-CNN (He et al., 2017)\n\n**Extends Faster R-CNN** with a parallel **mask head**:\n\n```\nImage → Backbone → FPN\n         ├─ RPN → RoI boxes\n         ├─ Box head → class + box refine\n         └─ Mask head → K×m×m binary masks (K classes)\n```\n\n**Key design:** Predict **class-agnostic** mask (one mask per RoI), then classify separately.\n\n**Mask loss:** Per-pixel sigmoid CE on positive RoIs only (typically 28×28 resolution)."),
    SETUP, MASK_IOU_CODE,
    code("try:\n    from torchvision.models.detection import maskrcnn_resnet50_fpn\n    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor\n    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor\n\n    def build_mask_rcnn(num_classes=2):\n        model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)\n        in_features = model.roi_heads.box_predictor.cls_score.in_features\n        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)\n        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels\n        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)\n        return model\n\n    model = build_mask_rcnn(num_classes=2)\n    model.eval()\n    images = [torch.rand(3, 256, 256)]\n    with torch.no_grad():\n        out = model(images)\n    print('Output keys:', out[0].keys())\n    print('Boxes:', out[0]['boxes'].shape)\n    print('Masks:', out[0]['masks'].shape)  # (N, 1, H, W)\nexcept ImportError as e:\n    print('torchvision detection required:', e)"),
    md("## Mask IoU\n\nSame as box IoU but on pixel masks:\n\n$$\\text{maskIoU} = \\frac{|M_{pred} \\cap M_{gt}|}{|M_{pred} \\cup M_{gt}|}$$\n\nCOCO instance seg benchmark uses mask AP (AP computed on masks, not boxes)."),
    code("# Mask IoU example\npred_m = torch.zeros(64, 64); pred_m[10:30, 10:30] = 1\ngt_m = torch.zeros(64, 64); gt_m[12:32, 12:32] = 1\nprint(f'Mask IoU: {mask_iou(pred_m, gt_m).item():.3f}')"),
    md("## GeoSpatial: Individual Pond Instance Segmentation\n\nAnnotate each pond polygon with unique instance ID → train Mask R-CNN → get separate masks for adjacent ponds."),
    footer("Mask R-CNN = Faster R-CNN + FCN mask head per RoI.", "03_YOLACT.ipynb"),
])

register("03_YOLACT.ipynb", [
    hdr("03", "YOLACT", "2 hrs",
        "1. Understand real-time instance segmentation\n2. Know prototype mask coefficients approach\n3. Compare speed vs Mask R-CNN"),
    md("## YOLACT (Bolya et al., 2019)\n\n**Real-time instance segmentation** by factorizing masks:\n\n$$M_i = \\sigma\\left(\\sum_k c_{i,k} \\cdot P_k\\right)$$\n\n- **Prototype masks** $P_k$: $K$ shared basis masks (e.g. 32 prototypes, H×W)\n- **Coefficients** $c_i$: per-instance vector from detection branch\n- **Assembly:** Linear combination → instance mask\n\n**Benefits:** Single forward pass like YOLO; much faster than Mask R-CNN.\n\n**Trade-off:** Slightly lower mAP on COCO vs Mask R-CNN."),
    SETUP,
    code("# YOLACT mask assembly (concept)\nK, H, W = 4, 32, 32\nprototypes = torch.randn(K, H, W)\ncoeffs = torch.tensor([0.8, -0.3, 0.5, 0.1])  # instance 1\nmask = torch.sigmoid((coeffs[:, None, None] * prototypes).sum(0))\nplt.imshow(mask.numpy(), cmap='Blues'); plt.title('Assembled instance mask'); plt.colorbar(); plt.show()"),
    md("## When to Use YOLACT\n\n- Video / real-time aquaculture monitoring\n- Edge deployment where Mask R-CNN too slow\n- Acceptable if mask boundaries less precise than UNet++"),
    footer("YOLACT assembles instance masks from shared prototypes — fast single-pass.", "04_SOLO_SOLOv2.ipynb"),
])

register("04_SOLO_SOLOv2.ipynb", [
    hdr("04", "SOLO / SOLOv2", "2 hrs",
        "1. Understand direct instance segmentation without proposals\n2. Know grid-based instance assignment\n3. Compare SOLOv2 dynamic conv to Mask R-CNN"),
    md("## SOLO (Wang et al., 2020)\n\n**Segmenting Objects by Locations** — no bounding box proposals.\n\n**Idea:** Divide feature map into $S \\times S$ grid. Each cell responsible for objects whose **center** falls in that cell.\n\n**Per grid cell predict:**\n- Category label\n- Instance mask ($E.g.$ 256 channels → upsample to full mask)\n\n**SOLOv2 improvements:**\n- Dynamic conv kernels per instance\n- Better multi-scale handling\n- Faster and higher AP than SOLO v1"),
    SETUP,
    code("S = 8\n# Each grid cell: class + mask coefficients (concept)\ngrid_classes = torch.randint(0, 5, (S, S))\ngrid_mask_dim = 32\ngrid_masks = torch.randn(S, S, grid_mask_dim)\n\nfig, ax = plt.subplots(figsize=(4, 4))\nax.imshow(grid_classes.numpy(), cmap='tab10'); ax.set_title('SOLO grid: class per cell')\nplt.show()"),
    md("## SOLO vs Mask R-CNN\n\n| | Mask R-CNN | SOLOv2 |\n|---|-----------|--------|\n| Proposals | RPN required | None |\n| Parallelism | Sequential RoI | Fully parallel grid |\n| Speed | Slower | Faster |\n| Small objects | Good with FPN | Improving |"),
    footer("SOLO assigns instances by center location on a grid — no proposals.", "05_Panoptic_Segmentation.ipynb"),
])

register("05_Panoptic_Segmentation.ipynb", [
    hdr("05", "Panoptic Segmentation", "2.5 hrs",
        "1. Define panoptic segmentation task\n2. Understand Panoptic Quality (PQ) metric\n3. Know stuff vs things labeling\n4. Design unified GeoSpatial scene understanding"),
    md("## Panoptic Segmentation (Kirillov et al., 2019)\n\n**Unified task:** Every pixel gets a label — either semantic class (stuff) or instance ID (things).\n\n**Output format:** $(semantic\\_id, instance\\_id)$ per pixel\n\n- Stuff pixels: instance_id = 0\n- Thing pixels: unique instance_id per object\n\n**No overlap:** Each pixel belongs to exactly one segment."),
    SETUP,
    code("# Panoptic label map (concept): encode as category_id * 1000 + instance_id\n# stuff: instance=0; things: instance=1,2,3...\nH, W = 64, 64\npanoptic = torch.zeros(H, W, dtype=torch.long)\npanoptic[:, :20] = 1 * 1000 + 0      # road (stuff, class 1)\npanoptic[10:30, 40:60] = 2 * 1000 + 1  # building instance 1\npanoptic[35:55, 45:65] = 2 * 1000 + 2  # building instance 2\n\nplt.imshow(panoptic.numpy(), cmap='nipy_spectral'); plt.title('Panoptic map (encoded)'); plt.colorbar(); plt.show()"),
    md("## Panoptic Quality (PQ)\n\n$$PQ = \\underbrace{\\frac{TP}{TP + \\frac{1}{2}FP + \\frac{1}{2}FN}}_{\\text{SQ (Segmentation Quality)}} \\times \\underbrace{\\frac{\\sum IoU_{matched}}{TP}}_{\\text{RQ (Recognition Quality)}}$$\n\nReported per class and as **mPQ** (mean PQ).\n\n**Matching:** Predicted segment matches GT if same class and IoU > 0.5."),
    code("# PQ components (conceptual)\nTP, FP, FN = 8, 2, 3\nmatched_ious = torch.tensor([0.7, 0.8, 0.75, 0.9, 0.85, 0.6, 0.72, 0.88])\nSQ = TP / (TP + 0.5*FP + 0.5*FN)\nRQ = matched_ious.mean().item()\nPQ = SQ * RQ\nprint(f'SQ={SQ:.3f}, RQ={RQ:.3f}, PQ={PQ:.3f}')"),
    md("## GeoSpatial Panoptic Land Cover\n\nCityscapes-style: road (stuff) + individual buildings/trees (things) from satellite — full aquaculture landscape understanding."),
    footer("Panoptic = semantic + instance with PQ metric; no pixel overlap.", "06_Panoptic_FPN.ipynb"),
])

register("06_Panoptic_FPN.ipynb", [
    hdr("06", "Panoptic FPN", "2.5 hrs",
        "1. Understand Panoptic FPN multi-task design\n2. Know semantic + instance branch fusion\n3. Handle overlapping predictions at merge step"),
    md("## Panoptic FPN (Kirillov et al., 2019)\n\n**Architecture:** Shared FPN backbone with two heads:\n\n1. **Semantic head:** Per-pixel class prediction (stuff + things classes)\n2. **Instance head:** Mask R-CNN style for things only\n\n**Fusion (panoptic merge):**\n1. Take instance masks (things) sorted by confidence\n2. Paste onto canvas (higher confidence wins overlaps)\n3. Fill remaining pixels with semantic stuff predictions\n\n**Heuristic:** Instance masks override semantic for thing classes."),
    SETUP,
    code("# Panoptic merge heuristic (concept)\nH, W = 64, 64\nsemantic = torch.randint(0, 5, (H, W))  # includes thing classes\ninst_masks = [torch.zeros(H, W) for _ in range(3)]\ninst_masks[0][10:25, 10:25] = 1\ninst_masks[1][30:45, 30:45] = 1\ninst_masks[2][15:35, 40:55] = 1\nscores = [0.9, 0.85, 0.7]\n\npanoptic_out = semantic.clone()\nfor mask, score in sorted(zip(inst_masks, scores), key=lambda x: -x[1]):\n    panoptic_out[mask.bool()] = 10 + scores.index(score)  # thing instance IDs\nprint('Unique panoptic labels:', panoptic_out.unique().tolist())"),
    md("## Limitations\n\n- Merge heuristics can fail on crowded scenes\n- Two separate training objectives\n- Mask2Former (Notebook 07) unifies with query-based approach"),
    footer("Panoptic FPN fuses Mask R-CNN instances with semantic FPN predictions.", "07_Mask2Former.ipynb"),
])

register("07_Mask2Former.ipynb", [
    hdr("07", "Mask2Former & Module Capstone", "3 hrs",
        "1. Understand query-based universal segmentation\n2. Know Mask2Former vs Panoptic FPN\n3. Choose strategy for GeoSpatial projects\n4. Complete Module 09"),
    md("## Mask2Former (Cheng et al., 2022)\n\n**Universal segmentation** — one model for semantic, instance, and panoptic via **mask classification**:\n\n1. Pixel decoder produces high-res features\n2. Transformer decoder with N learned queries\n3. Each query → class logits + mask logits\n4. Hungarian matching assigns queries to GT segments\n\n**Advantage over Panoptic FPN:** Single unified architecture, no merge heuristics.\n\n*(Also covered in Module 07 Notebook 13 — here we focus on instance/panoptic use.)*"),
    SETUP,
    md("## Architecture Comparison\n\n| Model | Type | Speed | mAP/mPQ | Best For |\n|-------|------|-------|---------|----------|\n| UNet++ dual-head | Semantic + boundary | Fast | High IoU (your project) | Adjacent ponds |\n| Mask R-CNN | Instance | Medium | Mask AP | Instance IDs needed |\n| YOLACT | Instance | Fast | Good | Real-time video |\n| SOLOv2 | Instance | Fast | Good | No proposals |\n| Panoptic FPN | Panoptic | Medium | PQ | Cityscapes-style |\n| Mask2Former | Universal | Medium | SOTA PQ | Research / unified model |"),
    md("## water-bodies-detection: Architectural Decision\n\n**Problem:** Adjacent aquaculture ponds merge in semantic segmentation.\n\n**Option A — Your dual-head (Module 07):**\n- Train aqua + bund boundary heads\n- Post-process: watershed on boundaries → separate polygons\n- Labels: semantic masks (simpler annotation)\n\n**Option B — Mask R-CNN (Module 09):**\n- Annotate each pond with instance ID\n- Direct instance masks at inference\n- Heavier labels, heavier model\n\n**Option C — Hybrid:**\n- UNet++ for boundaries + connected components for instance IDs\n- No Mask R-CNN needed for many aquaculture use cases\n\n**Recommendation for your project:** Dual-head is the right engineering choice. Know Mask R-CNN for when clients need instance tracking or panoptic dashboards."),
    md("## Module 09 Assignment\n\nImplement connected-components instance labeling from binary aqua mask + boundary mask (simulate your post-process). Compare to Mask R-CNN on synthetic adjacent circles.\n\nSee `exercises/README.md`."),
    code("# Connected components instance labeling (concept)\nfrom scipy import ndimage\naqua = np.zeros((64, 128), dtype=np.uint8)\naqua[:, 15:30] = 1; aqua[:, 35:50] = 1  # two adjacent ponds\nlabeled, n = ndimage.label(aqua)\nprint(f'Instances found: {n}')\nplt.imshow(labeled, cmap='nipy_spectral'); plt.title('Connected components'); plt.show()"),
    md("## Module 09 Complete\n\n**Next:** Module 10 Transformers — attention, ViT, BERT, GPT."),
    footer("Mask2Former unifies segmentation tasks; your dual-head boundary design is a pragmatic alternative to instance seg.", None),
])


def main():
    print("Building Module 09 notebooks...")
    for name, cells in NOTEBOOKS:
        save(name, cells)
    print(f"Done: {len(NOTEBOOKS)} notebooks created.")


if __name__ == "__main__":
    main()
