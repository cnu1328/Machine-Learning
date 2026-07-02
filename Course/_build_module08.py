#!/usr/bin/env python3
"""Generate Module 08 Object Detection notebooks (12 total)."""
import json
from pathlib import Path

M08 = Path(__file__).resolve().parent / "08_Object_Detection"


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
    M08.mkdir(parents=True, exist_ok=True)
    (M08 / name).write_text(json.dumps(nb(cells), indent=1))
    print(f"  {name}")


def hdr(num, title, dur, objs):
    return md(
        f"# Notebook {num}: {title}\n\n**Module:** 08 Object Detection  \n**Duration:** ~{dur}\n\n---\n\n## Learning Objectives\n\n{objs}"
    )


def footer(summary, nxt=None):
    nxt = f"\n\n**Next:** [{nxt}]({nxt})" if nxt else ""
    return md(f"---\n\n## Summary\n\n{summary}{nxt}")


SETUP = code(
    "import numpy as np\nimport matplotlib.pyplot as plt\nimport matplotlib.patches as patches\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nplt.rcParams['figure.figsize'] = (8, 5)\ntorch.manual_seed(42)\nrng = np.random.default_rng(42)\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nprint(f'Device: {device}')"
)

IOU_CODE = code(
    "def box_iou(boxes1, boxes2):\n    \"\"\"boxes: (N,4) and (M,4) in xyxy format. Returns (N,M).\"\"\"\n    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])\n    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])\n    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])\n    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])\n    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)\n    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])\n    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])\n    union = area1[:, None] + area2[None, :] - inter\n    return inter / (union + 1e-7)\n\n\ndef xywh_to_xyxy(boxes):\n    x, y, w, h = boxes.unbind(-1)\n    return torch.stack([x, y, x + w, y + h], dim=-1)\n\n\ndef xyxy_to_cxcywh(boxes):\n    x1, y1, x2, y2 = boxes.unbind(-1)\n    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)"
)

NMS_CODE = code(
    "def nms(boxes, scores, iou_thresh=0.5):\n    \"\"\"Greedy NMS. boxes: (N,4) xyxy, scores: (N,).\"\"\"\n    order = scores.argsort(descending=True)\n    keep = []\n    while order.numel() > 0:\n        i = order[0].item()\n        keep.append(i)\n        if order.numel() == 1:\n            break\n        ious = box_iou(boxes[i:i+1], boxes[order[1:]])[0]\n        mask = ious <= iou_thresh\n        order = order[1:][mask]\n    return torch.tensor(keep, dtype=torch.long)"
)

NOTEBOOKS = []


def register(name, cells):
    NOTEBOOKS.append((name, cells))


# ── FOUNDATIONS (01-03) ─────────────────────────────────────────────────

register("01_Detection_Fundamentals.ipynb", [
    hdr("01", "Detection Fundamentals", "2 hrs",
        "1. Define object detection vs classification\n2. Represent bounding boxes (xyxy, xywh, cxcywh)\n3. Compute IoU between boxes\n4. Connect to GeoSpatial pond/building counting"),
    md("## Object Detection\n\n**Classification:** What is in the image?\n\n**Detection:** What objects are present AND where are they?\n\n**Output:** Set of bounding boxes + class labels + confidence scores.\n\n| Format | Fields | Used By |\n|--------|--------|--------|\n| xyxy | x_min, y_min, x_max, y_max | PyTorch, NMS |\n| xywh | x_min, y_min, width, height | COCO annotations |\n| cxcywh | center_x, center_y, w, h | YOLO, DETR |\n\n**GeoSpatial:** Segmentation gives pixel masks; detection gives box counts for ponds, buildings, ships."),
    SETUP, IOU_CODE,
    code("# Visualize boxes and IoU\nfig, ax = plt.subplots(figsize=(6, 6))\nax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.invert_yaxis()\nbox_a = torch.tensor([20., 20., 60., 60.])\nbox_b = torch.tensor([40., 40., 80., 80.])\nfor b, c in [(box_a, 'tab:blue'), (box_b, 'tab:orange')]:\n    x1, y1, x2, y2 = b.tolist()\n    ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=c, lw=2))\niou = box_iou(box_a.unsqueeze(0), box_b.unsqueeze(0)).item()\nax.set_title(f'IoU = {iou:.3f}'); plt.show()"),
    md("## IoU (Intersection over Union)\n\n$$\\text{IoU} = \\frac{|A \\cap B|}{|A \\cup B|}$$\n\nUsed for matching predictions to ground truth and for NMS."),
    footer("Detection = classify + localize. IoU measures box overlap.", "02_NMS_and_mAP.ipynb"),
])

register("02_NMS_and_mAP.ipynb", [
    hdr("02", "NMS and mAP Evaluation", "2.5 hrs",
        "1. Implement Non-Maximum Suppression from scratch\n2. Understand precision-recall curves\n3. Define mAP@0.5 and mAP@0.5:0.95\n4. Know COCO evaluation protocol"),
    md("## Non-Maximum Suppression (NMS)\n\nProblem: detector outputs many overlapping boxes for one object.\n\n**Algorithm:**\n1. Sort boxes by confidence (descending)\n2. Keep highest-scoring box\n3. Remove boxes with IoU > threshold to kept box\n4. Repeat\n\nTypical IoU threshold: 0.5"),
    SETUP, IOU_CODE, NMS_CODE,
    code("# NMS demo: 5 overlapping boxes, one true object\nboxes = torch.tensor([\n    [10., 10., 50., 50.],\n    [12., 12., 52., 52.],\n    [15., 8., 48., 49.],\n    [200., 200., 240., 240.],\n    [202., 198., 242., 238.],\n])\nscores = torch.tensor([0.92, 0.88, 0.75, 0.85, 0.80])\nkeep = nms(boxes, scores, iou_thresh=0.5)\nprint('Kept indices:', keep.tolist())\nprint('Kept boxes:', boxes[keep].tolist())"),
    md("## mAP (mean Average Precision)\n\n1. Match predictions to GT by IoU threshold (e.g. 0.5)\n2. Sort by confidence → precision-recall curve\n3. **AP** = area under PR curve (per class)\n4. **mAP** = mean AP across classes\n\n**COCO:** mAP@0.5:0.95 = average AP at IoU 0.5, 0.55, ..., 0.95\n\n| Metric | Meaning |\n|--------|--------|\n| mAP@0.5 | Loose matching (PASCAL VOC style) |\n| mAP@0.5:0.95 | Strict COCO benchmark |"),
    code("# Simple AP intuition: 10 GT, sorted predictions\n# TP if IoU>=0.5 and unmatched GT; else FP\nprecisions = [1.0, 1.0, 0.67, 0.75, 0.8]\nrecalls = [0.1, 0.2, 0.3, 0.4, 0.5]\nplt.plot(recalls, precisions, 'o-'); plt.xlabel('Recall'); plt.ylabel('Precision')\nplt.title('Precision-Recall curve (concept)'); plt.grid(True); plt.show()"),
    footer("NMS removes duplicate detections; mAP is the standard detection metric.", "03_Detection_Loss_Functions.ipynb"),
])

register("03_Detection_Loss_Functions.ipynb", [
    hdr("03", "Detection Loss Functions", "2 hrs",
        "1. Decompose detection loss into classification + localization\n2. Derive Smooth L1 (Huber) loss for box regression\n3. Understand focal loss for detection\n4. Match losses to one-stage vs two-stage detectors"),
    md("## Detection Loss\n\n$$L = L_{\\text{cls}} + \\lambda L_{\\text{loc}}$$\n\n**Classification:** Cross-entropy (or focal loss) on object class + background.\n\n**Localization:** Smooth L1 between predicted and target box offsets.\n\n**Smooth L1:**\n$$L(x) = \\begin{cases} 0.5 x^2 & |x| < 1 \\\\ |x| - 0.5 & \\text{otherwise} \\end{cases}$$"),
    SETUP,
    code("def smooth_l1(pred, target, beta=1.0):\n    diff = (pred - target).abs()\n    return torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta).mean()\n\npred_box = torch.tensor([0.1, -0.2, 0.05, 0.3])\ntarget_box = torch.tensor([0.0, 0.0, 0.0, 0.0])\nprint(f'Smooth L1: {smooth_l1(pred_box, target_box):.4f}')\n\n# Focal loss for classification (RetinaNet)\ndef focal_loss_cls(logits, targets, gamma=2.0, alpha=0.25):\n    ce = F.cross_entropy(logits, targets, reduction='none')\n    p = torch.exp(-ce)\n    return (alpha * (1 - p) ** gamma * ce).mean()\n\nlogits = torch.randn(8, 5)\ntargets = torch.randint(0, 5, (8,))\nprint(f'Focal cls loss: {focal_loss_cls(logits, targets):.4f}')"),
    md("## Anchor Matching\n\nAssign each anchor/prior box to GT:\n- IoU > 0.7 → positive\n- IoU < 0.3 → negative (background)\n- Between → ignore\n\nRegression targets are **offsets** from anchor to GT box (Faster R-CNN parameterization)."),
    footer("Detection = cls loss + loc loss; focal loss handles foreground/background imbalance.", "04_R_CNN.ipynb"),
])

# ── TWO-STAGE (04-06) ───────────────────────────────────────────────────

register("04_R_CNN.ipynb", [
    hdr("04", "R-CNN", "2 hrs",
        "1. Understand the R-CNN pipeline (selective search → CNN → SVM)\n2. Know why it was slow\n3. Trace evolution to Fast/Faster R-CNN"),
    md("## R-CNN (Girshick et al., 2014)\n\n**Pipeline:**\n1. **Region proposals** (~2000) via Selective Search\n2. **Warp** each region to fixed size (227×227)\n3. **CNN** (AlexNet) → feature vector\n4. **SVM** classifier per class\n5. **BBox regressor** per class\n\n**Problems:** Slow (CNN run per region), multi-stage training, no end-to-end gradient."),
    SETUP,
    code("# R-CNN concept: many proposals, each classified separately\nn_proposals = 2000\nproposal_time = 2.0  # selective search seconds\nforward_per_region = 0.05  # AlexNet forward seconds\ntotal = proposal_time + n_proposals * forward_per_region\nprint(f'R-CNN inference ~{total:.0f}s per image (conceptual)')\nprint('Bottleneck: no shared computation across regions')"),
    md("## Historical Significance\n\nFirst to show CNN features + region proposals beat hand-crafted features on PASCAL VOC.\n\n**Your takeaway:** Detection = **where to look** (proposals) + **what is it** (classifier)."),
    footer("R-CNN proved CNNs for detection but was too slow for production.", "05_Fast_R_CNN.ipynb"),
])

register("05_Fast_R_CNN.ipynb", [
    hdr("05", "Fast R-CNN", "2.5 hrs",
        "1. Understand RoI Pooling and shared backbone\n2. Single-stage training with multi-task loss\n3. Compare speed vs R-CNN"),
    md("## Fast R-CNN (Girshick, 2015)\n\n**Key innovation:** Run CNN **once** on full image → feature map. Pool features per proposal with **RoI Pooling**.\n\n**RoI Pooling:** Map each proposal onto feature map, divide into fixed grid (e.g. 7×7), max-pool each cell.\n\n**Loss:** Multi-task — softmax cls + Smooth L1 bbox (end-to-end)."),
    SETUP,
    code("class RoIPool(nn.Module):\n    def __init__(self, out_h=7, out_w=7):\n        super().__init__()\n        self.out_h, self.out_w = out_h, out_w\n    def forward(self, x, rois):\n        # x: (1,C,H,W), rois: (N,5) [batch_idx,x1,y1,x2,y2] in feature-map coords\n        N = rois.shape[0]\n        C, H, W = x.shape[1:]\n        out = torch.zeros(N, C, self.out_h, self.out_w)\n        for i, roi in enumerate(rois):\n            _, x1, y1, x2, y2 = roi.tolist()\n            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)\n            crop = x[0, :, y1:y2, x1:x2]\n            if crop.numel() == 0:\n                continue\n            crop = crop.unsqueeze(0)\n            pooled = F.adaptive_max_pool2d(crop, (self.out_h, self.out_w))\n            out[i] = pooled[0]\n        return out\n\nfeat = torch.randn(1, 64, 32, 32)\nrois = torch.tensor([[0, 4, 4, 20, 20], [0, 10, 10, 28, 28]], dtype=torch.float32)\nprint('RoI pool output:', RoIPool()(feat, rois).shape)"),
    md("## Speed\n\n~200× faster than R-CNN at test time (shared conv). Still uses external Selective Search for proposals."),
    footer("Fast R-CNN shares convolution — one forward pass per image.", "06_Faster_R_CNN.ipynb"),
])

register("06_Faster_R_CNN.ipynb", [
    hdr("06", "Faster R-CNN", "3 hrs",
        "1. Understand Region Proposal Network (RPN)\n2. Know anchor boxes at each feature map location\n3. Run torchvision Faster R-CNN on synthetic data\n4. Apply to vehicle/building detection"),
    md("## Faster R-CNN (Ren et al., 2015)\n\n**RPN:** Small network on feature map predicts objectness + box offsets for **anchors** at each spatial location.\n\n**Two heads after RPN:**\n1. RPN → proposals\n2. RoI head → class + refined box\n\n**Anchors:** Multiple scales/aspect ratios per pixel (e.g. 9 anchors at each of 50×38 locations)."),
    SETUP,
    code("try:\n    from torchvision.models.detection import fasterrcnn_resnet50_fpn\n    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor\n\n    def build_faster_rcnn(num_classes=2):\n        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)\n        in_features = model.roi_heads.box_predictor.cls_score.in_features\n        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)\n        return model\n\n    model = build_faster_rcnn(num_classes=2)  # background + object\n    model.eval()\n    images = [torch.rand(3, 256, 256)]\n    with torch.no_grad():\n        out = model(images)\n    print('Faster R-CNN output keys:', out[0].keys())\n    print('Num detections:', len(out[0]['boxes']))\nexcept ImportError as e:\n    print('torchvision detection requires recent torchvision:', e)"),
    md("## GeoSpatial Use Case\n\n**Building detection:** Faster R-CNN on satellite tiles — good when objects are sparse and precise boxes needed.\n\n**vs Segmentation:** Detection counts instances; segmentation delineates boundaries."),
    footer("Faster R-CNN = RPN proposals + RoI head — two-stage SOTA baseline for years.", "07_SSD.ipynb"),
])

# ── SINGLE-STAGE (07-10) ────────────────────────────────────────────────

register("07_SSD.ipynb", [
    hdr("07", "SSD (Single Shot Detector)", "2.5 hrs",
        "1. Multi-scale feature maps for detection\n2. Default boxes (anchors) at multiple scales\n3. Compare single-pass vs two-stage speed"),
    md("## SSD (Liu et al., 2016)\n\n**Single shot:** Predict class + box at multiple feature map scales in one forward pass.\n\n**Default boxes:** Fixed anchor sizes per layer (e.g. 38×38 layer for small objects, 10×10 for large).\n\n**Multi-scale:** Early layers → small objects; deep layers → large objects."),
    SETUP,
    code("# SSD-style multi-scale heads (concept)\nscales = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]\nanchors_per_loc = 4\nfor h, w in scales:\n    n_anchors = h * w * anchors_per_loc\n    print(f'Feature map {h}x{w}: {n_anchors:,} default boxes')"),
    md("## Trade-offs\n\n| | Faster R-CNN | SSD |\n|---|-------------|-----|\n| Stages | 2 | 1 |\n| Speed | Slower | Faster |\n| Small objects | Better (FPN helps) | Harder without FPN |"),
    footer("SSD detects at multiple scales in one pass using default boxes.", "08_YOLO_v1_v3.ipynb"),
])

register("08_YOLO_v1_v3.ipynb", [
    hdr("08", "YOLO v1–v3", "2.5 hrs",
        "1. Grid-based detection formulation\n2. YOLO v2/v3 improvements (anchors, multi-scale)\n3. Decode YOLO-style predictions"),
    md("## YOLO v1 (Redmon et al., 2016)\n\nDivide image into $S \\times S$ grid. Each cell predicts:\n- $B$ bounding boxes (x, y, w, h, confidence)\n- Class probabilities\n\n**Loss:** MSE on coords + classification.\n\n**YOLO v2:** Anchor boxes, batch norm, multi-scale training.\n\n**YOLO v3:** Feature pyramid, 3 scales, logistic class prediction."),
    SETUP,
    code("S, B, C = 7, 2, 3  # grid, boxes per cell, classes\n# Prediction tensor per image: S x S x (B*5 + C)\npred = torch.randn(S, S, B * 5 + C)\n\n# Decode one cell prediction (concept)\ncell_pred = pred[3, 3]\nconf = torch.sigmoid(cell_pred[4])  # first box confidence\ncls = torch.softmax(cell_pred[B*5:], dim=0)\nprint(f'Cell (3,3) conf={conf:.3f}, best class={cls.argmax().item()}')"),
    md("## YOLO Philosophy\n\nTreat detection as **regression** — one network, one pass, real-time.\n\nLimitation of v1: struggles with small objects and nearby duplicates (one object per cell)."),
    footer("YOLO reframed detection as single regression problem — foundation for real-time CV.", "09_RetinaNet.ipynb"),
])

register("09_RetinaNet.ipynb", [
    hdr("09", "RetinaNet", "2.5 hrs",
        "1. Understand Focal Loss for class imbalance\n2. Feature Pyramid Network (FPN) for multi-scale\n3. Compare RetinaNet to Faster R-CNN"),
    md("## RetinaNet (Lin et al., 2017)\n\n**Problem:** Extreme foreground/background imbalance (~100k:1 anchors) — CE loss dominated by easy negatives.\n\n**Focal Loss:** $(1-p_t)^\\gamma$ down-weights easy examples.\n\n**FPN:** Top-down pathway + lateral connections → rich multi-scale features.\n\n**Result:** First single-stage detector to match two-stage mAP on COCO."),
    SETUP,
    code("def focal_loss_binary(logits, targets, gamma=2.0, alpha=0.25):\n    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')\n    p = torch.sigmoid(logits)\n    p_t = p * targets + (1 - p) * (1 - targets)\n    return (alpha * (1 - p_t) ** gamma * bce).mean()\n\n# Simulated anchor labels: 1000 bg, 10 fg\nlogits = torch.randn(1010)\ntargets = torch.cat([torch.zeros(1000), torch.ones(10)])\nprint(f'Focal loss (imbalanced): {focal_loss_binary(logits, targets):.4f}')"),
    md("## GeoSpatial: Ship Detection (HRSC2016)\n\nRetinaNet + FPN handles ships at multiple scales in satellite imagery."),
    footer("RetinaNet = FPN + focal loss — single-stage at two-stage accuracy.", "10_YOLOv5_v8_v11.ipynb"),
])

register("10_YOLOv5_v8_v11.ipynb", [
    hdr("10", "YOLOv5 / v8 / v11 (Ultralytics)", "3 hrs",
        "1. Understand modern YOLO architecture (CSPDarknet, PAN-FPN)\n2. Use Ultralytics API for training/inference\n3. Apply to building detection on satellite imagery\n4. Compare export formats (ONNX, TensorRT)"),
    md("## Modern YOLO Lineage\n\n| Version | Key Features |\n|---------|-------------|\n| YOLOv5 | PyTorch, easy training, auto-anchor |\n| YOLOv8 | Anchor-free decoupled head, unified API |\n| YOLOv11 | Improved efficiency, same Ultralytics stack |\n\n**Ultralytics API:**\n```python\nfrom ultralytics import YOLO\nmodel = YOLO('yolov8n.pt')\nresults = model.train(data='buildings.yaml', epochs=100)\nresults = model.predict(source='tile.tif')\n```"),
    SETUP,
    code("try:\n    from ultralytics import YOLO\n    print('Ultralytics available — YOLOv8 ready')\n    # model = YOLO('yolov8n.pt')  # download on first use\nexcept ImportError:\n    print('Install: pip install ultralytics')\n    print('Typical workflow: annotate → YOLO format → train → export ONNX')"),
    md("## GeoSpatial: Building Detection\n\n**Workflow:**\n1. Annotate buildings in QGIS/Label Studio → YOLO txt format\n2. Train YOLOv8 on 512×512 satellite tiles\n3. Run sliding-window inference on large GeoTIFF\n4. Export to ONNX for deployment\n\n**vs water-bodies segmentation:** Detection gives bounding boxes for quick counts; segmentation gives exact boundaries."),
    md("## Production Tips\n\n- Tile large images with overlap\n- Use lower conf threshold at inference, NMS to filter\n- Monitor mAP@0.5 on validation tiles\n- Augment: rotation, flip, color jitter (careful with NIR bands)"),
    footer("YOLOv8/v11 via Ultralytics — your go-to for real-time GeoSpatial detection.", "11_DETR.ipynb"),
])

# ── TRANSFORMERS (11-12) ────────────────────────────────────────────────

register("11_DETR.ipynb", [
    hdr("11", "DETR (Detection Transformer)", "3 hrs",
        "1. Set prediction with fixed N queries\n2. Hungarian matching for loss assignment\n3. Transformer encoder-decoder for detection\n4. Know when DETR beats CNN detectors"),
    md("## DETR (Carion et al., 2020)\n\n**Paradigm shift:** No anchors, no NMS.\n\n**Architecture:**\n1. CNN backbone → feature map\n2. Transformer encoder\n3. Transformer decoder with **N learned object queries** (e.g. 100)\n4. Each query → class + box (cxcywh normalized)\n\n**Training:** Hungarian algorithm matches predictions to GT (bipartite matching loss)."),
    SETUP,
    code("# DETR output format (concept)\nN_queries = 100\nnum_classes = 91  # COCO\nlogits = torch.randn(N_queries, num_classes)\nboxes = torch.sigmoid(torch.randn(N_queries, 4))  # cxcywh in [0,1]\n\n# Inference: take queries where class != 'no-object'\nprobs = logits.softmax(-1)\nscores, labels = probs[:, :-1].max(-1)\nkeep = scores > 0.7\nprint(f'High-confidence detections: {keep.sum().item()}')"),
    md("## Pros & Cons\n\n| Pros | Cons |\n|------|------|\n| End-to-end, no NMS | Slow to converge (~500 epochs) |\n| Simple pipeline | Large objects easier than small |\n| Set-based loss | Higher compute than YOLO |"),
    footer("DETR treats detection as direct set prediction with transformers.", "12_RT_DETR.ipynb"),
])

register("12_RT_DETR.ipynb", [
    hdr("12", "RT-DETR & Module Capstone", "2.5 hrs",
        "1. Understand real-time DETR optimizations\n2. Compare detection paradigms\n3. Choose detector for GeoSpatial tasks\n4. Complete Module 08"),
    md("## RT-DETR (Real-Time DETR)\n\n**Improvements over DETR:**\n- Efficient hybrid encoder\n- IoU-aware query selection\n- Multi-scale features without heavy FPN\n\n**Result:** DETR-quality with YOLO-like speed."),
    SETUP,
    md("## Detection Paradigm Comparison\n\n| Model | Type | Speed | mAP | Best For |\n|-------|------|-------|-----|----------|\n| Faster R-CNN | Two-stage | Medium | High | Sparse, precise boxes |\n| SSD | Single-stage | Fast | Medium | Embedded systems |\n| RetinaNet | Single-stage | Medium | High | Multi-scale objects |\n| YOLOv8 | Single-stage | Very fast | High | Production GeoSpatial |\n| DETR | Transformer | Medium | High | Research, no NMS |\n| RT-DETR | Transformer | Fast | High | Real-time transformer |"),
    md("## GeoSpatial Pipeline Integration\n\n```\nSatellite tile\n  ├─ Segmentation (Module 07) → pond boundaries, land cover\n  └─ Detection (Module 08)    → building/ship/vehicle counts\n\nCombined → panoptic aquaculture dashboard (Module 09)\n```"),
    md("## Module 08 Assignment\n\nTrain YOLOv8 (or Faster R-CNN) on a small custom dataset (≥50 annotated objects). Report mAP@0.5, inference speed, and failure cases.\n\nSee `exercises/README.md`."),
    code("# YOUR CODE HERE — paradigm comparison table on your hardware\n"),
    md("## Module 08 Complete\n\n**Next:** Module 09 Instance Segmentation — Mask R-CNN, panoptic segmentation."),
    footer("Choose YOLO for production speed, Faster R-CNN for precision, DETR for anchor-free research.", None),
])


def main():
    print("Building Module 08 notebooks...")
    for name, cells in NOTEBOOKS:
        save(name, cells)
    print(f"Done: {len(NOTEBOOKS)} notebooks created.")


if __name__ == "__main__":
    main()
