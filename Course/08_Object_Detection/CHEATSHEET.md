# Module 08 Cheat Sheet — Object Detection

## Task Definition

**Input:** Image  
**Output:** List of `(class, confidence, x1, y1, x2, y2)` bounding boxes

## Box Formats

| Format | Fields | Convert |
|--------|--------|---------|
| xyxy | x_min, y_min, x_max, y_max | PyTorch NMS |
| xywh | x_min, y_min, w, h | COCO JSON |
| cxcywh | cx, cy, w, h | YOLO, DETR |

```python
def xywh_to_xyxy(b):
    x, y, w, h = b.unbind(-1)
    return torch.stack([x, y, x+w, y+h], -1)
```

## IoU

$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$$

- Matching pred ↔ GT: IoU ≥ 0.5 (typical)
- NMS threshold: IoU ≤ 0.5 (suppress overlaps)

## NMS Algorithm

1. Sort by confidence ↓
2. Keep top box
3. Remove boxes with IoU > thresh to kept
4. Repeat until empty

## mAP

| Metric | Meaning |
|--------|---------|
| AP | Area under PR curve (one class) |
| mAP | Mean AP across classes |
| mAP@0.5 | IoU threshold 0.5 |
| mAP@0.5:0.95 | COCO standard (10 thresholds) |

## Loss Functions

$$L = L_{cls} + \lambda L_{loc}$$

| Component | Loss | Used In |
|-----------|------|---------|
| Classification | CE / Focal | All |
| Localization | Smooth L1 | R-CNN family, SSD |
| Objectness | BCE | RPN, YOLO |

**Focal Loss:** $L = -(1-p_t)^\gamma \log(p_t)$ — RetinaNet, hard examples

## Architecture Timeline

| Model | Year | Type | Key Idea |
|-------|------|------|----------|
| R-CNN | 2014 | Two-stage | CNN per region |
| Fast R-CNN | 2015 | Two-stage | RoI Pooling, shared conv |
| Faster R-CNN | 2015 | Two-stage | RPN + anchors |
| SSD | 2016 | Single | Multi-scale default boxes |
| YOLO | 2016 | Single | Grid regression |
| RetinaNet | 2017 | Single | FPN + focal loss |
| YOLOv8 | 2023 | Single | Anchor-free, Ultralytics |
| DETR | 2020 | Transformer | Set prediction, no NMS |
| RT-DETR | 2023 | Transformer | Real-time DETR |

## Two-Stage vs Single-Stage

| | Two-Stage | Single-Stage |
|---|-----------|--------------|
| Examples | Faster R-CNN | YOLO, SSD, RetinaNet |
| Proposals | RPN / selective search | Dense anchors / grid |
| Speed | Slower | Faster |
| Small objects | Better (with FPN) | Improving |

## Faster R-CNN Pipeline

```
Image → Backbone → Feature Map
                    ├─ RPN → proposals (anchors)
                    └─ RoI Head → class + box refine
```

## YOLO (Ultralytics)

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='data.yaml', epochs=100, imgsz=640)
results = model.predict(source='image.jpg', conf=0.25)
model.export(format='onnx')
```

## DETR

- N fixed object queries (e.g. 100)
- Hungarian matching during training
- No NMS at inference (optional score threshold)
- Slow convergence; RT-DETR fixes speed

## GeoSpatial Use Cases

| Task | Model | Output |
|------|-------|--------|
| Building count | YOLOv8 | Bounding boxes |
| Ship detection | RetinaNet + FPN | Oriented boxes |
| Vehicle detection | Faster R-CNN | Precise boxes |
| Pond boundaries | Segmentation (Mod 07) | Pixel masks |

**Combined:** Detection for counts + segmentation for boundaries → Module 09 panoptic.

## Production Checklist

- [ ] Tile large GeoTIFF with overlap
- [ ] Match train/inference resolution
- [ ] Tune conf threshold + NMS on val set
- [ ] Monitor mAP@0.5 per class
- [ ] Export ONNX/TensorRT for deployment

## Common Mistakes

- Wrong box format (xywh vs xyxy)
- Forgetting NMS → duplicate boxes
- Evaluating with wrong IoU threshold
- Training on COCO, testing on satellite without domain adaptation
- No overlap in sliding-window inference (missed edge objects)
