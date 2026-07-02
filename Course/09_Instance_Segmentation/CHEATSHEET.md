# Module 09 Cheat Sheet — Instance Segmentation

## Three Paradigms

| Type | Output | Instances? | Metric |
|------|--------|------------|--------|
| Semantic | Class per pixel | No | mIoU |
| Instance | Mask + ID per object | Yes | mask AP |
| Panoptic | Stuff + thing IDs | Yes (things) | PQ |

**Stuff:** sky, road, water (no count)  
**Things:** car, building, pond (countable)

## Mask R-CNN Pipeline

```
Image → FPN backbone
         ├─ RPN → proposals
         ├─ Box head → class + box
         └─ Mask head → 28×28 mask per RoI
```

- **Mask loss:** Sigmoid CE on positive RoIs
- **Class-agnostic mask:** One mask channel, classify separately
- **torchvision:** `maskrcnn_resnet50_fpn`

## Mask IoU

$$\text{maskIoU} = \frac{|M_{pred} \cap M_{gt}|}{|M_{pred} \cup M_{gt}|}$$

COCO instance seg uses **mask AP** (not box AP).

## Real-Time Alternatives

| Model | Idea | Speed |
|-------|------|-------|
| YOLACT | Prototype masks × coefficients | Fast |
| SOLOv2 | Grid cell → instance mask | Fast |
| Mask R-CNN | RoI mask head | Slower |

**YOLACT:** $M_i = \sigma(\sum_k c_{i,k} P_k)$

## Panoptic Segmentation

Every pixel: `(semantic_class, instance_id)`
- Stuff: instance_id = 0
- Things: unique instance_id
- **No overlapping segments**

## Panoptic Quality (PQ)

$$PQ = SQ \times RQ$$

- **SQ:** $\frac{TP}{TP + \frac{1}{2}FP + \frac{1}{2}FN}$
- **RQ:** mean IoU of matched segments
- Match if same class + IoU > 0.5

## Panoptic FPN

1. Semantic head (all classes)
2. Instance head (Mask R-CNN for things)
3. **Merge:** paste instances by confidence, fill stuff

## Mask2Former

- Query-based (like DETR)
- One model: semantic / instance / panoptic
- Hungarian matching
- No merge heuristics

## water-bodies: Dual-Head vs Instance Seg

| | Dual-head (Mod 07) | Mask R-CNN (Mod 09) |
|---|-------------------|---------------------|
| Labels | Aqua + boundary masks | Instance polygon per pond |
| Adjacent ponds | Boundary separates | Separate instance masks |
| Instance ID | Connected components post-process | Native |
| Compute | UNet++ (efficient) | Heavier two-stage |
| Your project | ✅ Production choice | Alternative |

## Post-Process: Connected Components

```python
from scipy import ndimage
labeled, n_instances = ndimage.label(aqua_mask)
```

Use boundary mask to split merged components (watershed).

## When to Use What

| Need | Choose |
|------|--------|
| Pond boundaries, GIS polygons | UNet++ dual-head (Mod 07) |
| Track individual pond IDs | Mask R-CNN or CC on dual-head |
| Real-time video instances | YOLACT |
| Full land cover + buildings | Panoptic FPN / Mask2Former |
| Research SOTA unified model | Mask2Former |

## Common Mistakes

- Using semantic mIoU for instance task (use mask AP or PQ)
- Forgetting NMS on instance predictions
- Overlapping instance masks without merge rule
- Instance annotations when boundary labels suffice
- Evaluating panoptic with semantic metrics only

## PyTorch Quick Reference

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

model = maskrcnn_resnet50_fpn(weights="DEFAULT")
out = model([image_tensor])
# out[0]['boxes'], out[0]['labels'], out[0]['masks'], out[0]['scores']
```
