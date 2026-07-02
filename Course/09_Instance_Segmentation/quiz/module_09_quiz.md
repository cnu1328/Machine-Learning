# Module 09 Quiz

**Passing score:** 12/15 (80%)

---

**Q1.** Instance segmentation assigns:
- (a) One class label per image
- (b) Unique mask per object instance
- (c) Bounding boxes only
- (d) Depth per pixel

**Q2.** Semantic segmentation of adjacent ponds:
- (a) Separates each pond automatically
- (b) May merge adjacent ponds into one region
- (c) Assigns instance IDs
- (d) Uses NMS

**Q3.** Mask R-CNN adds to Faster R-CNN:
- (a) RPN only
- (b) Parallel mask head per RoI
- (c) YOLO grid
- (d) Transformer decoder

**Q4.** Mask R-CNN mask head typically outputs:
- (a) Full-resolution mask directly
- (b) Low-res mask (e.g. 28×28) upsampled to RoI
- (c) Bounding box only
- (d) Class embedding only

**Q5.** YOLACT assembles masks via:
- (a) RoI pooling
- (b) Linear combination of prototype masks
- (c) K-means clustering
- (d) Graph cuts

**Q6.** SOLO assigns instances by:
- (a) Random assignment
- (b) Object center location on grid
- (c) Color histogram
- (d) Edge detection only

**Q7.** Panoptic segmentation combines:
- (a) Detection + classification only
- (b) Semantic (stuff) + instance (things)
- (c) Segmentation + regression
- (d) Two semantic models

**Q8.** Panoptic Quality (PQ) equals:
- (a) mAP × mIoU
- (b) SQ × RQ (segmentation × recognition quality)
- (c) Precision only
- (d) Box IoU mean

**Q9.** Stuff classes include:
- (a) Individual cars
- (b) Sky, road, water (amorphous regions)
- (c) Bounding boxes
- (d) Instance IDs only

**Q10.** Panoptic FPN merges predictions by:
- (a) Random pixel assignment
- (b) Paste instance masks by confidence, fill stuff semantically
- (c) Softmax over all classes only
- (d) KNN

**Q11.** Mask2Former uses:
- (a) Selective search
- (b) Query-based mask classification with transformer
- (c) R-CNN only
- (d) No learned queries

**Q12.** Your water-bodies dual-head approach:
- (a) Requires instance polygon labels per pond
- (b) Uses aqua + boundary heads to separate adjacent ponds
- (c) Is identical to Mask R-CNN
- (d) Cannot produce GIS polygons

**Q13.** Mask IoU is computed on:
- (a) Bounding boxes
- (b) Pixel masks
- (c) Class logits
- (d) Feature vectors

**Q14.** Connected components on binary mask:
- (a) Assigns semantic classes
- (b) Labels separated foreground blobs with unique IDs
- (c) Trains neural network
- (d) Computes PQ directly

**Q15.** When to prefer Mask R-CNN over dual-head UNet++:
- (a) Always — Mask R-CNN is always better
- (b) When instance tracking/IDs needed and instance labels available
- (c) For faster inference always
- (d) When you only have semantic labels

---

See [solutions/quiz_answers.md](solutions/quiz_answers.md) after submitting.
