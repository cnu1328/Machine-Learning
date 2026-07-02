# Module 08 Quiz

**Passing score:** 16/20 (80%)

---

**Q1.** Object detection outputs:
- (a) Single class label only
- (b) Bounding boxes + class labels + scores
- (c) Pixel-wise masks only
- (d) Keypoints only

**Q2.** IoU is used for:
- (a) Learning rate scheduling
- (b) Matching predictions to ground truth and NMS
- (c) Data augmentation
- (d) Batch normalization

**Q3.** NMS purpose:
- (a) Increase duplicate detections
- (b) Remove overlapping low-confidence boxes
- (c) Train the backbone
- (d) Compute mAP

**Q4.** mAP@0.5:0.95 means:
- (a) AP at IoU 0.5 only
- (b) Average AP across IoU thresholds 0.5 to 0.95
- (c) Maximum AP
- (d) Mean precision at recall 0.95

**Q5.** R-CNN bottleneck:
- (a) Too few proposals
- (b) CNN run separately per region proposal
- (c) No classification head
- (d) Missing NMS

**Q6.** Fast R-CNN improvement over R-CNN:
- (a) External selective search removed
- (b) Shared convolution + RoI pooling
- (c) Anchor boxes
- (d) Transformer encoder

**Q7.** Faster R-CNN adds:
- (a) YOLO grid
- (b) Region Proposal Network (RPN)
- (c) Mask head
- (d) SAM prompts

**Q8.** SSD detects at:
- (a) Single scale only
- (b) Multiple feature map scales
- (c) Image pyramid only
- (d) Video frames only

**Q9.** YOLO v1 treats detection as:
- (a) Graph cut
- (b) Single regression from grid cells
- (c) Clustering
- (d) Reinforcement learning

**Q10.** RetinaNet key innovation:
- (a) RoI pooling
- (b) Focal loss + FPN
- (c) Selective search
- (d) Hungarian matching

**Q11.** Focal loss parameter γ:
- (a) Controls anchor size
- (b) Down-weights easy examples
- (c) Sets NMS threshold
- (d) Defines grid size

**Q12.** YOLOv8 is typically used via:
- (a) Keras only
- (b) Ultralytics API
- (c) TensorFlow 1.x
- (d) OpenCV DNN only

**Q13.** DETR differs from Faster R-CNN by:
- (a) Using anchors
- (b) Set prediction with object queries, no NMS
- (c) No transformer
- (d) Single class only

**Q14.** DETR training uses:
- (a) Random assignment
- (b) Hungarian bipartite matching
- (c) K-means
- (d) Softmax only

**Q15.** Smooth L1 loss is used for:
- (a) Classification
- (b) Bounding box regression
- (c) Segmentation masks
- (d) Data loading

**Q16.** Two-stage vs single-stage:
- (a) Two-stage always faster
- (b) Single-stage runs one forward pass for dense predictions
- (c) Single-stage uses RPN
- (d) Two-stage has no proposals

**Q17.** GeoSpatial building detection often uses:
- (a) YOLOv8 on tiled satellite imagery
- (b) Linear regression
- (c) K-means only
- (d) Word2Vec

**Q18.** COCO box format in annotations:
- (a) cxcywh normalized
- (b) xywh absolute pixels
- (c) Polar coordinates
- (d) Lat/lon

**Q19.** RT-DETR improves DETR by:
- (a) Removing backbone
- (b) Real-time optimizations (efficient encoder, query selection)
- (c) Adding selective search
- (d) Using R-CNN only

**Q20.** Detection vs segmentation for pond counting:
- (a) Segmentation always faster for counts
- (b) Detection gives boxes for quick counts; segmentation gives exact boundaries
- (c) They are identical
- (d) Detection outputs pixel masks

---

See [solutions/quiz_answers.md](solutions/quiz_answers.md) after submitting.
