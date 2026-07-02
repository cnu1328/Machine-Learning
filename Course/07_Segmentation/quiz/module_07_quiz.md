# Module 07 Quiz

**Passing score:** 16/20 (80%)

---

**Q1.** Binary segmentation output activation:
- (a) Softmax
- (b) Sigmoid
- (c) ReLU
- (d) Tanh

**Q2.** Multi-class segmentation uses:
- (a) Sigmoid per channel
- (b) Softmax across classes
- (c) No activation
- (d) Linear only

**Q3.** Your water-bodies dual-head output is:
- (a) Multi-class
- (b) Multi-label (2 binary channels)
- (c) Instance segmentation
- (d) Panoptic only

**Q4.** UNet skip connections preserve:
- (a) Batch norm stats
- (b) Fine spatial details
- (c) Learning rate
- (d) Class labels

**Q5.** UNet++ differs from UNet by:
- (a) No encoder
- (b) Nested dense skip pathways
- (c) No decoder
- (d) Only 1×1 convolutions

**Q6.** DeepLab ASPP captures:
- (a) Single scale only
- (b) Multi-scale context via atrous conv
- (c) Temporal information
- (d) Text prompts

**Q7.** Dice loss is preferred when:
- (a) Classes balanced
- (b) Foreground is small (class imbalance)
- (c) Regression task
- (d) Detection task

**Q8.** IoU formula:
- (a) intersection / union
- (b) 2×intersection / sum
- (c) intersection / foreground
- (d) union / intersection

**Q9.** AquaBoundaryLoss combines:
- (a) MSE on both heads
- (b) BCE+Dice on aqua and boundary heads
- (c) Cross-entropy only
- (d) Focal only

**Q10.** water-bodies encoder is:
- (a) VGG-16
- (b) SE-ResNet50 UNet++
- (c) LeNet
- (d) AlexNet

**Q11.** Boundary head purpose:
- (a) Color correction
- (b) Separate adjacent ponds via bund detection
- (c) Increase image resolution
- (d) Data augmentation

**Q12.** Focal loss parameter γ controls:
- (a) Learning rate
- (b) Focus on hard examples
- (c) Number of classes
- (d) Batch size

**Q13.** SAM requires:
- (a) Fine-tuning always
- (b) Prompts (points/boxes)
- (c) 3-band RGB only
- (d) Instance labels

**Q14.** mIoU is:
- (a) Max IoU
- (b) Mean IoU across classes
- (c) Min IoU
- (d) Median IoU

**Q15.** Semantic segmentation assigns:
- (a) Unique ID per object
- (b) Class label per pixel
- (c) Bounding boxes
- (d) Keypoints

**Q16.** BCEDiceLoss in your project uses weights:
- (a) 0.5 BCE + 0.5 Dice
- (b) 0.35 BCE + 0.65 Dice
- (c) 0.65 BCE + 0.35 Dice
- (d) 1.0 BCE only

**Q17.** Post-process GIS threshold (0.8) vs inference (0.5):
- (a) Same purpose
- (b) Higher threshold for precision-friendly polygons
- (c) Lower is always better
- (d) Random choice

**Q18.** Planet input bands in water-bodies:
- (a) 3 (RGB)
- (b) 6 (bands 2,3,4,6,7,8)
- (c) 1 (grayscale)
- (d) 12

**Q19.** Lovász loss optimizes:
- (a) MSE directly
- (b) IoU surrogate directly
- (c) Accuracy only
- (d) Bounding box IoU

**Q20.** Instance segmentation differs from semantic by:
- (a) No pixels
- (b) Separating individual object instances
- (c) Using regression
- (d) No neural network

---

See [solutions/quiz_answers.md](solutions/quiz_answers.md) after submitting.
