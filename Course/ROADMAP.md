# Course Roadmap

**Pace:** 5–8 hours/week  
**Total duration:** 18–24 months  
**Prerequisite:** Python basics (Module 01 is ML-focused revision, not beginner Python)

---

## Phase 1: Foundation (Months 1–4)

### Module 00 — Course Introduction (1 day)

- Learning contract, setup verification, course map
- First conceptual question: Python list vs NumPy array

### Module 01 — Python Revision (1 week)

**Focus:** NumPy, Pandas, Matplotlib, vectorization — the exact tools every ML module uses.

| Notebook | Topics |
|----------|--------|
| 01 NumPy Foundations | ndarrays, broadcasting, dot product, axis operations |
| 02 Pandas for ML | DataFrames, EDA, missing values, your CSV datasets |
| 03 Matplotlib | scatter, histogram, heatmap, confusion matrix preview |
| 04 Vectorization | loops vs NumPy, timing, KNN distance preview |

**Mini project:** EDA on `TitanicSurvival.csv`  
**Links to legacy code:** Reimplement data loading from `Day - 11`, `Day - 6`

### Module 02 — Mathematics (6–8 weeks)

**Part A: Linear Algebra (3 weeks)**

Scalars, vectors, matrices, matrix multiplication, dot product, cross product, eigenvalues, eigenvectors, matrix inverse, rank, orthogonality, projection

**Part B: Calculus (2 weeks)**

Functions, derivatives, partial derivatives, chain rule, gradient, Jacobian, Hessian, optimization landscapes

**Part C: Probability & Statistics (2 weeks)**

Random variables, Bayes theorem, Gaussian, Bernoulli, Binomial, Poisson, conditional probability, likelihood, MLE, mean, variance, covariance, correlation, hypothesis testing, confidence intervals

**Part D: Optimization (1 week)**

Gradient descent, SGD, Momentum, RMSProp, Adam, AdamW — full mathematical derivation of each

**Part E: Loss Functions (1 week)**

MSE, MAE, Cross-Entropy, Hinge, Dice, IoU, Focal — intuition + math + implementation

---

## Phase 2: Classical Machine Learning (Months 5–8)

### Module 03 — Classical ML (8–10 weeks)

Every algorithm with: why it exists, math derivation, code walkthrough, hyperparameters, pros/cons, exercises.

| Week | Algorithm | Your Legacy Script |
|------|-----------|-------------------|
| 1 | Linear Regression | `Day - 11 House price Prediction Using Linear Regression.py` |
| 1 | Polynomial Regression | `Day - 12 Salary prediction using polynomial Regression.py` |
| 2 | Logistic Regression | `Day - 3 Logistic_regression_Heart_Diseases.py` |
| 2 | Model Evaluation (Classification) | `Day - 9 Evaluating Classification model performance.py` |
| 3 | Naive Bayes | `Day - 6 Titanic Survival prediction.py` |
| 3 | KNN | `Day - 4 Salary_Estimatiom_by_KNerst.py` |
| 4 | Decision Trees | `Day - 7 Leaf_species_Detection_DescisionTree.py` |
| 4 | Random Forest | `Day - 16 Car price Prediction using Random Forest REgression.py` |
| 5 | SVM | `Day - 5 Handwritten_Digit_Recognition.py`, `Day - 13 Stock Prediction Using Support Vector Regression.py` |
| 5 | AdaBoost, Gradient Boosting | *New notebooks* |
| 6 | XGBoost, LightGBM, CatBoost | *New notebooks* |
| 6 | Bias-Variance, Cross-Validation | `Day - 17 & 18 Evaluating Regression Model...` |
| 7 | K-Means, Hierarchical, DBSCAN | `Day - 20 Clustering income spent using Hierarchial clustering.py` |
| 7 | Gaussian Mixture Models | *New notebook* |
| 8 | PCA, LDA, t-SNE, UMAP | `Day - 21 Plant Iris Clustering Uing Principal Component Analysis.py` |
| 8 | Isolation Forest, Autoencoders | *New notebooks* |
| 9 | Feature Engineering & Selection | *New notebooks* |
| 10 | Multi-algorithm comparison | `Day - 10 Breat cancer Detection_VariousMLAlgorithm.py` |

**Projects:** Spam detection, house prices, Titanic, customer churn, movie recommendation (`Day - 22`)

### Module 04 — ML Paradigms (2 weeks)

Supervised, unsupervised, semi-supervised, self-supervised, reinforcement (`Day - 25`, `Day - 30`), active, online, transfer, federated, meta, curriculum, few-shot, zero-shot

For each: where used, when to choose, industry examples, mini projects.

---

## Phase 3: Deep Learning Core (Months 9–11)

### Module 05 — Deep Learning (4–5 weeks)

From scratch in PyTorch (extending `Day - 28`, `Day - 29`):

- Artificial neuron, perceptron, MLP
- Forward propagation, backpropagation (full derivation)
- Activation functions, loss functions, optimizers
- Weight initialization, regularization, dropout, batch norm
- Learning rate scheduling, residual learning

**Projects:** MNIST from scratch, COVID detection (modernized from Day 29)

### Module 06 — CNN (4 weeks)

Image representation, convolution, padding, stride, pooling, receptive field

Architectures implemented/studied: LeNet, AlexNet, VGG, GoogLeNet, ResNet, DenseNet, EfficientNet, ConvNeXt

**Projects:** CIFAR-10, plant disease detection

---

## Phase 4: Computer Vision (Months 12–16)

### Module 07 — Segmentation (5–6 weeks)

Binary, multi-class, multi-label, semantic, instance, panoptic, boundary detection

Architectures: FCN, UNet, UNet++, DeepLab, PSPNet, SegFormer, Mask2Former, SAM, HRNet

Losses: Dice, CE, Focal, IoU, Boundary, Lovász

Metrics: Dice, IoU, Precision, Recall, F1, mIoU, confusion matrix

**Bridge to capstone:** Introduction to water-bodies-detection architecture

### Module 08 — Object Detection (4 weeks)

R-CNN, Fast R-CNN, Faster R-CNN, SSD, YOLO (v5/v8/v11), RetinaNet, DETR, RT-DETR

### Module 09 — Instance Segmentation (3 weeks)

Mask R-CNN, YOLACT, SOLO, panoptic segmentation

---

## Phase 5: Modern DL + Production (Months 17–20)

### Module 10 — Transformers (4 weeks)

Attention, self-attention, multi-head attention, ViT, Swin, SegFormer, Mask2Former, CLIP, DINO, SAM, GroundingDINO

Full mathematical treatment of attention mechanism.

### Module 11 — Production ML (4 weeks)

Data pipelines, training pipelines, MLflow, DVC, TensorBoard, ONNX, TensorRT, Triton, Docker, FastAPI, model versioning, monitoring, CI/CD, mixed precision, distributed training, quantization, pruning, knowledge distillation

**Projects:** Model deployment, REST API for inference

---

## Phase 6: Capstone (Months 21–24)

### Module 12 — water-bodies-detection Walkthrough (4 weeks)

Line-by-line explanation of every file in [water-bodies-detection](../../water-bodies-detection/):

| Week | Focus | Files |
|------|-------|-------|
| 1 | Data prep & config | `config/default.yaml`, `tile_and_mask.py` |
| 2 | Model & loss | `model.py`, `losses.py` |
| 3 | Training & dataset | `dataset.py`, `train.py` |
| 4 | Inference, post-process, deploy | `predict.py`, `post_process/`, `automate/`, `Dockerfile` |

Topics: dual-head UNet++, Planet bands, dual masks, two-stage training, sliding-window inference, TTA, GIS vectorization.

---

## Progression Diagram

```
00 Intro → 01 Python → 02 Math → 03 Classical ML → 04 Paradigms
    → 05 Deep Learning → 06 CNN → 07 Segmentation → 08 Detection
    → 09 Instance Seg → 10 Transformers → 11 Production → 12 Capstone
```

## Module Completion Checklist

Each module ends with:

- Summary
- Cheat sheet
- Interview questions
- Coding exercises
- Assignments
- Mini project
- Real-world applications
- Revision notes
- Common mistakes
- Best practices
- Further reading
- Quiz (≥80% to pass)

## GIS / GeoSpatial Thread

Your domain expertise is woven throughout:

- Module 01: rasters as NumPy arrays
- Module 07: satellite segmentation, road/water/building detection
- Module 11: GDAL Docker, batch inference pipelines
- Module 12: full aquaculture pond detection pipeline

Other repos for future elective projects:

- `building-detection/`, `cultivation-land-detection/`, `vegetation-detection/`, `multi-class-road-detection/`, `road-hit-plots/`
