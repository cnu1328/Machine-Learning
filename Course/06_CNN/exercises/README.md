# Module 06 Exercises

Attempt before checking [solutions/](solutions/).

---

## Foundations (Notebooks 01–06)

### Exercise 1 — Manual Convolution (Notebook 02)
Implement 2D convolution with stride=2 and padding=1. Verify against `F.conv2d`.

### Exercise 2 — Output Size Calculator (Notebook 03)
Write a function that given a CNN architecture (list of conv/pool layers), prints output shape at each layer for 32×32 input.

### Exercise 3 — Receptive Field (Notebook 06)
Compute RF for ResNet-50 layer by layer. At which layer does RF exceed 512?

### Exercise 4 — Filter Visualization (Notebook 05)
Train a small CNN on CIFAR-10 for 5 epochs. Visualize first-layer filters and activation maps for one image.

---

## Architectures (Notebooks 07–14)

### Exercise 5 — MNIST CNN (Notebook 14)
Train `MNISTCNN` on full MNIST. Beat Module 05 MLP accuracy (target: >98%).

### Exercise 6 — CIFAR-10 ResNet (Notebook 11)
Train `ResNetSmall` for 20 epochs on full CIFAR-10. Target: >70% test accuracy.

### Exercise 7 — Architecture Comparison
Train LeNet, AlexNetCIFAR, and ResNetSmall on same CIFAR-10 subset (5000 samples). Compare params, training time, accuracy.

---

## Module Assignment: CIFAR-10 CNN Benchmark

**Deliverable:** `exercises/assignment_cifar10_cnn.ipynb`

1. **Implement** a CNN achieving >75% test accuracy on CIFAR-10
2. **Compare** at least 3 architectures (e.g., custom CNN, ResNetSmall, pretrained fine-tune)
3. **Report** params, training time, train/val/test accuracy, loss curves
4. **Visualize** 10 misclassified images with predicted vs true labels
5. **Document** design choices (architecture, augmentation, optimizer, scheduler)
6. **Connect** to water-bodies: explain how SE-ResNet50 encoder relates to your ResNet implementation

---

## GeoSpatial Extension (Optional)

Apply pretrained ResNet50 feature extractor to a small satellite image classification task (3–5 land cover classes). Fine-tune last layer only vs full fine-tune — compare.

---

## Submission

> Module 06 exercises complete. Assignment attached. Quiz score: X/20.
