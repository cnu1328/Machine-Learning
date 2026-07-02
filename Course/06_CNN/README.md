# Module 06 — Convolutional Neural Networks

**Duration:** 4 weeks  
**Prerequisites:** Module 05 complete  
**Status:** Ready

---

## Overview

Computer Vision fundamentals: convolution math from scratch, then every major CNN architecture from LeNet to ConvNeXt. Connects directly to your **water-bodies-detection** SE-ResNet50 encoder.

**Framework:** NumPy (conv from scratch) + PyTorch (architectures & training)

---

## Study Plan (4 weeks at 5–8 hrs/week)

| Week | Notebooks | Focus |
|------|-----------|-------|
| 1 | 01–06 | Images, conv, padding, pooling, feature maps, RF |
| 2 | 07–09 | LeNet, AlexNet, VGG |
| 3 | 10–12 | GoogLeNet, ResNet, DenseNet |
| 4 | 13–14 | EfficientNet, ConvNeXt, capstone projects |

---

## Part 1: Foundations (Notebooks 01–06)

| # | Notebook | Topic |
|---|----------|-------|
| 01 | [01_Image_Representation.ipynb](01_Image_Representation.ipynb) | Pixels, channels, NCHW tensors |
| 02 | [02_Convolution_Operation.ipynb](02_Convolution_Operation.ipynb) | Manual conv in NumPy |
| 03 | [03_Padding_and_Stride.ipynb](03_Padding_and_Stride.ipynb) | Output size formula |
| 04 | [04_Pooling.ipynb](04_Pooling.ipynb) | Max/avg pooling |
| 05 | [05_Feature_Maps.ipynb](05_Feature_Maps.ipynb) | Filter visualization |
| 06 | [06_Receptive_Field.ipynb](06_Receptive_Field.ipynb) | RF computation |

---

## Part 2: Architectures (Notebooks 07–14)

| # | Notebook | Year | Innovation |
|---|----------|------|------------|
| 07 | [07_LeNet.ipynb](07_LeNet.ipynb) | 1998 | First CNN |
| 08 | [08_AlexNet.ipynb](08_AlexNet.ipynb) | 2012 | ReLU + dropout |
| 09 | [09_VGG.ipynb](09_VGG.ipynb) | 2014 | 3×3 depth |
| 10 | [10_GoogLeNet.ipynb](10_GoogLeNet.ipynb) | 2015 | Inception |
| 11 | [11_ResNet.ipynb](11_ResNet.ipynb) | 2016 | Skip connections |
| 12 | [12_DenseNet.ipynb](12_DenseNet.ipynb) | 2017 | Dense connections |
| 13 | [13_EfficientNet.ipynb](13_EfficientNet.ipynb) | 2019 | Compound scaling |
| 14 | [14_ConvNeXt.ipynb](14_ConvNeXt.ipynb) | 2022 | Modernized CNN |

---

## Connection to Your Work

```
water-bodies-detection:
  Input (6, 512, 512) Planet bands
    → SE-ResNet50 encoder  ← Module 06 teaches this (ResNet + SE attention)
    → UNet++ decoder       ← Module 07
    → (2, 512, 512) masks
```

---

## Module Deliverables

- [ ] All 14 notebooks completed
- [ ] Manual convolution implemented (Exercise 1)
- [ ] MNIST CNN >98% accuracy
- [ ] CIFAR-10 assignment >75% accuracy
- [ ] Quiz ≥16/20 (80%)

---

## Assignment

**CIFAR-10 CNN Benchmark** — implement and compare 3 architectures, >75% test accuracy.

See [exercises/README.md](exercises/README.md).

---

## Key Formula

$$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

---

## Resources

- [CHEATSHEET.md](CHEATSHEET.md)
- [quiz/module_06_quiz.md](quiz/module_06_quiz.md)
- [exercises/README.md](exercises/README.md)

---

**Previous:** [05_Deep_Learning/](../05_Deep_Learning/)  
**Next:** [07_Segmentation/](../07_Segmentation/)
