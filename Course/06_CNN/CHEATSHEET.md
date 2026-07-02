# Module 06 Cheat Sheet — CNN

## Output Size Formula

$$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

## Tensor Shapes

| Object | Shape | Example |
|--------|-------|---------|
| Grayscale image | (H, W) | 28×28 MNIST |
| RGB image | (3, H, W) | 3×32×32 CIFAR |
| Batch | (N, C, H, W) | 32×3×224×224 |
| Planet tile | (6, 512, 512) | water-bodies input |

## Convolution

$$(I * K)_{ij} = \sum_m \sum_n I_{i+m,j+n} \cdot K_{m,n}$$

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
F.conv2d(input, weight, bias, stride, padding)
```

## Pooling

```python
F.max_pool2d(x, kernel_size=2, stride=2)
F.avg_pool2d(x, kernel_size=2, stride=2)
nn.AdaptiveAvgPool2d(1)  # global average pool
```

## Receptive Field

$$RF_l = RF_{l-1} + (K_l - 1) \times \prod_{i=1}^{l-1} S_i$$

## Architecture Timeline

| Model | Year | Key Innovation | Params (approx) |
|-------|------|----------------|-----------------|
| LeNet-5 | 1998 | First CNN | 60K |
| AlexNet | 2012 | ReLU + Dropout + GPU | 60M |
| VGG-16 | 2014 | 3×3 stacks | 138M |
| GoogLeNet | 2015 | Inception modules | 5M |
| ResNet-50 | 2016 | Skip connections | 25M |
| DenseNet | 2017 | Dense connections | 8M |
| EfficientNet | 2019 | Compound scaling | 5M (B0) |
| ConvNeXt | 2022 | Modernized ResNet | 28M (T) |

## ResNet Block

```python
def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    return F.relu(out + self.shortcut(x))  # skip connection
```

## PyTorch CNN Template

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128*8*8, 256), nn.ReLU(), nn.Linear(256, 10))
    def forward(self, x):
        return self.classifier(self.features(x))
```

## Pretrained Models (torchvision)

```python
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
```

## Your water-bodies Pipeline

```
Input (6, 512, 512) → SE-ResNet50 encoder → UNet++ decoder → (2, 512, 512)
         ↑ Module 06 teaches this
```

## Common Mistakes

- Wrong input shape (NHWC vs NCHW)
- Forgetting padding → shrinking spatial size
- Not using `model.eval()` for inference
- Training without normalizing inputs
- Ignoring receptive field for segmentation

## Interview Questions

1. Derive conv output size formula
2. Max pool vs avg pool — when to use each?
3. Why does ResNet use skip connections?
4. 1×1 convolution purpose?
5. What is receptive field and why does it matter for segmentation?
