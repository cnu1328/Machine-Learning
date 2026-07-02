# Module 04 Cheat Sheet — ML Paradigms

## Paradigm Selection Flowchart

```
Have labels?
├── Yes, plenty → Supervised Learning
│   └── Similar pretrained task? → Transfer Learning
├── Some labels → Semi-Supervised / Active Learning
└── No labels
    ├── Need clusters → Unsupervised
    ├── Need features → Self-Supervised
    └── Sequential decisions → Reinforcement Learning
```

## All Paradigms at a Glance

| Paradigm | Labels Needed | Key Idea | Your Code |
|----------|---------------|----------|-----------|
| Supervised | Full labels | Learn f: X→Y | All Day scripts, Module 03 |
| Unsupervised | None | Find structure | Day - 20, Day - 21 |
| Semi-Supervised | Few labels + unlabeled | Propagate/pseudo-label | New |
| Self-Supervised | None (self-generated) | Pretext tasks | Preview Module 05/10 |
| Reinforcement | Reward signal | Maximize cumulative reward | Day - 25 UCB |
| Active Learning | Few labels (query) | Pick most informative to label | New |
| Online Learning | Streaming labels | Incremental updates | SGD partial_fit |
| Transfer Learning | Target labels | Reuse pretrained model | water-bodies SE-ResNet50 |
| Federated Learning | Local labels | Aggregate without sharing data | New |
| Meta Learning | Many tasks | Learn to adapt fast | MAML |
| Curriculum Learning | Labels + ordering | Easy → hard training | water-bodies staging |
| Few-shot | K examples/class | Prototype/metric learning | Prototypical nets |
| Zero-shot | None (text/attrs) | CLIP-style embedding match | Module 10 preview |

## Key Formulas

**UCB (Day - 25):**
$$a_t = \arg\max_k \left[\hat{\mu}_k + \sqrt{\frac{2\ln t}{n_k}}\right]$$

**Q-Learning:**
$$Q(s,a) \leftarrow Q(s,a) + \alpha[R + \gamma\max_{a'}Q(s',a') - Q(s,a)]$$

**FedAvg:**
$$w_{global} = \sum_k \frac{n_k}{n} w_k$$

**Contrastive (InfoNCE):**
$$L = -\log\frac{\exp(\text{sim}(z_i,z_j)/\tau)}{\sum_k\exp(\text{sim}(z_i,z_k)/\tau)}$$

**Prototypical:**
$$c_k = \frac{1}{K}\sum_i f_\theta(x_i^k), \quad \hat{y} = \arg\min_k \|f_\theta(x_q) - c_k\|$$

## When to Choose What

| Scenario | Best Paradigm |
|----------|---------------|
| 10K labeled house prices | Supervised |
| 100 labels + 50K unlabeled tiles | Semi-Supervised + Active Learning |
| No labels, explore customer segments | Unsupervised (K-Means) |
| No labels, pretrain satellite encoder | Self-Supervised |
| Which ad to show? | Reinforcement (UCB/bandit) |
| 500 labeled ponds, ImageNet exists | Transfer Learning |
| Data on 5 farms, can't share | Federated Learning |
| New crop type, 5 examples | Few-shot |
| "Find solar panels" text query | Zero-shot (CLIP) |
| Model degrades over months | Online Learning |

## GeoSpatial Decision Examples

| Project | Paradigm |
|---------|----------|
| water-bodies-detection | Supervised + Transfer + Curriculum |
| Unlabeled land cover exploration | Unsupervised clustering |
| Label budget for road annotation | Active Learning |
| Multi-farm pond model | Federated Learning |
| New building type, 10 examples | Few-shot |
| Pretrain on unlabeled Planet tiles | Self-Supervised |

## Interview Questions

1. Supervised vs unsupervised vs semi-supervised?
2. Exploration vs exploitation in RL?
3. Transfer learning vs meta-learning?
4. When is active learning worth the complexity?
5. How does CLIP enable zero-shot classification?
