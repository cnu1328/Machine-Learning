# Module 04 — Types of Machine Learning

**Duration:** 2 weeks  
**Prerequisites:** Module 03 complete  
**Status:** Ready

---

## Overview

ML is not just "supervised vs unsupervised." This module covers every learning paradigm, when to choose each, and real-world examples from industry and your GeoSpatial work.

---

## Study Plan (2 weeks at 5–8 hrs/week)

| Day | Notebooks | Focus |
|-----|-----------|-------|
| 1–2 | 01–02 | Supervised & unsupervised recap |
| 3–4 | 03–04 | Semi-supervised & self-supervised |
| 5–6 | 05–07 | RL (UCB), active & online learning |
| 7–8 | 08–10 | Transfer, federated, meta learning |
| 9 | 11–12 | Curriculum, few-shot, zero-shot |
| 10 | Review | Exercises, assignment, quiz |

---

## Notebooks

| # | Notebook | Legacy Code |
|---|----------|-------------|
| 01 | [01_Supervised_Learning.ipynb](01_Supervised_Learning.ipynb) | All Day scripts |
| 02 | [02_Unsupervised_Learning.ipynb](02_Unsupervised_Learning.ipynb) | Day - 20, Day - 21 |
| 03 | [03_Semi_Supervised_Learning.ipynb](03_Semi_Supervised_Learning.ipynb) | New |
| 04 | [04_Self_Supervised_Learning.ipynb](04_Self_Supervised_Learning.ipynb) | New |
| 05 | [05_Reinforcement_Learning.ipynb](05_Reinforcement_Learning.ipynb) | Day - 25 UCB |
| 06 | [06_Active_Learning.ipynb](06_Active_Learning.ipynb) | New |
| 07 | [07_Online_Learning.ipynb](07_Online_Learning.ipynb) | New |
| 08 | [08_Transfer_Learning.ipynb](08_Transfer_Learning.ipynb) | water-bodies SE-ResNet50 |
| 09 | [09_Federated_Learning.ipynb](09_Federated_Learning.ipynb) | New |
| 10 | [10_Meta_Learning.ipynb](10_Meta_Learning.ipynb) | New |
| 11 | [11_Curriculum_Learning.ipynb](11_Curriculum_Learning.ipynb) | New |
| 12 | [12_Few_Shot_and_Zero_Shot_Learning.ipynb](12_Few_Shot_and_Zero_Shot_Learning.ipynb) | New |

---

## For Each Paradigm

Every notebook includes:

- Definition and intuition
- Mathematical formulation
- Working code demonstration
- When to choose it (decision criteria)
- Industry and GeoSpatial examples
- Exercises
- Interview questions

---

## Decision Guide

```
Have labels?
├── Yes, plenty → Supervised → Similar task? → Transfer Learning
├── Some labels → Semi-Supervised / Active Learning
└── No labels → Cluster? Unsupervised | Features? Self-Supervised | Rewards? RL
```

---

## Projects

| Paradigm | Project | Notebook |
|----------|---------|----------|
| Reinforcement Learning | UCB web ad optimization | 05 |
| Active Learning | Uncertainty sampling demo | 06 |
| Transfer Learning | ImageNet → GeoSpatial | 08 |
| Self-Supervised | Contrastive loss demo | 04 |
| Few-shot | Prototypical networks | 12 |

---

## Module Deliverables

- [ ] All 12 notebooks completed
- [ ] 6 exercises attempted
- [ ] Assignment: Paradigm selection report for 3 GeoSpatial projects
- [ ] Quiz ≥12/15 (80%)
- [ ] Gate questions answered in chat

---

## Assignment

**Paradigm Selection Report** — analyze 3 of your GeoSpatial projects, recommend paradigm combinations, implement one component.

See [exercises/README.md](exercises/README.md).

---

## Interview Questions

1. When would you use semi-supervised over supervised?
2. Explain exploration vs exploitation in RL.
3. How does transfer learning differ from meta-learning?
4. What is the UCB formula and why does it work?
5. How would you reduce labeling cost for satellite segmentation?
6. Explain CLIP zero-shot classification.
7. What is federated learning and when is it needed?

---

## Connection to Your Projects

| Project | Paradigms Used |
|---------|----------------|
| water-bodies-detection | Supervised + Transfer + Curriculum |
| All Day scripts | Supervised |
| Day - 20, 21 | Unsupervised |
| Day - 25 | Reinforcement (UCB) |

---

## Resources

- [CHEATSHEET.md](CHEATSHEET.md)
- [quiz/module_04_quiz.md](quiz/module_04_quiz.md)
- [exercises/README.md](exercises/README.md)

---

**Previous:** [03_Classical_ML/](../03_Classical_ML/)  
**Next:** [05_Deep_Learning/](../05_Deep_Learning/)
