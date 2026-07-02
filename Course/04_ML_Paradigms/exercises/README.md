# Module 04 Exercises

Attempt before checking [solutions/](solutions/).

---

## Notebook Exercises

### Exercise 1 — Paradigm Identification (Notebook 01)
For each problem, identify the best ML paradigm:
1. Predict house prices from features
2. Group customers by purchase behavior (no labels)
3. 50 labeled pond images + 10,000 unlabeled tiles
4. Choose which web ad to display
5. Detect new land cover type with 3 examples per class

### Exercise 2 — UCB vs Epsilon-Greedy (Notebook 05)
Implement epsilon-greedy (ε=0.1) bandit. Plot cumulative reward vs UCB over 5000 rounds.

### Exercise 3 — Active Learning Budget (Notebook 06)
With labeling budget of 50 samples, compare random sampling vs uncertainty sampling. Plot accuracy curve.

### Exercise 4 — Transfer Learning Experiment (Notebook 08)
On MNIST digits: compare (a) train from scratch, (b) PCA features + classifier, (c) train on subset with transfer from full dataset.

### Exercise 5 — FedAvg Simulation (Notebook 09)
Simulate 5 clients with non-IID data (each client has different class distribution). Compare FedAvg vs centralized training.

### Exercise 6 — Prototypical Network (Notebook 12)
Implement 5-way 1-shot classification on Iris using Euclidean distance to prototypes.

---

## Module Assignment: Paradigm Selection Report

**Deliverable:** `exercises/assignment_paradigm_selection.ipynb`

For **three** of your real GeoSpatial projects (water-bodies-detection, building-detection, road-detection, etc.):

1. **Describe the problem** — input, output, data available
2. **Identify the current paradigm** used in the project
3. **Evaluate alternatives** — would semi-supervised, active learning, or transfer learning improve it?
4. **Design an improved pipeline** using at least 2 paradigms combined
5. **Implement one component** in code (e.g., UCB for tile selection, pseudo-labeling demo, or FedAvg simulation)

Include a decision flowchart for each project.

---

## Mini Projects

| Project | Paradigm | Dataset |
|---------|----------|---------|
| Web ad UCB | Reinforcement | Simulated bandit |
| Pond tile active learning | Active Learning | Simulated pool |
| Satellite pretext task | Self-Supervised | Unlabeled tiles concept |
| Solar panel zero-shot | Zero-shot | CLIP prompts (conceptual) |

---

## Submission

> Module 04 exercises complete. Assignment attached. Quiz score: X/15.
