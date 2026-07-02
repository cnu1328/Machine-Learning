# Module 04 Quiz — Answer Key

| Q | Answer |
|---|--------|
| 1 | (b) |
| 2 | (b) |
| 3 | (c) |
| 4 | (b) |
| 5 | (b) |
| 6 | (b) |
| 7 | (b) |
| 8 | (b) |
| 9 | (b) |
| 10 | (b) |
| 11 | (c) |
| 12 | (b) |
| 13 | (b) |
| 14 | (b) |
| 15 | (b) |

## Exercise 1 Answers

1. Supervised (regression)
2. Unsupervised (clustering)
3. Semi-Supervised (+ possibly Active Learning for label selection)
4. Reinforcement Learning (multi-armed bandit / UCB)
5. Few-shot learning

## Exercise 2 — Epsilon-Greedy Sketch

```python
def epsilon_greedy(true_means, eps=0.1, rounds=5000):
    counts = np.zeros(len(true_means))
    values = np.zeros(len(true_means))
    rewards = []
    for t in range(rounds):
        if rng.random() < eps:
            arm = rng.integers(len(true_means))
        else:
            arm = values.argmax()
        reward = rng.random() < true_means[arm]
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        rewards.append(reward)
    return rewards
```
