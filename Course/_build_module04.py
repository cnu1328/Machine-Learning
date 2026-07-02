#!/usr/bin/env python3
"""Generate Module 04 ML Paradigms notebooks (12 total)."""
import json
from pathlib import Path

M04 = Path(__file__).resolve().parent / "04_ML_Paradigms"


def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": [s]}


def code(s):
    return {"cell_type": "code", "metadata": {}, "source": [s], "outputs": [], "execution_count": None}


def save(name, cells):
    M04.mkdir(parents=True, exist_ok=True)
    (M04 / name).write_text(json.dumps(nb(cells), indent=1))
    print(f"  {name}")


def hdr(num, title, dur, objs, legacy=""):
    leg = f"\n\n**Legacy script:** `{legacy}`" if legacy else ""
    return md(
        f"# Notebook {num}: {title}\n\n**Module:** 04 ML Paradigms  \n**Duration:** ~{dur}{leg}\n\n---\n\n## Learning Objectives\n\n{objs}"
    )


def footer(summary, nxt=None):
    nxt = f"\n\n**Next:** [{nxt}]({nxt})" if nxt else ""
    return md(f"---\n\n## Interview Questions\n\nSee module quiz.\n\n## Summary\n\n{summary}{nxt}")


SETUP = code(
    "import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom pathlib import Path\n\nREPO = Path('../../').resolve()\nplt.rcParams['figure.figsize'] = (8, 5)\nrng = np.random.default_rng(42)"
)

NOTEBOOKS = []


def register(name, cells):
    NOTEBOOKS.append((name, cells))


# ── 01 Supervised Learning ──────────────────────────────────────────────

register("01_Supervised_Learning.ipynb", [
    hdr("01", "Supervised Learning", "2 hrs",
        "1. Define supervised learning formally\n2. Distinguish classification vs regression\n3. Map every Module 03 algorithm to supervised learning\n4. Know when supervised learning is the right choice"),
    md("## 1. Intuition\n\n**Supervised learning** = learn a mapping $f: X \\rightarrow Y$ from **labeled** examples $(x_i, y_i)$.\n\n- **Classification:** $Y$ is discrete (survived/died, cat/dog)\n- **Regression:** $Y$ is continuous (price, temperature)\n\nYou already implemented 20+ supervised algorithms in Module 03."),
    md("## 2. Mathematical Framework\n\nGiven training set $\\mathcal{D} = \\{(x_1, y_1), \\ldots, (x_n, y_n)\\}$:\n\n$$\\hat{f} = \\arg\\min_{f \\in \\mathcal{F}} \\frac{1}{n}\\sum_{i=1}^{n} L(f(x_i), y_i) + \\lambda \\Omega(f)$$\n\n- $\\mathcal{F}$ = hypothesis space (linear, trees, neural nets)\n- $L$ = loss function (MSE, cross-entropy)\n- $\\Omega$ = regularization (L1, L2, tree depth)"),
    SETUP,
    md("## 3. Your Legacy Scripts = Supervised Learning\n\n| Type | Your Scripts | Target |\n|------|-------------|--------|\n| Regression | Day - 11, 12, 13, 15, 16 | Continuous price/salary |\n| Classification | Day - 3, 4, 5, 6, 7, 10 | Binary/multi-class labels |"),
    code("# Quick supervised learning recap — house prices\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.metrics import r2_score\n\nhouse = pd.read_csv(REPO / 'houseprice.csv')\nX, y = house[['area']].values, house['price'].values\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\nmodel = LinearRegression().fit(X_train, y_train)\nprint(f'Supervised regression R²: {r2_score(y_test, model.predict(X_test)):.4f}')"),
    code("# Classification recap — heart disease\nfrom sklearn.linear_model import LogisticRegression\n\nheart = pd.read_csv(REPO / 'heart_Disease.csv')\ntarget = heart.columns[-1]\nX = heart.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(heart.median(numeric_only=True))\ny = heart[target]\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n\nclf = LogisticRegression(max_iter=1000).fit(X_train, y_train)\nprint(f'Supervised classification accuracy: {clf.score(X_test, y_test):.4f}')"),
    md("## 4. When to Choose Supervised Learning\n\n✅ **Use when:**\n- You have labeled data (enough for your model complexity)\n- Clear input → output mapping exists\n- Goal is prediction on new data\n\n❌ **Don't use when:**\n- No labels available (→ unsupervised or self-supervised)\n- Labels are extremely expensive (→ active learning)\n- Task changes continuously (→ online learning)"),
    md("## 5. GeoSpatial Examples\n\n- **Supervised segmentation:** water-bodies-detection (labeled pond polygons)\n- **Supervised classification:** building vs non-building from satellite tiles\n- **Supervised regression:** predict crop yield from spectral indices"),
    md("## Exercise\n\nList 3 supervised learning problems from your GeoSpatial projects. For each: input X, output Y, algorithm used."),
    code("# YOUR CODE HERE\n"),
    footer("Supervised learning is the default paradigm when labels exist.", "02_Unsupervised_Learning.ipynb"),
])

# ── 02 Unsupervised Learning ────────────────────────────────────────────

register("02_Unsupervised_Learning.ipynb", [
    hdr("02", "Unsupervised Learning", "2 hrs",
        "1. Define unsupervised learning\n2. Distinguish clustering vs dimensionality reduction\n3. Connect to Module 03 notebooks 20–27\n4. Walk through Day - 20 and Day - 21",
        "Day - 20, Day - 21"),
    md("## 1. Intuition\n\n**Unsupervised learning** = find structure in data **without labels**.\n\nTwo main goals:\n1. **Clustering:** group similar samples (customer segments, land cover types)\n2. **Dimensionality reduction:** compress features while preserving information (PCA, visualization)"),
    md("## 2. Mathematical Framework\n\nNo target $y$. Optimize objectives like:\n\n- **K-Means:** $\\min \\sum_k \\sum_{x_i \\in C_k} \\|x_i - \\mu_k\\|^2$\n- **PCA:** $\\max_v v^T C v$ subject to $\\|v\\|=1$ (maximize variance)\n- **Autoencoder:** $\\min \\|x - \\text{decode}(\\text{encode}(x))\\|^2$"),
    SETUP,
    code("from sklearn.cluster import KMeans\nfrom sklearn.decomposition import PCA\nfrom sklearn.datasets import load_iris\n\nX, y_true = load_iris(return_X_y=True)  # y_true only for visualization\n\n# Clustering (unsupervised — don't use y_true for training)\nkm = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)\nprint('K-Means cluster sizes:', np.bincount(km.labels_))\n\n# PCA (unsupervised)\npca = PCA(n_components=2).fit(X)\nX_pca = pca.transform(X)\nprint(f'PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}')"),
    code("fig, axes = plt.subplots(1, 2, figsize=(12, 4))\naxes[0].scatter(X_pca[:,0], X_pca[:,1], c=km.labels_, cmap='viridis', alpha=0.7)\naxes[0].set_title('K-Means Clusters (unsupervised)')\naxes[1].scatter(X_pca[:,0], X_pca[:,1], c=y_true, cmap='viridis', alpha=0.7)\naxes[1].set_title('True Labels (for comparison only)')\nplt.tight_layout(); plt.show()"),
    md("## 3. When to Choose\n\n✅ Exploratory data analysis\n✅ Customer/market segmentation\n✅ Anomaly detection (Isolation Forest)\n✅ Feature preprocessing (PCA before supervised)\n✅ Visualization (t-SNE, UMAP)\n\n❌ When you need accurate predictions on labeled targets"),
    md("## GeoSpatial: Unsupervised land cover clustering on satellite bands without polygon labels."),
    footer("Unsupervised learning discovers hidden structure without labels.", "03_Semi_Supervised_Learning.ipynb"),
])

# ── 03 Semi-Supervised Learning ─────────────────────────────────────────

register("03_Semi_Supervised_Learning.ipynb", [
    hdr("03", "Semi-Supervised Learning", "2.5 hrs",
        "1. Understand learning with few labels + many unlabeled samples\n2. Implement pseudo-labeling\n3. Apply Label Propagation\n4. Know when semi-supervised beats supervised"),
    md("## 1. Intuition\n\n**Problem:** Labeling is expensive. You have 100 labeled images and 10,000 unlabeled.\n\n**Semi-supervised learning** uses both labeled and unlabeled data.\n\n**Assumption (smoothness):** Points close in feature space should have similar labels."),
    md("## 2. Methods\n\n| Method | Idea |\n|--------|------|\n| **Pseudo-labeling** | Train on labeled, predict unlabeled, add confident predictions to training set, repeat |\n| **Label Propagation** | Spread labels through a similarity graph |\n| **Self-training** | Model generates its own training data |\n| **MixMatch / FixMatch** | Deep semi-supervised (Module 05+) |"),
    SETUP,
    code("from sklearn.semi_supervised import LabelPropagation\nfrom sklearn.datasets import make_classification\nfrom sklearn.model_selection import train_test_split\n\nX, y = make_classification(n_samples=500, n_features=10, random_state=42)\n\n# Hide 90% of labels (set to -1)\ny_semi = y.copy()\nmask = rng.random(len(y)) < 0.9\ny_semi[mask] = -1\nprint(f'Labeled: {(y_semi >= 0).sum()}, Unlabeled: {(y_semi < 0).sum()}')\n\nlp = LabelPropagation(kernel='knn', n_neighbors=7).fit(X, y_semi)\nprint(f'Label Propagation accuracy: {(lp.predict(X) == y).mean():.4f}')"),
    code("# Pseudo-labeling demo\ndef pseudo_label(X_l, y_l, X_u, base_clf, threshold=0.9):\n    base_clf.fit(X_l, y_l)\n    proba = base_clf.predict_proba(X_u)\n    conf = proba.max(axis=1)\n    preds = proba.argmax(axis=1)\n    confident = conf >= threshold\n    X_new = np.vstack([X_l, X_u[confident]])\n    y_new = np.concatenate([y_l, preds[confident]])\n    return X_new, y_new, confident.sum()\n\nfrom sklearn.linear_model import LogisticRegression\nX_l, X_u, y_l, y_u = X[~mask], X[mask], y[~mask], y[mask]\nclf = LogisticRegression(max_iter=1000)\nX_new, y_new, n_added = pseudo_label(X_l, y_l, X_u, clf)\nprint(f'Added {n_added} pseudo-labeled samples')"),
    md("## GeoSpatial Example\n\n10 labeled aquaculture ponds + 5000 unlabeled tiles → semi-supervised segmentation reduces labeling cost by 90%."),
    md("## Exercise\n\nImplement 3 iterations of pseudo-labeling. Plot accuracy vs iteration."),
    code("# YOUR CODE HERE\n"),
    footer("Semi-supervised learning leverages cheap unlabeled data to boost limited labels.", "04_Self_Supervised_Learning.ipynb"),
])

# ── 04 Self-Supervised Learning ─────────────────────────────────────────

register("04_Self_Supervised_Learning.ipynb", [
    hdr("04", "Self-Supervised Learning", "2.5 hrs",
        "1. Define self-supervised learning\n2. Understand pretext tasks\n3. Learn contrastive learning intuition (SimCLR)\n4. Connect to GeoSpatial pretraining"),
    md("## 1. Intuition\n\n**Self-supervised learning** creates labels **from the data itself** — no human annotation.\n\nExamples of pretext tasks:\n- Predict rotation angle of image (0°, 90°, 180°, 270°)\n- Predict masked patch in image (BERT for images — MAE)\n- Contrastive: augmented views of same image should be similar\n\nAfter pretraining, fine-tune on small labeled dataset."),
    md("## 2. Contrastive Learning (SimCLR)\n\nFor image $x$, create two augmentations $x_i, x_j$:\n\n- **Positive pair:** $(x_i, x_j)$ from same image\n- **Negative pairs:** $(x_i, x_k)$ from different images\n\n**Loss (InfoNCE):** pull positives together, push negatives apart\n\n$$L = -\\log \\frac{\\exp(\\text{sim}(z_i, z_j)/\\tau)}{\\sum_k \\exp(\\text{sim}(z_i, z_k)/\\tau)}$$"),
    SETUP,
    code("# Simplified contrastive loss demo\ndef contrastive_loss(z_i, z_j, z_negatives, temperature=0.5):\n    \"\"\"InfoNCE for one positive pair vs negatives.\"\"\"\n    sim_pos = np.dot(z_i, z_j) / (np.linalg.norm(z_i) * np.linalg.norm(z_j))\n    sim_neg = [np.dot(z_i, z_n) / (np.linalg.norm(z_i) * np.linalg.norm(z_n)) for z_n in z_negatives]\n    logits = np.array([sim_pos] + sim_neg) / temperature\n    logits = logits - logits.max()  # stability\n    exp_logits = np.exp(logits)\n    return -np.log(exp_logits[0] / exp_logits.sum())\n\nz_anchor = rng.normal(0, 1, 64)\nz_positive = z_anchor + rng.normal(0, 0.1, 64)  # similar\nz_negs = [rng.normal(0, 1, 64) for _ in range(5)]  # different\nprint(f'Contrastive loss: {contrastive_loss(z_anchor, z_positive, z_negs):.4f}')"),
    md("## GeoSpatial Applications\n\n- Pretrain on unlabeled Planet imagery (millions of tiles)\n- Fine-tune on 500 labeled ponds for water-bodies-detection\n- DINO, MoCo, SimCLR on satellite data (Module 10)"),
    md("## When to Choose\n\n✅ Massive unlabeled data + small labeled set\n✅ Domain-specific pretraining (satellite, medical)\n❌ Already have large labeled dataset (→ supervised directly)"),
    footer("Self-supervised learning creates supervision from data structure — foundation of modern CV.", "05_Reinforcement_Learning.ipynb"),
])

# ── 05 Reinforcement Learning ───────────────────────────────────────────

register("05_Reinforcement_Learning.ipynb", [
    hdr("05", "Reinforcement Learning", "3 hrs",
        "1. Define MDP, agent, environment, reward\n2. Implement multi-armed bandit and UCB\n3. Understand Q-learning basics\n4. Extend Day - 25 web ad optimization",
        "Day - 25 Web Ad. Optimization upper Confidence Bound Reinforcement Learning.py"),
    md("## 1. Intuition\n\n**Reinforcement Learning (RL):** An **agent** takes **actions** in an **environment** to maximize **cumulative reward**.\n\nUnlike supervised learning:\n- No labeled (input, output) pairs\n- Agent discovers good actions through trial and error\n- Delayed rewards (action now affects future outcomes)"),
    md("## 2. Markov Decision Process (MDP)\n\n$(S, A, P, R, \\gamma)$:\n- $S$ = states\n- $A$ = actions\n- $P(s'|s,a)$ = transition probability\n- $R(s,a)$ = reward\n- $\\gamma$ = discount factor\n\n**Goal:** Find policy $\\pi(a|s)$ maximizing $G = \\sum_{t=0}^{\\infty} \\gamma^t R_t$"),
    md("## 3. Multi-Armed Bandit & UCB\n\nSimplest RL problem: $K$ slot machines (arms), each with unknown reward rate $\\mu_k$.\n\n**Exploration vs Exploitation:**\n- Exploit: pull best-known arm\n- Explore: try other arms to discover better ones\n\n**Upper Confidence Bound (UCB):**\n$$a_t = \\arg\\max_k \\left[ \\hat{\\mu}_k + \\sqrt{\\frac{2\\ln t}{n_k}} \\right]$$\n\nYour **Day - 25** implements UCB for web ad optimization."),
    SETUP,
    code("# Multi-Armed Bandit simulation\nn_arms = 5\ntrue_means = rng.uniform(0.2, 0.8, n_arms)\nprint('True arm means:', np.round(true_means, 3))\n\n# UCB algorithm\ncounts = np.zeros(n_arms)\nvalues = np.zeros(n_arms)\ntotal_reward = 0\nrewards_history = []\n\nfor t in range(1, 5001):\n    if t <= n_arms:\n        arm = t - 1  # explore each arm once\n    else:\n        ucb = values + np.sqrt(2 * np.log(t) / np.maximum(counts, 1))\n        arm = ucb.argmax()\n    reward = rng.random() < true_means[arm]\n    counts[arm] += 1\n    values[arm] += (reward - values[arm]) / counts[arm]\n    total_reward += reward\n    rewards_history.append(reward)\n\nprint(f'Best arm: {true_means.argmax()}, Selected most: {counts.argmax()}')\nprint(f'Arm pull counts: {counts.astype(int)}')"),
    code("plt.plot(pd.Series(rewards_history).rolling(100).mean())\nplt.xlabel('Round'); plt.ylabel('Avg reward (100-round window)')\nplt.title('UCB Multi-Armed Bandit — Day - 25 Concept'); plt.show()"),
    md("## 4. Q-Learning (Preview)\n\n$$Q(s,a) \\leftarrow Q(s,a) + \\alpha [R + \\gamma \\max_{a'} Q(s',a') - Q(s,a)]$$\n\nUsed in game playing (Day - 30 Snake), robotics, resource allocation."),
    md("## GeoSpatial RL Examples\n\n- Optimize satellite acquisition scheduling\n- Route planning for UAV surveys\n- Active learning as bandit problem (which tile to label next)"),
    md("## Exercise\n\nImplement epsilon-greedy bandit. Compare cumulative reward with UCB over 5000 rounds."),
    code("# YOUR CODE HERE\n"),
    footer("RL learns through interaction and reward — UCB solves the exploration-exploitation tradeoff.", "06_Active_Learning.ipynb"),
])

# ── 06 Active Learning ──────────────────────────────────────────────────

register("06_Active_Learning.ipynb", [
    hdr("06", "Active Learning", "2 hrs",
        "1. Define active learning\n2. Implement uncertainty sampling\n3. Simulate labeling budget constraints\n4. Apply to GeoSpatial annotation"),
    md("## 1. Intuition\n\n**Problem:** Labeling 10,000 satellite tiles costs $50,000. You have budget for 500 labels.\n\n**Active learning:** Model **chooses** which samples to label next — picks most informative ones.\n\n**Query strategies:**\n- **Uncertainty sampling:** label samples where model is least confident\n- **Margin sampling:** smallest difference between top-2 class probabilities\n- **Query-by-committee:** models disagree most"),
    SETUP,
    code("from sklearn.datasets import make_classification\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\n\nX, y = make_classification(n_samples=1000, n_features=10, random_state=42)\nX_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n# Start with 10 labeled samples\nlabeled_idx = rng.choice(len(X_pool), 10, replace=False)\nunlabeled_mask = np.ones(len(X_pool), dtype=bool)\nunlabeled_mask[labeled_idx] = False\n\naccuracies = []\nfor iteration in range(20):\n    X_l = X_pool[labeled_idx]\n    y_l = y_pool[labeled_idx]\n    clf = LogisticRegression(max_iter=1000).fit(X_l, y_l)\n    acc = clf.score(X_test, y_test)\n    accuracies.append(acc)\n    \n    # Uncertainty sampling: pick least confident unlabeled sample\n    X_unlabeled = X_pool[unlabeled_mask]\n    if len(X_unlabeled) == 0: break\n    proba = clf.predict_proba(X_unlabeled)\n    uncertainty = 1 - proba.max(axis=1)\n    pick = uncertainty.argmax()\n    unlabeled_indices = np.where(unlabeled_mask)[0]\n    labeled_idx = np.append(labeled_idx, unlabeled_indices[pick])\n    unlabeled_mask[unlabeled_indices[pick]] = False\n\nplt.plot(range(10, 10+len(accuracies)), accuracies, 'o-')\nplt.xlabel('Number of labeled samples'); plt.ylabel('Test accuracy')\nplt.title('Active Learning — Uncertainty Sampling'); plt.show()"),
    md("## GeoSpatial: Label the 500 most uncertain pond tiles instead of random 500 → better model with same budget."),
    footer("Active learning maximizes model quality per labeling dollar.", "07_Online_Learning.ipynb"),
])

# ── 07 Online Learning ──────────────────────────────────────────────────

register("07_Online_Learning.ipynb", [
    hdr("07", "Online Learning", "2 hrs",
        "1. Define online/incremental learning\n2. Understand concept drift\n3. Use SGDClassifier for streaming data\n4. Compare batch vs online updates"),
    md("## 1. Intuition\n\n**Online learning:** Update model as each new sample arrives — don't retrain from scratch.\n\n**Use cases:**\n- Streaming sensor data\n- Fraud detection (patterns change daily)\n- Recommendation systems (new users/items constantly)\n\n**Concept drift:** Data distribution changes over time. Model must adapt."),
    SETUP,
    code("from sklearn.linear_model import SGDClassifier\n\n# Simulate streaming data with concept drift\nX_stream, y_stream = make_classification(n_samples=2000, n_features=5, random_state=42)\n\n# Batch: train once on first 1000\nsgd_batch = SGDClassifier(loss='log_loss', random_state=42, max_iter=1000)\nsgd_batch.fit(X_stream[:1000], y_stream[:1000])\nbatch_acc = sgd_batch.score(X_stream[1000:], y_stream[1000:])\n\n# Online: update one sample at a time\nsgd_online = SGDClassifier(loss='log_loss', random_state=42)\nfor i in range(1000):\n    sgd_online.partial_fit(X_stream[i:i+1], y_stream[i:i+1], classes=[0, 1])\nfor i in range(1000, 2000):\n    sgd_online.partial_fit(X_stream[i:i+1], y_stream[i:i+1])\nonline_acc = sgd_online.score(X_stream[1000:], y_stream[1000:])\n\nprint(f'Batch accuracy (drift period):  {batch_acc:.4f}')\nprint(f'Online accuracy (drift period): {online_acc:.4f}')"),
    footer("Online learning adapts to streaming data and concept drift without full retraining.", "08_Transfer_Learning.ipynb"),
])

# ── 08 Transfer Learning ────────────────────────────────────────────────

register("08_Transfer_Learning.ipynb", [
    hdr("08", "Transfer Learning", "2.5 hrs",
        "1. Define transfer learning\n2. Understand fine-tuning vs feature extraction\n3. Preview ImageNet → GeoSpatial transfer\n4. Connect to water-bodies SE-ResNet50 encoder"),
    md("## 1. Intuition\n\n**Transfer learning:** Use knowledge from **source task** to improve **target task**.\n\n- Pretrain on ImageNet (1.2M images, 1000 classes)\n- Fine-tune on your aquaculture pond dataset (500 images)\n\n**Why it works:** Early layers learn universal features (edges, textures). Later layers are task-specific."),
    md("## 2. Strategies\n\n| Strategy | Description | When |\n|----------|-------------|------|\n| **Feature extraction** | Freeze pretrained layers, train new head | Small target dataset |\n| **Fine-tuning** | Unfreeze some/all layers, train with low LR | Medium target dataset |\n| **Full training** | Train from scratch | Large target dataset or very different domain |"),
    md("## 3. Your water-bodies-detection\n\n```python\n# model.py uses SE-ResNet50 with ImageNet weights\nencoder = smp.encoders.get_encoder('se_resnet50', weights='imagenet')\n# Stage 1: freeze encoder, train decoder\n# Stage 2: unfreeze all, fine-tune\n```\n\nThis is **exactly** transfer learning — covered in depth in Module 05–07."),
    SETUP,
    code("# Transfer learning with sklearn (feature extraction analogy)\nfrom sklearn.datasets import load_digits\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.decomposition import PCA\n\nX, y = load_digits(return_X_y=True)\n\n# 'Pretrain' = PCA learns structure from ALL data (unsupervised pretraining)\npca = PCA(n_components=32).fit(X)\nX_features = pca.transform(X)\n\n# 'Fine-tune' = classifier on extracted features\nX_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)\nclf = LogisticRegression(max_iter=1000).fit(X_train, y_train)\nprint(f'Transfer (PCA features) accuracy: {clf.score(X_test, y_test):.4f}')\n\n# Compare: raw pixels\nX_train_r, X_test_r, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)\nclf_raw = LogisticRegression(max_iter=1000).fit(X_train_r, y_train)\nprint(f'From scratch (raw pixels) accuracy: {clf_raw.score(X_test_r, y_test):.4f}')"),
    footer("Transfer learning is essential when target data is limited — your entire segmentation pipeline uses it.", "09_Federated_Learning.ipynb"),
])

# ── 09 Federated Learning ───────────────────────────────────────────────

register("09_Federated_Learning.ipynb", [
    hdr("09", "Federated Learning", "2 hrs",
        "1. Define federated learning\n2. Understand FedAvg algorithm\n3. Simulate multi-client training\n4. Know privacy and communication tradeoffs"),
    md("## 1. Intuition\n\n**Problem:** Hospital/clinic/farm data cannot leave premises (privacy, regulations).\n\n**Federated learning:** Train a global model by aggregating **local model updates** without sharing raw data.\n\n1. Server sends global model to clients\n2. Each client trains on local data\n3. Clients send weight updates (not data) to server\n4. Server averages updates → new global model"),
    md("## 2. FedAvg Algorithm\n\n$$w_{global} \\leftarrow \\sum_{k=1}^{K} \\frac{n_k}{n} w_k$$\n\nWeighted average of client model weights by dataset size."),
    SETUP,
    code("# Simulate FedAvg with 5 clients\nK = 5\nn_samples = [100, 200, 150, 80, 120]\ntotal = sum(n_samples)\n\n# Global model weights (1D for demo)\nw_global = np.zeros(10)\n\nfor round_num in range(10):\n    client_weights = []\n    for k in range(K):\n        # Each client trains locally (simulated as global + noise)\n        w_k = w_global + rng.normal(0, 0.1, 10)\n        client_weights.append(w_k)\n    # FedAvg\n    w_global = sum(n_k / total * w_k for n_k, w_k in zip(n_samples, client_weights))\n\nprint(f'Global model after 10 rounds: {np.round(w_global, 3)}')"),
    md("## GeoSpatial: Multiple farms share a pond detection model without sharing satellite imagery or farm locations."),
    footer("Federated learning enables collaborative ML under privacy constraints.", "10_Meta_Learning.ipynb"),
])

# ── 10 Meta Learning ────────────────────────────────────────────────────

register("10_Meta_Learning.ipynb", [
    hdr("10", "Meta Learning", "2.5 hrs",
        "1. Define 'learning to learn'\n2. Understand MAML intuition\n3. Distinguish meta-learning from transfer learning\n4. Preview few-shot applications"),
    md("## 1. Intuition\n\n**Meta-learning:** Train a model that can **quickly adapt** to new tasks with few examples.\n\n- **Transfer learning:** Source task → target task (one-time transfer)\n- **Meta-learning:** Many tasks → learn initialization that adapts fast to any new task\n\n**MAML (Model-Agnostic Meta-Learning):**\nFind $\\theta^*$ such that one gradient step on any new task gives good performance:\n$$\\theta^* = \\arg\\min_\\theta \\sum_{\\mathcal{T}_i} L_{\\mathcal{T}_i}(\\theta - \\alpha \\nabla_\\theta L_{\\mathcal{T}_i}(\\theta))$$"),
    SETUP,
    code("# MAML intuition demo: find initialization good for multiple tasks\n# Each task: y = a*x + b with different a, b\ntasks = [(rng.uniform(1,3), rng.uniform(-2,2)) for _ in range(5)]\n\n# Bad initialization\nw_bad = np.array([0.0, 0.0])\n# Good meta-initialization (average of task-optimal weights)\nw_meta = np.array([np.mean([a for a,_ in tasks]), np.mean([b for _,b in tasks])])\n\nfor label, w in [('Bad init', w_bad), ('Meta init', w_meta)]:\n    total_loss = 0\n    for a, b in tasks:\n        x = rng.uniform(-1, 1, 20)\n        y = a * x + b + rng.normal(0, 0.1, 20)\n        pred = w[0] * x + w[1]\n        total_loss += np.mean((pred - y)**2)\n    print(f'{label}: avg loss across tasks = {total_loss/len(tasks):.4f}')"),
    footer("Meta-learning optimizes for fast adaptation — foundation of few-shot learning.", "11_Curriculum_Learning.ipynb"),
])

# ── 11 Curriculum Learning ────────────────────────────────────────────

register("11_Curriculum_Learning.ipynb", [
    hdr("11", "Curriculum Learning", "2 hrs",
        "1. Define curriculum learning\n2. Implement easy-to-hard sample ordering\n3. Compare random vs curriculum training\n4. Apply to GeoSpatial training"),
    md("## 1. Intuition\n\nHumans learn addition before calculus. **Curriculum learning** trains on **easy samples first**, gradually increasing difficulty.\n\n**Benefits:** Faster convergence, better generalization, more stable training."),
    md("## 2. Difficulty Metrics\n\n| Domain | Easy | Hard |\n|--------|------|------|\n| Classification | High confidence correct | Low margin, misclassified |\n| Segmentation | Large uniform regions | Thin boundaries, small objects |\n| GeoSpatial | Single isolated pond | Dense adjacent ponds with shared bunds |"),
    SETUP,
    code("from sklearn.datasets import make_classification\nfrom sklearn.linear_model import SGDClassifier\n\nX, y = make_classification(n_samples=500, n_features=5, n_informative=3,\n                            n_redundant=1, class_sep=0.5, random_state=42)\n\n# Difficulty = distance from class center (easy = far from boundary)\ncenter_0, center_1 = X[y==0].mean(axis=0), X[y==1].mean(axis=0)\ndist_to_boundary = np.abs((X - center_0) @ (center_1 - center_0))\ndifficulty_order = np.argsort(-dist_to_boundary)  # easy first\nrandom_order = rng.permutation(len(X))\n\nfor order_name, order in [('Random', random_order), ('Curriculum', difficulty_order)]:\n    clf = SGDClassifier(loss='log_loss', random_state=42)\n    losses = []\n    for i in order:\n        clf.partial_fit(X[i:i+1], y[i:i+1], classes=[0, 1])\n        if len(losses) == 0 or i % 50 == 0:\n            losses.append(1 - clf.score(X, y))\n    plt.plot(losses, label=order_name)\n\nplt.xlabel('Checkpoint'); plt.ylabel('Training error'); plt.legend()\nplt.title('Curriculum vs Random Training Order'); plt.show()"),
    md("## water-bodies-detection: Train on isolated ponds first, then complex multi-pond scenes."),
    footer("Curriculum learning mirrors human education — easy examples first, hard examples later.", "12_Few_Shot_and_Zero_Shot_Learning.ipynb"),
])

# ── 12 Few-shot & Zero-shot ─────────────────────────────────────────────

register("12_Few_Shot_and_Zero_Shot_Learning.ipynb", [
    hdr("12", "Few-Shot and Zero-Shot Learning", "2.5 hrs",
        "1. Define N-way K-shot learning\n2. Understand prototypical networks\n3. Preview CLIP zero-shot classification\n4. Connect to GeoSpatial novel class detection"),
    md("## 1. Definitions\n\n**Few-shot:** Classify new categories from **K examples per class** (e.g., 5-way 1-shot = 5 classes, 1 example each).\n\n**Zero-shot:** Classify **without any examples** — using text descriptions or attributes.\n\n**CLIP (Contrastive Language-Image Pre-training):** Maps images and text to same embedding space. Zero-shot: compare image to text prompts like \"a satellite image of aquaculture pond\" vs \"forest\"."),
    md("## 2. Prototypical Networks\n\nFor N-way K-shot:\n1. Compute **prototype** (mean embedding) for each class from K support examples\n2. Classify query by nearest prototype\n\n$$c_k = \\frac{1}{K}\\sum_{i} f_\\theta(x_i^k) \\quad \\text{(prototype for class k)}$$"),
    SETUP,
    code("# Prototypical network demo\ndef compute_prototypes(X_support, y_support, n_way=3):\n    prototypes = {}\n    for c in range(n_way):\n        prototypes[c] = X_support[y_support == c].mean(axis=0)\n    return prototypes\n\ndef classify_query(x_query, prototypes):\n    dists = {c: np.linalg.norm(x_query - p) for c, p in prototypes.items()}\n    return min(dists, key=dists.get)\n\nfrom sklearn.datasets import load_iris\nX, y = load_iris(return_X_y=True)\n\n# 3-way 3-shot\nprototypes = compute_prototypes(X[:9], y[:9], n_way=3)\ncorrect = sum(classify_query(X[i], prototypes) == y[i] for i in range(9, 30))\nprint(f'Prototypical network accuracy (3-way 3-shot): {correct/21:.4f}')"),
    md("## GeoSpatial Zero-Shot\n\n```python\n# CLIP-style zero-shot (conceptual)\nprompts = ['aquaculture pond', 'river', 'forest', 'urban area']\n# image_embedding @ text_embeddings.T → classify without training examples\n```\n\nEnables detecting new land cover types without labeled training data."),
    md("## Module 04 Complete\n\nYou now understand all major ML paradigms and when to choose each.\n\n**Next module:** [05_Deep_Learning](../05_Deep_Learning/) — build neural networks from scratch."),
    md("## Exercise\n\nDesign a paradigm selection plan for a new GeoSpatial project: detect solar panels with 50 labeled images and 100,000 unlabeled satellite tiles."),
    code("# YOUR CODE HERE\n"),
    footer("Few-shot and zero-shot learning enable ML with minimal or no labeled examples.", None),
])


def main():
    print("Building Module 04 notebooks...")
    for name, cells in NOTEBOOKS:
        save(name, cells)
    print(f"Done: {len(NOTEBOOKS)} notebooks created.")


if __name__ == "__main__":
    main()
