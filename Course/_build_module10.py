#!/usr/bin/env python3
"""Generate Module 10 Transformers notebooks (13 total)."""
import json
from pathlib import Path

M10 = Path(__file__).resolve().parent / "10_Transformers"


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
    M10.mkdir(parents=True, exist_ok=True)
    (M10 / name).write_text(json.dumps(nb(cells), indent=1))
    print(f"  {name}")


def hdr(num, title, dur, objs):
    return md(
        f"# Notebook {num}: {title}\n\n**Module:** 10 Transformers  \n**Duration:** ~{dur}\n\n---\n\n## Learning Objectives\n\n{objs}"
    )


def footer(summary, nxt=None):
    nxt = f"\n\n**Next:** [{nxt}]({nxt})" if nxt else ""
    return md(f"---\n\n## Summary\n\n{summary}{nxt}")


SETUP = code(
    "import numpy as np\nimport matplotlib.pyplot as plt\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport math\n\nplt.rcParams['figure.figsize'] = (8, 5)\ntorch.manual_seed(42)\nnp.random.seed(42)\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nprint(f'Device: {device}')"
)

ATTENTION_NUMPY = code(
    "def scaled_dot_product_attention(Q, K, V, mask=None):\n    \"\"\"NumPy: Q,K,V shapes (..., seq, d_k). Returns output, weights.\"\"\"\n    d_k = Q.shape[-1]\n    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) / np.sqrt(d_k)\n    if mask is not None:\n        scores = np.where(mask, scores, -1e9)\n    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))\n    weights = weights / weights.sum(axis=-1, keepdims=True)\n    out = np.matmul(weights, V)\n    return out, weights"
)

ATTENTION_TORCH = code(
    "def scaled_dot_product_attention_torch(Q, K, V, mask=None):\n    d_k = Q.size(-1)\n    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)\n    if mask is not None:\n        scores = scores.masked_fill(mask == 0, float('-inf'))\n    weights = F.softmax(scores, dim=-1)\n    return torch.matmul(weights, V), weights"
)

NOTEBOOKS = []


def register(name, cells):
    NOTEBOOKS.append((name, cells))


# ── ATTENTION FOUNDATIONS (01-05) ───────────────────────────────────────

register("01_Attention_Intuition.ipynb", [
    hdr("01", "Attention Intuition", "2 hrs",
        "1. Understand why attention was invented\n2. Map Query, Key, Value analogy\n3. Contrast attention with CNN receptive fields\n4. Preview vision transformer applications"),
    md("## Why Attention?\n\n**RNN problem:** Sequential processing — hard to parallelize, vanishing gradients on long sequences.\n\n**CNN problem:** Local receptive field — must stack many layers for global context.\n\n**Attention:** Each token can **directly** attend to any other token in one step.\n\n## Query, Key, Value Analogy\n\nThink of a **search engine**:\n- **Query (Q):** What you're looking for\n- **Key (K):** Index labels on documents\n- **Value (V):** Document content\n\nAttention score = similarity(Q, K) → weighted sum of V."),
    SETUP,
    code("# Toy example: 3 words attend to each other\nwords = ['pond', 'bund', 'water']\nd_model = 4\n# Random embeddings\nX = np.random.randn(3, d_model)\n# Simple attention: dot product similarity\nscores = X @ X.T\nweights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)\nout = weights @ X\nprint('Attention weights (3x3):')\nprint(np.round(weights, 3))\nprint('\\nOutput shape:', out.shape)"),
    md("## Vision Context\n\nIn ViT, each **patch token** attends to all other patches — a 16×16 patch grid of 224×224 image = 196 tokens with global context in layer 1."),
    footer("Attention = soft lookup: compare queries to keys, aggregate values.", "02_Scaled_Dot_Product_Attention.ipynb"),
])

register("02_Scaled_Dot_Product_Attention.ipynb", [
    hdr("02", "Scaled Dot-Product Attention", "2.5 hrs",
        "1. Derive scaled dot-product attention\n2. Implement from scratch in NumPy\n3. Explain why scale by sqrt(d_k)\n4. Apply causal mask for decoders"),
    md("## Scaled Dot-Product Attention (Vaswani et al., 2017)\n\n$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right) V$$\n\n**Shapes:** $Q, K, V \\in \\mathbb{R}^{\\text{seq} \\times d_k}$\n\n**Why scale?** Dot products grow with $d_k$ → softmax saturates → tiny gradients. Dividing by $\\sqrt{d_k}$ keeps variance stable."),
    SETUP, ATTENTION_NUMPY,
    code("seq, d_k, d_v = 4, 8, 8\nQ = np.random.randn(seq, d_k)\nK = np.random.randn(seq, d_k)\nV = np.random.randn(seq, d_v)\nout, w = scaled_dot_product_attention(Q, K, V)\nprint('Output shape:', out.shape)\nprint('Weights sum per row:', w.sum(axis=1))\n\nplt.imshow(w, cmap='Blues'); plt.colorbar()\nplt.xticks(range(seq), ['t0','t1','t2','t3']); plt.yticks(range(seq), ['t0','t1','t2','t3'])\nplt.title('Attention weights'); plt.show()"),
    md("## Causal Mask (Decoder)\n\nAutoregressive models (GPT) mask future tokens:\n\n```\n[[1, 0, 0, 0],\n [1, 1, 0, 0],\n [1, 1, 1, 0],\n [1, 1, 1, 1]]\n```"),
    code("seq = 4\ncausal = np.tril(np.ones((seq, seq))).astype(bool)\nout_causal, w_causal = scaled_dot_product_attention(Q, K, V, mask=causal)\nplt.imshow(w_causal, cmap='Blues'); plt.title('Causal attention'); plt.show()"),
    footer("Scaled dot-product attention is the core operation of all transformers.", "03_Self_Attention.ipynb"),
])

register("03_Self_Attention.ipynb", [
    hdr("03", "Self-Attention", "2.5 hrs",
        "1. Derive self-attention with Q=K=V from input\n2. Track matrix dimensions through layers\n3. Implement self-attention in PyTorch\n4. Compute O(n²) complexity"),
    md("## Self-Attention\n\n**Self-attention:** Q, K, V all derived from the **same** input sequence X.\n\n$$Q = XW_Q, \\quad K = XW_K, \\quad V = XW_V$$\n\n**Input:** $X \\in \\mathbb{R}^{\\text{seq} \\times d_{model}}$\n\n**Output:** Same shape as X — each token replaced by weighted mix of all tokens.\n\n**Complexity:** $O(n^2 \\cdot d)$ for sequence length $n$ — bottleneck for long sequences."),
    SETUP, ATTENTION_TORCH,
    code("class SelfAttention(nn.Module):\n    def __init__(self, d_model, d_k):\n        super().__init__()\n        self.W_q = nn.Linear(d_model, d_k, bias=False)\n        self.W_k = nn.Linear(d_model, d_k, bias=False)\n        self.W_v = nn.Linear(d_model, d_k, bias=False)\n    def forward(self, x):\n        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)\n        out, w = scaled_dot_product_attention_torch(Q, K, V)\n        return out, w\n\nx = torch.randn(2, 6, 64)  # batch, seq, d_model\nsa = SelfAttention(64, 32)\nout, w = sa(x)\nprint(f'Input: {x.shape} -> Output: {out.shape}, Weights: {w.shape}')"),
    md("## Dimension Cheat Sheet\n\n| Tensor | Shape |\n|--------|-------|\n| X | (batch, seq, d_model) |\n| Q, K | (batch, seq, d_k) |\n| V | (batch, seq, d_v) |\n| Weights | (batch, seq, seq) |\n| Output | (batch, seq, d_v) |"),
    footer("Self-attention: each token attends to all tokens; Q,K,V from same input.", "04_Multi_Head_Attention.ipynb"),
])

register("04_Multi_Head_Attention.ipynb", [
    hdr("04", "Multi-Head Attention", "2.5 hrs",
        "1. Derive multi-head attention\n2. Implement MultiHeadAttention in PyTorch\n3. Understand head splitting and concatenation\n4. Build transformer encoder block"),
    md("## Multi-Head Attention\n\nInstead of one attention, run **h parallel heads** on different subspaces:\n\n$$\\text{MultiHead}(Q,K,V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_h) W_O$$\n\n$$\\text{head}_i = \\text{Attention}(QW_Q^i, KW_K^i, VW_V^i)$$\n\n**Why multiple heads?** Different heads learn different relationships (local vs global, syntax vs semantics)."),
    SETUP,
    code("class MultiHeadAttention(nn.Module):\n    def __init__(self, d_model, num_heads):\n        super().__init__()\n        assert d_model % num_heads == 0\n        self.num_heads = num_heads\n        self.d_k = d_model // num_heads\n        self.W_q = nn.Linear(d_model, d_model)\n        self.W_k = nn.Linear(d_model, d_model)\n        self.W_v = nn.Linear(d_model, d_model)\n        self.W_o = nn.Linear(d_model, d_model)\n\n    def forward(self, x, mask=None):\n        B, S, D = x.shape\n        Q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)\n        K = self.W_k(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)\n        V = self.W_v(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)\n        out, w = scaled_dot_product_attention_torch(Q, K, V, mask)\n        out = out.transpose(1, 2).contiguous().view(B, S, D)\n        return self.W_o(out), w\n\nmha = MultiHeadAttention(64, 8)\nx = torch.randn(2, 10, 64)\nout, w = mha(x)\nprint(f'MHA output: {out.shape}, weights: {w.shape}')"),
    code("class TransformerEncoderBlock(nn.Module):\n    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):\n        super().__init__()\n        self.mha = MultiHeadAttention(d_model, num_heads)\n        self.ff = nn.Sequential(\n            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))\n        self.norm1 = nn.LayerNorm(d_model)\n        self.norm2 = nn.LayerNorm(d_model)\n        self.drop = nn.Dropout(dropout)\n    def forward(self, x):\n        attn_out, _ = self.mha(x)\n        x = self.norm1(x + self.drop(attn_out))\n        ff_out = self.ff(x)\n        x = self.norm2(x + self.drop(ff_out))\n        return x\n\nblock = TransformerEncoderBlock(64, 8, 256)\nprint('Encoder block output:', block(torch.randn(2, 10, 64)).shape)"),
    footer("Multi-head attention runs parallel heads; encoder block = MHA + FFN + residuals.", "05_Positional_Encoding.ipynb"),
])

register("05_Positional_Encoding.ipynb", [
    hdr("05", "Positional Encoding", "2 hrs",
        "1. Explain why transformers need position information\n2. Derive sinusoidal positional encoding\n3. Compare learned vs fixed encodings\n4. Understand 2D extensions for vision"),
    md("## Why Positional Encoding?\n\nSelf-attention is **permutation invariant** — shuffling tokens gives same result. Must inject **position** information.\n\n## Sinusoidal Encoding (Original Transformer)\n\n$$PE_{(pos, 2i)} = \\sin(pos / 10000^{2i/d})$$\n$$PE_{(pos, 2i+1)} = \\cos(pos / 10000^{2i/d})$$\n\nAdded to input embeddings before first layer."),
    SETUP,
    code("def sinusoidal_pe(seq_len, d_model):\n    pe = np.zeros((seq_len, d_model))\n    pos = np.arange(seq_len)[:, None]\n    div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))\n    pe[:, 0::2] = np.sin(pos * div)\n    pe[:, 1::2] = np.cos(pos * div)\n    return pe\n\npe = sinusoidal_pe(100, 64)\nplt.imshow(pe, aspect='auto', cmap='RdBu'); plt.xlabel('dim'); plt.ylabel('position')\nplt.title('Sinusoidal positional encoding'); plt.colorbar(); plt.show()"),
    md("## Vision: 2D Positional Encoding\n\nViT uses **learned** 1D position embeddings for patch sequence.\n\nSwin uses **relative** position bias within windows.\n\nSegFormer uses efficient positional encoding in MLP decoder."),
    footer("Positional encoding injects order; vision uses learned or relative variants.", "06_Vision_Transformer.ipynb"),
])

# ── VISION TRANSFORMERS (06-13) ─────────────────────────────────────────

register("06_Vision_Transformer.ipynb", [
    hdr("06", "Vision Transformer (ViT)", "3 hrs",
        "1. Understand patch embedding\n2. Know ViT architecture end-to-end\n3. Implement ViT patch embedding in PyTorch\n4. Compare ViT vs CNN inductive bias"),
    md("## ViT (Dosovitskiy et al., 2020)\n\n**Treat image as sequence of patches:**\n\n1. Split $224 \\times 224$ image into $16 \\times 16$ patches → $14 \\times 14 = 196$ tokens\n2. Linear embed each patch → $d_{model}$\n3. Prepend [CLS] token\n4. Add positional embedding\n5. Transformer encoder × L layers\n6. [CLS] → MLP head for classification\n\n**Key insight:** With enough data, less inductive bias (no conv) is fine."),
    SETUP,
    code("class PatchEmbed(nn.Module):\n    def __init__(self, img_size=224, patch_size=16, in_ch=3, d_model=768):\n        super().__init__()\n        self.n_patches = (img_size // patch_size) ** 2\n        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch_size, stride=patch_size)\n    def forward(self, x):\n        x = self.proj(x)\n        x = x.flatten(2).transpose(1, 2)\n        return x\n\npe = PatchEmbed(224, 16, 3, 768)\nx = torch.randn(2, 3, 224, 224)\nprint(f'Patches: {pe(x).shape}')  # (2, 196, 768)"),
    code("try:\n    import timm\n    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=10)\n    print(f'timm ViT params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')\nexcept ImportError:\n    print('Optional: pip install timm for pretrained ViT')"),
    md("## ViT vs CNN\n\n| | CNN | ViT |\n|---|-----|-----|\n| Inductive bias | Locality, translation equivariance | None (learned) |\n| Data needed | Less | More (or pretrain) |\n| Global context | Deep layers | Every layer |\n| GeoSpatial | Proven (your UNet++) | Emerging (SegFormer) |"),
    footer("ViT = patchify image + standard transformer encoder + CLS classification.", "07_Swin_Transformer.ipynb"),
])

register("07_Swin_Transformer.ipynb", [
    hdr("07", "Swin Transformer", "2.5 hrs",
        "1. Understand hierarchical vision transformer\n2. Know shifted window attention\n3. Compare linear complexity vs ViT\n4. Apply to high-resolution satellite imagery"),
    md("## Swin Transformer (Liu et al., 2021)\n\n**Problems with ViT:** Quadratic cost on patch count; fixed resolution; no hierarchy.\n\n**Swin solutions:**\n1. **Window attention:** Self-attention within local $M \\times M$ windows → $O(n)$ per layer\n2. **Shifted windows:** Alternate layers shift window partition for cross-window connections\n3. **Hierarchical:** Patch merging between stages (like CNN pyramid)\n\n**Result:** ViT-like accuracy with CNN-like efficiency for dense prediction."),
    SETUP,
    md("## Window Attention\n\n```\nLayer L:   [W1][W2]     Layer L+1 (shifted):  [  W1'  ]\n           [W3][W4]                           [  W2'  ]\n```\n\nShift connects previously separated windows."),
    md("## GeoSpatial\n\nSwin backbones used in remote sensing classification and detection — handles large tiles via hierarchical design."),
    footer("Swin = hierarchical ViT with shifted window attention for efficiency.", "08_SegFormer.ipynb"),
])

register("08_SegFormer.ipynb", [
    hdr("08", "SegFormer", "2.5 hrs",
        "1. Understand MiT encoder + MLP decoder\n2. Compare SegFormer to UNet++ for segmentation\n3. Know when to choose transformer vs CNN seg\n4. Connect to modern land cover pipelines"),
    md("## SegFormer (Xie et al., 2021)\n\n**Efficient semantic segmentation transformer:**\n\n- **Encoder:** Hierarchical MiT (Mix Transformer) — like Swin-lite\n- **Decoder:** Lightweight all-MLP (no heavy upsampling conv)\n- **No positional encoding** in decoder — saves compute\n\n**Performance:** Competitive mIoU on ADE20K/Cityscapes with fewer params than many CNN seg models."),
    SETUP,
    md("## SegFormer vs Your UNet++\n\n| | UNet++ (water-bodies) | SegFormer |\n|---|----------------------|----------|\n| Backbone | SE-ResNet50 (CNN) | MiT (Transformer) |\n| Inductive bias | Strong spatial | Weak, data-driven |\n| Multi-band input | 6 Planet bands | Flexible |\n| Production maturity | Your deployed pipeline | Research/modern alt |\n| Best when | Proven GeoSpatial workflow | Large labeled land cover |\n\n**Recommendation:** Keep UNet++ for aquaculture; evaluate SegFormer for multi-class land cover at scale."),
    footer("SegFormer = hierarchical transformer encoder + simple MLP decoder for segmentation.", "09_Mask2Former.ipynb"),
])

register("09_Mask2Former.ipynb", [
    hdr("09", "Mask2Former", "2.5 hrs",
        "1. Understand mask classification paradigm\n2. Connect to Module 09 instance/panoptic\n3. Know query-based universal segmentation\n4. Compare to UNet and Mask R-CNN"),
    md("## Mask2Former (Cheng et al., 2022)\n\n**Mask classification:** Predict N mask proposals + class labels (like DETR for segmentation).\n\n**Architecture:**\n1. Pixel decoder (multi-scale features)\n2. Transformer decoder with learned queries\n3. Each query → class logits + mask logits\n4. Hungarian matching to GT masks\n\n**Universal:** Same architecture for semantic, instance, panoptic (switch training data).\n\n*(Deep dive also in Module 09 Notebook 07.)*"),
    SETUP,
    md("## Paradigm Shift\n\n| Old | Mask2Former |\n|-----|-------------|\n| Per-pixel softmax (semantic) | Query → mask |\n| Mask R-CNN RoI head (instance) | Query → mask |\n| Panoptic FPN merge heuristics | Unified queries |"),
    footer("Mask2Former unifies segmentation via query-based mask classification.", "10_DINO.ipynb"),
])

register("10_DINO.ipynb", [
    hdr("10", "DINO (Self-Supervised ViT)", "2 hrs",
        "1. Understand self-supervised vision pretraining\n2. Know teacher-student distillation in DINO\n3. Apply to unlabeled satellite imagery\n4. Compare to supervised ImageNet pretraining"),
    md("## DINO (Caron et al., 2021)\n\n**Self-supervised learning** for ViT without labels.\n\n**Method:**\n1. Student network sees augmented crop\n2. Teacher network (EMA of student) sees different crop\n3. Match teacher output distribution to student\n4. Teacher provides stable targets\n\n**Result:** ViT features cluster semantically without labels — excellent for downstream segmentation with few labels."),
    SETUP,
    md("## GeoSpatial Application\n\n**Pretrain DINO on unlabeled Planet/Sentinel tiles** → fine-tune UNet++/SegFormer with limited pond labels.\n\n**Your workflow extension:**\n```\nUnlabeled satellite corpus → DINO pretrain → fine-tune segmentation head\n```\n\nCheaper than labeling millions of pixels; complements your supervised water-bodies pipeline."),
    footer("DINO learns visual features from unlabeled images via self-distillation.", "11_CLIP.ipynb"),
])

register("11_CLIP.ipynb", [
    hdr("11", "CLIP (Vision-Language)", "2.5 hrs",
        "1. Understand contrastive image-text pretraining\n2. Use CLIP for zero-shot classification\n3. Apply text prompts to GeoSpatial detection\n4. Know limitations for remote sensing"),
    md("## CLIP (Radford et al., 2021)\n\n**Contrastive Language-Image Pretraining:**\n\n- Image encoder (ViT or ResNet) + text encoder (Transformer)\n- Train on 400M image-text pairs\n- Loss: maximize cosine similarity of matching pairs, minimize non-matching\n\n**Zero-shot:** Classify by comparing image embedding to text prompts:\n- \"a satellite image of aquaculture pond\"\n- \"a satellite image of bare soil\"\n- \"a satellite image of vegetation\""),
    SETUP,
    code("try:\n    import clip\n    print('OpenAI CLIP available')\n    # model, preprocess = clip.load('ViT-B/32')\n    # text = clip.tokenize(['pond', 'building', 'road'])\nexcept ImportError:\n    print('Optional: pip install git+https://github.com/openai/CLIP.git')\n\n# Concept: cosine similarity\nimg_emb = F.normalize(torch.randn(1, 512), dim=-1)\ntxt_emb = F.normalize(torch.randn(3, 512), dim=-1)\nsim = (img_emb @ txt_emb.T).softmax(dim=-1)\nprint('Zero-shot probs (concept):', sim.detach().numpy())"),
    md("## GeoSpatial: Text-Prompted Detection\n\n**Use case:** Find ponds without training detector — prompt \"water body\" / \"aquaculture pond\".\n\n**Limitations:** CLIP trained on web images, not multispectral satellite; domain gap significant. Fine-tune or use RemoteCLIP/GeoCLIP for RS."),
    footer("CLIP aligns images and text — enables zero-shot with natural language prompts.", "12_SAM.ipynb"),
])

register("12_SAM.ipynb", [
    hdr("12", "SAM (Segment Anything)", "2.5 hrs",
        "1. Understand SAM architecture (ViT encoder + prompt encoder + mask decoder)\n2. Use point/box prompts for segmentation\n3. Apply SAM to rapid GeoSpatial labeling\n4. Compare SAM vs supervised UNet++"),
    md("## SAM (Kirillov et al., 2023)\n\n**Promptable segmentation foundation model:**\n\n1. **Image encoder:** ViT-H (heavy, run once per image)\n2. **Prompt encoder:** Points, boxes, or coarse masks\n3. **Mask decoder:** Lightweight transformer → 3 mask candidates + IoU scores\n\n**Training:** 1.1B masks on 11M images — segmentation foundation model.\n\n**Zero-shot:** Works on new domains without fine-tuning (with varying quality)."),
    SETUP,
    code("try:\n    from segment_anything import sam_model_registry\n    print('segment-anything available')\nexcept ImportError:\n    print('Optional: pip install segment-anything')\n\n# SAM workflow (conceptual)\nprint('Workflow: encode image once -> prompt with points/box -> decode mask in ~50ms')"),
    md("## GeoSpatial Labeling Workflow\n\n**Accelerate annotation:**\n1. Click pond centers in QGIS/Label Studio\n2. SAM generates polygon masks\n3. Human corrects boundaries\n4. Export for UNet++ training\n\n**vs Your UNet++:** SAM for labeling; UNet++ for production inference at scale on 6-band Planet data."),
    footer("SAM = foundation model for promptable segmentation; excellent for labeling.", "13_GroundingDINO.ipynb"),
])

register("13_GroundingDINO.ipynb", [
    hdr("13", "Grounding DINO & Module Capstone", "2.5 hrs",
        "1. Understand open-vocabulary detection\n2. Connect vision-language to detection\n3. Design transformer stack for GeoSpatial AI\n4. Complete Module 10"),
    md("## Grounding DINO (Liu et al., 2023)\n\n**Open-vocabulary object detection:** Detect objects described by **text** — no fixed class list.\n\n**Combines:** DINO (detection transformer) + language model for text embeddings\n\n**Input:** Image + \"aquaculture pond . building . road\"\n\n**Output:** Boxes for each mentioned category — zero-shot on new classes."),
    SETUP,
    md("## Module 10: Transformer Stack for GeoSpatial AI\n\n```\n                    YOUR PRODUCTION STACK\n                    ─────────────────────\nTraining labels:     SAM (rapid annotation)\nPretraining:         DINO (unlabeled satellite)\nSegmentation:        UNet++ (water-bodies) OR SegFormer (land cover)\nDetection:           YOLOv8 (Module 08) OR Grounding DINO (zero-shot)\nPrompting:           CLIP / Grounding DINO (exploration)\nInstance/Panoptic:   Mask2Former (Module 09)\n```"),
    md("## When to Use Transformers in Production\n\n| Use Case | Recommendation |\n|----------|---------------|\n| Aquaculture pond boundaries | UNet++ (CNN) — proven |\n| Rapid prototype new class | SAM + CLIP prompts |\n| Unlabeled pretraining | DINO on satellite tiles |\n| Multi-class land cover | SegFormer |\n| Open-vocab exploration | Grounding DINO |\n| Real-time detection | YOLOv8 (not transformer) |"),
    md("## Module 10 Assignment\n\nImplement scaled dot-product attention in NumPy, MultiHeadAttention in PyTorch, and ViT patch embedding. Optional: zero-shot CLIP or SAM demo on one image.\n\nSee `exercises/README.md`."),
    code("# YOUR CODE HERE — transformer stack decision for your next project\n"),
    md("## Module 10 Complete\n\n**Next:** Module 11 Production ML — deployment, monitoring, MLOps."),
    footer("Transformers dominate vision; choose CNN vs transformer based on data, latency, and task.", None),
])


def main():
    print("Building Module 10 notebooks...")
    for name, cells in NOTEBOOKS:
        save(name, cells)
    print(f"Done: {len(NOTEBOOKS)} notebooks created.")


if __name__ == "__main__":
    main()
