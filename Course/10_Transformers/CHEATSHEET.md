# Module 10 Cheat Sheet — Transformers

## Scaled Dot-Product Attention

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| Symbol | Shape | Meaning |
|--------|-------|---------|
| Q | (seq, d_k) | Queries |
| K | (seq, d_k) | Keys |
| V | (seq, d_v) | Values |
| Weights | (seq, seq) | Attention map |

**Scale by √d_k:** Prevents softmax saturation when d_k is large.

## Self-Attention

$$Q = XW_Q,\; K = XW_K,\; V = XW_V$$

**Complexity:** O(n² · d) — quadratic in sequence length.

## Multi-Head Attention

$$\text{MultiHead} = \text{Concat}(\text{head}_1,...,\text{head}_h)W_O$$

- Split d_model into h heads of size d_k = d_model / h
- Each head learns different relationships

## Transformer Encoder Block

```
x → MultiHeadAttention → Add & Norm → FFN → Add & Norm → out
```

**FFN:** Linear → GELU → Linear (typically 4× expansion)

## Positional Encoding

**Sinusoidal:**
$$PE_{(pos,2i)} = \sin(pos/10000^{2i/d})$$
$$PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d})$$

**Vision:** Learned (ViT), relative bias (Swin)

## Vision Architectures

| Model | Key Idea | Use Case |
|-------|----------|----------|
| **ViT** | Patches as tokens | Classification |
| **Swin** | Shifted window attention | Efficient dense pred |
| **SegFormer** | MiT + MLP decoder | Segmentation |
| **Mask2Former** | Query → mask | Universal seg |
| **DINO** | Self-supervised ViT | Pretrain on unlabeled RS |
| **CLIP** | Image-text contrastive | Zero-shot with text |
| **SAM** | Promptable foundation | Rapid labeling |
| **Grounding DINO** | Text → detection | Open-vocab boxes |

## ViT Pipeline

```
Image 224×224
  → Patch embed 16×16 → 196 tokens
  → + [CLS] + positional embed
  → Transformer encoder × L
  → [CLS] → classifier
```

## PyTorch: Multi-Head Attention

```python
mha = nn.MultiheadAttention(d_model=512, num_heads=8, batch_first=True)
out, weights = mha(x, x, x)  # self-attention
```

## PyTorch: ViT Patch Embed

```python
nn.Conv2d(3, d_model, kernel_size=16, stride=16)  # patchify
# flatten → (B, n_patches, d_model)
```

## CNN vs Transformer

| | CNN | Transformer |
|---|-----|-------------|
| Inductive bias | Locality, translation | None |
| Global context | Deep layers | Every layer |
| Compute | O(n) per layer (conv) | O(n²) attention |
| Data efficiency | Better with small data | Needs pretrain/big data |
| GeoSpatial prod | UNet++, YOLO | SegFormer, SAM for labels |

## Your GeoSpatial Stack

| Stage | Tool |
|-------|------|
| Labeling | SAM (point prompts) |
| Pretrain | DINO (unlabeled tiles) |
| Segmentation prod | UNet++ (water-bodies) |
| Land cover alt | SegFormer |
| Detection prod | YOLOv8 |
| Zero-shot explore | CLIP, Grounding DINO |

## Common Mistakes

- Forgetting positional encoding (permutation invariant bug)
- Wrong tensor layout (batch_first vs seq_first)
- Using ViT on tiny datasets without pretrain
- Applying CLIP zero-shot to multispectral without domain adaptation
- Using SAM for production batch inference (slow ViT-H encoder)

## Interview Highlights

1. Derive attention from Q,K,V
2. Why scale by √d_k?
3. ViT patch embedding
4. Swin vs ViT complexity
5. CLIP contrastive loss intuition
