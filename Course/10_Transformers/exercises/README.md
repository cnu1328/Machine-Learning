# Module 10 Exercises

Attempt before checking [solutions/](solutions/).

---

## Attention Foundations (Notebooks 01–05)

### Exercise 1 — Attention from Scratch (Notebook 02)
Implement scaled dot-product attention in NumPy. Verify output shape and that weights sum to 1 per row.

### Exercise 2 — Causal Mask (Notebook 02)
Apply causal mask to 8-token sequence. Visualize attention weights — confirm no token attends to future.

### Exercise 3 — Multi-Head Attention (Notebook 04)
Implement `MultiHeadAttention` in PyTorch. Confirm output shape matches input (batch, seq, d_model).

### Exercise 4 — Transformer Encoder Block (Notebook 04)
Stack 4 encoder blocks. Pass random (2, 16, 128) tensor through. Count parameters.

### Exercise 5 — Positional Encoding (Notebook 05)
Plot sinusoidal PE for seq_len=128, d_model=64. Show that PE[pos] · PE[pos+k] depends only on k (near-constant for fixed k).

---

## Vision Transformers (Notebooks 06–13)

### Exercise 6 — ViT Patch Embedding (Notebook 06)
Implement `PatchEmbed` for 256×256 image, patch 16, d_model=384. Print number of patches and output shape.

### Exercise 7 — ViT vs CNN Params (Notebook 06)
Compare parameter count: small CNN (3 conv layers) vs ViT-Tiny on same input. Which has more params?

### Exercise 8 — SegFormer Decision (Notebook 08)
Write decision matrix: when would you choose SegFormer over UNet++ for a new GeoSpatial project?

### Exercise 9 — SAM Labeling Workflow (Notebook 12)
Document a 5-step workflow using SAM to accelerate pond polygon annotation for water-bodies training.

---

## Module Assignment: Transformer Components

**Deliverable:** `exercises/assignment_transformers.ipynb`

Implement and demonstrate:

1. **NumPy:** `scaled_dot_product_attention(Q, K, V, mask=None)`
2. **PyTorch:** `MultiHeadAttention(d_model, num_heads)` with forward pass
3. **PyTorch:** `TransformerEncoderBlock` (MHA + FFN + LayerNorm + residuals)
4. **PyTorch:** `PatchEmbed` for ViT-style patchification
5. **Demo:** Run a 4-layer mini-transformer on random sequence (batch=2, seq=20, d=64)
6. **Visualization:** Plot attention weights heatmap for one head
7. **Optional:** CLIP or SAM zero-shot demo on one image with commentary on GeoSpatial domain gap
8. **Report:** 1-page transformer stack recommendation for your GeoSpatial portfolio

**Targets:**
- All shapes documented in comments
- Attention weights sum to 1.0 (±1e-5)
- Mini-transformer runs without error on CPU

---

## GeoSpatial Extension

Research **RemoteCLIP** or **SatCLIP**. Write 200 words on why standard CLIP fails on multispectral satellite imagery and what domain-specific models fix.

---

## Submission

> Module 10 complete. Assignment attached. Quiz score: X/20.
