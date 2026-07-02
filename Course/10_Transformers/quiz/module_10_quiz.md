# Module 10 Quiz

**Passing score:** 16/20 (80%)

---

**Q1.** Attention computes:
- (a) Fixed convolution
- (b) Weighted sum of values based on query-key similarity
- (c) Max pooling
- (d) Batch normalization

**Q2.** Scaling dot products by √d_k prevents:
- (a) Overfitting
- (b) Softmax saturation when d_k is large
- (c) Underfitting
- (d) Data leakage

**Q3.** Self-attention means:
- (a) Q, K, V from different sequences
- (b) Q, K, V derived from same input
- (c) No learned weights
- (d) Only one head

**Q4.** Multi-head attention purpose:
- (a) Reduce parameters
- (b) Learn different attention patterns in parallel
- (c) Remove positional info
- (d) Replace FFN

**Q5.** Transformers need positional encoding because:
- (a) Attention is permutation invariant
- (b) Attention is too slow
- (c) GPUs require it
- (d) Softmax fails without it

**Q6.** ViT treats images as:
- (a) Single pixel
- (b) Sequence of patch tokens
- (c) Graph nodes only
- (d) RNN states

**Q7.** ViT [CLS] token is used for:
- (a) Positional encoding
- (b) Global image representation for classification
- (c) Mask decoding
- (d) Data augmentation

**Q8.** Swin Transformer improves ViT with:
- (a) Larger patches only
- (b) Shifted window attention + hierarchy
- (c) No attention
- (d) RNN layers

**Q9.** SegFormer decoder is:
- (a) Heavy UNet
- (b) Lightweight all-MLP
- (c) RNN
- (d) GAN

**Q10.** Mask2Former uses:
- (a) Per-pixel softmax only
- (b) Query-based mask classification
- (c) R-CNN RoI pooling
- (d) K-means

**Q11.** DINO is:
- (a) Supervised ImageNet only
- (b) Self-supervised ViT pretraining via distillation
- (c) Object detector
- (d) GAN

**Q12.** CLIP trains with:
- (a) Image-image contrastive loss
- (b) Image-text contrastive loss
- (c) MSE reconstruction
- (d) GAN loss

**Q13.** SAM accepts prompts:
- (a) Text only
- (b) Points, boxes, or coarse masks
- (c) Class labels only
- (d) GPS coordinates only

**Q14.** Grounding DINO enables:
- (a) Fixed 80 COCO classes only
- (b) Open-vocabulary detection from text
- (c) Segmentation only
- (d) Unsupervised clustering

**Q15.** Attention complexity in sequence length n:
- (a) O(n)
- (b) O(n log n)
- (c) O(n²)
- (d) O(1)

**Q16.** Transformer encoder block contains:
- (a) MHA + FFN with residual connections
- (b) Conv + pool only
- (c) LSTM only
- (d) Decision tree

**Q17.** For aquaculture pond production segmentation you recommended:
- (a) SAM inference at scale
- (b) UNet++ CNN pipeline (water-bodies)
- (c) GPT-4 only
- (d) Linear regression

**Q18.** SAM best use in your workflow:
- (a) Production batch inference on Planet tiles
- (b) Rapid annotation / labeling acceleration
- (c) Replacing all training data
- (d) mAP evaluation

**Q19.** Causal mask used in:
- (a) ViT encoder
- (b) Autoregressive decoder (GPT-style)
- (c) SegFormer
- (d) NMS

**Q20.** Standard CLIP on multispectral satellite:
- (a) Works perfectly zero-shot
- (b) Domain gap — needs RS-specific models or fine-tuning
- (c) Cannot process images
- (d) Requires 12-band input natively

---

See [solutions/quiz_answers.md](solutions/quiz_answers.md) after submitting.
