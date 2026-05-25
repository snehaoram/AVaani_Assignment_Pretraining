# BERT Pretraining from Scratch using MLM

This repository contains the implementation of a **BERT-like Transformer Encoder model** trained completely from scratch using the **Masked Language Modeling (MLM)** objective in PyTorch. The model is trained using **Distributed Data Parallel (DDP)** across **2 × 80GB GPUs** with support for **mixed precision BF16 training**.

The implementation includes several modern architectural and optimization components such as:

* Rotary Positional Embeddings (RoPE)
* Grouped Query Attention (GQA)
* RMSNorm
* Dynamic Masking
* Gradient Accumulation
* Cosine Learning Rate Scheduler with Warmup
* Weight Tying
* Distributed Training with NCCL backend

---

## Features

* BERT-style encoder-only Transformer architecture
* MLM pretraining objective with runtime dynamic masking
* RoPE-based positional encoding
* GQA attention implementation
* RMSNorm normalization
* BF16 mixed precision training using `torch.autocast`
* Multi-GPU training with PyTorch DDP
* Gradient clipping and accumulation
* Cosine decay scheduler with warmup
* Periodic checkpoint saving

---

## Model Configuration

| Component       | Value |
| --------------- | ----- |
| Hidden Size     | 512   |
| Encoder Layers  | 16    |
| Attention Heads | 8     |
| KV Heads (GQA)  | 2     |
| Sequence Length | 128   |
| Vocabulary Size | 32000 |
| MLM Mask Ratio  | 40%   |

---

## Training Setup

| Setting                     | Value                 |
| --------------------------- | --------------------- |
| GPUs                        | 2 × 80GB              |
| Micro Batch Size / GPU      | 8                     |
| Gradient Accumulation       | 16                    |
| Effective Global Batch Size | 256                   |
| Optimizer                   | AdamW                 |
| Learning Rate               | 3e-4                  |
| Scheduler                   | Cosine Decay + Warmup |
| Precision                   | BF16                  |

---

## Dataset Format

The training data is expected as tokenized chunks stored in a pickle file:

```python
hindi_chunks.pkl
```

Each chunk should contain token IDs of fixed sequence length:

```python
[List[int]]  # length = 128
```

---

## Running the Training

Example DDP launch:

```bash
torchrun --nproc_per_node=2 train.py
```

---

## Checkpoints

Model checkpoints are periodically saved during training:

```bash
checkpointsBPT/
```

Each checkpoint contains:

* Model state
* Optimizer state
* Scheduler state
* Current epoch
* Global step
* Training loss

---

## Main Components

* `TokenizedChunksDataset` → Dataset wrapper
* `DataCollatorForMLM` → Dynamic MLM masking
* `RMSNorm` → Stable normalization
* `GQAAttention` → Grouped Query Attention
* `TransformerEncoderLayer` → Encoder block
* `ModernBERT` → Full encoder model
* Training loop with DDP + BF16

---

## Key Tools Used

* Python
* PyTorch
* Distributed Data Parallel (DDP)
* CUDA / NCCL
* BF16 Mixed Precision

---

## Notes

* The architecture is inspired by modern Transformer optimizations while retaining the encoder-only MLM training paradigm.
* Flash Attention hooks are included for future integration.
