# GPT-2 Style Decoder Pretraining from Scratch

This repository contains the implementation of a **GPT-2 style decoder-only Transformer language model** trained completely from scratch using the **next-token prediction objective** in PyTorch. The model is trained using **Distributed Data Parallel (DDP)** across **2 × 80GB GPUs** with support for **BF16 mixed precision training**.

The implementation incorporates several modern architectural improvements inspired by contemporary LLM designs such as:

* Rotary Positional Embeddings (RoPE)
* Grouped Query Attention (GQA)
* RMSNorm
* SwiGLU Feed Forward Networks
* Weight Tying
* Gradient Accumulation
* Cosine Learning Rate Scheduling with Warmup
* Distributed Multi-GPU Training

---

## Features

* GPT-style decoder-only Transformer architecture
* Autoregressive next-token prediction training
* Rotary Positional Embeddings (RoPE)
* Grouped Query Attention (GQA)
* RMSNorm normalization
* SwiGLU feed-forward layers
* BF16 mixed precision training
* Multi-GPU training using PyTorch DDP
* Gradient clipping and accumulation
* Cosine decay scheduler with warmup
* Periodic checkpoint saving

---

## Model Configuration

| Component       | Value |
| --------------- | ----- |
| Hidden Size     | 512   |
| Decoder Layers  | 16    |
| Attention Heads | 8     |
| KV Heads (GQA)  | 4     |
| Sequence Length | 128   |
| Vocabulary Size | 32000 |
| Dropout         | 0.1   |

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

Training data is expected as tokenized sequences stored in a NumPy file:

```python id="6vsl4m"
hindi_chunksGP.npy
```

Each sample should contain token IDs of fixed sequence length:

```python id="6hcx6o"
[List[int]]  # length = 128
```

---

## Running the Training

Example DDP launch:

```bash id="jex5vs"
torchrun --nproc_per_node=2 train.py
```

---

## Checkpoints

Model checkpoints are periodically saved during training:

```bash id="7j13jy"
checkpointsGPT/
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

* `GPT2Dataset` → Dataset wrapper
* `RotaryEmbedding` → RoPE positional encoding
* `GroupedQueryAttention` → GQA attention mechanism
* `SwiGLUMLP` → Feed-forward network
* `TransformerDecoderLayer` → Decoder block
* `GPT2DecoderModel` → Full decoder-only model
* Training loop with DDP + BF16

---

## Technologies Used

* Python
* PyTorch
* CUDA / NCCL
* Distributed Data Parallel (DDP)
* BF16 Mixed Precision

---

## Notes

* The implementation is intended for educational and research purposes.
* The architecture follows modern decoder design principles inspired by recent large language models.
* Native PyTorch Scaled Dot Product Attention (SDPA) is utilized when available for improved efficiency.
* The model is trained using autoregressive causal language modeling for next-token prediction.
