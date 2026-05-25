import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset, DataLoader

import pickle
import os
from torch.amp import autocast # Core import for mixed precision
import numpy as np
import contextlib

# from torch.cuda.amp import autocast
tokenizer = ByteLevelBPETokenizer(
    vocab="hindi_bpe_tokenizerGP/vocab.json",
    merges="hindi_bpe_tokenizerGP/merges.txt"
)

PAD_ID = tokenizer.token_to_id("<pad>")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# Dataset 
# ---------------------------------------------------------

class GPT2Dataset(Dataset):
    def __init__(self, npy_file):
        self.data = np.load(npy_file)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # Convert directly to PyTorch tensor
        return torch.tensor(self.data[idx], dtype=torch.long)   


# ---------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------
    
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original data type (e.g., bfloat16)
        orig_dtype = x.dtype
        
        # Compute RMS in float32 to prevent underflow/overflow
        x_f32 = x.float()
        rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        # Scale and cast back to the original type before multiplying by weight
        # return (x_f32 * rms).to(orig_dtype) * self.weight
        return (x_f32 * rms).to(orig_dtype) * self.weight.to(orig_dtype)


# ---------------------------------------------------------
# RoPE
# ---------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires head_dim to be even")

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Notice we divide by head_dim entirely to handle the half-dimension indexing
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)

        # Standard RoPE replicates the frequencies for the split halves
        # Transforming (seq_len, head_dim // 2) -> (seq_len, head_dim)
        full_freqs = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos", full_freqs.cos(), persistent=False)
        self.register_buffer("sin", full_freqs.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)

        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        # Gather buffers and match dimensions: (1, 1, seq_len, head_dim)
        cos = self.cos[:seq_len].to(dtype=q.dtype, device=q.device)[None, None, :, :]
        sin = self.sin[:seq_len].to(dtype=q.dtype, device=q.device)[None, None, :, :]

        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        return q, k


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Splits the tensor in half along the last dimension and rotates it."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x shape: (batch, n_heads, seq_len, head_dim)
    cos/sin shape: (1, 1, seq_len, head_dim)
    """
    # Standard RoPE formula: R(x) = x * cos + rotate_half(x) * sin
    return (x * cos) + (rotate_half(x) * sin)


# ---------------------------------------------------------
# Grouped Query Attention
# ---------------------------------------------------------

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expands key/value heads to match query heads for standard attention calculations.
    Input: (B, n_kv_heads, T, head_dim)
    Output: (B, n_kv_heads * n_rep, T, head_dim)
    """
    if n_rep == 1:
        return x
    B, n_kv_heads, T, head_dim = x.shape
    # Expand along a new dimension to replicate head patterns cleanly
    x = x[:, :, None, :, :].expand(B, n_kv_heads, n_rep, T, head_dim)
    return x.reshape(B, n_kv_heads * n_rep, T, head_dim)


class GroupedQueryAttention(nn.Module):
    """
    GPT-style Grouped Query Attention aligned with custom RotaryEmbedding modules.
    Input: x: (batch, seq_len, n_embd)
    Output: y: (batch, seq_len, n_embd)
    """

    def __init__(self, n_embd: int, n_query_heads: int, n_kv_heads: int, block_size: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()

        if n_embd % n_query_heads != 0:
            raise ValueError("n_embd must be divisible by n_query_heads")
        if n_query_heads % n_kv_heads != 0:
            raise ValueError("n_query_heads must be divisible by n_kv_heads")

        self.n_embd = n_embd
        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = n_embd // n_query_heads
        self.n_rep = n_query_heads // n_kv_heads
        self.dropout = dropout

        # Projections
        self.q_proj = nn.Linear(n_embd, n_query_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(n_embd, n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(n_embd, n_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # self.out_proj = nn.Linear(n_embd, n_embd, bias=bias) #new
        self.out_proj._is_residual_proj = True #new

        # Dropouts
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Manual attention fallback causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, rope_embedder: nn.Module, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to separate heads: (B, T, H, D)
        q = q.view(B, T, self.n_query_heads, self.head_dim)
        k = k.view(B, T, self.n_kv_heads, self.head_dim)
        v = v.view(B, T, self.n_kv_heads, self.head_dim)

        # Transpose to conventional PyTorch format: (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE directly before calculating attention weights
        q, k = rope_embedder(q, k)

        # Compute Grouped Query Attention
        y = self._torch_attention(q, k, v, attention_mask)

        # Restore sequence shape: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final linear mapping and projection dropout
        return self.resid_dropout(self.out_proj(y))

    def _torch_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        T = q.size(-2)

        # Repeat key/value heads across grouped query configurations
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Use PyTorch Native SDPA whenever available
        # if hasattr(F, "scaled_dot_product_attention"):
        #     # If a custom attention padding mask isn't provided, use native causal layout
        #     is_causal = True if attention_mask is None else False
            
        #     y = F.scaled_dot_product_attention(
        #         q,
        #         k,
        #         v,
        #         attn_mask=attention_mask,
        #         dropout_p=self.dropout if self.training else 0.0,
        #         is_causal=is_causal,
        #     )
        if hasattr(F, "scaled_dot_product_attention"): #new
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True if attention_mask is None else False,
            )
        else:
            # Fallback manual calculation loop
            att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            
            # Apply causal layout boundary checks
            causal_mask = self.causal_mask[:, :, :T, :T]
            att = att.masked_fill(causal_mask == 0, float("-inf"))
            
            # Incorporate optional outside padding rules if provided
            if attention_mask is not None:
                att = att.masked_fill(attention_mask == 0, float("-inf"))

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        return y


# Initialization example
# rope = RotaryEmbedding(head_dim=64)
# attn = GroupedQueryAttention(n_embd=768, n_query_heads=12, n_kv_heads=3, block_size=2048)

# Forward call execution pattern
# output = attn(x, rope_embedder=rope)


# --------------------------------------------------------
# Transformer Decoder Layer
# --------------------------------------------------------

class SwiGLUMLP(nn.Module):
    """
    SwiGLU Feed-Forward Network. 
    Standard in modern decoders (LLaMA, Mistral) that use RMSNorm and RoPE.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        # SwiGLU splits the input processing into a gate and a value path
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        # self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.down_proj._is_residual_proj = True #new
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # silu(gate) * up_proj
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerDecoderLayer(nn.Module):
    """
    Modern decoder block using:
    - RMSNorm instead of LayerNorm
    - Grouped Query Attention (GQA)
    - SwiGLU MLP
    - Pre-norm residual structure
    """

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, block_size: int, dropout: float = 0.0):
        super().__init__()

        # Aligned with your specific GroupedQueryAttention naming convention
        self.attention = GroupedQueryAttention(
            n_embd=hidden_size,
            n_query_heads=num_heads,
            n_kv_heads=num_kv_heads,
            block_size=block_size,
            dropout=dropout,
        )

        self.attn_norm = RMSNorm(hidden_size)

        # Standard practice for SwiGLU is an intermediate size of ~ 8/3 * hidden_size
        # adjusted to the nearest multiple of 256 for optimal hardware alignment.
        intermediate_size = int(2 * (hidden_size * 4 / 3))
        intermediate_size = ((intermediate_size + 255) // 256) * 256

        self.mlp = SwiGLUMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )

        self.mlp_norm = RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor, rope_embedder: nn.Module, attention_mask: torch.Tensor | None = None ) -> torch.Tensor:
        # Self-attention block
        residual = x
        x = self.attn_norm(x)
        x = self.attention(
            x=x,
            rope_embedder=rope_embedder,  # Crucial: Passed down to GQA
            attention_mask=attention_mask,
        )
        x = residual + x

        # MLP block
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + x

        return x


# --------------------------------------------------------
# Prediction Head
# --------------------------------------------------------

class PredictionHead(nn.Module):
    """
    Final language modeling block for a modern pre-norm decoder.
    Includes final RMSNorm and supports weight tying.
    """

    def __init__(self, hidden_size: int, vocab_size: int, bias: bool = False):
        super().__init__()

        # 1. CRUCIAL: Final normalization before mapping to vocabulary logits
        self.final_norm = RMSNorm(hidden_size)

        # 2. Output linear layer mapping hidden states to vocab size
        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=bias,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states shape: (batch, seq_len, hidden_size)
        
        # Apply the final missing normalization layer
        normed_states = self.final_norm(hidden_states)
        
        # Compute vocabulary logits
        logits = self.lm_head(normed_states)
        return logits

# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------

class GPT2DecoderModel(nn.Module):
    """
    Modern decoder-only language model for next-token prediction.
    Assembled with custom RMSNorm, GQA, SwiGLU, and RoPE components.
    """

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, num_heads: int, num_kv_heads: int, max_seq_len: int, dropout: float = 0.0,
        tie_embeddings: bool = True, pad_token_id: int | None = None
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        # Token embedding layer
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # 1. Instantiate the single, shared RoPE module here
        # Assuming your head_dim = hidden_size // num_heads
        head_dim = hidden_size // num_heads
        self.rope_embedder = RotaryEmbedding(head_dim=head_dim, max_seq_len=max_seq_len)

        # 2. FIXED: Added block_size argument pass-through
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    block_size=max_seq_len,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # REMOVED: self.final_norm = RMSNorm(hidden_size) 
        # (It is handled inside PredictionHead to avoid double-normalization)

        self.prediction_head = PredictionHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            bias=False,
        )

        self.apply(self._init_weights)

        if tie_embeddings:
            self.tie_weights()

        # if tie_embeddings:
        #     self.tie_weights()


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02

            # Residual projection scaling, used for stability in deep decoder stacks.
            if getattr(module, "_is_residual_proj", False):
                std = std / math.sqrt(2 * self.num_layers)

            nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def tie_weights(self):
        """Ties input embedding weights to output prediction weights."""
        self.prediction_head.lm_head.weight = self.token_embeddings.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None):
        B, T = input_ids.shape

        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}")

        # Compute token embeddings
        x = self.token_embeddings(input_ids)
        x = self.dropout(x)

        # 3. FIXED: Shape 2D padding mask to 4D boolean mask for PyTorch native SDPA compatibility
        # (batch, seq_len) -> (batch, 1, 1, seq_len)
        # ---final_attn_mask = None
        # ---if attention_mask is not None:
            # Convert 1s to True (keep) and 0s to False (mask out)
            #--- final_attn_mask = attention_mask.view(B, 1, 1, T).bool()

        final_attn_mask = None

        if attention_mask is not None:
            # attention_mask: (B, T), 1 = valid token, 0 = pad
            padding_mask = attention_mask[:, None, None, :].bool()  # (B, 1, 1, T)

            causal_mask = torch.tril(
                torch.ones(T, T, device=input_ids.device, dtype=torch.bool)
            )[None, None, :, :]  # (1, 1, T, T)

            # True means allowed attention
            final_attn_mask = padding_mask & causal_mask  # (B, 1, T, T)
        else:
            final_attn_mask = None

        # Pass through decoder layers
        for layer in self.layers:
            # 4. FIXED: Passed down the self.rope_embedder instance
            x = layer(
                x,
                rope_embedder=self.rope_embedder,
                attention_mask=final_attn_mask,
            )

        # Map hidden states directly to vocab logits (includes internal final RMSNorm)
        logits = self.prediction_head(x)

        if labels is not None and self.pad_token_id is not None: #new
            labels = labels.masked_fill(labels == self.pad_token_id, -100)

        # if labels is None:
        #     return logits

        # # Next-token prediction loss routing
        # shift_logits = logits[:, :-1, :].contiguous()
        # shift_labels = labels[:, 1:].contiguous()

        # loss = F.cross_entropy(
        #     shift_logits.view(-1, self.vocab_size),
        #     shift_labels.view(-1),
        #     ignore_index=-100,
        # )
        if labels is None: #new
            return logits

        if self.pad_token_id is not None:
            labels = labels.masked_fill(labels == self.pad_token_id, -100)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        return loss, logits
    

# --------------------------------------------------------
# PyTorch Distributed Data Parallel (DDP)
# --------------------------------------------------------

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# --------------------------------------------------------
# Distributed Initialization
# --------------------------------------------------------
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# --------------------------------------------------------
# Model Configuration & Initialization
# --------------------------------------------------------
# Adjusting max_seq_len to match your token chunk testing sizes
max_seq_len = 128  #1024

model = GPT2DecoderModel(
    vocab_size=32000,
    hidden_size=512,
    num_layers=16,
    num_heads=8,
    num_kv_heads=4,
    max_seq_len=max_seq_len,
    dropout=0.1,
    pad_token_id=PAD_ID #tokenizer.token_to_id("<s>"), 0
)

# CRUCIAL FIX: Move model to device and wrap with DDP BEFORE executing forward loops
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])



# --------------------------------------------------------
# Sanity Check Forward Pass
# --------------------------------------------------------
# Match input sizes to maximum configured context boundaries
input_ids = torch.randint(0, 32000, (8, max_seq_len), device=local_rank)

with torch.no_grad():
    loss, logits = model(input_ids=input_ids, labels=input_ids)

if local_rank == 0:
    print(f"Initial Test Loss: {loss.item():.4f}")
    print(f"Logits Matrix Output Shape: {logits.shape}")  # (8, 1024, 32000)


# --------------------------------------------------------
# Hyperparameters & Optimization Schedule
# --------------------------------------------------------
def build_optimizer(model, lr=3e-4, weight_decay=0.1):
    decay_params = []
    no_decay_params = []
    seen_params = set() # Track unique memory pointers to handle weight-tied layers safely

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Deduplicate tied embedding matrices 
        if id(param) in seen_params:
            continue
        seen_params.add(id(param))

        # Filter out biases and normalization scales
        if (
            name.endswith(".bias")
            or "norm" in name.lower()
            or "rmsnorm" in name.lower()
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    return optimizer


def build_cosine_warmup_scheduler(optimizer, total_training_steps, warmup_ratio=0.03, min_lr_ratio=0.1):
    warmup_steps = int(total_training_steps * warmup_ratio)
    warmup_steps = max(1, warmup_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        progress = float(current_step - warmup_steps) / float(
            max(1, total_training_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------
# Data Loading Setup
# --------------------------------------------------------
save_path = "hindi_chunksGP.npy"
# Fix: np.load() reads the file; np.save() writes it
chunks = np.load(save_path) 

train_dataset = GPT2Dataset(save_path) #chunks



# --------------------------------------------------------
# Training Setup & Data Loaders
# --------------------------------------------------------
from torch.utils.data.distributed import DistributedSampler

num_epochs = 5
micro_batch_size_per_gpu = 8
num_gpus = 2
gradient_accumulation_steps = 16

global_batch_size = micro_batch_size_per_gpu * num_gpus * gradient_accumulation_steps

if local_rank == 0:
    print("Global batch size:", global_batch_size)

num_samples = len(train_dataset)
steps_per_epoch = math.ceil(num_samples / global_batch_size)
total_training_steps = steps_per_epoch * num_epochs

peak_lr = 3e-4
warmup_ratio = 0.03

# Initialize variables missing from original context snippet
device = torch.device(f"cuda:{local_rank}")
device_type = "cuda"

sampler = DistributedSampler(
    train_dataset, 
    num_replicas=num_gpus, 
    rank=local_rank, 
    shuffle=True
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=micro_batch_size_per_gpu, 
    num_workers=4, 
    pin_memory=True, 
    sampler=sampler
)

optimizer = build_optimizer(model, lr=peak_lr, weight_decay=0.1)
scheduler = build_cosine_warmup_scheduler(
    optimizer,
    total_training_steps=total_training_steps,
    warmup_ratio=warmup_ratio,
    min_lr_ratio=0.1,
)

# --------------------------------------------------------
# Checkpoint Helper Function (DDP Gated)
# --------------------------------------------------------
def save_checkpoint(model, optimizer, scheduler, epoch, global_step, loss, checkpoint_dir="checkpointsGPT"):
    # ONLY Rank 0 should touch the file system
    if local_rank != 0:
        return
        
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pt")
    
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        # Unpack the underlying model weights from the DDP wrapper container
        "model_state_dict": model.module.state_dict(), 
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"\n[CHECKPOINT] Saved checkpoint to {checkpoint_path}\n")



# --------------------------------------------------------
# Enhanced Training Loop
# --------------------------------------------------------
model.train()
optimizer.zero_grad(set_to_none=True)
global_step = 0

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)

    for micro_step, batch in enumerate(train_loader):
        # input_ids = batch["input_ids"].to(device, non_blocking=True)
        # attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        # labels = batch["labels"].to(device, non_blocking=True)

        input_ids = batch.to(device, non_blocking=True)
        labels = input_ids
        attention_mask = torch.ones_like(input_ids)

        should_step = (
            (micro_step + 1) % gradient_accumulation_steps == 0
            or (micro_step + 1) == len(train_loader)
        )

        # PERFORMANCE BOOST: Turn off DDP communication overhead during accumulation
        # context manager defaults to sync state automatically on final step pass
        context_manager = model.no_sync() if not should_step else contextlib.nullcontext()

        with context_manager:
            # INTEGRATION STEP 1: Mixed Precision Autocast Block
            with autocast(device_type=device_type, dtype=torch.bfloat16):
                # FIXED: Pass labels directly to let our model compute next-token loss internally
                loss, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Scale loss down to match the accumulation step metrics
                loss = loss / gradient_accumulation_steps

            # Backward pass runs inside context bounds safely
            loss.backward()

        if should_step:
            # INTEGRATION STEP 2: Gradient Clipping Execution
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            
            # Print performance tracks exclusively on Rank 0
            if local_rank == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"epoch={epoch + 1} | "
                    f"step={global_step}/{total_training_steps} | "
                    f"loss={loss.item() * gradient_accumulation_steps:.4f} | "
                    f"lr={current_lr:.6e}"
                )
            
            # INTEGRATION STEP 3: Checkpointing Cadence Check
            if global_step % 5000 == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1, 
                    global_step, loss.item() * gradient_accumulation_steps
                )

    # Save epoch wrap up state cleanly
    save_checkpoint(
        model, optimizer, scheduler, epoch + 1, 
        global_step, loss.item() * gradient_accumulation_steps
    )









