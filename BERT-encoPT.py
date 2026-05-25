import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pickle
import os
from torch.amp import autocast # Core import for mixed precision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_path = "hindi_chunks.pkl"

with open(load_path, "rb") as f:
    chunks = pickle.load(f)

print("Chunks loaded successfully.")
print("Total chunks:", len(chunks))

# ---------------------------------------------------------
# Dataset Wrapper
# ---------------------------------------------------------

class TokenizedChunksDataset(Dataset):
    def __init__(self, raw_chunks):
        """
        raw_chunks: List of lists containing token IDs. 
        Example: [[101, 43, 22, ...], [101, 99, 12, ...]]
        Each internal list must have length 128.
        """
        self.chunks = raw_chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        # Return as a tensor of long integers
        return torch.tensor(self.chunks[idx], dtype=torch.long)


# ---------------------------------------------------------
# The Dynamic Masking Collator
# ---------------------------------------------------------

class DataCollatorForMLM:
    def __init__(self, mask_token_id, vocab_size, mask_ratio=0.40, pad_token_id=0):
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.mask_ratio = mask_ratio
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        # Stack individual items into a batch tensor: [Batch_size, 128]
        input_ids = torch.stack(examples, dim=0)
        
        # Setup targets/labels: start with copies of the inputs
        labels = input_ids.clone()
        
        # Create a probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mask_ratio)
        
        # Prevent special tokens from being masked (e.g., [PAD], [CLS])
        # Modify this to include your specific special token IDs if they exist
        special_tokens_mask = (input_ids == self.pad_token_id) 
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Sample which indices to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Everywhere else that isn't masked should be ignored by the loss function (-100)
        labels[~masked_indices] = -100  
        
        # 80% of the selected tokens become the literal [MASK] token id
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id
        
        # 10% of the selected tokens get replaced with a random word from the vocab
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        # The remaining 10% of selected tokens remain unchanged (but are still evaluated in labels)
        
        # Create attention mask (1 for tokens, 0 for padding tokens)
        attention_mask = (input_ids != self.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# ---------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------
    

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # Keep the learnable weight in float32 for high-precision scaling
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x):
        # Save the original data type (e.g., torch.bfloat16) to cast back later
        orig_dtype = x.dtype
        
        # Upcast the input tensor to float32 for stable computation
        x_fp32 = x.to(torch.float32)
        
        # Compute inverse RMS in float32 to avoid underflow/overflow during squaring
        inv_rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        # Perform multiplication in float32, then cast back to the original bfloat16
        output = self.weight * x_fp32 * inv_rms
        
        return output.to(orig_dtype)
    

# --------------------------------------------------------
# RoPE
# --------------------------------------------------------


def build_rope_cache(seq_len, head_dim, device):
    assert head_dim % 2 == 0

    # 1. Correct theta calculation matching the RoPE formula
    # Exponent indices: 0, 2, 4, ..., head_dim - 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    # positions = torch.arange(seq_len, device=device).float()
    positions = torch.arange(seq_len, device=device).float() - (seq_len // 2)

    # freqs shape: [seq_len, head_dim // 2]
    freqs = torch.outer(positions, inv_freq)
    
    # 2. Repeat the frequencies so they map to the full head_dim
    # This makes broadcasting dead simple later
    emb = torch.cat((freqs, freqs), dim=-1) # Shape: [seq_len, head_dim]
    
    return emb.cos(), emb.sin()


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # Split head_dim in half, rotate one half negatively, and swap them
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    """
    x:   [batch, heads, seq_len, head_dim]
    cos: [seq_len, head_dim]
    sin: [seq_len, head_dim]
    """
    # Reshape cos and sin for perfect broadcasting: [1, 1, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    
    # The standard RoPE rotation formula: x * cos + rotate_half(x) * sin
    return (x * cos) + (rotate_half(x) * sin)
    

# --------------------------------------------------------
# GQA with Flash Attention
# --------------------------------------------------------


# try:
#     from flash_attn import flash_attn_func, flash_attn_varlen_func
#     from flash_attn.bert_padding import unpad_input, pad_input
#     FLASH_ATTN_AVAILABLE = True
# except ImportError:
#     FLASH_ATTN_AVAILABLE = False


class GQAAttention(nn.Module):
    def __init__(self, dim, num_q_heads=12, num_kv_heads=4, dropout=0.1):
        super().__init__()

        assert num_q_heads % num_kv_heads == 0
        assert dim % num_q_heads == 0

        self.dim = dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_q_heads
        self.groups = num_q_heads // num_kv_heads
        self.dropout_p = dropout

        self.q_proj = nn.Linear(dim, num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)

        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _is_causal_mask(self, attn_mask):
        if attn_mask is None:
            return False

        Tq, Tk = attn_mask.shape
        if Tq != Tk:
            return False

        causal = torch.triu(
            torch.ones(Tq, Tk, device=attn_mask.device, dtype=torch.bool),
            diagonal=1,
        )

        return torch.equal(attn_mask.bool(), causal)

    def _apply_rope_bthd(self, q, k, T, device):
        """
        q: [B, T, num_q_heads, head_dim]
        k: [B, T, num_kv_heads, head_dim]
        """
        # Assuming build_rope_cache and apply_rope are defined globally
        cos, sin = build_rope_cache(T, self.head_dim, device)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()

        return q, k

    def _manual_attention(self, q, k, v, attn_mask=None, key_padding_mask=None): # key_padding_mask=None
        """
        q: [B, T, num_q_heads, head_dim]
        k: [B, T, num_kv_heads, head_dim]
        v: [B, T, num_kv_heads, head_dim]
        """
        B, T, _, _ = q.shape

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = q.view(B, self.num_kv_heads, self.groups, T, self.head_dim)
        k = k.unsqueeze(2)
        v = v.unsqueeze(2)

        # scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        # Using a fixed numerical value rather than finfo min prevents fp16 underflow problems
        mask_value = -1e9

        if attn_mask is not None:
            # attn_mask = attn_mask.bool()
            mask_value = torch.finfo(scores.dtype).min # Max negative value for your precision (BF16)
            # scores = scores.masked_fill(attn_mask[None, None, None, :, :], mask_value)
            scores = scores.masked_fill(attn_mask[:, None, None, :], mask_value)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.bool()
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, None, :],
                mask_value,
            )

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        out = out.view(B, self.num_q_heads, T, self.head_dim)
        out = out.transpose(1, 2).contiguous()
        # attn_probs = torch.softmax(scores, dim=-1)
        # out = torch.matmul(attn_probs, v)

        return out

    def forward(self, x, key_padding_mask=None, attn_mask=None): #positional_embedding, key_padding_mask=None, , positional_embedding=None
        """
        x:                [B, T, D]
        attn_mask:        [T, T], True/1 means mask out
        key_padding_mask: [B, T], True/1 means padding / mask out
        """
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        q, k = self._apply_rope_bthd(q, k, T, x.device)

        causal = self._is_causal_mask(attn_mask)

        out = self._manual_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )

        out = out.reshape(B, T, D)
        return self.out_proj(out)
    

# --------------------------------------------------------
# Projection Layer and Model
# --------------------------------------------------------


class MLMPredictionHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        # Dense projection layer
        self.dense = nn.Linear(hidden_size, hidden_size)
        # Using RMSNorm as requested
        self.norm = RMSNorm(hidden_size) 
        # Modern activation function (e.g., GELU or Swish)
        self.activation = nn.GELU()
        
        # The final projection back to vocab size
        # self.output = nn.Linear(hidden_size, vocab_size, bias=False)
        # This is what ModernBERT is looking for!
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.norm(x)
        # x = self.output(x) #
        x = self.decoder(x)
        return x


# --------------------------------------------------------
# Transformer Encoder Layer
# --------------------------------------------------------


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__()
        # Initialize your ready-made GQA block
        self.attention = GQAAttention(hidden_size, num_heads, num_kv_heads)
        self.attn_norm = RMSNorm(hidden_size)
        
        # Standard FeedForward / SwiGLU / GELU block
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.mlp_norm = RMSNorm(hidden_size)

    def forward(self, x, rope_cache, attention_mask=None): #positional_embedding=rope_cache, key_padding_mask, , positional_embedding=rope_cache
        # Pre-LN / Pre-RMSNorm structure
        normED_x = self.attn_norm(x)
        # attn_out = self.attention(normED_x, key_padding_mask=None, attn_mask=attention_mask) #attn_mask=None, key_padding_mask=None, rope_cache=attn_mask=None, key_padding_mask=None
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # True means mask out PAD tokens

        key_padding_mask = attention_mask == 0 if attention_mask is not None else None

        attn_out = self.attention(
            normED_x,
            key_padding_mask=key_padding_mask,
            attn_mask=None,
        )
        x = x + attn_out
        
        # MLP Block
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------

class ModernBERT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, num_kv_heads, max_seq_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Token embedding matrix lives here
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # Pre-compute RoPE cache once during initialization
        # build_rope_cache should yield vectors sized for (max_seq_len, head_dim)
        dimen = hidden_size // num_heads
        self.rope_cache = build_rope_cache(seq_len=max_seq_len, head_dim=dimen, device=device) #hidden_size // num_headsseq_len, head_dim, device
        
        # Core Transformer Encoder blocks
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_heads, num_kv_heads) for _ in range(num_layers)
        ])
        
        # Final layer normalization layer before MLM head
        self.final_norm = RMSNorm(hidden_size)
        
        # MLM prediction head
        self.mlm_head = MLMPredictionHead(hidden_size, vocab_size)
        
        # Tie the weights of token embeddings with the output prediction layer
        self.tie_weights()

    def tie_weights(self):
        """ Ties the weights between input embedding and MLM final projection. """
        self.mlm_head.decoder.weight = self.token_embeddings.weight

    def forward(self, input_ids, attention_mask=None):
        # Get sequence length from current batch
        seq_len = input_ids.size(1)
        
        # Lookup dense embeddings
        x = self.token_embeddings(input_ids)
        
        # Fetch and slice RoPE cache dynamically to match the input batch size/length
        # Ensure your apply_rope logic works bidirectionally
        # current_rope_cache = self.rope_cache[:seq_len, :].to(input_ids.device)
        # 1. UNPACK AND SLICE INDIVIDUALLY
        # Assuming self.rope_cache is a tuple: (cos, sin)
        cos_cache, sin_cache = self.rope_cache
        
        # Slice along the sequence length dimension (dim 0) and send to GPU
        current_cos = cos_cache[:seq_len, :].to(input_ids.device)
        current_sin = sin_cache[:seq_len, :].to(input_ids.device)
        
        # Re-package them if your apply_rope expects a tuple, or keep them separate
        current_rope_cache = (current_cos, current_sin)
        
        # Cascade through encoder blocks
        for layer in self.encoder_layers:
            x = layer(x, rope_cache=current_rope_cache, attention_mask=attention_mask)  #attn_out = self.attention(normED_x, rope_cache=rope_cache, attention_mask=attention_mask)
            
        # Final normalizations and projection
        x = self.final_norm(x)
        logits = self.mlm_head(x)
        
        return logits
    

# --------------------------------------------------------
# PyTorch Distributed Data Parallel (DDP)
# --------------------------------------------------------


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# model = ModernBERT(vocab_size=32000, hidden_size=512, num_layers=16, num_heads=8, num_kv_heads=8, max_seq_len=128) 


model = ModernBERT(vocab_size=32000, hidden_size=512, num_layers=16, num_heads=8, num_kv_heads=2, max_seq_len=128) #num_kv_heads=2
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])


# --------------------------------------------------------
# Hyperparameters & Optimization Schedule
# --------------------------------------------------------


from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(model, lr=3e-4, weight_decay=0.1):
    """
    AdamW:
      beta1 = 0.9
      beta2 = 0.95
      weight_decay = 0.1
    """

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Usually avoid weight decay on bias and norm parameters.
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
            {
                "params": decay_params,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    return optimizer


def build_cosine_warmup_scheduler(
    optimizer,
    total_training_steps,
    warmup_ratio=0.03,
    min_lr_ratio=0.1,
):
    """
    Cosine decay with linear warmup.

    warmup_ratio:
      0.01 = first 1% of total steps
      0.03 = first 3% of total steps
      0.05 = first 5% of total steps

    min_lr_ratio:
      final_lr = peak_lr * min_lr_ratio
    """

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

    scheduler = LambdaLR(optimizer, lr_lambda)

    return scheduler

# --------------------------------------------------------
# Dataset and Runtime Dynamic Masking (40%)
# --------------------------------------------------------


# 1. Initialize Dataset
train_dataset = TokenizedChunksDataset(raw_chunks=chunks) #your_list_of_chunks

# 2. Setup Collator (Assuming e.g., MASK ID=103, Vocab=30522)
data_collator = DataCollatorForMLM(
    mask_token_id=103, 
    vocab_size=30522, 
    mask_ratio=0.40,
    pad_token_id=0
)

# print('Data collator: ', data_collator.shape)

# --------------------------------------------------------
# Training Setup
# --------------------------------------------------------

num_epochs = 5

micro_batch_size_per_gpu = 8
num_gpus = 2 #4
gradient_accumulation_steps = 16

global_batch_size = (
    micro_batch_size_per_gpu
    * num_gpus
    * gradient_accumulation_steps
)

print("Global batch size:", global_batch_size)
# 8 x 4 x 16 = 512

num_samples = len(train_dataset)

steps_per_epoch = math.ceil(
    num_samples / global_batch_size
)

total_training_steps = steps_per_epoch * num_epochs

peak_lr = 3e-4       # usually 1e-4 to 6e-4
warmup_ratio = 0.03  # 1% to 5%


# --------------------------------------------------------
# Data Loader
# --------------------------------------------------------


from torch.utils.data.distributed import DistributedSampler

# sampler = DistributedSampler(train_dataset, shuffle=True)
sampler = DistributedSampler(
    train_dataset, 
    num_replicas=num_gpus, 
    rank=local_rank, #local_rank, current_rank
    shuffle=True  # <--- Shuffling is managed here!
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=micro_batch_size_per_gpu, # From your script (8), data
    # shuffle=True,
    collate_fn=data_collator,
    num_workers=4,                       # Utilizes multi-process data loading
    pin_memory=True, 
    sampler=sampler)


optimizer = build_optimizer(
    model,
    lr=peak_lr,
    weight_decay=0.1,
)

scheduler = build_cosine_warmup_scheduler(
    optimizer,
    total_training_steps=total_training_steps,
    warmup_ratio=warmup_ratio,
    min_lr_ratio=0.1,
)

# --------------------------------------------------------
# Training loop with Gradient Accumulation
# --------------------------------------------------------

# from torch.amp import autocast # Core import for mixed precision

# Checkpoint Helper Function
def save_checkpoint(model, optimizer, scheduler, epoch, global_step, loss, checkpoint_dir="checkpointsBPT"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pt")
    
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
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

# Ensure device type is exactly 'cuda' for autocast mapping
device_type = "cuda" if "cuda" in str(device) else "cpu"

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)

    for micro_step, batch in enumerate(train_loader):
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # --- INTEGRATION STEP 1: BF16 AUTOCAST ---
        # Wraps the forward pass and loss calculations
        with autocast(device_type=device_type, dtype=torch.bfloat16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

        # Backward pass executes outside the autocast block
        loss.backward()

        should_step = (
            (micro_step + 1) % gradient_accumulation_steps == 0
            or (micro_step + 1) == len(train_loader)
        )

        if should_step:
            # --- INTEGRATION STEP 2: GRADIENT CLIPPING ---
            # Unscaling gradients isn't required since BF16 does not use a GradScaler
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            current_lr = scheduler.get_last_lr()[0]

            # Print actual un-scaled loss value for accurate tracking
            print(
                f"epoch={epoch + 1} "
                f"step={global_step}/{total_training_steps} "
                f"loss={loss.item() * gradient_accumulation_steps:.4f} "
                f"lr={current_lr:.6e}"
            )
            
            # --- INTEGRATION STEP 3: PERIODIC CHECKPOINTING ---
            # Save every 5000 global steps (adjust interval as you see fit)
            if global_step % 5000 == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1, 
                    global_step, loss.item() * gradient_accumulation_steps
                )

    # Always save a checkpoint at the conclusion of an entire epoch
    save_checkpoint(
        model, optimizer, scheduler, epoch + 1, 
        global_step, loss.item() * gradient_accumulation_steps
    )




# model.train()

# optimizer.zero_grad(set_to_none=True)

# global_step = 0

# for epoch in range(num_epochs):
#     for micro_step, batch in enumerate(train_loader):
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)

#         logits = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#         )

#         loss = F.cross_entropy(
#             logits.view(-1, logits.size(-1)),
#             labels.view(-1),
#             ignore_index=-100,
#         )

#         # Scale loss because gradients are accumulated over several micro-batches.
#         loss = loss / gradient_accumulation_steps
#         loss.backward()

#         should_step = (
#             (micro_step + 1) % gradient_accumulation_steps == 0
#             or (micro_step + 1) == len(train_loader)
#         )

#         if should_step:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

#             optimizer.step()
#             scheduler.step()
#             optimizer.zero_grad(set_to_none=True)

#             global_step += 1

#             current_lr = scheduler.get_last_lr()[0]

#             print(
#                 f"epoch={epoch + 1} "
#                 f"step={global_step}/{total_training_steps} "
#                 f"loss={loss.item() * gradient_accumulation_steps:.4f} "
#                 f"lr={current_lr:.6e}"
#             )

# 3. Create DataLoader
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=micro_batch_size_per_gpu, # From your script (8)
#     shuffle=True,
#     collate_fn=data_collator,
#     num_workers=4,                       # Utilizes multi-process data loading
#     pin_memory=True                      # Speeds up tensor transfer to GPU VRAM
# )

 # can_use_flash = (
        #     FLASH_ATTN_AVAILABLE
        #     and x.is_cuda
        #     and x.dtype in (torch.float16, torch.bfloat16)
        #     and (attn_mask is None or causal)
        # )

        # if can_use_flash:
        #     # cos, sin = positional_embedding
        #     # q = apply_rope(q, cos, sin)
        #     # k = apply_rope(k, cos, sin)
        #     out = self._flash_attention(
        #         q,
        #         k,
        #         v,
        #         key_padding_mask=key_padding_mask,
        #         causal=causal,
        #     )
        # else:

        # def _flash_attention(self, q, k, v, key_padding_mask=None, causal=False):
    #     """
    #     q: [B, T, num_q_heads, head_dim]
    #     k: [B, T, num_kv_heads, head_dim]
    #     v: [B, T, num_kv_heads, head_dim]
    #     """
    #     dropout_p = self.dropout_p if self.training else 0.0

    #     if key_padding_mask is None:
    #         # Native GQA handling works perfectly here out of the box
    #         return flash_attn_func(
    #             q,
    #             k,
    #             v,
    #             dropout_p=dropout_p,
    #             softmax_scale=None,
    #             causal=causal,
    #         )

    #     valid_mask = ~key_padding_mask.bool()

    #     # BUG FIX: Unpad Q to get the shared structural pooling indexes 
    #     q_unpad, indices_q, cu_seqlens, max_seqlen = unpad_input(q, valid_mask)
        
    #     # Manually unpad K and V using Q's sequence index maps to guarantee perfect 
    #     # cross-head sequence matching and prevent runtime slicing errors.
    #     B, T = key_padding_mask.shape
    #     k_flat = k.flatten(0, 1) # Combine B and T -> [B*T, num_kv_heads, head_dim]
    #     v_flat = v.flatten(0, 1)
        
    #     k_unpad = k_flat[indices_q]
    #     v_unpad = v_flat[indices_q]

    #     # Flash attention handles GQA with variable lengths smoothly if structures are identical
    #     out_unpad = flash_attn_varlen_func(
    #         q_unpad,
    #         k_unpad,
    #         v_unpad,
    #         cu_seqlens_q=cu_seqlens,
    #         cu_seqlens_k=cu_seqlens,
    #         max_seqlen_q=max_seqlen,
    #         max_seqlen_k=max_seqlen,
    #         dropout_p=dropout_p,
    #         softmax_scale=None,
    #         causal=causal,
    #     )

    #     return pad_input(out_unpad, indices_q, B, T)

    # class RMSNorm(nn.Module):
#     def __init__(self, dim, eps=1e-6):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(dim))

#     def forward(self, x):
#         # Using rsqrt() replaces .sqrt() and the division '/'
#         inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
#         return self.weight * x * inv_rms