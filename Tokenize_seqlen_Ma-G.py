import os
import numpy as np
from datasets import load_from_disk
from tokenizers import ByteLevelBPETokenizer

# ---------------------------------------------------
# Load Tokenizer
# ---------------------------------------------------
tokenizer = ByteLevelBPETokenizer(
    vocab="marathi_bpe_tokenizerGP/vocab.json",
    merges="marathi_bpe_tokenizerGP/merges.txt"
)

PAD_ID = tokenizer.token_to_id("<pad>")

# ---------------------------------------------------
# Create continuous token stream
# ---------------------------------------------------
dataset = load_from_disk("indiccorp_marathi_10percent")

all_tokens = []
print("Tokenizing dataset...")

for example in dataset:
    text = example["text"].strip()
    if len(text) == 0:
        continue
    
    # Corrected: We do NOT manually add SOS/EOS here because
    # the trained tokenizer's post-processor already injects them.
    encoded = tokenizer.encode(text)
    all_tokens.extend(encoded.ids)

# Convert to a highly compressed NumPy array
all_tokens = np.array(all_tokens, dtype=np.int32)
print("Total tokens:", len(all_tokens))

# ---------------------------------------------------
# Efficiently Chunk into constant-length sequences
# ---------------------------------------------------
MAX_LEN = 128

# Calculate how many full chunks we can make
num_chunks = len(all_tokens) // MAX_LEN
remainder = len(all_tokens) % MAX_LEN

# Slice the array to only include full blocks
chunks = all_tokens[:num_chunks * MAX_LEN].reshape(-1, MAX_LEN)

# Handle the final remainder chunk if it exists
if remainder > 0:
    last_chunk = all_tokens[-remainder:]
    padding_length = MAX_LEN - remainder
    padded_last_chunk = np.concatenate([last_chunk, np.full(padding_length, PAD_ID, dtype=np.int32)])
    
    # Append the padded last chunk to our 2D array
    chunks = np.vstack([chunks, padded_last_chunk])

print("Total chunks:", chunks.shape[0])
print("Chunk shape:", chunks.shape)

# ---------------------------------------------------
# Save as NumPy Binary (Much faster than Pickle)
# ---------------------------------------------------
save_path = "marathi_chunksGP.npy"
np.save(save_path, chunks)

print(f"Chunks saved successfully to {save_path}.")






























# ---------------------------------------------------
# Load tokenizer
# ---------------------------------------------------

# from tokenizers import ByteLevelBPETokenizer
# import pickle

# tokenizer = ByteLevelBPETokenizer(
#     vocab="hindi_bpe_tokenizerGP/vocab.json",
#     merges="hindi_bpe_tokenizerGP/merges.txt"
# )

# PAD_ID = tokenizer.token_to_id("<pad>")
# SOS_ID = tokenizer.token_to_id("<s>")
# EOS_ID = tokenizer.token_to_id("</s>")
# UNK_ID = tokenizer.token_to_id("<unk>")

# # ---------------------------------------------------
# # Create continuous token stream
# # ---------------------------------------------------

# from datasets import load_from_disk

# dataset = load_from_disk(
#     "indiccorp_hindi_10percent"
# )

# all_tokens = []

# for example in dataset:

#     text = example["text"].strip()

#     if len(text) == 0:
#         continue

#     encoded = tokenizer.encode(text)

#     # Optional: add CLS and SEP
#     tokens = [SOS_ID] + encoded.ids + [EOS_ID]

#     all_tokens.extend(tokens)

# print("Total tokens:", len(all_tokens))

# # ---------------------------------------------------
# # Chunk into constant-length sequences
# # ---------------------------------------------------

# MAX_LEN = 128

# chunks = []

# for i in range(0, len(all_tokens), MAX_LEN):

#     chunk = all_tokens[i:i + MAX_LEN]

#     # Only last chunk may need padding
#     if len(chunk) < MAX_LEN:

#         padding_length = MAX_LEN - len(chunk)

#         chunk = chunk + [PAD_ID] * padding_length

#     chunks.append(chunk)

# print("Total chunks:", len(chunks))
# print("Chunk length:", len(chunks[0]))

# save_path = "hindi_chunksGP.pkl"

# with open(save_path, "wb") as f:
#     pickle.dump(chunks, f)

# print("Chunks saved successfully.")