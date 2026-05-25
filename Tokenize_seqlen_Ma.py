# ---------------------------------------------------
# Load tokenizer
# ---------------------------------------------------

from tokenizers import ByteLevelBPETokenizer
import pickle

tokenizer = ByteLevelBPETokenizer(
    vocab="marathi_bpe_tokenizer/vocab.json",
    merges="marathi_bpe_tokenizer/merges.txt"
)

PAD_ID = tokenizer.token_to_id("[PAD]")
CLS_ID = tokenizer.token_to_id("[CLS]")
SEP_ID = tokenizer.token_to_id("[SEP]")
MASK_ID = tokenizer.token_to_id("[MASK]")

# ---------------------------------------------------
# Create continuous token stream
# ---------------------------------------------------

from datasets import load_from_disk

dataset = load_from_disk(
    "indiccorp_marathi_10percent"
)

all_tokens = []

for example in dataset:

    text = example["text"].strip()

    if len(text) == 0:
        continue

    encoded = tokenizer.encode(text)

    # Optional: add CLS and SEP
    tokens = [CLS_ID] + encoded.ids + [SEP_ID]

    all_tokens.extend(tokens)

print("Total tokens:", len(all_tokens))

# ---------------------------------------------------
# Chunk into constant-length sequences
# ---------------------------------------------------

MAX_LEN = 128

chunks = []

for i in range(0, len(all_tokens), MAX_LEN):

    chunk = all_tokens[i:i + MAX_LEN]

    # Only last chunk may need padding
    if len(chunk) < MAX_LEN:

        padding_length = MAX_LEN - len(chunk)

        chunk = chunk + [PAD_ID] * padding_length

    chunks.append(chunk)

print("Total chunks:", len(chunks))
print("Chunk length:", len(chunks[0]))

save_path = "marathi_chunks.pkl"

with open(save_path, "wb") as f:
    pickle.dump(chunks, f)

print("Chunks saved successfully.")