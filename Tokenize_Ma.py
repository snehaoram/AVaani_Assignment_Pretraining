from datasets import load_from_disk
from tokenizers import ByteLevelBPETokenizer

import os

# ---------------------------------------------------
# Load saved dataset
# ---------------------------------------------------

dataset = load_from_disk(
    "indiccorp_marathi_10percent"
)

print(dataset)

# ---------------------------------------------------
# Save raw text temporarily for tokenizer training
# Rust tokenizer trains efficiently from text files
# ---------------------------------------------------

text_file = "marathi_corpusPT.txt"

with open(text_file, "w", encoding="utf-8") as f:
    for example in dataset:
        text = example["text"].strip()

        if len(text) > 0:
            f.write(text + "\n")

# ---------------------------------------------------
# Train Byte-Level BPE tokenizer
# ---------------------------------------------------

print("Tokenizer training begins..")

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=[text_file],

    vocab_size=32000,
    min_frequency=2,

    special_tokens=[
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]"
    ]
)

print("Tokenizer training complete..")

# ---------------------------------------------------
# Save tokenizer
# ---------------------------------------------------

save_dir = "marathi_bpe_tokenizer"

os.makedirs(save_dir, exist_ok=True)

tokenizer.save_model(save_dir)

print("Tokenizer saved at:", save_dir)


