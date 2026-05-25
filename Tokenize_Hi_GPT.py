from datasets import load_from_disk
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import RobertaProcessing
import os

# ---------------------------------------------------
# Load saved dataset
# ---------------------------------------------------

dataset = load_from_disk(
    "indiccorp_hindi_10percent"
)

print(dataset)

# ---------------------------------------------------
# Save raw text temporarily for tokenizer training
# Rust tokenizer trains efficiently from text files
# ---------------------------------------------------

text_file = "hindi_corpusPT.txt"

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

# GPT-2 style special tokens
tokenizer.train(
    files=[text_file],
    vocab_size=32000,
    min_frequency=2,
    special_tokens=[
        "<s>",         # Start of text (BOS)
        "<pad>",       # Padding token
        "</s>",        # End of text (EOS) - Crucial for autoregressive decoders
        "<unk>",       # Unknown token
        # "<mask>"       # Mask token (Optional, good to have)
    ]
)

print("Tokenizer training complete..")

# ---------------------------------------------------
# Post-Processing: Enable automated special token handling
# ---------------------------------------------------
# This step tells the tokenizer to automatically wrap encoded 
# sentences with <s> and </s>, which decoders rely on.
tokenizer.post_processor = RobertaProcessing(
    sep=("</s>", tokenizer.token_to_id("</s>")),
    cls=("<s>", tokenizer.token_to_id("<s>")),
)

# ---------------------------------------------------
# Save tokenizer
# ---------------------------------------------------

save_dir = "hindi_bpe_tokenizerGP"

os.makedirs(save_dir, exist_ok=True)

tokenizer.save_model(save_dir)

print("Tokenizer saved at:", save_dir)





# from datasets import load_from_disk
# from tokenizers import ByteLevelBPETokenizer
# # from transformers import PreTrainedTokenizerFast
# import os

# # ---------------------------------------------------
# # Load saved dataset
# # ---------------------------------------------------

# dataset = load_from_disk(
#     "indiccorp_hindi_10percent"
# )

# print(dataset)

# # ---------------------------------------------------
# # Save raw text temporarily for tokenizer training
# # Rust tokenizer trains efficiently from text files
# # ---------------------------------------------------

# text_file = "hindi_corpusPT.txt"

# # with open(text_file, "w", encoding="utf-8") as f:
# #     for example in dataset:
# #         f.write(example["text"] + "\n")

# with open(text_file, "w", encoding="utf-8") as f:
#     for example in dataset:
#         text = example["text"].strip()

#         if len(text) > 0:
#             f.write(text + "\n")

# # ---------------------------------------------------
# # Train Byte-Level BPE tokenizer
# # ---------------------------------------------------

# # tokenizer = ByteLevelBPETokenizer()

# print("Tokenizer training begins..")

# # tokenizer.train(
# #     files=[text_file],

# #     vocab_size=32000,
# #     min_frequency=2,

# #     special_tokens=[
# #         "[PAD]",
# #         "[UNK]",
# #         "[CLS]",
# #         "[MASK]"
# #     ]
# # )

# tokenizer = ByteLevelBPETokenizer()

# tokenizer.train(
#     files=[text_file],

#     vocab_size=32000,
#     min_frequency=2,

#     special_tokens=[
#         "[PAD]",
#         "[UNK]",
#         "[CLS]",
#         "[SEP]",
#         "[MASK]"
#     ]
# )

# print("Tokenizer training complete..")
# # ---------------------------------------------------
# # Save tokenizer
# # ---------------------------------------------------

# save_dir = "hindi_bpe_tokenizer"

# os.makedirs(save_dir, exist_ok=True)

# tokenizer.save_model(save_dir)

# print("Tokenizer saved at:", save_dir)

# ---------------------------------------------------
# Convert to HuggingFace Fast tokenizer
# ---------------------------------------------------

# print("Converting to HF Fast tokenizer..")

# hf_tokenizer = PreTrainedTokenizerFast(
#     tokenizer_file=None,

#     vocab_file=f"{save_dir}/vocab.json",
#     merges_file=f"{save_dir}/merges.txt",

#     unk_token="[UNK]",
#     pad_token="[PAD]",
#     cls_token="[CLS]",
#     sep_token="[SEP]",
#     mask_token="[MASK]"
# )

# hf_tokenizer.save_pretrained(save_dir)

