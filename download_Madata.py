from datasets import load_dataset, Dataset

# Load dataset
dataset = load_dataset(
    "ai4bharat/IndicCorpV2",
    split="mar_Deva",
    streaming=True
)

print(dataset)

sampled = []

# Keep roughly 10%
for i, example in enumerate(dataset):

    if i % 10 == 0:
        sampled.append(example)

    # Optional stopping
    if len(sampled) >= 100000:
        break

# Convert to normal dataset
small_dataset = Dataset.from_list(sampled)

print("sampled")

# Save locally
small_dataset.save_to_disk(
    "indiccorp_marathi_10percent"
)

print("complete")
