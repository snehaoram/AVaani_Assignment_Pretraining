from datasets import load_dataset, Dataset

# Load dataset
dataset = load_dataset(
    "ai4bharat/IndicCorpV2",
    split="hin_Deva",
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
    "indiccorp_hindi_10percent"
)

print("complete")

# Assuming the dataset has a 'train' split
# train_data = dataset["train"]

# # Get 10% sample
# sampled_data = train_data.train_test_split(
#     test_size=0.1,
#     seed=42
# )["test"]

# print("sampled..!")

# # Save locally
# sampled_data.save_to_disk("indiccorpv2_hindi_10percent")

# print("Complete..!")