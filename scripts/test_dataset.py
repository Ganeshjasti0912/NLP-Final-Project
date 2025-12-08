from datasets import load_dataset

print("Loading TravelPlanner dataset")

# Load all splits
dataset = {
    "train": load_dataset("osunlp/TravelPlanner", "train"),
    "validation": load_dataset("osunlp/TravelPlanner", "validation"),
    "test": load_dataset("osunlp/TravelPlanner", "test")
}

print("\n Dataset splits loaded successfully:")
for split, data in dataset.items():
    print(f"{split}: {len(data[split])} samples")

print("\nExample from train split:")
print(dataset["train"]["train"][0])
