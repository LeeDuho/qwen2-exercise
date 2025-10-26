from datasets import load_dataset

# dataset = load_dataset("cais/mmlu", "all")
dataset = load_dataset("cais/mmlu", "abstract_algebra")

print(f"dataset : {dataset}")
print(f"dataset['test'] : {dataset['test']}")
print(f"dataset['test'][0] : {dataset['test'][0]}")
print(f"dataset['test'][1] : {dataset['test'][1]}")