import sys
import os
from tqdm import tqdm
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_clients.qwen2_client import VllmClient
from utils.mmlu_utils import MMLUUtils


qwen2_client = VllmClient()

dataset = MMLUUtils.load_MMLU_dataset(subject = "all")
dev_set = dataset["dev"]
test_set = dataset["test"]


fewshot_examples = []
num_fewshot = 5
np.random.seed(42)
num_available = len(dev_set)
num_samples = min(num_fewshot, num_available)
indices = list(range(num_samples)) 
fewshot_examples = [dev_set[i] for i in indices]

predictions = []
references = []

for example in tqdm(test_set, desc="Evaluating"):
    prompt = MMLUUtils.build_question_prompt(example, fewshot_examples)

    response = qwen2_client.generate_completion(prompt)
    print(f"prompt:{prompt}")

    pred = MMLUUtils.extract_answer(response)
    predictions.append(pred)
    references.append(example["answer"])

metrics = MMLUUtils.compute_accuracy(predictions, references)


print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
print(f"Valid predictions: {metrics['num_valid']}/{metrics['num_total']}")
