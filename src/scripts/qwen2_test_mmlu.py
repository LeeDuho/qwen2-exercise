import sys
import os

from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_clients.qwen2_client import Qwen2Client


dataset = load_dataset("cais/mmlu", "abstract_algebra")
print(dataset)
test_dataset = dataset['test']
print(test_dataset)

question_dataset = test_dataset['question']
subject_dataset = test_dataset['subject']
choices_dataset = test_dataset['choices']
answer_dataset = test_dataset['answer']

tmp = 0
for td in test_dataset:
    question_dataset = td['question']
    subject_dataset = td['subject']
    choices_dataset = td['choices']
    answer_dataset = td['answer']
    print(question_dataset)
    print(subject_dataset)
    print(choices_dataset)
    print(answer_dataset)

    response = Qwen2Client.generate_completion(question_dataset)
    print(f"q:{question_dataset}")
    print(f"a:{response}")
    print("----------")
    tmp= tmp+1
    if tmp == 10:
        break