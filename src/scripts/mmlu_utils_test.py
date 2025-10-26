import sys
import os

from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_clients.qwen2_client import Qwen2Client
from utils.mmlu_utils import MMLUUtils


dataset = MMLUUtils.load_MMLU_dataset(subject = "abstract_algebra")
print(dataset['test']['question'][0])

for td in dataset['test']:
    td_question = td['question']
    td_subject = td['subject']
    td_choices = td['choices']
    td_answer = td['answer']
    formatted_question = MMLUUtils.build_question_format(td_question,td_choices)
    ans = Qwen2Client.generate_completion(formatted_question)
    print(f"ans = {ans}")
    print(f"td ans = {td_answer}")
    