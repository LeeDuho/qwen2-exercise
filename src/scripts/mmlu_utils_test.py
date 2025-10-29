import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_clients.qwen2_client import VllmClient
from utils.mmlu_utils import MMLUUtils


dataset = MMLUUtils.load_MMLU_dataset(subject = "abstract_algebra")
print(dataset)
print(dataset['test'])
print(dataset['test']['question'][0])

qwen2_client = VllmClient()

for td in dataset['test']:
    td_question = td['question']
    td_subject = td['subject']
    td_choices = td['choices']
    td_answer = td['answer']
    formatted_question = MMLUUtils.build_question_prompt(td_subject,td_question,td_choices)

    
    print(f"prompt ={formatted_question}")
    ans = qwen2_client.generate_completion(prompt=formatted_question)
    ans2 = MMLUUtils.extract_answer(ans)
    print(f"ans = {ans2}")
    print(f"td ans = {td_answer}")
    