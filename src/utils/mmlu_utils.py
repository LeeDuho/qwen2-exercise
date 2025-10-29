import re
import numpy as np
from datasets import load_dataset

class MMLUUtils:
    def init():
        pass

    def load_MMLU_dataset(subject:str = "all"):
        dataset = load_dataset("cais/mmlu", subject)
        return dataset

    def build_question_prompt(dataset, fewshot_dataset):
        subject = dataset['subject']

        question_prompt = ""

        question_prompt += f"The following are multiple choice questions (with answers) about {subject}.\n\n"

        if fewshot_dataset:
            for fd in fewshot_dataset:
                question = fd["question"]
                choices = fd["choices"]

                question_prompt += f"{question}\n"
                for i, choice in enumerate(choices):
                    question_prompt += f"{chr(65 + i)}. {choice}\n"

                question_prompt += f"Answer: {chr(65 + fd['answer'])}\n"

        question = dataset["question"]
        choices = dataset["choices"]
        question_prompt += f"{question}\n"
        for i, choice in enumerate(choices):
            question_prompt += f"{chr(65 + i)}. {choice}\n"
        
        question_prompt += "Answer:"
        
        return question_prompt
    
    def extract_answer(response: str) -> str | None:
        response = response.strip()

        if len(response) > 0 and response[0].upper() in "ABCD":
            return response[0].upper()

        patterns = [
            r'[Aa]nswer\s*:\s*([ABCD])', 
            r'[Tt]he answer is\s*([ABCD])',  
            r'[\(\[]([ABCD])[\)\]]', 
            r'^([ABCD])\.',   
            r'\b([ABCD])\b'     
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        return None

    def compute_accuracy(predictions, references):
        ref_letters = [chr(65 + r) for r in references]

        correct = []
        for pred, ref in zip(predictions, ref_letters):
            if pred is None:
                correct.append(0)
            else:
                correct.append(1 if pred == ref else 0)

        num_valid = sum(1 for p in predictions if p is not None)

        return {
            "accuracy": float(np.mean(correct)),
            "num_valid": num_valid,
            "num_total": len(predictions)
        }
