import re
from typing import List
import numpy as np
from datasets import load_dataset
from typing import Dict, Optional, List

class DatasetUtils:
    def init():
        pass
    

    ##################### MMLU #####################

    @staticmethod
    def load_MMLU_dataset(subject: str = "all"):
        """Load MMLU dataset."""
        dataset = load_dataset("cais/mmlu", subject)
        return dataset
    
    @staticmethod
    def load_MMLU_Pro_dataset():
        """Load MMLU-Pro dataset (harder version with 10 choices)."""
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")
        return dataset
    
    @staticmethod
    def load_MMLU_Redux_dataset():
        """Load MMLU-Redux dataset (error-corrected version)."""
        dataset = load_dataset("edinburgh-dawg/mmlu-redux")
        return dataset


    def build_question_prompt(dataset:dict, fewshot_dataset:List[dict] = None) -> str:
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

        if response and response[0].upper() in "ABCD":
            return response[0].upper()

        patterns = [
            r'[Aa]nswer\s*:\s*([ABCD])',
            r'^([ABCD])\.',
            r'\b([ABCD])\b'
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
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










    @staticmethod
    def build_question_prompt(dataset: dict, fewshot_dataset: List[dict] = None) -> str:
        """Build MMLU question prompt with few-shot examples."""
        subject = dataset['subject']
        question_prompt = f"The following are multiple choice questions (with answers) about {subject}.\n\n"

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
    
    @staticmethod
    def build_mmlu_pro_prompt(dataset: dict, fewshot_dataset: List[dict] = None) -> str:
        """Build MMLU-Pro prompt (10 choices instead of 4)."""
        category = dataset.get('category', 'general')
        prompt = f"The following are multiple choice questions about {category}.\n\n"
        
        if fewshot_dataset:
            for fd in fewshot_dataset:
                prompt += f"{fd['question']}\n"
                for i, option in enumerate(fd['options']):
                    prompt += f"{chr(65 + i)}. {option}\n"
                prompt += f"Answer: {fd['answer']}\n\n"
        
        prompt += f"{dataset['question']}\n"
        for i, option in enumerate(dataset['options']):
            prompt += f"{chr(65 + i)}. {option}\n"
        prompt += "Answer:"
        
        return prompt



    ##################### GSM8K #####################
    @staticmethod
    def load_GSM8K_dataset():
        """Load GSM8K dataset."""
        dataset = load_dataset("gsm8k", "main")
        return dataset
    
    @staticmethod
    def build_gsm8k_prompt(dataset: dict, fewshot_dataset: List[dict] = None) -> str:
        """Build GSM8K question prompt with few-shot examples."""
        prompt = "Solve the following math problem step by step. Provide your final answer after ####.\n\n"
        
        if fewshot_dataset:
            for fd in fewshot_dataset:
                prompt += f"Question: {fd['question']}\n"
                prompt += f"Answer: {fd['answer']}\n\n"
        
        prompt += f"Question: {dataset['question']}\n"
        prompt += "Answer:"
        
        return prompt
    
    @staticmethod
    def extract_gsm8k_answer(response: str) -> Optional[str]:
        """Extract numerical answer from GSM8K response."""
        # Look for #### pattern
        if "####" in response:
            answer_part = response.split("####")[-1].strip()
            # Extract first number
            match = re.search(r'-?\d+(?:,\d+)*(?:\.\d+)?', answer_part)
            if match:
                return match.group(0).replace(',', '')
        
        # Fallback: extract last number in response
        numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', response)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return None
    
    @staticmethod
    def normalize_gsm8k_answer(answer: str) -> Optional[float]:
        """Normalize GSM8K answer to float."""
        try:
            return float(str(answer).replace(',', '').strip())
        except (ValueError, AttributeError):
            return None
        



    
    # ============ MATH Dataset Methods ============
    @staticmethod
    def load_MATH_dataset():
        """Load MATH dataset (competition mathematics)."""
        dataset = load_dataset("hendrycks/competition_math")
        return dataset
    
    @staticmethod
    def build_math_prompt(dataset: dict, fewshot_dataset: List[dict] = None) -> str:
        """Build MATH dataset prompt."""
        prompt = "Solve the following mathematics problem. Show your work and put your final answer in \\boxed{}.\n\n"
        
        if fewshot_dataset:
            for fd in fewshot_dataset:
                prompt += f"Problem: {fd['problem']}\n"
                prompt += f"Solution: {fd['solution']}\n\n"
        
        prompt += f"Problem: {dataset['problem']}\n"
        prompt += "Solution:"
        
        return prompt
    
    @staticmethod
    def extract_math_answer(response: str) -> Optional[str]:
        """Extract answer from MATH dataset response (looks for \\boxed{} format)."""
        # Look for \boxed{answer}
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        match = re.search(boxed_pattern, response)
        if match:
            return match.group(1).strip()
        
        # Fallback: look for final answer patterns
        final_patterns = [
            r'(?:final answer|answer|therefore)[:\s]+([^.\n]+)',
            r'= ([^=\n]+)$'
        ]
        
        for pattern in final_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    ##################### ARC #####################
    @staticmethod
    def load_ARC_dataset(difficulty: str = "ARC-Challenge"):
        """Load ARC dataset.
        
        Args:
            difficulty: "ARC-Challenge" or "ARC-Easy"
        """
        dataset = load_dataset("ai2_arc", difficulty)
        return dataset
    

    @staticmethod
    def build_arc_prompt(dataset: dict, fewshot_dataset: List[dict] = None) -> str:
        """Build ARC question prompt with few-shot examples."""
        prompt = "Answer the following multiple choice question.\n\n"
        
        if fewshot_dataset:
            for fd in fewshot_dataset:
                prompt += f"Question: {fd['question']}\n"
                choices = fd['choices']
                for label, text in zip(choices['label'], choices['text']):
                    prompt += f"{label}. {text}\n"
                prompt += f"Answer: {fd['answerKey']}\n\n"
        
        prompt += f"Question: {dataset['question']}\n"
        choices = dataset['choices']
        for label, text in zip(choices['label'], choices['text']):
            prompt += f"{label}. {text}\n"
        prompt += "Answer:"
        
        return prompt
    
    @staticmethod
    def extract_arc_answer(response: str) -> Optional[str]:
        """Extract answer from ARC response."""
        response = response.strip()
        
        # Check if first character is a valid label (1-5 or A-E)
        if response and response[0].upper() in "ABCDE12345":
            return response[0].upper()
        
        # Pattern matching
        patterns = [
            r'[Aa]nswer\s*:\s*([ABCDE12345])',
            r'^([ABCDE12345])\.',
            r'\b([ABCDE12345])\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).upper()
        
        return None
    
    @staticmethod
    def normalize_arc_answer(answer_key: str, choices: dict) -> str:
        """Normalize ARC answer to letter format (A-E)."""
        # If answerKey is numeric (1-5), convert to letter
        if answer_key.isdigit():
            return chr(64 + int(answer_key))
        # If already letter, return as is
        return answer_key.upper()
    
    ##################### HellaSwag #####################
    @staticmethod
    def load_HellaSwag_dataset():
        """Load HellaSwag dataset."""
        dataset = load_dataset("hellaswag")
        return dataset


    @staticmethod
    def build_hellaswag_prompt(dataset: dict, fewshot_dataset: List[dict] = None) -> str:
        """Build HellaSwag question prompt with few-shot examples."""
        prompt = "Complete the following scenario by choosing the most appropriate ending.\n\n"
        
        if fewshot_dataset:
            for fd in fewshot_dataset:
                prompt += f"Context: {fd['ctx']}\n"
                prompt += "Endings:\n"
                for i, ending in enumerate(fd['endings']):
                    prompt += f"{chr(65 + i)}. {ending}\n"
                prompt += f"Answer: {chr(65 + int(fd['label']))}\n\n"
        
        prompt += f"Context: {dataset['ctx']}\n"
        prompt += "Endings:\n"
        for i, ending in enumerate(dataset['endings']):
            prompt += f"{chr(65 + i)}. {ending}\n"
        prompt += "Answer:"
        
        return prompt
    
    @staticmethod
    def extract_hellaswag_answer(response: str) -> Optional[str]:
        """Extract answer from HellaSwag response."""
        response = response.strip()
        
        # Check if first character is A-D
        if response and response[0].upper() in "ABCD":
            return response[0].upper()
        
        # Pattern matching
        patterns = [
            r'[Aa]nswer\s*:\s*([ABCD])',
            r'^([ABCD])\.',
            r'\b([ABCD])\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).upper()
        
        return None






    # ============ BBH (Big-Bench Hard) Methods ============
    @staticmethod
    def load_BBH_dataset():
        """Load Big-Bench Hard dataset."""
        dataset = load_dataset("lukaemon/bbh")
        return dataset
    
    @staticmethod
    def build_bbh_prompt(dataset: dict, fewshot_dataset: List[dict] = None, use_cot: bool = True) -> str:
        """Build BBH prompt with optional Chain-of-Thought."""
        prompt = ""
        
        if fewshot_dataset and use_cot:
            for fd in fewshot_dataset:
                prompt += f"Q: {fd['input']}\n"
                if 'cot' in fd and fd['cot']:
                    prompt += f"A: {fd['cot']} So the answer is {fd['target']}\n\n"
                else:
                    prompt += f"A: {fd['target']}\n\n"
        elif fewshot_dataset:
            for fd in fewshot_dataset:
                prompt += f"Q: {fd['input']}\n"
                prompt += f"A: {fd['target']}\n\n"
        
        prompt += f"Q: {dataset['input']}\n"
        prompt += "A:"
        
        return prompt
    
    @staticmethod
    def extract_bbh_answer(response: str) -> Optional[str]:
        """Extract answer from BBH response."""
        # BBH answers vary by task, extract last non-empty line or after "answer is"
        if "answer is" in response.lower():
            parts = re.split(r'answer is', response, flags=re.IGNORECASE)
            if len(parts) > 1:
                answer = parts[-1].strip()
                # Remove trailing punctuation
                answer = re.sub(r'[.,!?]$', '', answer)
                return answer.strip()
        
        # Extract last line
        lines = [l.strip() for l in response.split('\n') if l.strip()]
        if lines:
            return lines[-1]
        
        return response.strip()



    # ============ GPQA Methods ============
    @staticmethod
    def load_GPQA_dataset(subset: str = "gpqa_main"):
        """Load GPQA dataset.
        
        Args:
            subset: "gpqa_extended", "gpqa_main", or "gpqa_diamond"
        """
        dataset = load_dataset("Idavidrein/gpqa", subset)
        return dataset
    
    @staticmethod
    def build_gpqa_prompt(dataset: dict, fewshot_dataset: List[dict] = None) -> str:
        """Build GPQA prompt (graduate-level science questions)."""
        prompt = "Answer the following graduate-level science question.\n\n"
        
        if fewshot_dataset:
            for fd in fewshot_dataset:
                prompt += f"Question: {fd['Question']}\n"
                prompt += f"A. {fd['Correct Answer']}\n"
                prompt += f"B. {fd['Incorrect Answer 1']}\n"
                prompt += f"C. {fd['Incorrect Answer 2']}\n"
                prompt += f"D. {fd['Incorrect Answer 3']}\n"
                prompt += f"Answer: A\n\n"  # Correct answer is always first in fewshot
        
        # Build choices list
        choices = [
            dataset['Correct Answer'],
            dataset['Incorrect Answer 1'],
            dataset['Incorrect Answer 2'],
            dataset['Incorrect Answer 3']
        ]
        
        prompt += f"Question: {dataset['Question']}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        prompt += "Answer:"
        
        return prompt
    
    @staticmethod
    def extract_gpqa_answer(response: str) -> Optional[str]:
        """Extract answer from GPQA response."""
        response = response.strip()
        
        if response and response[0].upper() in "ABCD":
            return response[0].upper()
        
        patterns = [
            r'[Aa]nswer\s*:\s*([ABCD])',
            r'^([ABCD])\.',
            r'\b([ABCD])\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).upper()
        
        return None

    # ============ Code Generation: HumanEval & MBPP ============
    @staticmethod
    def load_HumanEval_dataset():
        """Load HumanEval dataset."""
        dataset = load_dataset("openai_humaneval")
        return dataset
    
    @staticmethod
    def load_MBPP_dataset():
        """Load MBPP dataset."""
        dataset = load_dataset("mbpp", "sanitized")
        return dataset
    
    @staticmethod
    def build_humaneval_prompt(dataset: dict, fewshot_dataset: List[dict] = None) -> str:
        """Build HumanEval prompt."""
        prompt = ""
        
        if fewshot_dataset:
            for fd in fewshot_dataset:
                prompt += f"{fd['prompt']}\n"
                prompt += f"{fd['canonical_solution']}\n\n"
        
        prompt += dataset['prompt']
        
        return prompt
    
    @staticmethod
    def build_mbpp_prompt(dataset: dict, fewshot_dataset: List[dict] = None) -> str:
        """Build MBPP prompt."""
        prompt = ""
        
        if fewshot_dataset:
            for fd in fewshot_dataset:
                prompt += f'"""{fd["text"]}"""\n'
                prompt += f"{fd['code']}\n\n"
        
        prompt += f'"""{dataset["text"]}"""\n'
        
        return prompt
    
    @staticmethod
    def extract_code(response: str, language: str = "python") -> str:
        """Extract code from response."""
        # Look for code blocks
        code_block_pattern = f'```{language}\\n(.*?)```'
        match = re.search(code_block_pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Look for any code block
        code_block_pattern = r'```.*?\n(.*?)```'
        match = re.search(code_block_pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Return as is
        return response.strip()





    # ============ Common Methods ============

    @staticmethod
    def extract_answer(response: str) -> Optional[str]:
        """Extract answer for MMLU (kept for backward compatibility)."""
        response = response.strip()

        if response and response[0].upper() in "ABCD":
            return response[0].upper()

        patterns = [
            r'[Aa]nswer\s*:\s*([ABCD])',
            r'^([ABCD])\.',
            r'\b([ABCD])\b'
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).upper()

        return None

    @staticmethod
    def compute_accuracy(predictions: List, references: List) -> Dict:
        """Compute accuracy for MMLU."""
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
    
    @staticmethod
    def compute_exact_match_accuracy(predictions: List, references: List) -> Dict:
        """Compute exact match accuracy for general use."""
        correct = sum(1 for p, r in zip(predictions, references) if p == r)
        num_valid = sum(1 for p in predictions if p is not None)
        
        return {
            "accuracy": correct / len(predictions) if predictions else 0.0,
            "correct": correct,
            "num_valid": num_valid,
            "num_total": len(predictions)
        }
    
    @staticmethod
    def compute_gsm8k_accuracy(predictions: List, references: List) -> Dict:
        """Compute accuracy for GSM8K (numerical comparison)."""
        correct = 0
        num_valid = 0
        
        for pred, ref in zip(predictions, references):
            if pred is None:
                continue
            
            num_valid += 1
            
            # Extract reference answer (after ####)
            if "####" in ref:
                ref_answer = ref.split("####")[-1].strip()
            else:
                ref_answer = ref.strip()
            
            # Normalize both answers
            pred_norm = DatasetUtils.normalize_gsm8k_answer(pred)
            ref_norm = DatasetUtils.normalize_gsm8k_answer(ref_answer)
            
            if pred_norm is not None and ref_norm is not None:
                if abs(pred_norm - ref_norm) < 1e-6:  # Float comparison with tolerance
                    correct += 1
        
        return {
            "accuracy": correct / len(predictions) if predictions else 0.0,
            "correct": correct,
            "num_valid": num_valid,
            "num_total": len(predictions)
        }


