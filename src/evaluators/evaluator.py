"""Main evaluation engine for multiple benchmarks."""
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Optional, List
import random
import time

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.dataset_utils import DatasetUtils
from src.api_clients.qwen2_client import VllmClient


class Evaluator:
    """Main evaluator for multiple benchmarks.
    
    Supports:
    - MMLU: Multiple choice questions across 57 subjects
    - MMLU-Pro: Harder version with 10 choices
    - MMLU-Redux: Error-corrected version
    - GSM8K: Grade school math problems
    - MATH: Competition mathematics
    - ARC: AI2 Reasoning Challenge
    - HellaSwag: Commonsense NLI
    - BBH: Big-Bench Hard reasoning
    - GPQA: Graduate-level science
    - HumanEval: Code generation
    - MBPP: Code generation
    """
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        base_url: str = "http://localhost:16000/v1",
        output_dir: str = "results"
    ):
        self.client = VllmClient(model_name=model_name, base_url=base_url)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

    def evaluate(self, tasks: List[str] = None, limit: Optional[int] = None):
        results = {}
        results['mmlu'] = self._eval_mmlu()
        # """Run evaluation on specified tasks.
        
        # Args:
        #     tasks: List of task names. If None, runs all tasks.
        #     limit: Maximum number of samples per task (for quick testing)
        # """
        # available_tasks = {
        #     'mmlu': self._eval_mmlu,
        #     'mmlu_pro': self._eval_mmlu_pro,
        #     'mmlu_redux': self._eval_mmlu_redux,
        #     'gsm8k': self._eval_gsm8k,
        #     'math': self._eval_math,
        #     'arc_easy': lambda: self._eval_arc('ARC-Easy', limit),
        #     'arc_challenge': lambda: self._eval_arc('ARC-Challenge', limit),
        #     'hellaswag': self._eval_hellaswag,
        #     'bbh': self._eval_bbh,
        #     'gpqa': self._eval_gpqa,
        #     'humaneval': self._eval_humaneval,
        #     'mbpp': self._eval_mbpp,
        # }
        
        # if tasks is None:
        #     tasks = list(available_tasks.keys())
        
        # results = {}
        # for task_name in tasks:
        #     if task_name not in available_tasks:
        #         print(f"Warning: Unknown task '{task_name}', skipping...")
        #         continue
            
        #     print(f"\n{'='*80}")
        #     print(f"Evaluating: {task_name.upper()}")
        #     print(f"{'='*80}\n")
            
        #     try:
        #         task_results = available_tasks[task_name](limit=limit) if 'limit' in str(available_tasks[task_name].__code__.co_varnames) else available_tasks[task_name]()
        #         results[task_name] = task_results
                
        #         # Save intermediate results
        #         self._save_results(results)
                
        #     except Exception as e:
        #         print(f"Error evaluating {task_name}: {e}")
        #         import traceback
        #         traceback.print_exc()
        #         results[task_name] = {"error": str(e)}
        
        # return results

    def _eval_mmlu(self, limit: Optional[int] = None) -> Dict:
        """Evaluate MMLU task (57 subjects, 4 choices each).
        
        MMLU tests knowledge across diverse subjects from elementary to professional level.
        Uses 5-shot evaluation with examples from dev set.
        """
        dataset = DatasetUtils.load_MMLU_dataset(subject="all")
        dev_data = dataset["dev"]
        test_data = dataset["test"]
        
        # Apply limit if specified
        if limit:
            test_data = test_data.select(range(min(limit, len(test_data))))
        
        # Group dev data by subject for few-shot examples
        dev_by_subject = {}
        for dd in dev_data:
            subject = dd["subject"]
            if subject not in dev_by_subject:
                dev_by_subject[subject] = []
            dev_by_subject[subject].append(dd)
        
        # Sample 5-shot examples per subject
        fewshot_by_subject = {}
        for subject, subject_dev in dev_by_subject.items():
            num_samples = min(5, len(subject_dev))
            fewshot_by_subject[subject] = random.sample(subject_dev, num_samples)
        
        predictions = []
        references = []
        
        for td in tqdm(test_data, desc="MMLU"):
            subject_name = td["subject"]
            few_shot_data = fewshot_by_subject.get(subject_name, [])
            
            prompt = DatasetUtils.build_question_prompt(
                dataset=td,
                fewshot_dataset=few_shot_data
            )
            
            try:
                response = self.client.generate_completion(prompt, max_tokens=10)
                pred = DatasetUtils.extract_answer(response)
            except Exception as e:
                print(f"Error: {e}")
                pred = None
            
            predictions.append(pred)
            references.append(td["answer"])
            
            time.sleep(0.05)  # Rate limiting
        
        metrics = DatasetUtils.compute_accuracy(predictions, references)
        
        print(f"\nMMLU Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Valid: {metrics['num_valid']}/{metrics['num_total']}")
        
        return {
            "task": "mmlu",
            "metrics": metrics,
            "predictions": predictions,
            "references": references
        }

    def _eval_mmlu_pro(self, limit: Optional[int] = None) -> Dict:
        """Evaluate MMLU-Pro (10 choices, harder questions).
        
        MMLU-Pro is more challenging with:
        - 10 answer choices instead of 4
        - More reasoning-focused questions
        - Reduced noise and trivial questions
        """
        dataset = DatasetUtils.load_MMLU_Pro_dataset()
        test_data = dataset["test"]
        
        if limit:
            test_data = test_data.select(range(min(limit, len(test_data))))
        
        predictions = []
        references = []
        
        for sample in tqdm(test_data, desc="MMLU-Pro"):
            prompt = DatasetUtils.build_mmlu_pro_prompt(sample)
            
            try:
                response = self.client.generate_completion(prompt, max_tokens=10)
                # MMLU-Pro uses letters A-J for 10 choices
                pred = DatasetUtils.extract_answer(response)
            except Exception as e:
                print(f"Error: {e}")
                pred = None
            
            predictions.append(pred)
            references.append(sample['answer'])
            
            time.sleep(0.05)
        
        metrics = DatasetUtils.compute_exact_match_accuracy(predictions, references)
        
        print(f"\nMMLU-Pro Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Valid: {metrics['num_valid']}/{metrics['num_total']}")
        
        return {
            "task": "mmlu_pro",
            "metrics": metrics,
            "predictions": predictions,
            "references": references
        }

    def _eval_mmlu_redux(self, limit: Optional[int] = None) -> Dict:
        """Evaluate MMLU-Redux (error-corrected MMLU).
        
        MMLU-Redux fixes errors in original MMLU:
        - Corrected answer labels
        - Removed ambiguous questions
        - Improved question quality
        """
        dataset = DatasetUtils.load_MMLU_Redux_dataset()
        test_data = dataset["test"]
        
        if limit:
            test_data = test_data.select(range(min(limit, len(test_data))))
        
        predictions = []
        references = []
        
        for sample in tqdm(test_data, desc="MMLU-Redux"):
            # Use same prompt format as MMLU
            prompt = DatasetUtils.build_question_prompt(sample)
            
            try:
                response = self.client.generate_completion(prompt, max_tokens=10)
                pred = DatasetUtils.extract_answer(response)
            except Exception as e:
                print(f"Error: {e}")
                pred = None
            
            predictions.append(pred)
            references.append(sample["answer"])
            
            time.sleep(0.05)
        
        metrics = DatasetUtils.compute_accuracy(predictions, references)
        
        print(f"\nMMLU-Redux Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        
        return {
            "task": "mmlu_redux",
            "metrics": metrics,
            "predictions": predictions,
            "references": references
        }

    def _eval_gsm8k(self, limit: Optional[int] = None) -> Dict:
        """Evaluate GSM8K (Grade School Math 8K).
        
        GSM8K contains grade school math word problems requiring:
        - Multi-step reasoning
        - Arithmetic operations
        - Final numerical answer after ####
        
        Uses 8-shot evaluation (standard).
        """
        dataset = DatasetUtils.load_GSM8K_dataset()
        train_data = dataset["train"]
        test_data = dataset["test"]
        
        if limit:
            test_data = test_data.select(range(min(limit, len(test_data))))
        
        # Sample 8-shot examples (standard for GSM8K)
        num_shots = min(8, len(train_data))
        few_shot_examples = random.sample(list(train_data), num_shots)
        
        predictions = []
        references = []
        
        for sample in tqdm(test_data, desc="GSM8K"):
            prompt = DatasetUtils.build_gsm8k_prompt(
                sample,
                fewshot_dataset=few_shot_examples
            )
            
            try:
                # Need more tokens for step-by-step solution
                response = self.client.generate_completion(prompt, max_tokens=256)
                pred = DatasetUtils.extract_gsm8k_answer(response)
            except Exception as e:
                print(f"Error: {e}")
                pred = None
            
            predictions.append(pred)
            references.append(sample['answer'])
            
            time.sleep(0.05)
        
        metrics = DatasetUtils.compute_gsm8k_accuracy(predictions, references)
        
        print(f"\nGSM8K Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Correct: {metrics['correct']}/{metrics['num_total']}")
        
        return {
            "task": "gsm8k",
            "metrics": metrics,
            "predictions": predictions,
            "references": references
        }

    def _eval_math(self, limit: Optional[int] = None) -> Dict:
        """Evaluate MATH dataset (competition mathematics).
        
        MATH contains competition-level math problems:
        - Requires advanced mathematical reasoning
        - Answer in LaTeX \\boxed{} format
        - Multiple difficulty levels
        
        Uses 4-shot evaluation.
        """
        dataset = DatasetUtils.load_MATH_dataset()
        train_data = dataset["train"]
        test_data = dataset["test"]
        
        if limit:
            test_data = test_data.select(range(min(limit, len(test_data))))
        
        # Sample 4-shot examples
        num_shots = min(4, len(train_data))
        few_shot_examples = random.sample(list(train_data), num_shots)
        
        predictions = []
        references = []
        
        for sample in tqdm(test_data, desc="MATH"):
            prompt = DatasetUtils.build_math_prompt(
                sample,
                fewshot_dataset=few_shot_examples
            )
            
            try:
                # MATH needs many tokens for detailed solutions
                response = self.client.generate_completion(prompt, max_tokens=512)
                pred = DatasetUtils.extract_math_answer(response)
            except Exception as e:
                print(f"Error: {e}")
                pred = None
            
            predictions.append(pred)
            references.append(sample['solution'])
            
            time.sleep(0.05)
        
        metrics = DatasetUtils.compute_exact_match_accuracy(predictions, references)
        
        print(f"\nMATH Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        
        return {
            "task": "math",
            "metrics": metrics,
            "predictions": predictions,
            "references": references
        }

    def _eval_arc(self, difficulty: str = "ARC-Challenge", limit: Optional[int] = None) -> Dict:
        """Evaluate ARC (AI2 Reasoning Challenge).
        
        ARC tests scientific reasoning:
        - ARC-Easy: Easier questions
        - ARC-Challenge: Harder questions requiring reasoning
        - Multiple choice (A-E, up to 5 choices)
        
        Uses 25-shot for Challenge, 0-shot for Easy (standard).
        """
        dataset = DatasetUtils.load_ARC_dataset(difficulty)
        train_data = dataset["train"]
        test_data = dataset["test"]
        
        if limit:
            test_data = test_data.select(range(min(limit, len(test_data))))
        
        # ARC-Challenge uses 25-shot, ARC-Easy uses 0-shot
        num_shots = 25 if difficulty == "ARC-Challenge" else 0
        few_shot_examples = []
        if num_shots > 0 and len(train_data) > 0:
            num_shots = min(num_shots, len(train_data))
            few_shot_examples = random.sample(list(train_data), num_shots)
        
        predictions = []
        references = []
        
        for sample in tqdm(test_data, desc=f"ARC-{difficulty}"):
            prompt = DatasetUtils.build_arc_prompt(
                sample,
                fewshot_dataset=few_shot_examples if few_shot_examples else None
            )
            
            try:
                response = self.client.generate_completion(prompt, max_tokens=10)
                pred = DatasetUtils.extract_arc_answer(response)
            except Exception as e:
                print(f"Error: {e}")
                pred = None
            
            predictions.append(pred)
            # Normalize answer key to letter format
            ref = DatasetUtils.normalize_arc_answer(
                sample['answerKey'],
                sample['choices']
            )
            references.append(ref)
            
            time.sleep(0.05)
        
        metrics = DatasetUtils.compute_exact_match_accuracy(predictions, references)
        
        print(f"\nARC-{difficulty} Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        
        return {
            "task": f"arc_{difficulty.lower()}",
            "metrics": metrics,
            "predictions": predictions,
            "references": references
        }

    def _eval_hellaswag(self, limit: Optional[int] = None) -> Dict:
        """Evaluate HellaSwag (commonsense NLI).
        
        HellaSwag tests commonsense reasoning:
        - Complete scenario with most appropriate ending
        - 4 possible endings
        - Requires understanding of physical/social situations
        
        Uses 10-shot evaluation (standard).
        """
        dataset = DatasetUtils.load_HellaSwag_dataset()
        train_data = dataset["train"]
        test_data = dataset["validation"]  # HellaSwag uses validation as test
        
        if limit:
            test_data = test_data.select(range(min(limit, len(test_data))))
        
        # 10-shot (standard)
        num_shots = min(10, len(train_data))
        few_shot_examples = random.sample(list(train_data), num_shots)
        
        predictions = []
        references = []
        
        for sample in tqdm(test_data, desc="HellaSwag"):
            prompt = DatasetUtils.build_hellaswag_prompt(
                sample,
                fewshot_dataset=few_shot_examples
            )
            
            try:
                response = self.client.generate_completion(prompt, max_tokens=10)
                pred = DatasetUtils.extract_hellaswag_answer(response)
            except Exception as e:
                print(f"Error: {e}")
                pred = None
            
            predictions.append(pred)
            # Convert label to letter
            ref = chr(65 + int(sample['label']))
            references.append(ref)
            
            time.sleep(0.05)
        
        metrics = DatasetUtils.compute_exact_match_accuracy(predictions, references)
        
        print(f"\nHellaSwag Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        
        return {
            "task": "hellaswag",
            "metrics": metrics,
            "predictions": predictions,
            "references": references
        }

    def _eval_bbh(self, limit: Optional[int] = None) -> Dict:
        """Evaluate BBH (Big-Bench Hard).
        
        BBH tests challenging reasoning tasks:
        - 23 diverse reasoning tasks
        - Chain-of-Thought helpful
        - Various answer formats
        
        Uses 3-shot with CoT (standard).
        """
        dataset = DatasetUtils.load_BBH_dataset()
        # BBH doesn't have a standard train/test split
        # Use first 100 as few-shot pool, rest as test
        all_data = list(dataset["test"])
        
        few_shot_pool = all_data[:100]
        test_data = all_data[100:]
        
        if limit:
            test_data = test_data[:limit]
        
        # 3-shot with CoT (standard)
        num_shots = min(3, len(few_shot_pool))
        few_shot_examples = random.sample(few_shot_pool, num_shots)
        
        predictions = []
        references = []
        
        for sample in tqdm(test_data, desc="BBH"):
            prompt = DatasetUtils.build_bbh_prompt(
                sample,
                fewshot_dataset=few_shot_examples,
                use_cot=True
            )
            
            try:
                response = self.client.generate_completion(prompt, max_tokens=256)
                pred = DatasetUtils.extract_bbh_answer(response)
            except Exception as e:
                print(f"Error: {e}")
                pred = None
            
            predictions.append(pred)
            references.append(sample['target'])
            
            time.sleep(0.05)
        
        metrics = DatasetUtils.compute_exact_match_accuracy(predictions, references)
        
        print(f"\nBBH Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        
        return {
            "task": "bbh",
            "metrics": metrics,
            "predictions": predictions,
            "references": references
        }

    def _eval_gpqa(self, limit: Optional[int] = None) -> Dict:
        """Evaluate GPQA (Graduate-level science questions).
        
        GPQA contains very challenging science questions:
        - Graduate-level physics, chemistry, biology
        - Expert-vetted questions
        - 4 answer choices
        
        Uses 5-shot evaluation.
        """
        dataset = DatasetUtils.load_GPQA_dataset(subset="gpqa_main")
        train_data = dataset["train"]
        test_data = dataset["test"]
        
        if limit:
            test_data = test_data.select(range(min(limit, len(test_data))))
        
        # 5-shot
        num_shots = min(5, len(train_data))
        few_shot_examples = random.sample(list(train_data), num_shots)
        
        predictions = []
        references = []
        
        for sample in tqdm(test_data, desc="GPQA"):
            prompt = DatasetUtils.build_gpqa_prompt(
                sample,
                fewshot_dataset=few_shot_examples
            )
            
            try:
                response = self.client.generate_completion(prompt, max_tokens=10)
                pred = DatasetUtils.extract_gpqa_answer(response)
            except Exception as e:
                print(f"Error: {e}")
                pred = None
            
            predictions.append(pred)
            references.append('A')  # Correct answer is always A in processed format
            
            time.sleep(0.05)
        
        metrics = DatasetUtils.compute_exact_match_accuracy(predictions, references)
        
        print(f"\nGPQA Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        
        return {
            "task": "gpqa",
            "metrics": metrics,
            "predictions": predictions,
            "references": references
        }

    def _eval_humaneval(self, limit: Optional[int] = None) -> Dict:
        """Evaluate HumanEval (code generation).
        
        HumanEval tests Python code generation:
        - Function implementation from docstring
        - Unit test verification
        - 0-shot evaluation (standard)
        
        Note: Full evaluation requires code execution.
        """
        dataset = DatasetUtils.load_HumanEval_dataset()
        test_data = dataset["test"]
        
        if limit:
            test_data = test_data.select(range(min(limit, len(test_data))))
        
        predictions = []
        references = []
        
        for sample in tqdm(test_data, desc="HumanEval"):
            prompt = DatasetUtils.build_humaneval_prompt(sample)
            
            try:
                response = self.client.generate_completion(prompt, max_tokens=512)
                pred = DatasetUtils.extract_code(response, language="python")
            except Exception as e:
                print(f"Error: {e}")
                pred = None
            
            predictions.append(pred)
            references.append(sample['canonical_solution'])
            
            time.sleep(0.05)
        
        # Note: Proper HumanEval evaluation requires code execution
        print(f"\nHumanEval Results:")
        print(f"  Generated {len([p for p in predictions if p])} valid code samples")
        print(f"  Note: Full evaluation requires code execution (pass@k)")
        
        return {
            "task": "humaneval",
            "predictions": predictions,
            "references": references,
            "note": "Requires code execution for pass@k metrics"
        }

    def _eval_mbpp(self, limit: Optional[int] = None) -> Dict:
        """Evaluate MBPP (Mostly Basic Python Problems).
        
        MBPP tests Python programming:
        - Basic programming problems
        - Function implementation from description
        - 0-shot evaluation
        
        Note: Full evaluation requires code execution.
        """
        dataset = DatasetUtils.load_MBPP_dataset()
        test_data = dataset["test"]
        
        if limit:
            test_data = test_data.select(range(min(limit, len(test_data))))
        
        predictions = []
        references = []
        
        for sample in tqdm(test_data, desc="MBPP"):
            prompt = DatasetUtils.build_mbpp_prompt(sample)
            
            try:
                response = self.client.generate_completion(prompt, max_tokens=512)
                pred = DatasetUtils.extract_code(response, language="python")
            except Exception as e:
                print(f"Error: {e}")
                pred = None
            
            predictions.append(pred)
            references.append(sample['code'])
            
            time.sleep(0.05)
        
        print(f"\nMBPP Results:")
        print(f"  Generated {len([p for p in predictions if p])} valid code samples")
        print(f"  Note: Full evaluation requires code execution")
        
        return {
            "task": "mbpp",
            "predictions": predictions,
            "references": references,
            "note": "Requires code execution for pass@k metrics"
        }

    def _save_results(self, results: Dict):
        """Save evaluation results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"eval_results_{timestamp}.json"
        
        # Add metadata
        output_data = {
            "model": self.model_name,
            "timestamp": timestamp,
            "results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n✓ Results saved to: {output_file}")
        
        # Also save a summary
        summary_file = self.output_dir / f"eval_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Evaluation Summary\n")
            f.write(f"{'='*80}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            for task, result in results.items():
                f.write(f"\n{task.upper()}\n")
                f.write(f"{'-'*80}\n")
                if 'metrics' in result:
                    metrics = result['metrics']
                    if 'accuracy' in metrics:
                        f.write(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
                    if 'correct' in metrics:
                        f.write(f"Correct: {metrics['correct']}/{metrics.get('num_total', 0)}\n")
                elif 'error' in result:
                    f.write(f"Error: {result['error']}\n")
        
        print(f"✓ Summary saved to: {summary_file}")


# Example usage
if __name__ == "__main__":
    evaluator = Evaluator(
        model_name="Qwen/Qwen2.5-3B",
        base_url="http://localhost:16000/v1"
    )
    
    # Run specific tasks
    results = evaluator.evaluate(
        tasks=['mmlu', 'gsm8k', 'arc_challenge'],
        limit=10  # Quick test
    )
    
    # Or run all tasks
    # results = evaluator.evaluate()