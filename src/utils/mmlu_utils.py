from datasets import load_dataset

class MMLUUtils:
    def init():
        pass

    def load_MMLU_dataset(subject= "all"):
        dataset = load_dataset("cais/mmlu", subject)
        return dataset

    def build_question_format(question, choices):
        question_prompt = ""
        question_prompt += f"question: {question}\n choices:"
        for i, choice in enumerate(choices):
            question_prompt += f"{i}:{choice} "
        question_prompt += "answer just number, answer:"

        return question_prompt