from src.api_clients.qwen2_client import VllmClient
from src.evaluators.evaluator import Evaluator

def main():
    evaluator = Evaluator()

    results = evaluator.evaluate()
    print(results)
    print(results['mmlu'])

if __name__ == "__main__":
    main()