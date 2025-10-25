from src.api_clients.qwen2_client import Qwen2Client

def main():
    print("qwen2 completion 테스트")

    response = Qwen2Client.generate_completion()
    print(f"테스트 : '{response}'")

if __name__ == "__main__":
    main()