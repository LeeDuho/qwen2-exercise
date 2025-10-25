import requests

class Qwen2Client:
    def __init__():
        pass
    
    def generate_completion() -> str:

        url = "http://localhost:63700/v1/completions"
        
        payload = {
            "model": "Qwen/Qwen2.5-3B",
            "prompt": "hi my name is",
        }
        
        headers = {
            "Content-Type": "application/json"
        }
    
        res = requests.post(url, json=payload, headers=headers)
        
        print(f"res: {res.json()}")
        data = res.json()
        
        ret = data["choices"][0]["text"]
        return ret
        