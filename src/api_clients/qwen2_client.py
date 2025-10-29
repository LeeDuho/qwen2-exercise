import requests
import json

class VllmClient:
    def __init__(self, api_key:str = "" , base_url:str = "http://localhost:8000/v1", model="Qwen/Qwen2.5-3B-Instruct"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
    
    def generate_completion(
            self,
            prompt:str,
            max_tokens=10, 
            temperature=0.0
    ) -> str:
        
        url = self.base_url + "/completions"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        headers = {
            "Content-Type": "application/json"
        }
    
        res = requests.post(url, json=payload, headers=headers)
        
        data = res.json()
        print(data)
        ret = data["choices"][0]["text"]
        return ret