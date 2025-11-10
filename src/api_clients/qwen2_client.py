import requests
from openai import OpenAI
import json

class VllmClient:
    def __init__(
            self,
            api_key:str = "EMPTY",
            base_url:str = "http://localhost:16000/v1", 
            model_name="Qwen/Qwen3-VL-8B-Instruct"
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=3600)
    
    def generate_completion(
            self,
            prompt:str,
            image_url:str = None,
            max_tokens:int = 10, 
            temperature:float =0.0
    ) -> str:
        
        contents = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        if image_url:
            contents.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            )
        messages = [
            {
                "role": "user",
                "content": contents
            }
        ]

        # res = self.client.chat.completions.create(model = self.model_name, messages=messages)
        res = self.client.chat.completions.create(
                model=self.model_name, 
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
        res_message = res.choices[0].message.content
        
        return res_message