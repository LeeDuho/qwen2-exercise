import sys
import os
from tqdm import tqdm
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_clients.qwen2_client import VllmClient
from utils.mmlu_utils import MMLUUtils


qwen2_client = VllmClient()


response = qwen2_client.generate_completion("what is your name")

print(response)

