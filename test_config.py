# test_config.py
import os
from dotenv import load_dotenv

load_dotenv()

print("API Key de Google:", os.getenv("GOOGLE_API_KEY"))
print("API Key de OpenAI:", os.getenv("OPENAI_API_KEY"))
print("API Key de Anthropic:", os.getenv("ANTHROPIC_API_KEY"))