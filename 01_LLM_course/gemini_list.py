import google.generativeai as genai
from src.script import load_config

config = load_config()
api_key = config.get("gemini", {}).get("api_key", "add_your_api_key_here")

genai.configure(api_key=api_key)
# List available Gemini models
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
