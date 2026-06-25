import pathlib
import re

import google.generativeai as genai

text = pathlib.Path("test_gemini.py").read_text()
match = re.search(r'api_key="([^"]+)"', text)

if not match:
    raise RuntimeError("Could not find api_key in test_gemini.py")

genai.configure(api_key=match.group(1))

for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(model.name)
