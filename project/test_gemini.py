import os
import google.generativeai as genai

api_key = os.getenv("GEMINI_API_KEY", "")
if not api_key:
    # Fallback to empty string
    api_key = ""

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.0-flash")

response = model.generate_content(
    "Say hello"
)

print(response.text)
