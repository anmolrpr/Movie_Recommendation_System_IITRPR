import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def test_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("No API Key found. Skipping real test.")
        return

    print(f"Testing model: gemma-3-27b-it")
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel('gemma-3-27b-it')
        response = model.generate_content("Hello, do you exist?")
        print("Success! Model responded.")
        print(f"Response: {response.text[:50]}...")
    except Exception as e:
        print(f"Model Test Failed: {e}")

if __name__ == "__main__":
    test_model()
