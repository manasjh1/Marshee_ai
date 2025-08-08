import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_API_KEY not found!")
    exit(1)

print(f"API Key: {api_key[:10]}...")

genai.configure(api_key=api_key)

try:
    result = genai.embed_content(
        model="models/embedding-001",
        content="Test text"
    )
    
    if result and hasattr(result, 'embedding') and result.embedding:
        print(f"SUCCESS! Embedding dimension: {len(result.embedding)}")
    else:
        print(f"FAILED: No embedding returned")
        print(f"Response: {result}")
        
except Exception as e:
    print(f"ERROR: {e}")