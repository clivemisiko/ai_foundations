import os
import sys
from dotenv import load_dotenv

load_dotenv()

print('=== Environment & Platform Verification ===')
print(f'1. Python Version: {sys.version.split()[0]}')

# Check JupyterLab
try:
    import jupyterlab
    print(f'2. JupyterLab: Ready (v{jupyterlab.__version__})')
except ImportError as e:
    print(f'2. JupyterLab: Not installed ({e})')

# Check Hugging Face Hub connectivity
try:
    from huggingface_hub import HfApi
    api = HfApi()
    model_info = api.model_info('google/gemma-2-2b')
    print(f'3. Hugging Face Access: Verified (queried model: {model_info.id})')
except Exception as e:
    print(f'3. Hugging Face Access: Error ({e})')

# Check Kaggle library
try:
    import kaggle
    print('4. Kaggle Tooling: Verified (library installed and CLI ready)')
except Exception as e:
    print(f'4. Kaggle Tooling: Error ({e})')

# Check Gemini API
api_key = os.environ.get('GEMINI_API_KEY')
if api_key:
    masked = api_key[:4] + '...' + api_key[-4:] if len(api_key) > 8 else '***'
    print(f'5. API Key in Env: Loaded securely ({masked}) - not hardcoded')
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents='Respond with: ENVIRONMENT_READY'
        )
        print(f'6. Gemini API Call: Success! Response = {response.text.strip()}')
    except Exception as e:
        print(f'6. Gemini API Call: Failed ({e})')
else:
    print('5. API Key in Env: Not found in .env')

print('============================================')
