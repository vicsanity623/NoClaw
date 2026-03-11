import json
import logging
import os
import requests
import sys
import time
import threading

logger = logging.getLogger("PyOuroBoros")

# We need these from core_utils, so we redefine them here temporarily 
# or we can update the import structure later. For now, assume these are constants.
GEMINI_MODEL = os.environ.get("PYOB_GEMINI_MODEL", "gemini-2.5-flash")
LOCAL_MODEL = os.environ.get("PYOB_LOCAL_MODEL", "qwen3-coder:30b")


def stream_gemini(prompt: str, api_key: str, on_chunk) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:streamGenerateContent?alt=sse&key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1},
    }
    try:
        response = requests.post(url, headers=headers, json=data, stream=True, timeout=220)
        if response.status_code != 200:
            logger.error(f"❌ Gemini API Error {response.status_code}: {response.text}")
            return f"ERROR_CODE_{response.status_code}"
        response_text = ""
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                try:
                    chunk_data = json.loads(line[6:])
                    text = chunk_data["candidates"][0]["content"]["parts"][0]["text"]
                    on_chunk()
                    response_text += text
                except (KeyError, IndexError, json.JSONDecodeError):
                    pass
        return response_text
    except Exception as e:
        return f"ERROR_CODE_EXCEPTION: {e}"

def stream_ollama(prompt: str, on_chunk) -> str:
    response_text = ""
    try:
        stream = ollama.chat(
            model=LOCAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_ctx": 32000},
            stream=True,
        )
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                on_chunk()
                response_text += content
    except Exception as e:
        logger.error(f"Ollama Error: {e}")
    return response_text

def stream_github_models(prompt: str, on_chunk, model_name: str = "Phi-4") -> str:
    """Fallback to GitHub Models API with dynamic model selection."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return "ERROR_CODE_TOKEN_MISSING"

    endpoint = "https://models.inference.ai.azure.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "x-ms-model-name": model_name,
    }
    
    actual_model = "Phi-4" if model_name == "Phi-4" else "Meta-Llama-3.3-70B-Instruct"
    
    data = {
        "messages": [
            {"role": "system", "content": "You are a code generation engine. Output ONLY raw code, no intro/outro."},
            {"role": "user", "content": prompt}
        ],
        "model": actual_model,
        "stream": True,
        "temperature": 0.1,
    }

    full_text = ""
    try:
        response = requests.post(endpoint, headers=headers, json=data, stream=True, timeout=120)
        if response.status_code != 200:
            logger.error(f"❌ GitHub Models ({actual_model}) Error {response.status_code}: {response.text}")
            return f"ERROR_CODE_{response.status_code}"

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8").replace("data: ", "")
                if decoded_line == "[DONE]": break
                try:
                    chunk = json.loads(decoded_line)
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        full_text += content
                        on_chunk()
                except Exception:
                    continue
        return full_text
    except Exception as e:
        logger.error(f"❌ GitHub Models Exception: {e}")
        return f"ERROR_CODE_EXCEPTION: {e}"
