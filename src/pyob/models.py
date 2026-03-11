import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from typing import Callable, Optional

import requests

logger = logging.getLogger("PyOuroBoros")

try:
    if (
        os.environ.get("GITHUB_ACTIONS") == "true"
        or os.environ.get("CI") == "true"
        or "GITHUB_RUN_ID" in os.environ
    ):
        OLLAMA_AVAILABLE = False
    else:
        import ollama

        OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

GEMINI_MODEL = os.environ.get("PYOB_GEMINI_MODEL", "gemini-2.5-flash")
LOCAL_MODEL = os.environ.get("PYOB_LOCAL_MODEL", "qwen3-coder:30b")


def stream_gemini(prompt: str, api_key: str, on_chunk: Callable[[], None]) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:streamGenerateContent?alt=sse&key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1},
    }
    response = requests.post(url, headers=headers, json=data, stream=True, timeout=120)
    if response.status_code != 200:
        return f"ERROR_CODE_{response.status_code}: {response.text}"
    response_text = ""
    for line in response.iter_lines(decode_unicode=True):
        if line.startswith("data: "):
            try:
                chunk_data = json.loads(line[6:])
                text = chunk_data["candidates"][0]["content"]["parts"][0]["text"]
                on_chunk()
                print(text, end="", flush=True)
                response_text += text
            except (KeyError, IndexError, json.JSONDecodeError):
                pass
    return response_text


def stream_ollama(prompt: str, on_chunk: Callable[[], None]) -> str:
    if (
        os.environ.get("GITHUB_ACTIONS") == "true"
        or os.environ.get("CI") == "true"
        or "GITHUB_RUN_ID" in os.environ
    ):
        logger.error(
            "🚫 SECURITY VIOLATION: Ollama called in Cloud environment. ABORTING."
        )
        time.sleep(60)
        return "ERROR_CODE_CLOUD_OLLAMA_FORBIDDEN"

    if not OLLAMA_AVAILABLE:
        logger.error("🚫 Ollama is not available.")
        time.sleep(60)
        return "ERROR_CODE_OLLAMA_UNAVAILABLE"

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
                print(content, end="", flush=True)
                response_text += content
    except Exception as e:
        logger.error(f"Ollama Error: {e}")
        time.sleep(10)
    return response_text


def stream_github_models(
    prompt: str, on_chunk: Callable[[], None], model_name: str = "Llama-3"
) -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error("🚫 GITHUB_TOKEN is missing. Cannot use GitHub Models.")
        time.sleep(60)
        return "ERROR_CODE_TOKEN_MISSING"

    endpoint = "https://models.inference.ai.azure.com/chat/completions"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    actual_model = "Meta-Llama-3.3-70B-Instruct" if model_name == "Llama-3" else "Phi-4"

    data = {
        "model": actual_model,
        "messages": [
            {
                "role": "system",
                "content": "You are a code generation engine. Output ONLY raw code.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "temperature": 0.1,
        "max_tokens": 4096,
    }

    full_text = ""
    try:
        response = requests.post(
            endpoint, headers=headers, json=data, stream=True, timeout=120
        )

        if response.status_code != 200:
            error_body = response.text
            logger.error(
                f"❌ GitHub Models ({actual_model}) Error {response.status_code}: {error_body}"
            )
            return f"ERROR_CODE_{response.status_code}"

        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8").replace("data: ", "")
            if line_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(line_str)
                content = (
                    chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                )
                if content:
                    full_text += content
                    on_chunk()
            except Exception:
                continue
        return full_text
    except Exception as e:
        logger.error(f"❌ GitHub Models Exception: {e}")
        return f"ERROR_CODE_EXCEPTION: {str(e)}"


def stream_single_llm(
    prompt: str,
    key: Optional[str] = None,
    context: str = "",
    gh_model: str = "Llama-3",
) -> str:
    input_tokens = len(prompt) // 4
    first_chunk_received = [False]
    gen_start_time = time.time()
    is_cloud = (
        os.environ.get("GITHUB_ACTIONS") == "true"
        or os.environ.get("CI") == "true"
        or "GITHUB_RUN_ID" in os.environ
    )

    def spinner():
        spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while not first_chunk_received[0]:
            cols, _ = shutil.get_terminal_size((80, 20))
            elapsed = time.time() - gen_start_time
            expected_time = max(1, input_tokens / 12.0)
            progress = min(1.0, elapsed / expected_time)
            bar_len = max(10, cols - 65)
            filled = int(progress * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            status = f"{spinner_chars[i]} Reading [{context}] ~{input_tokens} ctx... [{bar}] {progress * 100:.1f}%"
            sys.stdout.write(f"\r\033[K{status[: cols - 1]}")
            sys.stdout.flush()
            i = (i + 1) % len(spinner_chars)
            time.sleep(0.1)

    t = threading.Thread(target=spinner, daemon=True)
    t.start()

    def on_chunk():
        if not first_chunk_received[0]:
            first_chunk_received[0] = True
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            source = f"Gemini ...{key[-4:]}" if key else f"GitHub Models ({gh_model})"
            if not key and not is_cloud:
                source = "Local Ollama"
            print(f"🤖 AI Output ({source}): ", end="", flush=True)

    response_text = ""
    try:
        if key is not None:
            response_text = stream_gemini(prompt, key, on_chunk)
        elif is_cloud:
            response_text = stream_github_models(prompt, on_chunk, model_name=gh_model)
        else:
            response_text = stream_ollama(prompt, on_chunk)
    except Exception as e:
        first_chunk_received[0] = True
        return f"ERROR_CODE_EXCEPTION: {e}"

    first_chunk_received[0] = True
    if response_text and not response_text.startswith("ERROR_CODE_"):
        print(
            f"\n\n[✅ Generation Complete: ~{len(response_text) // 4} tokens in {time.time() - gen_start_time:.1f}s]"
        )
    return response_text


def get_valid_llm_response_engine(
    prompt: str,
    validator: Callable[[str], bool],
    key_cooldowns: dict[str, float],
    context: str = "",
) -> str:
    attempts = 0
    is_cloud = (
        os.environ.get("GITHUB_ACTIONS") == "true"
        or os.environ.get("CI") == "true"
        or "GITHUB_RUN_ID" in os.environ
    )

    while True:
        key = None
        now = time.time()
        available_keys = [k for k, cd in key_cooldowns.items() if now > cd]
        response_text = None

        if available_keys:
            key = available_keys[attempts % len(available_keys)]
            logger.info(
                f"Attempting Gemini Key {attempts % len(available_keys) + 1}/{len(available_keys)}"
            )
            response_text = stream_single_llm(prompt, key=key, context=context)
        elif is_cloud:
            logger.warning("⏳ Gemini limited. Pivoting to GitHub Models (Llama-3)...")
            response_text = stream_single_llm(
                prompt, key=None, context=context, gh_model="Llama-3"
            )
        else:
            logger.info("🏠 Using Local Ollama Engine...")
            response_text = stream_single_llm(prompt, key=None, context=context)

        if not response_text or response_text.startswith("ERROR_CODE_"):
            if "429" in response_text and key:
                key_cooldowns[key] = time.time() + 1200
                logger.warning(f"⚠️ Key {key[-4:]} rate-limited. Rotating...")

            if is_cloud:
                if key:
                    logger.warning(
                        "☁️ Gemini failed/limited. Pivoting to GitHub Models (Llama-3)..."
                    )
                    response_text = stream_single_llm(
                        prompt, key=None, context=context, gh_model="Llama-3"
                    )

                if not response_text or response_text.startswith("ERROR_CODE_"):
                    logger.warning(
                        "☁️ Llama-3 failed. Pivoting to GitHub Models (Phi-4)..."
                    )
                    response_text = stream_single_llm(
                        prompt, key=None, context=context, gh_model="Phi-4"
                    )

                if not response_text or response_text.startswith("ERROR_CODE_"):
                    wait = 60
                    logger.warning(
                        f"⚠️ All Cloud Engines failed. Sleeping {wait}s for refill..."
                    )
                    time.sleep(wait)
                    attempts += 1
                    continue

        if not response_text or response_text.startswith("ERROR_CODE_"):
            wait = 10 if not is_cloud else 5
            logger.warning(f"⚠️ Generic LLM error. Backing off {wait}s...")
            time.sleep(wait)
            attempts += 1
            continue

        if validator(response_text):
            if is_cloud:
                time.sleep(5)
            return response_text
        else:
            clean_text = re.sub(
                r"^(Here is the code:)|(I suggest:)|(```)",
                "",
                response_text,
                flags=re.IGNORECASE,
            )
            if validator(clean_text):
                return clean_text

            wait = 15 if is_cloud else 5
            logger.warning(f"⚠️ Response invalid. Backing off {wait}s...")
            time.sleep(wait)
            attempts += 1
