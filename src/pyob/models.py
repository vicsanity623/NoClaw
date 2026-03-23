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

    # We use a long timeout on the request, but we will monitor chunk arrival internally
    response = requests.post(url, headers=headers, json=data, stream=True, timeout=120)
    if response.status_code != 200:
        return f"ERROR_CODE_{response.status_code}: {response.text}"

    response_text = ""
    last_chunk_time = time.time()

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            # Check for stall
            if time.time() - last_chunk_time > 30:
                logger.warning("Gemini stream stalled. Forcing closure.")
                break
            continue

        if line.startswith("data: "):
            last_chunk_time = time.time()  # Reset stall timer
            try:
                chunk_data = json.loads(line[6:])
                text = chunk_data["candidates"][0]["content"]["parts"][0]["text"]
                on_chunk()
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
            "SECURITY VIOLATION: Ollama called in Cloud environment. ABORTING."
        )
        time.sleep(60)  # CRITICAL: Hard sleep kills outer loop machine-gun attempts
        return "ERROR_CODE_CLOUD_OLLAMA_FORBIDDEN"

    if not OLLAMA_AVAILABLE:
        logger.error("Ollama is not available.")
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
        time.sleep(30)
        return f"ERROR_CODE_EXCEPTION: {e}"
    return response_text


def stream_github_models(
    prompt: str, on_chunk: Callable[[], None], model_name: str = "Llama-3"
) -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return "ERROR_CODE_TOKEN_MISSING"

    endpoint = "https://models.inference.ai.azure.com/chat/completions"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    actual_model = "Llama-3.3-70B-Instruct" if model_name == "Llama-3" else "Phi-4"

    data = {
        "model": actual_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 4096,
    }

    full_text = ""
    last_chunk_time = time.time()

    try:
        response = requests.post(
            endpoint, headers=headers, json=data, stream=True, timeout=120
        )

        for line in response.iter_lines():
            if not line:
                if time.time() - last_chunk_time > 30:
                    logger.warning("GitHub stream stalled. Forcing closure.")
                    break
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
                    last_chunk_time = time.time()
                    full_text += content
                    on_chunk()
            except Exception:
                continue
        return full_text
    except Exception as e:
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
            expected_time = max(1, input_tokens / 34.0)
            progress = min(1.0, elapsed / expected_time)
            bar_len = max(10, cols - 65)
            filled = int(progress * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            status = f"{spinner_chars[i]} Reading [{context}] ~{input_tokens} ctx... [{bar}] {progress * 100:.1f}%"
            sys.stdout.write(f"\r\033[K{status[: cols - 1]}")
            sys.stdout.flush()
            i = (i + 1) % len(spinner_chars)
            time.sleep(0.1)

    if is_cloud:
        print(f"Reading [{context}] ~{input_tokens} ctx...", flush=True)
    else:
        t = threading.Thread(target=spinner, daemon=True)
        t.start()

    def on_chunk():
        if not first_chunk_received[0]:
            first_chunk_received[0] = True
            if not is_cloud:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
            source = f"Gemini ...{key[-4:]}" if key else f"GitHub Models ({gh_model})"
            if not key and not is_cloud:
                source = "Local Ollama"
            print(f"AI Output ({source}): ", end="", flush=True)

    response_text = ""
    try:
        if key is not None:
            response_text = stream_gemini(prompt, key, on_chunk)
        elif is_cloud:
            response_text = stream_github_models(prompt, on_chunk, model_name=gh_model)

            # Immediately intercept 413, pause 60s, and force Gemini usage so outer loops don't panic
            if response_text and "413" in response_text:
                first_chunk_received[0] = True
                logger.warning(
                    "\nPayload too large. Sleeping 60s, then pivoting to Gemini..."
                )
                time.sleep(60)
                gemini_keys = [
                    k.strip()
                    for k in os.environ.get("PYOB_GEMINI_KEYS", "").split(",")
                    if k.strip()
                ]
                if gemini_keys:
                    # Return a specific signal string so the caller knows it worked
                    return stream_gemini(prompt, gemini_keys[0], on_chunk)
                else:
                    return "ERROR_CODE_413_NO_GEMINI_FALLBACK"

            # Force mandatory sleep if ANY cloud error escapes, breaking infinite loop triggers
            if response_text and response_text.startswith("ERROR_CODE_"):
                time.sleep(30)

        else:
            response_text = stream_ollama(prompt, on_chunk)
    except Exception as e:
        first_chunk_received[0] = True
        if is_cloud:
            time.sleep(30)
        return f"ERROR_CODE_EXCEPTION: {e}"

    first_chunk_received[0] = True
    if response_text and not response_text.startswith("ERROR_CODE_"):
        print(
            f"\n\n[Generation Complete: ~{len(response_text) // 4} tokens in {time.time() - gen_start_time:.1f}s]"
        )
    return response_text


def get_valid_llm_response_engine(
    prompt: str,
    validator: Callable[[str], bool],
    key_cooldowns: dict[str, float],
    context: str = "",
) -> str:
    """
    Robust engine that handles key rotation across multiple providers.
    Uses cooldown tracking to ensure maximum utilization of free-tier quotas.
    """
    attempts = 0
    is_cloud = (
        os.environ.get("GITHUB_ACTIONS") == "true"
        or os.environ.get("CI") == "true"
        or "GITHUB_RUN_ID" in os.environ
    )

    while True:
        key = None
        now = time.time()

        # 1. Identify all registered Gemini keys and find those not on cooldown
        gemini_keys = [k for k in list(key_cooldowns.keys()) if "github" not in k]
        available_keys = [k for k in gemini_keys if now > key_cooldowns[k]]
        response_text = None

        # 2. DECISION LOGIC: Prioritize Gemini rotation
        if available_keys:
            # Cycle through available keys using the attempt counter
            key = available_keys[attempts % len(available_keys)]
            logger.info(
                f"Attempting Gemini Key {attempts % len(available_keys) + 1}/{len(gemini_keys)}"
            )
            response_text = stream_single_llm(prompt, key=key, context=context)

        elif is_cloud:
            # 3. CLOUD FALLBACK: If all Gemini keys are cooling down, use GitHub Models
            if now < key_cooldowns.get("github_llama", 0):
                # If Llama is also cooling, try Phi
                logger.warning("Llama-3 limited. Trying Phi-4...")
                response_text = stream_single_llm(
                    prompt, key=None, context=context, gh_model="Phi-4"
                )
            else:
                logger.warning(
                    "Gemini exhausted. Pivoting to GitHub Models (Llama-3)..."
                )
                response_text = stream_single_llm(
                    prompt, key=None, context=context, gh_model="Llama-3"
                )
        else:
            # 4. LOCAL FALLBACK: Fallback to Ollama
            logger.info(" All Gemini keys exhausted. Falling back to Local Ollama...")
            response_text = stream_single_llm(prompt, key=None, context=context)

        # --- ERROR HANDLING BLOCK ---
        if not response_text or response_text.startswith("ERROR_CODE_"):
            # A. Gemini Rate Limit (429)
            if key and response_text and "429" in response_text:
                key_cooldowns[key] = (
                    time.time() + 180
                )  # 3 min rest for the specific key
                logger.warning(f"Key {key[-4:]} rate-limited. Pivoting to next key...")
                attempts += 1
                continue  # Immediately retry with the next key in the pool

            # B. GitHub Models Daily Quota (429)
            if (
                response_text
                and "429" in response_text
                and "RateLimitReached" in response_text
            ):
                match = re.search(r"wait (\d+) seconds", response_text)
                seconds_to_wait = int(match.group(1)) if match else 86400

                # Assign cooldown to the specific model that failed
                if "Llama" in (response_text or ""):
                    key_cooldowns["github_llama"] = time.time() + seconds_to_wait + 60
                else:
                    key_cooldowns["github_phi"] = time.time() + seconds_to_wait + 60

                logger.error(
                    f"GITHUB QUOTA REACHED. Cooling down model for {seconds_to_wait}s"
                )
                attempts += 1
                continue

            # C. Generic Error Handling / Fail-Safe Sleep
            if not available_keys:
                # If everything is exhausted, take a long nap
                wait = 300
                logger.warning(
                    f"All API resources exhausted. Sleeping {wait}s for refill..."
                )
                time.sleep(wait)
                attempts = 0  # RESET: Start fresh with Key 1 after the nap
                continue
            else:
                # Key failed for unknown reason, rotate and retry
                if key:
                    key_cooldowns[key] = time.time() + 30
                attempts += 1
                time.sleep(2)
                continue

        # --- VALIDATION BLOCK ---
        if validator(response_text):
            if is_cloud:
                time.sleep(
                    10
                )  # Slow down slightly in cloud to prevent 429 machine-gunning
            return response_text
        else:
            # Try cleaning AI chatter (e.g. "Here is the code:") and re-validating
            clean_text = (
                re.sub(
                    r"^(Here is the code:)|(I suggest:)|(```[a-z]*)",
                    "",
                    response_text,
                    flags=re.IGNORECASE,
                )
                .strip()
                .rstrip("`")
            )

            if validator(clean_text):
                return clean_text

            # If still invalid, back off and retry
            wait = 300 if is_cloud else 10
            logger.warning(f"AI response failed validation. Backing off {wait}s...")
            time.sleep(wait)
            attempts += 1
