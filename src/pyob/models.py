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

# --- ENVIRONMENT DETECTION ---
IS_CLOUD = (
    os.environ.get("GITHUB_ACTIONS") == "true"
    or os.environ.get("CI") == "true"
    or "GITHUB_RUN_ID" in os.environ
)


# --- CONFIGURATION LOADER ---
def load_config():
    # If we are in GitHub Actions, NEVER look for the local file.
    # This prevents the bot from accidentally using a committed .pyob_config
    if IS_CLOUD:
        return {}

    config_path = ".pyob_config"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading .pyob_config: {e}")
    return {}


config = load_config()

# --- KEY RESOLUTION ---
# Priority:
# 1. Environment Variables (Highest priority for Cloud/Manual overrides)
# 2. .pyob_config file (For your iMac local use)
# 3. Defaults

OPENROUTER_KEY = os.environ.get("PYOB_OPENROUTER_KEY") or config.get("openrouter_key")
_user_or_model = (
    os.environ.get("PYOB_OPENROUTER_MODEL")
    or config.get("openrouter_model")
    or "deepseek/deepseek-v4-flash:free"
)
OPENROUTER_MODELS = [
    _user_or_model,
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "openrouter/free",
]
OPENROUTER_MODELS = list(dict.fromkeys(OPENROUTER_MODELS))

GEMINI_MODEL = (
    os.environ.get("PYOB_GEMINI_MODEL")
    or config.get("gemini_model")
    or "gemini-3.7-flash"
)
raw_gemini_keys = os.environ.get("PYOB_GEMINI_KEYS") or config.get("gemini_keys") or ""
GEMINI_API_KEYS = [k.strip() for k in raw_gemini_keys.split(",") if k.strip()]

LOCAL_MODEL = (
    os.environ.get("PYOB_LOCAL_MODEL") or config.get("local_model") or "llama3.2:3b"
)


def stream_openrouter(
    prompt: str, key: str, on_chunk: Callable[[], None], model: str
) -> str:
    """Streams response from OpenRouter using a standard requests approach."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/vicsanity623/PyOB.git",  # Optional, good for OpenRouter rankings
        "X-Title": "PyOuroBoros",  # Optional
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }

    last_chunk_time = time.time()
    full_text = ""

    try:
        # Reduced timeout slightly for faster debugging
        response = requests.post(
            url, headers=headers, json=data, stream=True, timeout=60
        )

        # OpenRouter returns 429 if you hit the free tier rate limit
        if response.status_code != 200:
            error_msg = f"ERROR_CODE_{response.status_code}: {response.text}"
            logger.error(f"OpenRouter API Error: {error_msg}")
            return error_msg

        # Reset chunk time so connection latency isn't counted against stream stall
        last_chunk_time = time.time()

        for line in response.iter_lines():
            if not line:
                if time.time() - last_chunk_time > 90:
                    logger.warning("OpenRouter stream stalled. Forcing closure.")
                    break
                continue

            line_str = line.decode("utf-8", errors="ignore").replace("data: ", "")
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
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
        return full_text
    except (requests.RequestException, OSError) as e:
        return f"ERROR_CODE_EXCEPTION: {str(e)}"


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
    last_chunk_time = time.time()

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            if time.time() - last_chunk_time > 30:
                logger.warning("Gemini stream stalled. Forcing closure.")
                break
            continue

        if line.startswith("data: "):
            last_chunk_time = time.time()
            try:
                chunk_data = json.loads(line[6:])
                text = chunk_data["candidates"][0]["content"]["parts"][0]["text"]
                on_chunk()
                response_text += text
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
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
        time.sleep(60)
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
            try:
                content = chunk.get("message", {}).get("content", "")
            except AttributeError:
                content = getattr(getattr(chunk, "message", None), "content", "")

            if content:
                on_chunk()
                print(content, end="", flush=True)
                response_text += content
    except (
        RuntimeError,
        ConnectionError,
        OSError,
        AttributeError,
        requests.RequestException,
    ) as e:
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

            line_str = line.decode("utf-8", errors="ignore").replace("data: ", "")
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
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
        return full_text
    except (requests.RequestException, OSError) as e:
        return f"ERROR_CODE_EXCEPTION: {str(e)}"


def stream_single_llm(
    prompt: str,
    provider: str = "local",  # NEW: Explicitly define provider
    key: Optional[str] = None,
    context: str = "",
    gh_model: str = "Llama-3",
    or_model: Optional[str] = None,
) -> str:
    if or_model is None:
        or_model = OPENROUTER_MODELS[0]
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

            # Setup AI Identity String
            if provider == "openrouter":
                source = f"OpenRouter ({or_model.split('/')[-1]})"
            elif provider == "gemini":
                source = f"Gemini ...{key[-4:]}" if key else "Gemini"
            elif provider == "github":
                source = f"GitHub Models ({gh_model})"
            else:
                source = "Local Ollama"

            print(f"AI Output ({source}): ", end="", flush=True)

    response_text = ""
    try:
        if provider == "openrouter" and key:
            response_text = stream_openrouter(prompt, key, on_chunk, model=or_model)
        elif provider == "gemini" and key:
            response_text = stream_gemini(prompt, key, on_chunk)
        elif provider == "github":
            response_text = stream_github_models(prompt, on_chunk, model_name=gh_model)
            if response_text and "413" in response_text:
                first_chunk_received[0] = True
                logger.warning("\nPayload too large. Sleeping 60s, then pivoting...")
                time.sleep(60)
                # Fail out of github so the outer engine can handle the rotation
                return "ERROR_CODE_413_PAYLOAD_TOO_LARGE"
            if response_text and response_text.startswith("ERROR_CODE_"):
                time.sleep(30)
        else:
            response_text = stream_ollama(prompt, on_chunk)

    except (
        requests.RequestException,
        ConnectionError,
        OSError,
        ValueError,
        RuntimeError,
    ) as e:
        if is_cloud:
            time.sleep(30)
        return f"ERROR_CODE_EXCEPTION: {e}"
    finally:
        first_chunk_received[0] = True
    if response_text and not response_text.startswith("ERROR_CODE_"):
        # Strip DeepSeek / "thinking" model chain-of-thought preamble.
        # These models sometimes bleed raw reasoning text before the actual XML
        # answer. If we find a recognised XML opener, discard everything before it.
        xml_openers = [
            "<THOUGHT>",
            "<SNIPPET>",
            "<EDIT>",
            "<thought>",
            "<snippet>",
            "<edit>",
        ]
        first_tag_pos = -1
        for tag in xml_openers:
            pos = response_text.find(tag)
            if pos != -1 and (first_tag_pos == -1 or pos < first_tag_pos):
                first_tag_pos = pos
        if first_tag_pos > 0:
            stripped = response_text[first_tag_pos:]
            # Only strip if the preamble is suspiciously long (> 200 chars of non-XML text)
            if first_tag_pos > 200:
                logger.warning(
                    f"Stripping {first_tag_pos} chars of thinking-model preamble before XML."
                )
                response_text = stripped

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
    Robust engine that handles hierarchy: OpenRouter -> Gemini -> GitHub -> Ollama.
    """
    attempts = 0
    is_cloud = (
        os.environ.get("GITHUB_ACTIONS") == "true"
        or os.environ.get("CI") == "true"
        or "GITHUB_RUN_ID" in os.environ
    )

    while True:
        key = None
        provider = "local"
        now = time.time()

        # 1. Evaluate key availabilities
        global_or_cooldown = key_cooldowns.get("openrouter", 0)
        openrouter_available = OPENROUTER_KEY and now > global_or_cooldown

        available_or_models = []
        if openrouter_available:
            available_or_models = [
                m
                for m in OPENROUTER_MODELS
                if now > key_cooldowns.get(f"openrouter_{m}", 0)
            ]

        gemini_keys = [
            k
            for k in list(key_cooldowns.keys())
            if "github" not in k and not k.startswith("openrouter")
        ]
        # Add env keys to pool if not present in cooldowns yet
        env_gem_keys = [k.strip() for k in raw_gemini_keys.split(",") if k.strip()]
        for ek in env_gem_keys:
            if ek not in key_cooldowns:
                key_cooldowns[ek] = 0
                gemini_keys.append(ek)

        available_gemini_keys = [k for k in gemini_keys if now > key_cooldowns[k]]
        response_text = None

        # 2. DECISION LOGIC: Prioritize Gemini > OpenRouter > GitHub > Ollama
        if available_gemini_keys:
            key = available_gemini_keys[attempts % len(available_gemini_keys)]
            provider = "gemini"
            logger.info(
                f"Attempting Gemini Key {attempts % len(available_gemini_keys) + 1}/{len(available_gemini_keys)}"
            )
            response_text = stream_single_llm(
                prompt, provider=provider, key=key, context=context
            )

        elif available_or_models:
            current_or_model = available_or_models[attempts % len(available_or_models)]
            logger.info(f"Attempting OpenRouter ({current_or_model})...")
            key = OPENROUTER_KEY
            provider = "openrouter"
            response_text = stream_single_llm(
                prompt,
                provider=provider,
                key=key,
                context=context,
                or_model=current_or_model,
            )

        elif is_cloud:
            provider = "github"
            if now < key_cooldowns.get("github_llama", 0):
                logger.warning("Llama-3 limited. Trying Phi-4...")
                response_text = stream_single_llm(
                    prompt,
                    provider=provider,
                    key=None,
                    context=context,
                    gh_model="Phi-4",
                )
            else:
                logger.warning(
                    "Primary APIs exhausted. Pivoting to GitHub Models (Llama-3)..."
                )
                response_text = stream_single_llm(
                    prompt,
                    provider=provider,
                    key=None,
                    context=context,
                    gh_model="Llama-3",
                )
        else:
            provider = "local"
            logger.info("All Cloud APIs exhausted. Falling back to Local Ollama...")
            response_text = stream_single_llm(
                prompt, provider=provider, key=None, context=context
            )

        # --- ERROR HANDLING BLOCK ---
        if not response_text or response_text.startswith("ERROR_CODE_"):
            # A. OpenRouter Rate Limit (429)
            if provider == "openrouter" and "429" in (response_text or ""):
                key_cooldowns[f"openrouter_{current_or_model}"] = (
                    time.time() + 60
                )  # 1 minute rest for this openrouter model
                logger.warning(
                    f"OpenRouter ({current_or_model}) rate-limited. Pivoting..."
                )
                attempts += 1
                continue

            # B. Gemini Rate Limit (429)
            if provider == "gemini" and key and "429" in (response_text or ""):
                key_cooldowns[key] = (
                    time.time() + 180
                )  # 3 min rest for specific Gemini key
                logger.warning(f"Gemini Key {key[-4:]} rate-limited. Rotating...")
                attempts += 1
                continue

            # C. GitHub Models Daily Quota (429)
            if (
                provider == "github"
                and "429" in (response_text or "")
                and "RateLimitReached" in (response_text or "")
            ):
                match = re.search(r"wait (\d+) seconds", response_text)
                seconds_to_wait = int(match.group(1)) if match else 86400
                if "Llama" in (response_text or ""):
                    key_cooldowns["github_llama"] = time.time() + seconds_to_wait + 60
                else:
                    key_cooldowns["github_phi"] = time.time() + seconds_to_wait + 60
                logger.error(
                    f"GITHUB QUOTA REACHED. Cooling down model for {seconds_to_wait}s"
                )
                attempts += 1
                continue

            # D. Generic Error Handling / Fail-Safe Sleep
            if not available_or_models and not available_gemini_keys and not is_cloud:
                wait = 300
                logger.warning(
                    f"All API resources exhausted. Sleeping {wait}s for refill..."
                )
                time.sleep(wait)
                attempts = 0
                continue
            else:
                # Key failed for unknown reason
                if provider == "openrouter":
                    if "401" in (response_text or ""):
                        logger.warning(
                            f"OpenRouter API Key invalid ({response_text}). Disabling OpenRouter..."
                        )
                        key_cooldowns["openrouter"] = time.time() + 86400
                    else:
                        logger.warning(
                            f"OpenRouter failed ({response_text}). Pivoting to fallback..."
                        )
                        key_cooldowns[f"openrouter_{current_or_model}"] = (
                            time.time() + 60
                        )
                elif provider == "gemini" and key:
                    logger.warning(f"Gemini Key {key[-4:]} failed. Rotating...")
                    key_cooldowns[key] = time.time() + 180
                elif key:
                    key_cooldowns[key] = time.time() + 30

                attempts += 1
                time.sleep(2)
                continue

        # --- VALIDATION BLOCK ---
        if validator(response_text):
            if is_cloud:
                time.sleep(10)
            return response_text
        else:
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

            wait = 300 if is_cloud else 10
            logger.error(f"Validation Failed! The AI said: '{response_text}'")
            logger.warning(f"AI response failed validation. Backing off {wait}s...")
            time.sleep(wait)
            attempts += 1
