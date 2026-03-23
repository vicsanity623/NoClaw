import json
import logging
import os
import re
import select
import shutil
import subprocess
import sys
import textwrap
import time
from typing import Callable, Optional

from .models import (
    get_valid_llm_response_engine,
    stream_gemini,
    stream_github_models,
    stream_ollama,
    stream_single_llm,
)

env_keys = os.environ.get("PYOB_GEMINI_KEYS", "")
GEMINI_API_KEYS = [k.strip() for k in env_keys.split(",") if k.strip()]
GEMINI_MODEL = os.environ.get("PYOB_GEMINI_MODEL", "gemini-2.5-flash")
LOCAL_MODEL = os.environ.get("PYOB_LOCAL_MODEL", "qwen3-coder:30b")
PR_FILE_NAME = "PEER_REVIEW.md"
FEATURE_FILE_NAME = "FEATURE.md"
FAILED_PR_FILE_NAME = "FAILED_PEER_REVIEW.md"
FAILED_FEATURE_FILE_NAME = "FAILED_FEATURE.md"
MEMORY_FILE_NAME = "MEMORY.md"
ANALYSIS_FILE = "ANALYSIS.md"
HISTORY_FILE = "HISTORY.md"
SYMBOLS_FILE = "SYMBOLS.json"
PYOB_DATA_DIR = ".pyob"

IGNORE_DIRS = {
    ".git",
    ".github",
    # ".pyob",
    "autovenv",
    "build_env",
    "pyob.egg-info",
    "TapEvent",
    "build",
    "dist",
    "docs",
    "venv",
    ".venv",
    "code",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "patch_test",
    "env",
    "__pycache__",
    "node_modules",
    ".vscode",
    ".idea",
    "tests",
}

IGNORE_FILES = {
    "package-lock.json",
    "LICENSE",
    "manifest.json",
    "action.yml",
    "Dockerfile",
    "build_pyinstaller_multiOS.py",
    "check.sh",
    ".pyob_config",
    ".DS_Store",
    ".gitignore",
    "pyob.icns",
    "pyob.ico",
    "pyob.png",
    "ROADMAP.md",
    "README.md",
    "DOCUMENTATION.md",
    "observer.html",
}
SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".html", ".css", ".json", ".sh"}


class CyberpunkFormatter(logging.Formatter):
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        cols, _ = shutil.get_terminal_size((80, 20))
        color = self.RESET
        if record.levelno == logging.INFO:
            color = self.GREEN
        elif record.levelno == logging.WARNING:
            color = self.YELLOW
        elif record.levelno >= logging.ERROR:
            color = self.RED

        prefix = f"{time.strftime('%H:%M:%S')} | "
        available_width = max(cols - len(prefix) - 1, 20)
        message = record.getMessage()
        wrapped_lines = textwrap.wrap(message, width=available_width)

        formatted_msg = ""
        for i, line in enumerate(wrapped_lines):
            if i == 0:
                formatted_msg += (
                    f"{self.BLUE}{prefix}{self.RESET}{color}{line}{self.RESET}"
                )
            else:
                formatted_msg += f"\n{' ' * len(prefix)}{color}{line}{self.RESET}"
        return formatted_msg


logger = logging.getLogger("PyOuroBoros")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(CyberpunkFormatter())
logger.addHandler(handler)
logger.propagate = False


class CoreUtilsMixin:
    target_dir: str
    memory_path: str
    key_cooldowns: dict[str, float]

    def generate_pr_summary(self, rel_path: str, diff_text: str) -> dict:
        """Analyzes a git diff and returns a professional title and body for the PR."""
        prompt = f"""
        Analyze the following git diff for file `{rel_path}` and write a professional, high-quality PR title and description.
        RULES:
        1. PR Title: Start with a category (e.g., "Refactor:", "Feature:", "Fix:", "Security:") followed by a concise summary.
        2. PR Body: Use professional markdown. Include sections for 'Summary of Changes' and 'Technical Impact'.
        3. NO TIMESTAMPS: Do not mention the time or date.
        GIT DIFF:
        {diff_text}
        OUTPUT FORMAT (STRICT JSON):
        {{"title": "...", "body": "..."}}
        """

        try:
            response = self.get_valid_llm_response(
                prompt,
                lambda t: '"title":' in t and '"body":' in t,
                context="PR Architect",
            )

            json_match = re.search(r"(\{.*\})", response, re.DOTALL)
            if json_match:
                clean_json = json_match.group(1)
            else:
                clean_json = re.sub(
                    r"^```json\s*|\s*```$", "", response.strip(), flags=re.MULTILINE
                )

            data = json.loads(clean_json, strict=False)

            if isinstance(data, dict):
                return data

            raise ValueError("LLM response was not a valid dictionary object")

        except Exception as e:
            logger.warning(f"Librarian failed to generate AI summary: {e}")
            return {
                "title": f"Evolution: Refactor of `{rel_path}`",
                "body": f"Automated self-evolution update for `{rel_path}`. Verified stable via runtime testing.",
            }

    def stream_gemini(
        self, prompt: str, api_key: str, on_chunk: Callable[[], None]
    ) -> str:
        return stream_gemini(prompt, api_key, on_chunk)

    def stream_ollama(self, prompt: str, on_chunk: Callable[[], None]) -> str:
        return str(stream_ollama(prompt, on_chunk))

    def stream_github_models(
        self, prompt: str, on_chunk: Callable[[], None], model_name: str = "Llama-3"
    ) -> str:
        return str(stream_github_models(prompt, on_chunk, model_name))

    def _stream_single_llm(
        self,
        prompt: str,
        key: Optional[str] = None,
        context: str = "",
        gh_model: str = "Llama-3",
    ) -> str:
        return str(stream_single_llm(prompt, key, context, gh_model))

    def get_user_approval(self, prompt_text: str, timeout: int = 220) -> str:
        if (
            not sys.stdin.isatty()
            or os.environ.get("GITHUB_ACTIONS") == "true"
            or os.environ.get("CI") == "true"
            or "GITHUB_RUN_ID" in os.environ
        ):
            logger.info(" Headless environment detected: Auto-approving action.")
            return "PROCEED"
        print(f"\n{prompt_text}")
        start_time = time.time()
        input_str = ""
        if sys.platform == "win32":
            import msvcrt

            prev_line_len = 0
            while True:
                remaining = int(timeout - (time.time() - start_time))
                if remaining <= 0:
                    return "PROCEED"
                current_display_str = f" {remaining}s remaining | You: {input_str}"
                padding_needed = max(0, prev_line_len - len(current_display_str))
                sys.stdout.write(f"\r{current_display_str}{' ' * padding_needed}")
                prev_line_len = len(current_display_str) + padding_needed
                sys.stdout.flush()
                if msvcrt.kbhit():
                    char = msvcrt.getwch()
                    if char in ("\r", "\n"):
                        print()
                        val = input_str.strip().upper()
                        return val if val else "PROCEED"
                    elif char == "\x08":
                        input_str = input_str[:-1]
                    else:
                        input_str += char
                time.sleep(0.1)
        else:
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while True:
                    remaining = int(timeout - (time.time() - start_time))
                    if remaining <= 0:
                        return "PROCEED"
                    sys.stdout.write(
                        f"\r {remaining}s remaining | You: {input_str}\033[K"
                    )
                    sys.stdout.flush()
                    i, o, e = select.select([sys.stdin], [], [], 0.1)
                    if i:
                        char = sys.stdin.read(1)
                        if char in ("\n", "\r"):
                            print()
                            val = input_str.strip().upper()
                            return val if val else "PROCEED"
                        elif char in ("\x08", "\x7f"):
                            input_str = input_str[:-1]
                        else:
                            input_str += char
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _open_editor_for_content(
        self,
        initial_content: str,
        file_suffix: str = ".txt",
        log_message: str = "Opening editor",
        error_message: str = "Using original content.",
    ) -> str:
        import tempfile

        editor = os.environ.get("EDITOR", "nano")
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, encoding="utf-8", suffix=file_suffix
        ) as tmp_file:
            tmp_file.write(initial_content)
            tmp_file_path = tmp_file.name
        logger.info(f"{log_message}: {editor} {tmp_file_path}")
        try:
            subprocess.run([editor, tmp_file_path], check=True)
            with open(tmp_file_path, "r", encoding="utf-8") as f:
                edited_content = f.read()
            return edited_content
        except FileNotFoundError:
            logger.error(f"Editor '{editor}' not found. {error_message}")
            return initial_content
        except subprocess.CalledProcessError:
            logger.error(f"Editor '{editor}' exited with an error. {error_message}")
            return initial_content
        except Exception:
            logger.error(f"An unexpected error occurred with editor. {error_message}")
            return initial_content
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    def _launch_external_code_editor(
        self, initial_content: str, file_suffix: str = ".py"
    ) -> str:
        return self._open_editor_for_content(
            initial_content,
            file_suffix,
            log_message="Opening prompt augmentation editor",
            error_message="Using original content.",
        )

    def _edit_prompt_with_external_editor(self, initial_prompt: str) -> str:
        return self._open_editor_for_content(
            initial_prompt,
            log_message="Opening prompt in editor",
            error_message="Using original prompt.",
        )

    def backup_workspace(self) -> dict[str, str]:
        state: dict[str, str] = {}
        for root, dirs, files in os.walk(self.target_dir):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

            for file in files:
                if file in IGNORE_FILES:
                    continue

                if any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    path = os.path.join(root, file)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            state[path] = f.read()
                    except Exception:
                        pass
        return state

    def restore_workspace(self, state: dict[str, str]):
        for path, content in state.items():
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                logger.error(f"Failed to restore {path}: {e}")
        logger.warning("Workspace restored to safety due to unfixable AI errors.")

    def load_memory(self) -> str:
        """Loads persistent memory and injects repo-level human directives."""
        memory_content = ""

        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    memory_content = f.read().strip()
            except Exception:
                pass

        directives_path = os.path.join(self.target_dir, "DIRECTIVES.md")
        if os.path.exists(directives_path):
            try:
                with open(directives_path, "r", encoding="utf-8") as f:
                    human_orders = f.read().strip()
                    if human_orders:
                        logger.info(
                            f"🎯 BEACON ACTIVE: Injected {len(human_orders.splitlines())} lines from DIRECTIVES.md"
                        )

                        memory_content = (
                            f"# CRITICAL HUMAN DIRECTIVES - PRIORITY 1\n"
                            f"{human_orders}\n"
                            f"\n# END CRITICAL DIRECTIVES\n"
                            f"---\n{memory_content}"
                        )
            except Exception as e:
                logger.warning(f"Could not read DIRECTIVES.md: {e}")

        return memory_content

    def get_valid_llm_response(
        self, prompt: str, validator: Callable[[str], bool], context: str = ""
    ) -> str:
        """Wrapper that ensures key rotation is used for all requests."""
        return str(
            get_valid_llm_response_engine(
                prompt, validator, self.key_cooldowns, context
            )
        )

    def _find_entry_file(self) -> str | None:
        priority_files = [
            "entrance.py",
            "main.py",
            "app.py",
            "gui.py",
            "pyob_launcher.py",
            "package.json",
            "server.js",
            "app.js",
            "index.html",
        ]

        for f_name in priority_files:
            target = os.path.join(self.target_dir, f_name)
            if os.path.exists(target):
                if (
                    f_name == "package.json"
                    or f_name.endswith(".html")
                    or f_name.endswith(".htm")
                ):
                    return target

                try:
                    with open(target, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if target.endswith(".py"):
                            if (
                                'if __name__ == "__main__":' in content
                                or "if __name__ == '__main__':" in content
                            ):
                                return target
                        if target.endswith(".js") and len(content.strip()) > 10:
                            return target
                except Exception:
                    continue

        html_fallback = None
        for root, dirs, files in os.walk(self.target_dir):
            dirs[:] = [
                d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")
            ]

            for file in files:
                if file in IGNORE_FILES:
                    continue

                file_path = os.path.join(root, file)

                if file.endswith(".py"):
                    try:
                        with open(
                            file_path, "r", encoding="utf-8", errors="ignore"
                        ) as f_obj:
                            content = f_obj.read()
                            if (
                                'if __name__ == "__main__":' in content
                                or "if __name__ == '__main__':" in content
                            ):
                                return file_path
                    except Exception:
                        continue

                if file.endswith(".html") and not html_fallback:
                    html_fallback = file_path

        python_entry_points = []
        other_script_entry_points = []
        html_entry_points = []

        for root, dirs, files in os.walk(self.target_dir):
            dirs[:] = [
                d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")
            ]

            for file in files:
                if file in IGNORE_FILES:
                    continue

                file_path = os.path.join(root, file)

                if file.endswith(".py"):
                    try:
                        with open(
                            file_path, "r", encoding="utf-8", errors="ignore"
                        ) as f_obj:
                            content = f_obj.read()
                            if (
                                'if __name__ == "__main__":' in content
                                or "if __name__ == '__main__':" in content
                            ):
                                python_entry_points.append(file_path)
                    except Exception:
                        continue
                elif file.endswith((".js", ".ts", ".sh")):
                    other_script_entry_points.append(file_path)
                elif file.endswith((".html", ".htm")):
                    html_entry_points.append(file_path)
                elif file == "package.json":
                    html_entry_points.append(
                        file_path
                    )  # Treat package.json as a fallback similar to HTML

        if python_entry_points:
            # Prioritize common names if multiple python entry points are found
            for p_file in priority_files:
                for entry in python_entry_points:
                    if entry.endswith(p_file):
                        return entry
            return python_entry_points[
                0
            ]  # Fallback to first found if no priority match

        if other_script_entry_points:
            # Prioritize common names if multiple script entry points are found
            for p_file in priority_files:
                for entry in other_script_entry_points:
                    if entry.endswith(p_file):
                        return entry
            return other_script_entry_points[0]

        if html_entry_points:
            # Prioritize common names if multiple html/package.json entry points are found
            for p_file in priority_files:
                for entry in html_entry_points:
                    if entry.endswith(p_file):
                        return entry
            return html_entry_points[0]

        return None
