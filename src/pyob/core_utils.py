import ast
import logging
import os
import re
import select
import shutil
import subprocess
import sys
import termios
import textwrap
import time
import tty
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
    ".pyob",
    "autovenv",
    "build_env",
    "pyob.egg-info",
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
}

IGNORE_FILES = {
    "package-lock.json",
    "LICENSE",
    "action.yml",
    "Dockerfile",
    "PEER_REVIEW.md",
    "FEATURE.md",
    "FAILED_PEER_REVIEW.md",
    "FAILED_FEATURE.md",
    "MEMORY.md",
    "ANALYSIS.md",
    "HISTORY.md",
    "SYMBOLS.json",
    "UM.md",
    "RM.md",
    "PP.md",
    "ALF.md",
    "FRE.md",
    "PF.md",
    "IF.md",
    "PCF.md",
    "PIR.md",
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
    memory_file: str
    key_cooldowns: dict[str, float]

    def stream_gemini(
        self, prompt: str, api_key: str, on_chunk: Callable[[], None]
    ) -> str:
        return str(stream_gemini(prompt, api_key, on_chunk))

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

            while True:
                remaining = int(timeout - (time.time() - start_time))
                if remaining <= 0:
                    return "PROCEED"
                sys.stdout.write(
                    f"\r {remaining}s remaining | You: {input_str} \b\b\b\b"
                )
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

    def _launch_external_code_editor(
        self, initial_content: str, file_suffix: str = ".py"
    ) -> str:
        import tempfile

        editor = os.environ.get("EDITOR", "nano")
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, encoding="utf-8", suffix=file_suffix
        ) as tmp_file:
            tmp_file.write(initial_content)
            tmp_file_path = tmp_file.name
        logger.info(f"Opening prompt augmentation editor: {editor} {tmp_file_path}")
        logger.info(
            "Add your additional instructions/context. Save and close the file to continue."
        )
        try:
            subprocess.run([editor, tmp_file_path], check=True)
            with open(tmp_file_path, "r", encoding="utf-8") as f:
                edited_content = f.read()
            return edited_content
        except FileNotFoundError:
            logger.error(f"Editor '{editor}' not found. Using original content.")
            return initial_content
        except subprocess.CalledProcessError:
            logger.error(
                f"Editor '{editor}' exited with an error. Using original content."
            )
            return initial_content
        finally:
            os.remove(tmp_file_path)

    def _edit_prompt_with_external_editor(self, initial_prompt: str) -> str:
        import tempfile

        editor = os.environ.get("EDITOR", "nano")
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, encoding="utf-8"
        ) as tmp_file:
            tmp_file.write(initial_prompt)
            tmp_file_path = tmp_file.name
        logger.info(f"Opening prompt in editor: {editor} {tmp_file_path}")
        logger.info(
            "Save and close the file to continue. (e.g., Ctrl+X, then Y, then Enter for nano)"
        )
        try:
            subprocess.run([editor, tmp_file_path], check=True)
            with open(tmp_file_path, "r", encoding="utf-8") as f:
                edited_prompt = f.read()
            return edited_prompt
        except FileNotFoundError:
            logger.error(f"Editor '{editor}' not found. Using original prompt.")
            return initial_prompt
        except subprocess.CalledProcessError:
            logger.error(
                f"Editor '{editor}' exited with an error. Using original prompt."
            )
            return initial_prompt
        finally:
            os.remove(tmp_file_path)

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
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                pass
        return ""

    def get_valid_llm_response(
        self, prompt: str, validator: Callable[[str], bool], context: str = ""
    ) -> str:
        return str(
            get_valid_llm_response_engine(
                prompt, validator, self.key_cooldowns, context
            )
        )

    def _get_user_prompt_augmentation(self, initial_text: str = "") -> str:
        import tempfile

        editor = os.environ.get("EDITOR", "nano")
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, encoding="utf-8", suffix=".txt"
        ) as tmp_file:
            tmp_file.write(initial_text)
            tmp_file_path = tmp_file.name
        logger.info(f"Opening prompt augmentation editor: {editor}")
        try:
            subprocess.run([editor, tmp_file_path], check=True)
            with open(tmp_file_path, "r", encoding="utf-8") as f:
                edited_content = f.read()
            return edited_content
        except Exception:
            return initial_text
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    def ensure_imports_retained(
        self, orig_code: str, new_code: str, filepath: str
    ) -> str:
        try:
            orig_tree = ast.parse(orig_code)
            new_tree = ast.parse(new_code)
        except Exception:
            return new_code
        orig_imports = []
        for node in orig_tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                start_line = node.lineno - 1
                end_line = getattr(node, "end_lineno", node.lineno)
                import_text = "\n".join(orig_code.splitlines()[start_line:end_line])
                orig_imports.append((node, import_text))
        missing_imports = []
        for orig_node, import_text in orig_imports:
            found = False
            for new_node in new_tree.body:
                if isinstance(new_node, type(orig_node)):
                    if isinstance(orig_node, ast.Import) and isinstance(
                        new_node, ast.Import
                    ):
                        if {alias.name for alias in orig_node.names}.issubset(
                            {alias.name for alias in new_node.names}
                        ):
                            found = True
                            break
                    elif isinstance(orig_node, ast.ImportFrom) and isinstance(
                        new_node, ast.ImportFrom
                    ):
                        if orig_node.module == new_node.module and {
                            alias.name for alias in orig_node.names
                        }.issubset({alias.name for alias in new_node.names}):
                            found = True
                            break
            if not found:
                missing_imports.append(import_text)
        if missing_imports:
            return "\n".join(missing_imports) + "\n\n" + new_code
        return new_code

    def apply_xml_edits(
        self, source_code: str, llm_response: str
    ) -> tuple[str, str, bool]:
        source_code = source_code.replace("\r\n", "\n")
        llm_response = llm_response.replace("\r\n", "\n")
        thought_match = re.search(
            r"<THOUGHT>(.*?)</THOUGHT>", llm_response, re.DOTALL | re.IGNORECASE
        )
        explanation = (
            thought_match.group(1).strip()
            if thought_match
            else "No explanation provided."
        )
        new_code = source_code
        pattern = re.compile(
            r"<EDIT>\s*<SEARCH>\s*\n?(.*?)\n?\s*</SEARCH>\s*<REPLACE>\s*\n?(.*?)\n?\s*</REPLACE>\s*</EDIT>",
            re.DOTALL | re.IGNORECASE,
        )
        matches = list(pattern.finditer(llm_response))
        if not matches:
            return new_code, explanation, True
        all_edits_succeeded = True
        for m in matches:
            raw_search = re.sub(
                r"^```[\w]*\n|\n```$", "", m.group(1), flags=re.MULTILINE
            )
            raw_replace = re.sub(
                r"^```[\w]*\n|\n```$", "", m.group(2), flags=re.MULTILINE
            )
            search_lines = raw_search.split("\n")
            replace_lines = raw_replace.split("\n")
            search_indent = ""
            for line in search_lines:
                if line.strip():
                    search_indent = line[: len(line) - len(line.lstrip(" \t"))]
                    break
            replace_base_indent = ""
            for line in replace_lines:
                if line.strip():
                    replace_base_indent = line[: len(line) - len(line.lstrip(" \t"))]
                    break
            fixed_replace_lines = []
            for line in replace_lines:
                if line.strip():
                    if line.startswith(replace_base_indent):
                        clean_line = line[len(replace_base_indent) :]
                    else:
                        clean_line = line.lstrip(" \t")
                    fixed_replace_lines.append(search_indent + clean_line)
                else:
                    fixed_replace_lines.append("")
            raw_replace = "\n".join(fixed_replace_lines)
            block_applied = False
            if raw_search in new_code:
                new_code = new_code.replace(raw_search, raw_replace, 1)
                block_applied = True
            if not block_applied:
                clean_search = raw_search.strip("\n")
                clean_replace = raw_replace.strip("\n")
                if clean_search and clean_search in new_code:
                    new_code = new_code.replace(clean_search, clean_replace, 1)
                    block_applied = True
            if not block_applied:

                def normalize(t):
                    t = re.sub(r"#.*", "", t)
                    return re.sub(r"\s+", " ", t).strip()

                norm_search = normalize(raw_search)
                if norm_search:
                    lines = new_code.splitlines()
                    for i in range(len(lines)):
                        test_block = normalize(
                            "\n".join(lines[i : i + len(search_lines)])
                        )
                        if norm_search in test_block:
                            lines[i : i + len(search_lines)] = [raw_replace]
                            new_code = "\n".join(lines)
                            block_applied = True
                            logger.info(
                                f"Normalization match succeeded at line {i + 1}."
                            )
                            break
            if not block_applied:
                try:
                    search_lines_cleaned = [
                        line.strip()
                        for line in clean_search.split("\n")
                        if line.strip()
                    ]
                    if search_lines_cleaned:
                        regex_parts = [
                            r"^[ \t]*" + re.escape(line) + r"[ \t]*\n+"
                            for line in search_lines_cleaned[:-1]
                        ]
                        regex_parts.append(
                            r"^[ \t]*"
                            + re.escape(search_lines_cleaned[-1])
                            + r"[ \t]*\n?"
                        )
                        pattern_str = r"".join(regex_parts)
                        fuzzy_match = re.search(pattern_str, new_code, re.MULTILINE)
                        if fuzzy_match:
                            new_code = (
                                new_code[: fuzzy_match.start()]
                                + raw_replace
                                + "\n"
                                + new_code[fuzzy_match.end() :]
                            )
                            block_applied = True
                except Exception:
                    pass
            if not block_applied:
                search_lines_stripped = [
                    line.strip() for line in search_lines if line.strip()
                ]
                if search_lines_stripped:
                    code_lines = new_code.splitlines()
                    for i in range(len(code_lines) - len(search_lines_stripped) + 1):
                        match = True
                        for j, sline in enumerate(search_lines_stripped):
                            if sline not in code_lines[i + j].strip():
                                match = False
                                break
                        if match:
                            new_code_lines = (
                                code_lines[:i]
                                + replace_lines
                                + code_lines[i + len(search_lines) :]
                            )
                            new_code = "\n".join(new_code_lines)
                            if not new_code.endswith("\n") and source_code.endswith(
                                "\n"
                            ):
                                new_code += "\n"
                            block_applied = True
                            logger.info(
                                f"Robust fuzzy match succeeded at line {i + 1}."
                            )
                            break
            if not block_applied:
                all_edits_succeeded = False
        return new_code, explanation, all_edits_succeeded

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

        return html_fallback
