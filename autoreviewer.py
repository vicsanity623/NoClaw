import os
import re
import difflib
import ast
import subprocess
import random
import sys
import time
import shutil

from core_utils import (
    logger,
    IGNORE_DIRS,
    IGNORE_FILES,
    SUPPORTED_EXTENSIONS,
    CoreUtilsMixin,
    GEMINI_API_KEYS,
    PR_FILE_NAME,
    FEATURE_FILE_NAME,
    FAILED_PR_FILE_NAME,
    FAILED_FEATURE_FILE_NAME,
    MEMORY_FILE_NAME,
    ANALYSIS_FILE,
    HISTORY_FILE,
    SYMBOLS_FILE,
)

from prompts_and_memory import PromptsAndMemoryMixin


class AutoReviewer(CoreUtilsMixin, PromptsAndMemoryMixin):
    # FIXED: Added explicit type annotation for the shared class variable
    _shared_cooldowns: dict[str, float] | None = None

    def __init__(self, target_dir: str):
        self.target_dir = os.path.abspath(target_dir)
        self.pr_file = os.path.join(self.target_dir, PR_FILE_NAME)
        self.feature_file = os.path.join(self.target_dir, FEATURE_FILE_NAME)
        self.failed_pr_file = os.path.join(self.target_dir, FAILED_PR_FILE_NAME)
        self.failed_feature_file = os.path.join(
            self.target_dir, FAILED_FEATURE_FILE_NAME
        )
        self.memory_file = os.path.join(self.target_dir, MEMORY_FILE_NAME)
        self.analysis_path = os.path.join(self.target_dir, ANALYSIS_FILE)
        self.history_path = os.path.join(self.target_dir, HISTORY_FILE)
        self.symbols_path = os.path.join(self.target_dir, SYMBOLS_FILE)
        self.memory = self.load_memory()
        self.session_context: list[str] = []
        self._ensure_prompt_files()

        if AutoReviewer._shared_cooldowns is None:
            # FIXED: Used 0.0 (float) instead of 0 (int) to match type dict[str, float]
            AutoReviewer._shared_cooldowns = {
                key: 0.0 for key in GEMINI_API_KEYS if key.strip()
            }

        self.key_cooldowns = AutoReviewer._shared_cooldowns

    def get_language_info(self, filepath: str) -> tuple[str, str]:
        ext = os.path.splitext(filepath)[1].lower()
        mapping = {
            ".py": ("Python", "python"),
            ".js": ("JavaScript", "javascript"),
            ".ts": ("TypeScript", "typescript"),
            ".html": ("HTML", "html"),
            ".css": ("CSS", "css"),
            ".json": ("JSON", "json"),
            ".sh": ("Bash", "bash"),
            ".md": ("Markdown", "markdown"),
        }
        return mapping.get(ext, ("Code", ""))

    def scan_for_lazy_code(self, filepath: str, content: str) -> list[str]:
        issues = []
        lines = content.splitlines()

        if len(lines) > 800:
            issues.append(
                f"Architectural Bloat: File has {len(lines)} lines. This exceeds the 800-line modularity threshold. Priority: HIGH. Action: Split into smaller modules."
            )

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return [f"SyntaxError during AST parse: {e}"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "Any":
                issues.append("Found use of 'Any' type hint.")
            elif isinstance(node, ast.Attribute) and node.attr == "Any":
                issues.append("Found use of 'typing.Any'.")
        return issues

    def run_linters(self, filepath: str) -> tuple[str, str]:
        ruff_out, mypy_out = "", ""
        try:
            ruff_out = subprocess.run(
                ["ruff", "check", filepath], capture_output=True, text=True
            ).stdout.strip()
        except FileNotFoundError:
            pass
        try:
            res = subprocess.run(["mypy", filepath], capture_output=True, text=True)
            if res.returncode != 0:
                mypy_out = res.stdout.strip()
        except FileNotFoundError:
            pass
        return ruff_out, mypy_out

    def build_patch_prompt(
        self,
        lang_name: str,
        lang_tag: str,
        content: str,
        ruff_out: str,
        mypy_out: str,
        custom_issues: list[str],
    ) -> str:
        memory_section = self._get_rich_context()
        ruff_section = f"### Ruff Errors:\n{ruff_out}\n\n" if ruff_out else ""
        mypy_section = f"### Mypy Errors:\n{mypy_out}\n\n" if mypy_out else ""
        custom_issues_section = (
            "### Code Quality Issues:\n"
            + "\n".join(f"- {i}" for i in custom_issues)
            + "\n\n"
            if custom_issues
            else ""
        )
        return self.load_prompt(
            "PP.md",
            lang_name=lang_name,
            lang_tag=lang_tag,
            content=content,
            memory_section=memory_section,
            ruff_section=ruff_section,
            mypy_section=mypy_section,
            custom_issues_section=custom_issues_section,
        )

    def get_valid_edit(
        self,
        prompt: str,
        source_code: str,
        require_edit: bool = True,
        target_filepath: str = "",
    ) -> tuple[str, str, str]:
        if target_filepath:
            display_name = os.path.relpath(target_filepath, self.target_dir)
        else:
            display_name = "System Update"
        print("\n" + "=" * 50)
        print(f"💡 AI Generation Prompt Ready: [{display_name}]")
        print("=" * 50)
        print(
            f"The AI has prepared a prompt for code generation/review of: {display_name}"
        )
        user_choice_pre_llm = self.get_user_approval(
            "Hit ENTER to send as-is, type 'EDIT_PROMPT' to refine the full prompt, 'AUGMENT_PROMPT' to add quick instructions, or 'SKIP' to cancel.",
            timeout=220,
        )
        if user_choice_pre_llm == "SKIP":
            logger.info("AI generation skipped by user.")
            return source_code, "AI generation skipped by user.", ""
        elif user_choice_pre_llm == "EDIT_PROMPT":
            logger.info("Opening full prompt in editor for manual refinement...")
            prompt = self._edit_prompt_with_external_editor(prompt)
            if not prompt.strip():
                logger.warning("Edited prompt is empty. Skipping AI generation.")
                return source_code, "Edited prompt is empty.", ""
        elif user_choice_pre_llm == "AUGMENT_PROMPT":
            logger.info("Opening augmentation editor to add quick instructions...")
            augmentation_text = self._get_user_prompt_augmentation()
            if augmentation_text.strip():
                prompt += f"\n\n### User Augmentation:\n{augmentation_text.strip()}"
                logger.info("Prompt augmented with user input.")
            else:
                logger.info("No augmentation provided.")

        attempts: int = int(0)
        use_ollama = False
        while True:
            key = None
            now = time.time()
            available_keys = [
                k for k, cooldown in self.key_cooldowns.items() if now > cooldown
            ]
            if not available_keys:
                if not use_ollama:
                    logger.warning(
                        "🚫 All Gemini keys are currently rate-limited. Falling back to Local Ollama."
                    )
                    use_ollama = True
            else:
                use_ollama = False
                key = available_keys[attempts % len(available_keys)]
                logger.info(
                    f"\n[Attempting Gemini API Key {attempts % len(available_keys) + 1}/{len(available_keys)} Available]"
                )
            if use_ollama:
                logger.info("\n[Attempting Local Ollama]")
            response_text = self._stream_single_llm(
                prompt, key=key, context=display_name
            )
            if response_text.startswith("ERROR_CODE_429"):
                if key:
                    logger.warning(
                        "⚠️ Key hit a 429 rate limit. Putting it in a 20-minute timeout."
                    )
                    self.key_cooldowns[key] = time.time() + 1200
                attempts += 1
                continue
            if response_text.startswith("ERROR_CODE_") or not response_text.strip():
                logger.warning("⚠️ API Error or Empty Response. Rotating...")
                attempts += 1
                continue
            new_code, explanation, edit_success = self.apply_xml_edits(
                source_code, response_text
            )
            edit_count = len(re.findall(r"<EDIT>", response_text, re.IGNORECASE))
            lower_exp = explanation.lower()
            ai_approved_code = (
                "no fixes needed" in lower_exp
                or "looks good" in lower_exp
                or "no changes needed" in lower_exp
            )

            if not require_edit and ai_approved_code:
                if edit_count > 0:
                    logger.info(
                        "🤖 AI stated the code looks good, but hallucinated empty <EDIT> blocks. Ignoring them."
                    )
                return source_code, explanation, response_text
            if edit_count > 0 and not edit_success:
                logger.warning(
                    f"⚠️ Partial edit failure in {display_name} ({edit_count} blocks found, but some missed targets). Auto-regenerating..."
                )
                attempts += 1
                continue
            if require_edit and new_code == source_code:
                logger.warning("Search block mismatch. Rotating...")
                attempts += 1
                continue
            if not require_edit and new_code == source_code:
                lower_exp = explanation.lower()
                if (
                    "no fixes needed" in lower_exp
                    or "looks good" in lower_exp
                    or "no changes needed" in lower_exp
                ):
                    return new_code, explanation, response_text
                else:
                    if edit_count > 0:
                        logger.warning(
                            f"⚠️ AI generated {edit_count} <EDIT> blocks, but <SEARCH> text failed to match. Rotating..."
                        )
                    else:
                        logger.warning(
                            f"⚠️ AI provided no edit and didn't state: [{display_name}] looks good. Rotating..."
                        )
                    attempts += 1
                    continue
            if new_code != source_code:
                print("\n" + "=" * 50)
                print(f"💡 AI Proposed Edit Ready for: [{display_name}]")
                print("=" * 50)
                print(
                    f"AI Thought: {explanation[:400]}{'...' if len(explanation) > 400 else ''}"
                )
                diff_lines = list(
                    difflib.unified_diff(
                        source_code.splitlines(keepends=True),
                        new_code.splitlines(keepends=True),
                        fromfile="Original",
                        tofile="Proposed",
                    )
                )
                print("\nProposed Changes:\n")
                for line in diff_lines[2:22]:
                    clean_line = line.rstrip()
                    if clean_line.startswith("+"):
                        print(f"\033[92m{clean_line}\033[0m")
                    elif clean_line.startswith("-"):
                        print(f"\033[91m{clean_line}\033[0m")
                    elif clean_line.startswith("@@"):
                        print(f"\033[94m{clean_line}\033[0m")
                    else:
                        print(clean_line)
                if len(diff_lines) > 22:
                    print(f"\033[93m... and {len(diff_lines) - 22} more lines.\033[0m")
                user_choice = self.get_user_approval(
                    "Hit ENTER to APPLY, type 'FULL_DIFF' to view full diff, 'EDIT_CODE' to refine code manually, 'EDIT_XML' to refine AI XML, 'REGENERATE' to retry AI, or 'SKIP' to cancel.",
                    timeout=220,
                )
                if user_choice == "SKIP":
                    logger.info("AI proposed edit skipped by user.")
                    return source_code, "Edit skipped by user.", ""
                elif user_choice == "REGENERATE":
                    logger.info("Regenerating AI edit...")
                    attempts += 1
                    continue
                elif user_choice == "EDIT_XML":
                    logger.info(
                        "Opening AI XML response in editor for manual refinement..."
                    )
                    response_text = self._edit_prompt_with_external_editor(
                        response_text
                    )
                    new_code, explanation, _ = self.apply_xml_edits(
                        source_code, response_text
                    )
                    if new_code == source_code:
                        logger.warning(
                            "Edited XML failed to match code. Skipping edit."
                        )
                        return source_code, "Edit failed after manual refinement.", ""
                    return new_code, explanation, response_text
                elif user_choice == "EDIT_CODE":
                    logger.info(
                        "Opening proposed code in editor for manual refinement..."
                    )
                    file_ext = (
                        os.path.splitext(target_filepath)[1]
                        if target_filepath
                        else ".py"
                    )
                    edited_code = self._launch_external_code_editor(
                        new_code, file_suffix=file_ext
                    )
                    if edited_code == new_code:
                        logger.info(
                            "No changes made in external editor. Proceeding with original AI proposal."
                        )
                        return new_code, explanation, response_text
                    else:
                        logger.info(
                            "User manually refined code. Applying refined code."
                        )
                        return (
                            edited_code,
                            explanation + " (User refined code manually)",
                            response_text,
                        )
                elif user_choice == "FULL_DIFF":
                    full_diff_text = "".join(diff_lines)
                    try:
                        pager_cmd = os.environ.get("PAGER", "less -R").split()
                        if sys.platform == "win32":
                            pager_cmd = ["more"]
                        process = subprocess.Popen(
                            pager_cmd,
                            stdin=subprocess.PIPE,
                            stdout=sys.stdout,
                            stderr=sys.stderr,
                            text=True,
                        )
                        process.communicate(input=full_diff_text)
                    except FileNotFoundError:
                        for line in diff_lines:
                            clean_line = line.rstrip()
                            if clean_line.startswith("+"):
                                print(f"\033[92m{clean_line}\033[0m")
                            elif clean_line.startswith("-"):
                                print(f"\033[91m{clean_line}\033[0m")
                            elif clean_line.startswith("@@"):
                                print(f"\033[94m{clean_line}\033[0m")
                            else:
                                print(clean_line)
                    user_choice_after_diff = self.get_user_approval(
                        "Hit ENTER to APPLY, type 'EDIT_CODE' to refine code manually, 'EDIT_XML' to refine AI XML, 'REGENERATE' to retry AI, or 'SKIP' to cancel.",
                        timeout=220,
                    )
                    if user_choice_after_diff == "SKIP":
                        return source_code, "Edit skipped by user.", ""
                    elif user_choice_after_diff == "REGENERATE":
                        attempts += 1
                        continue
                    elif user_choice_after_diff == "EDIT_XML":
                        response_text = self._edit_prompt_with_external_editor(
                            response_text
                        )
                        new_code, explanation, _ = self.apply_xml_edits(
                            source_code, response_text
                        )
                        if new_code == source_code:
                            return (
                                source_code,
                                "Edit failed after manual refinement.",
                                "",
                            )
                        return new_code, explanation, response_text
                    elif user_choice_after_diff == "EDIT_CODE":
                        file_ext = (
                            os.path.splitext(target_filepath)[1]
                            if target_filepath
                            else ".py"
                        )
                        edited_code = self._launch_external_code_editor(
                            new_code, file_suffix=file_ext
                        )
                        if edited_code == new_code:
                            return new_code, explanation, response_text
                        else:
                            return (
                                edited_code,
                                explanation + " (User refined code manually)",
                                response_text,
                            )
                    return new_code, explanation, response_text
                return new_code, explanation, response_text

    def run_linter_fix_loop(self, context_of_change: str = "") -> bool:
        logger.info("\n🧹 Validating codebase syntax (Python, JS, CSS)...")
        success: bool = True
        try:
            subprocess.run(["ruff", "format", self.target_dir], capture_output=True)
            res = subprocess.run(
                ["ruff", "check", self.target_dir], capture_output=True, text=True
            )
            if res.returncode != 0:
                logger.warning(f"⚠️ Ruff found errors:\n{res.stdout.strip()}")
                py_fixed = False
                for attempt in range(3):
                    file_errors: dict[str, list[str]] = {}
                    for line in res.stdout.splitlines():
                        if ".py:" in line:
                            filepath = line.split(":")[0].strip()
                            if os.path.exists(filepath):
                                file_errors.setdefault(filepath, []).append(line)
                    if not file_errors:
                        break
                    for fpath, errs in file_errors.items():
                        self._apply_linter_fixes(
                            fpath, "\n".join(errs), context_of_change
                        )
                    recheck = subprocess.run(
                        ["ruff", "check", self.target_dir],
                        capture_output=True,
                        text=True,
                    )
                    if recheck.returncode == 0:
                        logger.info("✅ Python Auto-fix successful!")
                        py_fixed = True
                        break
                if not py_fixed:
                    logger.error("❌ Python errors remain unfixable.")
                    success = False
        except FileNotFoundError:
            pass
        try:
            js_files = []
            for root, dirs, files in os.walk(self.target_dir):
                dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
                for file in files:
                    if file.endswith(".js") and file not in IGNORE_FILES:
                        js_files.append(os.path.join(root, file))
            for js_file in js_files:
                res = subprocess.run(
                    ["node", "--check", js_file], capture_output=True, text=True
                )
                if res.returncode != 0:
                    rel_name = os.path.basename(js_file)
                    err_msg = res.stderr.strip()
                    logger.warning(f"⚠️ JS Syntax Error in {rel_name}:\n{err_msg}")
                    js_fixed = False
                    for attempt in range(3):
                        logger.info(
                            f"Asking AI to fix JS syntax (Attempt {attempt + 1}/3)..."
                        )
                        self._apply_linter_fixes(js_file, err_msg, context_of_change)
                        recheck = subprocess.run(
                            ["node", "--check", js_file], capture_output=True, text=True
                        )
                        if recheck.returncode == 0:
                            logger.info(f"✅ JS Auto-fix successful for {rel_name}!")
                            js_fixed = True
                            break
                    if not js_fixed:
                        logger.error(f"❌ JS syntax in {rel_name} remains broken.")
                        success = False
                        break
        except FileNotFoundError:
            logger.info("Node.js not installed. Skipping JS syntax validation.")
        for root, dirs, files in os.walk(self.target_dir):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for file in files:
                if file.endswith(".css") and file not in IGNORE_FILES:
                    path = os.path.join(root, file)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            css_content = f.read()
                        if css_content.count("{") != css_content.count("}"):
                            logger.error(
                                f"❌ CSS Syntax Error in {file}: Unbalanced braces."
                            )
                            success = False
                    except Exception:
                        pass
        return success

    def _apply_linter_fixes(
        self, filepath: str, err_text: str, context_of_change: str = ""
    ):
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()
        rel_path = os.path.relpath(filepath, self.target_dir)
        if context_of_change:
            logger.info(f"Applying CONTEXT-AWARE fix for `{rel_path}`...")
            prompt = self.load_prompt(
                "PIR.md",
                context_of_change=context_of_change,
                rel_path=rel_path,
                err_text=err_text,
                code=code,
            )
        else:
            logger.info(f"Applying standard linter fix for `{rel_path}`...")
            prompt = self.load_prompt(
                "ALF.md", rel_path=rel_path, err_text=err_text, code=code
            )
        new_code, _, _ = self.get_valid_edit(
            prompt, code, require_edit=True, target_filepath=filepath
        )
        if new_code != code:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_code)
            self.session_context.append(
                f"Auto-fixed syntax/linting errors in `{rel_path}`."
            )

    def run_and_verify_app(self, context_of_change: str = "") -> bool:
        entry_file = self._find_entry_file()
        if not entry_file:
            return True

        if getattr(sys, "frozen", False):
            python_cmd = shutil.which("python3") or shutil.which("python") or "python3"
        else:
            python_cmd = sys.executable

        for attempt in range(3):
            logger.info(
                f"\n🚀 PHASE 4: Runtime Verification. Launching {os.path.basename(entry_file)} for 10 seconds (Attempt {attempt + 1}/3)..."
            )
            process = subprocess.Popen(
                [python_cmd, entry_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.target_dir,
            )
            stdout, stderr = "", ""
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    stdout, stderr = process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
            error_keywords = [
                "Traceback (most recent call last):",
                "Exception:",
                "Error:",
                "NameError:",
                "AttributeError:",
            ]
            has_crash = any(kw in stderr for kw in error_keywords) or (
                process.returncode != 0
                and process.returncode not in (None, 0, 15, -15, 137, -9, 1)
            )
            if not has_crash and "Traceback" not in stdout:
                logger.info("✅ App ran successfully for 10 seconds with no crashes.")
                return True
            logger.warning(f"⚠️ App crashed or threw runtime errors!\n{stderr}")
            self._fix_runtime_errors(
                stderr + "\n" + stdout, entry_file, context_of_change
            )
        logger.error("❌ Exhausted runtime auto-fix attempts.")
        return False

    def _fix_runtime_errors(
        self, logs: str, entry_file: str, context_of_change: str = ""
    ):
        """Detects crashes. Handles missing packages automatically, otherwise asks AI."""

        package_match = re.search(r"ModuleNotFoundError: No module named '(.*?)'", logs)
        if not package_match:
            package_match = re.search(r"ImportError: No module named '(.*?)'", logs)

        if package_match:
            pkg = package_match.group(1)
            logger.info(
                f"📦 Auto-detected missing dependency: {pkg}. Attempting pip install..."
            )

            if getattr(sys, "frozen", False):
                python_cmd = (
                    shutil.which("python3") or shutil.which("python") or "python3"
                )
            else:
                python_cmd = sys.executable

            try:
                subprocess.run(
                    [
                        python_cmd,
                        "-m",
                        "pip",
                        "install",
                        pkg,
                        "--break-system-packages",
                    ],
                    check=True,
                )
                subprocess.run(
                    [
                        python_cmd,
                        "-m",
                        "pip",
                        "install",
                        f"types-{pkg}",
                        "--break-system-packages",
                    ],
                    capture_output=True,
                )
                logger.info(
                    "✅ Successfully installed {pkg}. System will now retry launch."
                )
                return
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install {pkg} automatically: {e}")
        tb_files = re.findall(r'File "([^"]+)"', logs)
        target_file = entry_file
        for f in reversed(tb_files):
            abs_f = os.path.abspath(f)
            if (
                abs_f.startswith(self.target_dir)
                and not any(ign in abs_f for ign in IGNORE_DIRS)
                and os.path.exists(abs_f)
            ):
                target_file = abs_f
                break
        rel_path = os.path.relpath(target_file, self.target_dir)
        with open(target_file, "r", encoding="utf-8") as f:
            code = f.read()
        if context_of_change:
            logger.info(
                f"Applying CONTEXT-AWARE fix for runtime crash in `{rel_path}`..."
            )
            prompt = self.load_prompt(
                "PIR.md",
                context_of_change=context_of_change,
                rel_path=rel_path,
                err_text=logs[-2000:],
                code=code,
            )
        else:
            logger.info(f"Applying standard fix for runtime crash in `{rel_path}`...")
            memory_section = (
                f"### Project Memory / Context:\n{self.memory}\n\n"
                if self.memory
                else ""
            )
            prompt = self.load_prompt(
                "FRE.md",
                memory_section=memory_section,
                logs=logs[-2000:],
                rel_path=rel_path,
                code=code,
            )
        new_code, explanation, _ = self.get_valid_edit(
            prompt, code, require_edit=True, target_filepath=target_file
        )
        if new_code != code:
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(new_code)
            logger.info(f"✅ AI Auto-patched runtime crash in `{rel_path}`")
            self.session_context.append(
                f"Auto-fixed runtime crash in `{rel_path}`: {explanation}"
            )
            self.run_linter_fix_loop()

    def write_pr(self, filepath: str, explanation: str, llm_response: str):
        rel_path = os.path.relpath(filepath, self.target_dir)
        mode = "a" if os.path.exists(self.pr_file) else "w"
        with open(self.pr_file, mode, encoding="utf-8") as f:
            if mode == "w":
                f.write(
                    "# 🚀 Autonomous Code Review & Patch Proposals\n\n*Automatically generated by AutoPR Reviewer*\n\n---\n"
                )
            f.write(
                f"\n## 🛠 Review for `{rel_path}`\n**AI Analysis & Fixes:**\n{explanation}\n\n"
            )
            edits = re.findall(
                r"<EDIT>.*?</EDIT>", llm_response, re.DOTALL | re.IGNORECASE
            )
            if edits:
                f.write("### Proposed Patch:\n```xml\n")
                for edit in edits:
                    f.write(edit.strip() + "\n\n")
                f.write("```\n")
            f.write("\n---\n")
        logger.info(f"Appended successful fix for {rel_path} to {PR_FILE_NAME}")

    def analyze_file(self, filepath: str, current_index: int, total_files: int):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
                content = "".join(lines)
        except UnicodeDecodeError:
            return
        lang_name, lang_tag = self.get_language_info(filepath)
        filename = os.path.basename(filepath)
        logger.info(
            f"[{current_index}/{total_files}] Scanning {filename} ({lang_name}) - Reading {len(lines)} lines into AI context..."
        )
        ruff_out, mypy_out, custom_issues = "", "", []
        if lang_tag == "python":
            ruff_out, mypy_out = self.run_linters(filepath)
            custom_issues = self.scan_for_lazy_code(filepath, content)
        prompt = self.build_patch_prompt(
            lang_name, lang_tag, content, ruff_out, mypy_out, custom_issues
        )
        new_code, explanation, llm_response = self.get_valid_edit(
            prompt, content, require_edit=False, target_filepath=filepath
        )
        if new_code == content:
            logger.info(
                f"✅ AI Analysis complete. No changes required for {filename}.\n"
            )
            return
        if new_code != content:
            self.write_pr(filepath, explanation, llm_response)
            self.session_context.append(
                f"Proposed patch for `{filename}` to fix: {explanation}"
            )
            logger.info(
                f"⚠️ Issues found and patched in {filename}. Added to {PR_FILE_NAME}.\n"
            )

    def scan_directory(self):
        file_list = []
        for root, dirs, files in os.walk(self.target_dir):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for file in files:
                if file in IGNORE_FILES or file == os.path.basename(__file__):
                    continue
                if any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    file_list.append(os.path.join(root, file))
        return file_list

    def propose_feature(self, target_path: str):
        rel_path = os.path.relpath(target_path, self.target_dir)
        lang_name, lang_tag = self.get_language_info(target_path)
        with open(target_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(
            f"\n🚀 PHASE 2: Generating an interactive feature proposal for [{rel_path}]..."
        )
        memory_section = self._get_rich_context()
        prompt = self.load_prompt(
            "PF.md",
            lang_name=lang_name,
            memory_section=memory_section,
            lang_tag=lang_tag,
            content=content,
            rel_path=rel_path,
        )
        print("\n" + "=" * 50)
        print(f"💡 Feature Proposal Prompt Ready: [{rel_path}]")
        print("=" * 50)
        print(
            f"The AI has prepared a prompt to generate a feature proposal for: {rel_path}"
        )
        user_choice = self.get_user_approval(
            "Hit ENTER to send as-is, type 'EDIT_PROMPT' to refine the full prompt, 'AUGMENT_PROMPT' to add quick instructions, or 'SKIP' to cancel this proposal.",
            timeout=220,
        )
        if user_choice == "SKIP":
            logger.info("Feature proposal skipped by user.")
            return
        if user_choice == "EDIT_PROMPT":
            prompt = self._edit_prompt_with_external_editor(prompt)
            if not prompt.strip():
                logger.warning("Edited prompt is empty. Skipping feature proposal.")
                return
        elif user_choice == "AUGMENT_PROMPT":
            logger.info("Opening augmentation editor to add quick instructions...")
            augmentation_text = self._get_user_prompt_augmentation()
            if augmentation_text.strip():
                prompt += f"\n\n### User Augmentation:\n{augmentation_text.strip()}"
                logger.info("Prompt augmented with user input for feature proposal.")
            else:
                logger.info("No augmentation provided.")

        def validator(text):
            return "<SNIPPET>" in text and "</SNIPPET>" in text

        llm_response = self.get_valid_llm_response(prompt, validator, context=rel_path)
        thought_match = re.search(
            r"<(?:THOUGHT|EXPLANATION)>(.*?)</(?:THOUGHT|EXPLANATION)>",
            llm_response,
            re.DOTALL | re.IGNORECASE,
        )
        snippet_match = re.search(
            r"<SNIPPET>\n?(.*?)\n?</SNIPPET>", llm_response, re.DOTALL | re.IGNORECASE
        )
        explanation = (
            thought_match.group(1).strip()
            if thought_match
            else "No explicit explanation generated."
        )
        suggested_code = snippet_match.group(1).strip() if snippet_match else ""
        suggested_code = re.sub(r"^```[a-zA-Z]*\n", "", suggested_code)
        suggested_code = re.sub(r"\n```$", "", suggested_code)
        suggested_code = suggested_code.strip()
        if not suggested_code:
            return logger.error("LLM failed to generate a valid feature snippet.")
        with open(self.feature_file, "w", encoding="utf-8") as f:
            f.write(
                f"# 💡 Feature Proposal\n\n**Target File:** `{rel_path}`\n\n**Explanation:**\n{explanation}\n\n### Suggested Addition/Optimization:\n```{lang_tag}\n{suggested_code}\n```\n\n---\n> **ACTION REQUIRED:** Review this feature proposal. Wait for terminal prompt to approve.\n"
            )
        logger.info(f"Feature proposal written to {FEATURE_FILE_NAME}.")

    def implement_feature(self, feature_content: str) -> bool:
        match = re.search(r"\*\*Target File:\*\* `(.*?)`", feature_content)
        if not match:
            logger.error("Could not determine target file from FEATURE.md formatting.")
            return False
        rel_path = match.group(1)
        target_path = os.path.join(self.target_dir, rel_path)
        lang_name, lang_tag = self.get_language_info(target_path)
        with open(target_path, "r", encoding="utf-8") as f:
            source_code = f.read()

        # --- NEW: ARCHITECTURAL SPLIT SUPPORT ---
        # Look for a <CREATE_FILE> tag in the AI proposal
        new_file_match = re.search(
            r'<CREATE_FILE path="(.*?)">(.*?)</CREATE_FILE>', feature_content, re.DOTALL
        )
        if new_file_match:
            new_path_rel = new_file_match.group(1)
            new_code_payload = new_file_match.group(2).strip()
            new_path_abs = os.path.join(self.target_dir, new_path_rel)
            if not os.path.exists(new_path_abs):
                logger.warning(
                    f"🏗️ ARCHITECTURAL SPLIT: Spawning new module `{new_path_rel}`"
                )
                with open(new_path_abs, "w", encoding="utf-8") as f:
                    f.write(new_code_payload)
                self.session_context.append(
                    f"Created new architectural module: `{new_path_rel}`"
                )
        # ----------------------------------------

        exp_match = re.search(
            r"\*\*Explanation:\*\*(.*?)(?:###|---|>)",
            feature_content,
            re.DOTALL | re.IGNORECASE,
        )
        feature_explanation = (
            exp_match.group(1).strip()
            if exp_match
            else f"Implemented a new structural feature for: [{rel_path}]..."
        )
        logger.info(
            f"Implementing approved feature seamlessly directly into {rel_path}..."
        )
        memory_section = self._get_rich_context()
        prompt = self.load_prompt(
            "IF.md",
            memory_section=memory_section,
            feature_content=feature_content,
            lang_name=lang_name,
            lang_tag=lang_tag,
            source_code=source_code,
            rel_path=rel_path,
        )
        new_code, _, _ = self.get_valid_edit(
            prompt, source_code, require_edit=True, target_filepath=target_path
        )
        if new_code == source_code:
            logger.error("Implementation failed. LLM did not apply valid changes.")
            return False
        if lang_tag == "python":
            new_code = self.ensure_imports_retained(source_code, new_code, target_path)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(new_code)
        if lang_tag == "python":
            if not self.run_linter_fix_loop(
                context_of_change=feature_content
            ) or not self.run_and_verify_app(context_of_change=feature_content):
                return False
            if not self.check_downstream_breakages(target_path, rel_path):
                logger.error(
                    "❌ Feature implementation failed downstream type checks (Mypy)."
                )
                return False
        logger.info(f"✅ Successfully implemented feature directly into {rel_path}.")
        self.session_context.append(
            f"Successfully implemented feature directly into `{rel_path}` -> {feature_explanation}"
        )
        if os.path.exists(self.feature_file):
            os.remove(self.feature_file)
        return True

    def check_downstream_breakages(self, target_path: str, rel_path: str) -> bool:
        logger.info(
            f"\n🔍 PHASE 3: Simulating workspace to check for downstream breakages caused by {rel_path} edits..."
        )
        try:
            excludes = (
                set(IGNORE_DIRS) | set(IGNORE_FILES) | {os.path.basename(__file__)}
            )
            exclude_regex = (
                r"(^|/|\\)(" + "|".join(re.escape(x) for x in excludes) + r")(/|\\|$)"
            )
            result = subprocess.run(
                [
                    "mypy",
                    self.target_dir,
                    "--exclude",
                    exclude_regex,
                    "--ignore-missing-imports",
                ],
                capture_output=True,
                text=True,
            )
            if "error:" in result.stdout:
                logger.warning(
                    f"⚠️ Downstream Breakage Detected!\n{result.stdout.strip()}"
                )
                return self.propose_cascade_fix(result.stdout.strip(), rel_path)
            logger.info("✅ No downstream breakages detected.")
            return True
        except Exception as e:
            logger.error(f"Error during Phase 3 assessment: {e}")
            return True

    def propose_cascade_fix(self, mypy_errors: str, trigger_file: str) -> bool:
        problem_file = None
        for line in mypy_errors.splitlines():
            if ".py:" in line:
                candidate = line.split(":")[0].strip()
                if (
                    os.path.exists(candidate)
                    and os.path.basename(candidate) not in IGNORE_FILES
                    and not any(ign in candidate for ign in IGNORE_DIRS)
                ):
                    problem_file = candidate
                    if trigger_file not in candidate:
                        break
        if not problem_file:
            return False
        rel_broken_path = os.path.relpath(problem_file, self.target_dir)
        with open(problem_file, "r", encoding="utf-8") as f:
            broken_code = f.read()
        memory_section = (
            f"### Project Context:\n{self.memory}\n\n" if self.memory else ""
        )
        prompt = self.load_prompt(
            "PCF.md",
            memory_section=memory_section,
            trigger_file=trigger_file,
            rel_broken_path=rel_broken_path,
            mypy_errors=mypy_errors,
            broken_code=broken_code,
        )
        new_code, _, _ = self.get_valid_edit(
            prompt, broken_code, require_edit=True, target_filepath=problem_file
        )
        if new_code != broken_code:
            with open(problem_file, "w", encoding="utf-8") as f:
                f.write(new_code)
            logger.info(
                f"✅ Auto-patched cascading fix directly into `{os.path.basename(problem_file)}`"
            )
            self.session_context.append(
                f"Auto-applied downstream cascade fix in `{os.path.basename(problem_file)}`"
            )
            self.run_linter_fix_loop()
            final_check = subprocess.run(
                ["mypy", problem_file, "--ignore-missing-imports"], capture_output=True
            )
            return final_check.returncode == 0
        logger.error(f"❌ Failed to auto-patch `{rel_broken_path}`.")
        return False

    def implement_pr(self, pr_content: str) -> bool:
        logger.info("Implementing approved PRs seamlessly from XML blocks...")
        file_sections = re.split(r"## 🛠 Review for `(.*?)`", pr_content)
        if len(file_sections) < 3:
            logger.error("No valid file patches found in PR.md to apply.")
            return False
        all_success = True
        for i in range(1, len(file_sections), 2):
            rel_path = file_sections[i].strip()
            section_content = file_sections[i + 1]
            target_path = os.path.join(self.target_dir, rel_path)
            if not os.path.exists(target_path):
                logger.error(f"Target file {rel_path} not found for patching.")
                all_success = False
                continue
            with open(target_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            new_code, _, _ = self.apply_xml_edits(source_code, section_content)
            if new_code == source_code:
                logger.error(
                    f"❌ Failed to apply XML patch for {rel_path}. The SEARCH blocks did not match."
                )
                all_success = False
            else:
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(new_code)
                logger.info(f"✅ Successfully applied patch to {rel_path}.")
        if all_success:
            if not self.run_linter_fix_loop(
                context_of_change=pr_content
            ) or not self.run_and_verify_app(context_of_change=pr_content):
                return False
            self.session_context.append(
                "Applied automated patch XML edits directly into the codebase from an approved PR.md."
            )
            if os.path.exists(self.pr_file):
                os.remove(self.pr_file)
            return True
        else:
            return False

    def run_pipeline(self, current_iteration: int):
        if not self.session_context:
            self.session_context = []
        changes_made = False
        try:
            if os.path.exists(self.pr_file) or os.path.exists(self.feature_file):
                logger.info("==================================================")
                logger.info(
                    f"⏸️ Found pending {PR_FILE_NAME} and/or {FEATURE_FILE_NAME} from a previous run."
                )
                user_input = self.get_user_approval(
                    "Hit ENTER to PROCEED, type 'SKIP' to ignore, or 'DELETE' to discard",
                    timeout=220,
                )
                if user_input == "PROCEED":
                    backup_state = self.backup_workspace()
                    success = True
                    if os.path.exists(self.pr_file):
                        with open(self.pr_file, "r", encoding="utf-8") as f:
                            if not self.implement_pr(f.read()):
                                success = False
                    if success and os.path.exists(self.feature_file):
                        with open(self.feature_file, "r", encoding="utf-8") as f:
                            if not self.implement_feature(f.read()):
                                success = False
                    if not success:
                        self.restore_workspace(backup_state)
                        logger.warning("🔄 Rollback performed due to unfixable errors.")

                        failure_report = f"\n\n### ❌ FAILURE ATTEMPT LOGS ({time.strftime('%Y-%m-%d %H:%M:%S')})\n"
                        failure_report += "\n".join(self.session_context[-3:])

                        if os.path.exists(self.pr_file):
                            content = open(self.pr_file).read()
                            with open(self.failed_pr_file, "w") as f:
                                f.write(content + failure_report)
                            os.remove(self.pr_file)

                        if os.path.exists(self.feature_file):
                            content = open(self.feature_file).read()
                            with open(self.failed_feature_file, "w") as f:
                                f.write(content + failure_report)
                            os.remove(self.feature_file)

                    changes_made = True
                elif user_input == "DELETE":
                    if os.path.exists(self.pr_file):
                        os.remove(self.pr_file)
                    if os.path.exists(self.feature_file):
                        os.remove(self.feature_file)
                    logger.info(
                        "Deleted pending proposal files. Starting fresh scan..."
                    )
            if not changes_made:
                logger.info("==================================================")
                logger.info("🚀 PHASE 1: Initial Assessment & Codebase Scan")
                logger.info("==================================================")
                all_files = self.scan_directory()
                if not all_files:
                    return logger.warning("No supported source files found.")
                for idx, filepath in enumerate(all_files, start=1):
                    self.analyze_file(filepath, idx, len(all_files))
                logger.info("==================================================")
                logger.info("✅ Phase 1 Complete.")
                logger.info("==================================================")
                if os.path.exists(self.pr_file):
                    logger.info(
                        "⏸️ Skipping Phase 2 (Feature Proposal) because Phase 1 found bugs."
                    )
                    logger.info("Applying fixes first to prevent code collisions...")
                elif all_files:
                    logger.info("🚀 Moving to Phase 2: Generating Feature Proposal...")
                    self.propose_feature(random.choice(all_files))
                if os.path.exists(self.pr_file) or os.path.exists(self.feature_file):
                    print("\n" + "=" * 50)
                    print("⏸️ ACTION REQUIRED: Proposals Generated")
                    user_input = self.get_user_approval(
                        "Hit ENTER to PROCEED, or type 'SKIP' to cancel", timeout=220
                    )
                    if user_input == "PROCEED":
                        backup_state = self.backup_workspace()
                        success = True
                        if os.path.exists(self.pr_file):
                            with open(self.pr_file, "r", encoding="utf-8") as f:
                                if not self.implement_pr(f.read()):
                                    success = False
                        if success and os.path.exists(self.feature_file):
                            with open(self.feature_file, "r", encoding="utf-8") as f:
                                if not self.implement_feature(f.read()):
                                    success = False
                        if not success:
                            self.restore_workspace(backup_state)
                            logger.warning(
                                "🔄 Rollback performed due to unfixable errors."
                            )

                            failure_report = f"\n\n### ❌ FAILURE ATTEMPT LOGS ({time.strftime('%Y-%m-%d %H:%M:%S')})\n"
                            failure_report += "\n".join(self.session_context[-3:])

                            if os.path.exists(self.pr_file):
                                content = open(self.pr_file).read()
                                with open(self.failed_pr_file, "w") as f:
                                    f.write(content + failure_report)
                                os.remove(self.pr_file)

                            if os.path.exists(self.feature_file):
                                content = open(self.feature_file).read()
                                with open(self.failed_feature_file, "w") as f:
                                    f.write(content + failure_report)
                                os.remove(self.feature_file)
                    else:
                        logger.info(
                            "Changes not applied manually. They will remain for the next loop iteration."
                        )
                else:
                    logger.info("\n🎉 No issues found, no features proposed.")
        finally:
            self.update_memory()
            if current_iteration % 2 == 0:
                self.refactor_memory()
            logger.info("Pipeline iteration complete.")


if __name__ == "__main__":
    print("Please run `python entrance.py` instead to use the targeted memory flow.")
    sys.exit(0)
