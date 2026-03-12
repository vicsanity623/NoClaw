import ast
import difflib
import os
import random
import re
import subprocess
import sys
import time

from pyob.core_utils import (
    ANALYSIS_FILE,
    FAILED_FEATURE_FILE_NAME,
    FAILED_PR_FILE_NAME,
    FEATURE_FILE_NAME,
    GEMINI_API_KEYS,
    HISTORY_FILE,
    IGNORE_DIRS,
    IGNORE_FILES,
    MEMORY_FILE_NAME,
    PR_FILE_NAME,
    SUPPORTED_EXTENSIONS,
    SYMBOLS_FILE,
    CoreUtilsMixin,
    logger,
)
from pyob.prompts_and_memory import PromptsAndMemoryMixin
from pyob.reviewer_mixins import ValidationMixin
from pyob.feature_mixins import FeatureOperationsMixin


class AutoReviewer(
    CoreUtilsMixin, PromptsAndMemoryMixin, ValidationMixin, FeatureOperationsMixin
):
    _shared_cooldowns: dict[str, float] | None = None

    def __init__(self, target_dir: str):
        self.target_dir = os.path.abspath(target_dir)
        self.pyob_dir = os.path.join(self.target_dir, ".pyob")
        os.makedirs(self.pyob_dir, exist_ok=True)
        self.pr_file = os.path.join(self.pyob_dir, PR_FILE_NAME)
        self.feature_file = os.path.join(self.pyob_dir, FEATURE_FILE_NAME)
        self.failed_pr_file = os.path.join(self.pyob_dir, FAILED_PR_FILE_NAME)
        self.failed_feature_file = os.path.join(self.pyob_dir, FAILED_FEATURE_FILE_NAME)
        self.memory_file = os.path.join(self.pyob_dir, MEMORY_FILE_NAME)
        self.analysis_path = os.path.join(self.pyob_dir, ANALYSIS_FILE)
        self.history_path = os.path.join(self.pyob_dir, HISTORY_FILE)
        self.symbols_path = os.path.join(self.pyob_dir, SYMBOLS_FILE)
        self.memory = self.load_memory()
        self.session_context: list[str] = []
        self.manual_target_file: str | None = None
        self._ensure_prompt_files()

        if AutoReviewer._shared_cooldowns is None:
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

    def set_manual_target_file(self, filepath: str | None):
        """
        Sets or clears the manual target file for analysis.
        If a filepath is provided, it will be used for the next analysis cycle.
        If None, the system reverts to scanning the directory.
        """
        if filepath:
            if not os.path.exists(filepath):
                logger.warning(
                    f"Manual target file '{filepath}' does not exist. Ignoring."
                )
                self.manual_target_file = None
            else:
                self.manual_target_file = os.path.abspath(filepath)
                logger.info(f"Manual target file set to: {self.manual_target_file}")
        else:
            self.manual_target_file = None
            logger.info("Manual target file cleared. Reverting to directory scan.")

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
        return str(
            self.load_prompt(
                "PP.md",
                lang_name=lang_name,
                lang_tag=lang_tag,
                content=content,
                memory_section=memory_section,
                ruff_section=ruff_section,
                mypy_section=mypy_section,
                custom_issues_section=custom_issues_section,
            )
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
        is_cloud = (
            os.environ.get("GITHUB_ACTIONS") == "true"
            or os.environ.get("CI") == "true"
            or "GITHUB_RUN_ID" in os.environ
        )

        while True:
            key = None
            now = time.time()
            available_keys = [
                k for k, cooldown in self.key_cooldowns.items() if now > cooldown
            ]

            if not available_keys:
                if is_cloud:
                    # Check if we've been stuck for a while
                    logger.warning(
                        "☁️ Cloud environment: Keys exhausted. Checking if cooldowns can be cleared..."
                    )

                    # FORCE RESET: If we are in the cloud and all keys are dead,
                    # we must clear the cooldowns to attempt a retry,
                    # otherwise the pipeline is effectively dead.
                    for key in self.key_cooldowns:
                        self.key_cooldowns[key] = 0.0

                    time.sleep(60)
                    attempts += 1
                    continue
                else:
                    if not use_ollama:
                        logger.warning(
                            "🚫 Gemini rate-limited. Falling back to Local Ollama."
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

            if "ERROR_CODE_413" in response_text:
                logger.warning(
                    "⚠️ GitHub Models context too large (413). Sleeping 60s..."
                )
                time.sleep(60)
                attempts += 1
                continue

            if response_text.startswith("ERROR_CODE_429"):
                if key:
                    logger.warning("⚠️ Key hit a 429 rate limit. Timeout 2m.")
                    self.key_cooldowns[key] = time.time() + 120
                time.sleep(60)
                attempts += 1
                continue

            if response_text.startswith("ERROR_CODE_") or not response_text.strip():
                logger.warning("⚠️ API Error or Empty Response. Backing off 60s...")
                time.sleep(60)
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
                    f"⚠️ Partial edit failure in {display_name}. Auto-regenerating..."
                )
                time.sleep(30)
                attempts += 1
                continue

            if require_edit and new_code == source_code:
                logger.warning("Search block mismatch. Rotating...")
                time.sleep(30)
                attempts += 1
                continue

            if not require_edit and new_code == source_code:
                if ai_approved_code:
                    return new_code, explanation, response_text
                else:
                    logger.warning("⚠️ AI provided no edit and no approval. Rotating...")
                    time.sleep(30)
                    attempts += 1
                    continue

            if new_code != source_code:
                print("\n" + "=" * 50)
                print(f"💡 AI Proposed Edit Ready for: [{display_name}]")
                print("=" * 50)
                diff_lines = list(
                    difflib.unified_diff(
                        source_code.splitlines(keepends=True),
                        new_code.splitlines(keepends=True),
                        fromfile="Original",
                        tofile="Proposed",
                    )
                )
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

                user_choice = self.get_user_approval(
                    "Hit ENTER to APPLY, type 'FULL_DIFF', 'EDIT_CODE', 'EDIT_XML', 'REGENERATE', or 'SKIP'.",
                    timeout=220,
                )

                if user_choice == "SKIP":
                    return source_code, "Edit skipped by user.", ""
                elif user_choice == "REGENERATE":
                    attempts += 1
                    continue
                elif user_choice == "EDIT_XML":
                    response_text = self._edit_prompt_with_external_editor(
                        response_text
                    )
                    new_code, explanation, _ = self.apply_xml_edits(
                        source_code, response_text
                    )
                    return new_code, explanation, response_text
                elif user_choice == "EDIT_CODE":
                    file_ext = (
                        os.path.splitext(target_filepath)[1]
                        if target_filepath
                        else ".py"
                    )
                    edited_code = self._launch_external_code_editor(
                        new_code, file_suffix=file_ext
                    )
                    return (
                        edited_code,
                        explanation + " (User refined code manually)",
                        response_text,
                    )

                return new_code, explanation, response_text

    def scan_directory(self) -> list[str]:
        file_list = []
        for root, dirs, files in os.walk(self.target_dir):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for file in files:
                if file in IGNORE_FILES:
                    continue
                if file.endswith(".spec") or file.endswith(".dmg"):
                    continue
                if any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    file_list.append(os.path.join(root, file))
        return file_list

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
                        self.session_context.append(
                            "CRITICAL: The last refactor/feature attempt FAILED and was ROLLED BACK. "
                            "The files on disk have NOT changed. Check FAILED_FEATURE.md for error logs."
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
                if self.manual_target_file:
                    if os.path.exists(self.manual_target_file):
                        all_files = [self.manual_target_file]
                        logger.info(
                            f"ð¯ Manual target file override active: {self.manual_target_file}"
                        )
                    else:
                        logger.warning(
                            f"Manual target file '{self.manual_target_file}' not found. Reverting to full scan."
                        )
                        self.manual_target_file = None  # Clear invalid target
                        all_files = self.scan_directory()
                else:
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
