import ast
import difflib
import os
import random
import re
import subprocess
import sys
import time

from .core_utils import (
    ANALYSIS_FILE,
    FAILED_FEATURE_FILE_NAME,
    FAILED_PR_FILE_NAME,
    FEATURE_FILE_NAME,
    GEMINI_API_KEYS,
    HISTORY_FILE,
    MEMORY_FILE_NAME,
    PR_FILE_NAME,
    SYMBOLS_FILE,
    CoreUtilsMixin,
    logger,
)
from .feature_mixins import FeatureOperationsMixin
from .prompts_and_memory import PromptsAndMemoryMixin
from .reviewer_mixins import ValidationMixin
from .scanner_mixins import ScannerMixin


class AutoReviewer(
    CoreUtilsMixin,
    PromptsAndMemoryMixin,
    ValidationMixin,
    FeatureOperationsMixin,
    ScannerMixin,
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
        display_name = os.path.relpath(target_filepath, self.target_dir) if target_filepath else "System Update"
        
        # 1. Pre-Flight Human Check
        prompt, skip = self._handle_pre_generation_approval(prompt, display_name)
        if skip:
            return source_code, "AI generation skipped by user.", ""

        attempts = 0
        while True:
            # 2. Fetch from AI (Handles keys, retries, and API limits)
            response_text, attempts = self._fetch_llm_with_retries(prompt, display_name, attempts)
            
            # 3. Validate and Apply XML Patch
            new_code, explanation, is_valid = self._validate_llm_patch(
                source_code, response_text, require_edit, display_name
            )
            
            if not is_valid:
                attempts += 1
                continue

            if new_code == source_code:
                return new_code, explanation, response_text

            # 4. Post-Flight Human Review (Diffs and Approval)
            final_code, final_exp, final_resp, action = self._handle_post_generation_review(
                source_code, new_code, explanation, response_text, target_filepath, display_name
            )
            
            if action == "REGENERATE":
                attempts += 1
                continue
                
            return final_code, final_exp, final_resp


    # ==========================================
    # PRIVATE HELPER METHODS FOR GET_VALID_EDIT
    # ==========================================

    def _handle_pre_generation_approval(self, prompt: str, display_name: str) -> tuple[str, bool]:
        print("\n" + "=" * 50)
        print(f"💡 AI Generation Prompt Ready: [{display_name}]")
        print("=" * 50)
        choice = self.get_user_approval(
            "Hit ENTER to send as-is, type 'EDIT_PROMPT', 'AUGMENT_PROMPT', or 'SKIP'.", timeout=220
        )
        if choice == "SKIP":
            return prompt, True
        elif choice == "EDIT_PROMPT":
            prompt = self._edit_prompt_with_external_editor(prompt)
        elif choice == "AUGMENT_PROMPT":
            aug = self._get_user_prompt_augmentation()
            if aug.strip():
                prompt += f"\n\n### User Augmentation:\n{aug.strip()}"
        return prompt, False

    def _fetch_llm_with_retries(self, prompt: str, display_name: str, attempts: int) -> tuple[str, int]:
        is_cloud = os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CI") == "true"
        use_ollama = False
        
        while True:
            now = time.time()
            available_keys = [k for k, cd in self.key_cooldowns.items() if now > cd]
            key = None

            if not available_keys:
                if is_cloud:
                    for k in self.key_cooldowns: self.key_cooldowns[k] = 0.0
                    time.sleep(60)
                    attempts += 1
                    continue
                else:
                    use_ollama = True
            else:
                use_ollama = False
                key = available_keys[attempts % len(available_keys)]
                logger.info(f"\n[Attempting Gemini API Key {attempts % len(available_keys) + 1}]")

            if use_ollama: logger.info("\n[Attempting Local Ollama]")

            response = self._stream_single_llm(prompt, key=key, context=display_name)

            if "ERROR_CODE_413" in response:
                time.sleep(60)
                attempts += 1
                continue
            if response.startswith("ERROR_CODE_429"):
                if key: self.key_cooldowns[key] = time.time() + 120
                time.sleep(60)
                attempts += 1
                continue
            if response.startswith("ERROR_CODE_") or not response.strip():
                time.sleep(60)
                attempts += 1
                continue
                
            return response, attempts

    def _validate_llm_patch(self, source_code: str, response_text: str, require_edit: bool, display_name: str) -> tuple[str, str, bool]:
        new_code, explanation, edit_success = self.apply_xml_edits(source_code, response_text)
        edit_count = len(re.findall(r"<EDIT>", response_text, re.IGNORECASE))
        lower_exp = explanation.lower()
        ai_approved = "no fixes needed" in lower_exp or "looks good" in lower_exp

        if not require_edit and ai_approved:
            return source_code, explanation, True
        if edit_count > 0 and not edit_success:
            logger.warning(f"⚠️ Partial edit failure in {display_name}. Auto-regenerating...")
            time.sleep(30)
            return source_code, explanation, False
        if require_edit and new_code == source_code:
            logger.warning("⚠️ Search block mismatch. Rotating...")
            time.sleep(30)
            return source_code, explanation, False
        if not require_edit and new_code == source_code and not ai_approved:
            time.sleep(30)
            return source_code, explanation, False
            
        return new_code, explanation, True

    def _handle_post_generation_review(self, source_code: str, new_code: str, explanation: str, response_text: str, target_filepath: str, display_name: str) -> tuple[str, str, str, str]:
        print("\n" + "=" * 50)
        print(f"✨ AI Proposed Edit Ready for: [{display_name}]")
        print("=" * 50)
        diff_lines = list(difflib.unified_diff(
            source_code.splitlines(keepends=True), new_code.splitlines(keepends=True), fromfile="Original", tofile="Proposed"
        ))
        for line in diff_lines[2:22]:
            clean = line.rstrip()
            if clean.startswith("+"): print(f"\033[92m{clean}\033[0m")
            elif clean.startswith("-"): print(f"\033[91m{clean}\033[0m")
            elif clean.startswith("@@"): print(f"\033[94m{clean}\033[0m")
            else: print(clean)

        choice = self.get_user_approval("Hit ENTER to APPLY, type 'EDIT_CODE', 'EDIT_XML', 'REGENERATE', or 'SKIP'.", timeout=220)

        if choice == "SKIP":
            return source_code, "Edit skipped.", "", "SKIP"
        elif choice == "REGENERATE":
            return source_code, explanation, response_text, "REGENERATE"
        elif choice == "EDIT_XML":
            resp = self._edit_prompt_with_external_editor(response_text)
            nc, exp, _ = self.apply_xml_edits(source_code, resp)
            return nc, exp, resp, "APPLY"
        elif choice == "EDIT_CODE":
            ext = os.path.splitext(target_filepath)[1] if target_filepath else ".py"
            ec = self._launch_external_code_editor(new_code, file_suffix=ext)
            return ec, explanation + " (User edited)", response_text, "APPLY"
            
        return new_code, explanation, response_text, "APPLY"

    def run_pipeline(self, current_iteration: int):
        if not self.session_context:
            self.session_context = []
        changes_made = False
        try:
            if os.path.exists(self.pr_file) or os.path.exists(self.feature_file):
                logger.info("==================================================")
                logger.info(
                    f"Found pending {PR_FILE_NAME} and/or {FEATURE_FILE_NAME} from a previous run."
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
                        logger.warning("Rollback performed due to unfixable errors.")
                        self.session_context.append(
                            "CRITICAL: The last refactor/feature attempt FAILED and was ROLLED BACK. "
                            "The files on disk have NOT changed. Check FAILED_FEATURE.md for error logs."
                        )

                        failure_report = f"\n\n### FAILURE ATTEMPT LOGS ({time.strftime('%Y-%m-%d %H:%M:%S')})\n"
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
                logger.info("PHASE 1: Initial Assessment & Codebase Scan")
                logger.info("==================================================")
                if self.manual_target_file:
                    if os.path.exists(self.manual_target_file):
                        all_files = [self.manual_target_file]
                        logger.info(
                            f"Manual target file override active: {self.manual_target_file}"
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
                logger.info(" Phase 1 Complete.")
                logger.info("==================================================")
                if os.path.exists(self.pr_file):
                    logger.info(
                        "Skipping Phase 2 (Feature Proposal) because Phase 1 found bugs."
                    )
                    logger.info("Applying fixes first to prevent code collisions...")
                elif all_files:
                    logger.info("Moving to Phase 2: Generating Feature Proposal...")
                    self.propose_feature(random.choice(all_files))
                if os.path.exists(self.pr_file) or os.path.exists(self.feature_file):
                    print("\n" + "=" * 50)
                    print(" ACTION REQUIRED: Proposals Generated")
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
                                " Rollback performed due to unfixable errors."
                            )

                            failure_report = f"\n\n###  FAILURE ATTEMPT LOGS ({time.strftime('%Y-%m-%d %H:%M:%S')})\n"
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
                    logger.info("\nNo issues found, no features proposed.")
        finally:
            self.update_memory()
            if current_iteration % 2 == 0:
                self.refactor_memory()
            logger.info("Pipeline iteration complete.")


if __name__ == "__main__":
    print("Please run `python entrance.py` instead to use the targeted memory flow.")
    sys.exit(0)
