import os
import re
import shutil
import subprocess
import sys

from pyob.core_utils import (
    FEATURE_FILE_NAME,
    IGNORE_DIRS,
    IGNORE_FILES,
    PR_FILE_NAME,
    logger,
)


class ValidationMixin:
    target_dir: str
    session_context: list[str]
    memory: str

    def run_linter_fix_loop(self, context_of_change: str = "") -> bool:
        logger.info("\n🧹 Validating codebase syntax (Python, JS, CSS)...")
        success: bool = True
        try:
            subprocess.run(["ruff", "format", self.target_dir], capture_output=True)

            subprocess.run(
                ["ruff", "check", self.target_dir, "--fix"], capture_output=True
            )

            res = subprocess.run(
                ["ruff", "check", self.target_dir], capture_output=True, text=True
            )
            if res.returncode != 0:
                logger.warning(f"⚠️ Ruff found logic errors:\n{res.stdout.strip()}")
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
            prompt = getattr(self, "load_prompt")(
                "PIR.md",
                context_of_change=context_of_change,
                rel_path=rel_path,
                err_text=err_text,
                code=code,
            )
        else:
            logger.info(f"Applying standard linter fix for `{rel_path}`...")
            prompt = getattr(self, "load_prompt")(
                "ALF.md", rel_path=rel_path, err_text=err_text, code=code
            )
        new_code, _, _ = getattr(self, "get_valid_edit")(
            prompt, code, require_edit=True, target_filepath=filepath
        )
        if new_code != code:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_code)
            self.session_context.append(
                f"Auto-fixed syntax/linting errors in `{rel_path}`."
            )

    def run_and_verify_app(self, context_of_change: str = "") -> bool:
        entry_file = getattr(self, "_find_entry_file")()
        if not entry_file:
            return True

        venv_python = os.path.join(self.target_dir, "build_env", "bin", "python3")
        if not os.path.exists(venv_python):
            venv_python = os.path.join(self.target_dir, "venv", "bin", "python3")

        if os.path.exists(venv_python):
            python_cmd = venv_python
        elif getattr(sys, "frozen", False):
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
            has_crash = any(kw in stderr or kw in stdout for kw in error_keywords) or (
                process.returncode != 0
                and process.returncode not in (None, 0, 15, -15, 137, -9, 1)
            )
            if not has_crash:
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
            venv_python = os.path.join(self.target_dir, "build_env", "bin", "python3")
            if not os.path.exists(venv_python):
                venv_python = os.path.join(self.target_dir, "venv", "bin", "python3")

            if os.path.exists(venv_python):
                python_cmd = venv_python
            elif getattr(sys, "frozen", False):
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
                    f"✅ Successfully installed {pkg}. System will now retry launch."
                )

                # --- AUTO-DEPENDENCY LOCKING ---
                try:
                    req_path = os.path.join(
                        getattr(self, "target_dir"), "requirements.txt"
                    )
                    subprocess.run(
                        f'"{python_cmd}" -m pip freeze > "{req_path}"',
                        shell=True,
                        check=True,
                    )
                    logger.info("🔒 Auto-locked dependencies in requirements.txt")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to lock dependencies: {e}")
                # -------------------------------

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
        with open(target_file, "r", encoding="utf-8") as f_obj:
            code = f_obj.read()
        if context_of_change:
            logger.info(
                f"Applying CONTEXT-AWARE fix for runtime crash in `{rel_path}`..."
            )
            prompt = getattr(self, "load_prompt")(
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
            prompt = getattr(self, "load_prompt")(
                "FRE.md",
                memory_section=memory_section,
                logs=logs[-2000:],
                rel_path=rel_path,
                code=code,
            )
        new_code, explanation, _ = getattr(self, "get_valid_edit")(
            prompt, code, require_edit=True, target_filepath=target_file
        )
        if new_code != code:
            with open(target_file, "w", encoding="utf-8") as f_out:
                f_out.write(new_code)
            logger.info(f"✅ AI Auto-patched runtime crash in `{rel_path}`")
            self.session_context.append(
                f"Auto-fixed runtime crash in `{rel_path}`: {explanation}"
            )
            self.run_linter_fix_loop()

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
        prompt = getattr(self, "load_prompt")(
            "PCF.md",
            memory_section=memory_section,
            trigger_file=trigger_file,
            rel_broken_path=rel_broken_path,
            mypy_errors=mypy_errors,
            broken_code=broken_code,
        )
        new_code, _, _ = getattr(self, "get_valid_edit")(
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


class FeatureOperationsMixin:
    target_dir: str
    pr_file: str
    feature_file: str
    session_context: list[str]

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
        lang_name, lang_tag = getattr(self, "get_language_info")(filepath)
        filename = os.path.basename(filepath)
        logger.info(
            f"[{current_index}/{total_files}] Scanning {filename} ({lang_name}) - Reading {len(lines)} lines into AI context..."
        )
        ruff_out, mypy_out, custom_issues = "", "", []
        if lang_tag == "python":
            ruff_out, mypy_out = getattr(self, "run_linters")(filepath)
            custom_issues = getattr(self, "scan_for_lazy_code")(filepath, content)
        prompt = getattr(self, "build_patch_prompt")(
            lang_name, lang_tag, content, ruff_out, mypy_out, custom_issues
        )
        new_code, explanation, llm_response = getattr(self, "get_valid_edit")(
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

    def propose_feature(self, target_path: str):
        rel_path = os.path.relpath(target_path, self.target_dir)
        lang_name, lang_tag = getattr(self, "get_language_info")(target_path)
        with open(target_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(
            f"\n🚀 PHASE 2: Generating an interactive feature proposal for [{rel_path}]..."
        )
        memory_section = getattr(self, "_get_rich_context")()
        prompt = getattr(self, "load_prompt")(
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
        user_choice = getattr(self, "get_user_approval")(
            "Hit ENTER to send as-is, type 'EDIT_PROMPT' to refine the full prompt, 'AUGMENT_PROMPT' to add quick instructions, or 'SKIP' to cancel this proposal.",
            timeout=220,
        )
        if user_choice == "SKIP":
            logger.info("Feature proposal skipped by user.")
            return
        if user_choice == "EDIT_PROMPT":
            prompt = getattr(self, "_edit_prompt_with_external_editor")(prompt)
            if not prompt.strip():
                logger.warning("Edited prompt is empty. Skipping feature proposal.")
                return
        elif user_choice == "AUGMENT_PROMPT":
            logger.info("Opening augmentation editor to add quick instructions...")
            augmentation_text = getattr(self, "_get_user_prompt_augmentation")()
            if augmentation_text.strip():
                prompt += f"\n\n### User Augmentation:\n{augmentation_text.strip()}"
                logger.info("Prompt augmented with user input for feature proposal.")
            else:
                logger.info("No augmentation provided.")

        def validator(text):
            return "<SNIPPET>" in text and "</SNIPPET>" in text

        llm_response = getattr(self, "get_valid_llm_response")(
            prompt, validator, context=rel_path
        )
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
        lang_name, lang_tag = getattr(self, "get_language_info")(target_path)

        with open(target_path, "r", encoding="utf-8") as f_handle:
            source_code = f_handle.read()
        created_files: list[str] = []
        new_file_matches = re.finditer(
            r'<CREATE_FILE path="(.*?)">(.*?)</CREATE_FILE>', feature_content, re.DOTALL
        )
        for file_match in new_file_matches:
            new_path_rel = file_match.group(1)
            new_code_payload = file_match.group(2).strip()
            new_path_abs = os.path.join(self.target_dir, new_path_rel)
            if not os.path.exists(new_path_abs):
                try:
                    logger.warning(
                        f"🏗️ ARCHITECTURAL SPLIT: Spawning new module `{new_path_rel}`"
                    )
                    with open(new_path_abs, "w", encoding="utf-8") as f_new:
                        f_new.write(new_code_payload)
                    created_files.append(new_path_abs)
                except Exception as e:
                    logger.error(f"Failed to create new module {new_path_rel}: {e}")
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
        memory_section = getattr(self, "_get_rich_context")()
        prompt = getattr(self, "load_prompt")(
            "IF.md",
            memory_section=memory_section,
            feature_content=feature_content,
            lang_name=lang_name,
            lang_tag=lang_tag,
            source_code=source_code,
            rel_path=rel_path,
        )

        new_code, _, _ = getattr(self, "get_valid_edit")(
            prompt, source_code, require_edit=True, target_filepath=target_path
        )

        if new_code == source_code:
            logger.error("Implementation failed. Rolling back created modules.")
            for file_path in created_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
            return False

        if lang_tag == "python":
            new_code = getattr(self, "ensure_imports_retained")(
                source_code, new_code, target_path
            )
        with open(target_path, "w", encoding="utf-8") as f_out:
            f_out.write(new_code)
        if lang_tag == "python":
            if not getattr(self, "run_linter_fix_loop")(
                context_of_change=feature_content
            ) or not getattr(self, "run_and_verify_app")(
                context_of_change=feature_content
            ):
                for file_path in created_files:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                return False

            if not getattr(self, "check_downstream_breakages")(target_path, rel_path):
                for file_path in created_files:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                return False

        logger.info(f"✅ Successfully implemented feature directly into {rel_path}.")

        self.session_context.append(
            f"SUCCESSFUL CHANGE in `{rel_path}`: {feature_explanation}"
        )

        if created_files:
            self.session_context.append(
                "Created new modules: "
                + ", ".join(
                    [os.path.basename(file_path) for file_path in created_files]
                )
            )

        if os.path.exists(self.feature_file):
            os.remove(self.feature_file)
        return True

    def implement_pr(self, pr_content: str) -> bool:
        """
        Implements PR patches atomically.
        Only applies changes if ALL patches in the PR pass the XML,
        safety, and verification gates.
        """
        logger.info("Implementing approved PRs seamlessly from XML blocks...")
        file_sections = re.split(r"## 🛠 Review for `(.*?)`", pr_content)
        
        # Guard: Ensure content actually contains patches
        if len(file_sections) < 3:
            logger.error("No valid file patches found in PR.md to apply.")
            return False
            
        all_success = True
        patches_to_apply = [] # Atomic transaction buffer

        # 1. PRE-FLIGHT VALIDATION (Check all patches before applying any)
        for i in range(1, len(file_sections), 2):
            rel_path = file_sections[i].strip()
            section_content = file_sections[i + 1]
            target_path = os.path.join(self.target_dir, rel_path)
            
            if not os.path.exists(target_path):
                logger.error(f"Target file {rel_path} not found.")
                all_success = False
                continue

            with open(target_path, "r", encoding="utf-8") as f:
                source_code = f.read()

            # Apply XML edit logic
            new_code, explanation, success = getattr(self, "apply_xml_edits")(source_code, section_content)
            
            # --- EMERGENCY DELETION GUARD ---
            # If the patch deletes > 50% of the file, it is highly likely a hallucination.
            if len(new_code) < (len(source_code) * 0.5):
                logger.error(f"🚫 EMERGENCY STOP: Patch for {rel_path} would delete too much code.")
                all_success = False
                continue
            
            # Patch Validation
            if not success:
                logger.error(f"❌ Failed to match XML blocks for {rel_path}.")
                all_success = False
            elif new_code == source_code:
                logger.warning(f"⚠️ Patch for {rel_path} changed nothing. Skipping.")
            else:
                patches_to_apply.append((target_path, new_code, rel_path))

        # 2. ATOMIC COMMIT (Only write to disk if everything checked out)
        if all_success and patches_to_apply:
            for target_path, new_code, rel_path in patches_to_apply:
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(new_code)
                logger.info(f"✅ Successfully applied patch to {rel_path}.")
                
            # 3. VERIFICATION PIPELINE (The final test)
            if not getattr(self, "run_linter_fix_loop")(context_of_change=pr_content) or \
               not getattr(self, "run_and_verify_app")(context_of_change=pr_content):
                logger.error("❌ Verification failed. Changes will be rolled back.")
                return False
                
            self.session_context.append("Applied automated patch XML edits.")
            if os.path.exists(self.pr_file):
                os.remove(self.pr_file)
            return True
            
        return False
