import os
import re
import shutil
import subprocess
import sys

from .core_utils import (
    IGNORE_DIRS,
    IGNORE_FILES,
    logger,
)


class ValidationMixin:
    target_dir: str
    session_context: list[str]
    memory: str

    def run_linter_fix_loop(self, context_of_change: str = "") -> bool:
        logger.info("\nValidating codebase syntax (Python, JS, CSS)...")
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
                logger.warning(f"Ruff found logic errors:\n{res.stdout.strip()}")
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
                        logger.info("Python Auto-fix successful!")
                        py_fixed = True
                        break
                if not py_fixed:
                    logger.error("Python errors remain unfixable.")
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
                    logger.warning(f"JS Syntax Error in {rel_name}:\n{err_msg}")
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
                            logger.info(f"JS Auto-fix successful for {rel_name}!")
                            js_fixed = True
                            break
                    if not js_fixed:
                        logger.error(f"JS syntax in {rel_name} remains broken.")
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
                                f"CSS Syntax Error in {file}: Unbalanced braces."
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
        check_script = os.path.join(self.target_dir, "check.sh")

        if os.path.exists(check_script):
            logger.info("PHASE 3.5: Running full validation suite (check.sh)...")
            try:
                os.chmod(check_script, 0o755)

                subprocess.run(
                    [check_script, "--fix"],
                    capture_output=True,
                    text=True,
                    cwd=self.target_dir,
                )

                res = subprocess.run(
                    [check_script], capture_output=True, text=True, cwd=self.target_dir
                )

                if res.returncode != 0:
                    logger.warning(
                        f"Validation suite failed after auto-fix!\n{res.stdout.strip()}"
                    )
                    self._fix_runtime_errors(
                        res.stdout + "\n" + res.stderr,
                        "Validation Suite",
                        context_of_change,
                    )
                    return False
            except Exception as e:
                logger.error(f"Failed to execute validation script: {e}")
                return False
        else:
            logger.warning("No check.sh found in target project. Skipping PHASE 3.5.")

        entry_file = getattr(self, "_find_entry_file")()
        if not entry_file:
            logger.warning("No entry point detected. Skipping runtime smoke test.")
            return True

        venv_python = os.path.join(self.target_dir, "build_env", "bin", "python3")
        if not os.path.exists(venv_python):
            venv_python = os.path.join(self.target_dir, "venv", "bin", "python3")

        python_cmd = venv_python if os.path.exists(venv_python) else sys.executable

        for attempt in range(3):
            logger.info(
                f"\nPHASE 4: Runtime Verification. Launching {os.path.basename(entry_file)} (Attempt {attempt + 1}/3)..."
            )

            is_html = entry_file.endswith((".html", ".htm"))

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
                "Traceback",
                "Exception:",
                "Error:",
                "NameError:",
                "AttributeError:",
            ]

            if is_html:
                logger.info("HTML Entry detected. Verification assumed successful.")
                return True

            has_crash = any(kw in stderr or kw in stdout for kw in error_keywords) or (
                process.returncode != 0 and process.returncode not in (0, 15, -15, None)
            )

            if not has_crash:
                logger.info("App ran successfully for 10 seconds.")
                return True

            logger.warning(f"App crashed!\n{stderr}")
            self._fix_runtime_errors(
                stderr + "\n" + stdout, entry_file, context_of_change
            )

        logger.error("Exhausted runtime auto-fix attempts.")
        return False

    def _fix_runtime_errors(
        self, logs: str, entry_file: str, context_of_change: str = ""
    ):
        """Detects crashes. Handles missing packages automatically, otherwise asks AI."""

        # --- 0. SANITIZE TARGET ---
        # Prevent the bot from hallucinating log headers as filenames
        if entry_file == "Validation Suite" or "Validation Suite" in entry_file:
            # Fallback to searching for a real file in the logs or use the project main
            found_file = getattr(self, "_find_entry_file")()
            entry_file = found_file if found_file else "main.js"

        package_match = re.search(r"ModuleNotFoundError: No module named '(.*?)'", logs)
        if not package_match:
            package_match = re.search(r"ImportError: No module named '(.*?)'", logs)

        if package_match:
            pkg = package_match.group(1)
            logger.info(
                f"Auto-detected missing dependency: {pkg}. Attempting pip install..."
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
                    f"Successfully installed {pkg}. System will now retry launch."
                )

                # --- AUTO-DEPENDENCY LOCKING ---
                try:
                    req_path = os.path.join(
                        getattr(self, "target_dir"), "requirements.txt"
                    )
                    with open(req_path, "w", encoding="utf-8") as f_req:
                        subprocess.run(
                            [python_cmd, "-m", "pip", "freeze"],
                            stdout=f_req,
                            check=True,
                        )
                    logger.info("Auto-locked dependencies in requirements.txt")
                except Exception as e:
                    logger.warning(f"Failed to lock dependencies: {e}")
                # -------------------------------

                return
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {pkg} automatically: {e}")

        # --- 1. SMART FILE IDENTIFICATION ---
        # Look for Python tracebacks first
        tb_files = re.findall(r'File "([^"]+)"', logs)

        # NEW: Look for JavaScript/HTML filenames if Python search fails
        if not tb_files:
            tb_files = re.findall(r"([\w\-/]+\.(?:js|html|css))", logs)

        target_file = entry_file
        for f in reversed(tb_files):
            # Block the "Validation Suite" hallucination here too
            if "Validation Suite" in f:
                continue

            abs_f = (
                os.path.abspath(f)
                if os.path.isabs(f)
                else os.path.join(self.target_dir, f)
            )

            if (
                abs_f.startswith(self.target_dir)
                and not any(ign in abs_f for ign in getattr(self, "IGNORE_DIRS", []))
                and os.path.exists(abs_f)
            ):
                target_file = abs_f
                break

        # Ensure we have a valid path before proceeding
        if not os.path.exists(target_file):
            target_file = entry_file

        rel_path = os.path.relpath(target_file, self.target_dir)

        # --- 2. LOAD CODE AND PROMPT ---
        with open(target_file, "r", encoding="utf-8", errors="ignore") as f_obj:
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
                f"### Project Memory / Context:\n{getattr(self, 'memory', '')}\n\n"
                if getattr(self, "memory", None)
                else ""
            )
            prompt = getattr(self, "load_prompt")(
                "FRE.md",
                memory_section=memory_section,
                logs=logs[-2000:],
                rel_path=rel_path,
                code=code,
            )

        # --- 3. EXECUTE REPAIR ---
        new_code, explanation, _ = getattr(self, "get_valid_edit")(
            prompt, code, require_edit=True, target_filepath=target_file
        )

        if new_code != code:
            with open(target_file, "w", encoding="utf-8") as f_out:
                f_out.write(new_code)
            logger.info(f"AI Auto-patched runtime crash in `{rel_path}`")
            self.session_context.append(
                f"Auto-fixed runtime crash in `{rel_path}`: {explanation}"
            )
            if hasattr(self, "run_linter_fix_loop"):
                self.run_linter_fix_loop()

    def check_downstream_breakages(self, target_path: str, rel_path: str) -> bool:
        logger.info(
            f"\nPHASE 3: Simulating workspace to check for downstream breakages caused by {rel_path} edits..."
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
                    f"Downstream Breakage Detected!\n{result.stdout.strip()}"
                )
                return self.propose_cascade_fix(result.stdout.strip(), rel_path)
            logger.info("No downstream breakages detected.")
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
                f"Auto-patched cascading fix directly into `{os.path.basename(problem_file)}`"
            )
            self.session_context.append(
                f"Auto-applied downstream cascade fix in `{os.path.basename(problem_file)}`"
            )
            self.run_linter_fix_loop()
            final_check = subprocess.run(
                ["mypy", problem_file, "--ignore-missing-imports"], capture_output=True
            )
            return final_check.returncode == 0
        logger.error(f"Failed to auto-patch `{rel_broken_path}`.")
        return False
