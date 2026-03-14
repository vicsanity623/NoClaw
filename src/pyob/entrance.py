import ast
import atexit
import difflib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Optional

_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_pyob_package_root_dir = os.path.dirname(_current_file_dir)
if _pyob_package_root_dir not in sys.path:
    sys.path.insert(0, _pyob_package_root_dir)

from pyob.autoreviewer import AutoReviewer  # noqa: E402
from pyob.core_utils import CoreUtilsMixin  # noqa: E402
from pyob.entrance_mixins import EntranceMixin  # noqa: E402
from pyob.pyob_code_parser import CodeParser  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Loads configuration and validates gemini_keys.

    As per MEMORY.md, non-empty gemini_keys are required.
    This function checks for the GEMINI_API_KEY environment variable.
    """
    config = {}
    # Define paths to search for config.json, in order of increasing precedence
    config_paths = [
        os.path.join(_current_file_dir, "config.json"),
        os.path.join(_pyob_package_root_dir, "config.json"),
        os.path.join(os.getcwd(), "config.json"),
    ]

    # Merge configurations from files
    for path in config_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    file_config = json.load(f)
                    config.update(file_config)  # Merge, later files override earlier
            except json.JSONDecodeError as e:
                logger.warning(f"WARNING: Could not parse config.json at {path}: {e}")

    # Environment variable overrides config.json value
    gemini_key_env = os.environ.get("PYOB_GEMINI_KEYS")
    if gemini_key_env:
        config["gemini_api_key"] = gemini_key_env
    # Now, gemini_key should be retrieved from the merged config for validation
    gemini_key = config.get("gemini_api_key")
    if not gemini_key:
        logger.critical(
            "CRITICAL ERROR: GEMINI_API_KEY environment variable is not set or is empty."
        )
        logger.critical(
            "Please set the GEMINI_API_KEY environment variable to proceed."
        )
        sys.exit(1)
    return config


class EntranceController(EntranceMixin, CoreUtilsMixin):
    ENGINE_FILES = [
        "apply_xml_mixins.py",
        "autoreviewer.py",
        "core_utils.py",
        "dashboard_html.py",
        "dashboard_server.py",
        "entrance.py",
        "entrance_mixins.py",
        "feature_mixins.py",
        "models.py",
        "prompts_and_memory.py",
        "pyob_code_parser.py",
        "pyob_dashboard.py",
        "pyob_launcher.py",
        "reviewer_mixins.py",
        "scanner_mixins.py",
        "get_valid_edit.py",
        "targeted_reviewer.py",
    ]

    def __init__(self, target_dir: str, dashboard_active: bool = True):
        self.target_dir = os.path.abspath(target_dir)
        self.pyob_dir = os.path.join(self.target_dir, ".pyob")
        os.makedirs(self.pyob_dir, exist_ok=True)
        self.skip_dashboard = ("--no-dashboard" in sys.argv) or (not dashboard_active)

        self.analysis_path = os.path.join(self.pyob_dir, "ANALYSIS.md")
        self.history_path = os.path.join(self.pyob_dir, "HISTORY.md")
        self.symbols_path = os.path.join(self.pyob_dir, "SYMBOLS.json")
        self.memory_path = os.path.join(self.pyob_dir, "MEMORY.md")
        self.llm_engine = AutoReviewer(self.target_dir)
        self.code_parser = CodeParser()
        self.ledger = self.load_ledger()
        self.cascade_queue: list[str] = []
        self.cascade_diffs: dict[str, str] = {}
        self.self_evolved_flag: bool = False
        self.manual_target_file: Optional[str] = None
        self.dashboard_process: Optional[subprocess.Popen] = None

        self.current_iteration = 1

        if not self.skip_dashboard:
            logger.info(
                "Dashboard active: Initializing with EntranceController instance."
            )
            self.start_dashboard()

    def set_manual_target_file(self, file_path: str) -> tuple[bool, str]:
        """Sets a file path to be targeted in the next iteration, overriding LLM choice."""
        abs_path = os.path.join(self.target_dir, file_path)
        if os.path.exists(abs_path):
            self.manual_target_file = file_path
            logger.info(f"Manual target set for next iteration: {file_path}")
            return True, f"Manual target set for next iteration: {file_path}"
        else:
            logger.warning(f"Manual target file not found: {file_path}")
            return False, f"Error: File not found at path: {file_path}"

    def start_dashboard(self):
        """Starts the Flask dashboard server in a separate process."""
        logger.info("Starting PyOB Dashboard server...")
        try:
            env = os.environ.copy()
            env["PYOB_DIR"] = self.pyob_dir  # Pass the .pyob directory path
            self.dashboard_process = subprocess.Popen(
                [sys.executable, "-m", "pyob.dashboard_server"],
                cwd=self.target_dir,  # Run from target_dir so relative paths (if any) are correct
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,  # Pass the modified environment
            )
            logger.info(
                f"Dashboard server started with PID: {self.dashboard_process.pid}"
            )
            # Register a cleanup function to terminate the dashboard process on exit
            atexit.register(self._terminate_dashboard_process)
        except Exception as e:
            logger.error(f"Failed to start dashboard server: {e}")
            self.dashboard_process = None

    def _terminate_dashboard_process(self):
        """Terminates the dashboard server process if it's running."""
        if self.dashboard_process and self.dashboard_process.poll() is None:
            logger.info("Terminating PyOB Dashboard server...")
            self.dashboard_process.terminate()
            try:
                self.dashboard_process.wait(timeout=5)  # Give it some time to shut down
            except subprocess.TimeoutExpired:
                self.dashboard_process.kill()
                logger.warning("Dashboard server did not terminate gracefully, killed.")
            logger.info("PyOB Dashboard server terminated.")

    def sync_with_remote(self) -> bool:
        """Fetches remote updates and merges main if we are behind."""
        if not os.path.exists(os.path.join(self.target_dir, ".git")):
            return False

        logger.info("Checking for remote updates from main...")
        self._run_git_command(["git", "fetch", "origin"])

        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD..origin/main"],
            cwd=self.target_dir,
            capture_output=True,
            text=True,
        )

        commits_behind = int(result.stdout.strip() or 0)

        if commits_behind > 0:
            logger.warning(
                f"Project is behind main by {commits_behind} commits. Syncing..."
            )

            if self._run_git_command(["git", "merge", "origin/main"]):
                logger.info("Sync complete. Local files updated.")
                return True
            else:
                logger.error(
                    "Sync failed (likely a merge conflict). Manual intervention required."
                )

        return False

    def reboot_pyob(self):
        """Verified Hot-Reboot: Checks for syntax/import errors before restarting."""
        logger.info("PRE-FLIGHT: Verifying engine integrity before reboot...")

        test_cmd = [sys.executable, "-c", "import pyob.entrance; print('SUCCESS')"]
        env = os.environ.copy()
        # Ensure the directory containing the 'pyob' package is in PYTHONPATH for the subprocess
        current_pythonpath_list = env.get("PYTHONPATH", "").split(os.pathsep)
        if _pyob_package_root_dir not in current_pythonpath_list:
            if env.get("PYTHONPATH"):
                env["PYTHONPATH"] = (
                    f"{_pyob_package_root_dir}{os.pathsep}{env['PYTHONPATH']}"
                )
            else:
                env["PYTHONPATH"] = _pyob_package_root_dir

        try:
            result = subprocess.run(test_cmd, capture_output=True, text=True, env=env)
            if "SUCCESS" in result.stdout:
                logger.warning(
                    "SELF-EVOLUTION COMPLETE: Rebooting fresh PYOB engine..."
                )
                os.execv(
                    sys.executable,
                    [sys.executable, "-m", "pyob.pyob_launcher", self.target_dir],
                )
            else:
                logger.error(
                    f"REBOOT ABORTED: The evolved code has import/syntax errors:\n{result.stderr}"
                )
                self.self_evolved_flag = False
        except Exception as e:
            logger.error(f"Pre-flight check failed: {e}")
            self.self_evolved_flag = False

    def trigger_production_build(self):
        """Advanced Build: Compiles PYOB into a DMG and replaces the system version."""
        build_script = os.path.join(self.target_dir, "build_pyinstaller_multiOS.py")
        if not os.path.exists(build_script):
            logger.error("Build script not found. Skipping production deploy.")
            return

        logger.info("STARTING PRODUCTION BUILD... This will take 2-3 minutes.")
        try:
            subprocess.run([sys.executable, build_script], check=True)

            app_name = "PyOuroBoros.app"
            dist_path = os.path.join(self.target_dir, "dist", app_name)
            applications_path = f"/Applications/{app_name}"

            if sys.platform == "darwin" and os.path.exists(dist_path):
                logger.warning(
                    f"FORGE COMPLETE: Deploying new version to {applications_path}..."
                )

                if os.path.exists(applications_path):
                    shutil.rmtree(applications_path)
                shutil.copytree(dist_path, applications_path)

                subprocess.Popen(
                    ["open", "-a", applications_path, "--args", self.target_dir],
                    shell=True,
                )

                logger.info("NEW VERSION RELAYED. ENGINE SHUTTING DOWN.")
                sys.exit(0)

        except Exception as e:
            logger.error(f"Production Build Failed: {e}")

    def load_ledger(self):
        if os.path.exists(self.symbols_path):
            try:
                with open(self.symbols_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(
                    f"Failed to load SYMBOLS.json, initializing empty ledger: {e}"
                )
        return {"definitions": {}, "references": {}}

    def save_ledger(self):
        with open(self.symbols_path, "w", encoding="utf-8") as f:
            json.dump(self.ledger, f, indent=2)

    def run_master_loop(self):
        logger.info(
            "\n" + "=" * 60 + "\nENTRANCE CONTROLLER: SYMBOLIC MODE ACTIVE\n" + "=" * 60
        )
        if not os.path.exists(self.analysis_path):
            self.build_initial_analysis()

        iteration = 1
        while True:
            self.current_iteration = iteration
            self.self_evolved_flag = False

            if self.sync_with_remote():
                res = subprocess.run(
                    ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"],
                    cwd=self.target_dir,
                    capture_output=True,
                    text=True,
                )
                changed_files = res.stdout.strip().splitlines()
                if any(os.path.basename(f) in self.ENGINE_FILES for f in changed_files):
                    logger.warning(
                        "REMOTE EVOLUTION: Engine files updated via sync. Rebooting..."
                    )
                    self.self_evolved_flag = True

            if self.self_evolved_flag:
                if getattr(sys, "frozen", False):
                    logger.warning("COMPILED ENGINE EVOLVED: Initiating Forge Build.")
                    self.trigger_production_build()
                else:
                    logger.warning(" SCRIPT ENGINE EVOLVED: Initiating Hot-Reboot.")
                    self.reboot_pyob()

            logger.info(
                f"\n\n{'=' * 70}\nTargeted Pipeline Loop (Iteration {iteration})\n{'=' * 70}"
            )

            try:
                self.execute_targeted_iteration(iteration)
            except KeyboardInterrupt:
                logger.info("\nExiting Entrance Controller...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in master loop: {e}", exc_info=True)

            iteration += 1
            logger.info("Iteration complete. Waiting for system cooldown...")
            time.sleep(120)

    def _extract_path_from_llm_response(self, text: str) -> str:
        """Extracts a clean relative file path from a potentially conversational LLM response."""
        cleaned_text = re.sub(r"[`\"*]", "", text).strip()
        if " " in cleaned_text:
            parts = cleaned_text.split()
            for part in parts:
                if "/" in part or part.endswith((".py", ".js", ".ts", ".html", ".css")):
                    return part
            return parts[0]
        return cleaned_text

    def _run_git_command(self, cmd: list[str]) -> bool:
        """Helper to run git commands safely."""
        try:
            result = subprocess.run(
                cmd, cwd=self.target_dir, capture_output=True, text=True
            )
            if result.returncode != 0:
                logger.warning(
                    f"Git Command Failed: {' '.join(cmd)}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Git Execution Error: {e}")
            return False

    def handle_git_librarian(self, rel_path: str, iteration: int):
        """Creates a branch, generates an AI summary, and opens a professional PR."""
        if not os.path.exists(os.path.join(self.target_dir, ".git")):
            return

        # 1. Capture the raw diff before committing
        diff_proc = subprocess.run(
            ["git", "diff", "HEAD", rel_path],
            cwd=self.target_dir, capture_output=True, text=True
        )
        diff_text = diff_proc.stdout or "Minor structural refinement."

        # 2. Ask the AI to write the PR summary
        summary = self.generate_pr_summary(rel_path, diff_text)
        title = summary.get("title", f"Evolution: Refactor of {rel_path}")
        body = summary.get("body", f"This PR was automatically generated by PyOB.\n\nFile: `{rel_path}`")

        # 3. Git Operations
        timestamp = int(time.time())
        branch_name = f"pyob-evolution-v{iteration}-{timestamp}"
        
        logger.info(f" LIBRARIAN: Publishing Evolution: {title}")

        self._run_git_command(["git", "checkout", "-b", branch_name])
        self._run_git_command(["git", "add", rel_path])
        
        # Use the AI-generated title as the commit message
        if not self._run_git_command(["git", "commit", "-m", title]):
            return

        if shutil.which("gh"):
            logger.info("Pushing to GitHub and opening Pull Request...")
            self._run_git_command(["git", "push", "origin", branch_name])
            self._run_git_command([
                "gh", "pr", "create",
                "--title", title,
                "--body", body,
                "--base", "main"
            ])
        else:
            logger.warning("GitHub CLI (gh) not found. Committed locally only.")

    def _run_final_verification_and_heal(self, backup_state: dict) -> bool:
        """
        Runs the project entry file for 10 seconds. Supports Python,
        Node.js (package.json), and static HTML projects.
        """
        entry_file = self.llm_engine._find_entry_file()
        if not entry_file:
            logger.warning("No main entry file found. Skipping runtime test.")
            return True

        rel_entry_file = os.path.relpath(entry_file, self.target_dir)

        for attempt in range(3):
            logger.info(
                f"Launching `{rel_entry_file}` for test... (Attempt {attempt + 1}/3)"
            )

            cmd: list[str] = []
            is_html = entry_file.endswith((".html", ".htm"))
            is_js = entry_file.endswith((".js", "package.json"))
            is_py = entry_file.endswith(".py")

            if is_py:
                venv_python = os.path.join(
                    self.target_dir, "build_env", "bin", "python3"
                )
                python_cmd = (
                    venv_python if os.path.exists(venv_python) else sys.executable
                )
                cmd = [python_cmd, entry_file]
            elif is_js:
                cmd = (
                    ["npm", "start"]
                    if entry_file.endswith("package.json")
                    else ["node", entry_file]
                )
            elif is_html:
                # SAFE GUARD: Skip opening browser in GitHub Actions
                if os.environ.get("GITHUB_ACTIONS") == "true":
                    logger.info("Skipping browser launch in GitHub Actions.")
                    return True

                # Command to open browser based on OS
                if sys.platform == "darwin":
                    cmd = ["open", entry_file]
                elif sys.platform == "win32":
                    cmd = [f'start "" "{entry_file}"']
                else:
                    cmd = ["xdg-open", entry_file]

            if not cmd:
                return True

            use_shell = bool(cmd and (cmd[0].startswith("start") or cmd[0] == "open"))

            start_time = time.time()
            try:
                # We use a context manager for popen to ensure cleanup
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=self.target_dir,
                    shell=use_shell,
                    close_fds=sys.platform != "win32",
                )

                if is_html:
                    time.sleep(5)
                    return True

                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                stdout, stderr = process.communicate()
            except Exception as e:
                logger.error(f"Execution failed: {e}")
                stdout, stderr = "", str(e)

            duration = time.time() - start_time

            # Check for common crash keywords in output
            has_error_logs = any(
                kw in stderr or kw in stdout
                for kw in [
                    "Traceback",
                    "Exception",
                    "Error:",
                    "ModuleNotFoundError",
                    "ImportError",
                    "ReferenceError",
                ]
            )
            is_crash_code = process.returncode not in (0, 15, -15, None)

            if is_crash_code or has_error_logs:
                logger.warning(f"App crashed after {duration:.1f}s!")
                if attempt < 2:
                    # Let the AI try to fix the crash
                    self.llm_engine._fix_runtime_errors(
                        stderr + "\n" + stdout, entry_file
                    )
                else:
                    logger.error(" Exhausted all 3 auto-fix attempts.")
            else:
                logger.info(f"App ran successfully for {duration:.1f}s.")
                return True

        # If we got through all 3 attempts and still failed, rollback
        self.llm_engine.restore_workspace(backup_state)
        return False

    def detect_symbolic_ripples(
        self, old: str, new: str, source_file: str
    ) -> list[str]:
        diff = list(difflib.unified_diff(old.splitlines(), new.splitlines()))
        changed_text = "\n".join(
            [line for line in diff if line.startswith("+") or line.startswith("-")]
        )
        potential_symbols = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", changed_text))
        impacted_files = []
        for sym in potential_symbols:
            if self.ledger["definitions"].get(sym) == source_file:
                for target_file, refs in self.ledger["references"].items():
                    if sym in refs and target_file != source_file:
                        impacted_files.append(target_file)
        return list(set(impacted_files))

    def pick_target_file(self) -> str:
        if self.manual_target_file:
            target = self.manual_target_file
            self.manual_target_file = None
            logger.info(f" Using manual target: {target}")
            return target

        analysis = self._read_file(self.analysis_path)
        history = self._read_file(self.history_path) or "No history yet."

        last_file = ""
        history_lines = history.strip().split("\n")
        for line in reversed(history_lines):
            if line.startswith("## "):
                match = re.search(r"`([^`]+)`", line)
                if match:
                    last_file = match.group(1)
                    break

        prompt = f"""
Choose ONE relative file path to review next based on the ANALYSIS.md and HISTORY.md.

STRATEGIC RULES:
1. DO NOT pick `{last_file}`. It was just edited.
2. If logic was recently added to a backend file (like `logic.py`), your MUST pick the UI or Main file (like `main.py`) next to implement a way for the user to access it.
3. If a configuration file was changed, pick a file that depends on it.
4. Rotate between logic, UI, and styles to ensure features are complete and visible.

OUTPUT FORMAT RULES (MANDATORY):
- Reply ONLY with the raw relative file path.
- DO NOT use Markdown, quotes, bold, or explanation.
- DO NOT say "I suggest...". Just the path.

Example Output:
src/pyob/core_utils.py

### Analysis:
{analysis}
### History:
{history}
"""

        def val(text: str) -> bool:
            """Smarter validation that extracts a path from conversational AI."""
            extracted_path = self._extract_path_from_llm_response(text)
            exists = os.path.exists(os.path.join(self.target_dir, extracted_path))
            not_duplicate = extracted_path != last_file
            return exists and not_duplicate

        response = self.llm_engine.get_valid_llm_response(
            prompt, val, context="Target Selector"
        )

        final_path = self._extract_path_from_llm_response(response)

        return str(final_path)

    def build_initial_analysis(self):
        logger.info("ANALYSIS.md not found. Bootstrapping Deep Symbolic Scan...")
        all_files = sorted(self.llm_engine.scan_directory())
        structure_map = "\n".join(
            os.path.relpath(f, self.target_dir) for f in all_files
        )
        proj_prompt = f"Write a 2-sentence summary of this project based on these files:\n{structure_map}"
        project_summary = self.llm_engine.get_valid_llm_response(
            proj_prompt, lambda t: len(t) > 5, context="Project Genesis"
        ).strip()
        content = f"# Project Analysis\n\n**Project Summary:**\n{project_summary}\n\n---\n\n## 📂 File Directory\n\n"
        file_structures = {}
        for f_path in all_files:
            rel = os.path.relpath(f_path, self.target_dir)
            with open(f_path, "r", encoding="utf-8", errors="ignore") as f:
                full_code = f.read()
            self.update_ledger_for_file(rel, full_code)
            file_structures[rel] = self.code_parser.generate_structure_dropdowns(
                f_path, full_code
            )

        logger.info(f"Batching Deep Symbolic Parsing for {len(all_files)} files...")
        batch_prompt = "Provide a succinct one-sentence plain text summary for EACH of the following files based on their structure. CRITICAL: Output MUST be strictly in 'filepath: summary' format on each line. Do NOT include any HTML tags, markdown, <details> blocks, or code signatures.\n\n"
        for rel_path, struct in file_structures.items():
            batch_prompt += f"File `{rel_path}` Structure:\n{struct}\n\n"

        batch_response = self.llm_engine.get_valid_llm_response(
            batch_prompt, lambda t: ":" in t and len(t) > 10, context="Batch Genesis"
        ).strip()

        summaries = {}
        for line in batch_response.splitlines():
            if ":" in line:
                path_part, summary_part = line.split(":", 1)
                path_clean = path_part.replace("`", "").replace("*", "").strip()
                summaries[path_clean] = summary_part.strip()

        for f_path in all_files:
            rel = os.path.relpath(f_path, self.target_dir)
            structured_summary = summaries.get(rel, "No summary generated.")
            if structured_summary == "No summary generated.":
                for k, v in summaries.items():
                    if k in rel or rel in k:
                        structured_summary = v
                        break
            content += f"### `{rel}`\n**Summary:** {structured_summary}\n\n{file_structures[rel]}\n---\n"

        with open(self.analysis_path, "w", encoding="utf-8") as f:
            f.write(content)
        self.save_ledger()
        logger.info(" ANALYSIS.md and SYMBOLS.json successfully initialized.")

    def update_analysis_for_single_file(self, target_abs_path: str, rel_path: str):
        if not os.path.exists(self.analysis_path):
            return
        with open(target_abs_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        structure = self.code_parser.generate_structure_dropdowns(target_abs_path, code)
        sum_prompt = f"Provide a one-sentence plain text summary of what the file `{rel_path}` does. \n\nCRITICAL: Do NOT include any HTML, <details> tags, or code in your response.\n\nStructure:\n{structure}"
        desc = self.llm_engine.get_valid_llm_response(
            sum_prompt, lambda t: "<details>" not in t and len(t) > 5, context=rel_path
        ).strip()
        new_block = f"### `{rel_path}`\n**Summary:** {desc}\n\n" + structure + "\n---\n"
        with open(self.analysis_path, "r", encoding="utf-8") as f:
            analysis_text = f.read()
        pattern = rf"### `{re.escape(rel_path)}`.*?(?=### `|---\n\Z)"
        updated = re.sub(pattern, new_block, analysis_text, flags=re.DOTALL)
        with open(self.analysis_path, "w", encoding="utf-8") as f:
            f.write(updated)

    def update_ledger_for_file(self, rel_path: str, code: str):
        ext = os.path.splitext(rel_path)[1]
        definitions_to_remove = [
            name
            for name, path in self.ledger["definitions"].items()
            if path == rel_path
        ]
        for name in definitions_to_remove:
            del self.ledger["definitions"][name]

        if ext == ".py":
            try:
                tree = ast.parse(code)
                for n in ast.walk(tree):
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                        self.ledger["definitions"][n.name] = rel_path
                    elif isinstance(n, ast.Assign):
                        for target in n.targets:
                            if isinstance(target, ast.Name) and target.id.isupper():
                                self.ledger["definitions"][target.id] = rel_path
            except Exception as e:
                logger.warning(f"Failed to parse Python AST for {rel_path}: {e}")
        elif ext in [".js", ".ts"]:
            defs = re.findall(
                r"(?:export\s+|async\s+)?(?:function\*?|class|const|var|let)\s+([a-zA-Z0-9_$]+)",
                code,
            )
            for d in defs:
                if len(d) > 3:
                    self.ledger["definitions"][d] = rel_path

        if rel_path in self.ledger["references"]:
            del self.ledger["references"][rel_path]

        # Capture any word-like identifier of at least 4 characters for potential references
        potential_refs = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b", code)
        self.ledger["references"][rel_path] = list(set(potential_refs))
        self.save_ledger()

    def append_to_history(self, rel_path: str, old_code: str, new_code: str):
        diff_lines = list(
            difflib.unified_diff(
                old_code.splitlines(keepends=True),
                new_code.splitlines(keepends=True),
                fromfile="Original",
                tofile="Proposed",
            )
        )
        if not diff_lines:
            return
        summary_diff = (
            "".join(diff_lines[:5])
            + "\n... [TRUNCATED] ...\n"
            + "".join(diff_lines[-5:])
            if len(diff_lines) > 20
            else "".join(diff_lines)
        )
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(
                f"\n## {timestamp} - `{rel_path}`\n```diff\n{summary_diff}\n```\n---\n"
            )

    def _read_file(self, f: str) -> str:
        if os.path.exists(f):
            try:
                with open(f, "r", encoding="utf-8", errors="ignore") as f_obj:
                    return f_obj.read()
            except Exception:
                return ""
        return ""


if __name__ == "__main__":
    EntranceController(sys.argv[1] if len(sys.argv) > 1 else ".").run_master_loop()
