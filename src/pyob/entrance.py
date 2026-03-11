import ast
import difflib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import HTTPServer
from pathlib import Path
from typing import Optional

from pyob.autoreviewer import AutoReviewer
from pyob.pyob_code_parser import CodeParser
from pyob.pyob_dashboard import OBSERVER_HTML, ObserverHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


class TargetedReviewer(AutoReviewer):
    def __init__(self, target_dir: str, target_file: str):
        super().__init__(target_dir)
        self.forced_target_file = target_file

    def scan_directory(self) -> list[str]:
        if os.path.exists(self.forced_target_file):
            return [self.forced_target_file]
        return []


class EntranceController:
    ENGINE_FILES = [
        "autoreviewer.py",
        "core_utils.py",
        "prompts_and_memory.py",
        "entrance.py",
        "reviewer_mixins.py",
        "pyob_code_parser.py",
        "pyob_dashboard.py",
    ]

    def __init__(self, target_dir: str):
        self.target_dir = os.path.abspath(target_dir)
        self.pyob_dir = os.path.join(self.target_dir, ".pyob")
        os.makedirs(self.pyob_dir, exist_ok=True)
        self.skip_dashboard = "--no-dashboard" in sys.argv
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

        self.current_iteration = 1

        if not self.skip_dashboard:
            self.start_dashboard()

    def start_dashboard(self):
        # 1. Save to the internal .pyob folder
        obs_path = os.path.join(self.pyob_dir, "observer.html")
        with open(obs_path, "w", encoding="utf-8") as f:
            f.write(OBSERVER_HTML)

        # 2. Initialize and Start the Live Server
        ObserverHandler.controller = self

        def run_server():
            try:
                server = HTTPServer(("localhost", 5000), ObserverHandler)
                server.serve_forever()
            except Exception as e:
                logger.error(f"Dashboard failed to start: {e}")

        threading.Thread(target=run_server, daemon=True).start()

        print("\n" + "=" * 60)
        print("⚡ PyOuroBoros (PyOB) OBSERVER IS LIVE")
        print("🔗 URL: http://localhost:5000")
        print(f"📂 FILE: {obs_path}")
        print("=" * 60 + "\n")

    def set_manual_target_file(self, file_path: str):
        """Sets a file path to be targeted in the next iteration, overriding LLM choice."""
        abs_path = os.path.join(self.target_dir, file_path)
        if os.path.exists(abs_path):
            self.manual_target_file = file_path
            logger.info(f"Ã°Å¸Å½Â¯ Manual target set for next iteration: {file_path}")
        else:
            logger.warning(f"Ã¢ Å Manual target file not found: {file_path}")

    def sync_with_remote(self) -> bool:
        """Fetches remote updates and merges main if we are behind."""
        if not os.path.exists(os.path.join(self.target_dir, ".git")):
            return False

        logger.info("📡 Checking for remote updates from main...")
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
                f"🔄 Project is behind main by {commits_behind} commits. Syncing..."
            )

            if self._run_git_command(["git", "merge", "origin/main"]):
                logger.info("✅ Sync complete. Local files updated.")
                return True
            else:
                logger.error(
                    "❌ Sync failed (likely a merge conflict). Manual intervention required."
                )

        return False

    def reboot_pyob(self):
        """Verified Hot-Reboot: Checks for syntax/import errors before restarting."""
        logger.info("🧪 PRE-FLIGHT: Verifying engine integrity before reboot...")

        # Test if the new code can actually be loaded
        test_cmd = [sys.executable, "-c", "import pyob.entrance; print('SUCCESS')"]
        env = os.environ.copy()
        # Ensure 'src' is in the path for the test
        env["PYTHONPATH"] = os.path.join(self.target_dir, "src")

        try:
            result = subprocess.run(test_cmd, capture_output=True, text=True, env=env)
            if "SUCCESS" in result.stdout:
                logger.warning(
                    "🔄 SELF-EVOLUTION COMPLETE: Rebooting fresh PYOB engine..."
                )
                os.execv(
                    sys.executable,
                    [sys.executable, "-m", "pyob.pyob_launcher", self.target_dir],
                )
            else:
                logger.error(
                    f"🚫 REBOOT ABORTED: The evolved code has import/syntax errors:\n{result.stderr}"
                )
                self.self_evolved_flag = False  # Cancel reboot to stay alive
        except Exception as e:
            logger.error(f"❌ Pre-flight check failed: {e}")
            self.self_evolved_flag = False

    def trigger_production_build(self):
        """Advanced Build: Compiles PYOB into a DMG and replaces the system version."""
        build_script = os.path.join(self.target_dir, "build_pyinstaller_multiOS.py")
        if not os.path.exists(build_script):
            logger.error("❌ Build script not found. Skipping production deploy.")
            return

        logger.info("🛠️ STARTING PRODUCTION BUILD... This will take 2-3 minutes.")
        try:
            subprocess.run([sys.executable, build_script], check=True)

            app_name = "PyOuroBoros.app"
            dist_path = os.path.join(self.target_dir, "dist", app_name)
            applications_path = f"/Applications/{app_name}"

            if sys.platform == "darwin" and os.path.exists(dist_path):
                logger.warning(
                    f"🚀 FORGE COMPLETE: Deploying new version to {applications_path}..."
                )

                if os.path.exists(applications_path):
                    shutil.rmtree(applications_path)
                shutil.copytree(dist_path, applications_path)

                subprocess.Popen(
                    ["open", "-a", applications_path, "--args", self.target_dir],
                    shell=True,
                )

                logger.info("🔥 NEW VERSION RELAYED. ENGINE SHUTTING DOWN.")
                sys.exit(0)

        except Exception as e:
            logger.error(f"❌ Production Build Failed: {e}")

    def load_ledger(self):
        if os.path.exists(self.symbols_path):
            try:
                with open(self.symbols_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(
                    f"Failed to load SYMBOLS.json, initializing empty ledger: {e}"
                )
        return {"definitions": {}, "references": {}}

    def save_ledger(self):
        with open(self.symbols_path, "w") as f:
            json.dump(self.ledger, f, indent=2)

    def run_master_loop(self):
        logger.info(
            "\n"
            + "=" * 60
            + "\n🧠 ENTRANCE CONTROLLER: SYMBOLIC MODE ACTIVE\n"
            + "=" * 60
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

                if any(ef in res.stdout for ef in self.ENGINE_FILES):
                    logger.warning(
                        "🧠 REMOTE EVOLUTION: Engine files updated via sync. Rebooting..."
                    )
                    self.self_evolved_flag = True

            if self.self_evolved_flag:
                if getattr(sys, "frozen", False):
                    logger.warning(
                        "💎 COMPILED ENGINE EVOLVED: Initiating Forge Build."
                    )
                    self.trigger_production_build()
                else:
                    logger.warning("🐍 SCRIPT ENGINE EVOLVED: Initiating Hot-Reboot.")
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
        """Creates a branch, commits the change, and opens a PR."""
        if not os.path.exists(os.path.join(self.target_dir, ".git")):
            return

        timestamp = int(time.time())
        branch_name = f"pyob-evolution-v{iteration}-{timestamp}"

        logger.info(f"📂 LIBRARIAN: Archiving change in branch `{branch_name}`...")

        if not self._run_git_command(["git", "checkout", "-b", branch_name]):
            return

        self._run_git_command(["git", "add", rel_path])

        commit_msg = f"PyOB Evolution: Automated refactor of `{rel_path}` (Iteration {iteration})"
        if not self._run_git_command(["git", "commit", "-m", commit_msg]):
            return

        if shutil.which("gh"):
            logger.info("🚀 Pushing to GitHub and opening Pull Request...")
            self._run_git_command(["git", "push", "origin", branch_name])
            self._run_git_command(
                [
                    "gh",
                    "pr",
                    "create",
                    "--title",
                    commit_msg,
                    "--body",
                    f"This PR was automatically generated by the PyOB Self-Evolution engine.\n\nFile modified: `{rel_path}`",
                    "--base",
                    "main",
                ]
            )
        else:
            logger.warning(
                "⚠️ GitHub CLI (gh) not found. Change committed locally but not pushed."
            )

    def execute_targeted_iteration(self, iteration: int):
        backup_state = self.llm_engine.backup_workspace()
        target_diff = ""
        if self.cascade_queue:
            target_rel_path = self.cascade_queue.pop(0)
            target_diff = self.cascade_diffs.get(target_rel_path, "")
            logger.warning(
                f"🔗 SYMBOLIC CASCADE: Targeting impacted file: {target_rel_path}"
            )
            is_cascade = True
        else:
            target_rel_path = self.pick_target_file()
            is_cascade = False

        if not target_rel_path:
            return

        is_engine_file = any(f in target_rel_path for f in self.ENGINE_FILES)

        if is_engine_file:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            project_name = os.path.basename(self.target_dir)
            base_backup_path = Path.home() / "Documents" / "PYOB_Backups" / project_name
            pod_path = base_backup_path / f"safety_pod_v{iteration}_{timestamp}"

            try:
                pod_path.mkdir(parents=True, exist_ok=True)
                logger.warning(
                    f"🛡️ SELF-EVOLUTION: Sheltering engine source EXTERNALLY in {pod_path}"
                )
                for f_name in self.ENGINE_FILES:
                    src = os.path.join(self.target_dir, "src", "pyob", f_name)
                    # Assuming src/pyob is the canonical location for engine files,
                    # consistent with PYTHONPATH setup in reboot_pyob.
                    if os.path.exists(src):
                        shutil.copy(src, str(pod_path))
            except Exception as e:
                logger.error(f"Failed to create external safety pod: {e}")

        target_abs_path = os.path.join(self.target_dir, target_rel_path)
        self.llm_engine.session_context = []
        if is_cascade and target_diff:
            msg = f"CRITICAL SYMBOLIC RIPPLE: This file depends on code that was just modified. Ensure this file is updated to support these changes:\n\n### DEPDENDENCY CHANGE DIFF:\n{target_diff}"
            self.llm_engine.session_context.append(msg)

        old_content = ""
        if os.path.exists(target_abs_path):
            with open(target_abs_path, "r", encoding="utf-8", errors="ignore") as f:
                old_content = f.read()

        reviewer = TargetedReviewer(self.target_dir, target_abs_path)
        reviewer.session_context = self.llm_engine.session_context[:]
        reviewer.run_pipeline(iteration)

        self.llm_engine.session_context = reviewer.session_context[:]

        new_content = ""
        if os.path.exists(target_abs_path):
            with open(target_abs_path, "r", encoding="utf-8", errors="ignore") as f:
                new_content = f.read()

        logger.info(f"🔄 Refreshing metadata for `{target_rel_path}`...")
        self.update_analysis_for_single_file(target_abs_path, target_rel_path)
        self.update_ledger_for_file(target_rel_path, new_content)

        if old_content != new_content:
            logger.info(
                f"📝 Edit successful. Checking ripples and running final verification for {target_rel_path}..."
            )
            self.append_to_history(target_rel_path, old_content, new_content)

            current_diff = "".join(
                difflib.unified_diff(
                    old_content.splitlines(keepends=True),
                    new_content.splitlines(keepends=True),
                )
            )

            ripples = self.detect_symbolic_ripples(
                old_content, new_content, target_rel_path
            )
            if ripples:
                logger.warning(
                    f"⚠️ CROSS-FILE DEPTH TRIGGERED! Queuing {len(ripples)} files."
                )
                for r in ripples:
                    if r not in self.cascade_queue:
                        self.cascade_queue.append(r)
                        self.cascade_diffs[r] = current_diff

            logger.info("\n" + "=" * 20 + " FINAL VERIFICATION " + "=" * 20)
            if not self._run_final_verification_and_heal(backup_state):
                logger.error(
                    "🔴 Final verification failed and could not be auto-repaired. Iteration changes have been rolled back."
                )
            else:
                logger.info("✅ Final verification successful. Application is stable.")
                self.handle_git_librarian(target_rel_path, iteration)

                if is_engine_file:
                    logger.warning(
                        f"🚀 SELF-EVOLUTION: `{target_rel_path}` was successfully updated."
                    )
                    self.self_evolved_flag = True

            logger.info("=" * 60 + "\n")

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
            logger.info(f"🚀 Launching `{rel_entry_file}` for test... (Attempt {attempt + 1}/3)")

            cmd: list[str] = []
            is_html = entry_file.endswith((".html", ".htm"))
            is_js = entry_file.endswith((".js", "package.json"))
            is_py = entry_file.endswith(".py")

            if is_py:
                venv_python = os.path.join(self.target_dir, "build_env", "bin", "python3")
                python_cmd = venv_python if os.path.exists(venv_python) else sys.executable
                cmd = [python_cmd, entry_file]
                if os.path.basename(entry_file) == "entrance.py":
                    cmd.append("--no-dashboard")
            elif is_js:
                cmd = ["npm", "start"] if entry_file.endswith("package.json") else ["node", entry_file]
            elif is_html:
                if os.environ.get("GITHUB_ACTIONS") == "true":
                    return True
                cmd = ["open", entry_file] if sys.platform == "darwin" else ["start", entry_file]

            if not cmd:
                return True

            # Determine shell usage before Popen to satisfy Mypy type checker
            use_shell = bool(cmd and (cmd[0] == "start" or cmd[0] == "open"))

            start_time = time.time()
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=self.target_dir,
                    shell=use_shell,
                    close_fds=sys.platform != "win32"
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
                logger.warning(f"⚠️ App crashed after {duration:.1f}s!")
                if attempt < 2:
                    self.llm_engine._fix_runtime_errors(stderr + "\n" + stdout, entry_file)
                else:
                    logger.error("❌ Exhausted all 3 auto-fix attempts.")
            else:
                logger.info(f"✅ App ran successfully for {duration:.1f}s.")
                return True

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
            logger.info(f"🎯 Using manual target: {target}")
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
            p = text.strip().replace("`", "").replace('"', "").replace("*", "")

            if " " in p:
                parts = p.split()
                for part in parts:
                    if "/" in part or part.endswith(".py") or part.endswith(".js"):
                        p = part
                        break

            exists = os.path.exists(os.path.join(self.target_dir, p))
            not_duplicate = p != last_file
            return exists and not_duplicate

        response = self.llm_engine.get_valid_llm_response(
            prompt, val, context="Target Selector"
        )

        final_path = response.strip().replace("`", "").replace('"', "").replace("*", "")
        if " " in final_path:
            for part in final_path.split():
                if "/" in part or part.endswith(".py"):
                    final_path = part
                    break

        return str(final_path)

    def build_initial_analysis(self):
        logger.info("⏳ ANALYSIS.md not found. Bootstrapping Deep Symbolic Scan...")
        all_files = sorted(self.llm_engine.scan_directory())
        structure_map = "\n".join(
            os.path.relpath(f, self.target_dir) for f in all_files
        )
        proj_prompt = f"Write a 2-sentence summary of this project based on these files:\n{structure_map}"
        project_summary = self.llm_engine.get_valid_llm_response(
            proj_prompt, lambda t: len(t) > 5, context="Project Genesis"
        ).strip()
        content = f"# 🧠 Project Analysis\n\n**Project Summary:**\n{project_summary}\n\n---\n\n## 📂 File Directory\n\n"
        for f_path in all_files:
            rel = os.path.relpath(f_path, self.target_dir)
            logger.info(f"Deep Symbolic Parsing: {rel}")
            with open(f_path, "r", encoding="utf-8", errors="ignore") as f:
                full_code = f.read()
            self.update_ledger_for_file(rel, full_code)
            structure_dropdowns = self.code_parser.generate_structure_dropdowns(
                f_path, full_code
            )
            sum_prompt = f"Provide a one-sentence plain text summary of what the file `{rel}` does. \n\nCRITICAL: Do NOT include any HTML tags, <details> blocks, or code signatures in your response. Just the sentence.\n\nFile Structure for context:\n{structure_dropdowns}"
            desc = self.llm_engine.get_valid_llm_response(
                sum_prompt, lambda t: "<details>" not in t and len(t) > 5, context=rel
            ).strip()
            content += (
                f"### `{rel}`\n**Summary:** {desc}\n\n{structure_dropdowns}\n---\n"
            )
        with open(self.analysis_path, "w", encoding="utf-8") as f:
            f.write(content)
        self.save_ledger()
        logger.info("✅ ANALYSIS.md and SYMBOLS.json successfully initialized.")

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
                r"(?:function|class|const|var|let)\s+([a-zA-Z0-9_$]+)", code
            )
            for d in defs:
                if len(d) > 3:
                    self.ledger["definitions"][d] = rel_path

        if rel_path in self.ledger["references"]:
            del self.ledger["references"][rel_path]

        potential_refs = re.findall(r"([a-zA-Z0-9_$]{4,})(?=\s*\(|\s*\.)", code)
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
