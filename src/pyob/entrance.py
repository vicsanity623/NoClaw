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
from pyob.evolution_mixins import EvolutionMixin  # noqa: E402
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


class EntranceController(EntranceMixin, CoreUtilsMixin, EvolutionMixin):
    ENGINE_FILES = [
        "autoreviewer.py",
        "cascade_queue_handler.py",
        "core_utils.py",
        "dashboard_html.py",
        "dashboard_server.py",
        "data_parser.py",
        "entrance.py",
        "entrance_mixins.py",
        "evolution_mixins.py",
        "feature_mixins.py",
        "models.py",
        "prompts_and_memory.py",
        "pyob_code_parser.py",
        "pyob_dashboard.py",
        "pyob_launcher.py",
        "reviewer_mixins.py",
        "scanner_mixins.py",
        "get_valid_edit.py",
        "stats_updater.py",
        "targeted_reviewer.py",
        "xml_mixin.py",
    ]

    def __init__(self, target_dir: str, dashboard_active: bool = True):
        self.target_dir = os.path.abspath(target_dir)
        self.pyob_dir = os.path.join(self.target_dir, ".pyob")
        os.makedirs(self.pyob_dir, exist_ok=True)
        self.key_cooldowns: dict[str, float] = {}
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
                cwd=self.target_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            logger.info(
                f"Dashboard server started with PID: {self.dashboard_process.pid}"
            )
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
                self.dashboard_process.wait(timeout=5)
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
                    ["open", "-a", applications_path, "--args", self.target_dir]
                )

                logger.info("NEW VERSION RELAYED. ENGINE SHUTTING DOWN.")
                sys.exit(0)

        except Exception as e:
            logger.error(f"Production Build Failed: {e}")

    def load_ledger(self) -> dict:
        """Loads the symbolic ledger from disk with type safety."""
        if os.path.exists(self.symbols_path):
            try:
                with open(self.symbols_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(
                    f"Failed to load SYMBOLS.json, initializing empty ledger: {e}"
                )
        return {"definitions": {}, "references": {}}

    def save_ledger(self):
        """Saves the current symbolic ledger back to disk."""
        with open(self.symbols_path, "w", encoding="utf-8") as f:
            json.dump(self.ledger, f, indent=2)

    def run_master_loop(self):
        logger.info(
            "\n" + "=" * 60 + "\nENTRANCE CONTROLLER: SYMBOLIC MODE ACTIVE\n" + "=" * 60
        )
        iteration = 1
        while True:
            self.current_iteration = iteration
            self.self_evolved_flag = False

            # --- CONTINUOUS EVOLUTION REFRESH ---
            # Every iteration, we wipe the bot's 'Short-Term Memory' so it 
            # re-scans the files. This allows Iteration 2 to 'see' the 
            # changes made in Iteration 1, even if they are on a PR branch.
            logger.info(f"--- REFRESHING SYMBOLIC CONTEXT FOR ITERATION {iteration} ---")
            
            # 1. Clear the derived project map files
            if os.path.exists(self.analysis_path):
                os.remove(self.analysis_path)
            if os.path.exists(self.symbols_path):
                os.remove(self.symbols_path)

            # 2. Perform a Fresh Genesis Scan of the current filesystem state.
            # This ensures the bot's analysis is never 'stale'.
            self.build_initial_analysis()
            # ------------------------------------

            # Check if main branch has been updated by the user (manual merges)
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

            # If the engine itself was modified, reboot to load new logic
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
        """Extracts a clean relative file path from a LLM response."""
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

    def update_analysis_for_single_file(self, target_abs_path: str, rel_path: str):
        if not os.path.exists(self.analysis_path):
            return
        with open(target_abs_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
        structure = self.code_parser.generate_structure_dropdowns(target_abs_path, code)
        sum_prompt = f"Provide a one-sentence plain text summary of what the file `{rel_path}` does. \n\nStructure:\n{structure}"
        desc = self.get_valid_llm_response(
            sum_prompt, lambda t: "<details>" not in t and len(t) > 5, context=rel_path
        ).strip()
        new_block = f"### `{rel_path}`\n**Summary:** {desc}\n\n" + structure + "\n---\n"
        with open(self.analysis_path, "r", encoding="utf-8") as f:
            analysis_text = f.read()
        pattern = rf"### `{re.escape(rel_path)}`.*?(?=### `|---\n\Z)"
        updated_text, num_subs = re.subn(
            pattern, new_block, analysis_text, flags=re.DOTALL
        )

        if num_subs == 0:
            # If no existing block was found, append the new block
            updated_text = analysis_text + new_block

        with open(self.analysis_path, "w", encoding="utf-8") as f:
            f.write(updated_text)

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
