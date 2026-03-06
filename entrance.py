import os
import sys
import ast
import re
import difflib
import time
import logging
import json
import subprocess
import shutil
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
from autoreviewer import AutoReviewer
from pathlib import Path
from typing import Any

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
    def __init__(self, target_dir: str):
        self.target_dir = os.path.abspath(target_dir)
        # Check if we should skip dashboard (for recursive test runs)
        self.skip_dashboard = "--no-dashboard" in sys.argv

        self.analysis_path = os.path.join(self.target_dir, "ANALYSIS.md")
        self.history_path = os.path.join(self.target_dir, "HISTORY.md")
        self.symbols_path = os.path.join(self.target_dir, "SYMBOLS.json")
        self.llm_engine = AutoReviewer(self.target_dir)
        self.ledger = self.load_ledger()

        # FIXED: Added Type Annotations for Mypy
        self.cascade_queue: list[str] = []
        self.cascade_diffs: dict[str, str] = {}

        self.current_iteration = 1

        if not self.skip_dashboard:
            self.start_dashboard()

    def start_dashboard(self):
        # 1. Build the physical file for the user to see
        obs_path = os.path.join(self.target_dir, "observer.html")
        with open(obs_path, "w", encoding="utf-8") as f:
            f.write(OBSERVER_HTML)

        # 2. Set the controller reference for the handler
        ObserverHandler.controller = self

        # 3. Start the background server
        def run_server():
            try:
                server = HTTPServer(("localhost", 5000), ObserverHandler)
                server.serve_forever()
            except Exception as e:
                logger.error(f"Dashboard failed to start: {e}")

        threading.Thread(target=run_server, daemon=True).start()

        # 4. Notify the user with a Cyberpunk banner
        print("\n" + "=" * 60)
        print("⚡ NOCLAW OBSERVER IS LIVE")
        print("🔗 URL: http://localhost:5000")
        print(f"📂 FILE: {obs_path}")
        print("=" * 60 + "\n")

    def reboot_noclaw(self):
        """Standard Hot-Reboot: Replaces the current process with a fresh one."""
        logger.warning("🔄 SELF-EVOLUTION COMPLETE: Rebooting fresh NoClaw engine...")
        # Ensure we pass the same arguments (like the target directory)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    def trigger_production_build(self):
        """Advanced Build: Compiles NoClaw into a DMG and replaces the system version."""
        build_script = os.path.join(self.target_dir, "build_pyinstaller_multiOS.py")
        if not os.path.exists(build_script):
            logger.error("❌ Build script not found. Skipping production deploy.")
            return

        logger.info("🛠️ STARTING PRODUCTION BUILD... This will take 2-3 minutes.")
        try:
            # 1. Run the PyInstaller build
            subprocess.run([sys.executable, build_script], check=True)

            # 2. Define Mac paths
            app_name = "NoClaw.app"
            dist_path = os.path.join(self.target_dir, "dist", app_name)
            applications_path = f"/Applications/{app_name}"

            if sys.platform == "darwin" and os.path.exists(dist_path):
                logger.warning(
                    f"🚀 FORGE COMPLETE: Deploying new version to {applications_path}..."
                )

                # Delete old version and copy new one
                if os.path.exists(applications_path):
                    shutil.rmtree(applications_path)
                shutil.copytree(dist_path, applications_path)

                # --- THE PASSING OF THE TORCH ---
                # Launch new app and pass the CURRENT target directory as an argument.
                subprocess.Popen(
                    ["open", "-a", applications_path, "--args", self.target_dir]
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
            logger.info(
                f"\n\n{'=' * 70}\n🎯 TARGETED PIPELINE LOOP (Iteration {iteration})\n{'=' * 70}"
            )

            # Track if NoClaw's own logic was upgraded this turn
            self_evolved = False

            try:
                self.execute_targeted_iteration(iteration)

                # --- SELF-EVOLUTION DETECTION ---
                history_text = self._read_file(self.history_path)
                engine_files = [
                    "autoreviewer.py",
                    "core_utils.py",
                    "prompts_and_memory.py",
                    "entrance.py",
                ]

                # Check the very last entry in HISTORY.md
                if history_text:
                    last_entry = history_text.split("##")[-1]
                    if any(f"`{ef}`" in last_entry for ef in engine_files):
                        self_evolved = True
                # --------------------------------

            except KeyboardInterrupt:
                logger.info("\nExiting Entrance Controller...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in master loop: {e}", exc_info=True)

            # --- AUTOMATED ENGINE RELOAD ---
            if self_evolved:
                if getattr(sys, "frozen", False):
                    logger.warning(
                        "💎 COMPILED ENGINE EVOLVED: Initiating full Forge Build."
                    )
                    self.trigger_production_build()
                else:
                    logger.warning("🐍 SCRIPT ENGINE EVOLVED: Initiating Hot-Reboot.")
                    self.reboot_noclaw()
            # -------------------------------

            iteration += 1
            logger.info("Iteration complete. Waiting for system cooldown...")
            time.sleep(120)

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

        # --- RECURSIVE SAFETY PATCH (EXTERNAL STORAGE) ---
        engine_files = [
            "autoreviewer.py",
            "core_utils.py",
            "prompts_and_memory.py",
            "entrance.py",
        ]
        if any(f in target_rel_path for f in engine_files):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            project_name = os.path.basename(self.target_dir)
            base_backup_path = (
                Path.home() / "Documents" / "NoClaw_Backups" / project_name
            )
            pod_path = base_backup_path / f"safety_pod_v{iteration}_{timestamp}"

            try:
                pod_path.mkdir(parents=True, exist_ok=True)
                logger.warning(
                    f"🛡️ SELF-EVOLUTION: Sheltering engine source EXTERNALLY in {pod_path}"
                )
                for f_name in engine_files:
                    src = os.path.join(self.target_dir, f_name)
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

        # Sync intent context back to main engine
        self.llm_engine.session_context = reviewer.session_context[:]

        new_content = ""
        if os.path.exists(target_abs_path):
            with open(target_abs_path, "r", encoding="utf-8", errors="ignore") as f:
                new_content = f.read()

        # Brain-to-Queue Intent Detection
        all_text_context = " ".join(self.llm_engine.session_context).lower()
        for other_file in self.llm_engine.scan_directory():
            rel_other = os.path.relpath(other_file, self.target_dir)
            if rel_other != target_rel_path and rel_other in all_text_context:
                if rel_other not in self.cascade_queue:
                    logger.warning(
                        f"🧠 AI INTENT DETECTED: Queuing {rel_other} for follow-up."
                    )
                    self.cascade_queue.append(rel_other)

        logger.info(f"🔄 Refreshing metadata for `{target_rel_path}`...")
        self.update_analysis_for_single_file(target_abs_path, target_rel_path)
        self.update_ledger_for_file(target_rel_path, new_content)

        if old_content != new_content:
            logger.info(
                f"📝 Edit successful. Checking ripples and running final verification for {target_rel_path}..."
            )
            self.append_to_history(target_rel_path, old_content, new_content)

            # FIXED: Used keepends=True for Mypy compatibility
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
            logger.info("=" * 60 + "\n")

    def _run_final_verification_and_heal(self, backup_state: dict) -> bool:
        """
        Runs the main application file for 10 seconds. If it crashes, it
        attempts to fix the error up to 3 times.
        """
        entry_file = self.llm_engine._find_entry_file()
        if not entry_file:
            logger.warning("No main entry file found. Skipping runtime test.")
            return True
        rel_entry_file = os.path.relpath(entry_file, self.target_dir)

        # 3. Execution and Healing Loop
        for attempt in range(3):
            logger.info(
                f"🚀 Launching `{rel_entry_file}` for test... (Attempt {attempt + 1}/3)"
            )

            if getattr(sys, "frozen", False):
                python_cmd = (
                    shutil.which("python3") or shutil.which("python") or "python3"
                )
            else:
                python_cmd = sys.executable

            # RECURSIVE PORT SAFETY
            cmd = [python_cmd, entry_file]
            if os.path.basename(entry_file) == "entrance.py":
                cmd.append("--no-dashboard")

            start_time = time.time()
            process = subprocess.Popen(
                cmd,
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
                stdout, stderr = process.communicate()

            duration = time.time() - start_time
            has_error_logs = any(
                kw in stderr or kw in stdout
                for kw in [
                    "Traceback",
                    "Exception",
                    "Error:",
                    "ModuleNotFoundError",
                    "ImportError",
                ]
            )

            is_crash_code = process.returncode not in (0, 15, -15, None)

            if is_crash_code or has_error_logs:
                logger.warning(f"⚠️ App crashed or threw errors after {duration:.1f}s!")
                logger.warning(
                    f"--- STDERR ---\n{stderr}\n--- STDOUT ---\n{stdout}\n--------------"
                )
                if attempt < 2:
                    logger.info("Attempting auto-repair...")
                    self.llm_engine._fix_runtime_errors(
                        stderr + "\n" + stdout, entry_file
                    )
                else:
                    logger.error("❌ Exhausted all 3 auto-fix attempts.")
            else:
                logger.info(
                    f"✅ App ran successfully for {duration:.1f}s without tracebacks."
                )
                return True

        logger.warning(
            "Restoring workspace to pre-iteration state due to unfixable crash."
        )
        self.llm_engine.restore_workspace(backup_state)
        return False

    def detect_symbolic_ripples(
        self, old: str, new: str, source_file: str
    ) -> list[str]:
        diff = list(difflib.unified_diff(old.splitlines(), new.splitlines()))
        changed_text = "\n".join(
            [line for line in diff if line.startswith("+") or line.startswith("-")]
        )
        potential_symbols = set(re.findall(r"([a-zA-Z0-9_$]{4,})", changed_text))
        impacted_files = []
        for sym in potential_symbols:
            if self.ledger["definitions"].get(sym) == source_file:
                for target_file, refs in self.ledger["references"].items():
                    if sym in refs and target_file != source_file:
                        impacted_files.append(target_file)
        return list(set(impacted_files))

    def pick_target_file(self) -> str:
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
Read the ANALYSIS.md and HISTORY.md. Choose ONE relative file path to review next.
CRITICAL RULES:
1. DO NOT pick `{last_file}` again. It was just edited.
2. If logic was recently added to a backend file (like `logic.py`), your MUST pick the UI or Main file (like `main.py`) next to implement a way for the user to use that logic.
3. If a configuration file (like `theme.py`) was changed, pick a file that depends on it.
4. Rotate between logic, UI, and styles to ensure the features are actually visible to the user.
### Analysis:
{analysis}
### History:
{history}
Reply ONLY with the relative file path.
"""

        def val(text: str) -> bool:
            p = text.strip().strip("`").strip()
            return os.path.exists(os.path.join(self.target_dir, p)) and p != last_file

        return (
            self.llm_engine.get_valid_llm_response(
                prompt, val, context="Target Selector"
            )
            .strip()
            .strip("`")
        )

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
            structure_dropdowns = self.generate_structure_dropdowns(f_path, full_code)
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
        structure = self.generate_structure_dropdowns(target_abs_path, code)
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
        if ext == ".py":
            try:
                tree = ast.parse(code)
                for n in ast.walk(tree):
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                        self.ledger["definitions"][n.name] = rel_path
            except Exception as e:
                logger.warning(f"Failed to parse Python AST for {rel_path}: {e}")
        elif ext in [".js", ".ts"]:
            defs = re.findall(
                r"(?:function|class|const|var|let)\s+([a-zA-Z0-9_$]+)", code
            )
            for d in defs:
                if len(d) > 3:
                    self.ledger["definitions"][d] = rel_path
        potential_refs = re.findall(r"([a-zA-Z0-9_$]{4,})(?=\s*\(|\s*\.)", code)
        self.ledger["references"][rel_path] = list(set(potential_refs))
        self.save_ledger()

    def generate_structure_dropdowns(self, filepath: str, code: str) -> str:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".py":
            return self._parse_python(code)
        elif ext in [".js", ".ts"]:
            return self._parse_javascript(code)
        elif ext == ".html":
            return self._parse_html(code)
        elif ext == ".css":
            return self._parse_css(code)
        return ""

    def _parse_python(self, code: str) -> str:
        try:
            tree = ast.parse(code)
            imports, classes, functions, consts = [], [], [], []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.unparse(node))
                elif isinstance(node, ast.ClassDef):
                    classes.append(f"class {node.name}")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = []
                    for arg in node.args.args:
                        if arg.arg != "self":
                            args.append(arg.arg)
                    if node.args.vararg:
                        args.append(f"*{node.args.vararg.arg}")
                    if node.args.kwarg:
                        args.append(f"**{node.args.kwarg.arg}")
                    sig = f"def {node.name}({', '.join(args)})"
                    functions.append(sig)
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id.isupper():
                            consts.append(t.id)
            return self._format_dropdowns(imports, classes, functions, consts)
        except Exception as e:
            logger.warning(f"Failed to parse Python AST for dropdowns: {e}")
            return ""

    def _parse_javascript(self, code: str) -> str:
        imports = re.findall(r"(?:import|from|require)\s+['\"].*?['\"]", code)
        classes = re.findall(r"class\s+([a-zA-Z0-9_$]+)", code)
        fn_patterns = [
            r"function\s+([a-zA-Z0-9_$]+)\s*\(([^)]*)\)",
            r"(?:const|let|var|window\.)\s*([a-zA-Z0-9_$]+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>",
            r"^\s*([a-zA-Z0-9_$]+)\s*\(([^)]*)\)\s*\{",
        ]
        raw_fns = []
        for pattern in fn_patterns:
            raw_fns.extend(re.findall(pattern, code, re.MULTILINE))
        clean_fns = []
        seen = set()
        for name, params in raw_fns:
            if name not in seen and name not in [
                "if",
                "for",
                "while",
                "switch",
                "return",
                "await",
            ]:
                params = re.sub(r"\s+", " ", params).strip()
                clean_fns.append(f"{name}({params})")
                seen.add(name)
        entities = re.findall(
            r"(?:const|var|let)\s+([A-Z0-9_]{3,})|([a-zA-Z0-9_$]+)\s*:\s*(?:['\"].*?['\"]|[0-9\.]+|true|false|null|{)",
            code,
        )
        clean_entities = sorted(
            list(
                set(
                    [
                        e[0] or e[1]
                        for e in entities
                        if (e[0] or e[1]) and (e[0] or e[1]) not in seen
                    ]
                )
            )
        )
        return self._format_dropdowns(
            imports, classes, sorted(clean_fns), clean_entities
        )

    def _parse_html(self, code: str) -> str:
        scripts = re.findall(r"<script.*?src=['\"](.*?)['\"]", code)
        styles = re.findall(r"<link.*?href=['\"](.*?)['\"]", code)
        ids = re.findall(r"id=['\"](.*?)['\"]", code)
        return self._format_dropdowns(
            [],
            [f"Script: {s}" for s in scripts],
            [f"ID: #{i}" for i in ids],
            [f"CSS: {s}" for s in styles],
        )

    def _parse_css(self, code: str) -> str:
        selectors = re.findall(r"(\.[a-zA-Z0-9_-]+)\s*\{", code)
        return self._format_dropdowns([], [], selectors[:50], [])

    def _format_dropdowns(
        self, imp: list[str], cls: list[str], fn: list[str], cnst: list[str]
    ) -> str:
        res = ""
        if imp:
            res += f"<details><summary>Imports ({len(imp)})</summary>{'<br>'.join(sorted(imp))}</details>\n"
        if cnst:
            res += f"<details><summary>Entities/Constants ({len(cnst)})</summary>{'<br>'.join(sorted(cnst))}</details>\n"
        if cls:
            res += f"<details><summary>Classes/Structures ({len(cls)})</summary>{'<br>'.join(sorted(cls))}</details>\n"
        if fn:
            res += f"<details><summary>Logic/Functions ({len(fn)})</summary>{'<br>'.join(sorted(fn))}</details>\n"
        return res

    def append_to_history(self, rel_path: str, old_code: str, new_code: str):
        # FIXED: Change splitlines(1) to keepends=True for Mypy
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


OBSERVER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>NOCLAW // OBSERVER</title>
    <style>
        body { background: #050505; color: #00FF41; font-family: 'Menlo', monospace; margin: 0; padding: 20px; overflow-x: hidden; }
        .glow { text-shadow: 0 0 10px #00FF41, 0 0 20px #00FF41; }
        .border { border: 1px solid #00FF41; box-shadow: 0 0 15px rgba(0, 255, 65, 0.2); padding: 20px; margin-bottom: 20px; }
        h1 { font-size: 2em; border-bottom: 2px solid #00FF41; padding-bottom: 10px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card { background: rgba(0, 255, 65, 0.05); }
        .label { color: #008F11; font-weight: bold; margin-bottom: 5px; }
        .data { font-size: 0.9em; white-space: pre-wrap; height: 300px; overflow-y: auto; border: 1px solid #004411; padding: 10px; background: #000; }
        .stat-bar { display: flex; justify-content: space-between; font-size: 1.2em; margin-bottom: 20px; }
        .queue-item { background: #00FF41; color: #000; padding: 2px 5px; margin: 2px; display: inline-block; font-size: 0.8em; }
        #iteration { font-size: 1.5em; color: #fff; }
    </style>
</head>
<body>
    <h1 class="glow">NOCLAW_OS // OBSERVER_DASHBOARD</h1>
    <div class="stat-bar border card">
        <div>ITERATION: <span id="iteration" class="glow">--</span></div>
        <div>LEDGER: <span id="ledger">--</span> symbols</div>
        <div>STATUS: <span id="status" style="color: #fff">SCANNING...</span></div>
    </div>
    <div class="border card"><div class="label">SYMBOLIC CASCADE QUEUE:</div><div id="queue">--</div></div>
    <div class="grid">
        <div class="border card"><div class="label">LIVE MEMORY (MEMORY.md):</div><div id="memory" class="data">--</div></div>
        <div class="border card"><div class="label">RECENT HISTORY (HISTORY.md):</div><div id="history" class="data">--</div></div>
    </div>
    <div class="border card"><div class="label">LATEST ARCHITECTURAL ANALYSIS:</div><div id="analysis" class="data">--</div></div>
    <script>
        async function updateStats() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                document.getElementById('iteration').innerText = data.iteration;
                document.getElementById('ledger').innerText = data.ledger_stats.definitions;
                document.getElementById('memory').innerText = data.memory || "Initializing brain...";
                document.getElementById('history').innerText = data.history || "No history recorded yet.";
                document.getElementById('analysis').innerText = data.analysis || "Parsing directory structure...";
                const queueDiv = document.getElementById('queue');
                queueDiv.innerHTML = data.cascade_queue.length > 0 ? data.cascade_queue.map(f => `<span class='queue-item'>${f}</span>`).join('') : "IDLE // NO PENDING CASCADES";
                document.getElementById('status').innerText = data.cascade_queue.length > 0 ? "EVOLVING" : "READY";
            } catch (e) { document.getElementById('status').innerText = "OFFLINE"; }
        }
        setInterval(updateStats, 3000);
        updateStats();
    </script>
</body>
</html>
"""


class ObserverHandler(BaseHTTPRequestHandler):
    controller = None

    def do_GET(self):
        if self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            status = {
                "iteration": getattr(self.controller, "current_iteration", 1),
                "cascade_queue": self.controller.cascade_queue,
                "ledger_stats": {
                    "definitions": len(self.controller.ledger["definitions"]),
                    "references": len(self.controller.ledger["references"]),
                },
                "analysis": self.controller._read_file(self.controller.analysis_path),
                "memory": self.controller._read_file(
                    os.path.join(self.controller.target_dir, "MEMORY.md")
                ),
                "history": self.controller._read_file(self.controller.history_path)[
                    -5000:
                ],
            }
            self.wfile.write(json.dumps(status).encode())
        elif self.path == "/" or self.path == "/observer.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(OBSERVER_HTML.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return


if __name__ == "__main__":
    EntranceController(sys.argv[1] if len(sys.argv) > 1 else ".").run_master_loop()
