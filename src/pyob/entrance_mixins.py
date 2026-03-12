import difflib
import json
import logging
import os
import shutil
import threading
import time
import urllib.parse
from http.server import HTTPServer
from pathlib import Path
from typing import Any

from pyob.autoreviewer import AutoReviewer
from pyob.pyob_dashboard import OBSERVER_HTML, ObserverHandler

logger = logging.getLogger(__name__)


class TargetedReviewer(AutoReviewer):
    def __init__(self, target_dir: str, target_file: str):
        super().__init__(target_dir)
        self.forced_target_file = target_file

    def scan_directory(self) -> list[str]:
        if os.path.exists(self.forced_target_file):
            return [self.forced_target_file]
        return []


class EntranceMixin:
    def start_dashboard(self: Any):
        # 1. Save to the internal .pyob folder
        obs_path = os.path.join(self.pyob_dir, "observer.html")

        # Dynamically modify OBSERVER_HTML string to inject manual target UI and logic
        modified_html_content = OBSERVER_HTML

        # Insert HTML block for manual target selection
        html_to_insert = """
        <!-- NEW INTERACTIVE FEATURE START -->
        <div class="control-section" style="background-color: #333333; padding: 15px; border-radius: 6px; margin-top: 20px;">
            <h3>Manual Target Selection</h3>
            <form id="set-target-form" action="/set_target" method="POST">
                <input type="text" id="target-file-path" name="file_path" placeholder="e.g., src/my_module/my_file.py" style="width: 70%; padding: 8px; margin-right: 10px; border: 1px solid #555; background-color: #1c1c1c; color: #d4d4d4; border-radius: 4px;">
                <button type="submit" style="padding: 8px 15px; background-color: #6a9955; color: white; border: none; border-radius: 4px; cursor: pointer;">Set Next Target</button>
            </form>
            <p id="target-message" style="margin-top: 10px; color: #dcdcaa;"></p>
        </div>
        <!-- NEW INTERACTIVE FEATURE END -->
"""
        # Assuming OBSERVER_HTML has a structure like: ... </div>\n\n        <h2>Live Log</h2>
        # This is a fragile string replacement based on the proposal's description.
        insertion_marker_html = "        </div>\n\n        <h2>Live Log</h2>"
        if insertion_marker_html in modified_html_content:
            modified_html_content = modified_html_content.replace(
                insertion_marker_html, html_to_insert + "\n" + insertion_marker_html
            )
        else:
            logger.warning(
                "HTML insertion marker not found in OBSERVER_HTML. Manual target UI may not be visible."
            )

        # Insert JavaScript for manual target form submission handling
        js_to_insert = """
        // Handle manual target form submission
        document.getElementById('set-target-form').addEventListener('submit', function(event) {
            event.preventDefault(); // Prevent default form submission

            const filePathInput = document.getElementById('target-file-path');
            const filePath = filePathInput.value;
            const targetMessage = document.getElementById('target-message');

            if (!filePath) {
                targetMessage.textContent = 'Please enter a file path.';
                targetMessage.style.color = 'red';
                return;
            }

            fetch('/set_target', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `file_path=${encodeURIComponent(filePath)}`
            })
            .then(response => {
                const isOk = response.ok; // Capture response.ok status
                return response.json().then(data => ({ data, isOk })); // Pass data and status
            })
            .then(({ data, isOk }) => { // Destructure to get both
                targetMessage.textContent = data.message;
                // Check response.ok from the original fetch response, not data
                targetMessage.style.color = isOk ? '#6a9955' : 'red';
                if (isOk) {
                    filePathInput.value = ''; // Clear input on success
                }
            })
            .catch(error => {
                console.error('Error setting manual target:', error);
                targetMessage.textContent = 'An error occurred while setting target.';
                targetMessage.style.color = 'red';
            });
        });
"""
        # Assuming OBSERVER_HTML has a script tag ending with '    </script>'
        insertion_marker_js = "    </script>"
        if insertion_marker_js in modified_html_content:
            modified_html_content = modified_html_content.replace(
                insertion_marker_js, js_to_insert + "\n" + insertion_marker_js
            )
        else:
            logger.warning(
                "JavaScript insertion marker not found in OBSERVER_HTML. Manual target logic may not be functional."
            )

        with open(obs_path, "w", encoding="utf-8") as f:
            f.write(modified_html_content)

        # 2. Initialize and Start the Live Server

        # Dynamically add do_POST method for manual target handling
        def _dynamic_do_POST_method(handler_instance: Any):
            if handler_instance.path == "/set_target":
                content_length = int(handler_instance.headers["Content-Length"])
                post_data = handler_instance.rfile.read(content_length).decode("utf-8")
                parsed_data = urllib.parse.parse_qs(post_data)
                file_path = parsed_data.get("file_path", [""])[0]

                if file_path and handler_instance.controller:
                    handler_instance.controller.set_manual_target_file(file_path)
                    message = f"Manual target set to: {file_path}"
                    status_code = 200
                else:
                    message = (
                        "Error: No file path provided or controller not available."
                    )
                    status_code = 400

                handler_instance.send_response(status_code)
                handler_instance.send_header("Content-type", "application/json")
                handler_instance.end_headers()
                handler_instance.wfile.write(
                    json.dumps({"message": message}).encode("utf-8")
                )
            else:
                handler_instance.send_error(404)

        # Assign the function as a method to the imported class
        ObserverHandler.do_POST = _dynamic_do_POST_method

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

    def execute_targeted_iteration(self: Any, iteration: int):
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
