import difflib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

class EntranceMixin:
    """
    Mixin providing core iteration logic.
    Attributes are declared here to satisfy strict Mypy checks.
    """
    target_dir: str
    pyob_dir: str
    ENGINE_FILES: list[str]
    llm_engine: Any
    code_parser: Any
    cascade_queue: list[str]
    cascade_diffs: dict[str, str]
    session_pr_count: int
    self_evolved_flag: bool
    memory_path: str
    history_path: str
    analysis_path: str
    symbols_path: str
    manual_target_file: Optional[str]
    key_cooldowns: dict[str, float]

    def pick_target_file(self) -> str:
        return ""
    def _read_file(self, path: str) -> str:
        return ""
    def _extract_path_from_llm_response(self, text: str) -> str:
        return ""
    def get_valid_llm_response(self, p: str, v: Callable[[str], bool], context: str) -> str:
        return ""
    def update_analysis_for_single_file(self, abs_p: str, rel_p: str):
        pass
    def update_ledger_for_file(self, rel_p: str, code: str):
        pass
    def detect_symbolic_ripples(self, o: str, n: str, p: str) -> list[str]:
        return []
    def _run_final_verification_and_heal(self, b: dict) -> bool:
        return False
    def handle_git_librarian(self, p: str, i: int):
        pass
    def append_to_history(self, p: str, o: str, n: str):
        pass
    def wrap_up_evolution_session(self):
        pass
    def generate_pr_summary(self, rel_path: str, diff_text: str) -> dict:
        return {}
    def execute_targeted_iteration(self, iteration: int):
        """Orchestrates a single targeted evolution step."""
        backup_state = self.llm_engine.backup_workspace()
        target_diff = ""

        if self.cascade_queue:
            target_rel_path = self.cascade_queue.pop(0)
            target_diff = self.cascade_diffs.get(target_rel_path, "")
            logger.warning(f"SYMBOLIC CASCADE: Targeting impacted file: {target_rel_path}")
            is_cascade = True
        else:
            target_rel_path = self.pick_target_file()
            is_cascade = False

        if not target_rel_path:
            return

        is_engine_file = any(Path(target_rel_path).name == f for f in self.ENGINE_FILES)

        if is_engine_file:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            project_name = os.path.basename(self.target_dir)
            base_backup_path = Path.home() / "Documents" / "PYOB_Backups" / project_name
            pod_path = base_backup_path / f"safety_pod_v{iteration}_{timestamp}"
            try:
                pod_path.mkdir(parents=True, exist_ok=True)
                for f_name in self.ENGINE_FILES:
                    src = os.path.join(self.target_dir, "src", "pyob", f_name)
                    if os.path.exists(src):
                        shutil.copy(src, str(pod_path))
            except Exception as e:
                logger.error(f"Failed to create safety pod: {e}")

        target_abs_path = os.path.join(self.target_dir, target_rel_path)
        self.llm_engine.session_context = []
        if is_cascade and target_diff:
            msg = f"CRITICAL SYMBOLIC RIPPLE: This file depends on code that was just modified.\n### CHANGE DIFF:\n{target_diff}"
            self.llm_engine.session_context.append(msg)

        old_content = ""
        if os.path.exists(target_abs_path):
            with open(target_abs_path, "r", encoding="utf-8", errors="ignore") as f:
                old_content = f.read()

        from pyob.targeted_reviewer import TargetedReviewer
        reviewer = TargetedReviewer(self.target_dir, target_abs_path)
        reviewer.session_context = self.llm_engine.session_context[:]
        if hasattr(self, 'key_cooldowns'):
            reviewer.key_cooldowns = self.key_cooldowns
        if hasattr(self, 'session_pr_count'):
            reviewer.session_pr_count = self.session_pr_count

        reviewer.run_pipeline(iteration)

        self.llm_engine.session_context = reviewer.session_context[:]
        if hasattr(reviewer, 'session_pr_count'):
            self.session_pr_count = reviewer.session_pr_count

        new_content = ""
        if os.path.exists(target_abs_path):
            with open(target_abs_path, "r", encoding="utf-8", errors="ignore") as f:
                new_content = f.read()

        logger.info(f"Refreshing metadata for `{target_rel_path}`...")
        self.update_analysis_for_single_file(target_abs_path, target_rel_path)
        self.update_ledger_for_file(target_rel_path, new_content)

        if old_content != new_content:
            logger.info(f"Edit successful. Verifying {target_rel_path}...")
            self.append_to_history(target_rel_path, old_content, new_content)
            current_diff = "".join(difflib.unified_diff(old_content.splitlines(keepends=True), new_content.splitlines(keepends=True)))
            ripples = self.detect_symbolic_ripples(old_content, new_content, target_rel_path)
            if ripples:
                for r in ripples:
                    if r not in self.cascade_queue:
                        self.cascade_queue.append(r)
                        self.cascade_diffs[r] = current_diff

            if not self._run_final_verification_and_heal(backup_state):
                logger.error("Final verification failed. Changes rolled back.")
            else:
                self.handle_git_librarian(target_rel_path, iteration)
                if is_engine_file:
                    self.self_evolved_flag = True

            # --- THE FINAL WRAP-UP GATE ---
            if getattr(self, "session_pr_count", 0) >= 8 and not getattr(self, "cascade_queue", []):
                logger.info(f"🏆 MISSION ACCOMPLISHED: {self.session_pr_count} PRs achieved.")
                self.wrap_up_evolution_session()
