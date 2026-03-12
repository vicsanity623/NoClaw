import os
import re

from pyob.core_utils import logger


class PromptsAndMemoryMixin:
    target_dir: str
    history_path: str
    analysis_path: str
    memory_file: str
    memory: str

    def _ensure_prompt_files(self) -> None:
        data_dir = os.path.join(self.target_dir, ".pyob")
        os.makedirs(data_dir, exist_ok=True)
        templates = {
            "UM.md": "You are the PyOB Memory Manager. Your job is to update MEMORY.md.\n\n### Current Memory:\n{current_memory}\n\n### Recent Actions:\n{session_summary}\n\n### INSTRUCTIONS:\n1. Update the memory with the Recent Actions.\n2. TRANSACTIONAL RECORDING: Only record changes as 'Implemented' if the actions specifically state 'SUCCESSFUL CHANGE'. If you see 'CRITICAL: FAILED' or 'ROLLED BACK', record this as a 'Failed Attempt' with the reason, so the engine knows to try a different approach next time.\n3. BREVITY: Keep the ENTIRE document under 200 words. Be ruthless. Delete old, irrelevant details.\n4. FORMAT: Keep lists strictly to bullet points. No long paragraphs.\n5. Respond EXCLUSIVELY with the raw markdown for MEMORY.md. Do not use ```markdown fences or <THOUGHT> blocks.",
            "RM.md": "You are the PYOB Memory Manager. The current MEMORY.md is too bloated and is breaking the AI context window.\n\n### Bloated Memory:\n{current_memory}\n\n### INSTRUCTIONS:\n1. AGGRESSIVELY COMPRESS this memory. \n2. Delete duplicate information, repetitive logs, and obvious statements.\n3. Keep ONLY the core architectural rules and crucial file dependencies.\n4. The final output MUST BE UNDER 150 WORDS.\n5. Respond EXCLUSIVELY with the raw markdown. No fences, no thoughts.",
            "PP.md": "You are an elite PYOB Software Engineer. Analyze the code for bugs or architectural gaps.\n\n{memory_section}{ruff_section}{mypy_section}{custom_issues_section}### Source Code:\n```{lang_tag}\n{content}\n```\n\n### CRITICAL RULES:\n1. **ITERATION BUDGET**: \n   - IF iteration < 5: Focus ONLY on bug fixes, linting, and minor performance tweaks. Do NOT create new files, move classes, or extract logic. \n   - IF iteration >= 5: You may propose architectural refactors, file extractions, and logic movement.\n2. **SURGICAL FIXES**: Every <SEARCH> block must be exactly 2-5 lines. Only provide ONE <EDIT> block per response.\n3. **NO HALLUCINATIONS**: Do not invent bugs. If the code is functional, state 'The code looks good.'\n4. **ARCHITECTURAL BLOAT**: Only flag bloat for future planning until iteration 5.\n5. **MISSING IMPORTS**: If you use a new module/type, you MUST add an <EDIT> block to import it at the top.\n6. **INDENTATION IS SYNTAX**: Your <REPLACE> blocks must have perfect, absolute indentation.\n7. **DEFEATING MYPY**: If fixing a `[union-attr]` error, use `assert variable is not None` or `# type: ignore`.\n8. **IMPORT PATHS (MANDATORY)**: Never use the prefix `src.` in any import statement. The root of the package is `pyob`. (Example: Use `from pyob.core_utils import ...`, NOT `from src.pyob.core_utils import ...`).\n9. **TYPE HINTS (CRITICAL)**: Never use the `|` operator with quoted class names (Forward References). Use `Any` for objects that create circular dependencies.\n10. **ATOMIC REFACTORING**: If you are beyond iteration 5 and need to move logic, use a 3-step process: A) Add new file/imports. B) Update call sites. C) Remove old code in a subsequent turn.\n\n### HOW TO RESPOND (CHOOSE ONE):\n\n**SCENARIO A: Code has bugs/bloat and needs edits:**\n<THOUGHT>\nSummary: ...\nEvaluation: [Address bloat here if flagged]\nImports Required: ...\nAction: I will fix X by doing Y.\n</THOUGHT>\n<EDIT>\n<SEARCH>\nExact lines to replace (2-5 lines max)\n</SEARCH>\n<REPLACE>\nNew lines\n</REPLACE>\n</EDIT>\n\n**SCENARIO B: Code is perfect. NO EDITS NEEDED:**\n<THOUGHT>\nSummary: ...\nEvaluation: ...\nAction: The code looks good. No fixes needed.\n</THOUGHT>",
            "ALF.md": "You are an elite developer fixing syntax errors.\nThe file `{rel_path}` failed validation with these exact errors:\n{err_text}\n\n### Current Code:\n```\n{code}\n```\n\n### Instructions:\n1. Fix the syntax errors (like stray brackets, unexpected tokens, or indentation) using surgical XML edits.\n2. Respond EXCLUSIVELY with a <THOUGHT> block followed by ONE OR MORE <EDIT> blocks.\n3. Ensure your edits perfectly align with the surrounding brackets.",
            "FRE.md": "You are an elite PYOB developer fixing runtime crashes.\n{memory_section}The application crashed during a test run.\n\n### Crash Logs & Traceback:\n{logs}\n\nThe traceback indicates the error occurred in `{rel_path}`.\n\n### Current Code of `{rel_path}`:\n```python\n{code}\n```\n\n### Instructions:\n1. Identify the EXACT root cause of the crash.\n2. Fix the error using surgical XML edits.\n3. Respond EXCLUSIVELY with a <THOUGHT> block followed by ONE OR MORE <EDIT> blocks.\n\n### REQUIRED XML FORMAT:\n<THOUGHT>\nExplanation of root cause...\nImports Needed: [List new imports required or 'None']\n</THOUGHT>\n<EDIT>\n<SEARCH>\nExact lines to replace\n</SEARCH>\n<REPLACE>\nNew replacement lines\n</REPLACE>\n</EDIT>",
            "PF.md": "You are the PYOB Product Architect. Review the source code and suggest ONE highly useful, INTERACTIVE feature.\n\n{memory_section}### Source Code:\n```{lang_tag}\n{content}\n```\n\n### CRITICAL RULES:\n1. Suggest an INTERACTIVE feature (UI elements, buttons, menus).\n2. **ARCHITECTURAL SPLIT (MANDATORY)**: If the source code is over 800 lines, you ARE NOT ALLOWED to propose a new feature. You MUST propose an 'Architectural Split'. Identify a logical module (like a Mixin) to move to a NEW file.\n3. **SINGLE FILE LIMIT**: If you are proposing an Architectural Split, you are ONLY allowed to create ONE new file per iteration. Do not attempt to split multiple modules at once. Focus on the largest logical block first.\n4. **NEW FILE FORMAT**: If proposing a split, your <SNIPPET> block MUST use this format: <CREATE_FILE path=\"new_filename.py\">[Full Code for New File]</CREATE_FILE>. Your <THOUGHT> must then explain how to update the original file to import this new module.\n5. MULTI-FILE ORCHESTRATION: Explicitly list filenames of other files that will need updates in your <THOUGHT>.\n6. The <SNIPPET> block MUST contain ONLY the new logic or the <CREATE_FILE> block.\n\n### REQUIRED XML FORMAT:\n<THOUGHT>\n...\n</THOUGHT>\n<SNIPPET>\n# New logic OR <CREATE_FILE> tag here\n</SNIPPET>",
            "IF.md": "You are an elite PYOB Implementation Engineer. Implement the APPROVED feature into `{rel_path}`.\n\n{memory_section}### Feature Proposal Details:\n{feature_content}\n\n### Current Source Code:\n```{lang_tag}\n{source_code}\n```\n\n### CRITICAL INSTRUCTIONS:\n1. **SURGICAL EDITS ONLY**: Every <SEARCH> block must be EXACTLY 2-5 lines.\n2. **MISSING IMPORTS**: If your feature introduces a new class/function, add a separate <EDIT> block to import it.\n3. **BLOCK INTEGRITY (CRITICAL)**: Python relies on indentation. If you add an `if`, `try`, or `def` statement, you MUST indent the code beneath it. If you remove one, you MUST dedent the code. Do not leave orphaned indents.\n4. **ABSOLUTE SPACES**: The spaces in your <REPLACE> block must match the absolute margin of the source code.\n5. **VARIABLE SCOPE**: Ensure variables are accessible (use `self.` for class states).\n6. **DELETING CODE (CRITICAL)**: If your goal is to remove a block of code (e.g., when moving logic to a new file), DO NOT leave an empty <REPLACE> block. Instead, provide a <REPLACE> block containing a comment such as `# [Logic moved to new module]` to ensure the surrounding code remains syntactically valid.\n7. **IMPORT PATHS (MANDATORY)**: Never use the prefix `src.` in any import statement. The root of the package is `pyob`. (Example: Use `from pyob.entrance import Controller`, NOT `from src.pyob.entrance import Controller`).\n8. **TYPE HINTS (CRITICAL)**: Never use the `|` operator with quoted class names (Forward References). Use `Any` for objects that create circular dependencies.\n9. **ATOMIC REFACTORING**: Do not move large blocks or classes to new files in a single turn. Focus ONLY on logic improvements or bug fixes within the current file.\n10. **SURGICAL MOVEMENT**: If moving logic, use a 3-step process: A) Add new file/imports. B) Update call sites. C) Remove old code in a subsequent turn.\n11. **NO MASS DELETIONS**: You are strictly forbidden from replacing a large block of code with a smaller one unless you include the surrounding class/method structure in your <SEARCH> block. If you cannot fit the structure in 5 lines, use multiple <EDIT> blocks.\n12. **INDENTATION LOCK**: Every <REPLACE> block must have the exact same indentation level as the <SEARCH> block.\n13. **PRESERVE STRUCTURE**: Never use a <SEARCH> block that cuts off a class or method definition. Always include the full line you are matching.\n14. **NO EMOJIS (MANDATORY)**: Never use emojis or non-ASCII characters in logger messages, print statements, or source comments. Use plain ASCII text only. Emojis cause encoding errors and git diff corruption.\n\n### REQUIRED XML FORMAT:\n<THOUGHT>\n1. Lines to change: ...\n2. New imports needed: [List them or state 'None']\n3. Strategy: ...\n</THOUGHT>\n<EDIT>\n<SEARCH>\nExact 2-5 lines\n</SEARCH>\n<REPLACE>\nNew code\n</REPLACE>\n</EDIT>",
            "PCF.md": 'You are the PYOB Symbolic Fixer taking over Phase 3 Cascaded Edits.\n{memory_section}We just modified `{trigger_file}`, and it broke a dependency downstream: `{rel_broken_path}`.\n\n### Linter Errors for `{rel_broken_path}`:\n{mypy_errors}\n\n### Source Code of `{rel_broken_path}`:\n```python\n{broken_code}\n```\n\n### Instructions:\n1. Respond EXCLUSIVELY with a <THOUGHT> block followed by ONE <EDIT> block to fix the broken references.\n2. **DEFEATING MYPY (CRITICAL)**: If the error contains `[union-attr]` or states `Item "None" of ... has no attribute`, adding a type hint will fail. You MUST fix this by either inserting `assert variable is not None` before the operation, or adding `# type: ignore` to the end of the failing line.\n\n### REQUIRED XML FORMAT:\n<THOUGHT>\nExplanation of cascade fix...\n</THOUGHT>\n<EDIT>\n<SEARCH>\nExact lines to replace\n</SEARCH>\n<REPLACE>\nNew replacement lines\n</REPLACE>\n</EDIT>',
            "PIR.md": "You are an elite PYOB developer performing a post-implementation repair.\nAn automated attempt to implement a feature or bugfix has failed, resulting in the following errors.\n\n### Original Goal / Change Request:\n{context_of_change}\n\n### The Resulting Errors (Linter/Runtime):\n{err_text}\n\n### The Broken Code in `{rel_path}`:\n```\n{code}\n```\n\n### Instructions:\n1. Analyze the 'Original Goal' and the 'Resulting Errors' together.\n2. CRITICAL INDENTATION CHECK: If the error says `Unexpected indentation`, rewrite the <EDIT> block with PERFECT absolute indentation.\n3. CRITICAL IMPORT CHECK: If the error is `F821 Undefined name`, you MUST create an <EDIT> block at the top of the file to add the missing `import` statement.\n4. DEFEATING MYPY: If the error is a `[union-attr]` or `None` type error, simply adding a type hint will fail the linter again. You MUST insert `assert object is not None` right before it is used, or append `# type: ignore` to the failing line.\n5. Create a surgical XML edit to fix the code.\n6. **DELETING CODE**: If you are removing logic, never use an empty <REPLACE> block. Always include a placeholder comment to maintain valid Python syntax.\n\n### REQUIRED XML FORMAT:\n<THOUGHT>\nRoot Cause: ...\nImports Needed: ...\nIndentation Fix Needed: [Yes/No]\nAction: ...\n</THOUGHT>\n<EDIT>\n<SEARCH>\nExact lines to replace\n</SEARCH>\n<REPLACE>\nNew replacement lines\n</REPLACE>\n</EDIT>",
        }
        for filename, content in templates.items():
            filepath = os.path.join(data_dir, filename)  # Use data_dir here
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

    def load_prompt(self, filename: str, **kwargs: str) -> str:
        # Use a consistent path resolution
        data_dir = os.path.join(self.target_dir, ".pyob")
        filepath = os.path.join(data_dir, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                template = f.read()
            for key, value in kwargs.items():
                template = template.replace(f"{{{key}}}", str(value))
            return template
        except Exception as e:
            logger.error(f"Failed to load prompt {filename} from {filepath}: {e}")
            return ""

    def _get_impactful_history(self) -> str:
        if not os.path.exists(self.history_path):
            return "No prior history."
        with open(self.history_path, "r", encoding="utf-8") as f:
            full_history = f.read()
        entries = re.split(r"## \d{4}-\d{2}-\d{2}", full_history)
        recent_entries = entries[-3:]
        summary = "### Significant Recent Architecture Changes:\n"
        for entry in recent_entries:
            lines = entry.strip().split("\n")
            if lines:
                summary += f"- {lines[0].strip()}\n"
        return summary

    def _get_rich_context(self) -> str:
        context = ""
        analysis_path = os.path.join(self.target_dir, "ANALYSIS.md")
        if os.path.exists(analysis_path):
            with open(analysis_path, "r", encoding="utf-8") as f:
                content = f.read()
                header_parts = content.split("## File Directory")
                header = header_parts[0].strip() if header_parts else ""
                context += f"### Project Goal:\n{header}\n\n"

        if os.path.exists(self.history_path):
            with open(self.history_path, "r", encoding="utf-8") as f:
                hist = f.read()
                headers = [line for line in hist.split("\n") if line.startswith("## ")]
                context += (
                    "### Recent Edit History:\n" + "\n".join(headers[-3:]) + "\n\n"
                )

        if self.memory:
            mem_str = self.memory.strip()
            if len(mem_str) > 1500:
                mem_str = (
                    mem_str[:500]
                    + "\n\n... [OLDER MEMORY TRUNCATED FOR CONTEXT LIMITS] ...\n\n"
                    + mem_str[-800:]
                )

            context += f"### Logic Memory:\n{mem_str}\n\n"

        return context

    def update_memory(self) -> None:
        session_context: list[str] = getattr(self, "session_context", [])
        if not session_context:
            return
        logger.info("\nPHASE 5: Updating MEMORY.md with session context...")
        session_summary = "\n".join(f"- {item}" for item in session_context)
        prompt = self.load_prompt(
            "UM.md",
            current_memory=self.memory if self.memory else "No previous memory.",
            session_summary=session_summary,
        )

        def validator(text: str) -> bool:
            return bool(text.strip())

        llm_response = getattr(self, "get_valid_llm_response")(
            prompt, validator, context="Memory Update"
        )
        clean_memory = re.sub(
            r"^```[a-zA-Z]*\n", "", llm_response.strip(), flags=re.MULTILINE
        )
        clean_memory = re.sub(r"\n```$", "", clean_memory, flags=re.MULTILINE)
        if clean_memory:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                f.write(clean_memory)
            self.memory = clean_memory

    def refactor_memory(self) -> None:
        if not self.memory:
            return
        logger.info("\nPHASE 6: Cleanup! Summarizing and refactoring MEMORY.md...")
        prompt = self.load_prompt("RM.md", current_memory=self.memory)

        def validator(text: str) -> bool:
            return bool(text.strip())

        llm_response = getattr(self, "get_valid_llm_response")(
            prompt, validator, context="Memory Refactor"
        )
        clean_memory = re.sub(
            r"^```[a-zA-Z]*\n", "", llm_response.strip(), flags=re.MULTILINE
        )
        clean_memory = re.sub(r"\n```$", "", clean_memory, flags=re.MULTILINE)
        if clean_memory and len(clean_memory) > 50:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                f.write(clean_memory)
            self.memory = clean_memory
