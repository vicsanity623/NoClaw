import ast
import re

from .core_utils import logger


# This class is being moved to `xml_mixin.py` in an architectural split.
# It will be removed from this file in a subsequent turn.
class ApplyXMLMixin:
    def ensure_imports_retained(
        self, orig_code: str, new_code: str, filepath: str
    ) -> str:
        try:
            orig_tree = ast.parse(orig_code)
            new_tree = ast.parse(new_code)
        except Exception:
            return new_code
        orig_imports = []
        for node in orig_tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                start_line = node.lineno - 1
                end_line = getattr(node, "end_lineno", node.lineno)
                import_text = "\n".join(orig_code.splitlines()[start_line:end_line])
                orig_imports.append((node, import_text))
        missing_imports = []
        for orig_node, import_text in orig_imports:
            found = False
            for new_node in new_tree.body:
                if isinstance(new_node, type(orig_node)):
                    if isinstance(orig_node, ast.Import) and isinstance(
                        new_node, ast.Import
                    ):
                        if {alias.name for alias in orig_node.names}.issubset(
                            {alias.name for alias in new_node.names}
                        ):
                            found = True
                            break
                    elif isinstance(orig_node, ast.ImportFrom) and isinstance(
                        new_node, ast.ImportFrom
                    ):
                        if orig_node.module == new_node.module and {
                            alias.name for alias in orig_node.names
                        }.issubset({alias.name for alias in new_node.names}):
                            found = True
                            break
            if not found:
                missing_imports.append(import_text)
        if missing_imports:
            return "\n".join(missing_imports) + "\n\n" + new_code
        return new_code

    def apply_xml_edits(
        self, source_code: str, llm_response: str
    ) -> tuple[str, str, bool]:
        source_code = source_code.replace("\r\n", "\n")
        llm_response = llm_response.replace("\r\n", "\n")

        explanation = self._extract_explanation(llm_response)
        matches = self._extract_edit_blocks(llm_response)

        if not matches:
            return source_code, explanation, True

        new_code = source_code
        all_edits_succeeded = True

        for m in matches:
            raw_search = re.sub(
                r"^```[\w]*\n|\n```$", "", m.group(1), flags=re.MULTILINE
            )
            raw_replace = re.sub(
                r"^```[\w]*\n|\n```$", "", m.group(2), flags=re.MULTILINE
            )

            raw_replace = self._fix_replace_indentation(raw_search, raw_replace)

            new_code, success = self._apply_single_block(
                new_code, raw_search, raw_replace
            )
            if not success:
                all_edits_succeeded = False

        return new_code, explanation, all_edits_succeeded

    # ==========================================
    # PRIVATE HELPER METHODS FOR XML PATCHING
    # ==========================================

    def _extract_explanation(self, llm_response: str) -> str:
        thought_match = re.search(
            r"<THOUGHT>(.*?)</THOUGHT>", llm_response, re.DOTALL | re.IGNORECASE
        )
        return (
            thought_match.group(1).strip()
            if thought_match
            else "No explanation provided."
        )

    def _extract_edit_blocks(self, llm_response: str) -> list[re.Match]:
        pattern = re.compile(
            r"<EDIT>\s*<SEARCH>\s*\n?(.*?)\n?\s*</SEARCH>\s*<REPLACE>\s*\n?(.*?)\n?\s*</REPLACE>\s*</EDIT>",
            re.DOTALL | re.IGNORECASE,
        )
        return list(pattern.finditer(llm_response))

    def _fix_replace_indentation(self, search: str, replace: str) -> str:
        search_lines = search.split("\n")
        replace_lines = replace.split("\n")

        search_indent = ""
        for line in search_lines:
            if line.strip():
                search_indent = line[: len(line) - len(line.lstrip(" \t"))]
                break

        replace_base_indent = ""
        for line in replace_lines:
            if line.strip():
                replace_base_indent = line[: len(line) - len(line.lstrip(" \t"))]
                break

        fixed_replace_lines = []
        for line in replace_lines:
            if line.strip():
                if line.startswith(replace_base_indent):
                    clean_line = line[len(replace_base_indent) :]
                else:
                    clean_line = line.lstrip(" \t")
                fixed_replace_lines.append(search_indent + clean_line)
            else:
                fixed_replace_lines.append("")
        return "\n".join(fixed_replace_lines)

    def _apply_single_block(
        self, source: str, search: str, replace: str
    ) -> tuple[str, bool]:
        # Strategy 1: Exact Match
        if search in source:
            return source.replace(search, replace, 1), True

        clean_search = search.strip("\n")
        clean_replace = replace.strip("\n")

        # Strategy 2: Clean Exact Match
        if clean_search and clean_search in source:
            return source.replace(clean_search, clean_replace, 1), True

        # Strategy 3: Normalized Match
        source, success = self._attempt_normalized_match(source, search, replace)
        if success:
            return source, True

        # Strategy 4: Regex Fuzzy Match
        source, success = self._attempt_regex_fuzzy_match(source, clean_search, replace)
        if success:
            return source, True

        # Strategy 5: Line-by-Line Robust Match
        source, success = self._attempt_line_by_line_match(source, search, replace)
        if success:
            return source, True

        return source, False

    def _attempt_normalized_match(
        self, source: str, search: str, replace: str
    ) -> tuple[str, bool]:
        def normalize(t: str) -> str:
            t = re.sub(r"#.*", "", t)
            return re.sub(r"\s+", " ", t).strip()

        norm_search = normalize(search)
        if not norm_search:
            return source, False

        search_lines = search.split("\n")
        lines = source.splitlines()
        for i in range(len(lines)):
            test_block = normalize("\n".join(lines[i : i + len(search_lines)]))
            if norm_search in test_block:
                lines[i : i + len(search_lines)] = replace.splitlines()
                logger.info(f"Normalization match succeeded at line {i + 1}.")
                return "\n".join(lines), True
        return source, False

    def _attempt_regex_fuzzy_match(
        self, source: str, clean_search: str, replace: str
    ) -> tuple[str, bool]:
        try:
            search_lines_cleaned = [
                line.strip() for line in clean_search.split("\n") if line.strip()
            ]
            if not search_lines_cleaned:
                return source, False

            regex_parts = [
                r"^[ \t]*" + re.escape(line) + r"[ \t]*\n+"
                for line in search_lines_cleaned[:-1]
            ]
            regex_parts.append(
                r"^[ \t]*" + re.escape(search_lines_cleaned[-1]) + r"[ \t]*\n?"
            )
            pattern_str = r"".join(regex_parts)
            fuzzy_match = re.search(pattern_str, source, re.MULTILINE)
            if fuzzy_match:
                new_code = (
                    source[: fuzzy_match.start()]
                    + replace
                    + "\n"
                    + source[fuzzy_match.end() :]
                )
                return new_code, True
        except Exception:
            pass
        return source, False

    def _attempt_line_by_line_match(
        self, source: str, search: str, replace: str
    ) -> tuple[str, bool]:
        search_lines = search.split("\n")
        search_lines_stripped = [line.strip() for line in search_lines if line.strip()]
        if not search_lines_stripped:
            return source, False

        replace_lines = replace.split("\n")
        code_lines = source.splitlines()
        for i in range(len(code_lines) - len(search_lines_stripped) + 1):
            match = True
            for j, sline in enumerate(search_lines_stripped):
                if sline not in code_lines[i + j].strip():
                    match = False
                    break
            if match:
                new_code_lines = (
                    code_lines[:i]
                    + replace_lines
                    + code_lines[i + len(search_lines_stripped) :]
                )
                new_code = "\n".join(new_code_lines)
                if not new_code.endswith("\n") and source.endswith("\n"):
                    new_code += "\n"
                logger.info(f"Robust fuzzy match succeeded at line {i + 1}.")
                return new_code, True
        return source, False
