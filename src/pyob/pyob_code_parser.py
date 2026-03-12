import ast
import logging
import os
import re

logger = logging.getLogger(__name__)


class CodeParser:
    def generate_structure_dropdowns(self, filepath: str, code: str) -> str:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".py":
            return self._parse_python(code)
        elif ext in [".js", ".ts", ".jsx", ".tsx"]:
            return self._parse_javascript(code)
        elif ext in [".html", ".htm"]:
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
                    try:
                        imports.append(ast.unparse(node))
                    except Exception:
                        pass
                elif isinstance(node, ast.ClassDef):
                    classes.append(f"class {node.name}")
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            args = [arg.arg for arg in child.args.args if arg.arg != "self"]
                            functions.append(f"def {node.name}.{child.name}({', '.join(args)})")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not any(f".{node.name}(" in fn for fn in functions):
                        args = [arg.arg for arg in node.args.args]
                        if node.args.vararg:
                            args.append(f"*{node.args.vararg.arg}")
                        if node.args.kwarg:
                            args.append(f"**{node.args.kwarg.arg}")
                        functions.append(f"def {node.name}({', '.join(args)})")
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id.isupper():
                            consts.append(t.id)
                            
            return self._format_dropdowns(imports, classes, functions, consts)
            
        except SyntaxError as e:
            logger.warning(f"AST parsing failed (SyntaxError: {e}). Falling back to Regex for structure map.")
            return self._parse_python_regex_fallback(code)
        except Exception as e:
            logger.error(f"Unexpected AST parse error: {e}")
            return ""

    def _parse_python_regex_fallback(self, code: str) -> str:
        """Used when a Python file has syntax errors so the AI isn't blinded."""
        imports = re.findall(r"^(?:import|from)\s+[a-zA-Z0-9_\.]+", code, re.MULTILINE)
        classes = [f"class {c}" for c in re.findall(r"^class\s+([a-zA-Z0-9_]+)", code, re.MULTILINE)]
        functions = [f"def {f}()" for f in re.findall(r"^[ \t]*def\s+([a-zA-Z0-9_]+)", code, re.MULTILINE)]
        consts = list(set(re.findall(r"^([A-Z_][A-Z0-9_]+)\s*=", code, re.MULTILINE)))
        
        return self._format_dropdowns(imports, classes, functions, consts)

    def _parse_javascript(self, code: str) -> str:
        imports = re.findall(r"(?:import|from|require)\s+['\"].*?['\"]", code)
        classes = re.findall(r"(?:class|interface)\s+([a-zA-Z0-9_$]+)", code)
        types = re.findall(r"type\s+([a-zA-Z0-9_$]+)\s*=", code)
        classes.extend([f"type {t}" for t in types])

        fn_patterns = [
            r"function\s+([a-zA-Z0-9_$]+)\s*\(([^)]*)\)",
            r"(?:const|let|var|window\.)\s*([a-zA-Z0-9_$]+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>",
            r"^\s*(?:async\s*)?([a-zA-Z0-9_$]+)\s*\(([^)]*)\)\s*\{",
        ]
        raw_fns = []
        for pattern in fn_patterns:
            raw_fns.extend(re.findall(pattern, code, re.MULTILINE))

        clean_fns = []
        seen = set()
        for name, params in raw_fns:
            if name not in seen and name not in ["if", "for", "while", "return", "catch", "switch"]:
                clean_fns.append(f"{name}({params.strip()})")
                seen.add(name)

        entities = re.findall(r"(?:const|var|let)\s+([A-Z0-9_]{3,})", code)
        return self._format_dropdowns(
            imports, classes, sorted(clean_fns), sorted(list(set(entities)))
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
        selectors = re.findall(r"([#\.][a-zA-Z0-9_-]+)\s*\{", code)
        unique_selectors = list(dict.fromkeys(selectors))
        return self._format_dropdowns([], [], unique_selectors[:50], [])

    def _format_dropdowns(self, imp: list, cls: list, fn: list, cnst: list) -> str:
        res = ""
        if imp:
            res += f"<details><summary>Imports ({len(imp)})</summary>{'<br>'.join(sorted(imp))}</details>\n"
        if cnst:
            res += f"<details><summary>Entities ({len(cnst)})</summary>{'<br>'.join(sorted(cnst))}</details>\n"
        if cls:
            res += f"<details><summary>Classes/Types ({len(cls)})</summary>{'<br>'.join(sorted(cls))}</details>\n"
        if fn:
            res += f"<details><summary>Logic ({len(fn)})</summary>{'<br>'.join(sorted(fn))}</details>\n"
        return res
