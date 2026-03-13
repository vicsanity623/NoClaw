import os

from .core_utils import IGNORE_DIRS, IGNORE_FILES, SUPPORTED_EXTENSIONS


class ScannerMixin:
    # Type hint so Mypy knows this mixin expects a target_dir attribute
    target_dir: str

    def scan_directory(self) -> list[str]:
        file_list = []
        for root, dirs, files in os.walk(self.target_dir):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for file in files:
                if file in IGNORE_FILES:
                    continue
                if (
                    file.endswith(".spec")
                    or file.endswith(".json")
                    or file.endswith(".dmg")
                ):
                    continue
                if any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    file_list.append(os.path.join(root, file))
        return file_list
