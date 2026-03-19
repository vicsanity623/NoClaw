import os

from .autoreviewer import AutoReviewer
from .xml_mixin import ApplyXMLMixin


class TargetedReviewer(AutoReviewer, ApplyXMLMixin):
    """
    Specialized reviewer that targets a single specific file.
    Automatically handles both absolute and relative path inputs for cross-platform stability.
    """

    def __init__(self, target_dir: str, target_file: str):
        super().__init__(target_dir)

        if os.path.isabs(target_file):
            self.forced_target_file = os.path.relpath(target_file, self.target_dir)
        else:
            self.forced_target_file = target_file

    def scan_directory(self) -> list[str]:
        """Returns only the specific targeted file for the pipeline to process."""
        full_target_path = os.path.join(self.target_dir, self.forced_target_file)
        if os.path.exists(full_target_path):
            return [full_target_path]
        return []
