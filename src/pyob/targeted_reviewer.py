import os

from .autoreviewer import AutoReviewer
from .xml_mixin import ApplyXMLMixin


class TargetedReviewer(AutoReviewer, ApplyXMLMixin):
    def __init__(self, target_dir: str, target_file: str):
        super().__init__(target_dir)
        if os.path.isabs(target_file):
            raise ValueError(f"target_file must be a relative path, got: {target_file}")
        self.forced_target_file = target_file

    def scan_directory(self) -> list[str]:
        full_target_path = os.path.join(self.target_dir, self.forced_target_file)
        if os.path.exists(full_target_path):
            return [full_target_path]
        return []
