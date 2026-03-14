import os

from .apply_xml_mixins import ApplyXMLMixin
from .autoreviewer import AutoReviewer


class TargetedReviewer(AutoReviewer, ApplyXMLMixin):
    def __init__(self, target_dir: str, target_file: str):
        super().__init__(target_dir)
        self.forced_target_file = target_file

    def scan_directory(self) -> list[str]:
        if os.path.exists(self.forced_target_file):
            return [self.forced_target_file]
        return []
