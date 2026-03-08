from pyob.core_utils import CoreUtilsMixin


class DummyAgent(CoreUtilsMixin):
    def __init__(self, target_dir):
        self.target_dir = target_dir


def test_directory_scanner_ignores_hidden_folders(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    valid_file = src_dir / "logic.py"
    valid_file.write_text("print('hello')")

    ignored_file = git_dir / "config"
    ignored_file.write_text("ignore me")

    from pyob.autoreviewer import AutoReviewer

    reviewer = AutoReviewer(str(tmp_path))
    files = reviewer.scan_directory()

    assert any("logic.py" in f for f in files)
    assert not any(".git" in f for f in files)


def test_directory_scanner_finds_all_relevant_files(tmp_path):
    # Create a more complex directory structure
    root_dir = tmp_path / "project"
    root_dir.mkdir()

    # Files in root
    (root_dir / "main.py").write_text("print('main')")
    (root_dir / "README.md").write_text("# Project")

    # Subdirectory 1
    sub_dir_1 = root_dir / "src"
    sub_dir_1.mkdir()
    (sub_dir_1 / "module_a.py").write_text("class A: pass")
    (sub_dir_1 / "data.json").write_text("{}")

    # Subdirectory 2 (nested)
    sub_dir_2 = sub_dir_1 / "utils"
    sub_dir_2.mkdir()
    (sub_dir_2 / "helper.py").write_text("def help(): pass")

    # Hidden directory and its contents - should be ignored
    hidden_dir = root_dir / ".venv"
    hidden_dir.mkdir()
    (hidden_dir / "activate").write_text("source")

    # Hidden file - should be ignored
    (root_dir / ".gitignore").write_text("*.log")

    from pyob.autoreviewer import AutoReviewer

    reviewer = AutoReviewer(str(root_dir))
    files = reviewer.scan_directory()

    # Expected files (absolute paths)
    expected_files = [
        str(root_dir / "main.py"),
        str(root_dir / "README.md"),
        str(sub_dir_1 / "module_a.py"),
        str(sub_dir_1 / "data.json"),
        str(sub_dir_2 / "helper.py"),
    ]

    # Sort both lists for consistent comparison
    files.sort()
    expected_files.sort()

    assert len(files) == len(expected_files)
    assert files == expected_files
