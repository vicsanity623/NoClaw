from pyob.autoreviewer import AutoReviewer


def test_directory_scanner_ignores_hidden_folders(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    src_dir = project_dir / "src"
    src_dir.mkdir()

    git_dir = project_dir / ".git"
    git_dir.mkdir()

    valid_file = src_dir / "logic.py"
    valid_file.write_text("print('hello')")

    ignored_file = git_dir / "config"
    ignored_file.write_text("ignore me")

    reviewer = AutoReviewer(str(project_dir))
    files = reviewer.scan_directory()

    assert any("logic.py" in f for f in files)
    assert not any(".git" in f for f in files)


def test_directory_scanner_finds_all_relevant_files(tmp_path):
    # Setup project structure
    root_dir = tmp_path / "project"
    root_dir.mkdir()

    # Files in root
    (root_dir / "main.py").write_text("print('main')")
    (root_dir / "README.md").write_text("# Project")

    # Subdirectory
    sub_dir_1 = root_dir / "src"
    sub_dir_1.mkdir()
    (sub_dir_1 / "module_a.py").write_text("class A: pass")
    (sub_dir_1 / "data.json").write_text("{}")

    # Nested Subdirectory
    sub_dir_2 = sub_dir_1 / "utils"
    sub_dir_2.mkdir()
    (sub_dir_2 / "helper.py").write_text("def help(): pass")

    # Hidden directory (should be ignored)
    hidden_dir = root_dir / ".venv"
    hidden_dir.mkdir()
    (hidden_dir / "activate").write_text("source")

    # Hidden file (should be ignored)
    (root_dir / ".gitignore").write_text("*.log")

    reviewer = AutoReviewer(str(root_dir))
    files = reviewer.scan_directory()

    # Assert specific presence of expected files rather than exact list length,
    # making the test robust to changes in supported extensions.
    expected_filenames = [
        "main.py",
        "module_a.py",
        "data.json",
        "helper.py",
    ]

    for filename in expected_filenames:
        assert any(filename in f for f in files), (
            f"Expected file {filename} not found in scanner results."
        )

    # Assert exclusion of hidden files/folders
    assert not any(".venv" in f for f in files), (
        "Scanner incorrectly included .venv folder."
    )
    assert not any(".gitignore" in f for f in files), (
        "Scanner incorrectly included .gitignore file."
    )
