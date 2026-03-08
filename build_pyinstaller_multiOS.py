import platform
import subprocess
import sys
from pathlib import Path


def main():
    os_name = platform.system().lower()
    project_root = Path(__file__).parent.absolute()
    VERSION = "0.2.4"
    APP_NAME = "Py-OB"
    print(f"🚀 Forging {APP_NAME} v{VERSION} for {os_name}...")
    common = [
        f"--name={APP_NAME}",
        "--clean",
        "--noconfirm",
        "--paths=src",
        "--collect-all=pyob",
        "--hidden-import=pyob.autoreviewer",
        "--hidden-import=pyob.core_utils",
        "--hidden-import=pyob.prompts_and_memory",
        "--hidden-import=pyob.reviewer_mixins",
        "--hidden-import=pyob.entrance",
        "--hidden-import=pyob.pyob_code_parser",
        "--hidden-import=pyob.pyob_dashboard",
        "--hidden-import=requests",
        "--hidden-import=ollama",
        "--hidden-import=textwrap",
        "--hidden-import=pathlib",
        "--hidden-import=charset_normalizer",
        "--hidden-import=chardet",
        "--copy-metadata=requests",
        "--copy-metadata=charset-normalizer",
        "--copy-metadata=chardet",
        "--collect-all=ruff",
        "--collect-all=mypy",
        "--collect-all=ollama",
        "--collect-all=charset_normalizer",
        "--collect-all=chardet",
        "src/pyob/pyob_launcher.py",
    ]

    if os_name == "darwin":
        cmd = (
            ["pyinstaller"]
            + common
            + [
                "--windowed",
                "--icon=pyob.icns",
            ]
        )
        dist_output = project_root / "dist" / f"{APP_NAME}.app"

    elif os_name == "windows":
        cmd = (
            ["pyinstaller"]
            + common
            + [
                "--onefile",
                "--console",
                "--icon=pyob.ico",
            ]
        )
        dist_output = project_root / "dist" / f"{APP_NAME}.exe"

    else:
        cmd = ["pyinstaller"] + common + ["--onefile", "--console"]
        dist_output = project_root / "dist" / APP_NAME

    print(f"🛠️  Running PyInstaller for {APP_NAME}...")
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ PyInstaller Build complete: {dist_output}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ PyInstaller Build failed: {e}")
        sys.exit(1)

    if os_name == "darwin":
        print("\n📦 Starting DMG creation...")

        dmg_name = f"{APP_NAME}-v{VERSION}.dmg"
        dmg_path = project_root / dmg_name

        if dmg_path.exists():
            dmg_path.unlink()

        dmg_cmd = [
            "create-dmg",
            "--volname",
            f"{APP_NAME} Installer",
            "--app-drop-link",
            "400",
            "120",
            "--window-size",
            "600",
            "400",
            "--icon-size",
            "150",
            "--icon",
            f"{APP_NAME}.app",
            "150",
            "120",
            str(dmg_path),
            str(dist_output),
        ]

        try:
            subprocess.run(dmg_cmd, check=True)
            print(f"\n🔥 SUCCESS! DMG created at: {dmg_path}")
            print(
                f"You can now distribute the DMG. Users can drag {APP_NAME} to Applications."
            )
        except FileNotFoundError:
            print("\n❌ Error: 'create-dmg' tool not found.")
            print("Fix: brew install create-dmg")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error creating DMG: {e}")


if __name__ == "__main__":
    main()
