import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

CONFIG_FILE = Path.home() / ".pyob_config"

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_LOCAL_MODEL = "qwen3-coder:30b"

# OBSERVER_PATCH_REVIEW_HTML content has been moved to pyob_dashboard.py
# or a dedicated UI template file, as it is UI-specific content.


def load_config():
    """Load config from file or environment, or prompt user if missing."""
    # 1. Try loading from the local configuration file first
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(
                f"Warning: Configuration file {CONFIG_FILE} is invalid or inaccessible ({e}). Re-creating."
            )
        except Exception as e:
            print(
                f"Warning: An unexpected error occurred while loading config ({e}). Re-creating."
            )

    # 2. Check for Environment Variables (Ensures it works in GitHub Actions/Docker)
    env_keys = os.environ.get("PYOB_GEMINI_KEYS")
    if env_keys:
        return {
            "gemini_keys": env_keys,
            "gemini_model": os.environ.get("PYOB_GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            "local_model": os.environ.get("PYOB_LOCAL_MODEL", DEFAULT_LOCAL_MODEL),
        }

    # 3. Safety Check for Headless Environments
    if not sys.stdin.isatty():
        print("Error: No API keys found in environment and stdin is not a TTY.")
        print("   In GitHub Actions, please set the PYOB_GEMINI_KEYS secret.")
        sys.exit(1)

    # 4. Standard Interactive Setup (reached on local iMac first-run)
    print("PYOB First-Time Setup")
    print("═" * 40)
    print("\nStep 1: Gemini API Keys")
    print("Enter up to 10 keys separated by commas:")
    keys = input("Keys: ").strip()

    print("\nStep 2: Model Configuration")
    print("WARNING: PYOB is optimized for 'gemini-2.0-flash' and 'qwen3-coder:30b'.")
    print("   Changing these may result in parsing errors or logic loops.")

    g_model = (
        input(f"\nEnter Gemini Model [default: {DEFAULT_GEMINI_MODEL}]: ").strip()
        or DEFAULT_GEMINI_MODEL
    )
    l_model = (
        input(f"Enter Local Ollama Model [default: {DEFAULT_LOCAL_MODEL}]: ").strip()
        or DEFAULT_LOCAL_MODEL
    )

    config = {"gemini_keys": keys, "gemini_model": g_model, "local_model": l_model}

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\nConfiguration saved to {CONFIG_FILE}")
    print("   (To change these later, simply delete that file and restart PYOB.)\n")
    return config


def ensure_terminal():
    if ".app/Contents/MacOS" in sys.executable and not os.isatty(sys.stdin.fileno()):
        script_path = shlex.quote(sys.argv[0])
        args = " ".join(shlex.quote(arg) for arg in sys.argv[1:])
        full_command = f"{sys.executable} {script_path} {args}".strip()
        cmd = f'tell application "Terminal" to do script "{full_command}"'
        subprocess.run(["osascript", "-e", cmd])
        sys.exit(0)


def main():
    if sys.platform == "darwin":
        ensure_terminal()

    print("═" * 70)
    print("                  PYOB Launcher")
    print("═" * 70)

    config = load_config()

    # Prioritize environment variables if set (e.g. by Docker/Actions)
    os.environ.setdefault("PYOB_GEMINI_KEYS", config.get("gemini_keys", ""))
    os.environ.setdefault(
        "PYOB_GEMINI_MODEL", config.get("gemini_model", DEFAULT_GEMINI_MODEL)
    )
    os.environ.setdefault(
        "PYOB_LOCAL_MODEL", config.get("local_model", DEFAULT_LOCAL_MODEL)
    )

    from pyob.entrance import EntranceController

    target_dir = ""
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        # Filter out internal macOS process paths
        if ".app/Contents/MacOS" in arg or arg == sys.executable:
            if sys.stdin.isatty():
                target_dir = input(
                    "\nEnter the FULL PATH to your target project directory\n"
                    "(or just press Enter to use the current folder): "
                ).strip()
            else:
                target_dir = "."
        else:
            target_dir = arg
    else:
        if sys.stdin.isatty():
            target_dir = input(
                "\nEnter the FULL PATH to your target project directory\n"
                "(or just press Enter to use the current folder): "
            ).strip()
        else:
            target_dir = "."

    if not target_dir:
        target_dir = "."

    target_dir = os.path.abspath(target_dir)

    if not os.path.isdir(target_dir):
        print(f"Error: Directory does not exist → {target_dir}")
        if sys.stdin.isatty():
            input("\nPress Enter to exit...")
        sys.exit(1)

    # --- NEW: CLOUD GIT CONFIGURATION FIX ---
    if os.environ.get("GITHUB_ACTIONS") == "true":
        # 1. Fix the "dubious ownership" block for Docker volumes
        subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", "*"],
            capture_output=True,
        )

        # 2. Auto-set Bot Identity (Prevents "Author unknown" errors for Marketplace users)
        check_name = subprocess.run(
            ["git", "config", "user.name"], capture_output=True, text=True
        )
        if not check_name.stdout.strip():
            subprocess.run(
                ["git", "config", "--global", "user.name", "pyob-bot"],
                capture_output=True,
            )
            subprocess.run(
                [
                    "git",
                    "config",
                    "--global",
                    "user.email",
                    "pyob-bot@users.noreply.github.com",
                ],
                capture_output=True,
            )
    # ----------------------------------------

    print(f"\nStarting PYOB on: {target_dir}")
    print(f"Gemini Model: {os.environ['PYOB_GEMINI_MODEL']}")
    print(f"Local Model:  {os.environ['PYOB_LOCAL_MODEL']}")
    print("   (Terminal will stay open — press Ctrl+C to stop)\n")

    try:
        controller = EntranceController(target_dir)
        controller.run_master_loop()
    except KeyboardInterrupt:
        print("\n\nPYOB stopped by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if sys.stdin.isatty():
            input("\nPress Enter to close this window...")


if __name__ == "__main__":
    main()
