import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

CONFIG_FILE = Path.home() / ".pyob_config"

DEFAULT_GEMINI_MODEL = "gemini-3.7-flash"
DEFAULT_LOCAL_MODEL = "llama3.2:3b"

# OBSERVER_PATCH_REVIEW_HTML content has been moved to pyob_dashboard.py
# or a dedicated UI template file, as it is UI-specific content.


def load_config():
    """Load config from file or environment, or prompt user if missing."""
    # 1. Highest Priority: Environment Variables (Cloud/Docker)
    if os.environ.get("PYOB_OPENROUTER_KEY") or os.environ.get("PYOB_GEMINI_KEYS"):
        return {
            "openrouter_key": os.environ.get("PYOB_OPENROUTER_KEY", ""),
            "openrouter_model": os.environ.get(
                "PYOB_OPENROUTER_MODEL", "deepseek/deepseek-v4-flash"
            ),
            "gemini_keys": os.environ.get("PYOB_GEMINI_KEYS", ""),
            "gemini_model": os.environ.get("PYOB_GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            "local_model": os.environ.get("PYOB_LOCAL_MODEL", DEFAULT_LOCAL_MODEL),
        }

    # 2. Try loading from the local configuration file (.pyob_config)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                config_data = json.load(f)
                if isinstance(config_data, dict):
                    if "openrouter_key" not in config_data:
                        config_data["openrouter_key"] = ""
                    if "openrouter_model" not in config_data:
                        config_data["openrouter_model"] = (
                            "deepseek/deepseek-v4-flash"
                        )
                    return config_data
        except Exception as e:
            print(f"Warning: Configuration file error ({e}).")

    # 3. Standard Interactive Setup (First run on iMac)
    if not sys.stdin.isatty():
        sys.exit(1)

    print("PYOB First-Time Setup")
    print("═" * 40)
    print("\nStep 1: OpenRouter API Key (Optional but Recommended)")
    or_key = input("Enter OpenRouter Key (press Enter to skip): ").strip()

    print("\nStep 2: Gemini API Keys (Fallback)")
    print("Enter keys separated by commas:")
    g_keys = input("Keys: ").strip()

    print("\nStep 3: Model Configuration")
    g_model = (
        input(f"Gemini Model [default: {DEFAULT_GEMINI_MODEL}]: ").strip()
        or DEFAULT_GEMINI_MODEL
    )
    l_model = (
        input(f"Local Model [default: {DEFAULT_LOCAL_MODEL}]: ").strip()
        or DEFAULT_LOCAL_MODEL
    )

    config = {
        "openrouter_key": or_key,
        "openrouter_model": "deepseek/deepseek-v4-flash",
        "gemini_keys": g_keys,
        "gemini_model": g_model,
        "local_model": l_model,
    }

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)
    return config


def ensure_terminal():
    if ".app/Contents/MacOS" in sys.executable and not os.isatty(sys.stdin.fileno()):
        script_path = shlex.quote(sys.argv[0])
        args = " ".join(shlex.quote(arg) for arg in sys.argv[1:])
        full_command = f"{sys.executable} {script_path} {args}".strip()

        escaped_command = full_command.replace('"', '\\"')
        cmd = f'tell application "Terminal" to do script "{escaped_command}"'

        subprocess.run(["osascript", "-e", cmd])
        sys.exit(0)


def main():
    if sys.platform == "darwin":
        ensure_terminal()

    print("═" * 70)
    print("                  PYOB Launcher")
    print("═" * 70)

    config = load_config()

    # Inject into environment so models.py can see them
    os.environ.setdefault("PYOB_OPENROUTER_KEY", config.get("openrouter_key", ""))
    os.environ.setdefault(
        "PYOB_OPENROUTER_MODEL",
        config.get("openrouter_model", "deepseek/deepseek-v4-flash"),
    )
    os.environ.setdefault("PYOB_GEMINI_KEYS", config.get("gemini_keys", ""))
    os.environ.setdefault(
        "PYOB_GEMINI_MODEL", config.get("gemini_model", DEFAULT_GEMINI_MODEL)
    )
    os.environ.setdefault(
        "PYOB_LOCAL_MODEL", config.get("local_model", DEFAULT_LOCAL_MODEL)
    )

    # Determine if dashboard is active, e.g., via environment variable
    dashboard_active = (
        os.environ.get("PYOB_DASHBOARD_ACTIVE", "false").lower() == "true"
    )

    from pyob.entrance import EntranceController

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

    target_dir = os.path.abspath(os.path.expanduser(target_dir))

    if not os.path.isdir(target_dir):
        print(f"Error: Directory does not exist → {target_dir}")
        if sys.stdin.isatty():
            input("\nPress Enter to exit...")
        sys.exit(1)

    if os.environ.get("GITHUB_ACTIONS") == "true":
        try:
            subprocess.run(
                ["git", "config", "--global", "--add", "safe.directory", "*"],
                capture_output=True,
            )

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
        except (FileNotFoundError, Exception) as e:
            print(
                f"Warning: Could not configure Git settings. git binary may be missing: {e}"
            )

    print(f"\nStarting PYOB on: {target_dir}")
    or_status = (
        config.get("openrouter_model") if config.get("openrouter_key") else "DISABLED"
    )
    print(f"OpenRouter:   {or_status}")
    print(f"Gemini Model: {os.environ['PYOB_GEMINI_MODEL']}")
    print(f"Local Model:  {os.environ['PYOB_LOCAL_MODEL']}")
    print("   (Terminal will stay open — press Ctrl+C to stop)\n")

    try:
        # Pass dashboard_active status to the EntranceController
        controller = EntranceController(target_dir, dashboard_active=dashboard_active)
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
