import sys
import os
import json
import subprocess
from pathlib import Path

CONFIG_FILE = Path.home() / ".pyob_config"


def load_config():
    """Load config, or prompt user if missing."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass

    print("🛠️  PYOB First-Time Setup")
    print("═" * 40)
    print("\n🔑 Step 1: Gemini API Keys")
    print("Enter up to 10 keys separated by commas:")
    keys = input("Keys: ").strip()
    print("\n🤖 Step 2: Model Configuration")
    print("⚠️  WARNING: PYOB is optimized for 'gemini-2.5-flash' and 'qwen3-coder:30b'.")
    print("   Changing these may result in parsing errors or logic loops.")

    g_model = input("\nEnter Gemini Model [default: gemini-2.5-flash]: ").strip()
    if not g_model:
        g_model = "gemini-2.5-flash"

    l_model = input("Enter Local Ollama Model [default: qwen3-coder:30b]: ").strip()
    if not l_model:
        l_model = "qwen3-coder:30b"

    config = {"gemini_keys": keys, "gemini_model": g_model, "local_model": l_model}

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\n✅ Configuration saved to {CONFIG_FILE}")
    print("   (To change these later, simply delete that file and restart PYOB.)\n")
    return config


def ensure_terminal():
    if ".app/Contents/MacOS" in sys.executable and not os.isatty(sys.stdin.fileno()):
        cmd = f'tell application "Terminal" to do script "{sys.executable}"'
        subprocess.run(["osascript", "-e", cmd])
        sys.exit(0)


if __name__ == "__main__":
    if sys.platform == "darwin":
        ensure_terminal()

    print("═" * 70)
    print("                  PYOB Launcher")
    print("═" * 70)
    config = load_config()
    os.environ["PYOB_GEMINI_KEYS"] = config.get("gemini_keys", "")
    os.environ["PYOB_GEMINI_MODEL"] = config.get("gemini_model", "gemini-2.5-flash")
    os.environ["PYOB_LOCAL_MODEL"] = config.get("local_model", "qwen3-coder:30b")

    #  Import App Logic (After Env is set)
    from entrance import EntranceController

    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = input(
            "\nEnter the FULL PATH to your target project directory\n"
            "(or just press Enter to use the current folder): "
        ).strip()

        if not target_dir:
            target_dir = "."

    target_dir = os.path.abspath(target_dir)

    if not os.path.isdir(target_dir):
        print(f"❌ Error: Directory does not exist → {target_dir}")
        input("\nPress Enter to exit...")
        sys.exit(1)

    print(f"\n🚀 Starting PYOB on: {target_dir}")
    print(f"🧠 Gemini Model: {os.environ['PYOB_GEMINI_MODEL']}")
    print(f"🏠 Local Model:  {os.environ['PYOB_LOCAL_MODEL']}")
    print("   (Terminal will stay open — press Ctrl+C to stop)\n")

    try:
        controller = EntranceController(target_dir)
        controller.run_master_loop()
    except KeyboardInterrupt:
        print("\n\nPYOB stopped by user.")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        input("\nPress Enter to close this window...")
