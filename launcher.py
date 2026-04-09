"""
Roundtable Launcher - Double-click to start!
Opens the web app in your default browser.
"""

import sys
import os

# When running as exe, set the correct working directory and fix imports
if getattr(sys, 'frozen', False):
    # Get the directory where the exe is running from (temp extraction folder)
    bundle_dir = sys._MEIPASS
    os.chdir(os.path.dirname(sys.executable))
    # Add sd-rpg to Python path so image_gen can import from it
    # Append (not insert) so our main config.py takes precedence over sd-rpg's config.py
    sd_rpg_path = os.path.join(bundle_dir, 'sd-rpg')
    if os.path.exists(sd_rpg_path) and sd_rpg_path not in sys.path:
        sys.path.append(sd_rpg_path)


def show_error(title, message, details=None):
    """Show a friendly error message and wait for user to read it."""
    print()
    print("!" * 60)
    print(f"  ERROR: {title}")
    print("!" * 60)
    print()
    print(message)
    if details:
        print()
        print("Technical details:")
        print(f"  {details}")
    print()
    print("-" * 60)
    print("Press Enter to exit...")
    try:
        input()
    except:
        pass
    sys.exit(1)


def check_dependencies():
    """Check that all required dependencies are available."""
    missing = []

    # Core Python packages
    try:
        import flask
    except ImportError as e:
        missing.append(("Flask", str(e)))

    try:
        import pydantic
    except ImportError as e:
        missing.append(("Pydantic", str(e)))

    try:
        import requests
    except ImportError as e:
        missing.append(("Requests", str(e)))

    try:
        import httpx
    except ImportError as e:
        missing.append(("HTTPX", str(e)))

    try:
        from PIL import Image
    except ImportError as e:
        missing.append(("Pillow (PIL)", str(e)))

    # API clients (optional but good to check)
    try:
        import anthropic
    except ImportError:
        pass  # Optional - only needed for Anthropic API

    try:
        import openai
    except ImportError:
        pass  # Optional - only needed for OpenAI API

    if missing:
        pkg_list = "\n".join([f"  - {name}" for name, _ in missing])
        detail_list = "\n".join([f"  {name}: {err}" for name, err in missing])
        show_error(
            "Missing Dependencies",
            f"The following required packages could not be loaded:\n{pkg_list}\n\n"
            "This might mean:\n"
            "  1. The executable is corrupted - try re-downloading\n"
            "  2. Your antivirus quarantined some files\n"
            "  3. You need Visual C++ Redistributable installed\n"
            "     Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe",
            detail_list
        )


def check_data_directory():
    """Check that we can create/write to the data directory."""
    from pathlib import Path

    data_dir = Path.home() / ".roundtable"
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        # Try to write a test file
        test_file = data_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
    except PermissionError:
        show_error(
            "Cannot Write Data",
            f"Roundtable needs to save data to:\n  {data_dir}\n\n"
            "But it doesn't have permission to write there.\n\n"
            "Try:\n"
            "  1. Run Roundtable as Administrator\n"
            "  2. Check that your user folder isn't read-only\n"
            "  3. Check your antivirus isn't blocking writes"
        )
    except Exception as e:
        show_error(
            "Cannot Create Data Directory",
            f"Roundtable needs to save data to:\n  {data_dir}\n\n"
            "But something went wrong creating that folder.",
            str(e)
        )


def check_port_available():
    """Check if port 5055 is available."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('127.0.0.1', 5055))
        sock.close()
    except OSError:
        show_error(
            "Port 5055 Already In Use",
            "Another program is using port 5055.\n\n"
            "This could mean:\n"
            "  1. Roundtable is already running - check your browser!\n"
            "  2. Another app is using port 5055\n\n"
            "Try:\n"
            "  1. Close other Roundtable windows\n"
            "  2. Restart your computer if nothing else works"
        )


def main():
    """Main entry point with full error handling."""
    print()
    print("=" * 60)
    print("  ROUNDTABLE - AI Conversation Partners")
    print("=" * 60)
    print()
    print("Running startup checks...")

    # Run all checks
    check_dependencies()
    check_data_directory()
    check_port_available()

    print("  [OK] All checks passed!")
    print()
    print("Starting server...")
    print("Opening browser to http://127.0.0.1:5055")
    print()
    print("Keep this window open while using Roundtable.")
    print("Close this window to stop the server.")
    print()
    print("-" * 60)

    # Open browser in background thread
    import webbrowser
    import threading
    import time

    def open_browser():
        time.sleep(1.5)
        webbrowser.open('http://127.0.0.1:5055')

    threading.Thread(target=open_browser, daemon=True).start()

    # Import and run the Flask app
    try:
        from web_app import app
    except Exception as e:
        show_error(
            "Failed to Load Application",
            "The main application code couldn't be loaded.\n\n"
            "This usually means a bug in the code or a missing file.",
            str(e)
        )

    # Run without debug mode, suppress most logging
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)

    try:
        app.run(host='127.0.0.1', port=5055, debug=False, use_reloader=False)
    except Exception as e:
        show_error(
            "Server Crashed",
            "The web server encountered an error and stopped.",
            str(e)
        )


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # Catch-all for anything we didn't anticipate
        show_error(
            "Unexpected Error",
            "Something unexpected went wrong.\n\n"
            "Please report this at:\n"
            "  https://github.com/yourusername/roundtable/issues",
            f"{type(e).__name__}: {e}"
        )
