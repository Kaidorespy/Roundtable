"""
App-wide settings that persist between sessions.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

SETTINGS_FILE = Path(__file__).parent / "settings.json"


@dataclass
class Settings:
    # Ollama settings
    use_ollama_review: bool = False
    ollama_model: str = "llava"
    ollama_auto_reject_below: float = 5.0  # Auto-reject images scoring below this

    # Generation settings
    batch_count: int = 1                   # How many images to generate at once
    pick_random: bool = False              # If batch > 1, pick one at random vs keep all
    auto_open_image: bool = True           # Open image after generation
    auto_open_gallery: bool = False        # Open gallery after generation

    # ComfyUI connection
    comfy_host: str = "127.0.0.1"
    comfy_port: int = 8188

    # Defaults
    default_template: Optional[str] = None
    default_checkpoint: Optional[str] = None
    default_steps: int = 20
    default_cfg: float = 7.0
    default_width: int = 1024
    default_height: int = 1024
    default_negative: str = "blurry, low quality, text, watermark"

    # Behavior
    save_all_images: bool = True           # Save even rejected images (in subfolder)
    confirm_before_generate: bool = False  # Ask before each generation
    show_seed_in_filename: bool = True     # Include seed in saved filename

    # Queue ahead (for RPG integration later)
    enable_queue_ahead: bool = False
    queue_ahead_count: int = 3

    def save(self):
        """Save settings to disk."""
        SETTINGS_FILE.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from disk, or create defaults."""
        if SETTINGS_FILE.exists():
            try:
                data = json.loads(SETTINGS_FILE.read_text())
                return cls(**data)
            except Exception as e:
                print(f"Warning: couldn't load settings: {e}")
        return cls()


def print_settings(settings: Settings):
    """Pretty print current settings."""
    print("\n=== Current Settings ===\n")

    print("Ollama Review:")
    print(f"  use_ollama_review: {settings.use_ollama_review}")
    print(f"  ollama_model: {settings.ollama_model}")
    print(f"  auto_reject_below: {settings.ollama_auto_reject_below}")

    print("\nGeneration:")
    print(f"  batch_count: {settings.batch_count}")
    print(f"  pick_random: {settings.pick_random}")
    print(f"  auto_open_image: {settings.auto_open_image}")

    print("\nDefaults:")
    print(f"  default_template: {settings.default_template or '(none)'}")
    print(f"  default_checkpoint: {settings.default_checkpoint or '(auto)'}")
    print(f"  default_steps: {settings.default_steps}")
    print(f"  default_cfg: {settings.default_cfg}")
    print(f"  default_size: {settings.default_width}x{settings.default_height}")

    print("\nBehavior:")
    print(f"  save_all_images: {settings.save_all_images}")
    print(f"  show_seed_in_filename: {settings.show_seed_in_filename}")

    print()


# Interactive settings editor
def edit_settings():
    """Simple CLI settings editor."""
    settings = Settings.load()

    print_settings(settings)

    print("Enter setting name and value (e.g., 'use_ollama_review true')")
    print("Type 'save' to save, 'quit' to exit without saving\n")

    while True:
        try:
            line = input("settings> ").strip()

            if not line:
                continue

            if line == "quit":
                return

            if line == "save":
                settings.save()
                print("Settings saved!")
                return

            if line == "show":
                print_settings(settings)
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                print("Usage: <setting_name> <value>")
                continue

            key, value = parts

            if not hasattr(settings, key):
                print(f"Unknown setting: {key}")
                continue

            # Parse value based on current type
            current = getattr(settings, key)

            if isinstance(current, bool):
                setattr(settings, key, value.lower() in ("true", "1", "yes", "on"))
            elif isinstance(current, int):
                setattr(settings, key, int(value))
            elif isinstance(current, float):
                setattr(settings, key, float(value))
            else:
                setattr(settings, key, value if value.lower() != "none" else None)

            print(f"{key} = {getattr(settings, key)}")

        except KeyboardInterrupt:
            print("\nExiting without saving.")
            return
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    edit_settings()
