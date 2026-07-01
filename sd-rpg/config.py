"""
Config and prompt management.
Save/load system prompts, settings, generation history.
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

CONFIG_DIR = Path(__file__).parent
PROMPTS_DIR = CONFIG_DIR / "prompts"
GALLERY_DIR = CONFIG_DIR / "gallery"

@dataclass
class PromptTemplate:
    name: str
    system_prompt: str  # The base context (e.g., "fantasy RPG, medieval setting")
    negative: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 20
    cfg: float = 7.0
    checkpoint: Optional[str] = None
    created: str = ""

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now().isoformat()

    def build_prompt(self, scene_description: str) -> str:
        """Combine system prompt with scene description."""
        if self.system_prompt:
            return f"{self.system_prompt}, {scene_description}"
        return scene_description


class PromptManager:
    def __init__(self):
        PROMPTS_DIR.mkdir(exist_ok=True)
        self.prompts: dict[str, PromptTemplate] = {}
        self._load_all()

    def _load_all(self):
        """Load all saved prompt templates."""
        for f in PROMPTS_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                self.prompts[data["name"]] = PromptTemplate(**data)
            except Exception as e:
                print(f"Warning: couldn't load {f}: {e}")

    def save(self, template: PromptTemplate):
        """Save a prompt template."""
        self.prompts[template.name] = template
        filepath = PROMPTS_DIR / f"{template.name}.json"
        filepath.write_text(json.dumps(asdict(template), indent=2))

    def get(self, name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name."""
        return self.prompts.get(name)

    def list_all(self) -> list[str]:
        """List all saved template names."""
        return list(self.prompts.keys())

    def delete(self, name: str):
        """Delete a prompt template."""
        if name in self.prompts:
            del self.prompts[name]
            filepath = PROMPTS_DIR / f"{name}.json"
            if filepath.exists():
                filepath.unlink()


@dataclass
class GenerationRecord:
    prompt: str
    negative: str
    template_name: Optional[str]
    image_path: str
    seed: int
    timestamp: str
    width: int
    height: int
    steps: int
    cfg: float


class GalleryManager:
    def __init__(self):
        GALLERY_DIR.mkdir(exist_ok=True)
        self.history_file = GALLERY_DIR / "history.json"
        self.history: list[GenerationRecord] = []
        self._load_history()

    def _load_history(self):
        """Load generation history."""
        if self.history_file.exists():
            try:
                data = json.loads(self.history_file.read_text())
                self.history = [GenerationRecord(**r) for r in data]
            except:
                self.history = []

    def _save_history(self):
        """Save generation history."""
        data = [asdict(r) for r in self.history]
        self.history_file.write_text(json.dumps(data, indent=2))

    def add(self, record: GenerationRecord):
        """Add a generation to history."""
        self.history.append(record)
        self._save_history()

    def save_image(self, image_data: bytes, prefix: str = "gen") -> str:
        """Save image bytes to gallery, return filepath."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        filepath = GALLERY_DIR / filename
        filepath.write_bytes(image_data)
        return str(filepath)

    def get_recent(self, count: int = 20) -> list[GenerationRecord]:
        """Get most recent generations."""
        return self.history[-count:][::-1]


# Create some default templates
def create_default_templates():
    manager = PromptManager()

    if not manager.list_all():
        # Fantasy RPG
        manager.save(PromptTemplate(
            name="fantasy",
            system_prompt="fantasy art, medieval setting, detailed illustration, dramatic lighting",
            negative="blurry, low quality, text, watermark, signature, modern elements",
            steps=25,
            cfg=7.5
        ))

        # Dark/Horror RPG
        manager.save(PromptTemplate(
            name="dark-fantasy",
            system_prompt="dark fantasy art, gothic, ominous atmosphere, detailed illustration",
            negative="blurry, low quality, text, watermark, bright colors, cheerful",
            steps=25,
            cfg=8.0
        ))

        # Sci-fi
        manager.save(PromptTemplate(
            name="scifi",
            system_prompt="science fiction art, futuristic, detailed illustration, cinematic lighting",
            negative="blurry, low quality, text, watermark, fantasy elements, medieval",
            steps=25,
            cfg=7.5
        ))

        print("Created default prompt templates: fantasy, dark-fantasy, scifi")


if __name__ == "__main__":
    create_default_templates()
    pm = PromptManager()
    print("Available templates:", pm.list_all())
