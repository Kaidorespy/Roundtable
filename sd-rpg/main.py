"""
SD-RPG: Simple prompt-to-image CLI
Type a prompt, get an image. That's it.
"""

import sys
import os
import random
import webbrowser
from datetime import datetime
from pathlib import Path

from comfy_client import ComfyClient
from config import PromptManager, GalleryManager, GenerationRecord, create_default_templates
from settings import Settings, print_settings
from ollama_reviewer import OllamaReviewer


def print_banner():
    print("""
╔═══════════════════════════════════════╗
║           SD-RPG Generator            ║
╚═══════════════════════════════════════╝
    """)


def print_help():
    print("""
Commands:
  /templates     - List saved prompt templates
  /use <name>    - Use a template (combines with your prompt)
  /clear         - Clear current template
  /save <name>   - Save current settings as template
  /gallery       - Open gallery in browser
  /recent        - Show recent generations
  /set <option>  - Set options (steps, cfg, size, checkpoint, batch, etc.)
  /settings      - Show all settings
  /config        - Edit persistent settings
  /status        - Check ComfyUI/Ollama connection
  /review <path> - Review an image with Ollama
  /help          - Show this help
  /quit          - Exit

Just type a prompt to generate an image!
""")


class Generator:
    def __init__(self):
        # Load persistent settings
        self.settings = Settings.load()

        # Initialize clients
        self.client = ComfyClient(
            host=self.settings.comfy_host,
            port=self.settings.comfy_port
        )
        self.prompts = PromptManager()
        self.gallery = GalleryManager()
        self.reviewer = OllamaReviewer(model=self.settings.ollama_model)

        self.current_template = self.settings.default_template

        # Current session settings (override from settings)
        self.steps = self.settings.default_steps
        self.cfg = self.settings.default_cfg
        self.width = self.settings.default_width
        self.height = self.settings.default_height
        self.negative = self.settings.default_negative
        self.checkpoint = self.settings.default_checkpoint
        self.batch_count = self.settings.batch_count

        # Create defaults if needed
        create_default_templates()
        self.prompts = PromptManager()  # Reload after creating defaults

    def check_connection(self) -> bool:
        comfy_ok = self.client.is_running()
        ollama_ok = self.reviewer.is_running()

        if comfy_ok:
            print("ComfyUI: connected")
        else:
            print("ComfyUI: not running (localhost:8188)")

        if self.settings.use_ollama_review:
            if ollama_ok:
                print(f"Ollama: connected (model: {self.settings.ollama_model})")
            else:
                print("Ollama: not running (localhost:11434)")

        return comfy_ok

    def generate(self, prompt: str):
        """Generate image(s) from prompt."""
        if not self.client.is_running():
            print("Error: ComfyUI not running!")
            return

        # Build full prompt with template
        full_prompt = prompt
        template_name = None

        if self.current_template:
            template = self.prompts.get(self.current_template)
            if template:
                full_prompt = template.build_prompt(prompt)
                template_name = self.current_template
                if not self.negative and template.negative:
                    self.negative = template.negative

        print(f"\nGenerating: {full_prompt[:80]}{'...' if len(full_prompt) > 80 else ''}")
        print(f"Settings: {self.width}x{self.height}, {self.steps} steps, cfg {self.cfg}")

        if self.batch_count > 1:
            print(f"Batch: generating {self.batch_count} images...")

        all_results = []

        for i in range(self.batch_count):
            seed = random.randint(0, 2**32 - 1)

            if self.batch_count > 1:
                print(f"  [{i+1}/{self.batch_count}] seed {seed}...", end="", flush=True)
            else:
                print(f"Seed: {seed}, waiting for ComfyUI...", end="", flush=True)

            try:
                images = self.client.generate_image(
                    prompt=full_prompt,
                    negative=self.negative,
                    width=self.width,
                    height=self.height,
                    steps=self.steps,
                    cfg=self.cfg,
                    seed=seed,
                    checkpoint=self.checkpoint
                )

                print(" done!")

                for img_data in images:
                    # Save image
                    prefix = f"gen_{seed}" if self.settings.show_seed_in_filename else "gen"
                    filepath = self.gallery.save_image(img_data, prefix=prefix)

                    result = {
                        "filepath": filepath,
                        "seed": seed,
                        "prompt": full_prompt,
                        "template": template_name
                    }

                    # Ollama review if enabled
                    if self.settings.use_ollama_review and self.reviewer.is_running():
                        print(f"    Reviewing with {self.settings.ollama_model}...", end="", flush=True)
                        try:
                            review = self.reviewer.review_image(filepath, full_prompt)
                            result["review"] = review
                            # Try to parse score
                            for line in review.split('\n'):
                                if 'score:' in line.lower():
                                    try:
                                        score = float(line.split('/')[0].split(':')[-1].strip())
                                        result["score"] = score
                                    except:
                                        pass
                            print(" done!")
                            if "score" in result:
                                print(f"    Score: {result['score']}/10")
                        except Exception as e:
                            print(f" error: {e}")

                    all_results.append(result)

                    # Record in history
                    self.gallery.add(GenerationRecord(
                        prompt=full_prompt,
                        negative=self.negative,
                        template_name=template_name,
                        image_path=filepath,
                        seed=seed,
                        timestamp=datetime.now().isoformat(),
                        width=self.width,
                        height=self.height,
                        steps=self.steps,
                        cfg=self.cfg
                    ))

            except Exception as e:
                print(f" error: {e}")

        # Handle results
        if not all_results:
            print("No images generated.")
            return

        # Pick random if enabled and batch > 1
        if self.settings.pick_random and len(all_results) > 1:
            # If we have scores, pick highest; else random
            scored = [r for r in all_results if "score" in r]
            if scored:
                winner = max(scored, key=lambda x: x["score"])
                print(f"\nBest image (score {winner['score']}/10): {winner['filepath']}")
            else:
                winner = random.choice(all_results)
                print(f"\nRandom pick: {winner['filepath']}")
        else:
            # Show all
            print(f"\nSaved {len(all_results)} image(s):")
            for r in all_results:
                score_str = f" (score: {r['score']}/10)" if "score" in r else ""
                print(f"  {r['filepath']}{score_str}")
            winner = all_results[0]

        # Auto-open
        if self.settings.auto_open_image and winner:
            try:
                os.startfile(winner["filepath"])
            except:
                pass

    def handle_command(self, cmd: str) -> bool:
        """Handle a command. Returns False if should quit."""
        parts = cmd.strip().split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command in ["/quit", "/exit", "/q"]:
            return False

        elif command == "/help":
            print_help()

        elif command == "/status":
            self.check_connection()

        elif command == "/settings":
            print_settings(self.settings)

        elif command == "/config":
            from settings import edit_settings
            edit_settings()
            # Reload
            self.settings = Settings.load()

        elif command == "/templates":
            templates = self.prompts.list_all()
            if templates:
                print("Available templates:")
                for name in templates:
                    t = self.prompts.get(name)
                    marker = " [active]" if name == self.current_template else ""
                    print(f"  {name}{marker}: {t.system_prompt[:50]}...")
            else:
                print("No templates saved.")

        elif command == "/use":
            if not arg:
                print("Usage: /use <template_name>")
            elif arg in self.prompts.list_all():
                self.current_template = arg
                t = self.prompts.get(arg)
                self.steps = t.steps
                self.cfg = t.cfg
                self.width = t.width
                self.height = t.height
                self.negative = t.negative
                self.checkpoint = t.checkpoint
                print(f"Using template: {arg}")
                print(f"System prompt: {t.system_prompt}")
            else:
                print(f"Template '{arg}' not found. Use /templates to see available.")

        elif command == "/clear":
            self.current_template = None
            print("Template cleared.")

        elif command == "/recent":
            recent = self.gallery.get_recent(10)
            if recent:
                print("Recent generations:")
                for r in recent:
                    print(f"  {r.timestamp[:19]} - {r.prompt[:40]}...")
                    print(f"    {r.image_path}")
            else:
                print("No generations yet.")

        elif command == "/gallery":
            self.open_gallery()

        elif command == "/set":
            self.handle_set(arg)

        elif command == "/review":
            if not arg:
                print("Usage: /review <image_path>")
            else:
                self.review_image(arg)

        else:
            print(f"Unknown command: {command}")
            print("Type /help for commands.")

        return True

    def handle_set(self, arg: str):
        """Handle /set command."""
        parts = arg.split(maxsplit=1)
        if len(parts) < 2:
            print("Usage: /set <option> <value>")
            print("Options: steps, cfg, width, height, size, negative, checkpoint, batch, ollama")
            print(f"\nCurrent session:")
            print(f"  Size: {self.width}x{self.height}")
            print(f"  Steps: {self.steps}, CFG: {self.cfg}")
            print(f"  Batch: {self.batch_count}")
            print(f"  Checkpoint: {self.checkpoint or '(auto)'}")
            print(f"  Ollama review: {self.settings.use_ollama_review}")
            return

        option, value = parts[0].lower(), parts[1]

        try:
            if option == "steps":
                self.steps = int(value)
                print(f"Steps set to {self.steps}")
            elif option == "cfg":
                self.cfg = float(value)
                print(f"CFG set to {self.cfg}")
            elif option == "width":
                self.width = int(value)
                print(f"Width set to {self.width}")
            elif option == "height":
                self.height = int(value)
                print(f"Height set to {self.height}")
            elif option == "size":
                w, h = value.lower().split("x")
                self.width, self.height = int(w), int(h)
                print(f"Size set to {self.width}x{self.height}")
            elif option == "negative":
                self.negative = value
                print(f"Negative prompt set.")
            elif option == "checkpoint":
                self.checkpoint = value
                print(f"Checkpoint set to {self.checkpoint}")
            elif option == "batch":
                self.batch_count = int(value)
                print(f"Batch count set to {self.batch_count}")
            elif option == "ollama":
                self.settings.use_ollama_review = value.lower() in ("true", "1", "yes", "on")
                print(f"Ollama review: {self.settings.use_ollama_review}")
            elif option == "random" or option == "pick_random":
                self.settings.pick_random = value.lower() in ("true", "1", "yes", "on")
                print(f"Pick random: {self.settings.pick_random}")
            else:
                print(f"Unknown option: {option}")
        except Exception as e:
            print(f"Error setting {option}: {e}")

    def review_image(self, path: str):
        """Review a single image with Ollama."""
        if not self.reviewer.is_running():
            print("Ollama not running!")
            return

        path = path.strip('"').strip("'")
        if not Path(path).exists():
            print(f"File not found: {path}")
            return

        print(f"Reviewing with {self.settings.ollama_model}...")
        try:
            result = self.reviewer.describe_image(path)
            print(f"\n{result}")
        except Exception as e:
            print(f"Error: {e}")

    def open_gallery(self):
        """Generate and open gallery HTML."""
        gallery_html = Path(__file__).parent / "gallery" / "index.html"
        recent = self.gallery.get_recent(50)

        html = """<!DOCTYPE html>
<html>
<head>
    <title>SD-RPG Gallery</title>
    <style>
        body { background: #1a1a1a; color: #eee; font-family: sans-serif; padding: 20px; }
        h1 { color: #fff; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .card { background: #2a2a2a; border-radius: 8px; overflow: hidden; }
        .card img { width: 100%; height: auto; cursor: pointer; }
        .card .info { padding: 10px; font-size: 12px; }
        .card .prompt { color: #aaa; margin-bottom: 5px; }
        .card .meta { color: #666; }
    </style>
</head>
<body>
    <h1>SD-RPG Gallery</h1>
    <div class="grid">
"""
        for r in recent:
            img_path = Path(r.image_path).name
            html += f"""
        <div class="card">
            <img src="{img_path}" alt="generated" onclick="window.open('{img_path}')">
            <div class="info">
                <div class="prompt">{r.prompt[:100]}...</div>
                <div class="meta">{r.width}x{r.height} | {r.steps} steps | seed {r.seed}</div>
            </div>
        </div>
"""
        html += """
    </div>
</body>
</html>
"""
        gallery_html.write_text(html)
        print(f"Gallery saved to: {gallery_html}")

        webbrowser.open(f"file://{gallery_html}")

    def run(self):
        """Main loop."""
        print_banner()

        # Check connection
        self.check_connection()

        if self.current_template:
            print(f"\nUsing template: {self.current_template}")

        print_help()

        while True:
            try:
                user_input = input("\n> ").strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        print("Goodbye!")
                        break
                else:
                    # It's a prompt, generate!
                    self.generate(user_input)

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                break


if __name__ == "__main__":
    gen = Generator()
    gen.run()
