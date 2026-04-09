"""
Image generation for Roundtable using ComfyUI.
Borrowed from sd-rpg setup.
"""

import sys
import random
import threading
from pathlib import Path
from typing import Optional

# Lock to serialize ComfyUI requests - it doesn't handle concurrent calls well
_comfy_lock = threading.Lock()

# Add sd-rpg to path so we can import from it (now bundled with Roundtable)
# Handle both normal running and frozen exe
if getattr(sys, 'frozen', False):
    # Running as PyInstaller exe - sd-rpg is in the temp extraction folder
    import os
    SD_RPG_PATH = Path(sys._MEIPASS) / "sd-rpg"
else:
    # Running from source
    SD_RPG_PATH = Path(__file__).parent / "sd-rpg"

if SD_RPG_PATH.exists() and str(SD_RPG_PATH) not in sys.path:
    sys.path.insert(0, str(SD_RPG_PATH))

# Try to import ComfyUI client - optional dependency
try:
    from comfy_client import ComfyClient
    from settings import Settings as SDSettings
    COMFY_AVAILABLE = True
except ImportError as e:
    COMFY_AVAILABLE = False
    ComfyClient = None
    SDSettings = None
    print("""
====================================================================
  IMAGE GENERATION UNAVAILABLE

  The 'sd-rpg' folder is missing. Image generation won't work.

  To enable image generation:
  1. Make sure the 'sd-rpg' folder is in your Roundtable folder
  2. Make sure ComfyUI is installed and running

  Chat features will still work normally without images.
====================================================================
""")

# Default negative prompt for Illustrious models
DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, extra digits, multiple views, "
    "text, error, worst quality, jpeg artifacts, low quality, "
    "watermark, unfinished, displeasing, signature, username, scan, blurry"
)

# Model presets with optimal settings
MODEL_PRESETS = {
    'illustrious': {
        'name': 'Illustrious (Anime)',
        'checkpoint': None,  # Use default from sd-rpg settings
        'steps': 28,
        'cfg': 5.0,
        'sampler': 'dpmpp_2m_sde',
        'scheduler': 'karras',
        'clip_skip': 2,
        'negative': DEFAULT_NEGATIVE,
        'width': 896,
        'height': 1152,
    },
    'flux': {
        'name': 'Flux Dev (Realistic)',
        'checkpoint': 'flux1-dev-fp8.safetensors',  # Flux FP8 checkpoint
        'steps': 24,
        'cfg': 1.0,
        'sampler': 'euler',
        'scheduler': 'simple',
        'clip_skip': 1,
        'negative': '',  # Flux doesn't use negative prompts effectively
        'width': 512,
        'height': 768,
    },
    'pony': {
        'name': 'Pony Diffusion V6 XL',
        'checkpoint': 'ponyDiffusionV6XL_v6StartWithThisOne.safetensors',
        'steps': 25,
        'cfg': 7.0,
        'sampler': 'euler_ancestral',
        'scheduler': 'normal',
        'clip_skip': 2,
        'negative': '',  # Use score tags in prompt instead (score_9, score_8_up, etc.)
        'width': 768,
        'height': 1024,
    },
}

# Legacy reference
ILLUSTRIOUS_SETTINGS = MODEL_PRESETS['illustrious']


class ImageGenerator:
    """Generates images via ComfyUI for Roundtable."""

    def __init__(self, model_preset: str = 'illustrious'):
        self.sd_settings = SDSettings.load()
        self.client = ComfyClient(
            host=self.sd_settings.comfy_host,
            port=self.sd_settings.comfy_port
        )
        self.base_dir = Path.home() / ".roundtable"
        self.avatars_dir = self.base_dir / "avatars"
        self.scenes_dir = self.base_dir / "scenes"
        self.avatars_dir.mkdir(parents=True, exist_ok=True)
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        # User overrides (empty/0 = use preset default)
        self.sampler_override = ''
        self.scheduler_override = ''
        self.steps_override = 0
        self.cfg_override = 0.0
        self.width_override = 0
        self.height_override = 0
        self.negative_prompt = ''  # User's custom negative prompt (empty = use preset default)

        # LoRA settings - list of {name, weight, enabled, trigger}
        self.loras = []

        # Hi-res upscaler settings
        self.hires_enabled = False
        self.hires_upscaler = ''  # Upscaler model name (e.g., "4x-UltraSharp")
        self.hires_scale = 2.0
        self.hires_denoise = 0.4

        # Custom checkpoint override (takes precedence over preset)
        self.custom_checkpoint = ''  # e.g., "flux1-dev.safetensors"
        self.custom_checkpoint_type = 'sdxl'  # 'sdxl' or 'flux'

        # Load persisted settings (preset + overrides + LoRAs) from JSON
        self._load_persisted_settings()

        # If no persisted preset, use the default
        if not hasattr(self, 'model_preset') or not self.model_preset:
            self.model_preset = model_preset
            self._load_preset(model_preset)

    def _load_persisted_settings(self):
        """Load image generation settings from persisted JSON."""
        import json
        settings_file = self.base_dir / "settings.json"
        if settings_file.exists():
            try:
                data = json.loads(settings_file.read_text())

                # Load model preset
                preset_name = data.get('model_preset', 'illustrious')
                if preset_name in MODEL_PRESETS:
                    self.model_preset = preset_name
                    self._load_preset(preset_name)
                    print(f"[image_gen] Loaded preset: {preset_name}")

                # Load overrides
                self.sampler_override = data.get('sampler_override', '')
                self.scheduler_override = data.get('scheduler_override', '')
                self.steps_override = int(data.get('steps_override', 0))
                self.cfg_override = float(data.get('cfg_override', 0))
                self.width_override = int(data.get('width_override', 0))
                self.height_override = int(data.get('height_override', 0))
                self.negative_prompt = data.get('negative_prompt', '')

                # LoRAs are now room-specific, not loaded from global settings
                # self.loras stays empty - captured_loras from room is passed at generation time

                # Load hi-res upscaler settings
                self.hires_enabled = data.get('hires_enabled', False)
                self.hires_upscaler = data.get('hires_upscaler', '')
                self.hires_scale = float(data.get('hires_scale', 2.0))
                self.hires_denoise = float(data.get('hires_denoise', 0.4))

                # Log what we loaded
                overrides = []
                if self.sampler_override: overrides.append(f"sampler={self.sampler_override}")
                if self.scheduler_override: overrides.append(f"scheduler={self.scheduler_override}")
                if self.steps_override: overrides.append(f"steps={self.steps_override}")
                if self.cfg_override: overrides.append(f"cfg={self.cfg_override}")
                if self.width_override: overrides.append(f"width={self.width_override}")
                if self.height_override: overrides.append(f"height={self.height_override}")

                if overrides:
                    print(f"[image_gen] Loaded overrides: {', '.join(overrides)}")

                # Log LoRAs
                enabled_loras = [l for l in self.loras if l.get('enabled', True)]
                if enabled_loras:
                    lora_info = [f"{l['name']}@{l.get('weight', 1.0)}" for l in enabled_loras]
                    print(f"[image_gen] Loaded LoRAs: {', '.join(lora_info)}")

            except Exception as e:
                print(f"[image_gen] Failed to load settings: {e}")

    def _load_preset(self, preset_name: str):
        """Load settings from a model preset."""
        preset = MODEL_PRESETS.get(preset_name, MODEL_PRESETS['illustrious'])
        self.checkpoint = preset['checkpoint']  # None means use sd-rpg default
        self._preset_steps = preset['steps']
        self._preset_cfg = preset['cfg']
        self._preset_sampler = preset['sampler']
        self._preset_scheduler = preset['scheduler']
        self.clip_skip = preset['clip_skip']
        self.negative = preset['negative']
        self._preset_width = preset['width']
        self._preset_height = preset['height']

    @property
    def steps(self):
        return self.steps_override if self.steps_override > 0 else self._preset_steps

    @property
    def cfg(self):
        return self.cfg_override if self.cfg_override > 0 else self._preset_cfg

    @property
    def sampler(self):
        return self.sampler_override if self.sampler_override else self._preset_sampler

    @property
    def scheduler(self):
        return self.scheduler_override if self.scheduler_override else self._preset_scheduler

    @property
    def default_width(self):
        return self.width_override if self.width_override > 0 else self._preset_width

    @property
    def default_height(self):
        return self.height_override if self.height_override > 0 else self._preset_height

    @property
    def effective_negative(self):
        """Return user's custom negative prompt if set, otherwise preset default."""
        return self.negative_prompt if self.negative_prompt else self.negative

    def _get_checkpoint(self) -> str:
        """Get the checkpoint to use (custom > preset > default)."""
        # Custom checkpoint takes highest priority
        if self.custom_checkpoint:
            return self.custom_checkpoint
        return self.checkpoint or self.sd_settings.default_checkpoint

    def is_flux_model(self) -> bool:
        """Check if current model is Flux (affects workflow)."""
        # If custom checkpoint is set, use its type
        if self.custom_checkpoint:
            return self.custom_checkpoint_type == 'flux'
        # Otherwise, check if preset is flux
        return self.model_preset == 'flux'

    def set_model_preset(self, preset_name: str):
        """Switch to a different model preset."""
        if preset_name in MODEL_PRESETS:
            self.model_preset = preset_name
            self._load_preset(preset_name)
            return True
        return False

    def get_available_presets(self) -> dict:
        """Return available model presets."""
        return {k: v['name'] for k, v in MODEL_PRESETS.items()}

    def is_available(self) -> bool:
        """Check if ComfyUI is running."""
        result = self.client.is_running()
        print(f"[image_gen] is_available: {result}")
        return result

    def get_partner_dir(self, partner_id: str) -> Path:
        """Get or create the dedicated folder for a partner's images."""
        partner_dir = self.avatars_dir / partner_id
        partner_dir.mkdir(parents=True, exist_ok=True)
        return partner_dir

    def generate_avatar(
        self,
        prompt: str,
        partner_id: str,
        count: int = 1,
        width: int = None,
        height: int = None,
        partner_loras: list = None,
        partner_name: str = None,
        model_name: str = None,
        system_prompt_prefix: str = None,
        room_id: str = None,
        captured_loras: list = None
    ) -> list[Path]:
        """
        Generate avatar options for a partner.

        Saves to partner's dedicated folder: ~/.roundtable/avatars/{partner_id}/
        Returns list of saved image paths.
        Uses model preset's default resolution if not specified.

        Args:
            partner_loras: Optional list of partner-specific LoRAs to merge with room/global LoRAs
            partner_name: Character name for filename
            model_name: AI model used for description (e.g., 'haiku', 'sonnet')
            system_prompt_prefix: First 15 chars of system prompt for filename
            room_id: Room where image was generated (for room-scoped galleries)
            captured_loras: Room LoRAs captured at queue time (overrides self.loras if provided)
        """
        if not self.is_available():
            raise RuntimeError("ComfyUI not running")

        # Use preset defaults if not specified
        width = width or self.default_width
        height = height or self.default_height

        # Use captured_loras (room-specific, captured at queue time) if provided,
        # otherwise fall back to global self.loras
        base_loras = captured_loras if captured_loras is not None else self.loras
        combined_loras = list(base_loras)  # Copy
        if partner_loras:
            combined_loras.extend(partner_loras)

        # Collect trigger words from enabled LoRAs and add to prompt
        trigger_words = []
        for lora in combined_loras:
            if lora.get('enabled', True) and lora.get('trigger'):
                trigger_words.append(lora['trigger'])
        if trigger_words:
            prompt = f"{' '.join(trigger_words)}, {prompt}"

        partner_dir = self.get_partner_dir(partner_id)
        results = []

        for i in range(count):
            seed = random.randint(0, 2**32 - 1)

            # Serialize ComfyUI requests - it doesn't handle concurrent calls
            print(f"[image_gen] Waiting for lock (request {i+1}/{count})...")
            print(f"[image_gen] Using preset: {self.model_preset} (steps={self.steps}, cfg={self.cfg}, sampler={self.sampler}, {width}x{height})")
            with _comfy_lock:
                print(f"[image_gen] Got lock, generating image {i+1}/{count} with seed {seed}")
                images = self.client.generate_image(
                    prompt=prompt,
                    negative=self.effective_negative,
                    width=width,
                    height=height,
                    steps=self.steps,
                    cfg=self.cfg,
                    seed=seed,
                    checkpoint=self._get_checkpoint(),
                    sampler=self.sampler,
                    scheduler=self.scheduler,
                    clip_skip=self.clip_skip,
                    loras=combined_loras
                )

                for j, img_data in enumerate(images):
                    # Save with timestamp for uniqueness and easy sorting
                    from datetime import datetime
                    import re
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    # Build filename with generation metadata
                    # Sanitize names for filesystem
                    def sanitize(s, max_len=20):
                        return re.sub(r'[^\w\-]', '', s.replace(' ', '_'))[:max_len] if s else None

                    name_part = sanitize(partner_name) or 'selfie'
                    model_part = f"_{sanitize(model_name)}" if model_name else ''
                    prompt_part = f"_{sanitize(system_prompt_prefix, 15)}" if system_prompt_prefix else ''
                    room_part = f"_room_{sanitize(room_id, 30)}" if room_id else ''

                    # Add generation settings: preset, sampler, steps, cfg
                    preset_part = f"_{self.model_preset}" if self.model_preset else ''
                    sampler_short = (self.sampler or 'euler')[:6]  # Truncate sampler name
                    settings_part = f"_{sampler_short}_{self.steps}s_{self.cfg}cfg"

                    # Add LoRA names (if any enabled)
                    lora_names = [sanitize(l.get('name', ''), 30) for l in combined_loras if l.get('enabled', True) and l.get('name')]
                    lora_part = f"_lora-{'-'.join(lora_names)}" if lora_names else ''

                    filename = f"{name_part}{model_part}{prompt_part}{preset_part}{settings_part}{lora_part}{room_part}_{timestamp}_{seed}.png"
                    filepath = partner_dir / filename
                    filepath.write_bytes(img_data)
                    results.append(filepath)
                    print(f"[image_gen] Saved selfie to: {filepath}")

                    # Save prompt alongside image for gallery tooltips
                    prompt_file = partner_dir / f"{name_part}{model_part}{prompt_part}{preset_part}{settings_part}{lora_part}{room_part}_{timestamp}_{seed}.txt"
                    prompt_file.write_text(prompt, encoding='utf-8')
                    print(f"[image_gen] Saved prompt to: {prompt_file}")
                print(f"[image_gen] Released lock after image {i+1}/{count}")

        return results

    def generate_scene(
        self,
        prompt: str,
        room_id: str,
        width: int = None,
        height: int = None,
        captured_loras: list = None
    ) -> Path:
        """
        Generate a scene image for the conversation.

        Returns path to saved image.
        Uses model preset's default resolution (landscape) if not specified.

        Args:
            captured_loras: Room LoRAs captured at queue time (overrides self.loras if provided)
        """
        if not self.is_available():
            raise RuntimeError("ComfyUI not running")

        # Use preset defaults, swapped for landscape
        width = width or self.default_height  # Swap for landscape
        height = height or self.default_width

        # Use captured_loras (room-specific) if provided, otherwise fall back to global
        effective_loras = captured_loras if captured_loras is not None else self.loras

        # Collect trigger words from enabled LoRAs and add to prompt
        trigger_words = []
        for lora in effective_loras:
            if lora.get('enabled', True) and lora.get('trigger'):
                trigger_words.append(lora['trigger'])
        if trigger_words:
            prompt = f"{' '.join(trigger_words)}, {prompt}"

        seed = random.randint(0, 2**32 - 1)

        # Serialize ComfyUI requests - it doesn't handle concurrent calls
        print(f"[image_gen] Scene: Waiting for lock...")
        print(f"[image_gen] Using preset: {self.model_preset} (steps={self.steps}, cfg={self.cfg}, sampler={self.sampler}, {width}x{height})")
        with _comfy_lock:
            print(f"[image_gen] Scene: Got lock, generating with seed {seed}")
            images = self.client.generate_image(
                prompt=prompt,
                negative=self.effective_negative,
                width=width,
                height=height,
                steps=self.steps,
                cfg=self.cfg,
                seed=seed,
                checkpoint=self._get_checkpoint(),
                sampler=self.sampler,
                scheduler=self.scheduler,
                clip_skip=self.clip_skip,
                loras=effective_loras
            )
            print(f"[image_gen] Scene: Released lock")

        if images:
            filename = f"scene_{room_id}_{seed}.png"
            filepath = self.scenes_dir / filename
            filepath.write_bytes(images[0])

            # Save prompt alongside image for gallery tooltips
            prompt_file = self.scenes_dir / f"scene_{room_id}_{seed}.txt"
            prompt_file.write_text(prompt, encoding='utf-8')
            print(f"[image_gen] Saved scene prompt to: {prompt_file}")

            return filepath

        raise RuntimeError("No image generated")

    def set_avatar(self, partner_id: str, source_path: Path) -> Path:
        """
        Set a partner's avatar from a generated option.

        Copies to partner's folder as avatar.png.
        """
        partner_dir = self.get_partner_dir(partner_id)
        final_path = partner_dir / "avatar.png"
        final_path.write_bytes(source_path.read_bytes())
        return final_path

    def get_avatar_path(self, partner_id: str) -> Optional[Path]:
        """Get the avatar path for a partner if it exists."""
        path = self.get_partner_dir(partner_id) / "avatar.png"
        return path if path.exists() else None

    def get_all_images(self, partner_id: str) -> list[Path]:
        """Get all images for a partner from their folder."""
        partner_dir = self.get_partner_dir(partner_id)
        return sorted(partner_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)

    def cleanup_options(self, partner_id: str):
        """Remove temporary avatar option files (legacy - kept for compatibility)."""
        # Old format cleanup
        for f in self.avatars_dir.glob(f"{partner_id}_option_*.png"):
            f.unlink()


# Singleton for easy access
_generator: Optional[ImageGenerator] = None


def get_generator() -> Optional[ImageGenerator]:
    """Get or create the image generator instance. Returns None if ComfyUI not available."""
    global _generator
    if not COMFY_AVAILABLE:
        print("[image_gen] get_generator: COMFY_AVAILABLE is False, returning None")
        return None
    if _generator is None:
        print("[image_gen] get_generator: Creating new ImageGenerator instance")
        _generator = ImageGenerator()
    return _generator


async def generate_scene_prompt(messages: list[dict], model: str = "llama3.2") -> str:
    """
    Use Ollama to generate an SD prompt from recent conversation.

    Args:
        messages: List of recent messages with 'speaker' and 'content' keys
        model: Ollama model to use

    Returns:
        A stable diffusion prompt describing the scene
    """
    import httpx

    # Format the conversation
    conversation = "\n".join(
        f"{m['speaker']}: {m['content']}"
        for m in messages[-5:]  # Last 5 messages
    )

    system = """You are a prompt writer for Stable Diffusion image generation.
Given a conversation, write a vivid, descriptive prompt that captures the current scene or moment.
Focus on visual elements: setting, mood, lighting, atmosphere.
Output ONLY the prompt, no explanation. Keep it under 100 words.
Use comma-separated descriptive phrases. Include art style hints like "cinematic lighting" or "dramatic atmosphere"."""

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": f"Conversation:\n{conversation}\n\nWrite a Stable Diffusion prompt for this scene:",
                "system": system,
                "stream": False
            }
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("response", "").strip()

    return "atmospheric scene, cinematic lighting, dramatic mood"


async def ask_character_self_description(
    character_description: str,
    character_name: str,
    context: str = "",
    model: str = "llama3.2"
) -> str:
    """
    Ask a character to describe their own appearance for portrait generation.

    The character stays in-character and describes themselves vividly,
    which then becomes the SD prompt.

    Args:
        character_description: The character's system prompt / who they are
        character_name: The character's name
        context: Optional recent conversation context
        model: Ollama model to use

    Returns:
        A self-description suitable for SD image generation
    """
    import httpx

    system = f"""{character_description}

You are {character_name}. Stay completely in character.

When asked to describe your appearance, give a vivid, visual description of yourself as you appear RIGHT NOW.
Describe: your face, expression, clothing, posture, the lighting on you, your surroundings.
Be specific and painterly. This description will be used to create a portrait of you.
Keep it under 100 words. Use comma-separated descriptive phrases.
Include art style hints naturally (dramatic lighting, oil painting style, etc).
Do NOT break character. Do NOT explain what you're doing. Just describe yourself."""

    prompt = "Describe your appearance right now, as if for a portrait artist who will paint you in this moment."
    if context:
        prompt = f"The conversation so far:\n{context}\n\n{prompt}"

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": system,
                "stream": False
            }
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("response", "").strip()

    return f"portrait of {character_name}, detailed face, artistic style"


async def generate_scene_from_conversation(
    messages: list[dict],
    room_context: str = "",
    model: str = "llama3.2"
) -> str:
    """
    Generate an SD prompt for the current scene/moment in a conversation.

    Args:
        messages: Recent messages with 'speaker' and 'content'
        room_context: Description of the room/setting
        model: Ollama model to use

    Returns:
        An SD prompt for the scene
    """
    import httpx

    conversation = "\n".join(
        f"{m['speaker']}: {m['content']}"
        for m in messages[-7:]
    )

    system = """You are a cinematographer and prompt writer for Stable Diffusion.
Given a conversation, visualize the SCENE - not portraits of speakers, but the moment itself.
Think: where are they? What's the atmosphere? What would a camera capture right now?

Output ONLY the image prompt. Under 100 words.
Use comma-separated phrases.
Include: setting, mood, lighting, composition, art style.
Be specific and cinematic."""

    prompt = f"""Setting context: {room_context if room_context else "A roundtable discussion space"}

Recent conversation:
{conversation}

Describe this scene for Stable Diffusion:"""

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": system,
                "stream": False
            }
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("response", "").strip()

    return "roundtable discussion, dramatic lighting, cinematic composition, atmospheric"
