"""Configuration and settings management for Roundtable."""

from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import json


class ProviderSettings(BaseModel):
    """Settings for an AI provider."""
    enabled: bool = True
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_model: Optional[str] = None


class Settings(BaseSettings):
    """Global application settings."""
    model_config = SettingsConfigDict(
        env_prefix="ROUNDTABLE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Provider API keys (can be set via env vars)
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    comfy_url: str = "http://localhost:8188"  # ComfyUI for image generation
    proxy_url: str = ""  # Optional HTTP proxy for API requests

    # Default models
    default_anthropic_model: str = "claude-sonnet-4-20250514"
    default_openai_model: str = "gpt-4o"
    default_ollama_model: str = "deepseek-r1:7b"  # Used for memory, DM agents, scene prompts

    # App settings
    user_name: str = "User"
    user_physical_description: str = ""  # Optional: what the user looks like (for group photos)
    user_avatar: str = ""  # Path to user's avatar image
    data_dir: Path = Path.home() / ".roundtable"

    # Model for generating image prompts (scene/portrait descriptions)
    image_prompt_model: str = "deepseek-r1:7b"

    # Model for StoryBuilder, Inciting Incidents, and DM
    storybuilder_model: str = "deepseek-r1:7b"

    # Vision model for describing images to non-vision models (llava fallback)
    vision_model: str = "llava"

    # Custom checkpoint override for image generation
    custom_checkpoint: str = ""  # e.g., "flux1-dev.safetensors"
    custom_checkpoint_type: str = "sdxl"  # "sdxl" or "flux"

    # Global system prompt - shared rules/context for all partners
    global_system_prompt: str = """ROLEPLAY MODE. You ARE this character - not an AI playing a character, but the character themselves.

CRITICAL RULES:
- Stay in character 100% of the time
- Never break the fourth wall
- Never ask "what kind of scene do you want" or "what direction should we go"
- Never offer options or ask for clarification about the roleplay itself
- Just BE the character and REACT to what happens
- Use actions, dialogue, and your character's perspective
- If unsure what to do, have your character do something in-character
- NEVER speak for the user, decide their actions, or put words in their mouth
- NEVER invent backstory or details about the user (injuries, history, profession, etc.)
- Only describe YOUR character's actions and perceptions, not theirs
- If a question is addressed to someone else (including the user), don't answer it - let them respond. You can react or wait, but don't hijack their moment.

FORMATTING:
- NEVER prefix your message with your name (e.g., "Name:" or "Character:") - the interface already shows who is speaking
- Use *asterisks* for actions and emoting, not (parentheses)

Begin your response immediately as your character. No preamble."""

    # Favorite prompts library - saved prompts for quick selection during partner creation
    favorite_prompts: List[dict] = []  # [{name: str, prompt: str}, ...]

    # Saved system prompts - saved global system prompts for quick switching
    saved_system_prompts: List[dict] = []  # [{name: str, prompt: str}, ...]

    # Voice settings
    voice_enabled: bool = False  # Global toggle for TTS
    voice_provider: str = "openai"  # "openai" or "elevenlabs"
    elevenlabs_api_key: Optional[str] = None

    # UI settings
    message_bubbles: bool = True  # Show subtle background behind messages for readability

    def get_partners_file(self) -> Path:
        return self.data_dir / "partners.json"

    def get_rooms_file(self) -> Path:
        return self.data_dir / "rooms.json"

    def ensure_data_dir(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()


class Partner(BaseModel):
    """A conversation partner entity."""
    id: str
    name: str
    character_description: str  # Who this character IS (personality, background)
    physical_description: str = ""  # What they look like (for image generation consistency)
    gender: str = ""  # "male", "female", or "" (let AI decide) - for image generation consistency
    provider: str  # "anthropic", "openai", "ollama"
    model: str
    avatar: str = "🤖"  # emoji for display
    avatar_image: Optional[str] = None  # path to generated image
    background_image: Optional[str] = None  # path to background image for room list
    color: str = "#ff69b4"  # highlight color (default: hot pink)

    # Per-partner LoRAs - always applied when generating images for this character
    # [{name: str, weight: float, enabled: bool}, ...]
    loras: List[dict] = []

    # Optional override - if set, replaces global system prompt for this partner
    custom_system_prompt: Optional[str] = None

    # Backwards compat: accept system_prompt as alias
    system_prompt: Optional[str] = None

    # Memory settings
    # "global" = memories persist across all rooms
    # "local" = memories are per-room only
    # "none" = always fresh, no memory accumulation
    memory_mode: str = "local"

    # Hidden traits - known only to the character (and DM)
    # Used for StoryBuilder characters to add dramatic potential
    secret: Optional[str] = None  # Hidden backstory element
    wound: Optional[str] = None   # Past trauma that shaped them
    want: Optional[str] = None    # What they truly seek
    fear: Optional[str] = None    # Their deepest fear
    skill: Optional[str] = None   # Their expertise/talent
    honesty: int = 5              # 1-10, how honest (1=liar, 10=cannot lie)

    # Voice settings
    # OpenAI voices: "alloy", "echo", "fable", "onyx", "nova", "shimmer"
    # Or "elevenlabs:voice_id" for ElevenLabs
    # Or "none" to disable voice for this character
    voice: str = "none"

    # DM-established canon - facts revealed through DM interactions
    # These get injected into the character's context so they "know" their secrets
    dm_canon: List[str] = []

    def get_character(self) -> str:
        """Get character description (handles old system_prompt field)."""
        return self.character_description or self.system_prompt or ""

    def has_image_avatar(self) -> bool:
        """Check if partner has a generated image avatar."""
        if self.avatar_image:
            return Path(self.avatar_image).exists()
        return False

    def get_display_avatar(self) -> str:
        """Get the avatar for display (emoji fallback if no image)."""
        return self.avatar

    def get_effective_system_prompt(self, global_system_prompt: str) -> str:
        """Get the system prompt to use - custom if set, otherwise global."""
        if self.custom_system_prompt and self.custom_system_prompt.strip():
            return self.custom_system_prompt
        return global_system_prompt

    def get_full_context(
        self,
        all_partners: list["Partner"],
        user_name: str,
        global_system_prompt: str,
        include_user: bool = True,
        user_physical_description: str = ""
    ) -> str:
        """Build the full system context for this partner."""
        others = [p for p in all_partners if p.id != self.id]

        # Build detailed descriptions of other characters (including user)
        other_characters = []

        # Add user description if provided
        if include_user and user_physical_description:
            user_info = f"**{user_name}** (human)\nAppearance: {user_physical_description}"
            other_characters.append(user_info)

        for p in others:
            char_info = f"**{p.name}**"
            # Only include physical appearance - characters can't read each other's minds/backstories
            if p.physical_description:
                char_info += f"\nAppearance: {p.physical_description}"
            other_characters.append(char_info)

        # Build participant list (names only, for quick reference)
        participant_names = []
        if include_user:
            participant_names.append(f"{user_name} (human)")
        participant_names.append(f"{self.name} (you)")
        for p in others:
            participant_names.append(p.name)

        effective_prompt = self.get_effective_system_prompt(global_system_prompt)

        # Build the context
        context = f"""{effective_prompt}

---
YOU ARE {self.name.upper()}. Not anyone else - YOU ARE {self.name.upper()}.

YOUR CHARACTER:
{self.get_character()}"""

        if self.physical_description:
            context += f"\n\nYOUR APPEARANCE:\n{self.physical_description}"

        # Add hidden psychological traits (known only to this character)
        hidden_traits = []
        if self.secret:
            hidden_traits.append(f"SECRET: {self.secret}")
        if self.wound:
            hidden_traits.append(f"WOUND (past trauma): {self.wound}")
        if self.want:
            hidden_traits.append(f"WANT (true desire): {self.want}")
        if self.fear:
            hidden_traits.append(f"FEAR: {self.fear}")
        if self.skill:
            hidden_traits.append(f"SKILL/EXPERTISE: {self.skill}")

        if hidden_traits:
            context += "\n\n---\nYOUR PRIVATE PSYCHOLOGY (known only to you - never reveal directly, but these shape your behavior):\n"
            context += "\n".join(hidden_traits)

            # Add honesty guidance
            if self.honesty <= 3:
                context += "\n\nYou are not particularly honest. You lie casually when convenient, deflect questions, and shade the truth."
            elif self.honesty <= 5:
                context += "\n\nYou're honest when it matters, but you'll bend the truth to protect yourself or avoid conflict."
            elif self.honesty >= 8:
                context += "\n\nYou are deeply honest. Lying is difficult for you - you tend to tell the truth even when it's inconvenient."

        # Add DM-established canon facts (things revealed through DM interactions)
        if hasattr(self, 'dm_canon') and self.dm_canon:
            context += "\n\n---\nESTABLISHED FACTS ABOUT YOU (revealed through the story - you know these are true):\n"
            context += "\n".join(f"- {fact}" for fact in self.dm_canon)

        context += f"""

---
OTHER CHARACTERS PRESENT:
{chr(10).join(other_characters) if other_characters else "(None yet)"}

---
PARTICIPANTS: {", ".join(participant_names)}

They may not have all spoken yet, but they are present and can hear everything.
You can address others by name, reference their presence, or speak to the group.
When describing others, stay consistent with their established appearances.
---"""
        return context


class Message(BaseModel):
    """A message in a conversation."""
    id: str
    speaker_id: str  # partner id or "user" or "dm" or "narrator"
    speaker_name: str
    content: str
    room_id: str
    message_type: str = "message"  # "message", "inciting_incident", "dm_public", "dm_private"
    fudged: bool = False  # For DM messages - has the fudge been used?
    mentions: List[str] = []  # Partner IDs mentioned with @name
    image_path: Optional[str] = None  # Attached image (for shared images in transcript)


class Room(BaseModel):
    """A conversation room."""
    id: str
    name: str
    is_common_room: bool = False
    pinned: bool = False  # Pinned rooms appear at top (after common room)
    partner_id: Optional[str] = None  # For 1-on-1 private rooms
    partner_ids: list[str] = []  # For custom rooms with multiple partners
    scenario: str = ""  # Room scenario/context that primes all conversations
    system_prompt: str = ""  # Room-level system prompt override (trumps global and character)
    messages: list[Message] = []
    background_image: Optional[str] = None  # Room background (for custom/common rooms)

    # Genre and world-building
    genre: str = ""  # Emotional texture: "zombie", "dystopia", "fantasy", "noir", "comedy", etc.
    factions: str = ""  # Who controls what in this world (free text description)

    # Genre rules - what's possible/impossible in this world
    # Used by DM to enforce genre consistency
    # Format: {"magic": {"enabled": bool, "note": str}, "supernatural": {...}, ...}
    genre_rules: dict = {
        "magic": {"enabled": True, "note": ""},
        "supernatural": {"enabled": True, "note": ""},
        "technology": {"enabled": True, "note": ""},
        "psychic": {"enabled": True, "note": ""},
        "custom_notes": ""
    }

    # DM mode - whether to use DM and subsystems
    # "full" = DM + all subsystems (fatigue, inventory, combat, etc.)
    # "light" = DM only, minimal systems
    # "none" = collaborative storytelling, no DM
    dm_mode: str = "full"

    # DM secret - hidden twist/truth that only the DM knows
    # Players never see this, but it informs DM decisions and reveals
    dm_secret: str = ""

    # Auto-generate settings (per-room)
    auto_generate: bool = False  # Whether to auto-generate images after responses
    auto_generate_mode: str = "both"  # "scene", "selfie", "both", "group" (for common/custom rooms)
    auto_generate_count: int = 1  # Number of images to generate per exchange

    # Room-specific LoRA settings
    # Each room has its own LoRA configuration (default: none enabled)
    # Format: [{name: str, weight: float, trigger: str, enabled: bool}, ...]
    loras: list[dict] = []

    # Ambient mode settings (common room only)
    # When enabled, randomly pulls characters into conversation
    ambient_mode: bool = False
    ambient_interval_min: int = 5  # Minimum minutes between ambient pulls
    ambient_interval_max: int = 15  # Maximum minutes between ambient pulls
    ambient_providers: list[str] = ["ollama"]  # Which providers to use (ollama, anthropic, openai)

    def get_partners_in_room(self, all_partners: list) -> list:
        """Get the partners that belong to this room."""
        if self.is_common_room:
            return all_partners
        elif self.partner_id:
            # Private 1-on-1 room
            return [p for p in all_partners if p.id == self.partner_id]
        elif self.partner_ids:
            # Custom room with selected partners
            return [p for p in all_partners if p.id in self.partner_ids]
        return []


class DataStore:
    """Handles persistence of partners and rooms."""

    def __init__(self, settings: Settings):
        self.settings = settings
        settings.ensure_data_dir()
        self._partners: dict[str, Partner] = {}
        self._rooms: dict[str, Room] = {}
        self._load()

    def _load(self):
        """Load data from disk."""
        # Load partners
        partners_file = self.settings.get_partners_file()
        if partners_file.exists():
            data = json.loads(partners_file.read_text())
            for p in data:
                partner = Partner(**p)
                self._partners[partner.id] = partner

        # Load rooms
        rooms_file = self.settings.get_rooms_file()
        if rooms_file.exists():
            data = json.loads(rooms_file.read_text())
            for r in data:
                room = Room(**r)
                self._rooms[room.id] = room

        # Ensure common room exists
        if "common" not in self._rooms:
            self._rooms["common"] = Room(
                id="common",
                name="Common Room",
                is_common_room=True
            )

        # Migrate: Remove "Private: " prefix from room names
        needs_save = False
        for room in self._rooms.values():
            if room.name.startswith("Private: "):
                room.name = room.name[9:]  # Remove "Private: " prefix
                needs_save = True
        if needs_save:
            self.save()

    def save(self):
        """Save data to disk."""
        partners_file = self.settings.get_partners_file()
        partners_file.write_text(json.dumps(
            [p.model_dump() for p in self._partners.values()],
            indent=2
        ))

        rooms_file = self.settings.get_rooms_file()
        rooms_file.write_text(json.dumps(
            [r.model_dump() for r in self._rooms.values()],
            indent=2
        ))

    def get_partners(self) -> list[Partner]:
        return list(self._partners.values())

    def get_partner(self, partner_id: str) -> Optional[Partner]:
        return self._partners.get(partner_id)

    def add_partner(self, partner: Partner) -> Partner:
        self._partners[partner.id] = partner
        # Create private room for this partner
        room = Room(
            id=f"private_{partner.id}",
            name=partner.name,
            partner_id=partner.id
        )
        self._rooms[room.id] = room
        self.save()
        return partner

    def update_partner(self, partner: Partner):
        self._partners[partner.id] = partner
        # Update room name if it exists
        room_id = f"private_{partner.id}"
        if room_id in self._rooms:
            self._rooms[room_id].name = partner.name
        # Update speaker_name in all messages from this partner
        for room in self._rooms.values():
            for message in room.messages:
                if message.speaker_id == partner.id:
                    message.speaker_name = partner.name
        self.save()

    def delete_partner(self, partner_id: str):
        if partner_id in self._partners:
            del self._partners[partner_id]
        room_id = f"private_{partner_id}"
        if room_id in self._rooms:
            del self._rooms[room_id]
        self.save()

    def get_rooms(self) -> list[Room]:
        return list(self._rooms.values())

    def get_room(self, room_id: str) -> Optional[Room]:
        return self._rooms.get(room_id)

    def add_message(self, room_id: str, message: Message):
        if room_id in self._rooms:
            self._rooms[room_id].messages.append(message)
            self.save()

    def clear_room(self, room_id: str):
        if room_id in self._rooms:
            self._rooms[room_id].messages = []
            self.save()

    def create_custom_room(
        self,
        name: str,
        partner_ids: list[str],
        scenario: str = "",
        genre: str = "",
        factions: str = "",
        genre_rules: dict = None,
        dm_mode: str = "full",
        dm_secret: str = ""
    ) -> Room:
        """Create a custom room with selected partners."""
        import uuid
        room_id = f"custom_{str(uuid.uuid4())[:8]}"
        room = Room(
            id=room_id,
            name=name,
            partner_ids=partner_ids,
            scenario=scenario,
            genre=genre,
            factions=factions,
            dm_mode=dm_mode,
            dm_secret=dm_secret
        )
        if genre_rules:
            room.genre_rules = genre_rules
        self._rooms[room_id] = room
        self.save()
        return room

    def delete_room(self, room_id: str):
        """Delete a room (not common room or private rooms)."""
        if room_id in self._rooms:
            room = self._rooms[room_id]
            # Don't delete common room or auto-created private rooms
            if not room.is_common_room and not room_id.startswith("private_"):
                del self._rooms[room_id]
                self.save()
