"""
DM Narrator System - Keeps the world alive during play.

"Every 3-10 turns, the world pushes back."

This system adds periodic DM interjections that:
- Add texture and atmosphere (75-80%)
- Suggest minor events (15-20%)
- Push the story forward with dramatic events (5-10%)

Unlike the StoryDaemon (background timer), this is turn-synchronized.
It only speaks when players are actively engaged.
"""

import random
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path


@dataclass
class NarratorState:
    """Per-room narrator state."""
    room_id: str
    turn_count: int = 0
    next_interjection_at: int = 0  # Turn number when DM will speak
    last_interjection: Optional[str] = None  # ISO timestamp
    last_interjection_type: Optional[str] = None  # "texture", "minor_event", "story_push"
    consecutive_textures: int = 0  # Track to avoid too many texture-only interjections

    def __post_init__(self):
        if self.next_interjection_at == 0:
            self._roll_next_target()

    def _roll_next_target(self):
        """Roll 3-10 turns until next interjection."""
        self.next_interjection_at = self.turn_count + random.randint(3, 10)

    def increment_turn(self) -> bool:
        """
        Increment turn counter. Returns True if it's time to interject.
        """
        self.turn_count += 1
        return self.turn_count >= self.next_interjection_at

    def mark_interjection(self, interjection_type: str):
        """Record that an interjection happened."""
        self.last_interjection = datetime.now().isoformat()
        self.last_interjection_type = interjection_type

        if interjection_type == "texture":
            self.consecutive_textures += 1
        else:
            self.consecutive_textures = 0

        # Roll next target
        self._roll_next_target()

    def to_dict(self) -> Dict:
        return {
            "room_id": self.room_id,
            "turn_count": self.turn_count,
            "next_interjection_at": self.next_interjection_at,
            "last_interjection": self.last_interjection,
            "last_interjection_type": self.last_interjection_type,
            "consecutive_textures": self.consecutive_textures,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "NarratorState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SeparatedTickState:
    """Per-room state for message-based separated character updates."""
    room_id: str
    turn_count: int = 0
    next_tick_at: int = 0  # Turn number when we update separated characters
    last_tick: Optional[str] = None  # ISO timestamp

    def __post_init__(self):
        if self.next_tick_at == 0:
            self._roll_next_target()

    def _roll_next_target(self):
        """Roll 5-8 turns until next separated character update."""
        self.next_tick_at = self.turn_count + random.randint(5, 8)

    def increment_turn(self) -> bool:
        """Increment turn counter. Returns True if it's time to update separated characters."""
        self.turn_count += 1
        return self.turn_count >= self.next_tick_at

    def mark_tick(self):
        """Record that a tick happened."""
        self.last_tick = datetime.now().isoformat()
        self._roll_next_target()

    def to_dict(self) -> Dict:
        return {
            "room_id": self.room_id,
            "turn_count": self.turn_count,
            "next_tick_at": self.next_tick_at,
            "last_tick": self.last_tick,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SeparatedTickState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SeparatedTickTracker:
    """
    Tracks message-based ticks for separated character updates.

    Every 5-8 messages, generates an update for separated characters.
    Only active when clock-based auto_tick is disabled.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".roundtable"
        self.state_file = self.data_dir / "separated_tick_state.json"
        self.states: Dict[str, SeparatedTickState] = {}
        self._load()

    def _load(self):
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                for room_id, state_data in data.items():
                    self.states[room_id] = SeparatedTickState.from_dict(state_data)
            except Exception as e:
                print(f"[SeparatedTick] Error loading state: {e}")

    def _save(self):
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            data = {k: v.to_dict() for k, v in self.states.items()}
            self.state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[SeparatedTick] Error saving state: {e}")

    def get_state(self, room_id: str) -> SeparatedTickState:
        if room_id not in self.states:
            self.states[room_id] = SeparatedTickState(room_id=room_id)
            self._save()
        return self.states[room_id]

    def record_turn(self, room_id: str) -> bool:
        """Record that a turn happened. Returns True if we should update separated characters."""
        state = self.get_state(room_id)
        should_tick = state.increment_turn()
        self._save()
        return should_tick

    def mark_tick_done(self, room_id: str):
        """Mark that a separated tick was processed."""
        state = self.get_state(room_id)
        state.mark_tick()
        self._save()


# Singleton
_separated_tick_tracker: Optional[SeparatedTickTracker] = None


def get_separated_tick_tracker() -> SeparatedTickTracker:
    global _separated_tick_tracker
    if _separated_tick_tracker is None:
        _separated_tick_tracker = SeparatedTickTracker()
    return _separated_tick_tracker


def init_separated_tick_tracker(data_dir: Path) -> SeparatedTickTracker:
    global _separated_tick_tracker
    _separated_tick_tracker = SeparatedTickTracker(data_dir)
    return _separated_tick_tracker


class DMNarrator:
    """
    The DM Narrator - adds life to the world between player actions.

    Usage:
        narrator = get_dm_narrator()

        # After each complete turn (user + AI response):
        if narrator.should_interject(room_id):
            interjection = await narrator.generate_interjection(room, world_state, recent_messages)
            # Post interjection to room
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".roundtable"
        self.state_file = self.data_dir / "narrator_state.json"
        self.states: Dict[str, NarratorState] = {}
        self._load()

    def _load(self):
        """Load narrator state from disk."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                for room_id, state_data in data.items():
                    self.states[room_id] = NarratorState.from_dict(state_data)
            except Exception as e:
                print(f"[DMNarrator] Error loading state: {e}")

    def _save(self):
        """Save narrator state to disk."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            data = {k: v.to_dict() for k, v in self.states.items()}
            self.state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[DMNarrator] Error saving state: {e}")

    def get_state(self, room_id: str) -> NarratorState:
        """Get or create state for a room."""
        if room_id not in self.states:
            self.states[room_id] = NarratorState(room_id=room_id)
            self._save()
        return self.states[room_id]

    def record_turn(self, room_id: str) -> bool:
        """
        Record that a turn happened. Returns True if DM should interject.

        Call this after each complete turn (user message + AI response).
        """
        state = self.get_state(room_id)
        should_speak = state.increment_turn()
        self._save()
        return should_speak

    def determine_interjection_type(self, room_id: str, threat_level: float = 0) -> str:
        """
        Determine what type of interjection to make.

        Returns: "texture", "minor_event", or "story_push"

        Base probabilities:
        - texture: 75-80%
        - minor_event: 15-20%
        - story_push: 5-10%

        Modified by:
        - Threat level (higher = more events)
        - Consecutive textures (avoid too many in a row)
        """
        state = self.get_state(room_id)

        # Base weights
        texture_weight = 75
        minor_event_weight = 18
        story_push_weight = 7

        # Modify based on threat level (0-10)
        # Higher threat = more events, fewer pure textures
        threat_modifier = threat_level / 10  # 0.0 to 1.0
        texture_weight -= int(threat_modifier * 20)  # Can drop to 55%
        minor_event_weight += int(threat_modifier * 10)  # Can rise to 28%
        story_push_weight += int(threat_modifier * 10)  # Can rise to 17%

        # Avoid too many consecutive textures
        if state.consecutive_textures >= 2:
            # Force something to happen
            texture_weight = max(30, texture_weight - 30)
            minor_event_weight += 20
            story_push_weight += 10

        if state.consecutive_textures >= 4:
            # Really force an event
            texture_weight = 10
            minor_event_weight = 50
            story_push_weight = 40

        # Roll
        roll = random.randint(1, 100)

        if roll <= texture_weight:
            return "texture"
        elif roll <= texture_weight + minor_event_weight:
            return "minor_event"
        else:
            return "story_push"

    def get_interjection_prompt(
        self,
        interjection_type: str,
        scenario: str,
        genre: str,
        recent_conversation: str,
        world_context: str,
        time_of_day: str = "day",
        weather: str = "clear",
        threat_level: float = 0,
        player_name: str = "the player",
        present_characters: list = None,
        player_location: str = "",
    ) -> tuple[str, str]:
        """
        Generate the prompt for the interjection.

        Returns: (system_prompt, user_prompt)
        """
        present_chars = ", ".join(present_characters) if present_characters else "the group"
        location_line = f"\nLOCATION: {player_location}" if player_location else ""

        base_system = f"""You are the Dungeon Master/Narrator for this roleplay scene.
Your role is to add texture and life to the world WITHOUT taking actions for characters.

SETTING: {scenario}
GENRE: {genre}
TIME: {time_of_day}
WEATHER: {weather}{location_line}
PRESENT: {present_chars} and {player_name}

CRITICAL RULES:
- NEVER speak for or control player characters or AI characters
- NEVER resolve actions or make decisions for characters
- You are describing the WORLD, not the characters' reactions
- Keep it SHORT (2-4 sentences max)
- Write in present tense, third person
- Be specific and sensory (sounds, smells, sights, textures)
- If a LOCATION is specified, ground your descriptions in that place"""

        if interjection_type == "texture":
            # Add situation awareness
            tension_note = ""
            if threat_level >= 5:
                tension_note = "\n⚠️ HIGH TENSION: Match the mood. If they're hiding/sneaking, texture should build dread not break immersion."

            user_prompt = f"""The scene is unfolding. Add TEXTURE to the world.

Recent conversation:
{recent_conversation}

{world_context}
{tension_note}

Read the recent conversation and match your texture to the CURRENT MOOD.
If it's tense/dangerous: ominous details, held breath, creaking sounds
If it's calm: peaceful ambiance, mundane life, gentle details

Describe something ambient that adds atmosphere:
- Environmental details (weather shifting, light changing, sounds in distance)
- Background activity that FITS the current scene
- Sensory details (smells, temperatures, textures)
- The passage of time or changing conditions

DO NOT: introduce threats, NPCs with dialogue, or events requiring response.
Just paint the world. Make it feel alive and present.

Your brief description:"""

        elif interjection_type == "minor_event":
            # Add situation awareness based on threat level
            situation_warning = ""
            if threat_level >= 5:
                situation_warning = """
⚠️ HIGH TENSION: Read the recent conversation carefully. If characters are:
- HIDING/sneaking: Don't introduce NPCs that would spot them or blow cover
- In DANGER: Events should relate to the threat, not random occurrences
- In STEALTH: Sounds/events should build tension, not break immersion
Match the dramatic energy of the scene."""

            user_prompt = f"""The scene is unfolding. Add a MINOR EVENT that could spark interaction.

Recent conversation:
{recent_conversation}

{world_context}
{situation_warning}

IMPORTANT: Read the recent conversation! Your event must FIT the current dramatic situation.
If they're hiding from armed men, don't have a random maintenance worker appear.
If there's pursuit, your event should relate to that tension.

Describe something that catches attention:
- Something contextually appropriate to the current scene
- A sound, movement, or change that fits the moment
- A complication or opportunity related to what's happening
- Background activity that makes sense for the situation

The event should be OPTIONAL to engage with - characters can ignore it or investigate.
Don't resolve it - just present it.

NPC INTRODUCTION: If you introduce a person the characters could talk to, end with a tag on its own line:
[NPC: name="Their Name" role="brief role" personality="2-3 trait words"]
Example: [NPC: name="Old Gus" role="fisherman mending nets" personality="gruff, helpful, superstitious"]
Only include this tag if you're introducing someone who could speak. Skip it for background figures.

Your brief description:"""

        else:  # story_push
            threat_desc = ""
            if threat_level >= 5:
                threat_desc = f"\nThreat level is HIGH ({threat_level}/10). Something dangerous is appropriate."

            user_prompt = f"""The scene is unfolding. Time to PUSH THE STORY forward.

Recent conversation:
{recent_conversation}

{world_context}
{threat_desc}

Describe something that DEMANDS attention:
- An NPC arrives with urgent news or need
- A threat becomes immediate (if appropriate to genre/threat level)
- A dramatic reveal or discovery
- A complication that changes the situation

This should feel like an inciting incident - something the characters can't easily ignore.
BUT: Don't resolve it. Present the moment just before action is required.

NPC INTRODUCTION: If you introduce a person the characters could interact with, end with a tag on its own line:
[NPC: name="Their Name" role="brief role" personality="2-3 trait words" want="what they need"]
Example: [NPC: name="Mira" role="frantic mother" personality="desperate, grateful" want="find her lost son"]
Only include this tag if you're introducing someone who could speak. Skip it for threats/monsters.

Your brief, dramatic description:"""

        return base_system, user_prompt

    def mark_complete(self, room_id: str, interjection_type: str):
        """Mark that an interjection was posted."""
        state = self.get_state(room_id)
        state.mark_interjection(interjection_type)
        self._save()

    def reset_room(self, room_id: str):
        """Reset narrator state for a room (e.g., on room clear)."""
        if room_id in self.states:
            del self.states[room_id]
            self._save()

    def get_status(self, room_id: str) -> Dict[str, Any]:
        """Get narrator status for debugging/display."""
        state = self.get_state(room_id)
        return {
            "turn_count": state.turn_count,
            "next_at": state.next_interjection_at,
            "turns_until": max(0, state.next_interjection_at - state.turn_count),
            "last_type": state.last_interjection_type,
            "consecutive_textures": state.consecutive_textures,
        }


# =============================================================================
# Global instance
# =============================================================================

_narrator: Optional[DMNarrator] = None


def get_dm_narrator() -> DMNarrator:
    """Get the global DM narrator instance."""
    global _narrator
    if _narrator is None:
        _narrator = DMNarrator()
    return _narrator


def init_dm_narrator(data_dir: Optional[Path] = None) -> DMNarrator:
    """Initialize the global DM narrator."""
    global _narrator
    _narrator = DMNarrator(data_dir)
    return _narrator
