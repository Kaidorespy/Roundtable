"""
Consequence Engine - Ripple effects from actions in the world.

"Every action has consequences."

This system tracks and propagates the effects of significant actions:
- A gunshot attracts nearby zombies
- An explosion is heard for miles
- A fire spreads if not contained
- A scream alerts nearby enemies

The Story Daemon calls this engine when events occur,
and the engine calculates what should happen as a result.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import json
import math
import debug_logger as dbg


class ConsequenceType(Enum):
    """Types of consequences that can ripple through the world."""
    SOUND = "sound"           # Noise that attracts/alerts
    VISUAL = "visual"         # Visible events (fire, explosion flash)
    SMELL = "smell"           # Smoke, blood, etc.
    REPUTATION = "reputation" # Word spreads about your actions
    RESOURCE = "resource"     # Resource depletion/discovery
    THREAT = "threat"         # Danger increase/decrease


class ConsequenceSeverity(Enum):
    """How significant is this consequence?"""
    MINOR = "minor"       # Barely noticeable
    MODERATE = "moderate" # Noteworthy
    MAJOR = "major"       # Significant
    CRITICAL = "critical" # World-changing


# Sound propagation - how far different sounds travel (in map units)
SOUND_RANGES = {
    "whisper": 1,
    "conversation": 3,
    "shout": 10,
    "gunshot": 50,
    "explosion": 200,
    "car_engine": 30,
    "car_horn": 80,
    "scream": 20,
    "breaking_glass": 15,
    "footsteps": 2,
    "running": 5,
}


@dataclass
class Consequence:
    """A single consequence of an action."""
    id: str
    consequence_type: ConsequenceType
    severity: ConsequenceSeverity
    description: str

    # Where did this originate?
    source_location_id: Optional[str] = None
    source_character_id: Optional[str] = None

    # How far does this consequence reach? (in map units)
    radius: float = 0.0

    # When does this resolve?
    # Some consequences are instant, others unfold over time
    delay_hours: float = 0.0  # Hours until this triggers
    duration_hours: float = 0.0  # How long the effect lasts

    # What does this consequence produce?
    effects: List[str] = field(default_factory=list)
    # e.g., ["zombies_attracted:15", "threat_increase:2", "npc_alerted:guard_1"]

    # Tracking
    created_at: str = ""
    triggered: bool = False
    triggered_at: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "consequence_type": self.consequence_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "source_location_id": self.source_location_id,
            "source_character_id": self.source_character_id,
            "radius": self.radius,
            "delay_hours": self.delay_hours,
            "duration_hours": self.duration_hours,
            "effects": self.effects,
            "created_at": self.created_at,
            "triggered": self.triggered,
            "triggered_at": self.triggered_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Consequence":
        data = data.copy()
        data["consequence_type"] = ConsequenceType(data["consequence_type"])
        data["severity"] = ConsequenceSeverity(data["severity"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PendingConsequence:
    """A consequence waiting to be applied to the world."""
    consequence: Consequence
    world_id: str
    game_hour_triggers_at: float  # game_day * 24 + game_hour


class ConsequenceEngine:
    """
    Calculates and tracks ripple effects from actions.

    Usage:
        engine = ConsequenceEngine()

        # When a gunshot happens
        consequences = engine.calculate_sound_consequence(
            world_id="room_123",
            sound_type="gunshot",
            source_location="warehouse_district",
            source_character="player_1"
        )

        # These consequences get queued in the Story Daemon
        # and applied over time
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".roundtable"
        self.consequences_file = self.data_dir / "consequences.json"
        self.pending: Dict[str, List[PendingConsequence]] = {}  # world_id -> pending
        self._load()

    def _load(self):
        """Load pending consequences from disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.consequences_file.exists():
            try:
                data = json.loads(self.consequences_file.read_text())
                for world_id, pending_list in data.items():
                    self.pending[world_id] = []
                    for p in pending_list:
                        self.pending[world_id].append(PendingConsequence(
                            consequence=Consequence.from_dict(p["consequence"]),
                            world_id=p["world_id"],
                            game_hour_triggers_at=p["game_hour_triggers_at"]
                        ))
            except Exception as e:
                print(f"[ConsequenceEngine] Error loading: {e}")

    def _save(self):
        """Save pending consequences to disk."""
        try:
            data = {}
            for world_id, pending_list in self.pending.items():
                data[world_id] = [{
                    "consequence": p.consequence.to_dict(),
                    "world_id": p.world_id,
                    "game_hour_triggers_at": p.game_hour_triggers_at
                } for p in pending_list]
            self.consequences_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[ConsequenceEngine] Error saving: {e}")

    def calculate_sound_consequence(
        self,
        world_id: str,
        sound_type: str,
        source_location_id: Optional[str] = None,
        source_character_id: Optional[str] = None,
        custom_range: Optional[float] = None,
        genre: str = "fantasy"
    ) -> List[Consequence]:
        """
        Calculate consequences of a sound event.

        Returns a list of consequences that should be applied.
        """
        import uuid

        radius = custom_range or SOUND_RANGES.get(sound_type.lower(), 10)
        consequences = []

        # Determine severity based on sound type
        if sound_type in ["whisper", "footsteps"]:
            severity = ConsequenceSeverity.MINOR
        elif sound_type in ["conversation", "running", "breaking_glass"]:
            severity = ConsequenceSeverity.MODERATE
        elif sound_type in ["shout", "scream", "car_engine"]:
            severity = ConsequenceSeverity.MAJOR
        else:  # gunshot, explosion, car_horn
            severity = ConsequenceSeverity.CRITICAL

        # Build effects based on genre and sound type
        effects = []

        # Zombie world: sounds attract zombies
        if "zombie" in genre.lower():
            if sound_type == "gunshot":
                # Rough estimate: 1 zombie per 10 units of radius, random variance
                estimated_zombies = max(1, int(radius / 10))
                effects.append(f"zombies_attracted:{estimated_zombies}")
                effects.append(f"threat_increase:2")
            elif sound_type == "explosion":
                estimated_zombies = max(5, int(radius / 5))
                effects.append(f"zombies_attracted:{estimated_zombies}")
                effects.append(f"threat_increase:4")
            elif sound_type in ["scream", "car_horn"]:
                estimated_zombies = max(1, int(radius / 15))
                effects.append(f"zombies_attracted:{estimated_zombies}")
                effects.append(f"threat_increase:1")

        # Fantasy/general: sounds alert NPCs
        else:
            if severity in [ConsequenceSeverity.MAJOR, ConsequenceSeverity.CRITICAL]:
                effects.append(f"npcs_alerted_in_radius:{radius}")
                if sound_type == "explosion":
                    effects.append("threat_increase:3")
                elif sound_type == "gunshot":
                    effects.append("threat_increase:2")

        # Create the consequence
        consequence = Consequence(
            id=str(uuid.uuid4())[:8],
            consequence_type=ConsequenceType.SOUND,
            severity=severity,
            description=f"A {sound_type} echoes through the area",
            source_location_id=source_location_id,
            source_character_id=source_character_id,
            radius=radius,
            delay_hours=0.0,  # Sound is instant
            duration_hours=0.0,
            effects=effects,
            created_at=datetime.now().isoformat(),
        )
        consequences.append(consequence)

        return consequences

    def calculate_visual_consequence(
        self,
        world_id: str,
        visual_type: str,
        source_location_id: Optional[str] = None,
        source_character_id: Optional[str] = None,
    ) -> List[Consequence]:
        """Calculate consequences of a visual event (fire, explosion flash, etc.)."""
        import uuid

        consequences = []

        # Visual ranges (how far can this be seen?)
        visual_ranges = {
            "fire_small": 20,
            "fire_large": 100,
            "explosion_flash": 200,
            "smoke_column": 500,
            "signal_flare": 300,
            "flashlight": 15,
            "torch": 10,
        }

        radius = visual_ranges.get(visual_type.lower(), 20)
        severity = ConsequenceSeverity.MAJOR if "fire" in visual_type or "explosion" in visual_type else ConsequenceSeverity.MODERATE

        effects = []
        if "fire" in visual_type:
            effects.append(f"visible_from_distance:{radius}")
            effects.append("investigation_likely")
            effects.append("threat_increase:1")
        elif "explosion" in visual_type:
            effects.append(f"visible_from_distance:{radius}")
            effects.append("investigation_certain")
            effects.append("threat_increase:2")
        elif "smoke" in visual_type:
            effects.append(f"visible_from_distance:{radius}")
            effects.append("investigation_possible")

        consequence = Consequence(
            id=str(uuid.uuid4())[:8],
            consequence_type=ConsequenceType.VISUAL,
            severity=severity,
            description=f"A {visual_type.replace('_', ' ')} is visible in the distance",
            source_location_id=source_location_id,
            source_character_id=source_character_id,
            radius=radius,
            delay_hours=0.0,
            duration_hours=1.0 if "fire" in visual_type else 0.1,
            effects=effects,
            created_at=datetime.now().isoformat(),
        )
        consequences.append(consequence)

        return consequences

    def queue_consequence(
        self,
        world_id: str,
        consequence: Consequence,
        current_game_day: int,
        current_game_hour: int
    ):
        """Queue a consequence to be applied later."""
        if world_id not in self.pending:
            self.pending[world_id] = []

        current_total_hours = current_game_day * 24 + current_game_hour
        trigger_at = current_total_hours + consequence.delay_hours

        self.pending[world_id].append(PendingConsequence(
            consequence=consequence,
            world_id=world_id,
            game_hour_triggers_at=trigger_at
        ))
        self._save()

        # Debug output for queued consequence
        delay_str = f"in {consequence.delay_hours}h" if consequence.delay_hours > 0 else "NOW"
        print(f"\033[38;5;220m[CONSEQUENCE QUEUED] {consequence.consequence_type.value}: {consequence.description[:50]}... ({delay_str})\033[0m")

    def check_and_trigger(
        self,
        world_id: str,
        current_game_day: int,
        current_game_hour: int
    ) -> List[Consequence]:
        """
        Check for consequences that should trigger now.

        Returns list of consequences that fired.
        """
        if world_id not in self.pending:
            return []

        current_total_hours = current_game_day * 24 + current_game_hour
        triggered = []
        remaining = []

        for pending in self.pending[world_id]:
            if current_total_hours >= pending.game_hour_triggers_at:
                pending.consequence.triggered = True
                pending.consequence.triggered_at = datetime.now().isoformat()
                triggered.append(pending.consequence)
                # Debug output for triggered consequence
                print(f"\n\033[38;5;220m{'─'*60}")
                print(f"⚡ CONSEQUENCE TRIGGERED: {pending.consequence.consequence_type.value}")
                print(f"   {pending.consequence.description}")
                if pending.consequence.effects:
                    print(f"   Effects: {', '.join(pending.consequence.effects)}")
                print(f"{'─'*60}\033[0m\n")
            else:
                remaining.append(pending)

        self.pending[world_id] = remaining
        if triggered:
            self._save()

        return triggered

    def get_pending_for_world(self, world_id: str) -> List[Consequence]:
        """Get all pending consequences for a world."""
        if world_id not in self.pending:
            return []
        return [p.consequence for p in self.pending[world_id]]

    def get_consequence_context(self, world_id: str) -> str:
        """Get a context string about pending consequences for the DM."""
        pending = self.get_pending_for_world(world_id)
        if not pending:
            return ""

        lines = ["=== PENDING CONSEQUENCES ==="]
        for c in pending[:5]:  # Show at most 5
            effect_str = ", ".join(c.effects[:2]) if c.effects else "unknown"
            lines.append(f"  - {c.description} [{c.severity.value}] -> {effect_str}")

        if len(pending) > 5:
            lines.append(f"  ... and {len(pending) - 5} more")

        return "\n".join(lines)

    def process_action_text(
        self,
        world_id: str,
        text: str,
        source_location_id: Optional[str] = None,
        source_character_id: Optional[str] = None,
        genre: str = "fantasy",
        current_game_day: int = 0,
        current_game_hour: int = 0
    ) -> List[Consequence]:
        """
        Parse text for consequential actions and queue appropriate consequences.

        This is called after player/NPC actions to detect things like:
        - "I fire my gun" -> gunshot consequence
        - "The building explodes" -> explosion consequence
        - "I scream for help" -> scream consequence
        """
        import re

        all_consequences = []
        text_lower = text.lower()

        # Sound triggers
        sound_patterns = {
            "gunshot": [r"fire[sd]?\s+(my\s+)?(gun|pistol|rifle|weapon)", r"shoot[s]?\s", r"pull[s]?\s+the\s+trigger", r"gunshot"],
            "explosion": [r"explod", r"detonat", r"blast", r"bomb\s+go"],
            "scream": [r"scream", r"shriek", r"yell[s]?\s+(for|out)", r"cry\s+out"],
            "shout": [r"shout", r"holler", r"call[s]?\s+out"],
            "breaking_glass": [r"break[s]?\s+(the\s+)?(glass|window)", r"shatter[s]?\s+(the\s+)?(glass|window)", r"smash[es]?\s+(the\s+)?(glass|window)"],
            "car_engine": [r"start[s]?\s+(the\s+)?(car|engine|vehicle|truck)", r"engine\s+roar", r"rev[s]?\s+(the\s+)?engine"],
            "car_horn": [r"honk[s]?", r"horn\s+blast", r"sound[s]?\s+(the\s+)?horn"],
        }

        for sound_type, patterns in sound_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    consequences = self.calculate_sound_consequence(
                        world_id=world_id,
                        sound_type=sound_type,
                        source_location_id=source_location_id,
                        source_character_id=source_character_id,
                        genre=genre
                    )
                    for c in consequences:
                        self.queue_consequence(world_id, c, current_game_day, current_game_hour)
                        all_consequences.append(c)
                    break  # Only trigger once per sound type

        # Visual triggers
        visual_patterns = {
            "fire_small": [r"light[s]?\s+a\s+fire", r"start[s]?\s+a\s+fire", r"campfire"],
            "fire_large": [r"building\s+(is\s+)?on\s+fire", r"engulf[ed]?\s+in\s+flame", r"inferno"],
            "explosion_flash": [r"explod", r"detonat"],
            "signal_flare": [r"fire[s]?\s+a\s+flare", r"signal\s+flare", r"shoot[s]?\s+a\s+flare"],
        }

        for visual_type, patterns in visual_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    consequences = self.calculate_visual_consequence(
                        world_id=world_id,
                        visual_type=visual_type,
                        source_location_id=source_location_id,
                        source_character_id=source_character_id,
                    )
                    for c in consequences:
                        self.queue_consequence(world_id, c, current_game_day, current_game_hour)
                        all_consequences.append(c)
                    break

        return all_consequences


# =============================================================================
# Global instance
# =============================================================================

_engine: Optional[ConsequenceEngine] = None


def get_consequence_engine() -> ConsequenceEngine:
    """Get the global consequence engine."""
    global _engine
    if _engine is None:
        _engine = ConsequenceEngine()
    return _engine


def init_consequence_engine(data_dir: Optional[Path] = None) -> ConsequenceEngine:
    """Initialize the global consequence engine."""
    global _engine
    _engine = ConsequenceEngine(data_dir)
    return _engine
