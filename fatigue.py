"""
Fatigue System - Sleep, rest, and exhaustion tracking.

"Your character has to sleep. Otherwise what's the point of time?"

This system is FORGIVING:
- Characters can stay up 2-3 days
- Effects are gradual (tired → exhausted → impaired)
- Not punishing, but makes time matter
"""

from typing import Dict, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import json
import debug_logger as dbg


class FatigueLevel(Enum):
    """How tired is this character?"""
    RESTED = "rested"           # Full energy, no penalties
    FINE = "fine"               # Normal, slight tiredness (12-18 hrs awake)
    TIRED = "tired"             # Noticeable fatigue (18-24 hrs awake)
    EXHAUSTED = "exhausted"     # Significant impairment (24-36 hrs awake)
    DEPLETED = "depleted"       # Severe impairment (36-48 hrs awake)
    CRITICAL = "critical"       # Barely functional (48+ hrs awake)


# Fatigue thresholds (hours awake)
FATIGUE_THRESHOLDS = {
    FatigueLevel.RESTED: 0,
    FatigueLevel.FINE: 12,
    FatigueLevel.TIRED: 18,
    FatigueLevel.EXHAUSTED: 24,
    FatigueLevel.DEPLETED: 36,
    FatigueLevel.CRITICAL: 48,
}

# Recovery rates (hours of rest needed to fully recover from each level)
RECOVERY_RATES = {
    FatigueLevel.RESTED: 0,
    FatigueLevel.FINE: 2,
    FatigueLevel.TIRED: 4,
    FatigueLevel.EXHAUSTED: 6,
    FatigueLevel.DEPLETED: 8,
    FatigueLevel.CRITICAL: 10,
}


@dataclass
class CharacterFatigue:
    """Fatigue state for a single character."""
    character_id: str
    character_name: str

    # Core fatigue tracking
    hours_awake: float = 0.0
    hours_since_rest: float = 0.0
    last_sleep: Optional[str] = None  # ISO timestamp (game time)
    last_rest: Optional[str] = None   # ISO timestamp (real time update)

    # Current state
    fatigue_level: FatigueLevel = FatigueLevel.RESTED
    is_resting: bool = False
    is_sleeping: bool = False

    # Modifiers
    endurance_bonus: float = 0.0      # Positive = can stay awake longer
    recovery_bonus: float = 0.0       # Positive = recovers faster

    # History
    total_hours_slept: float = 0.0
    times_hit_critical: int = 0

    def to_dict(self) -> Dict:
        return {
            "character_id": self.character_id,
            "character_name": self.character_name,
            "hours_awake": self.hours_awake,
            "hours_since_rest": self.hours_since_rest,
            "last_sleep": self.last_sleep,
            "last_rest": self.last_rest,
            "fatigue_level": self.fatigue_level.value,
            "is_resting": self.is_resting,
            "is_sleeping": self.is_sleeping,
            "endurance_bonus": self.endurance_bonus,
            "recovery_bonus": self.recovery_bonus,
            "total_hours_slept": self.total_hours_slept,
            "times_hit_critical": self.times_hit_critical,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CharacterFatigue":
        data = data.copy()
        if "fatigue_level" in data:
            data["fatigue_level"] = FatigueLevel(data["fatigue_level"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class FatigueTracker:
    """
    Tracks fatigue for all characters in the game.

    Design principles:
    - Forgiving: You can push through for days if needed
    - Gradual: Effects accumulate slowly
    - Meaningful: Makes time and rest actually matter
    - Genre-flexible: Works for any setting
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".roundtable"
        self.fatigue_file = self.data_dir / "fatigue.json"
        self.characters: Dict[str, CharacterFatigue] = {}
        self._load()

    def _load(self):
        """Load fatigue data from disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.fatigue_file.exists():
            try:
                data = json.loads(self.fatigue_file.read_text())
                for char_id, char_data in data.items():
                    self.characters[char_id] = CharacterFatigue.from_dict(char_data)
            except Exception as e:
                print(f"[Fatigue] Error loading: {e}")

    def _save(self):
        """Save fatigue data to disk."""
        try:
            data = {k: v.to_dict() for k, v in self.characters.items()}
            self.fatigue_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[Fatigue] Error saving: {e}")

    def get_or_create(self, character_id: str, character_name: str) -> CharacterFatigue:
        """Get or create fatigue tracking for a character."""
        if character_id not in self.characters:
            self.characters[character_id] = CharacterFatigue(
                character_id=character_id,
                character_name=character_name,
            )
            self._save()
        return self.characters[character_id]

    def get_fatigue(self, character_id: str) -> Optional[CharacterFatigue]:
        """Get fatigue state for a character."""
        return self.characters.get(character_id)

    def advance_time(self, character_id: str, hours: float, load_multiplier: float = 1.0):
        """
        Advance time for a character (they've been awake).

        Call this when game time passes and the character is active.

        Args:
            character_id: The character's ID
            hours: How many hours have passed
            load_multiplier: Encumbrance penalty (1.0 = normal, 1.2 = heavy, 1.5 = overburdened)
        """
        char = self.get_fatigue(character_id)
        if not char:
            return

        if char.is_sleeping or char.is_resting:
            # Resting recovers fatigue
            self._process_rest(char, hours)
        else:
            # Awake time accumulates fatigue (load makes it worse)
            effective_hours = hours * load_multiplier
            char.hours_awake += effective_hours
            char.hours_since_rest += effective_hours
            self._update_fatigue_level(char)

        self._save()

    def get_load_multiplier(self, load_percent: float) -> float:
        """
        Calculate fatigue multiplier based on load percentage.

        Args:
            load_percent: Current weight / capacity * 100

        Returns:
            Multiplier for fatigue accumulation (1.0 = normal)
        """
        if load_percent < 50:
            return 1.0  # Light load - no penalty
        elif load_percent < 70:
            return 1.0  # Normal load - no penalty
        elif load_percent < 85:
            return 1.2  # Heavy load - 20% faster fatigue
        else:
            return 1.5  # Overburdened - 50% faster fatigue

    def _update_fatigue_level(self, char: CharacterFatigue):
        """Update fatigue level based on hours awake."""
        effective_hours = char.hours_awake - char.endurance_bonus
        old_level = char.fatigue_level

        # Find appropriate fatigue level
        new_level = FatigueLevel.RESTED
        for level, threshold in sorted(FATIGUE_THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
            if effective_hours >= threshold:
                new_level = level
                break

        # Track if hitting critical for first time
        if new_level == FatigueLevel.CRITICAL and char.fatigue_level != FatigueLevel.CRITICAL:
            char.times_hit_critical += 1
            dbg.fatigue(f"⚠ {char.character_name} hit CRITICAL fatigue! ({char.times_hit_critical}x)")

        if new_level != old_level:
            dbg.fatigue(f"● {char.character_name} fatigue: {old_level.value} → {new_level.value} ({effective_hours:.1f}h awake)")

        char.fatigue_level = new_level

    def _process_rest(self, char: CharacterFatigue, hours: float):
        """Process rest/sleep time."""
        # Recovery rate: 1 hour of sleep = recover ~3 hours of fatigue
        # Resting (not sleeping) is half as effective
        recovery_multiplier = 3.0 if char.is_sleeping else 1.5
        recovery_multiplier *= (1.0 + char.recovery_bonus)

        recovered_hours = hours * recovery_multiplier
        char.hours_awake = max(0, char.hours_awake - recovered_hours)

        if char.is_sleeping:
            char.total_hours_slept += hours

        self._update_fatigue_level(char)

    def start_sleep(self, character_id: str, game_time_iso: str = None, character_name: str = None):
        """Character starts sleeping. Auto-creates tracking if needed."""
        dbg.fatigue(f"● {character_name or character_id} starts sleeping")
        char = self.get_fatigue(character_id)
        if not char:
            # Auto-create fatigue tracking on first rest
            name = character_name or character_id.replace('player_', 'You (') + ')' if character_id.startswith('player_') else character_id
            char = self.get_or_create(character_id, name)

        char.is_sleeping = True
        char.is_resting = True
        char.last_sleep = game_time_iso or datetime.now().isoformat()
        self._save()

    def start_rest(self, character_id: str, character_name: str = None):
        """Character starts resting (not full sleep). Auto-creates tracking if needed."""
        char = self.get_fatigue(character_id)
        if not char:
            # Auto-create fatigue tracking on first rest
            name = character_name or character_id.replace('player_', 'You (') + ')' if character_id.startswith('player_') else character_id
            char = self.get_or_create(character_id, name)

        char.is_resting = True
        char.is_sleeping = False
        char.last_rest = datetime.now().isoformat()
        self._save()

    def wake_up(self, character_id: str):
        """Character wakes up / stops resting."""
        char = self.get_fatigue(character_id)
        if not char:
            return

        char.is_sleeping = False
        char.is_resting = False
        self._save()

    def get_fatigue_effects(self, character_id: str) -> Dict[str, any]:
        """
        Get the gameplay effects of current fatigue level.

        Returns modifiers that can be applied to actions.
        """
        char = self.get_fatigue(character_id)
        if not char:
            return {"level": "unknown", "modifiers": {}}

        effects = {
            FatigueLevel.RESTED: {
                "description": "Well-rested and alert",
                "reaction_modifier": 0,
                "perception_modifier": 0,
                "willpower_modifier": 0,
                "can_concentrate": True,
            },
            FatigueLevel.FINE: {
                "description": "Slightly tired but functional",
                "reaction_modifier": 0,
                "perception_modifier": 0,
                "willpower_modifier": 0,
                "can_concentrate": True,
            },
            FatigueLevel.TIRED: {
                "description": "Noticeably fatigued, yawning",
                "reaction_modifier": -1,
                "perception_modifier": -1,
                "willpower_modifier": 0,
                "can_concentrate": True,
            },
            FatigueLevel.EXHAUSTED: {
                "description": "Exhausted, struggling to focus",
                "reaction_modifier": -2,
                "perception_modifier": -2,
                "willpower_modifier": -1,
                "can_concentrate": False,  # Disadvantage on concentration
            },
            FatigueLevel.DEPLETED: {
                "description": "Running on fumes, microsleeps",
                "reaction_modifier": -3,
                "perception_modifier": -3,
                "willpower_modifier": -2,
                "can_concentrate": False,
            },
            FatigueLevel.CRITICAL: {
                "description": "Barely conscious, hallucinating",
                "reaction_modifier": -5,
                "perception_modifier": -5,
                "willpower_modifier": -3,
                "can_concentrate": False,
            },
        }

        level_effects = effects.get(char.fatigue_level, effects[FatigueLevel.RESTED])

        return {
            "level": char.fatigue_level.value,
            "hours_awake": char.hours_awake,
            "is_resting": char.is_resting,
            "is_sleeping": char.is_sleeping,
            **level_effects,
        }

    def get_fatigue_context(self, character_id: str) -> str:
        """Get a context string for the DM about a character's fatigue."""
        effects = self.get_fatigue_effects(character_id)
        if effects["level"] == "unknown":
            return ""

        char = self.get_fatigue(character_id)

        lines = [f"{char.character_name}'s Fatigue Status:"]
        lines.append(f"  Level: {effects['level'].upper()}")
        lines.append(f"  {effects['description']}")
        lines.append(f"  Hours awake: {char.hours_awake:.1f}")

        if char.is_sleeping:
            lines.append("  Currently SLEEPING")
        elif char.is_resting:
            lines.append("  Currently RESTING")

        # Only mention modifiers if they exist
        if effects.get("reaction_modifier", 0) != 0:
            lines.append(f"  Reaction penalty: {effects['reaction_modifier']}")

        return "\n".join(lines)

    def get_all_fatigue_context(self) -> str:
        """Get fatigue summary for all tracked characters."""
        if not self.characters:
            return "No fatigue tracking active."

        lines = ["=== Fatigue Status ===\n"]
        for char in self.characters.values():
            status = "💤" if char.is_sleeping else "🛋️" if char.is_resting else ""
            level_emoji = {
                "rested": "✨",
                "fine": "😊",
                "tired": "😐",
                "exhausted": "😫",
                "depleted": "😵",
                "critical": "💀",
            }.get(char.fatigue_level.value, "")

            lines.append(
                f"{level_emoji} {char.character_name}: {char.fatigue_level.value} "
                f"({char.hours_awake:.1f}h awake) {status}"
            )

        return "\n".join(lines)


# =============================================================================
# Global instance
# =============================================================================

_tracker: Optional[FatigueTracker] = None


def get_fatigue_tracker() -> FatigueTracker:
    """Get the global fatigue tracker."""
    global _tracker
    if _tracker is None:
        _tracker = FatigueTracker()
    return _tracker


def init_fatigue_tracker(data_dir: Optional[Path] = None) -> FatigueTracker:
    """Initialize the global fatigue tracker."""
    global _tracker
    _tracker = FatigueTracker(data_dir)
    return _tracker
