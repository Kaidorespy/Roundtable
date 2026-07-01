"""
Narrative Condition Tracker - No HP, just story.

Tracks character state through descriptive conditions:
- Injuries: "deep gash on left arm", "bruised ribs"
- Condition: Healthy → Scratched up → Wounded → Badly wounded → Critical → Incapacitated
- Hunger: Satisfied → Peckish → Hungry → Famished → Starving
- Thirst: Hydrated → Thirsty → Parched → Desperate
- Bleeding: Active bleeding that needs to be stopped

The DM describes what happens, the system tracks the consequences.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import json
import debug_logger as dbg


# =============================================================================
# Condition Scales
# =============================================================================

class OverallCondition(Enum):
    """Overall physical condition - how bad off are you?"""
    HEALTHY = "healthy"
    SCRATCHED_UP = "scratched up"
    WOUNDED = "wounded"
    BADLY_WOUNDED = "badly wounded"
    CRITICAL = "critical"
    INCAPACITATED = "incapacitated"

    @classmethod
    def from_string(cls, s: str) -> "OverallCondition":
        """Parse condition from string, case-insensitive."""
        s = s.lower().strip().replace("_", " ").replace("-", " ")
        for c in cls:
            if c.value == s:
                return c
        # Fuzzy matching
        if "healthy" in s or "fine" in s or "good" in s:
            return cls.HEALTHY
        if "scratch" in s or "minor" in s:
            return cls.SCRATCHED_UP
        if "bad" in s or "severe" in s:
            return cls.BADLY_WOUNDED
        if "critical" in s or "dying" in s:
            return cls.CRITICAL
        if "incap" in s or "unconscious" in s or "down" in s:
            return cls.INCAPACITATED
        if "wound" in s or "hurt" in s or "injured" in s:
            return cls.WOUNDED
        return cls.HEALTHY


class HungerStatus(Enum):
    """How hungry are you?"""
    SATISFIED = "satisfied"
    PECKISH = "peckish"
    HUNGRY = "hungry"
    FAMISHED = "famished"
    STARVING = "starving"

    def worsen(self) -> "HungerStatus":
        """Get the next worse hunger state."""
        order = list(HungerStatus)
        idx = order.index(self)
        return order[min(idx + 1, len(order) - 1)]

    def improve(self, steps: int = 1) -> "HungerStatus":
        """Get a better hunger state."""
        order = list(HungerStatus)
        idx = order.index(self)
        return order[max(idx - steps, 0)]


class ThirstStatus(Enum):
    """How thirsty are you?"""
    HYDRATED = "hydrated"
    THIRSTY = "thirsty"
    PARCHED = "parched"
    DESPERATE = "desperate"

    def worsen(self) -> "ThirstStatus":
        """Get the next worse thirst state."""
        order = list(ThirstStatus)
        idx = order.index(self)
        return order[min(idx + 1, len(order) - 1)]

    def improve(self, steps: int = 1) -> "ThirstStatus":
        """Get a better thirst state."""
        order = list(ThirstStatus)
        idx = order.index(self)
        return order[max(idx - steps, 0)]


class InjurySeverity(Enum):
    """How bad is this specific injury?"""
    MINOR = "minor"          # Scrape, bruise
    MODERATE = "moderate"    # Cut, sprain
    SEVERE = "severe"        # Deep wound, fracture
    CRITICAL = "critical"    # Life-threatening


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Injury:
    """A single injury on a character."""
    id: str                          # Unique ID for this injury
    description: str                 # "deep gash on left arm"
    severity: InjurySeverity = InjurySeverity.MINOR
    bleeding: bool = False           # Is it actively bleeding?
    infected: bool = False           # Has it become infected?
    treated: bool = False            # Has it been treated?
    inflicted_at: str = ""           # When it happened
    inflicted_by: str = ""           # What caused it
    notes: str = ""                  # Additional details

    def __post_init__(self):
        if not self.inflicted_at:
            self.inflicted_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "description": self.description,
            "severity": self.severity.value,
            "bleeding": self.bleeding,
            "infected": self.infected,
            "treated": self.treated,
            "inflicted_at": self.inflicted_at,
            "inflicted_by": self.inflicted_by,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Injury":
        data = data.copy()
        if "severity" in data:
            data["severity"] = InjurySeverity(data["severity"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def get_display(self) -> str:
        """Get display string for this injury."""
        parts = [self.description]
        if self.bleeding:
            parts.append("(bleeding)")
        if self.infected:
            parts.append("(infected)")
        if self.treated:
            parts.append("(treated)")
        return " ".join(parts)


@dataclass
class CharacterCondition:
    """Complete condition state for a character."""
    character_id: str
    character_name: str

    # Overall state
    condition: OverallCondition = OverallCondition.HEALTHY

    # Specific injuries
    injuries: List[Injury] = field(default_factory=list)

    # Needs
    hunger: HungerStatus = HungerStatus.SATISFIED
    thirst: ThirstStatus = ThirstStatus.HYDRATED

    # Timestamps for need degradation
    last_ate: str = ""
    last_drank: str = ""

    # Tracking
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.last_ate:
            self.last_ate = now
        if not self.last_drank:
            self.last_drank = now
        self.updated_at = now

    @property
    def is_bleeding(self) -> bool:
        """Is the character actively bleeding from any injury?"""
        return any(i.bleeding and not i.treated for i in self.injuries)

    @property
    def is_incapacitated(self) -> bool:
        """Has the character lost control due to injuries?"""
        return self.condition == OverallCondition.INCAPACITATED

    @property
    def untreated_injuries(self) -> List[Injury]:
        """Get all untreated injuries."""
        return [i for i in self.injuries if not i.treated]

    @property
    def bleeding_injuries(self) -> List[Injury]:
        """Get all actively bleeding injuries."""
        return [i for i in self.injuries if i.bleeding and not i.treated]

    def add_injury(
        self,
        description: str,
        severity: str = "minor",
        bleeding: bool = False,
        inflicted_by: str = ""
    ) -> Injury:
        """Add a new injury."""
        import uuid
        injury = Injury(
            id=str(uuid.uuid4())[:8],
            description=description,
            severity=InjurySeverity(severity) if isinstance(severity, str) else severity,
            bleeding=bleeding,
            inflicted_by=inflicted_by,
        )
        self.injuries.append(injury)
        self._update_condition_from_injuries()
        self.updated_at = datetime.now().isoformat()
        return injury

    def treat_injury(self, injury_id: str = None, description_match: str = None) -> Optional[Injury]:
        """Mark an injury as treated. Stops bleeding."""
        for injury in self.injuries:
            if injury_id and injury.id == injury_id:
                injury.treated = True
                injury.bleeding = False
                self.updated_at = datetime.now().isoformat()
                return injury
            if description_match and description_match.lower() in injury.description.lower():
                injury.treated = True
                injury.bleeding = False
                self.updated_at = datetime.now().isoformat()
                return injury
        return None

    def heal_injury(self, injury_id: str = None, description_match: str = None) -> Optional[Injury]:
        """Fully heal (remove) an injury."""
        for i, injury in enumerate(self.injuries):
            if injury_id and injury.id == injury_id:
                removed = self.injuries.pop(i)
                self._update_condition_from_injuries()
                self.updated_at = datetime.now().isoformat()
                return removed
            if description_match and description_match.lower() in injury.description.lower():
                removed = self.injuries.pop(i)
                self._update_condition_from_injuries()
                self.updated_at = datetime.now().isoformat()
                return removed
        return None

    def stop_bleeding(self, injury_id: str = None) -> int:
        """Stop bleeding on specific injury or all injuries. Returns count stopped."""
        count = 0
        for injury in self.injuries:
            if injury.bleeding:
                if injury_id is None or injury.id == injury_id:
                    injury.bleeding = False
                    count += 1
        if count:
            self.updated_at = datetime.now().isoformat()
        return count

    def set_condition(self, condition: str) -> OverallCondition:
        """Set overall condition."""
        self.condition = OverallCondition.from_string(condition)
        self.updated_at = datetime.now().isoformat()
        return self.condition

    def _update_condition_from_injuries(self):
        """Update overall condition based on injuries."""
        if not self.injuries:
            self.condition = OverallCondition.HEALTHY
            return

        # Count injuries by severity
        severity_counts = {s: 0 for s in InjurySeverity}
        for injury in self.injuries:
            if not injury.treated:
                severity_counts[injury.severity] += 1

        # Determine condition
        if severity_counts[InjurySeverity.CRITICAL] >= 1:
            self.condition = OverallCondition.CRITICAL
        elif severity_counts[InjurySeverity.SEVERE] >= 2:
            self.condition = OverallCondition.CRITICAL
        elif severity_counts[InjurySeverity.SEVERE] >= 1:
            self.condition = OverallCondition.BADLY_WOUNDED
        elif severity_counts[InjurySeverity.MODERATE] >= 2:
            self.condition = OverallCondition.BADLY_WOUNDED
        elif severity_counts[InjurySeverity.MODERATE] >= 1:
            self.condition = OverallCondition.WOUNDED
        elif severity_counts[InjurySeverity.MINOR] >= 2:
            self.condition = OverallCondition.WOUNDED
        elif severity_counts[InjurySeverity.MINOR] >= 1:
            self.condition = OverallCondition.SCRATCHED_UP
        else:
            self.condition = OverallCondition.HEALTHY

    def eat(self, quality: str = "normal") -> HungerStatus:
        """
        Eat something. Quality affects how much hunger improves.
        quality: "snack" (1 step), "normal" (2 steps), "feast" (full reset)
        """
        steps = {"snack": 1, "normal": 2, "feast": 5}.get(quality, 2)
        self.hunger = self.hunger.improve(steps)
        self.last_ate = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        return self.hunger

    def drink(self, quality: str = "normal") -> ThirstStatus:
        """
        Drink something. Quality affects how much thirst improves.
        quality: "sip" (1 step), "normal" (2 steps), "plenty" (full reset)
        """
        steps = {"sip": 1, "normal": 2, "plenty": 4}.get(quality, 2)
        self.thirst = self.thirst.improve(steps)
        self.last_drank = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        return self.thirst

    def worsen_hunger(self) -> HungerStatus:
        """Hunger gets worse over time."""
        self.hunger = self.hunger.worsen()
        self.updated_at = datetime.now().isoformat()
        return self.hunger

    def worsen_thirst(self) -> ThirstStatus:
        """Thirst gets worse over time."""
        self.thirst = self.thirst.worsen()
        self.updated_at = datetime.now().isoformat()
        return self.thirst

    def to_dict(self) -> Dict:
        return {
            "character_id": self.character_id,
            "character_name": self.character_name,
            "condition": self.condition.value,
            "injuries": [i.to_dict() for i in self.injuries],
            "hunger": self.hunger.value,
            "thirst": self.thirst.value,
            "last_ate": self.last_ate,
            "last_drank": self.last_drank,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            # Computed properties
            "is_bleeding": self.is_bleeding,
            "is_incapacitated": self.is_incapacitated,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CharacterCondition":
        data = data.copy()
        if "condition" in data:
            data["condition"] = OverallCondition(data["condition"])
        if "injuries" in data:
            data["injuries"] = [Injury.from_dict(i) for i in data["injuries"]]
        if "hunger" in data:
            data["hunger"] = HungerStatus(data["hunger"])
        if "thirst" in data:
            data["thirst"] = ThirstStatus(data["thirst"])
        # Remove computed properties
        data.pop("is_bleeding", None)
        data.pop("is_incapacitated", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def get_summary(self) -> str:
        """Get a text summary for display."""
        lines = [f"=== {self.character_name}'s Condition ==="]
        lines.append(f"Status: {self.condition.value.upper()}")

        if self.is_bleeding:
            lines.append("⚠️ ACTIVELY BLEEDING")

        if self.injuries:
            lines.append("\nInjuries:")
            for injury in self.injuries:
                marker = "  •" if not injury.treated else "  ✓"
                lines.append(f"{marker} {injury.get_display()} [{injury.severity.value}]")

        lines.append(f"\nHunger: {self.hunger.value}")
        lines.append(f"Thirst: {self.thirst.value}")

        return "\n".join(lines)

    def get_dm_context(self) -> str:
        """Get condition info formatted for DM context."""
        lines = [f"CONDITION: {self.condition.value.upper()}"]

        if self.is_bleeding:
            lines.append("⚠️ ACTIVELY BLEEDING - needs immediate attention")

        if self.injuries:
            lines.append("INJURIES:")
            for injury in self.injuries:
                status_parts = []
                if injury.bleeding:
                    status_parts.append("bleeding")
                if injury.infected:
                    status_parts.append("infected")
                if injury.treated:
                    status_parts.append("treated")
                status = f" ({', '.join(status_parts)})" if status_parts else ""
                lines.append(f"  - {injury.description} [{injury.severity.value}]{status}")

        lines.append(f"HUNGER: {self.hunger.value}")
        lines.append(f"THIRST: {self.thirst.value}")

        if self.is_incapacitated:
            lines.append("\n⚠️ CHARACTER IS INCAPACITATED - cannot act on their own")

        return "\n".join(lines)


# =============================================================================
# Condition Tracker (manages all characters)
# =============================================================================

class ConditionTracker:
    """Tracks condition for all characters."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".roundtable"
        self.conditions_file = self.data_dir / "conditions.json"
        self.conditions: Dict[str, CharacterCondition] = {}
        self._load()

    def _load(self):
        """Load conditions from disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.conditions_file.exists():
            try:
                data = json.loads(self.conditions_file.read_text())
                for char_id, cond_data in data.items():
                    self.conditions[char_id] = CharacterCondition.from_dict(cond_data)
                dbg.debug("condition", f"Loaded conditions for {len(self.conditions)} characters")
            except Exception as e:
                print(f"[Condition] Error loading: {e}")

    def save(self):
        """Save conditions to disk."""
        try:
            data = {k: v.to_dict() for k, v in self.conditions.items()}
            self.conditions_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[Condition] Error saving: {e}")

    def get_or_create(self, character_id: str, character_name: str) -> CharacterCondition:
        """Get or create condition for a character."""
        if character_id not in self.conditions:
            self.conditions[character_id] = CharacterCondition(
                character_id=character_id,
                character_name=character_name,
            )
            self.save()
        return self.conditions[character_id]

    def get(self, character_id: str) -> Optional[CharacterCondition]:
        """Get condition if it exists."""
        return self.conditions.get(character_id)

    def add_injury(
        self,
        character_id: str,
        character_name: str,
        description: str,
        severity: str = "minor",
        bleeding: bool = False,
        inflicted_by: str = ""
    ) -> Injury:
        """Add an injury to a character."""
        condition = self.get_or_create(character_id, character_name)
        injury = condition.add_injury(description, severity, bleeding, inflicted_by)
        self.save()
        dbg.debug("condition", f"Added injury to {character_name}: {description} [{severity}]")
        return injury

    def treat_injury(
        self,
        character_id: str,
        injury_id: str = None,
        description_match: str = None
    ) -> Optional[Injury]:
        """Treat an injury."""
        condition = self.get(character_id)
        if not condition:
            return None
        injury = condition.treat_injury(injury_id, description_match)
        if injury:
            self.save()
            dbg.debug("condition", f"Treated injury for {condition.character_name}: {injury.description}")
        return injury

    def heal_injury(
        self,
        character_id: str,
        injury_id: str = None,
        description_match: str = None
    ) -> Optional[Injury]:
        """Fully heal (remove) an injury."""
        condition = self.get(character_id)
        if not condition:
            return None
        injury = condition.heal_injury(injury_id, description_match)
        if injury:
            self.save()
            dbg.debug("condition", f"Healed injury for {condition.character_name}: {injury.description}")
        return injury

    def stop_bleeding(self, character_id: str, injury_id: str = None) -> int:
        """Stop bleeding."""
        condition = self.get(character_id)
        if not condition:
            return 0
        count = condition.stop_bleeding(injury_id)
        if count:
            self.save()
            dbg.debug("condition", f"Stopped {count} bleeding injuries for {condition.character_name}")
        return count

    def set_condition(self, character_id: str, condition_str: str) -> Optional[OverallCondition]:
        """Set overall condition."""
        condition = self.get(character_id)
        if not condition:
            return None
        result = condition.set_condition(condition_str)
        self.save()
        dbg.debug("condition", f"Set condition for {condition.character_name}: {result.value}")
        return result

    def eat(self, character_id: str, quality: str = "normal") -> Optional[HungerStatus]:
        """Character eats something."""
        condition = self.get(character_id)
        if not condition:
            return None
        result = condition.eat(quality)
        self.save()
        dbg.debug("condition", f"{condition.character_name} ate ({quality}): now {result.value}")
        return result

    def drink(self, character_id: str, quality: str = "normal") -> Optional[ThirstStatus]:
        """Character drinks something."""
        condition = self.get(character_id)
        if not condition:
            return None
        result = condition.drink(quality)
        self.save()
        dbg.debug("condition", f"{condition.character_name} drank ({quality}): now {result.value}")
        return result


# =============================================================================
# Global instance
# =============================================================================

_tracker: Optional[ConditionTracker] = None


def get_condition_tracker() -> ConditionTracker:
    """Get the global condition tracker."""
    global _tracker
    if _tracker is None:
        _tracker = ConditionTracker()
    return _tracker


def init_condition_tracker(data_dir: Optional[Path] = None) -> ConditionTracker:
    """Initialize the global condition tracker."""
    global _tracker
    _tracker = ConditionTracker(data_dir)
    return _tracker
