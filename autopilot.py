"""
Autopilot System - Player characters continue when players are away.

"The world doesn't pause. Neither should you."

When a player goes on autopilot:
- Their character becomes a background thread
- Makes decisions based on alignment + stats
- Survives (or doesn't) based on world conditions
- Journal tracks everything that happens

Design principles:
- Safe places stay safe (enclave in zombie world = probably fine)
- Risk scales with threat level and time
- No major plot decisions (won't declare yourself king)
- Small interactions happen (someone brings you soup)
- Death is possible in hostile worlds
- Good death descriptions if it happens
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import json
import uuid
import debug_logger as dbg


class Alignment(Enum):
    """Classic D&D 3x3 alignment grid - the interface people see."""
    LAWFUL_GOOD = "lawful_good"
    NEUTRAL_GOOD = "neutral_good"
    CHAOTIC_GOOD = "chaotic_good"
    LAWFUL_NEUTRAL = "lawful_neutral"
    TRUE_NEUTRAL = "true_neutral"
    CHAOTIC_NEUTRAL = "chaotic_neutral"
    LAWFUL_EVIL = "lawful_evil"
    NEUTRAL_EVIL = "neutral_evil"
    CHAOTIC_EVIL = "chaotic_evil"


class Priority(Enum):
    """The four things a character can prioritize."""
    SELF = "self"       # Personal survival, comfort, desires
    GROUP = "group"     # The party, friends, loved ones
    MISSION = "mission" # The objective, the job, getting it done
    IDEALS = "ideals"   # Principles, morals, what they believe in


class Drive(Enum):
    """
    Maslow-ish hierarchy - what is this character currently optimizing for?
    This can evolve over time as circumstances change.
    """
    SURVIVAL = "survival"     # Will do anything to not die
    SAFETY = "safety"         # Will sacrifice freedom for security
    BELONGING = "belonging"   # Will sacrifice safety for connection
    ESTEEM = "esteem"         # Will sacrifice connection for status
    MEANING = "meaning"       # Will sacrifice everything else for purpose


class Role(Enum):
    """Character's narrative role preference."""
    LEAD = "lead"           # Takes initiative, makes decisions
    SUPPORT = "support"     # Assists others, looks for chances to help
    BALANCED = "balanced"   # Context-dependent


# =============================================================================
# Priority Stack - The Engine Under the D&D Grid
# =============================================================================

# Maps D&D alignment to priority ordering (highest to lowest)
ALIGNMENT_TO_PRIORITIES = {
    Alignment.LAWFUL_GOOD: [Priority.IDEALS, Priority.GROUP, Priority.MISSION, Priority.SELF],
    Alignment.NEUTRAL_GOOD: [Priority.GROUP, Priority.IDEALS, Priority.SELF, Priority.MISSION],
    Alignment.CHAOTIC_GOOD: [Priority.SELF, Priority.GROUP, Priority.IDEALS, Priority.MISSION],
    Alignment.LAWFUL_NEUTRAL: [Priority.MISSION, Priority.IDEALS, Priority.GROUP, Priority.SELF],
    Alignment.TRUE_NEUTRAL: [Priority.SELF, Priority.GROUP, Priority.MISSION, Priority.IDEALS],  # Balanced
    Alignment.CHAOTIC_NEUTRAL: [Priority.SELF, Priority.MISSION, Priority.GROUP, Priority.IDEALS],
    Alignment.LAWFUL_EVIL: [Priority.MISSION, Priority.SELF, Priority.IDEALS, Priority.GROUP],
    Alignment.NEUTRAL_EVIL: [Priority.SELF, Priority.MISSION, Priority.GROUP, Priority.IDEALS],
    Alignment.CHAOTIC_EVIL: [Priority.SELF, Priority.IDEALS, Priority.MISSION, Priority.GROUP],
}


def get_priority_stack(alignment: Alignment) -> List[Priority]:
    """Get the priority ordering for an alignment."""
    return ALIGNMENT_TO_PRIORITIES.get(alignment, [Priority.SELF, Priority.GROUP, Priority.MISSION, Priority.IDEALS])


def would_sacrifice(alignment: Alignment, sacrifice_what: Priority, for_what: Priority) -> bool:
    """
    Would this character sacrifice one priority for another?
    Returns True if for_what ranks higher than sacrifice_what.
    """
    stack = get_priority_stack(alignment)
    try:
        sacrifice_rank = stack.index(sacrifice_what)
        for_rank = stack.index(for_what)
        return for_rank < sacrifice_rank  # Lower index = higher priority
    except ValueError:
        return False


# Drive evolution keywords - what we look for in recent context
DRIVE_INDICATORS = {
    Drive.SURVIVAL: ["dying", "death", "kill", "starving", "bleeding", "wounded", "escape", "run", "hide"],
    Drive.SAFETY: ["shelter", "safe", "secure", "protect", "walls", "fortify", "defend", "camp", "rest"],
    Drive.BELONGING: ["friend", "trust", "together", "we", "us", "group", "family", "love", "care"],
    Drive.ESTEEM: ["respect", "leader", "reputation", "prove", "worthy", "honor", "recognition", "best"],
    Drive.MEANING: ["purpose", "why", "cause", "believe", "mission", "destiny", "matter", "change"],
}


# Alignment display names
ALIGNMENT_NAMES = {
    Alignment.LAWFUL_GOOD: "Lawful Good",
    Alignment.NEUTRAL_GOOD: "Neutral Good",
    Alignment.CHAOTIC_GOOD: "Chaotic Good",
    Alignment.LAWFUL_NEUTRAL: "Lawful Neutral",
    Alignment.TRUE_NEUTRAL: "True Neutral",
    Alignment.CHAOTIC_NEUTRAL: "Chaotic Neutral",
    Alignment.LAWFUL_EVIL: "Lawful Evil",
    Alignment.NEUTRAL_EVIL: "Neutral Evil",
    Alignment.CHAOTIC_EVIL: "Chaotic Evil",
}

# Alignment behavior tendencies
ALIGNMENT_TRAITS = {
    Alignment.LAWFUL_GOOD: {
        "help_stranger": 0.9,
        "follow_rules": 0.95,
        "share_supplies": 0.8,
        "fight_for_others": 0.85,
        "take_risks": 0.6,
        "trust_authority": 0.8,
    },
    Alignment.NEUTRAL_GOOD: {
        "help_stranger": 0.85,
        "follow_rules": 0.6,
        "share_supplies": 0.75,
        "fight_for_others": 0.8,
        "take_risks": 0.5,
        "trust_authority": 0.5,
    },
    Alignment.CHAOTIC_GOOD: {
        "help_stranger": 0.8,
        "follow_rules": 0.3,
        "share_supplies": 0.7,
        "fight_for_others": 0.85,
        "take_risks": 0.75,
        "trust_authority": 0.2,
    },
    Alignment.LAWFUL_NEUTRAL: {
        "help_stranger": 0.5,
        "follow_rules": 0.9,
        "share_supplies": 0.4,
        "fight_for_others": 0.5,
        "take_risks": 0.4,
        "trust_authority": 0.85,
    },
    Alignment.TRUE_NEUTRAL: {
        "help_stranger": 0.5,
        "follow_rules": 0.5,
        "share_supplies": 0.5,
        "fight_for_others": 0.5,
        "take_risks": 0.5,
        "trust_authority": 0.5,
    },
    Alignment.CHAOTIC_NEUTRAL: {
        "help_stranger": 0.4,
        "follow_rules": 0.2,
        "share_supplies": 0.4,
        "fight_for_others": 0.4,
        "take_risks": 0.8,
        "trust_authority": 0.15,
    },
    Alignment.LAWFUL_EVIL: {
        "help_stranger": 0.2,
        "follow_rules": 0.85,
        "share_supplies": 0.15,
        "fight_for_others": 0.3,
        "take_risks": 0.5,
        "trust_authority": 0.7,
    },
    Alignment.NEUTRAL_EVIL: {
        "help_stranger": 0.1,
        "follow_rules": 0.4,
        "share_supplies": 0.1,
        "fight_for_others": 0.2,
        "take_risks": 0.6,
        "trust_authority": 0.3,
    },
    Alignment.CHAOTIC_EVIL: {
        "help_stranger": 0.05,
        "follow_rules": 0.1,
        "share_supplies": 0.05,
        "fight_for_others": 0.15,
        "take_risks": 0.85,
        "trust_authority": 0.05,
    },
}


@dataclass
class JournalEntry:
    """A single entry in the player's autopilot journal."""
    timestamp: str  # Real time
    game_time: str  # Game time (Day X, Hour Y)
    event_type: str  # "routine", "encounter", "combat", "social", "death"
    description: str
    severity: str = "minor"  # "minor", "notable", "major", "critical"

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "game_time": self.game_time,
            "event_type": self.event_type,
            "description": self.description,
            "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "JournalEntry":
        return cls(**data)


@dataclass
class PlayerCharacter:
    """
    Player character state for a specific world/room.

    This tracks everything needed for autopilot to work.
    """
    player_id: str  # Usually "player_{room_id}" or a user ID
    room_id: str
    character_name: str

    # Alignment (for autopilot decision-making) - the interface
    alignment: Alignment = Alignment.TRUE_NEUTRAL

    # Drive (Maslow-ish) - what are they currently optimizing for?
    # This evolves over time based on circumstances
    current_drive: Drive = Drive.SAFETY

    # Role - narrative function preference
    role: Role = Role.BALANCED

    # Autopilot state
    autopilot_enabled: bool = False
    autopilot_engaged_at: Optional[str] = None  # ISO timestamp
    last_activity: Optional[str] = None  # ISO timestamp

    # Auto-engage settings
    auto_engage_after_minutes: int = 60  # Auto-engage after 1 hour inactivity

    # Sleepy giant - drive evolution tracking
    interaction_count: int = 0
    last_drive_check: int = 0  # Interaction count when we last checked
    drive_check_interval: int = 25  # Check every N interactions
    drive_journal: List[str] = field(default_factory=list)  # Private journal for DM

    # Skills from backstory - queryable list for DM
    skills: List[str] = field(default_factory=list)
    backstory_title: str = ""
    backstory_description: str = ""

    # Current state
    is_alive: bool = True
    death_description: Optional[str] = None
    current_location_id: Optional[str] = None

    # Journal of events while on autopilot
    journal: List[JournalEntry] = field(default_factory=list)

    # Stats reference (links to Pathfinder stats if available)
    combat_stats_id: Optional[str] = None

    # The Understudy Note - a short freeform instruction for your autopilot
    # "fuck Marcus and his bike" / "never leave Opus behind" / "I burp a lot"
    # This is the soul of your stand-in. 90 chars max.
    understudy_note: str = ""

    # =========================================================================
    # Separated Character System - for characters living their own story
    # =========================================================================
    is_separated: bool = False  # Is this character away from the main party?
    separation_started: Optional[str] = None  # ISO timestamp when separated
    last_background_tick: Optional[str] = None  # Last time we processed their story
    background_tick_interval_minutes: int = 30  # How often to generate events

    # Location tracking for separated characters
    last_known_location: str = ""  # Description of where they were last seen
    location_radius_km: float = 0.0  # How far they might have traveled
    movement_speed_km_per_hour: float = 3.0  # Walking speed for radius expansion

    # Condition tracking
    condition: str = "healthy"  # "healthy", "tired", "injured", "critical", "incapacitated"
    supplies_status: str = "adequate"  # "abundant", "adequate", "low", "critical", "none"
    morale: str = "stable"  # "high", "stable", "low", "broken"

    # Full simulation fields - same systems as player
    separation_matrix: Dict = field(default_factory=dict)  # Opening matrix rolled at separation
    current_weather: str = ""  # Synced with world state
    current_time_of_day: str = ""  # Synced with world state
    hours_since_separation: float = 0.0  # For tracking fatigue/hunger progression

    # Encounters and discoveries (things they can share when reunited)
    discoveries: List[str] = field(default_factory=list)  # Locations, caches, intel found
    encounters_survived: int = 0  # Combat/dangerous encounters survived
    allies_met: List[str] = field(default_factory=list)  # NPCs befriended
    enemies_made: List[str] = field(default_factory=list)  # NPCs hostile to them

    def to_dict(self) -> Dict:
        return {
            "player_id": self.player_id,
            "room_id": self.room_id,
            "character_name": self.character_name,
            "alignment": self.alignment.value,
            "current_drive": self.current_drive.value,
            "role": self.role.value,
            "autopilot_enabled": self.autopilot_enabled,
            "autopilot_engaged_at": self.autopilot_engaged_at,
            "last_activity": self.last_activity,
            "auto_engage_after_minutes": self.auto_engage_after_minutes,
            "interaction_count": self.interaction_count,
            "last_drive_check": self.last_drive_check,
            "drive_check_interval": self.drive_check_interval,
            "drive_journal": self.drive_journal,
            "skills": self.skills,
            "backstory_title": self.backstory_title,
            "backstory_description": self.backstory_description,
            "is_alive": self.is_alive,
            "death_description": self.death_description,
            "current_location_id": self.current_location_id,
            "journal": [e.to_dict() for e in self.journal],
            "combat_stats_id": self.combat_stats_id,
            "understudy_note": self.understudy_note,
            # Separated character fields
            "is_separated": self.is_separated,
            "separation_started": self.separation_started,
            "last_background_tick": self.last_background_tick,
            "background_tick_interval_minutes": self.background_tick_interval_minutes,
            "last_known_location": self.last_known_location,
            "location_radius_km": self.location_radius_km,
            "movement_speed_km_per_hour": self.movement_speed_km_per_hour,
            "condition": self.condition,
            "supplies_status": self.supplies_status,
            "morale": self.morale,
            "separation_matrix": self.separation_matrix,
            "current_weather": self.current_weather,
            "current_time_of_day": self.current_time_of_day,
            "hours_since_separation": self.hours_since_separation,
            "discoveries": self.discoveries,
            "encounters_survived": self.encounters_survived,
            "allies_met": self.allies_met,
            "enemies_made": self.enemies_made,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PlayerCharacter":
        data = data.copy()
        data["alignment"] = Alignment(data.get("alignment", "true_neutral"))
        data["current_drive"] = Drive(data.get("current_drive", "safety"))
        data["role"] = Role(data.get("role", "balanced"))
        data["journal"] = [JournalEntry.from_dict(e) for e in data.get("journal", [])]
        data["drive_journal"] = data.get("drive_journal", [])
        data["skills"] = data.get("skills", [])
        data["backstory_title"] = data.get("backstory_title", "")
        data["backstory_description"] = data.get("backstory_description", "")
        data["understudy_note"] = data.get("understudy_note", "")
        # Separated character fields
        data["is_separated"] = data.get("is_separated", False)
        data["separation_started"] = data.get("separation_started")
        data["last_background_tick"] = data.get("last_background_tick")
        data["background_tick_interval_minutes"] = data.get("background_tick_interval_minutes", 30)
        data["last_known_location"] = data.get("last_known_location", "")
        data["location_radius_km"] = data.get("location_radius_km", 0.0)
        data["movement_speed_km_per_hour"] = data.get("movement_speed_km_per_hour", 3.0)
        data["condition"] = data.get("condition", "healthy")
        data["supplies_status"] = data.get("supplies_status", "adequate")
        data["morale"] = data.get("morale", "stable")
        data["separation_matrix"] = data.get("separation_matrix", {})
        data["current_weather"] = data.get("current_weather", "")
        data["current_time_of_day"] = data.get("current_time_of_day", "")
        data["hours_since_separation"] = data.get("hours_since_separation", 0.0)
        data["discoveries"] = data.get("discoveries", [])
        data["encounters_survived"] = data.get("encounters_survived", 0)
        data["allies_met"] = data.get("allies_met", [])
        data["enemies_made"] = data.get("enemies_made", [])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def record_activity(self):
        """Record that the player did something (resets auto-engage timer)."""
        self.last_activity = datetime.now().isoformat()

    def should_auto_engage(self) -> bool:
        """Check if autopilot should auto-engage due to inactivity."""
        if self.autopilot_enabled:
            return False  # Already on
        if not self.last_activity:
            return False  # No activity recorded yet

        try:
            last = datetime.fromisoformat(self.last_activity)
            elapsed = datetime.now() - last
            return elapsed.total_seconds() > (self.auto_engage_after_minutes * 60)
        except:
            return False

    def engage_autopilot(self):
        """Turn on autopilot."""
        self.autopilot_enabled = True
        self.autopilot_engaged_at = datetime.now().isoformat()
        self.add_journal_entry(
            event_type="routine",
            description="You settle into a watchful rest, staying alert but conserving energy.",
            severity="minor"
        )

    def disengage_autopilot(self):
        """Turn off autopilot (player is back)."""
        self.autopilot_enabled = False
        self.autopilot_engaged_at = None
        self.record_activity()

    def add_journal_entry(
        self,
        event_type: str,
        description: str,
        severity: str = "minor",
        game_day: int = 0,
        game_hour: int = 0
    ):
        """Add an entry to the autopilot journal."""
        entry = JournalEntry(
            timestamp=datetime.now().isoformat(),
            game_time=f"Day {game_day}, {game_hour:02d}:00",
            event_type=event_type,
            description=description,
            severity=severity,
        )
        self.journal.append(entry)

        # Keep journal to reasonable size (last 100 entries)
        if len(self.journal) > 100:
            self.journal = self.journal[-100:]

    def get_journal_summary(self, since_timestamp: Optional[str] = None) -> str:
        """Get a summary of journal entries since a timestamp."""
        entries = self.journal

        if since_timestamp:
            try:
                since = datetime.fromisoformat(since_timestamp)
                entries = [e for e in entries if datetime.fromisoformat(e.timestamp) > since]
            except:
                pass

        if not entries:
            return "Nothing notable happened while you were away."

        # Group by severity
        critical = [e for e in entries if e.severity == "critical"]
        major = [e for e in entries if e.severity == "major"]
        notable = [e for e in entries if e.severity == "notable"]
        minor = [e for e in entries if e.severity == "minor"]

        lines = [f"While you were away ({len(entries)} events):"]

        # Show critical first (deaths, major danger)
        for e in critical:
            lines.append(f"  [CRITICAL] {e.description}")

        # Then major
        for e in major:
            lines.append(f"  - {e.description}")

        # Then notable
        for e in notable:
            lines.append(f"  - {e.description}")

        # Summarize minor if there are many
        if len(minor) > 5:
            lines.append(f"  - ...and {len(minor)} routine events")
        else:
            for e in minor:
                lines.append(f"  - {e.description}")

        return "\n".join(lines)

    def get_alignment_tendency(self, trait: str) -> float:
        """Get the probability of this character exhibiting a trait."""
        return ALIGNMENT_TRAITS.get(self.alignment, {}).get(trait, 0.5)

    def would_help_stranger(self) -> float:
        return self.get_alignment_tendency("help_stranger")

    def would_follow_rules(self) -> float:
        return self.get_alignment_tendency("follow_rules")

    def would_share_supplies(self) -> float:
        return self.get_alignment_tendency("share_supplies")

    def would_fight_for_others(self) -> float:
        return self.get_alignment_tendency("fight_for_others")

    # =========================================================================
    # Priority Stack Engine
    # =========================================================================

    def get_priorities(self) -> List[Priority]:
        """Get this character's priority ordering based on alignment."""
        return get_priority_stack(self.alignment)

    def would_sacrifice_for(self, sacrifice: Priority, for_what: Priority) -> bool:
        """Would this character sacrifice one priority for another?"""
        return would_sacrifice(self.alignment, sacrifice, for_what)

    def top_priority(self) -> Priority:
        """What does this character care about most?"""
        return self.get_priorities()[0]

    def bottom_priority(self) -> Priority:
        """What would this character sacrifice first?"""
        return self.get_priorities()[-1]

    # =========================================================================
    # Sleepy Giant - Drive Evolution
    # =========================================================================

    def record_interaction(self, context_text: str = "") -> Optional[str]:
        """
        Record an interaction and potentially evolve the drive.
        Returns a DM-only journal entry if drive changed, None otherwise.

        The "sleepy giant" wakes up every N interactions, looks at recent
        context, and decides if the character's fundamental drive has shifted.
        """
        self.interaction_count += 1

        # Check if it's time for the sleepy giant to wake
        if self.interaction_count - self.last_drive_check < self.drive_check_interval:
            return None

        self.last_drive_check = self.interaction_count

        if not context_text:
            return None

        # Analyze context for drive indicators
        context_lower = context_text.lower()
        drive_scores = {}

        for drive, keywords in DRIVE_INDICATORS.items():
            score = sum(1 for kw in keywords if kw in context_lower)
            if score > 0:
                drive_scores[drive] = score

        if not drive_scores:
            return None

        # Find the dominant drive in recent context
        dominant = max(drive_scores, key=drive_scores.get)

        # Only change if there's a clear signal (score > 2) and it's different
        if drive_scores[dominant] > 2 and dominant != self.current_drive:
            old_drive = self.current_drive
            self.current_drive = dominant

            # Create DM-only journal entry
            entry = (
                f"[Drive Evolution] {self.character_name}'s focus has shifted from "
                f"{old_drive.value.upper()} to {dominant.value.upper()}. "
                f"Recent context shows strong {dominant.value} indicators."
            )
            self.drive_journal.append(entry)

            # Keep drive journal reasonable
            if len(self.drive_journal) > 20:
                self.drive_journal = self.drive_journal[-20:]

            return entry

        return None

    def has_skill(self, skill_query: str) -> bool:
        """
        Check if character has a skill (fuzzy match).
        Example: has_skill("hotwire") matches "hotwiring", "hotwire cars", etc.
        """
        query_lower = skill_query.lower().strip()
        query_root = query_lower.rstrip('eing')  # Handle verb forms: hotwire/hotwiring

        for skill in self.skills:
            skill_lower = skill.lower()
            skill_root = skill_lower.rstrip('eing')

            # Direct substring match
            if query_lower in skill_lower or skill_lower in query_lower:
                return True

            # Root match (handles hotwire/hotwiring, repair/repairing, etc.)
            if query_root and skill_root:
                if query_root in skill_root or skill_root in query_root:
                    return True

            # Prefix match (at least 4 chars)
            if len(query_lower) >= 4 and len(skill_lower) >= 4:
                if skill_lower.startswith(query_lower[:4]) or query_lower.startswith(skill_lower[:4]):
                    return True

        return False

    def get_skills_string(self) -> str:
        """Get skills as a comma-separated string."""
        return ", ".join(self.skills) if self.skills else "No specialized skills"

    def get_drive_context_for_dm(self) -> str:
        """
        Get drive information formatted for DM context injection.
        This is private info the DM knows but players don't see directly.
        """
        priorities = self.get_priorities()
        priority_str = " > ".join(p.value.capitalize() for p in priorities)

        lines = [
            f"[{self.character_name} - Internal State]",
            f"  Alignment: {ALIGNMENT_NAMES.get(self.alignment, 'Unknown')}",
            f"  Priority Stack: {priority_str}",
            f"  Current Drive: {self.current_drive.value.upper()} (what they're optimizing for)",
            f"  Role: {self.role.value.capitalize()}",
        ]

        # Add skills (queryable)
        if self.skills:
            lines.append(f"  Skills: {self.get_skills_string()}")

        # Add backstory summary if present
        if self.backstory_title:
            lines.append(f"  Background: {self.backstory_title}")
            if self.backstory_description:
                # Truncate long descriptions
                desc = self.backstory_description[:150]
                if len(self.backstory_description) > 150:
                    desc += "..."
                lines.append(f"    {desc}")

        # Add recent drive evolution if any
        if self.drive_journal:
            lines.append(f"  Recent Evolution: {self.drive_journal[-1]}")

        # The understudy note - the soul of the stand-in
        if self.understudy_note:
            lines.append(f"  UNDERSTUDY NOTE: \"{self.understudy_note}\"")

        return "\n".join(lines)

    def get_role_modifier(self, action_type: str) -> float:
        """
        Get role-based probability modifier for an action type.

        Lead: +20% for initiative, decisions, stepping forward
        Support: +20% for assisting, protecting, following
        Balanced: No modifier (1.0)

        Returns a multiplier (0.8 to 1.2)
        """
        lead_actions = ["take_initiative", "make_decision", "lead_group", "step_forward", "confront"]
        support_actions = ["assist", "protect", "follow", "help_other", "watch_back", "heal"]

        if self.role == Role.LEAD:
            if action_type in lead_actions:
                return 1.2  # +20% for lead actions
            elif action_type in support_actions:
                return 0.9  # -10% for support actions
        elif self.role == Role.SUPPORT:
            if action_type in support_actions:
                return 1.2  # +20% for support actions
            elif action_type in lead_actions:
                return 0.9  # -10% for lead actions

        return 1.0  # Balanced or neutral action

    def would_take_initiative(self) -> float:
        """Would this character step up and take charge?"""
        base = self.get_alignment_tendency("take_risks")
        return min(1.0, base * self.get_role_modifier("take_initiative"))

    def would_assist_other(self) -> float:
        """Would this character help another party member?"""
        base = self.get_alignment_tendency("fight_for_others")
        return min(1.0, base * self.get_role_modifier("assist"))

    def would_protect_group(self) -> float:
        """Would this character put themselves in harm's way for the group?"""
        base = 0.5
        if self.would_sacrifice_for(Priority.SELF, Priority.GROUP):
            base = 0.8
        return min(1.0, base * self.get_role_modifier("protect"))

    def get_decision_factors(self, situation_type: str) -> Dict[str, Any]:
        """
        Get all factors that would influence a decision.
        Used by story daemon when making autopilot decisions.
        """
        return {
            "alignment": self.alignment.value,
            "priorities": [p.value for p in self.get_priorities()],
            "top_priority": self.top_priority().value,
            "current_drive": self.current_drive.value,
            "role": self.role.value,
            "skills": self.skills,
            "traits": {
                "help_stranger": self.would_help_stranger(),
                "follow_rules": self.would_follow_rules(),
                "share_supplies": self.would_share_supplies(),
                "fight_for_others": self.would_fight_for_others(),
                "take_initiative": self.would_take_initiative(),
                "assist_other": self.would_assist_other(),
                "protect_group": self.would_protect_group(),
            },
            "would_sacrifice_self_for_group": self.would_sacrifice_for(Priority.SELF, Priority.GROUP),
            "would_sacrifice_group_for_mission": self.would_sacrifice_for(Priority.GROUP, Priority.MISSION),
            "would_sacrifice_mission_for_ideals": self.would_sacrifice_for(Priority.MISSION, Priority.IDEALS),
            "understudy_note": self.understudy_note,
        }

    # =========================================================================
    # Separated Character Methods
    # =========================================================================

    def separate(self, starting_location: str):
        """
        Mark this character as separated from the main party.
        They'll now live their own story in the background.
        """
        self.is_separated = True
        self.separation_started = datetime.now().isoformat()
        self.last_background_tick = datetime.now().isoformat()
        self.last_known_location = starting_location
        self.location_radius_km = 0.0  # Start at exact location
        self.add_journal_entry(
            event_type="separation",
            description=f"Separated from the group at {starting_location}. Beginning solo journey.",
            severity="notable"
        )
        print(f"[Autopilot] {self.character_name} separated at {starting_location}. Journal entries: {len(self.journal)}")

    def reunite(self):
        """
        Reunite this character with the main party.
        Returns a summary of what happened while separated.
        """
        if not self.is_separated:
            return "Was never separated."

        self.is_separated = False
        separation_summary = self.get_separation_summary()
        self.add_journal_entry(
            event_type="reunion",
            description="Reunited with the group.",
            severity="notable"
        )
        return separation_summary

    def get_separation_summary(self) -> str:
        """Get a summary of what happened while separated."""
        if not self.separation_started:
            return "No separation recorded."

        lines = [f"=== {self.character_name}'s Journey ==="]
        lines.append(f"Condition: {self.condition}")
        lines.append(f"Supplies: {self.supplies_status}")
        lines.append(f"Morale: {self.morale}")
        lines.append(f"Encounters survived: {self.encounters_survived}")

        if self.discoveries:
            lines.append(f"Discoveries: {', '.join(self.discoveries[-5:])}")  # Last 5
        if self.allies_met:
            lines.append(f"Allies made: {', '.join(self.allies_met[-3:])}")
        if self.enemies_made:
            lines.append(f"Enemies made: {', '.join(self.enemies_made[-3:])}")

        return "\n".join(lines)

    def needs_background_tick(self) -> bool:
        """Check if this character needs a background story tick."""
        if not self.is_separated or not self.is_alive:
            return False

        if not self.last_background_tick:
            return True

        try:
            last_tick = datetime.fromisoformat(self.last_background_tick)
            elapsed = datetime.now() - last_tick
            return elapsed.total_seconds() > (self.background_tick_interval_minutes * 60)
        except:
            return True

    def record_background_tick(self):
        """Record that a background tick was processed."""
        self.last_background_tick = datetime.now().isoformat()

    def expand_search_radius(self, hours_elapsed: float = 1.0):
        """
        Expand the location radius based on time elapsed.
        A separated character could have traveled during this time.
        """
        # Reduce speed based on condition
        speed_multiplier = {
            "healthy": 1.0,
            "tired": 0.7,
            "injured": 0.4,
            "critical": 0.1,
            "incapacitated": 0.0,
        }.get(self.condition, 0.5)

        distance = self.movement_speed_km_per_hour * hours_elapsed * speed_multiplier
        self.location_radius_km += distance

    def update_condition(self, new_condition: str, reason: str = ""):
        """Update character condition with journal entry."""
        old_condition = self.condition
        self.condition = new_condition

        severity = "minor"
        if new_condition in ["injured", "critical"]:
            severity = "major"
        elif new_condition == "incapacitated":
            severity = "critical"

        desc = f"Condition changed: {old_condition} → {new_condition}"
        if reason:
            desc += f" ({reason})"

        self.add_journal_entry(
            event_type="condition",
            description=desc,
            severity=severity
        )

    def add_discovery(self, discovery: str):
        """Record something the character discovered."""
        self.discoveries.append(discovery)
        if len(self.discoveries) > 20:
            self.discoveries = self.discoveries[-20:]
        self.add_journal_entry(
            event_type="discovery",
            description=f"Discovered: {discovery}",
            severity="notable"
        )

    def record_encounter(self, survived: bool, description: str):
        """Record a dangerous encounter."""
        if survived:
            self.encounters_survived += 1
            self.add_journal_entry(
                event_type="encounter",
                description=f"Survived: {description}",
                severity="notable"
            )
        else:
            self.add_journal_entry(
                event_type="encounter",
                description=f"Overwhelmed: {description}",
                severity="critical"
            )

    def get_rss_feed_entries(self, limit: int = 50) -> List[Dict]:
        """Get journal entries formatted for RSS feed display."""
        entries = self.journal[-limit:] if limit else self.journal
        return [
            {
                "character_id": self.player_id,
                "character_name": self.character_name,
                "timestamp": e.timestamp,
                "game_time": e.game_time,
                "event_type": e.event_type,
                "description": e.description,
                "severity": e.severity,
                "is_separated": self.is_separated,
                "condition": self.condition,
            }
            for e in reversed(entries)  # Most recent first
        ]


class AutopilotTracker:
    """
    Tracks autopilot state for all players across all rooms.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".roundtable"
        self.autopilot_file = self.data_dir / "autopilot.json"
        self.players: Dict[str, PlayerCharacter] = {}  # key = "{player_id}_{room_id}"
        self._load()

    def _load(self):
        """Load autopilot data from disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.autopilot_file.exists():
            try:
                data = json.loads(self.autopilot_file.read_text())
                for key, pdata in data.items():
                    self.players[key] = PlayerCharacter.from_dict(pdata)
            except Exception as e:
                print(f"[Autopilot] Error loading: {e}")

    def _save(self):
        """Save autopilot data to disk."""
        try:
            data = {k: v.to_dict() for k, v in self.players.items()}
            self.autopilot_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[Autopilot] Error saving: {e}")

    def _key(self, player_id: str, room_id: str) -> str:
        return f"{player_id}_{room_id}"

    def get_or_create(
        self,
        player_id: str,
        room_id: str,
        character_name: str = "Player"
    ) -> PlayerCharacter:
        """Get or create a player character for autopilot tracking."""
        key = self._key(player_id, room_id)
        if key not in self.players:
            self.players[key] = PlayerCharacter(
                player_id=player_id,
                room_id=room_id,
                character_name=character_name,
            )
            self._save()
        return self.players[key]

    def get(self, player_id: str, room_id: str) -> Optional[PlayerCharacter]:
        """Get player character if it exists."""
        return self.players.get(self._key(player_id, room_id))

    def record_activity(self, player_id: str, room_id: str):
        """Record player activity (resets auto-engage timer)."""
        pc = self.get(player_id, room_id)
        if pc:
            pc.record_activity()
            # If autopilot was on, turn it off (player is back)
            if pc.autopilot_enabled:
                pc.disengage_autopilot()
            self._save()

    def toggle_autopilot(self, player_id: str, room_id: str) -> bool:
        """Toggle autopilot on/off. Returns new state."""
        pc = self.get(player_id, room_id)
        if not pc:
            return False

        if pc.autopilot_enabled:
            pc.disengage_autopilot()
        else:
            pc.engage_autopilot()

        self._save()
        return pc.autopilot_enabled

    def set_alignment(self, player_id: str, room_id: str, alignment: Alignment):
        """Set player's alignment."""
        pc = self.get(player_id, room_id)
        if pc:
            pc.alignment = alignment
            self._save()

    def set_role(self, player_id: str, room_id: str, role: Role):
        """Set player's narrative role preference."""
        pc = self.get(player_id, room_id)
        if pc:
            pc.role = role
            self._save()

    def set_drive(self, player_id: str, room_id: str, drive: Drive):
        """Manually set player's current drive (usually auto-evolves)."""
        pc = self.get(player_id, room_id)
        if pc:
            pc.current_drive = drive
            self._save()

    def set_understudy_note(self, player_id: str, room_id: str, note: str):
        """
        Set the understudy note - the soul of your stand-in.

        This is a short freeform instruction (90 chars max) that tells your
        autopilot character what to never forget:
        - "fuck Marcus and his bike"
        - "never leave Opus behind, she's ride or die"
        - "I burp a lot"
        """
        pc = self.get(player_id, room_id)
        if pc:
            # Enforce 90 char limit
            pc.understudy_note = note[:90] if note else ""
            self._save()

    def process_interaction(
        self,
        player_id: str,
        room_id: str,
        context_text: str
    ) -> Optional[str]:
        """
        Process an interaction - records activity and checks for drive evolution.
        Returns DM-only journal entry if drive changed.
        """
        pc = self.get(player_id, room_id)
        if not pc:
            return None

        pc.record_activity()
        dm_note = pc.record_interaction(context_text)
        self._save()
        return dm_note

    def get_dm_context(self, player_id: str, room_id: str) -> Optional[str]:
        """Get drive/priority context for DM injection."""
        pc = self.get(player_id, room_id)
        if pc:
            return pc.get_drive_context_for_dm()
        return None

    def check_auto_engage(self) -> List[PlayerCharacter]:
        """Check all players for auto-engage conditions. Returns those who engaged."""
        engaged = []
        for pc in self.players.values():
            if pc.should_auto_engage():
                pc.engage_autopilot()
                engaged.append(pc)

        if engaged:
            self._save()

        return engaged

    def get_autopilot_players_in_room(self, room_id: str) -> List[PlayerCharacter]:
        """Get all players on autopilot in a specific room."""
        return [
            pc for pc in self.players.values()
            if pc.room_id == room_id and pc.autopilot_enabled and pc.is_alive
        ]

    def add_journal_entry(
        self,
        player_id: str,
        room_id: str,
        event_type: str,
        description: str,
        severity: str = "minor",
        game_day: int = 0,
        game_hour: int = 0
    ):
        """Add a journal entry for a player."""
        pc = self.get(player_id, room_id)
        if pc:
            pc.add_journal_entry(event_type, description, severity, game_day, game_hour)
            self._save()

    def kill_player(
        self,
        player_id: str,
        room_id: str,
        death_description: str,
        game_day: int = 0,
        game_hour: int = 0
    ):
        """Record a player death."""
        pc = self.get(player_id, room_id)
        if pc:
            pc.is_alive = False
            pc.death_description = death_description
            pc.add_journal_entry(
                event_type="death",
                description=death_description,
                severity="critical",
                game_day=game_day,
                game_hour=game_hour
            )
            self._save()

    # =========================================================================
    # Separated Character Management
    # =========================================================================

    def separate_character(
        self,
        player_id: str,
        room_id: str,
        starting_location: str
    ) -> bool:
        """
        Mark a character as separated from the main party.
        They will now live their own background story.
        """
        pc = self.get(player_id, room_id)
        if pc:
            pc.separate(starting_location)
            self._save()
            return True
        return False

    def reunite_character(self, player_id: str, room_id: str) -> Optional[str]:
        """
        Reunite a separated character with the main party.
        Returns their journey summary.
        """
        pc = self.get(player_id, room_id)
        if pc:
            summary = pc.reunite()
            self._save()
            return summary
        return None

    def get_separated_characters_in_room(self, room_id: str) -> List[PlayerCharacter]:
        """Get all separated characters in a specific room."""
        return [
            pc for pc in self.players.values()
            if pc.room_id == room_id and pc.is_separated and pc.is_alive
        ]

    def get_characters_needing_tick(self, room_id: str) -> List[PlayerCharacter]:
        """Get separated characters that need a background tick."""
        return [
            pc for pc in self.players.values()
            if pc.room_id == room_id and pc.needs_background_tick()
        ]

    def get_all_characters_in_room(self, room_id: str) -> List[PlayerCharacter]:
        """Get all characters (separated or not) in a room."""
        return [
            pc for pc in self.players.values()
            if pc.room_id == room_id and pc.is_alive
        ]

    def get_rss_feed(self, room_id: str, character_ids: List[str] = None, limit: int = 100) -> List[Dict]:
        """
        Get RSS feed entries for characters in a room.

        Args:
            room_id: The room to get feed for
            character_ids: Optional filter for specific characters (None = all)
            limit: Max entries per character
        """
        feed = []
        for pc in self.players.values():
            if pc.room_id != room_id:
                continue
            if character_ids and pc.player_id not in character_ids:
                continue

            feed.extend(pc.get_rss_feed_entries(limit=limit // max(1, len(character_ids or [1]))))

        # Sort by timestamp, most recent first
        feed.sort(key=lambda x: x["timestamp"], reverse=True)
        return feed[:limit]

    def check_proximity(
        self,
        room_id: str,
        player_location: str,
        player_radius_km: float = 5.0,
        character_relationships: List[Dict] = None
    ) -> List[Dict]:
        """
        Check if any separated characters might be within range of the player.
        Returns list of potential encounters with probability.

        Factors in relationships:
        - Strangers: No encounter (ships passing in the night)
        - Acquaintances: Low probability, "wait, is that...?"
        - Friends/Allies: Higher probability, sixth sense pull
        - Close bonds (family, lover): Highest probability
        - Rivals/Enemies: Can trigger confrontation instead of reunion

        This is a simplified check - the cartographer has more detailed location tracking.
        """
        # Build relationship lookup: character_id -> {type, note, target_name}
        relationship_map = {}
        if character_relationships:
            for char_rel in character_relationships:
                char_id = char_rel.get('character_id', '')
                for rel in char_rel.get('relationships', []):
                    target_id = rel.get('target_id', '')
                    if target_id:
                        # Store relationship from this character's perspective
                        key = f"{char_id}_{target_id}"
                        relationship_map[key] = {
                            'type': rel.get('type', 'stranger'),
                            'note': rel.get('note', ''),
                            'from_name': char_rel.get('character_name', ''),
                            'to_name': rel.get('target_name', '')
                        }

        # Relationship type -> probability modifier and encounter type
        RELATIONSHIP_MODIFIERS = {
            'stranger': (0.0, None),           # No encounter
            'acquaintance': (1.0, 'sighting'),  # Base probability, might recognize
            'colleague': (1.2, 'sighting'),     # Slightly better
            'friend': (1.5, 'reunion'),         # Good chance, warm reunion
            'close_friend': (2.0, 'reunion'),   # Strong pull
            'family': (2.5, 'reunion'),         # Blood calls to blood
            'lover': (3.0, 'reunion'),          # Strongest bond
            'ally': (1.5, 'reunion'),           # Shared purpose
            'rival': (1.3, 'confrontation'),    # They'll find each other
            'enemy': (1.5, 'confrontation'),    # Drawn together by hate
            'mentor': (1.8, 'reunion'),         # Teacher-student bond
            'protege': (1.8, 'reunion'),        # Student-teacher bond
        }

        potential_encounters = []
        player_id = f"player_{room_id}"

        for pc in self.get_separated_characters_in_room(room_id):
            # Simple heuristic: if locations share keywords, they might be close
            player_keywords = set(player_location.lower().split())
            char_keywords = set(pc.last_known_location.lower().split())
            overlap = player_keywords & char_keywords

            if not (overlap or pc.location_radius_km > 10):
                continue  # Too far, no chance

            # Calculate base probability from distance
            base_probability = min(0.5, 5.0 / max(1.0, pc.location_radius_km))

            # Look up relationship (check both directions)
            rel_key_1 = f"{pc.player_id}_{player_id}"
            rel_key_2 = f"{player_id}_{pc.player_id}"
            relationship = relationship_map.get(rel_key_1) or relationship_map.get(rel_key_2)

            if relationship:
                rel_type = relationship.get('type', 'stranger').lower().replace(' ', '_')
            else:
                rel_type = 'stranger'

            # Get modifier and encounter type
            modifier, encounter_type = RELATIONSHIP_MODIFIERS.get(rel_type, (0.0, None))

            # Strangers don't encounter each other
            if encounter_type is None:
                continue

            # Apply relationship modifier
            final_probability = min(0.9, base_probability * modifier)

            # Must meet minimum threshold to show up
            if final_probability < 0.05:
                continue

            potential_encounters.append({
                "character_id": pc.player_id,
                "character_name": pc.character_name,
                "last_known_location": pc.last_known_location,
                "search_radius_km": pc.location_radius_km,
                "encounter_probability": round(final_probability, 2),
                "encounter_type": encounter_type,  # 'reunion', 'sighting', or 'confrontation'
                "relationship_type": rel_type,
                "condition": pc.condition,
                "morale": pc.morale,
                "supplies": pc.supplies_status,
                "shared_location_hints": list(overlap),
            })

        # Sort by probability (highest first)
        potential_encounters.sort(key=lambda x: x['encounter_probability'], reverse=True)
        return potential_encounters


# =============================================================================
# Global instance
# =============================================================================

_tracker: Optional[AutopilotTracker] = None


def get_autopilot_tracker() -> AutopilotTracker:
    """Get the global autopilot tracker."""
    global _tracker
    if _tracker is None:
        _tracker = AutopilotTracker()
    return _tracker


def init_autopilot_tracker(data_dir: Optional[Path] = None) -> AutopilotTracker:
    """Initialize the global autopilot tracker."""
    global _tracker
    _tracker = AutopilotTracker(data_dir)
    return _tracker
