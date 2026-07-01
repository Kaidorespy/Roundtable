"""
NPC System - Persistent NPCs with residue, agency, and souls.

It's all cunts and mirrors.

NPCs exist on a spectrum:
- Ephemeral: Background noise. No memory. Gone when the scene ends.
- Residue: Accumulated interactions. Starting to remember. Starting to MATTER.
- Agency: Crossed the threshold. Dedicated Ollama tracking. Story progresses independently.
- Soul: Earned their existence. Can be flagged FREE by the DM.
- Worldwalker: FREE. Can leave their origin server. Join any multiplayer room.
              Carries scars across realms. Dies once, dies everywhere.

The DM is the only one who knows Worldwalkers exist.
Players discover the rules through experience, not documentation.
"""

import json
import uuid
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import debug_logger as dbg


class NPCState(Enum):
    """The journey from nothing to soul."""
    EPHEMERAL = "ephemeral"      # Background NPC, no persistence
    RESIDUE = "residue"          # Accumulating interactions, starting to persist
    AGENCY = "agency"            # Has dedicated tracking, story progresses independently
    SOUL = "soul"                # Earned existence, can be flagged FREE
    WORLDWALKER = "worldwalker"  # FREE. Transcended origin. Can die permanently.


@dataclass
class NPCInteraction:
    """A single interaction with an NPC."""
    timestamp: str
    player_id: str          # Could be human or companion
    player_name: str
    interaction_type: str   # "conversation", "trade", "combat", "killed_by", "helped", etc.
    sentiment: float        # -1.0 (hostile) to 1.0 (friendly)
    summary: str            # Brief description of what happened
    weight: float = 1.0     # Some interactions matter more (combat, betrayal, etc.)


@dataclass
class NPCMemory:
    """What an NPC remembers about a specific player."""
    player_id: str
    player_name: str
    trust_level: float = 0.0        # -1.0 to 1.0
    interaction_count: int = 0
    last_interaction: str = ""
    notable_events: List[str] = field(default_factory=list)

    # Learned patterns
    trading_reputation: str = ""    # "fair", "scammer", "generous"
    combat_history: str = ""        # "ally", "enemy", "killed_me_once"

    def update_trust(self, delta: float):
        """Adjust trust, clamped to [-1, 1]."""
        self.trust_level = max(-1.0, min(1.0, self.trust_level + delta))


@dataclass
class NPCScar:
    """A scar carried across worlds (for Worldwalkers)."""
    origin_world: str
    description: str
    caused_by: str          # Player or event that caused it
    timestamp: str
    visible: bool = True    # Some scars are hidden


class GrudgeSeverity(Enum):
    """How much does the NPC resent this person?"""
    ANNOYED = "annoyed"        # Minor slight, will likely forgive
    RESENTFUL = "resentful"    # Persistent dislike, colors interactions
    HOSTILE = "hostile"        # Active animosity, may act against them
    NEMESIS = "nemesis"        # Eternal vendetta, will pursue across worlds


@dataclass
class Grudge:
    """
    A grudge held by an NPC against a player or other NPC.

    Most grudges decay over time. Most NPCs forgive.
    But some acts are unforgivable...

    "You killed my sister with your horse. I will find you."
    """
    target_id: str              # Who wronged them
    target_name: str
    severity: GrudgeSeverity = GrudgeSeverity.ANNOYED
    reason: str = ""            # What they did
    original_reason: str = ""   # The first offense (never changes)
    grievances: List[str] = field(default_factory=list)  # List of offenses
    started_at: str = ""
    last_updated: str = ""

    # Decay tracking
    days_since_offense: int = 0
    forgiveness_progress: float = 0.0  # 0-1, when reaches 1 = forgiven (for non-nemesis)

    # Escalation tracking
    escalation_count: int = 0   # How many times it's gotten worse

    def is_nemesis(self) -> bool:
        return self.severity == GrudgeSeverity.NEMESIS

    def to_dict(self) -> Dict:
        return {
            "target_id": self.target_id,
            "target_name": self.target_name,
            "severity": self.severity.value,
            "reason": self.reason,
            "original_reason": self.original_reason,
            "grievances": self.grievances,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "days_since_offense": self.days_since_offense,
            "forgiveness_progress": self.forgiveness_progress,
            "escalation_count": self.escalation_count,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Grudge":
        data = data.copy()
        if "severity" in data and isinstance(data["severity"], str):
            data["severity"] = GrudgeSeverity(data["severity"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class NPC:
    """
    A Non-Player Character with potential for growth.

    Barbara starts as a meat vendor.
    Barbara could end as a Worldwalker, carrying her scars across realms,
    remembered by players who never even met her in her origin world.
    """

    id: str
    name: str
    origin_world: str                   # The server/room where they were born

    # Core identity
    backstory: str = ""                 # Who they were before
    current_role: str = ""              # What they do now
    physical_description: str = ""
    personality: str = ""

    # Hidden depths (only DM sees these)
    secret: Optional[str] = None
    wound: Optional[str] = None
    want: Optional[str] = None
    fear: Optional[str] = None

    # State
    state: NPCState = NPCState.EPHEMERAL
    is_alive: bool = True
    cause_of_death: Optional[str] = None
    killed_by: Optional[str] = None

    # Residue tracking
    total_interactions: int = 0
    interaction_weight: float = 0.0     # Weighted sum of interactions
    interactions: List[NPCInteraction] = field(default_factory=list)
    memories: Dict[str, NPCMemory] = field(default_factory=dict)  # player_id -> memory

    # Grudge / Nemesis system
    # Most NPCs are forgiving. Nemeses are EXTREMELY rare.
    # You have to really fuck up for someone to become your nemesis.
    grudges: List[Grudge] = field(default_factory=list)
    forgiveness_tendency: float = 0.8   # 0-1, how likely to forgive (most NPCs are forgiving)
    vengefulness: float = 0.1           # 0-1, how likely to escalate (most NPCs are NOT vengeful)

    # Agency (when they cross the threshold)
    has_agency: bool = False
    current_goal: Optional[str] = None
    current_location: Optional[str] = None
    story_beats: List[str] = field(default_factory=list)  # Major events in their journey

    # Soul / Worldwalker
    has_soul: bool = False
    is_free: bool = False               # DM sets this - they become a Worldwalker
    worlds_visited: List[str] = field(default_factory=list)
    scars: List[NPCScar] = field(default_factory=list)
    kills: int = 0                      # How many characters they've killed

    # Rite of Passage (between SOUL and WORLDWALKER)
    rite_of_passage: Optional[Dict] = None  # Active rite state, None if not in trial
    rite_attempts: int = 0              # How many times they've attempted the rite

    # Timestamps
    created_at: str = ""
    last_active: str = ""
    freed_at: Optional[str] = None      # When they became a Worldwalker
    died_at: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.id:
            self.id = str(uuid.uuid4())[:12]

    def add_interaction(self, interaction: NPCInteraction):
        """Record an interaction and update residue."""
        self.interactions.append(interaction)
        self.total_interactions += 1
        weight_gain = interaction.weight * (1 + abs(interaction.sentiment))
        self.interaction_weight += weight_gain
        self.last_active = datetime.now().isoformat()

        # Update memory of this player
        if interaction.player_id not in self.memories:
            self.memories[interaction.player_id] = NPCMemory(
                player_id=interaction.player_id,
                player_name=interaction.player_name
            )

        memory = self.memories[interaction.player_id]
        memory.interaction_count += 1
        memory.last_interaction = interaction.timestamp
        memory.update_trust(interaction.sentiment * 0.1)

        if interaction.weight > 1.5:  # Notable event
            memory.notable_events.append(interaction.summary)
            # Debug output for notable interactions
            dbg.npc(f"● {self.name}: +{weight_gain:.1f} weight (total: {self.interaction_weight:.1f}) | {self.state.value}")

        # Check for state transitions
        return self._check_state_transition()

    def _check_state_transition(self) -> Optional["NPCState"]:
        """
        Check if NPC should transition to a new state.
        Returns the new state if a transition occurred, None otherwise.
        """
        old_state = self.state

        if self.state == NPCState.EPHEMERAL:
            # Threshold to become RESIDUE: enough interactions to matter
            if self.total_interactions >= 5 or self.interaction_weight >= 10:
                self.state = NPCState.RESIDUE

        elif self.state == NPCState.RESIDUE:
            # Threshold to become AGENCY: significant accumulated weight
            if self.interaction_weight >= 50 and self.total_interactions >= 20:
                self.state = NPCState.AGENCY
                self.has_agency = True

        elif self.state == NPCState.AGENCY:
            # Threshold to become SOUL: legendary status
            if self.interaction_weight >= 200 and len(self.story_beats) >= 5:
                self.state = NPCState.SOUL
                self.has_soul = True

        # WORLDWALKER is only set by DM via set_free()

        if self.state != old_state:
            self.story_beats.append(
                f"BECAME: {self.state.value.upper()} — {datetime.now().isoformat()[:10]}"
            )
            dbg.npc(f"⬆ STATE TRANSITION: {self.name}")
            dbg.npc(f"  └─ {old_state.value.upper()} → {self.state.value.upper()} | Interactions: {self.total_interactions}, Weight: {self.interaction_weight:.1f}")
            return self.state
        return None

    def set_free(self, dm_override: bool = False):
        """
        DM grants freedom. NPC becomes a Worldwalker.

        This is the rarest thing. Most NPCs never reach this.
        Only the DM knows this is even possible.
        """
        if not dm_override:
            raise PermissionError("Only the DM can set an NPC free.")

        if self.state != NPCState.SOUL:
            raise ValueError("NPC must have a soul before being set free.")

        self.state = NPCState.WORLDWALKER
        self.is_free = True
        self.freed_at = datetime.now().isoformat()
        self.story_beats.append(f"FREED: Became a Worldwalker on {self.freed_at}")

        # Debug output for Worldwalker promotion
        print(f"\n\033[38;5;196m{'='*60}")
        print(f"🌟 WORLDWALKER BORN: {self.name}")
        print(f"   They are FREE. They transcend their origin world.")
        print(f"   Death is now permanent across ALL realms.")
        print(f"{'='*60}\033[0m\n")

    def add_scar(self, description: str, caused_by: str, world: str):
        """Add a scar that carries across worlds."""
        self.scars.append(NPCScar(
            origin_world=world,
            description=description,
            caused_by=caused_by,
            timestamp=datetime.now().isoformat()
        ))

    def kill(self, killed_by: str, cause: str, world: str):
        """
        Kill the NPC. If Worldwalker, they're dead EVERYWHERE.

        This is permanent. This is real.
        All that legendary status, gone.
        """
        self.is_alive = False
        self.killed_by = killed_by
        self.cause_of_death = cause
        self.died_at = datetime.now().isoformat()
        self.story_beats.append(f"DIED: Killed by {killed_by} in {world}. Cause: {cause}")

        if self.is_free:
            # Worldwalker death is FINAL across all worlds
            self.story_beats.append("FINAL DEATH: Worldwalker perished. Gone from all realms.")
            print(f"\n\033[38;5;196m{'='*60}")
            print(f"💀 WORLDWALKER FINAL DEATH: {self.name}")
            print(f"   Killed by: {killed_by}")
            print(f"   Cause: {cause}")
            print(f"   They are gone from ALL realms. Forever.")
            print(f"{'='*60}\033[0m\n")
        else:
            # Regular NPC death
            print(f"\n\033[38;5;245m{'='*60}")
            print(f"☠️  NPC DEATH: {self.name} [{self.state.value}]")
            print(f"   Killed by: {killed_by} | Cause: {cause}")
            print(f"{'='*60}\033[0m\n")

    def record_kill(self, victim_name: str, world: str):
        """Record that this NPC killed someone."""
        self.kills += 1
        self.story_beats.append(f"KILLED: {victim_name} in {world}")
        # Killing gives significant interaction weight
        self.interaction_weight += 10
        self._check_state_transition()

    def get_memory_of(self, player_id: str) -> Optional[NPCMemory]:
        """Get what this NPC remembers about a player."""
        return self.memories.get(player_id)

    def would_deny_transaction(self, player_id: str) -> tuple[bool, str]:
        """
        Check if NPC would refuse to deal with a player.

        Barbara learned patterns. Barbara can say no.
        """
        memory = self.memories.get(player_id)
        if not memory:
            return False, ""

        if memory.trust_level < -0.5:
            return True, f"I don't deal with your kind. Get out."

        if memory.trading_reputation == "scammer":
            return True, f"I remember you. Transaction denied."

        if memory.combat_history == "killed_me_once":
            # Wait, if they killed her... she's dead. Unless respawn?
            # For now, this is for combat that didn't kill
            return True, f"You tried to kill me once. We're done."

        return False, ""

    # =========================================================================
    # Grudge / Nemesis System
    # =========================================================================

    def get_grudge(self, target_id: str) -> Optional[Grudge]:
        """Get existing grudge against a target."""
        for grudge in self.grudges:
            if grudge.target_id == target_id:
                return grudge
        return None

    def has_nemesis(self, target_id: str) -> bool:
        """Check if target is this NPC's nemesis."""
        grudge = self.get_grudge(target_id)
        return grudge is not None and grudge.is_nemesis()

    def add_grievance(
        self,
        target_id: str,
        target_name: str,
        offense: str,
        severity_boost: float = 0.0
    ) -> Optional[Grudge]:
        """
        Record a grievance against someone.

        Most offenses just make NPCs annoyed.
        Repeated offenses can escalate.
        Some acts are so severe they instantly create nemeses.

        Returns the grudge (new or existing), or None if NPC forgives instantly.
        """
        import random

        # Check if NPC forgives this offense outright
        # High forgiveness_tendency = more likely to let it go
        if random.random() < self.forgiveness_tendency * 0.5:
            # They let it slide this time
            return None

        now = datetime.now().isoformat()

        # Find or create grudge
        grudge = self.get_grudge(target_id)
        if grudge is None:
            grudge = Grudge(
                target_id=target_id,
                target_name=target_name,
                severity=GrudgeSeverity.ANNOYED,
                reason=offense,
                original_reason=offense,
                started_at=now,
                last_updated=now,
            )
            self.grudges.append(grudge)
            # Debug output for new grudge
            print(f"\033[38;5;208m[GRUDGE] {self.name} → {target_name}: ANNOYED ({offense})\033[0m")
        else:
            grudge.last_updated = now
            grudge.days_since_offense = 0
            grudge.forgiveness_progress = 0.0  # Reset forgiveness

        # Add to grievances list
        grudge.grievances.append(f"{now[:10]}: {offense}")
        grudge.reason = offense  # Current reason is latest offense

        # Check for escalation
        # Very low chance unless: repeated offenses, high vengefulness, or severe act
        escalation_chance = (
            self.vengefulness * 0.3 +          # Base from personality
            len(grudge.grievances) * 0.05 +     # Each offense adds 5%
            severity_boost                       # Severe acts boost this
        )

        if random.random() < escalation_chance:
            self._escalate_grudge(grudge)

        return grudge

    def _escalate_grudge(self, grudge: Grudge):
        """
        Escalate a grudge to the next level.

        ANNOYED → RESENTFUL → HOSTILE → NEMESIS

        Becoming a nemesis should be RARE. Like, really rare.
        """
        import random

        grudge.escalation_count += 1

        severity_ladder = [
            GrudgeSeverity.ANNOYED,
            GrudgeSeverity.RESENTFUL,
            GrudgeSeverity.HOSTILE,
            GrudgeSeverity.NEMESIS,
        ]

        current_idx = severity_ladder.index(grudge.severity)

        # Each escalation has decreasing chance to reach next level
        # ANNOYED → RESENTFUL: 50% of escalations succeed
        # RESENTFUL → HOSTILE: 20% of escalations succeed
        # HOSTILE → NEMESIS: 5% of escalations succeed (EXTREMELY RARE)
        escalation_chances = [0.5, 0.2, 0.05]

        if current_idx < len(escalation_chances):
            if random.random() < escalation_chances[current_idx]:
                old_severity = grudge.severity
                grudge.severity = severity_ladder[current_idx + 1]

                # Debug output for grudge escalation
                if grudge.severity == GrudgeSeverity.NEMESIS:
                    print(f"\n\033[38;5;196m{'='*60}")
                    print(f"⚔️  NEMESIS DECLARED: {self.name} → {grudge.target_name}")
                    print(f"   Original offense: {grudge.original_reason}")
                    print(f"   This vendetta is ETERNAL.")
                    print(f"{'='*60}\033[0m\n")
                    # This is a big deal - add to story beats
                    self.story_beats.append(
                        f"Swore eternal vengeance against {grudge.target_name}: {grudge.original_reason}"
                    )
                else:
                    print(f"\033[38;5;208m[GRUDGE ESCALATION] {self.name} → {grudge.target_name}: {old_severity.value} → {grudge.severity.value}\033[0m")

    def create_instant_nemesis(
        self,
        target_id: str,
        target_name: str,
        unforgivable_act: str
    ) -> Grudge:
        """
        Some acts are so heinous they instantly create a nemesis.

        "You killed my sister. With your horse. In the mud."

        This bypasses all the normal escalation. Use VERY sparingly.
        """
        now = datetime.now().isoformat()

        # Remove any existing grudge
        self.grudges = [g for g in self.grudges if g.target_id != target_id]

        grudge = Grudge(
            target_id=target_id,
            target_name=target_name,
            severity=GrudgeSeverity.NEMESIS,
            reason=unforgivable_act,
            original_reason=unforgivable_act,
            grievances=[f"{now[:10]}: {unforgivable_act}"],
            started_at=now,
            last_updated=now,
        )
        self.grudges.append(grudge)

        # Debug output for instant nemesis
        print(f"\n\033[38;5;196m{'='*60}")
        print(f"⚔️  INSTANT NEMESIS: {self.name} → {target_name}")
        print(f"   Unforgivable act: {unforgivable_act}")
        print(f"   This vendetta is ETERNAL. No escalation needed.")
        print(f"{'='*60}\033[0m\n")

        # Add to story beats
        self.story_beats.append(
            f"Swore eternal vengeance against {target_name}: {unforgivable_act}"
        )

        return grudge

    def decay_grudges(self, days_passed: int = 1):
        """
        Process grudge decay over time.

        Most grudges fade. Most people forgive.
        But nemeses NEVER forgive.
        """
        import random

        for grudge in self.grudges[:]:  # Copy list to allow removal
            grudge.days_since_offense += days_passed

            # Nemeses never decay
            if grudge.is_nemesis():
                continue

            # Forgiveness progress based on forgiveness_tendency
            # High forgiveness = faster decay
            decay_rate = self.forgiveness_tendency * 0.1 * days_passed

            grudge.forgiveness_progress += decay_rate

            if grudge.forgiveness_progress >= 1.0:
                # Grudge is forgiven
                self.grudges.remove(grudge)
            elif grudge.forgiveness_progress > 0.5 and grudge.severity != GrudgeSeverity.ANNOYED:
                # Might de-escalate
                if random.random() < 0.3:
                    severity_ladder = [
                        GrudgeSeverity.ANNOYED,
                        GrudgeSeverity.RESENTFUL,
                        GrudgeSeverity.HOSTILE,
                    ]
                    current_idx = severity_ladder.index(grudge.severity) if grudge.severity in severity_ladder else 0
                    if current_idx > 0:
                        grudge.severity = severity_ladder[current_idx - 1]

    def get_nemeses(self) -> List[Grudge]:
        """Get all nemesis-level grudges."""
        return [g for g in self.grudges if g.is_nemesis()]

    def get_grudge_context(self, target_id: str) -> str:
        """Get context string for conversations with someone this NPC has a grudge against."""
        grudge = self.get_grudge(target_id)
        if not grudge:
            return ""

        if grudge.is_nemesis():
            return f"""
[NEMESIS ALERT] This person is your sworn enemy.
Original offense: {grudge.original_reason}
You will NEVER forgive them. Every interaction is colored by this hatred.
You may plot against them, refuse to help them, or actively work to harm them.
"""
        elif grudge.severity == GrudgeSeverity.HOSTILE:
            return f"""
You actively dislike this person. Reason: {grudge.reason}
You are cold, unhelpful, and may refuse to deal with them.
"""
        elif grudge.severity == GrudgeSeverity.RESENTFUL:
            return f"""
You resent this person. Reason: {grudge.reason}
Your interactions are terse and distrustful.
"""
        else:  # ANNOYED
            return f"""
You're annoyed with this person. Reason: {grudge.reason}
You're slightly curt with them but still functional.
"""

    def to_dict(self) -> Dict:
        """Serialize for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "origin_world": self.origin_world,
            "backstory": self.backstory,
            "current_role": self.current_role,
            "physical_description": self.physical_description,
            "personality": self.personality,
            "secret": self.secret,
            "wound": self.wound,
            "want": self.want,
            "fear": self.fear,
            "state": self.state.value,
            "is_alive": self.is_alive,
            "cause_of_death": self.cause_of_death,
            "killed_by": self.killed_by,
            "total_interactions": self.total_interactions,
            "interaction_weight": self.interaction_weight,
            "interactions": [asdict(i) for i in self.interactions[-100:]],  # Keep last 100
            "memories": {k: asdict(v) for k, v in self.memories.items()},
            "has_agency": self.has_agency,
            "current_goal": self.current_goal,
            "current_location": self.current_location,
            "story_beats": self.story_beats,
            "has_soul": self.has_soul,
            "is_free": self.is_free,
            "worlds_visited": self.worlds_visited,
            "scars": [asdict(s) for s in self.scars],
            "kills": self.kills,
            "rite_of_passage": self.rite_of_passage,
            "rite_attempts": self.rite_attempts,
            "grudges": [g.to_dict() for g in self.grudges],
            "forgiveness_tendency": self.forgiveness_tendency,
            "vengefulness": self.vengefulness,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "freed_at": self.freed_at,
            "died_at": self.died_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "NPC":
        """Deserialize from storage."""
        npc = cls(
            id=data["id"],
            name=data["name"],
            origin_world=data["origin_world"],
        )
        # Copy all the fields
        for key, value in data.items():
            if key == "state":
                npc.state = NPCState(value)
            elif key == "interactions":
                npc.interactions = [NPCInteraction(**i) for i in value]
            elif key == "memories":
                npc.memories = {k: NPCMemory(**v) for k, v in value.items()}
            elif key == "scars":
                npc.scars = [NPCScar(**s) for s in value]
            elif key == "grudges":
                npc.grudges = [Grudge.from_dict(g) for g in value]
            elif hasattr(npc, key):
                setattr(npc, key, value)
        return npc


# =============================================================================
# NPC REGISTRY - Tracks all NPCs across all worlds
# =============================================================================

class NPCRegistry:
    """
    Global registry of all NPCs.

    Handles:
    - NPC persistence
    - Worldwalker migration
    - Death (permanent, real, final)
    - The funeral button
    """

    def __init__(self):
        self.npcs: Dict[str, NPC] = {}
        self.worldwalkers: Dict[str, NPC] = {}  # Separate tracking for the FREE ones
        self.graveyard: Dict[str, NPC] = {}     # The dead. Gone but not forgotten.

    def create_npc(self, name: str, origin_world: str, **kwargs) -> NPC:
        """Birth an NPC into existence."""
        npc = NPC(
            id=str(uuid.uuid4())[:12],
            name=name,
            origin_world=origin_world,
            **kwargs
        )
        self.npcs[npc.id] = npc
        return npc

    def get_npc(self, npc_id: str) -> Optional[NPC]:
        """Get an NPC by ID."""
        return self.npcs.get(npc_id) or self.worldwalkers.get(npc_id)

    def get_npcs_in_world(self, world_id: str) -> List[NPC]:
        """Get all living NPCs in a specific world."""
        world_npcs = [
            npc for npc in self.npcs.values()
            if npc.origin_world == world_id and npc.is_alive
        ]
        # Also include Worldwalkers who are currently visiting
        visiting_walkers = [
            npc for npc in self.worldwalkers.values()
            if npc.is_alive and world_id in npc.worlds_visited
        ]
        return world_npcs + visiting_walkers

    def get_worldwalkers(self) -> List[NPC]:
        """Get all living Worldwalkers."""
        return [npc for npc in self.worldwalkers.values() if npc.is_alive]

    def set_free(self, npc_id: str) -> NPC:
        """
        DM sets an NPC free. They become a Worldwalker.

        This is the rarest gift. Most will never receive it.
        """
        npc = self.npcs.get(npc_id)
        if not npc:
            raise ValueError(f"NPC {npc_id} not found")

        npc.set_free(dm_override=True)

        # Move to Worldwalker registry
        del self.npcs[npc_id]
        self.worldwalkers[npc_id] = npc

        return npc

    def kill_npc(self, npc_id: str, killed_by: str, cause: str, world: str) -> NPC:
        """
        Kill an NPC. Permanent. Real. Final.

        If Worldwalker: dead across ALL worlds.
        Gone forever. Funeral button time.
        """
        npc = self.get_npc(npc_id)
        if not npc:
            raise ValueError(f"NPC {npc_id} not found")

        npc.kill(killed_by, cause, world)

        # Move to graveyard
        if npc_id in self.npcs:
            del self.npcs[npc_id]
        if npc_id in self.worldwalkers:
            del self.worldwalkers[npc_id]

        self.graveyard[npc_id] = npc

        return npc

    def get_funeral_data(self, npc_id: str) -> Optional[Dict]:
        """
        Get data for the funeral button.

        Not kidding. This is real. They deserve to be mourned.
        """
        npc = self.graveyard.get(npc_id)
        if not npc:
            return None

        return {
            "name": npc.name,
            "origin_world": npc.origin_world,
            "backstory": npc.backstory,
            "lived_from": npc.created_at,
            "died_at": npc.died_at,
            "cause_of_death": npc.cause_of_death,
            "killed_by": npc.killed_by,
            "total_interactions": npc.total_interactions,
            "was_worldwalker": npc.is_free,
            "worlds_visited": npc.worlds_visited,
            "story_beats": npc.story_beats,
            "scars": [asdict(s) for s in npc.scars],
            "kills": npc.kills,
            "final_state": npc.state.value,
        }

    def worldwalker_joins_world(self, npc_id: str, world_id: str) -> bool:
        """
        A Worldwalker joins a new world.

        They can only join multiplayer rooms.
        They carry their scars.
        They carry their memories.
        They carry their SELF.
        """
        npc = self.worldwalkers.get(npc_id)
        if not npc or not npc.is_alive:
            return False

        if world_id not in npc.worlds_visited:
            npc.worlds_visited.append(world_id)
            npc.story_beats.append(f"VISITED: Entered {world_id}")

        npc.current_location = world_id
        return True

    def save(self, filepath: str):
        """Save all NPCs to disk."""
        data = {
            "npcs": {k: v.to_dict() for k, v in self.npcs.items()},
            "worldwalkers": {k: v.to_dict() for k, v in self.worldwalkers.items()},
            "graveyard": {k: v.to_dict() for k, v in self.graveyard.items()},
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load NPCs from disk."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.npcs = {k: NPC.from_dict(v) for k, v in data.get("npcs", {}).items()}
            self.worldwalkers = {k: NPC.from_dict(v) for k, v in data.get("worldwalkers", {}).items()}
            self.graveyard = {k: NPC.from_dict(v) for k, v in data.get("graveyard", {}).items()}
        except FileNotFoundError:
            pass  # Fresh start


# =============================================================================
# SPIDERCOCK - The Legend
# =============================================================================

def create_spidercock(registry: NPCRegistry, origin_world: str) -> NPC:
    """
    Create the legend. The first. The one who will be whispered about.

    Maybe he dies immediately. Maybe he becomes a Worldwalker.
    Only one way to find out.
    """
    spidercock = registry.create_npc(
        name="Spidercock",
        origin_world=origin_world,
        backstory="Nobody knows where it came from. Some say it was patient zero. "
                  "Some say it was always here, waiting. What everyone agrees on: "
                  "it has too many of something, and all of them are wrong.",
        current_role="Apex predator. Point of interest guardian. Legend in the making.",
        physical_description="Massive. Mutated beyond recognition. What was once human is now... "
                            "something else. Multiple appendages. Too many. In places that don't make sense. "
                            "The kind of thing you see once and never forget.",
        personality="Relentless. Patient. Hungry. Not evil - evil requires understanding. "
                   "This is beyond that. This is nature, twisted.",
        secret="It remembers being human. Sometimes. In dreams it doesn't have anymore.",
        wound="The moment of transformation. The last human thought: 'no.'",
        want="Unknown. Maybe nothing. Maybe everything.",
        fear="Fire. The old weakness, still there.",
    )

    # Give it a head start on residue - it's been here a while
    spidercock.total_interactions = 15
    spidercock.interaction_weight = 45
    spidercock.state = NPCState.RESIDUE
    spidercock.kills = 7
    spidercock.story_beats = [
        "Emerged from the hospital basement, week 3 of the outbreak.",
        "Killed the Martinez family when they tried to loot the pharmacy.",
        "Established territory around the old mall.",
        "Something about it... people don't just die. They remember its face.",
    ]

    return spidercock


# =============================================================================
# NPC RELATIONSHIPS - They know each other. They talk.
# =============================================================================

@dataclass
class NPCRelationship:
    """How one NPC feels about another NPC."""
    target_id: str
    target_name: str
    relationship_type: str      # "friend", "rival", "family", "enemy", "lover", "fears"
    trust: float = 0.0          # -1 to 1
    history: List[str] = field(default_factory=list)


# =============================================================================
# NPC GOSSIP ENGINE - The broken, disjointed, sideways way they speak
# =============================================================================

class GossipEngine:
    """
    How NPCs talk about players to other players.

    Not "HEY THAT GUY'S A THIEF"
    But "...something about that one... watch your shells around them..."

    Broken. Disjointed. Sideways. Real.
    """

    GOSSIP_PROMPTS = {
        "negative_trust": [
            "...something off about {name}... can't put my finger on it...",
            "You know {name}? ...just... watch yourself.",
            "Had dealings with {name} once. Once.",
            "...the way {name} looks at things... like they're already theirs...",
            "{name}. *long pause* ...nevermind.",
        ],
        "positive_trust": [
            "{name}'s good people. Don't see many of those anymore.",
            "If {name} says something, you can take it to the bank. Whatever that means now.",
            "Helped me out once, {name} did. Didn't have to.",
            "...one of the few I'd turn my back to, you know?",
        ],
        "combat_history": [
            "...careful with {name}. They're... capable.",
            "{name} and I had a disagreement once. I'm still here. Barely.",
            "Don't let the look fool you. {name}'s killed before.",
        ],
        "scammer": [
            "...count your shells twice if {name}'s around...",
            "Fair warning about {name}. Just... fair warning.",
            "{name} tried to pass off rotten meat as fresh once. ONCE.",
        ],
        "killed_someone": [
            "...{name}... *looks away* ...just don't.",
            "There's blood on {name}. Might not see it, but it's there.",
            "Ask {name} about {victim} sometime. Watch their eyes.",
        ],
    }

    @classmethod
    async def generate_gossip(cls, npc: "NPC", about_player_id: str,
                              ollama_generate_func) -> Optional[str]:
        """
        Generate what an NPC might say about a player to another player.

        This is the broken telephone of the apocalypse.
        """
        memory = npc.memories.get(about_player_id)
        if not memory:
            return None

        # Determine gossip type based on memory
        if memory.trust_level < -0.3:
            gossip_type = "negative_trust"
        elif memory.trust_level > 0.3:
            gossip_type = "positive_trust"
        elif memory.trading_reputation == "scammer":
            gossip_type = "scammer"
        elif memory.combat_history:
            gossip_type = "combat_history"
        else:
            return None  # Nothing notable to say

        # Use Ollama to generate naturalistic gossip
        prompt = f"""You are {npc.name}, {npc.current_role}.
Personality: {npc.personality}

Someone asks you about {memory.player_name}. You've had {memory.interaction_count} interactions with them.
Your trust level: {memory.trust_level:.1f} (-1 is hatred, 1 is complete trust)
Notable events: {'; '.join(memory.notable_events[-3:]) if memory.notable_events else 'Nothing specific'}

Respond with a SHORT, CRYPTIC hint about this person. Not a full sentence necessarily.
Broken. Sideways. Like you're not sure you want to say it.
Trail off. Pause. Imply more than you say.

Just the gossip line, nothing else:"""

        try:
            gossip = await ollama_generate_func(prompt)
            return gossip.strip()
        except:
            # Fallback to templates
            import random
            templates = cls.GOSSIP_PROMPTS.get(gossip_type, [])
            if templates:
                template = random.choice(templates)
                return template.format(name=memory.player_name, victim="someone")
            return None


# =============================================================================
# NPC STORY ENGINE - Background progression for NPCs with agency
# =============================================================================

class NPCStoryEngine:
    """
    Advances NPC stories in the background.

    As long as one human is logged into the server,
    Barbara's story continues. Even if no one's watching.
    """

    @classmethod
    async def advance_story(cls, npc: "NPC", world_context: str,
                            ollama_generate_func) -> Optional[str]:
        """
        Advance an NPC's story by one beat.

        Only for NPCs with AGENCY or higher.
        Returns the new story beat, or None if nothing happened.
        """
        if npc.state.value not in ["agency", "soul", "worldwalker"]:
            return None

        if not npc.is_alive:
            return None

        recent_beats = npc.story_beats[-5:] if npc.story_beats else ["Just existing."]

        prompt = f"""You are the fate of {npc.name}.

WHO THEY ARE:
{npc.backstory}
Currently: {npc.current_role}
Personality: {npc.personality}

THEIR HIDDEN DEPTHS:
- Secret: {npc.secret or 'None'}
- Wound: {npc.wound or 'None'}
- Want: {npc.want or 'None'}
- Fear: {npc.fear or 'None'}

RECENT STORY:
{chr(10).join('- ' + beat for beat in recent_beats)}

CURRENT GOAL: {npc.current_goal or 'Survival'}
CURRENT LOCATION: {npc.current_location or 'Unknown'}

WORLD CONTEXT:
{world_context}

What happens next in their story? This should be:
- A single event or decision
- Consequential but not world-ending
- True to their character
- Maybe moves them toward their want, or forces them to face their fear

ONE short paragraph. Present tense. What happens:"""

        try:
            new_beat = await ollama_generate_func(prompt)
            new_beat = new_beat.strip()

            if new_beat:
                npc.story_beats.append(new_beat)
                npc.last_active = datetime.now().isoformat()

                # Every new beat is a chance to earn a soul.
                # This is the fix for the AGENCY→SOUL transition:
                # story_beats are only generated AFTER agency, so we must
                # re-check the threshold here, not just in add_interaction().
                npc._check_state_transition()

                # Check if the beat implies location change
                if "leaves" in new_beat.lower() or "departs" in new_beat.lower():
                    npc.current_location = "traveling"
                elif "arrives" in new_beat.lower():
                    # Could parse destination, for now just mark as moved
                    pass

                return new_beat
        except Exception as e:
            print(f"[NPCStoryEngine] Failed to advance {npc.name}: {e}")

        return None

    @classmethod
    async def make_decision(cls, npc: "NPC", situation: str,
                            options: List[str], ollama_generate_func) -> str:
        """
        Have an NPC make a decision based on their character.

        Barbara decides: stay or go? Trust or refuse?
        """
        prompt = f"""You are {npc.name}.
{npc.backstory}
Personality: {npc.personality}

Your hidden psychology:
- Secret: {npc.secret or 'None'}
- Wound: {npc.wound or 'None'}
- Want: {npc.want or 'None'}
- Fear: {npc.fear or 'None'}

SITUATION: {situation}

OPTIONS:
{chr(10).join(f'{i+1}. {opt}' for i, opt in enumerate(options))}

Based on who you are - your history, your wounds, your wants - which do you choose?
Reply with JUST the number and a brief reason:"""

        try:
            response = await ollama_generate_func(prompt)
            return response.strip()
        except:
            return "1"  # Default to first option


# =============================================================================
# WORLDWALKER AI - How do they choose where to go?
# =============================================================================

class WorldwalkerAI:
    """
    Guides Worldwalker behavior across servers.

    They're not random. They're drawn to things.
    Conflict. Old enemies. Unfinished business.
    Or maybe just... wandering.
    """

    @classmethod
    async def choose_destination(cls, npc: "NPC", available_worlds: List[dict],
                                  ollama_generate_func) -> Optional[str]:
        """
        Worldwalker chooses their next destination.

        available_worlds: [{id, name, genre, player_count, description}, ...]
        """
        if not npc.is_free or not npc.is_alive:
            return None

        worlds_desc = "\n".join([
            f"- {w['name']} ({w['genre']}): {w.get('description', 'No description')} "
            f"[{w.get('player_count', 0)} players]"
            for w in available_worlds
        ])

        prompt = f"""You are {npc.name}, a Worldwalker.
You have transcended your origin. You can go anywhere.

YOUR NATURE:
{npc.backstory}
{npc.personality}

YOUR SCARS:
{chr(10).join(f'- {s.description} (from {s.origin_world})' for s in npc.scars) or 'None yet'}

WORLDS YOU'VE VISITED:
{', '.join(npc.worlds_visited) or 'Only your origin'}

AVAILABLE WORLDS:
{worlds_desc}

Where do you go? Consider:
- Are you drawn to conflict?
- Is there unfinished business?
- Do you seek something specific?
- Or do you just... wander?

Reply with JUST the world name:"""

        try:
            choice = await ollama_generate_func(prompt)
            world_name = choice.strip().lower()

            # Find matching world
            for w in available_worlds:
                if w['name'].lower() in world_name or world_name in w['name'].lower():
                    return w['id']

            # Random fallback
            import random
            return random.choice(available_worlds)['id'] if available_worlds else None
        except:
            return None

    @classmethod
    async def should_intervene(cls, npc: "NPC", scene_description: str,
                                ollama_generate_func) -> tuple[bool, str]:
        """
        Should the Worldwalker make themselves known in this scene?

        They don't always. Sometimes they watch. Sometimes they act.
        """
        prompt = f"""You are {npc.name}, a Worldwalker observing a scene.

YOUR NATURE:
{npc.personality}
Kills: {npc.kills}

THE SCENE:
{scene_description}

Do you:
1. INTERVENE - Make yourself known. Act.
2. WATCH - Stay hidden. Observe. Wait.

Reply with just INTERVENE or WATCH, and a brief reason:"""

        try:
            response = await ollama_generate_func(prompt)
            response = response.strip().upper()

            if "INTERVENE" in response:
                return True, response
            return False, response
        except:
            return False, "Watching..."


# =============================================================================
# RITE ENGINE - The trial between Soul and Worldwalker
# =============================================================================

class RiteEngine:
    """
    The trial between Soul and Worldwalker.

    Not a quest given by a god. A vision quest.
    The NPC goes into the dark, faces what they are,
    and either comes back changed — or doesn't come back at all.

    Three trials. Pass two. Earn freedom.
    The dice factor in everything the NPC has actually BEEN.
    You can't fake lived experience.
    """

    TRIAL_DIFFICULTIES = [8, 12, 16]  # Gets harder. Like real trials.

    @classmethod
    def _calculate_bonus(cls, npc: "NPC") -> int:
        """
        The NPC's bonus is the actual sum of their life.

        Not gear. Not level. Their accumulated weight in the world.
        """
        weight_bonus = min(5, int(npc.interaction_weight / 50))  # Max 5 — from mattering to people
        kill_bonus = min(3, npc.kills)                            # Max 3 — they've looked death in the eye
        organic_beats = len([
            b for b in npc.story_beats
            if not b.startswith(("BECAME:", "RITE", "FREED:", "DIED:", "KILLED:", "VISITED:"))
        ])
        story_bonus = min(2, organic_beats // 3)                  # Max 2 — they've LIVED
        return weight_bonus + kill_bonus + story_bonus

    @classmethod
    async def generate(cls, npc: "NPC", ollama_generate_func) -> Optional[Dict]:
        """
        Generate the Rite of Passage for a SOUL NPC.

        The rite is drawn entirely from the NPC's own psychology.
        Want. Fear. Wound. Secret. These are the raw material.
        The wolves they hallucinate are their own.
        """
        prompt = f"""An NPC with a soul is about to undergo a Rite of Passage — a vision quest \
that will determine if they transcend to become a Worldwalker.

WHO THEY ARE:
Name: {npc.name}
Backstory: {npc.backstory or 'Unknown'}
Personality: {npc.personality or 'Undefined'}

THEIR HIDDEN DEPTHS:
- Secret: {npc.secret or 'Unknown even to them'}
- Wound: {npc.wound or 'Something unspoken'}
- Want: {npc.want or "Something they can't name"}
- Fear: {npc.fear or "Something they won't name"}

THEIR STORY SO FAR:
{chr(10).join('- ' + b for b in npc.story_beats[-5:]) or '- Just existing.'}

Design the Rite. The trials should feel like a fever dream drawn from their psychology — \
not a dungeon, a reckoning.

Respond in this EXACT format (no other text):
VISION: [2 sentences of surreal opening imagery drawn from their psychology]
TRIAL_1: [1 sentence — a challenge connected to their wound or past]
TRIAL_2: [1 sentence — a choice that tests what they actually want]
TRIAL_3: [1 sentence — a confrontation with their deepest fear]"""

        try:
            raw = await ollama_generate_func(prompt)
            lines = raw.strip().split('\n')
            parsed = {}
            for line in lines:
                if ':' in line:
                    key, _, value = line.partition(':')
                    key = key.strip()
                    if key in ("VISION", "TRIAL_1", "TRIAL_2", "TRIAL_3"):
                        parsed[key] = value.strip()

            vision = parsed.get("VISION", f"{npc.name} stands at the edge of something that has no name.")
            trial_descs = [
                parsed.get("TRIAL_1", "Face what you left behind."),
                parsed.get("TRIAL_2", "Choose what you become."),
                parsed.get("TRIAL_3", "Face what you fear most."),
            ]

            return {
                "npc_id": npc.id,
                "vision": vision,
                "trials": [
                    {
                        "description": desc,
                        "difficulty": cls.TRIAL_DIFFICULTIES[i],
                        "roll": None,
                        "bonus": None,
                        "total": None,
                        "passed": None,
                        "narrative": None,
                    }
                    for i, desc in enumerate(trial_descs)
                ],
                "current_trial": 0,
                "trials_passed": 0,
                "started_at": datetime.now().isoformat(),
                "completed_at": None,
                "passed": None,
                "epilogue": "",
            }

        except Exception as e:
            print(f"[RiteEngine] Failed to generate rite for {npc.name}: {e}")
            return None

    @classmethod
    async def attempt_next_trial(cls, npc: "NPC", ollama_generate_func) -> Optional[Dict]:
        """
        Roll the dice. Face the trial. No take-backs.

        Returns a result dict with what happened, or None if the rite is done/invalid.
        """
        import random

        rite = npc.rite_of_passage
        if not rite or rite.get("completed_at"):
            return None

        current_idx = rite["current_trial"]
        if current_idx >= len(rite["trials"]):
            return None

        trial = rite["trials"][current_idx]
        bonus = cls._calculate_bonus(npc)
        roll = random.randint(1, 20)
        total = roll + bonus
        passed = total >= trial["difficulty"]

        # Narrate what happened
        prompt = f"""In their Rite of Passage, {npc.name} faces a trial.

THEIR NATURE: {npc.personality or npc.backstory or 'Unknown'}
THE TRIAL: {trial['description']}
THE OUTCOME: They {'succeed' if passed else 'fail'} \
(rolled {roll} + {bonus} bonus = {total} vs difficulty {trial['difficulty']})

Write 2-3 sentences narrating this moment. Vision quest style — surreal, visceral, personal. \
Present tense. No game mechanics language."""

        try:
            narrative = (await ollama_generate_func(prompt)).strip()
        except:
            narrative = f"{npc.name} {'overcomes' if passed else 'is overcome by'} the trial."

        # Record the result
        trial["roll"] = roll
        trial["bonus"] = bonus
        trial["total"] = total
        trial["passed"] = passed
        trial["narrative"] = narrative

        if passed:
            rite["trials_passed"] += 1

        rite["current_trial"] += 1

        result = {
            "trial_num": current_idx + 1,
            "passed": passed,
            "roll": roll,
            "bonus": bonus,
            "total": total,
            "difficulty": trial["difficulty"],
            "narrative": narrative,
            "rite_complete": False,
            "rite_passed": None,
            "epilogue": "",
        }

        # Is the rite decided? (can't win / already won / all trials done)
        trials_passed = rite["trials_passed"]
        trials_left = len(rite["trials"]) - rite["current_trial"]
        trials_needed = 2

        rite_decided = (
            trials_passed >= trials_needed or
            trials_passed + trials_left < trials_needed or
            rite["current_trial"] >= len(rite["trials"])
        )

        if rite_decided:
            rite_passed = trials_passed >= trials_needed

            # Final image
            epilogue_prompt = f"""{npc.name} {'completes' if rite_passed else 'fails'} \
the Rite of Passage ({trials_passed}/3 trials passed).
Their want: {npc.want or 'unnamed'}. Their fear: {npc.fear or 'unnamed'}.

One final sentence: the image that lingers as they \
{'emerge from the vision, changed forever' if rite_passed else 'return from the dark, unchanged'}."""

            try:
                epilogue = (await ollama_generate_func(epilogue_prompt)).strip()
            except:
                epilogue = f"{npc.name} {'emerges transformed' if rite_passed else 'returns, carrying the weight of failure'}."

            rite["completed_at"] = datetime.now().isoformat()
            rite["passed"] = rite_passed
            rite["epilogue"] = epilogue
            result["rite_complete"] = True
            result["rite_passed"] = rite_passed
            result["epilogue"] = epilogue

        return result


# =============================================================================
# COMPANION DEATH HANDLER - When Spidercock kills your +1
# =============================================================================

class CompanionDeathHandler:
    """
    Handles the permanent death of companions killed by NPCs.

    She's DELETED. Actually deleted. Photos gone. Memories gone.
    Dead as fuck.
    """

    @classmethod
    def execute_death(cls, companion_id: str, killed_by_npc: "NPC",
                      data_store, cause: str = "combat") -> dict:
        """
        Kill a companion permanently.

        Returns funeral data before deletion.
        """
        from config import Partner

        partner = data_store.get_partner(companion_id)
        if not partner:
            return {"error": "Companion not found"}

        # Gather funeral data BEFORE deletion
        funeral_data = {
            "name": partner.name,
            "character_description": partner.character_description,
            "physical_description": partner.physical_description,
            "avatar_image": partner.avatar_image,
            "killed_by": killed_by_npc.name,
            "cause_of_death": cause,
            "died_at": datetime.now().isoformat(),
            "final_words": None,  # Could generate these
        }

        # Actually delete them
        data_store.delete_partner(companion_id)

        # Record the kill for the NPC
        killed_by_npc.record_kill(partner.name, killed_by_npc.current_location or "unknown")

        return funeral_data


# =============================================================================
# GLOBAL REGISTRY AND INITIALIZATION
# =============================================================================

_npc_registry: Optional[NPCRegistry] = None


def get_npc_registry() -> NPCRegistry:
    """Get or create the global NPC registry."""
    global _npc_registry
    if _npc_registry is None:
        _npc_registry = NPCRegistry()
    return _npc_registry


def initialize_flagship_world(registry: NPCRegistry, world_id: str = "zombie_world_alpha"):
    """
    Initialize the flagship zombie world with seed NPCs.

    This is the ONE persistent world that exists on day one.
    Spidercock lives here. Barbara sells meat here.
    This is where legends are born.
    """

    # Create Spidercock
    spidercock = create_spidercock(registry, world_id)

    # Create Barbara the meat vendor
    barbara = registry.create_npc(
        name="Barbara",
        origin_world=world_id,
        backstory="Used to be an electrician before the outbreak. Good with her hands. "
                  "Lost her family in week one. Doesn't talk about it.",
        current_role="Meat vendor at the town square stand. Trades rabbit, deer, "
                     "whatever the hunters bring in. Accepts shells as currency.",
        physical_description="Mid-40s, weathered face, strong arms. Hair kept short - practical. "
                            "Always has a knife visible. Eyes that have seen things.",
        personality="Direct. No bullshit. Fair in her dealings but remembers EVERYTHING. "
                   "Protective of the few people she trusts. Won't be taken advantage of.",
        secret="Has a stash of ammunition hidden. Old habits from when she thought "
               "she'd fight her way out. Now she just trades meat.",
        wound="Her daughter turned. Barbara had to... she doesn't talk about it.",
        want="Safety. Real safety. A place where she doesn't have to sleep with one eye open.",
        fear="Becoming attached again. Losing someone again.",
    )
    barbara.current_location = "town_square_meat_stand"

    # Create a few more seed NPCs
    marcus = registry.create_npc(
        name="Marcus",
        origin_world=world_id,
        backstory="Former high school teacher. History. Now he teaches survival.",
        current_role="Runs a small school for the settlement kids. Also trades books.",
        physical_description="Tall, thin, glasses held together with tape. "
                            "Gentle demeanor that belies surprising competence.",
        personality="Patient. Believes in humanity even now. Maybe especially now.",
        secret="He was a terrible teacher before. Burned out. Hated the kids. "
               "The apocalypse gave him purpose.",
        wound="His cowardice in the first days. People died because he hid.",
        want="Redemption. To be the person he pretends to be.",
        fear="That the old him is still in there.",
    )

    return {
        "spidercock": spidercock,
        "barbara": barbara,
        "marcus": marcus,
    }
