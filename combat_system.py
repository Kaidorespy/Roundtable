"""
Pathfinder 2e Combat System for Companion Theater.

This is how Spidercock kills your companion for real.

Core Rules (from Archives of Nethys - 2e.aonprd.com):
- Attack Roll: d20 + modifier vs AC
- Critical Hit: Natural 20 OR beat AC by 10+ = double damage
- Critical Failure: Natural 1 OR miss AC by 10+
- Multiple Attack Penalty: -5 second attack, -10 third+

CHARACTER TYPES:
- Companions: FULL PLAYER CHARACTERS. Full stats, full progression, full sheets.
  They're not pets. They're not summons. They're PEOPLE.
- NPCs: Use simplified stat blocks based on role (vendor, guard, boss, etc.)
- Eidolons: If someone plays a Summoner, they get an actual Eidolon by the rules.
  That's a separate thing entirely.

DEATH:
- NPCs can kill companions permanently (deleted from database)
- Companion vs companion combat: TBD (unconscious? death saves?)
- NPC death: permanent, funeral button

All rolls happen behind the scenes. The DM sees everything.
Players see the narrative results.
"""

import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum
from datetime import datetime
import debug_logger as dbg


# =============================================================================
# DICE SYSTEM
# =============================================================================

class Dice:
    """Pathfinder dice roller. The bones don't lie."""

    @staticmethod
    def d20() -> int:
        return random.randint(1, 20)

    @staticmethod
    def d12() -> int:
        return random.randint(1, 12)

    @staticmethod
    def d10() -> int:
        return random.randint(1, 10)

    @staticmethod
    def d8() -> int:
        return random.randint(1, 8)

    @staticmethod
    def d6() -> int:
        return random.randint(1, 6)

    @staticmethod
    def d4() -> int:
        return random.randint(1, 4)

    @staticmethod
    def roll(notation: str) -> Tuple[int, List[int]]:
        """
        Roll dice using standard notation: "2d6+3", "1d20", "3d8-2"
        Returns (total, individual_rolls)
        """
        notation = notation.lower().replace(" ", "")

        # Parse modifier
        modifier = 0
        if "+" in notation:
            parts = notation.split("+")
            notation = parts[0]
            modifier = int(parts[1])
        elif "-" in notation:
            parts = notation.split("-")
            notation = parts[0]
            modifier = -int(parts[1])

        # Parse dice
        if "d" not in notation:
            return (int(notation) + modifier, [int(notation)])

        parts = notation.split("d")
        num_dice = int(parts[0]) if parts[0] else 1
        die_size = int(parts[1])

        rolls = [random.randint(1, die_size) for _ in range(num_dice)]
        total = sum(rolls) + modifier

        return (total, rolls)

    @staticmethod
    def check(dc: int, modifier: int = 0) -> Dict[str, Any]:
        """
        Make a d20 check against a DC.
        Returns full result with degree of success.
        """
        natural = Dice.d20()
        total = natural + modifier

        # Determine degree of success (PF2e rules)
        # Crit success: natural 20 OR beat DC by 10+
        # Crit failure: natural 1 OR miss DC by 10+
        # Success upgrades to crit on nat 20, downgrades on nat 1

        diff = total - dc

        if natural == 20:
            if diff >= 0:
                degree = "critical_success"
            else:
                degree = "success"  # Nat 20 upgrades failure to success
        elif natural == 1:
            if diff < 0:
                degree = "critical_failure"
            else:
                degree = "failure"  # Nat 1 downgrades success to failure
        elif diff >= 10:
            degree = "critical_success"
        elif diff >= 0:
            degree = "success"
        elif diff >= -10:
            degree = "failure"
        else:
            degree = "critical_failure"

        return {
            "natural": natural,
            "modifier": modifier,
            "total": total,
            "dc": dc,
            "degree": degree,
            "is_crit": degree == "critical_success",
            "is_crit_fail": degree == "critical_failure",
        }


# =============================================================================
# STAT BLOCKS
# =============================================================================

class Size(Enum):
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    HUGE = "huge"
    GARGANTUAN = "gargantuan"


@dataclass
class Attack:
    """A single attack option."""
    name: str
    attack_bonus: int
    damage: str           # Dice notation: "2d6+4"
    damage_type: str      # "slashing", "piercing", "bludgeoning", etc.
    traits: List[str] = field(default_factory=list)  # "finesse", "reach", etc.
    range: Optional[int] = None  # None = melee, number = ranged in feet


@dataclass
class CombatStats:
    """
    Combat statistics for any combatant (NPC, companion, player).
    Simplified Pathfinder 2e stat block.
    """

    # Identity
    name: str
    level: int = 1

    # Defenses
    ac: int = 10
    hp_max: int = 10
    hp_current: int = 10

    # Saves
    fortitude: int = 0
    reflex: int = 0
    will: int = 0

    # Offense
    attacks: List[Attack] = field(default_factory=list)

    # Movement
    speed: int = 25  # feet
    size: Size = Size.MEDIUM

    # Special
    immunities: List[str] = field(default_factory=list)
    resistances: Dict[str, int] = field(default_factory=dict)  # type -> amount
    weaknesses: Dict[str, int] = field(default_factory=dict)   # type -> amount

    # Status
    conditions: List[str] = field(default_factory=list)
    is_alive: bool = True

    def take_damage(self, amount: int, damage_type: str = "untyped") -> Dict[str, Any]:
        """
        Apply damage, accounting for resistances and weaknesses.
        Returns damage report.
        """
        if not self.is_alive:
            return {"error": "Already dead", "damage_dealt": 0}

        # Check immunity
        if damage_type in self.immunities:
            return {
                "original_damage": amount,
                "damage_dealt": 0,
                "reason": f"Immune to {damage_type}",
                "hp_remaining": self.hp_current,
            }

        modified = amount

        # Apply resistance
        if damage_type in self.resistances:
            reduction = min(self.resistances[damage_type], modified)
            modified -= reduction

        # Apply weakness
        if damage_type in self.weaknesses:
            modified += self.weaknesses[damage_type]

        # Apply damage
        modified = max(0, modified)
        self.hp_current -= modified

        result = {
            "original_damage": amount,
            "damage_type": damage_type,
            "damage_dealt": modified,
            "hp_remaining": self.hp_current,
            "hp_max": self.hp_max,
        }

        # Check for death
        if self.hp_current <= 0:
            self.is_alive = False
            self.hp_current = 0
            result["died"] = True
            result["overkill"] = abs(self.hp_current) if self.hp_current < 0 else 0

        return result

    def heal(self, amount: int) -> Dict[str, Any]:
        """Heal HP, capped at max."""
        if not self.is_alive:
            return {"error": "Cannot heal the dead", "healed": 0}

        old_hp = self.hp_current
        self.hp_current = min(self.hp_max, self.hp_current + amount)
        healed = self.hp_current - old_hp

        return {
            "healed": healed,
            "hp_current": self.hp_current,
            "hp_max": self.hp_max,
        }

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "level": self.level,
            "ac": self.ac,
            "hp_current": self.hp_current,
            "hp_max": self.hp_max,
            "fortitude": self.fortitude,
            "reflex": self.reflex,
            "will": self.will,
            "speed": self.speed,
            "size": self.size.value,
            "attacks": [
                {
                    "name": a.name,
                    "attack_bonus": a.attack_bonus,
                    "damage": a.damage,
                    "damage_type": a.damage_type,
                    "traits": a.traits,
                    "range": a.range,
                }
                for a in self.attacks
            ],
            "immunities": self.immunities,
            "resistances": self.resistances,
            "weaknesses": self.weaknesses,
            "conditions": self.conditions,
            "is_alive": self.is_alive,
        }


# =============================================================================
# COMBAT RESOLUTION
# =============================================================================

class RollVisibility(Enum):
    """Who can see dice rolls."""
    PUBLIC = "public"       # Everyone sees the roll
    DM_ONLY = "dm_only"     # Only DM sees numbers, players see narrative
    HIDDEN = "hidden"       # No one sees (for secret checks)


@dataclass
class AttackResult:
    """Result of a single attack."""
    attacker: str
    defender: str
    attack_name: str

    # Roll details
    natural_roll: int
    attack_bonus: int
    total_attack: int
    target_ac: int

    # Outcome
    degree: str  # "critical_success", "success", "failure", "critical_failure"
    hit: bool
    crit: bool

    # Damage (if hit)
    damage_rolls: List[int] = field(default_factory=list)
    base_damage: int = 0
    final_damage: int = 0  # After crit multiplier, resistances, etc.
    damage_type: str = ""

    # Consequences
    defender_hp_remaining: int = 0
    defender_died: bool = False

    # Visibility
    visibility: RollVisibility = RollVisibility.PUBLIC

    def to_narrative(self, include_numbers: bool = True) -> str:
        """Generate narrative description of the attack."""
        if self.degree == "critical_failure":
            return f"{self.attacker} swings wildly with {self.attack_name}, completely missing {self.defender}."
        elif self.degree == "failure":
            return f"{self.attacker} attacks {self.defender} with {self.attack_name}, but the blow fails to connect."
        elif self.degree == "success":
            if include_numbers:
                result = f"{self.attacker} strikes {self.defender} with {self.attack_name} for {self.final_damage} {self.damage_type} damage!"
            else:
                result = f"{self.attacker} strikes {self.defender} with {self.attack_name}!"
            if self.defender_died:
                result += f" {self.defender} falls!"
            return result
        else:  # critical_success
            if include_numbers:
                result = f"CRITICAL HIT! {self.attacker} devastates {self.defender} with {self.attack_name} for {self.final_damage} {self.damage_type} damage!"
            else:
                result = f"CRITICAL HIT! {self.attacker} devastates {self.defender} with {self.attack_name}!"
            if self.defender_died:
                result += f" {self.defender} is DESTROYED!"
            return result

    def to_player_view(self) -> Dict[str, Any]:
        """Get player-facing view (no numbers if DM-only visibility)."""
        if self.visibility == RollVisibility.PUBLIC:
            return {
                "narrative": self.to_narrative(include_numbers=True),
                "hit": self.hit,
                "crit": self.crit,
                "damage": self.final_damage,
                "defender_died": self.defender_died,
            }
        else:
            # DM_ONLY or HIDDEN - players just see narrative
            return {
                "narrative": self.to_narrative(include_numbers=False),
                "hit": self.hit,
                "crit": self.crit,
                "defender_died": self.defender_died,
            }

    def to_dm_view(self) -> Dict[str, Any]:
        """Get full DM view with all numbers."""
        return {
            "narrative": self.to_narrative(include_numbers=True),
            "natural_roll": self.natural_roll,
            "attack_bonus": self.attack_bonus,
            "total_attack": self.total_attack,
            "target_ac": self.target_ac,
            "degree": self.degree,
            "hit": self.hit,
            "crit": self.crit,
            "damage_rolls": self.damage_rolls,
            "base_damage": self.base_damage,
            "final_damage": self.final_damage,
            "damage_type": self.damage_type,
            "defender_hp_remaining": self.defender_hp_remaining,
            "defender_died": self.defender_died,
            "visibility": self.visibility.value,
        }


class CombatResolver:
    """
    Resolves combat using Pathfinder 2e rules.

    This is where the dice hit the table.
    """

    @staticmethod
    def attack(
        attacker: CombatStats,
        defender: CombatStats,
        attack_index: int = 0,
        map_penalty: int = 0,  # Multiple Attack Penalty: 0, -5, or -10
    ) -> AttackResult:
        """
        Resolve a single attack.

        map_penalty: Multiple Attack Penalty
            - First attack: 0
            - Second attack: -5
            - Third+ attack: -10
        """
        if not attacker.attacks:
            raise ValueError(f"{attacker.name} has no attacks!")

        attack = attacker.attacks[min(attack_index, len(attacker.attacks) - 1)]

        # Roll the attack
        natural = Dice.d20()
        total_bonus = attack.attack_bonus + map_penalty
        total_attack = natural + total_bonus

        # Determine degree of success
        diff = total_attack - defender.ac

        if natural == 20:
            degree = "critical_success" if diff >= -10 else "success"
        elif natural == 1:
            degree = "critical_failure" if diff < 10 else "failure"
        elif diff >= 10:
            degree = "critical_success"
        elif diff >= 0:
            degree = "success"
        elif diff >= -10:
            degree = "failure"
        else:
            degree = "critical_failure"

        hit = degree in ["success", "critical_success"]
        crit = degree == "critical_success"

        # Roll damage if hit
        damage_rolls = []
        base_damage = 0
        final_damage = 0

        if hit:
            base_damage, damage_rolls = Dice.roll(attack.damage)
            final_damage = base_damage

            # Critical doubles damage
            if crit:
                final_damage = base_damage * 2

            # Apply to defender
            damage_result = defender.take_damage(final_damage, attack.damage_type)
            final_damage = damage_result["damage_dealt"]

        return AttackResult(
            attacker=attacker.name,
            defender=defender.name,
            attack_name=attack.name,
            natural_roll=natural,
            attack_bonus=total_bonus,
            total_attack=total_attack,
            target_ac=defender.ac,
            degree=degree,
            hit=hit,
            crit=crit,
            damage_rolls=damage_rolls,
            base_damage=base_damage,
            final_damage=final_damage,
            damage_type=attack.damage_type,
            defender_hp_remaining=defender.hp_current,
            defender_died=not defender.is_alive,
        )

    @staticmethod
    def full_attack(
        attacker: CombatStats,
        defender: CombatStats,
        num_attacks: int = 1,
    ) -> List[AttackResult]:
        """
        Execute multiple attacks with MAP (Multiple Attack Penalty).

        PF2e rule: -5 on second attack, -10 on third+
        """
        results = []

        for i in range(num_attacks):
            if i == 0:
                map_penalty = 0
            elif i == 1:
                map_penalty = -5
            else:
                map_penalty = -10

            # Check if defender is still alive
            if not defender.is_alive:
                break

            result = CombatResolver.attack(attacker, defender, map_penalty=map_penalty)
            results.append(result)

        return results

    @staticmethod
    def saving_throw(
        target: CombatStats,
        save_type: str,  # "fortitude", "reflex", "will"
        dc: int,
    ) -> Dict[str, Any]:
        """
        Make a saving throw.
        """
        save_bonus = getattr(target, save_type, 0)
        result = Dice.check(dc, save_bonus)
        result["save_type"] = save_type
        result["target"] = target.name
        return result


# =============================================================================
# MONSTER/NPC STAT BLOCK TEMPLATES
# =============================================================================

def create_spidercock_stats() -> CombatStats:
    """
    Spidercock's combat stats.

    Level 8 creature. Boss monster. Point of interest guardian.
    Multiple attacks because... well, multiple appendages.
    """
    return CombatStats(
        name="Spidercock",
        level=8,
        ac=26,
        hp_max=155,
        hp_current=155,
        fortitude=17,
        reflex=14,
        will=12,
        speed=35,
        size=Size.LARGE,
        attacks=[
            Attack(
                name="Grotesque Appendage",
                attack_bonus=19,
                damage="2d10+8",
                damage_type="bludgeoning",
                traits=["reach"],
            ),
            Attack(
                name="Venomous Bite",
                attack_bonus=19,
                damage="2d8+6",
                damage_type="piercing",
                traits=["poison"],
            ),
            Attack(
                name="Constrict",
                attack_bonus=17,
                damage="2d6+8",
                damage_type="bludgeoning",
                traits=["grab"],
            ),
        ],
        weaknesses={"fire": 10},  # "Fire. The old weakness, still there."
        immunities=["fear", "mental"],
    )


def create_companion_stats(
    name: str,
    level: int = 1,
    archetype: str = "balanced",
) -> CombatStats:
    """
    PROTOTYPE: Quick-start combat stats for a companion.

    TODO: Companions are FULL PLAYER CHARACTERS. They need:
    - Ancestry (human, elf, etc.) with ancestry feats
    - Class (fighter, rogue, wizard, etc.) with class features
    - Background
    - Full ability scores: Str, Dex, Con, Int, Wis, Cha
    - Skills with proficiency ranks
    - Feats (general, skill, class, ancestry)
    - Equipment and inventory
    - Spells (if caster)

    This function is a PLACEHOLDER for quick testing.
    Real implementation needs a full CharacterSheet class.

    Quick-start archetypes (for prototyping only):
    - "tank": High AC and HP, lower offense
    - "striker": High offense, lower defense
    - "balanced": Middle of the road
    - "support": Lower combat stats
    """
    base_hp = 8 + (level * 8)
    base_ac = 10 + level
    base_attack = level + 2

    if archetype == "tank":
        return CombatStats(
            name=name,
            level=level,
            ac=base_ac + 4,
            hp_max=base_hp + (level * 4),
            hp_current=base_hp + (level * 4),
            fortitude=level + 4,
            reflex=level + 1,
            will=level + 2,
            attacks=[
                Attack(
                    name="Shield Bash",
                    attack_bonus=base_attack,
                    damage=f"1d6+{level // 2 + 2}",
                    damage_type="bludgeoning",
                ),
            ],
        )
    elif archetype == "striker":
        return CombatStats(
            name=name,
            level=level,
            ac=base_ac,
            hp_max=base_hp,
            hp_current=base_hp,
            fortitude=level + 2,
            reflex=level + 4,
            will=level + 1,
            attacks=[
                Attack(
                    name="Twin Blades",
                    attack_bonus=base_attack + 2,
                    damage=f"2d6+{level // 2 + 3}",
                    damage_type="slashing",
                    traits=["finesse", "twin"],
                ),
            ],
        )
    else:  # balanced
        return CombatStats(
            name=name,
            level=level,
            ac=base_ac + 2,
            hp_max=base_hp + (level * 2),
            hp_current=base_hp + (level * 2),
            fortitude=level + 2,
            reflex=level + 2,
            will=level + 2,
            attacks=[
                Attack(
                    name="Longsword",
                    attack_bonus=base_attack + 1,
                    damage=f"1d8+{level // 2 + 2}",
                    damage_type="slashing",
                ),
            ],
        )


def create_npc_combatant(
    name: str,
    role: str,
    level: int = 1,
) -> CombatStats:
    """
    Create combat stats for an NPC based on their role.

    Roles map to rough stat distributions:
    - "vendor": Weak, non-combatant
    - "guard": Moderate, defensive
    - "bandit": Moderate, offensive
    - "boss": Strong, multiple attacks
    - "civilian": Very weak
    """
    if role == "vendor":
        return CombatStats(
            name=name,
            level=max(1, level - 2),
            ac=12,
            hp_max=15,
            hp_current=15,
            fortitude=2,
            reflex=1,
            will=3,
            attacks=[
                Attack(
                    name="Kitchen Knife",
                    attack_bonus=2,
                    damage="1d4",
                    damage_type="slashing",
                ),
            ],
        )
    elif role == "guard":
        return CombatStats(
            name=name,
            level=level,
            ac=16 + level,
            hp_max=20 + (level * 10),
            hp_current=20 + (level * 10),
            fortitude=level + 3,
            reflex=level + 1,
            will=level + 2,
            attacks=[
                Attack(
                    name="Spear",
                    attack_bonus=level + 4,
                    damage=f"1d8+{level + 2}",
                    damage_type="piercing",
                    traits=["reach"],
                ),
            ],
        )
    elif role == "bandit":
        return CombatStats(
            name=name,
            level=level,
            ac=14 + level,
            hp_max=15 + (level * 8),
            hp_current=15 + (level * 8),
            fortitude=level + 1,
            reflex=level + 3,
            will=level + 1,
            attacks=[
                Attack(
                    name="Rusty Sword",
                    attack_bonus=level + 5,
                    damage=f"1d6+{level + 3}",
                    damage_type="slashing",
                ),
                Attack(
                    name="Dagger Throw",
                    attack_bonus=level + 4,
                    damage=f"1d4+{level}",
                    damage_type="piercing",
                    range=20,
                ),
            ],
        )
    elif role == "boss":
        return CombatStats(
            name=name,
            level=level + 2,
            ac=18 + level,
            hp_max=40 + (level * 15),
            hp_current=40 + (level * 15),
            fortitude=level + 4,
            reflex=level + 3,
            will=level + 3,
            attacks=[
                Attack(
                    name="Greataxe",
                    attack_bonus=level + 7,
                    damage=f"1d12+{level + 5}",
                    damage_type="slashing",
                ),
                Attack(
                    name="Backhand",
                    attack_bonus=level + 5,
                    damage=f"1d6+{level + 3}",
                    damage_type="bludgeoning",
                ),
            ],
        )
    else:  # civilian
        return CombatStats(
            name=name,
            level=0,
            ac=10,
            hp_max=6,
            hp_current=6,
            fortitude=0,
            reflex=1,
            will=1,
            attacks=[
                Attack(
                    name="Fists",
                    attack_bonus=0,
                    damage="1d4-1",
                    damage_type="bludgeoning",
                ),
            ],
        )


# =============================================================================
# COMBAT ENCOUNTER
# =============================================================================

@dataclass
class Combatant:
    """A participant in combat."""
    id: str
    stats: CombatStats
    initiative: int = 0
    team: str = "neutral"  # "players", "enemies", "neutral"
    is_companion: bool = False
    companion_id: Optional[str] = None  # Link to actual companion if applicable
    is_npc: bool = False
    npc_id: Optional[str] = None  # Link to NPC system


class CombatEncounter:
    """
    Manages a combat encounter.

    Initiative, turn order, round tracking.
    The whole bloody business.
    """

    def __init__(self, room_id: str):
        self.room_id = room_id
        self.combatants: Dict[str, Combatant] = {}
        self.turn_order: List[str] = []
        self.current_turn: int = 0
        self.round: int = 0
        self.is_active: bool = False
        self.combat_log: List[Dict] = []
        self.started_at: Optional[str] = None
        self.ended_at: Optional[str] = None

        # Team-based turn system
        self.active_side: str = "players"  # "players" or "enemies"
        self.player_turns_max: int = 0
        self.enemy_turns_max: int = 0
        self.player_turns_remaining: int = 0
        self.enemy_turns_remaining: int = 0
        self.actions_since_switch: int = 0  # Track actions for interjection chance

    def add_combatant(
        self,
        id: str,
        stats: CombatStats,
        team: str = "neutral",
        initiative_bonus: int = 0,
        is_companion: bool = False,
        companion_id: str = None,
        is_npc: bool = False,
        npc_id: str = None,
    ) -> Combatant:
        """Add a combatant to the encounter."""
        # Roll initiative
        init_roll = Dice.d20() + initiative_bonus

        combatant = Combatant(
            id=id,
            stats=stats,
            initiative=init_roll,
            team=team,
            is_companion=is_companion,
            companion_id=companion_id,
            is_npc=is_npc,
            npc_id=npc_id,
        )

        self.combatants[id] = combatant
        self._update_turn_order()

        return combatant

    def _update_turn_order(self):
        """Sort combatants by initiative (highest first)."""
        self.turn_order = sorted(
            self.combatants.keys(),
            key=lambda x: self.combatants[x].initiative,
            reverse=True,
        )

    def start_combat(self, initiator_side: str = "players"):
        """Begin the encounter."""
        self.is_active = True
        self.round = 1
        self.current_turn = 0
        self.started_at = datetime.now().isoformat()

        # Count combatants per side and set up team turns
        players = [c for c in self.combatants.values() if c.team == "players" and c.stats.is_alive]
        enemies = [c for c in self.combatants.values() if c.team == "enemies" and c.stats.is_alive]

        self.player_turns_max = len(players)
        self.enemy_turns_max = len(enemies)
        self.player_turns_remaining = self.player_turns_max
        self.enemy_turns_remaining = self.enemy_turns_max

        # Initiator's side goes first
        self.active_side = initiator_side
        self.actions_since_switch = 0

        combatant_names = [self.combatants[c_id].stats.name for c_id in self.turn_order]
        dbg.combat(f"▶ COMBAT START in {self.room_id}: {', '.join(combatant_names)}")
        dbg.combat(f"  Players: {self.player_turns_max} turns | Enemies: {self.enemy_turns_max} turns | {initiator_side} go first")

        self.combat_log.append({
            "type": "combat_start",
            "round": 1,
            "turn_order": [
                {"id": c_id, "name": self.combatants[c_id].stats.name, "initiative": self.combatants[c_id].initiative}
                for c_id in self.turn_order
            ],
            "player_turns": self.player_turns_max,
            "enemy_turns": self.enemy_turns_max,
            "active_side": self.active_side,
            "timestamp": self.started_at,
        })

    def record_action(self, side: str = None) -> dict:
        """
        Record that an action was taken by the active side.
        Returns info about the combat state after this action.
        """
        if side is None:
            side = self.active_side

        # Decrement the active side's remaining turns
        if side == "players":
            self.player_turns_remaining = max(0, self.player_turns_remaining - 1)
        else:
            self.enemy_turns_remaining = max(0, self.enemy_turns_remaining - 1)

        self.actions_since_switch += 1

        remaining = self.player_turns_remaining if side == "players" else self.enemy_turns_remaining

        dbg.combat(f"  Action recorded for {side}: {remaining} turns remaining (actions since switch: {self.actions_since_switch})")

        return {
            "side": side,
            "turns_remaining": remaining,
            "actions_since_switch": self.actions_since_switch,
            "should_auto_switch": remaining == 0,
        }

    def switch_sides(self, reason: str = "interjection") -> str:
        """
        Switch active side. Called by DM interjection or when turns exhaust.
        Returns the new active side.
        """
        old_side = self.active_side
        self.active_side = "enemies" if old_side == "players" else "players"
        self.actions_since_switch = 0

        dbg.combat(f"  ⚡ SIDE SWITCH ({reason}): {old_side} → {self.active_side}")

        # Check if both sides have exhausted turns (new round)
        if self.player_turns_remaining == 0 and self.enemy_turns_remaining == 0:
            self._start_new_round()

        self.combat_log.append({
            "type": "side_switch",
            "round": self.round,
            "from_side": old_side,
            "to_side": self.active_side,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })

        return self.active_side

    def _start_new_round(self):
        """Reset turns for a new combat round."""
        self.round += 1
        self.player_turns_remaining = self.player_turns_max
        self.enemy_turns_remaining = self.enemy_turns_max

        dbg.combat(f"  🔄 NEW ROUND {self.round}: Players {self.player_turns_max} | Enemies {self.enemy_turns_max}")

        self.combat_log.append({
            "type": "new_round",
            "round": self.round,
            "player_turns": self.player_turns_max,
            "enemy_turns": self.enemy_turns_max,
            "timestamp": datetime.now().isoformat(),
        })

    def get_combat_state(self) -> dict:
        """Get current combat state for UI/decision making."""
        return {
            "active": self.is_active,
            "round": self.round,
            "active_side": self.active_side,
            "player_turns_remaining": self.player_turns_remaining,
            "enemy_turns_remaining": self.enemy_turns_remaining,
            "player_turns_max": self.player_turns_max,
            "enemy_turns_max": self.enemy_turns_max,
            "actions_since_switch": self.actions_since_switch,
        }

    def get_current_combatant(self) -> Optional[Combatant]:
        """Get whose turn it is."""
        if not self.turn_order:
            return None
        return self.combatants.get(self.turn_order[self.current_turn])

    def next_turn(self) -> Combatant:
        """Advance to the next combatant's turn."""
        # Skip dead combatants
        while True:
            self.current_turn += 1

            if self.current_turn >= len(self.turn_order):
                self.current_turn = 0
                self.round += 1
                self.combat_log.append({
                    "type": "new_round",
                    "round": self.round,
                    "timestamp": datetime.now().isoformat(),
                })

            current = self.get_current_combatant()
            if current and current.stats.is_alive:
                return current

            # Prevent infinite loop if everyone's dead
            if not any(c.stats.is_alive for c in self.combatants.values()):
                self.end_combat()
                return None

    def execute_attack(
        self,
        attacker_id: str,
        defender_id: str,
        attack_index: int = 0,
        map_penalty: int = 0,
    ) -> AttackResult:
        """Execute an attack between combatants."""
        attacker = self.combatants.get(attacker_id)
        defender = self.combatants.get(defender_id)

        if not attacker or not defender:
            raise ValueError("Invalid attacker or defender")

        result = CombatResolver.attack(
            attacker.stats,
            defender.stats,
            attack_index,
            map_penalty,
        )

        self.combat_log.append({
            "type": "attack",
            "round": self.round,
            "attacker": attacker_id,
            "defender": defender_id,
            "result": {
                "hit": result.hit,
                "crit": result.crit,
                "damage": result.final_damage,
                "natural_roll": result.natural_roll,
                "total": result.total_attack,
                "ac": result.target_ac,
                "defender_hp": result.defender_hp_remaining,
                "died": result.defender_died,
            },
            "narrative": result.to_narrative(),
            "timestamp": datetime.now().isoformat(),
        })

        return result

    def end_combat(self, reason: str = "resolved"):
        """End the encounter."""
        self.is_active = False
        self.ended_at = datetime.now().isoformat()

        # Determine winners
        surviving_teams = set()
        for c in self.combatants.values():
            if c.stats.is_alive:
                surviving_teams.add(c.team)

        casualties = [c.stats.name for c in self.combatants.values() if not c.stats.is_alive]
        dbg.combat(f"■ COMBAT END: {reason} after {self.round} rounds | Casualties: {casualties or 'none'}")

        self.combat_log.append({
            "type": "combat_end",
            "reason": reason,
            "rounds": self.round,
            "surviving_teams": list(surviving_teams),
            "casualties": casualties,
            "timestamp": self.ended_at,
        })

    def get_casualties(self) -> List[Combatant]:
        """Get all dead combatants."""
        return [c for c in self.combatants.values() if not c.stats.is_alive]

    def to_dict(self) -> Dict:
        """Serialize the encounter."""
        return {
            "room_id": self.room_id,
            "is_active": self.is_active,
            "round": self.round,
            "current_turn": self.current_turn,
            "turn_order": self.turn_order,
            "combatants": {
                c_id: {
                    "id": c.id,
                    "stats": c.stats.to_dict(),
                    "initiative": c.initiative,
                    "team": c.team,
                    "is_companion": c.is_companion,
                    "companion_id": c.companion_id,
                    "is_npc": c.is_npc,
                    "npc_id": c.npc_id,
                }
                for c_id, c in self.combatants.items()
            },
            "combat_log": self.combat_log,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


# =============================================================================
# ENCOUNTER MANAGER
# =============================================================================

class EncounterManager:
    """
    Manages all combat encounters across rooms.
    """

    _encounters: Dict[str, CombatEncounter] = {}

    @classmethod
    def get_or_create(cls, room_id: str) -> CombatEncounter:
        """Get or create an encounter for a room."""
        if room_id not in cls._encounters:
            cls._encounters[room_id] = CombatEncounter(room_id)
        return cls._encounters[room_id]

    @classmethod
    def get(cls, room_id: str) -> Optional[CombatEncounter]:
        """Get an existing encounter."""
        return cls._encounters.get(room_id)

    @classmethod
    def clear(cls, room_id: str):
        """Clear an encounter."""
        if room_id in cls._encounters:
            del cls._encounters[room_id]

    @classmethod
    def has_active_combat(cls, room_id: str) -> bool:
        """Check if a room has active combat."""
        enc = cls._encounters.get(room_id)
        return enc.is_active if enc else False


# =============================================================================
# COMPANION DEATH INTEGRATION
# =============================================================================

def process_companion_death_in_combat(
    encounter: CombatEncounter,
    dead_combatant: Combatant,
    killer_combatant: Combatant,
    npc_registry,
    data_store,
    room=None,
) -> Dict[str, Any]:
    """
    Process what happens when a companion dies in combat.

    If hardcore_mode is ON and killed by NPC: companion is ACTUALLY DELETED.
    If hardcore_mode is OFF: companion is just "knocked out" (incapacitated).
    If killed by another player's companion: just unconscious regardless.
    """
    from npc_system import CompanionDeathHandler

    result = {
        "companion_id": dead_combatant.companion_id,
        "companion_name": dead_combatant.stats.name,
        "killed_by": killer_combatant.stats.name,
        "permanent_death": False,
        "knocked_out": False,
        "funeral_data": None,
    }

    # Check if hardcore mode is enabled
    hardcore_mode = room.hardcore_mode if room else False

    # If killer is an NPC, check hardcore mode
    if killer_combatant.is_npc and killer_combatant.npc_id:
        npc = npc_registry.get_npc(killer_combatant.npc_id)
        if npc:
            if hardcore_mode:
                # PERMANENT DEATH - delete the companion
                funeral_data = CompanionDeathHandler.execute_death(
                    dead_combatant.companion_id,
                    npc,
                    data_store,
                    cause=f"Killed in combat by {npc.name}",
                )
                result["permanent_death"] = True
                result["funeral_data"] = funeral_data
            else:
                # Just knocked out - they can return later
                result["knocked_out"] = True
    else:
        # PvP or unknown - just knocked out
        result["knocked_out"] = True

    return result
