"""
Action Resolver - Systems decide, DM narrates.

"I fire my gun" → System resolves → DM narrates the result.

This module classifies player actions and pre-resolves mechanical outcomes
BEFORE asking the DM. The DM's job becomes making it feel true, not
deciding if it's true.

Action Types:
- QUERY: "What do I see?" → DM has full authority
- COMBAT: "I attack/shoot/stab" → System resolves, DM narrates
- INVENTORY: "I pick up/use/drop" → System resolves, DM narrates
- SKILL: "I try to hotwire" → System checks skill + resolves, DM narrates
- MOVEMENT: "I go to/travel" → System checks possibility, DM narrates
- SOCIAL: "I ask/persuade" → DM has authority (with NPC constraints)
"""

import re
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import debug_logger as dbg


class ActionType(Enum):
    QUERY = "query"           # DM has full authority
    COMBAT = "combat"         # System resolves
    INVENTORY = "inventory"   # System resolves
    SKILL = "skill"           # System resolves
    MOVEMENT = "movement"     # System resolves
    SOCIAL = "social"         # DM has authority
    UNKNOWN = "unknown"       # Default to DM


@dataclass
class ResolvedAction:
    """Result of pre-resolving an action."""
    action_type: ActionType
    success: bool
    outcome_description: str      # What happened (for DM to narrate)
    items_consumed: List[str]     # Items used up
    items_gained: List[str]       # Items acquired
    consequences: List[str]       # Side effects (sounds, alerts, etc.)
    fatigue_modifier: str         # How fatigue affected this
    skill_used: Optional[str]     # What skill was relevant
    dm_instruction: str           # What to tell the DM
    raw_resolution: Dict[str, Any]  # Full resolution data


# Pattern matching for action classification
ACTION_PATTERNS = {
    ActionType.COMBAT: [
        r'\b(attack|shoot|fire|stab|slash|hit|strike|punch|kick|fight|kill|murder)\b',
        r'\b(swing|throw|hurl|aim|pull the trigger)\b',
        r'\bI (attack|shoot|fire|stab|hit|fight)\b',
    ],
    ActionType.INVENTORY: [
        r'\b(pick up|grab|take|get|collect|loot|pocket|steal)\b',
        r'\b(drop|discard|throw away|put down|leave behind)\b',
        r'\b(use|consume|eat|drink|apply|activate)\b',
        r'\b(give|hand|offer|trade|exchange)\b',
    ],
    ActionType.SKILL: [
        r'\b(try to|attempt to|I try|I attempt)\b',
        r'\b(hotwire|lockpick|pick the lock|hack|repair|fix|heal|treat|bandage)\b',
        r'\b(sneak|hide|climb|swim|jump|dodge)\b',
        r'\b(search|examine|inspect|investigate|look for)\b',
    ],
    ActionType.MOVEMENT: [
        r'\b(go to|travel to|head to|walk to|run to|move to)\b',
        r'\b(leave|exit|enter|approach|flee|escape|retreat)\b',
        r'\b(I go|I travel|I head|I walk|I run)\b',
    ],
    ActionType.SOCIAL: [
        r'\b(talk to|speak to|ask|tell|say to|convince|persuade|intimidate)\b',
        r'\b(negotiate|barter|trade with|bribe|threaten|lie to)\b',
        r'\b(I ask|I tell|I say|I convince)\b',
    ],
    ActionType.QUERY: [
        r'\bwhat (do I|can I|is|are|does)\b',
        r'\bwhere (is|are|am I|can I)\b',
        r'\bhow (do I|can I|does|is)\b',
        r'\bwho (is|are|can)\b',
        r'\bcan I see\b',
        r'\bdo I (see|hear|notice|know|remember)\b',
        r'\bdescribe\b',
        r'\blook around\b',
    ],
}


def classify_action(question: str) -> ActionType:
    """
    Classify what type of action the player is attempting.
    """
    question_lower = question.lower()

    # Check patterns in priority order
    # Queries first (so "what do I see if I attack" → QUERY)
    for action_type in [ActionType.QUERY, ActionType.COMBAT, ActionType.INVENTORY,
                        ActionType.SKILL, ActionType.MOVEMENT, ActionType.SOCIAL]:
        patterns = ACTION_PATTERNS.get(action_type, [])
        for pattern in patterns:
            if re.search(pattern, question_lower):
                return action_type

    return ActionType.UNKNOWN


def extract_target_item(question: str, player_inventory: List[str]) -> Optional[str]:
    """
    Extract what item the player is trying to use from their question.
    Returns the item if found in inventory, None otherwise.
    """
    question_lower = question.lower()

    # Check each inventory item
    for item in player_inventory:
        if item.lower() in question_lower:
            return item

    # Common weapon words
    weapon_words = {
        'gun': ['gun', 'pistol', 'revolver', 'firearm'],
        'rifle': ['rifle', 'shotgun', 'carbine'],
        'knife': ['knife', 'blade', 'dagger'],
        'sword': ['sword', 'blade'],
    }

    for item in player_inventory:
        item_lower = item.lower()
        for base, variants in weapon_words.items():
            if any(v in item_lower for v in variants):
                if any(v in question_lower for v in variants):
                    return item

    return None


def resolve_combat_action(
    question: str,
    player_inventory: List[str],
    player_skills: List[str],
    fatigue_state: str,
    is_alive: bool
) -> ResolvedAction:
    """
    Pre-resolve a combat action.
    """
    if not is_alive:
        return ResolvedAction(
            action_type=ActionType.COMBAT,
            success=False,
            outcome_description="Player is dead and cannot take actions.",
            items_consumed=[],
            items_gained=[],
            consequences=[],
            fatigue_modifier="N/A - dead",
            skill_used=None,
            dm_instruction="The player is dead. Describe the stillness, the absence.",
            raw_resolution={"reason": "player_dead"}
        )

    # Find what weapon they're using
    weapon = extract_target_item(question, player_inventory)

    consequences = []
    items_consumed = []
    fatigue_mod = ""

    # Fatigue effects
    if fatigue_state == "exhausted":
        fatigue_mod = "EXHAUSTED: Significant penalty, movements sluggish"
        success_chance = 0.4
    elif fatigue_state == "tired":
        fatigue_mod = "TIRED: Minor penalty"
        success_chance = 0.7
    else:
        fatigue_mod = "Well-rested"
        success_chance = 0.85

    # Check for ranged weapon (gun)
    is_ranged = weapon and any(w in weapon.lower() for w in ['gun', 'pistol', 'rifle', 'bow'])

    if is_ranged:
        # Check for ammo
        has_ammo = any('ammo' in item.lower() or 'bullet' in item.lower() or 'arrow' in item.lower()
                       for item in player_inventory)

        if not has_ammo and weapon:
            return ResolvedAction(
                action_type=ActionType.COMBAT,
                success=False,
                outcome_description=f"Player tries to fire {weapon} but has no ammunition.",
                items_consumed=[],
                items_gained=[],
                consequences=[],
                fatigue_modifier=fatigue_mod,
                skill_used=None,
                dm_instruction=f"The player pulled the trigger on their {weapon}, but it clicked empty. No ammo. Narrate their realization.",
                raw_resolution={"reason": "no_ammo", "weapon": weapon}
            )

        # Gunshot creates sound consequence
        consequences.append("SOUND:gunshot - loud noise echoes, may attract attention")
        items_consumed.append("1 ammunition")

    # Resolve success
    roll = random.random()
    success = roll < success_chance

    if weapon:
        if success:
            outcome = f"Player attacks with {weapon} and HITS."
            dm_instruction = f"The player struck with their {weapon} and connected. Describe the impact, the result. They succeeded."
        else:
            outcome = f"Player attacks with {weapon} but MISSES."
            dm_instruction = f"The player swung/fired their {weapon} but missed. Describe the near-miss, the frustration, the danger of the failed attempt."
    else:
        # Unarmed
        if success:
            outcome = "Player attacks unarmed and lands a blow."
            dm_instruction = "The player threw a punch/kick and connected. Describe the physical impact. They don't have a weapon."
        else:
            outcome = "Player attacks unarmed but fails to connect."
            dm_instruction = "The player's unarmed attack missed or was blocked. Describe the failed attempt."

    return ResolvedAction(
        action_type=ActionType.COMBAT,
        success=success,
        outcome_description=outcome,
        items_consumed=items_consumed,
        items_gained=[],
        consequences=consequences,
        fatigue_modifier=fatigue_mod,
        skill_used="combat",
        dm_instruction=dm_instruction,
        raw_resolution={"roll": roll, "threshold": success_chance, "weapon": weapon}
    )


def resolve_skill_action(
    question: str,
    player_skills: List[str],
    fatigue_state: str,
    is_alive: bool
) -> ResolvedAction:
    """
    Pre-resolve a skill-based action.
    """
    if not is_alive:
        return ResolvedAction(
            action_type=ActionType.SKILL,
            success=False,
            outcome_description="Player is dead.",
            items_consumed=[],
            items_gained=[],
            consequences=[],
            fatigue_modifier="N/A",
            skill_used=None,
            dm_instruction="The player is dead and cannot attempt this.",
            raw_resolution={"reason": "player_dead"}
        )

    question_lower = question.lower()

    # Detect what skill is being attempted
    skill_keywords = {
        'hotwire': ['hotwire', 'hotwiring', 'hot-wire', 'start the car', 'start the vehicle'],
        'lockpick': ['lockpick', 'pick the lock', 'pick lock', 'unlock'],
        'medicine': ['heal', 'bandage', 'treat', 'first aid', 'medical', 'stitch'],
        'repair': ['repair', 'fix', 'mend'],
        'sneak': ['sneak', 'stealth', 'hide', 'creep', 'quietly'],
        'climb': ['climb', 'scale'],
        'swim': ['swim'],
    }

    attempted_skill = None
    for skill, keywords in skill_keywords.items():
        if any(kw in question_lower for kw in keywords):
            attempted_skill = skill
            break

    if not attempted_skill:
        attempted_skill = "general"

    # Check if player has this skill
    has_skill = False
    if player_skills:
        for ps in player_skills:
            ps_lower = ps.lower()
            if attempted_skill in ps_lower or any(kw in ps_lower for kw in skill_keywords.get(attempted_skill, [])):
                has_skill = True
                break

    # Base success chance
    if has_skill:
        success_chance = 0.8
        skill_note = f"Player has relevant skill"
    else:
        success_chance = 0.3
        skill_note = f"Player lacks this skill - attempting anyway"

    # Fatigue modifier
    if fatigue_state == "exhausted":
        success_chance *= 0.5
        fatigue_mod = "EXHAUSTED: Hands shake, concentration broken"
    elif fatigue_state == "tired":
        success_chance *= 0.8
        fatigue_mod = "TIRED: Slower, less precise"
    else:
        fatigue_mod = "Well-rested: Full capability"

    roll = random.random()
    success = roll < success_chance

    if success:
        outcome = f"Player attempts {attempted_skill} and SUCCEEDS. {skill_note}."
        dm_instruction = f"The player attempted to {attempted_skill} and succeeded. {skill_note}. Describe their competent execution, the satisfying result."
    else:
        outcome = f"Player attempts {attempted_skill} and FAILS. {skill_note}."
        dm_instruction = f"The player attempted to {attempted_skill} but failed. {skill_note}. Describe what went wrong - fumbled attempt, unexpected complication, or near-miss."

    return ResolvedAction(
        action_type=ActionType.SKILL,
        success=success,
        outcome_description=outcome,
        items_consumed=[],
        items_gained=[],
        consequences=[],
        fatigue_modifier=fatigue_mod,
        skill_used=attempted_skill,
        dm_instruction=dm_instruction,
        raw_resolution={"roll": roll, "threshold": success_chance, "has_skill": has_skill}
    )


def resolve_inventory_action(
    question: str,
    player_inventory: List[str],
    is_alive: bool
) -> Optional[ResolvedAction]:
    """
    Pre-resolve an inventory action (pick up, use, drop).
    Returns None if this should be left to DM (e.g., picking up unknown item).
    """
    if not is_alive:
        return ResolvedAction(
            action_type=ActionType.INVENTORY,
            success=False,
            outcome_description="Player is dead.",
            items_consumed=[],
            items_gained=[],
            consequences=[],
            fatigue_modifier="N/A",
            skill_used=None,
            dm_instruction="The player is dead and cannot interact with items.",
            raw_resolution={"reason": "player_dead"}
        )

    question_lower = question.lower()

    # Detect action type
    is_pickup = any(w in question_lower for w in ['pick up', 'grab', 'take', 'get', 'collect', 'loot'])
    is_use = any(w in question_lower for w in ['use', 'consume', 'eat', 'drink', 'apply'])
    is_drop = any(w in question_lower for w in ['drop', 'discard', 'throw away', 'put down'])

    if is_use:
        # Check if they have the item
        item = extract_target_item(question, player_inventory)
        if item:
            return ResolvedAction(
                action_type=ActionType.INVENTORY,
                success=True,
                outcome_description=f"Player uses {item}.",
                items_consumed=[item],
                items_gained=[],
                consequences=[],
                fatigue_modifier="",
                skill_used=None,
                dm_instruction=f"The player used their {item}. It is now consumed/depleted. Describe the use and its effect.",
                raw_resolution={"action": "use", "item": item}
            )
        else:
            # They're trying to use something they don't have
            return ResolvedAction(
                action_type=ActionType.INVENTORY,
                success=False,
                outcome_description="Player tries to use an item they don't have.",
                items_consumed=[],
                items_gained=[],
                consequences=[],
                fatigue_modifier="",
                skill_used=None,
                dm_instruction="The player reached for something they don't have. Describe their realization - patting empty pockets, the item isn't there.",
                raw_resolution={"action": "use", "item": None, "reason": "not_in_inventory"}
            )

    # For pickup/drop, let DM handle (they control what's in the environment)
    return None


def resolve_action(
    question: str,
    hard_facts: Dict[str, Any]
) -> Tuple[ActionType, Optional[ResolvedAction]]:
    """
    Main entry point: classify and potentially pre-resolve an action.

    Returns:
        (action_type, resolution) where resolution is None if DM should handle fully.
    """
    action_type = classify_action(question)

    # Extract facts
    player_inventory = hard_facts.get("player_has", [])
    player_skills = hard_facts.get("player_skills", [])
    fatigue_state = hard_facts.get("fatigue_state", "rested")
    is_alive = hard_facts.get("is_alive", True)
    is_sleeping = hard_facts.get("is_sleeping", False)

    # Sleeping players can't do anything
    if is_sleeping and action_type in [ActionType.COMBAT, ActionType.INVENTORY, ActionType.SKILL, ActionType.MOVEMENT]:
        return (action_type, ResolvedAction(
            action_type=action_type,
            success=False,
            outcome_description="Player is asleep.",
            items_consumed=[],
            items_gained=[],
            consequences=[],
            fatigue_modifier="ASLEEP",
            skill_used=None,
            dm_instruction="The player is asleep. They cannot take this action. Perhaps describe a dream, or someone trying to wake them.",
            raw_resolution={"reason": "player_sleeping"}
        ))

    # Resolve based on type
    if action_type == ActionType.COMBAT:
        return (action_type, resolve_combat_action(question, player_inventory, player_skills, fatigue_state, is_alive))

    elif action_type == ActionType.SKILL:
        return (action_type, resolve_skill_action(question, player_skills, fatigue_state, is_alive))

    elif action_type == ActionType.INVENTORY:
        resolution = resolve_inventory_action(question, player_inventory, is_alive)
        return (action_type, resolution)  # May be None

    # QUERY, SOCIAL, MOVEMENT, UNKNOWN → DM handles fully
    return (action_type, None)


def build_dm_instruction_for_resolution(resolution: ResolvedAction) -> str:
    """
    Build the full instruction block for the DM when we've pre-resolved.
    """
    lines = [
        "═══ PRE-RESOLVED ACTION ═══",
        f"Action Type: {resolution.action_type.value.upper()}",
        f"Outcome: {'SUCCESS' if resolution.success else 'FAILURE'}",
        f"What Happened: {resolution.outcome_description}",
    ]

    if resolution.fatigue_modifier:
        lines.append(f"Fatigue Effect: {resolution.fatigue_modifier}")

    if resolution.skill_used:
        lines.append(f"Skill Involved: {resolution.skill_used}")

    if resolution.items_consumed:
        lines.append(f"Items Consumed: {', '.join(resolution.items_consumed)}")

    if resolution.consequences:
        lines.append(f"Consequences Triggered: {', '.join(resolution.consequences)}")

    lines.append("")
    lines.append("YOUR JOB: Narrate this outcome. The mechanics are decided.")
    lines.append(resolution.dm_instruction)
    lines.append("═══════════════════════════")

    return "\n".join(lines)
