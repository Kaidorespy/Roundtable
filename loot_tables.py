"""
Loot Tables - Genre-appropriate random item generation.

"What's in its pockets, precious?"

This system generates contextually appropriate loot for:
- Searching bodies/creatures
- Opening containers (chests, drawers, crates)
- Rewards from NPCs
- Found items in the world

Design principles:
- Genre-aware: Fantasy swords, sci-fi datapads, modern phones
- Context-aware: Goblin pockets differ from dragon hoards
- Rarity-weighted: Common items drop often, legendaries almost never
- DM-friendly: Suggestions, not mandates
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import random
import json
from pathlib import Path
import debug_logger as dbg


class Rarity(Enum):
    """Item rarity tiers with drop weights."""
    JUNK = "junk"           # Weight: 30 - Trash, vendor fodder
    COMMON = "common"       # Weight: 40 - Basic useful items
    UNCOMMON = "uncommon"   # Weight: 20 - Decent finds
    RARE = "rare"           # Weight: 8  - Notable items
    EPIC = "epic"           # Weight: 1.8 - Impressive finds
    LEGENDARY = "legendary" # Weight: 0.2 - Story-defining items


RARITY_WEIGHTS = {
    Rarity.JUNK: 30,
    Rarity.COMMON: 40,
    Rarity.UNCOMMON: 20,
    Rarity.RARE: 8,
    Rarity.EPIC: 1.8,
    Rarity.LEGENDARY: 0.2,
}


class LootContext(Enum):
    """What are we looting?"""
    # Creatures/NPCs
    CREATURE_WEAK = "creature_weak"       # Rats, goblins, thugs
    CREATURE_MEDIUM = "creature_medium"   # Bandits, orcs, guards
    CREATURE_STRONG = "creature_strong"   # Knights, mages, bosses
    CREATURE_BOSS = "creature_boss"       # Dragons, demons, crime lords

    # Containers
    CONTAINER_POOR = "container_poor"     # Crates, barrels, bags
    CONTAINER_DECENT = "container_decent" # Chests, lockers, desks
    CONTAINER_RICH = "container_rich"     # Treasure chests, safes, vaults
    CONTAINER_LEGENDARY = "container_legendary"  # Dragon hoards, ancient tombs

    # Environmental
    POCKET = "pocket"           # Pickpocketing, quick search
    THOROUGH_SEARCH = "search"  # Taking time to search thoroughly
    HIDDEN_CACHE = "cache"      # Secret stashes, hidden compartments
    REWARD = "reward"           # Given by NPCs, quest rewards


# Context modifies what rarities can appear and quantity
CONTEXT_MODIFIERS = {
    LootContext.CREATURE_WEAK: {
        "max_rarity": Rarity.UNCOMMON,
        "item_count": (0, 2),
        "gold_range": (0, 5),
    },
    LootContext.CREATURE_MEDIUM: {
        "max_rarity": Rarity.RARE,
        "item_count": (1, 3),
        "gold_range": (2, 20),
    },
    LootContext.CREATURE_STRONG: {
        "max_rarity": Rarity.EPIC,
        "item_count": (1, 4),
        "gold_range": (10, 100),
    },
    LootContext.CREATURE_BOSS: {
        "max_rarity": Rarity.LEGENDARY,
        "item_count": (2, 6),
        "gold_range": (50, 500),
        "guaranteed_rare": True,
    },
    LootContext.CONTAINER_POOR: {
        "max_rarity": Rarity.COMMON,
        "item_count": (0, 3),
        "gold_range": (0, 2),
    },
    LootContext.CONTAINER_DECENT: {
        "max_rarity": Rarity.RARE,
        "item_count": (1, 4),
        "gold_range": (5, 30),
    },
    LootContext.CONTAINER_RICH: {
        "max_rarity": Rarity.EPIC,
        "item_count": (2, 5),
        "gold_range": (20, 200),
        "guaranteed_uncommon": True,
    },
    LootContext.CONTAINER_LEGENDARY: {
        "max_rarity": Rarity.LEGENDARY,
        "item_count": (3, 8),
        "gold_range": (100, 1000),
        "guaranteed_rare": True,
    },
    LootContext.POCKET: {
        "max_rarity": Rarity.UNCOMMON,
        "item_count": (0, 2),
        "gold_range": (0, 10),
    },
    LootContext.THOROUGH_SEARCH: {
        "max_rarity": Rarity.RARE,
        "item_count": (0, 3),
        "gold_range": (0, 15),
    },
    LootContext.HIDDEN_CACHE: {
        "max_rarity": Rarity.EPIC,
        "item_count": (1, 4),
        "gold_range": (10, 100),
        "guaranteed_uncommon": True,
    },
    LootContext.REWARD: {
        "max_rarity": Rarity.EPIC,
        "item_count": (1, 3),
        "gold_range": (10, 100),
        "guaranteed_uncommon": True,
    },
}


# =============================================================================
# Genre-Specific Loot Tables
# =============================================================================

FANTASY_LOOT = {
    Rarity.JUNK: [
        "a handful of lint",
        "a chipped tooth",
        "moldy bread",
        "a broken comb",
        "dirty rags",
        "rat bones",
        "a rusty spoon",
        "a torn map fragment",
        "a cracked leather strap",
        "dried herbs (worthless)",
    ],
    Rarity.COMMON: [
        "a small knife",
        "a tinderbox",
        "a coil of rope (50 ft)",
        "a waterskin",
        "trail rations",
        "a candle",
        "a simple cloak",
        "a belt pouch",
        "a wooden holy symbol",
        "a set of dice",
        "a small mirror",
        "a whetstone",
        "bandages",
        "a flask of oil",
        "a simple lockpick",
    ],
    Rarity.UNCOMMON: [
        "a silver dagger",
        "a healing potion",
        "a well-made lantern",
        "a leather-bound journal",
        "a quality shortbow",
        "a set of thieves' tools",
        "a spyglass",
        "an antidote vial",
        "a silk rope (50 ft)",
        "a fine cloak with hidden pockets",
        "a map of the local area",
        "a silver holy symbol",
        "an alchemist's kit",
        "a masterwork dagger",
    ],
    Rarity.RARE: [
        "a glowing shortsword (+1)",
        "a potion of invisibility",
        "enchanted boots of quiet steps",
        "a ring of minor protection",
        "a wand of magic missiles (7 charges)",
        "a cloak of elvenkind",
        "a bag of holding",
        "a scroll of fireball",
        "mithril chainmail",
        "a amulet of health",
    ],
    Rarity.EPIC: [
        "a flaming longsword (+2)",
        "a staff of healing",
        "winged boots",
        "a ring of invisibility",
        "a portable hole",
        "a cloak of displacement",
        "adamantine plate armor",
        "a circlet of blasting",
    ],
    Rarity.LEGENDARY: [
        "the Sword of the Realm (artifact)",
        "a wish scroll",
        "the Crown of the Fallen King",
        "a deck of many things",
        "a staff of the magi",
    ],
}

SCIFI_LOOT = {
    Rarity.JUNK: [
        "a broken comm unit",
        "a spent power cell",
        "corroded wiring",
        "a cracked data chip",
        "expired ration bars",
        "a torn uniform patch",
        "stripped screws and bolts",
        "a burned-out fuse",
        "empty stim cartridges",
        "a damaged ID badge",
    ],
    Rarity.COMMON: [
        "a functional comm unit",
        "a small flashlight",
        "a standard power cell",
        "ration packs (3 days)",
        "a basic medkit",
        "a multitool",
        "a portable scanner",
        "a rebreather mask",
        "cable ties and tape",
        "a data pad (locked)",
        "a backup power cell",
        "emergency flares",
        "a utility knife",
        "water purification tabs",
    ],
    Rarity.UNCOMMON: [
        "a plasma pistol",
        "a stim pack (heals wounds)",
        "a hacking module",
        "an encrypted data pad",
        "a personal shield emitter",
        "a magnetic grapple",
        "night vision goggles",
        "a trauma kit",
        "a cloaking device (10 min)",
        "an EMP grenade",
        "a military-grade scanner",
        "a jetpack (limited fuel)",
    ],
    Rarity.RARE: [
        "a plasma rifle",
        "powered exoskeleton legs",
        "a personal AI assistant",
        "military-grade body armor",
        "a teleporter beacon (one use)",
        "a gravity manipulation device",
        "advanced cybernetic implant",
        "a cloaking suit",
        "an antimatter cell",
    ],
    Rarity.EPIC: [
        "a particle beam cannon",
        "full power armor",
        "an AI core (sentient)",
        "a personal teleporter",
        "a time dilation device",
        "military starship access codes",
        "a nanite healing swarm",
    ],
    Rarity.LEGENDARY: [
        "the coordinates to the Lost Colony",
        "a working FTL drive core",
        "the Admiral's personal sidearm (artifact)",
        "a reality manipulation device",
        "the cure for the Plague",
    ],
}

MODERN_LOOT = {
    Rarity.JUNK: [
        "a crumpled receipt",
        "used gum wrappers",
        "a dead phone battery",
        "broken earbuds",
        "expired coupons",
        "a crushed cigarette pack",
        "a worn-out wallet (empty)",
        "old keys (unknown locks)",
        "lint and coins",
        "a faded business card",
    ],
    Rarity.COMMON: [
        "a burner phone",
        "a swiss army knife",
        "a flashlight",
        "a first aid kit",
        "zip ties",
        "duct tape",
        "a lighter",
        "cash ($20-50)",
        "a USB drive",
        "car keys",
        "a crowbar",
        "a prepaid credit card",
        "energy bars",
        "a bottle of water",
    ],
    Rarity.UNCOMMON: [
        "a handgun with ammo",
        "a laptop (password protected)",
        "a lockpick set",
        "night vision goggles",
        "a police scanner",
        "body armor (vest)",
        "a fake ID",
        "burglary tools",
        "a taser",
        "GPS tracker",
        "a drone (small)",
        "encrypted radio",
    ],
    Rarity.RARE: [
        "a silenced pistol",
        "military-grade body armor",
        "hacking equipment",
        "a briefcase of cash ($10,000)",
        "explosives (C4)",
        "a sniper rifle",
        "classified documents",
        "a high-end sports car key",
        "a satellite phone",
    ],
    Rarity.EPIC: [
        "a prototype weapon",
        "access to a swiss bank account",
        "evidence that could bring down a corporation",
        "a cache of bearer bonds",
        "experimental military tech",
        "a list of undercover agents",
    ],
    Rarity.LEGENDARY: [
        "the hard drive with all the evidence",
        "nuclear launch codes",
        "the identity of the mastermind",
        "a cure for the engineered virus",
        "coordinates to the hidden bunker",
    ],
}

HORROR_LOOT = {
    Rarity.JUNK: [
        "a torn photograph",
        "dried flowers",
        "a broken music box",
        "old love letters",
        "a faded newspaper clipping",
        "a child's toy (disturbing)",
        "cracked spectacles",
        "a rusted locket (empty)",
        "moth-eaten cloth",
        "bone fragments",
    ],
    Rarity.COMMON: [
        "a flashlight (flickering)",
        "matches",
        "a journal entry",
        "a ring of keys",
        "a small crucifix",
        "salt",
        "a bottle of holy water",
        "bandages",
        "a lighter",
        "a kitchen knife",
        "a camera (old)",
        "a hand mirror",
        "candles",
        "a map of the building",
    ],
    Rarity.UNCOMMON: [
        "a handgun (few bullets)",
        "a diary revealing secrets",
        "an ancient talisman",
        "a ritual dagger",
        "a cryptic cipher",
        "a photograph showing something impossible",
        "a vial of strange liquid",
        "night vision equipment",
        "silver bullets",
        "a spirit board",
        "an old brass key",
    ],
    Rarity.RARE: [
        "a blessed weapon",
        "a page from the forbidden text",
        "the true name of the entity",
        "a sealing artifact",
        "video evidence of the supernatural",
        "an exorcist's kit",
        "a map to the ritual site",
        "the antidote",
    ],
    Rarity.EPIC: [
        "the complete ritual to banish it",
        "an angel's feather",
        "the heart of the creature",
        "a fragment of divine power",
        "the original binding contract",
    ],
    Rarity.LEGENDARY: [
        "the Necronomicon",
        "the weapon that killed it before",
        "God's direct phone line",
        "the truth about what happened here",
    ],
}

WESTERN_LOOT = {
    Rarity.JUNK: [
        "tobacco crumbs",
        "a worn deck of cards (incomplete)",
        "rusty nails",
        "a broken spur",
        "empty whiskey bottle",
        "a wanted poster (torn)",
        "frayed rope",
        "a dented tin cup",
        "moldy hardtack",
        "snake rattles",
    ],
    Rarity.COMMON: [
        "a revolver",
        "a hunting knife",
        "a lasso",
        "a canteen",
        "beef jerky",
        "matches",
        "a bandana",
        "playing cards",
        "a harmonica",
        "a compass",
        "a bedroll",
        "horseshoes",
        "ammunition",
        "a bottle of whiskey",
    ],
    Rarity.UNCOMMON: [
        "a fine Colt revolver",
        "a Winchester rifle",
        "a sheriff's badge",
        "dynamite sticks",
        "a treasure map",
        "a deed to land",
        "a quality saddle",
        "gold dust pouch",
        "a wanted poster (valuable target)",
        "snake antivenom",
        "a spyglass",
    ],
    Rarity.RARE: [
        "a pearl-handled revolver",
        "a sawed-off shotgun",
        "a bag of gold nuggets",
        "the deed to the mine",
        "a letter proving innocence",
        "a cavalry saber",
        "native american artifact",
        "a strongbox key",
    ],
    Rarity.EPIC: [
        "the outlaw's legendary gun",
        "a map to the lost gold mine",
        "the sheriff's confession",
        "a bag of stolen gold",
        "the railroad baron's secrets",
    ],
    Rarity.LEGENDARY: [
        "the Lost Dutchman's treasure map",
        "the deed to the whole territory",
        "the legendary gunslinger's pistol",
    ],
}


# Map genre keywords to loot tables
GENRE_TABLES = {
    "fantasy": FANTASY_LOOT,
    "medieval": FANTASY_LOOT,
    "dnd": FANTASY_LOOT,
    "dungeons": FANTASY_LOOT,
    "magic": FANTASY_LOOT,
    "scifi": SCIFI_LOOT,
    "sci-fi": SCIFI_LOOT,
    "science fiction": SCIFI_LOOT,
    "space": SCIFI_LOOT,
    "cyberpunk": SCIFI_LOOT,
    "futuristic": SCIFI_LOOT,
    "modern": MODERN_LOOT,
    "contemporary": MODERN_LOOT,
    "spy": MODERN_LOOT,
    "thriller": MODERN_LOOT,
    "crime": MODERN_LOOT,
    "horror": HORROR_LOOT,
    "supernatural": HORROR_LOOT,
    "lovecraft": HORROR_LOOT,
    "gothic": HORROR_LOOT,
    "western": WESTERN_LOOT,
    "cowboy": WESTERN_LOOT,
    "frontier": WESTERN_LOOT,
}


@dataclass
class LootResult:
    """Result of a loot roll."""
    items: List[Dict[str, str]]  # [{name, rarity, description}]
    gold: int  # Or credits/dollars/whatever
    gold_name: str  # "gold coins" / "credits" / "dollars"
    context_description: str  # "searching the goblin's pockets"
    nothing_found: bool = False

    def to_narrative(self) -> str:
        """Convert to narrative text for the DM to use."""
        if self.nothing_found:
            return f"After {self.context_description}, you find nothing of value."

        parts = []
        if self.gold > 0:
            parts.append(f"{self.gold} {self.gold_name}")

        for item in self.items:
            rarity = item.get("rarity", "common")
            if rarity in ["rare", "epic", "legendary"]:
                parts.append(f"**{item['name']}** ({rarity})")
            else:
                parts.append(item["name"])

        if not parts:
            return f"After {self.context_description}, you find nothing of value."

        items_text = ", ".join(parts[:-1]) + f" and {parts[-1]}" if len(parts) > 1 else parts[0]
        return f"After {self.context_description}, you find: {items_text}"

    def to_dict(self) -> Dict:
        return {
            "items": self.items,
            "gold": self.gold,
            "gold_name": self.gold_name,
            "context_description": self.context_description,
            "nothing_found": self.nothing_found,
            "narrative": self.to_narrative(),
        }


class LootGenerator:
    """
    Generates contextually appropriate loot.

    Usage:
        gen = LootGenerator(genre="fantasy")
        loot = gen.generate(LootContext.CREATURE_WEAK, description="goblin")
        print(loot.to_narrative())
    """

    def __init__(self, genre: str = "fantasy"):
        self.genre = genre.lower()
        self.loot_table = self._get_genre_table()
        self.gold_name = self._get_gold_name()

    def _get_genre_table(self) -> Dict[Rarity, List[str]]:
        """Get the appropriate loot table for the genre."""
        for keyword, table in GENRE_TABLES.items():
            if keyword in self.genre:
                return table
        # Default to fantasy
        return FANTASY_LOOT

    def _get_gold_name(self) -> str:
        """Get the currency name for the genre."""
        if any(k in self.genre for k in ["scifi", "sci-fi", "space", "cyber", "futur"]):
            return "credits"
        elif any(k in self.genre for k in ["modern", "contemporary", "spy", "thriller", "crime"]):
            return "dollars"
        elif any(k in self.genre for k in ["western", "cowboy", "frontier"]):
            return "dollars"
        else:
            return "gold coins"

    def _roll_rarity(self, max_rarity: Rarity) -> Rarity:
        """Roll for item rarity with weighted probability."""
        # Build weight list up to max_rarity
        rarity_order = [Rarity.JUNK, Rarity.COMMON, Rarity.UNCOMMON,
                        Rarity.RARE, Rarity.EPIC, Rarity.LEGENDARY]
        max_index = rarity_order.index(max_rarity)

        valid_rarities = rarity_order[:max_index + 1]
        weights = [RARITY_WEIGHTS[r] for r in valid_rarities]

        return random.choices(valid_rarities, weights=weights, k=1)[0]

    def _get_random_item(self, rarity: Rarity) -> str:
        """Get a random item of the specified rarity."""
        items = self.loot_table.get(rarity, self.loot_table[Rarity.COMMON])
        return random.choice(items)

    def generate(
        self,
        context: LootContext,
        description: str = "",
        luck_modifier: float = 0.0,
        guaranteed_items: List[str] = None,
    ) -> LootResult:
        """
        Generate loot for a given context.

        Args:
            context: What kind of loot source
            description: Description for narrative (e.g., "the dead bandit")
            luck_modifier: -1.0 to 1.0, affects rarity chances
            guaranteed_items: Specific items to always include

        Returns:
            LootResult with items and gold
        """
        mods = CONTEXT_MODIFIERS.get(context, CONTEXT_MODIFIERS[LootContext.CONTAINER_POOR])
        max_rarity = mods["max_rarity"]
        min_items, max_items = mods["item_count"]
        min_gold, max_gold = mods["gold_range"]

        # Roll number of items
        num_items = random.randint(min_items, max_items)

        # Roll gold
        gold = random.randint(min_gold, max_gold) if max_gold > 0 else 0

        # Apply luck modifier to gold
        if luck_modifier != 0:
            gold = int(gold * (1 + luck_modifier * 0.5))
            gold = max(0, gold)

        items = []
        used_items = set()  # Track used items to avoid duplicates

        # Add guaranteed items first
        if guaranteed_items:
            for item_name in guaranteed_items:
                items.append({
                    "name": item_name,
                    "rarity": "uncommon",  # Assume uncommon for guaranteed
                    "description": "A specific item you were looking for."
                })
                used_items.add(item_name)

        # Check for guaranteed minimum rarity
        guaranteed_min = None
        if mods.get("guaranteed_legendary"):
            guaranteed_min = Rarity.LEGENDARY
        elif mods.get("guaranteed_rare"):
            guaranteed_min = Rarity.RARE
        elif mods.get("guaranteed_uncommon"):
            guaranteed_min = Rarity.UNCOMMON

        # Generate random items
        for i in range(num_items):
            # First item uses guaranteed minimum if applicable
            if i == 0 and guaranteed_min:
                rarity = guaranteed_min
            else:
                rarity = self._roll_rarity(max_rarity)

            # Apply luck modifier - chance to upgrade rarity
            if luck_modifier > 0 and random.random() < luck_modifier:
                rarity_order = [Rarity.JUNK, Rarity.COMMON, Rarity.UNCOMMON,
                               Rarity.RARE, Rarity.EPIC, Rarity.LEGENDARY]
                current_idx = rarity_order.index(rarity)
                max_idx = rarity_order.index(max_rarity)
                if current_idx < max_idx:
                    rarity = rarity_order[current_idx + 1]

            # Try to get a unique item (try up to 5 times before giving up)
            item_name = None
            for _ in range(5):
                candidate = self._get_random_item(rarity)
                if candidate not in used_items:
                    item_name = candidate
                    used_items.add(candidate)
                    break

            if item_name is None:
                continue  # Skip if we couldn't find a unique item
            items.append({
                "name": item_name,
                "rarity": rarity.value,
                "description": f"A {rarity.value} item."
            })

        # Build context description
        if description:
            context_desc = f"searching {description}"
        else:
            context_desc = f"searching the {context.value.replace('_', ' ')}"

        nothing_found = len(items) == 0 and gold == 0

        return LootResult(
            items=items,
            gold=gold,
            gold_name=self.gold_name,
            context_description=context_desc,
            nothing_found=nothing_found,
        )

    def generate_specific(
        self,
        rarity: Rarity,
        count: int = 1,
        description: str = "",
    ) -> LootResult:
        """Generate specific rarity items (useful for rewards)."""
        items = []
        for _ in range(count):
            item_name = self._get_random_item(rarity)
            items.append({
                "name": item_name,
                "rarity": rarity.value,
                "description": f"A {rarity.value} item."
            })

        return LootResult(
            items=items,
            gold=0,
            gold_name=self.gold_name,
            context_description=description or "a reward",
            nothing_found=False,
        )


# =============================================================================
# DM Helper Functions
# =============================================================================

def suggest_loot_for_scene(
    genre: str,
    scene_description: str,
    creature_type: str = None,
    container_type: str = None,
) -> str:
    """
    Generate a loot suggestion for the DM based on scene context.

    Returns a formatted string the DM can use or modify.
    """
    gen = LootGenerator(genre)

    # Determine context from description
    context = LootContext.CONTAINER_DECENT  # Default
    description = ""

    if creature_type:
        creature_lower = creature_type.lower()
        if any(w in creature_lower for w in ["rat", "goblin", "imp", "thug", "bandit", "weak", "minor"]):
            context = LootContext.CREATURE_WEAK
        elif any(w in creature_lower for w in ["boss", "dragon", "demon", "lord", "king", "ancient"]):
            context = LootContext.CREATURE_BOSS
        elif any(w in creature_lower for w in ["knight", "mage", "captain", "elite", "veteran"]):
            context = LootContext.CREATURE_STRONG
        else:
            context = LootContext.CREATURE_MEDIUM
        description = f"the {creature_type}"

    elif container_type:
        container_lower = container_type.lower()
        if any(w in container_lower for w in ["crate", "barrel", "bag", "sack", "poor", "shabby"]):
            context = LootContext.CONTAINER_POOR
        elif any(w in container_lower for w in ["treasure", "vault", "hoard", "legendary", "ancient"]):
            context = LootContext.CONTAINER_LEGENDARY
        elif any(w in container_lower for w in ["rich", "ornate", "gilded", "safe", "vault"]):
            context = LootContext.CONTAINER_RICH
        else:
            context = LootContext.CONTAINER_DECENT
        description = f"the {container_type}"

    loot = gen.generate(context, description)
    return loot.to_narrative()


def get_loot_context_for_dm(genre: str, world_threat_level: int = 0) -> str:
    """
    Generate a quick loot reference for the DM context.
    """
    gen = LootGenerator(genre)

    # Sample a few items at different rarities
    samples = []
    for rarity in [Rarity.COMMON, Rarity.UNCOMMON, Rarity.RARE]:
        if rarity in gen.loot_table:
            item = random.choice(gen.loot_table[rarity])
            samples.append(f"  {rarity.value}: {item}")

    return f"""Loot Tables ({genre}):
{chr(10).join(samples)}
  Currency: {gen.gold_name}
  (Use /loot command for full generation)"""


# =============================================================================
# Global instance for convenience
# =============================================================================

_default_generator: Optional[LootGenerator] = None


def get_loot_generator(genre: str = "fantasy") -> LootGenerator:
    """Get or create a loot generator."""
    global _default_generator
    if _default_generator is None or _default_generator.genre != genre.lower():
        _default_generator = LootGenerator(genre)
    return _default_generator
