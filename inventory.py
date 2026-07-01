"""
Inventory Tracker - What's in everyone's backpack?

A DM assistant that tracks items for all characters.
"Hey DM, what's in my backpack?" → DM checks and answers.

Simple but essential for coherent storytelling.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import json
import uuid
import debug_logger as dbg


class ItemCategory(Enum):
    """Categories of items."""
    WEAPON = "weapon"
    ARMOR = "armor"
    CONSUMABLE = "consumable"      # Food, potions, ammo
    TOOL = "tool"                  # Lockpicks, rope, etc.
    KEY_ITEM = "key_item"          # Quest items, plot devices
    TREASURE = "treasure"          # Valuables, currency
    CLOTHING = "clothing"
    CONTAINER = "container"        # Bags, boxes
    MISC = "misc"


class AmmoLevel(Enum):
    """
    Fuzzy ammo tracking for weapons. Not exact counts - vibes.
    The DM adjusts this based on narrative usage.
    """
    EMPTY = "empty"              # 🔴 No ammo - weapon is dead weight
    ALMOST_OUT = "almost_out"    # 🔴 Last few shots, make them count
    LOW = "low"                  # 🟡 Running low, be conservative
    DECENT = "decent"            # 🟢 Comfortable supply
    PLENTIFUL = "plentiful"      # 🟢 Ammo for days
    NA = "n/a"                   # Not applicable (melee, etc.)


@dataclass
class Item:
    """An item that can be owned."""
    id: str
    name: str
    category: ItemCategory = ItemCategory.MISC

    # Description
    description: str = ""
    short_description: str = ""  # One line for lists

    # Quantity (for stackable items)
    quantity: int = 1
    stackable: bool = False
    max_stack: int = 99

    # Properties
    weight: float = 0.0          # Weight in lbs (0 = not tracked)
    value: float = 0.0           # Currency value (optional)
    capacity: float = 0.0        # For containers: how much weight they can hold (lbs)

    # Special flags
    is_equipped: bool = False
    is_quest_item: bool = False
    is_consumable: bool = False

    # Ammo tracking (for weapons) - fuzzy, not exact counts
    ammo_level: str = "n/a"      # AmmoLevel value: empty, almost_out, low, decent, plentiful, n/a

    # Metadata
    acquired_at: str = ""
    acquired_from: str = ""      # Where/who they got it from
    notes: str = ""              # DM notes

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:12]
        if not self.acquired_at:
            self.acquired_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "short_description": self.short_description,
            "quantity": self.quantity,
            "stackable": self.stackable,
            "max_stack": self.max_stack,
            "weight": self.weight,
            "value": self.value,
            "capacity": self.capacity,
            "is_equipped": self.is_equipped,
            "is_quest_item": self.is_quest_item,
            "is_consumable": self.is_consumable,
            "ammo_level": self.ammo_level,
            "acquired_at": self.acquired_at,
            "acquired_from": self.acquired_from,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Item":
        data = data.copy()
        if "category" in data:
            data["category"] = ItemCategory(data["category"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Inventory:
    """A character's inventory."""
    owner_id: str               # Character/partner/NPC ID
    owner_name: str
    owner_type: str = "player"  # "player", "partner", "npc"

    items: List[Item] = field(default_factory=list)

    # Capacity (optional)
    max_weight: float = 0.0     # 0 = unlimited
    max_slots: int = 0          # 0 = unlimited

    # Currency (genre-flexible)
    currency: Dict[str, float] = field(default_factory=dict)  # "gold": 100, "caps": 50

    # Timestamps
    created_at: str = ""
    last_updated: str = ""

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        self.last_updated = now

    def add_item(
        self,
        name: str,
        category: ItemCategory = ItemCategory.MISC,
        quantity: int = 1,
        **kwargs
    ) -> Item:
        """Add an item to the inventory."""
        # Check if stackable item already exists
        existing = self.find_item_by_name(name)
        if existing and existing.stackable:
            existing.quantity = min(existing.quantity + quantity, existing.max_stack)
            self.last_updated = datetime.now().isoformat()
            return existing

        # Auto-estimate weight if not provided
        if 'weight' not in kwargs or kwargs.get('weight', 0) == 0:
            # Import here to avoid circular import
            from inventory import estimate_weight
            kwargs['weight'] = estimate_weight(name, category)

        # Auto-estimate capacity for containers if not provided
        if category == ItemCategory.CONTAINER and ('capacity' not in kwargs or kwargs.get('capacity', 0) == 0):
            from inventory import estimate_capacity
            kwargs['capacity'] = estimate_capacity(name)

        # Create new item
        item = Item(
            id=str(uuid.uuid4())[:12],
            name=name,
            category=category,
            quantity=quantity,
            **kwargs
        )
        self.items.append(item)
        self.last_updated = datetime.now().isoformat()
        return item

    def remove_item(self, item_id: str, quantity: int = 1) -> bool:
        """Remove an item (or reduce quantity)."""
        for i, item in enumerate(self.items):
            if item.id == item_id:
                if item.stackable and item.quantity > quantity:
                    item.quantity -= quantity
                else:
                    self.items.pop(i)
                self.last_updated = datetime.now().isoformat()
                return True
        return False

    def find_item_by_name(self, name: str) -> Optional[Item]:
        """Find an item by name (case-insensitive)."""
        name_lower = name.lower()
        for item in self.items:
            if item.name.lower() == name_lower:
                return item
        return None

    def find_item_fuzzy(self, name: str) -> Optional[Item]:
        """
        Find an item by fuzzy matching - catches variations like:
        - "notebook" matches "Waterlogged Sketchbook"
        - "my notebook" matches "Notebook"
        - "the sketchbook" matches "Waterlogged Sketchbook"
        """
        if not name:
            return None

        name_lower = name.lower().strip()
        # Remove common prefixes
        for prefix in ['my ', 'the ', 'a ', 'an ', 'your ', 'his ', 'her ', 'their ']:
            if name_lower.startswith(prefix):
                name_lower = name_lower[len(prefix):]

        # Extract key words (skip very short ones)
        search_words = [w for w in name_lower.split() if len(w) > 2]
        if not search_words:
            search_words = [name_lower]

        for item in self.items:
            item_lower = item.name.lower()
            # Exact match
            if name_lower == item_lower:
                return item
            # Any search word appears in item name
            for word in search_words:
                if word in item_lower:
                    return item
            # Item name word appears in search
            for item_word in item_lower.split():
                if len(item_word) > 3 and item_word in name_lower:
                    return item

        return None

    def find_item_by_id(self, item_id: str) -> Optional[Item]:
        """Find an item by ID."""
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def get_items_by_category(self, category: ItemCategory) -> List[Item]:
        """Get all items in a category."""
        return [item for item in self.items if item.category == category]

    def get_equipped_items(self) -> List[Item]:
        """Get all equipped items."""
        return [item for item in self.items if item.is_equipped]

    def get_equipped_container(self) -> Optional[Item]:
        """Get the currently equipped container (backpack, pouch, etc.)."""
        for item in self.items:
            if item.is_equipped and item.category == ItemCategory.CONTAINER:
                return item
        return None

    def get_equipped_consumable(self) -> Optional[Item]:
        """Get currently equipped consumable (for use intent detection)."""
        for item in self.items:
            if item.is_equipped and item.category == ItemCategory.CONSUMABLE:
                return item
        return None

    def get_equipped_tool(self) -> Optional[Item]:
        """Get currently equipped tool (for use intent detection)."""
        for item in self.items:
            if item.is_equipped and item.category == ItemCategory.TOOL:
                return item
        return None

    def get_total_weight(self) -> float:
        """Calculate total weight of all items."""
        return sum(item.weight * item.quantity for item in self.items)

    def get_available_capacity(self) -> float:
        """
        Get available carrying capacity based on equipped container.
        Returns 0 if no container equipped, or remaining capacity.
        """
        container = self.get_equipped_container()
        if not container or container.capacity <= 0:
            return 0.0
        current_weight = self.get_total_weight()
        return max(0, container.capacity - current_weight)

    def can_add_item(self, weight: float) -> tuple[bool, str]:
        """
        Check if an item of given weight can be added.
        Returns (can_add, reason).
        """
        container = self.get_equipped_container()
        if not container:
            return False, "No container equipped"

        if container.capacity <= 0:
            # Container has no capacity set, allow anything
            return True, "OK"

        current_weight = self.get_total_weight()
        if current_weight + weight > container.capacity:
            available = container.capacity - current_weight
            return False, f"Too heavy ({weight:.1f} lbs). Only {available:.1f} lbs capacity remaining."

        return True, "OK"

    def get_total_value(self) -> float:
        """Calculate total value of all items."""
        return sum(item.value * item.quantity for item in self.items)

    def add_currency(self, currency_type: str, amount: float):
        """Add currency."""
        if currency_type not in self.currency:
            self.currency[currency_type] = 0
        self.currency[currency_type] += amount
        self.last_updated = datetime.now().isoformat()

    def remove_currency(self, currency_type: str, amount: float) -> bool:
        """Remove currency (returns False if insufficient)."""
        current = self.currency.get(currency_type, 0)
        if current < amount:
            return False
        self.currency[currency_type] = current - amount
        self.last_updated = datetime.now().isoformat()
        return True

    def to_dict(self) -> Dict:
        return {
            "owner_id": self.owner_id,
            "owner_name": self.owner_name,
            "owner_type": self.owner_type,
            "items": [item.to_dict() for item in self.items],
            "max_weight": self.max_weight,
            "max_slots": self.max_slots,
            "currency": self.currency,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Inventory":
        data = data.copy()
        if "items" in data:
            data["items"] = [Item.from_dict(i) for i in data["items"]]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def get_summary(self) -> str:
        """Get a text summary of the inventory."""
        lines = [f"=== {self.owner_name}'s Inventory ===\n"]

        # Currency
        if self.currency:
            currency_str = ", ".join(f"{v:.0f} {k}" for k, v in self.currency.items() if v > 0)
            if currency_str:
                lines.append(f"Currency: {currency_str}\n")

        # Items by category
        categories_with_items = {}
        for item in self.items:
            cat = item.category.value
            if cat not in categories_with_items:
                categories_with_items[cat] = []
            categories_with_items[cat].append(item)

        for cat, items in categories_with_items.items():
            lines.append(f"{cat.title()}:")
            for item in items:
                qty_str = f" x{item.quantity}" if item.quantity > 1 else ""
                equipped_str = " [EQUIPPED]" if item.is_equipped else ""
                lines.append(f"  • {item.name}{qty_str}{equipped_str}")
                if item.short_description:
                    lines.append(f"    {item.short_description}")
            lines.append("")

        if not self.items:
            lines.append("(Empty)\n")

        # Weight/capacity
        if self.max_weight > 0:
            total = self.get_total_weight()
            lines.append(f"Weight: {total:.1f} / {self.max_weight:.1f}")

        return "\n".join(lines)


class InventoryTracker:
    """
    Tracks inventories for all characters in the game.

    The DM's assistant for "what's in my backpack?"
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".roundtable"
        self.inventories_file = self.data_dir / "inventories.json"
        self.inventories: Dict[str, Inventory] = {}  # owner_id -> Inventory
        # Combat readiness: character_id -> True if ready to fight
        # This is separate from equipped weapon - you can be ready with fists
        self.combat_ready: Dict[str, bool] = {}
        self._load()

    def _load(self):
        """Load inventories from disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.inventories_file.exists():
            try:
                data = json.loads(self.inventories_file.read_text())
                for owner_id, inv_data in data.items():
                    self.inventories[owner_id] = Inventory.from_dict(inv_data)
            except Exception as e:
                print(f"[Inventory] Error loading: {e}")

    def _save(self):
        """Save inventories to disk."""
        try:
            data = {k: v.to_dict() for k, v in self.inventories.items()}
            self.inventories_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[Inventory] Error saving: {e}")

    def get_or_create_inventory(
        self,
        owner_id: str,
        owner_name: str,
        owner_type: str = "player"
    ) -> Inventory:
        """Get or create an inventory for a character."""
        if owner_id not in self.inventories:
            self.inventories[owner_id] = Inventory(
                owner_id=owner_id,
                owner_name=owner_name,
                owner_type=owner_type,
            )
            self._save()
        return self.inventories[owner_id]

    def get_inventory(self, owner_id: str) -> Optional[Inventory]:
        """Get an inventory if it exists."""
        return self.inventories.get(owner_id)

    def add_item(
        self,
        owner_id: str,
        item_name: str,
        category: str = "misc",
        quantity: int = 1,
        **kwargs
    ) -> Optional[Item]:
        """Add an item to a character's inventory."""
        inv = self.get_inventory(owner_id)
        if not inv:
            return None

        try:
            cat = ItemCategory(category)
        except ValueError:
            cat = ItemCategory.MISC

        item = inv.add_item(item_name, cat, quantity, **kwargs)
        self._save()
        return item

    def remove_item(
        self,
        owner_id: str,
        item_id: str,
        quantity: int = 1
    ) -> bool:
        """Remove an item from a character's inventory."""
        inv = self.get_inventory(owner_id)
        if not inv:
            return False

        result = inv.remove_item(item_id, quantity)
        if result:
            self._save()
        return result

    def transfer_item(
        self,
        from_owner_id: str,
        to_owner_id: str,
        item_id: str,
        quantity: int = 1
    ) -> bool:
        """Transfer an item between characters."""
        from_inv = self.get_inventory(from_owner_id)
        to_inv = self.get_inventory(to_owner_id)

        if not from_inv or not to_inv:
            return False

        item = from_inv.find_item_by_id(item_id)
        if not item:
            return False

        # Remove from source
        if not from_inv.remove_item(item_id, quantity):
            return False

        # Add to destination
        to_inv.add_item(
            name=item.name,
            category=item.category,
            quantity=quantity,
            description=item.description,
            short_description=item.short_description,
            stackable=item.stackable,
            weight=item.weight,
            value=item.value,
            is_quest_item=item.is_quest_item,
            is_consumable=item.is_consumable,
        )

        self._save()
        return True

    def get_dm_context(self, owner_id: str) -> str:
        """Get inventory summary for DM context."""
        inv = self.get_inventory(owner_id)
        if not inv:
            return f"No inventory found for {owner_id}"
        return inv.get_summary()

    def get_all_inventories_context(self) -> str:
        """Get summary of all inventories for DM."""
        if not self.inventories:
            return "No inventories tracked yet."

        lines = []
        for inv in self.inventories.values():
            lines.append(inv.get_summary())
            lines.append("-" * 40 + "\n")

        return "\n".join(lines)

    # =========================================================================
    # Combat Readiness System
    # =========================================================================

    def is_combat_ready(self, owner_id: str) -> bool:
        """Check if a character is in combat-ready stance (weapon drawn or fists up)."""
        return self.combat_ready.get(owner_id, False)

    def set_combat_ready(self, owner_id: str, ready: bool = True) -> dict:
        """
        Set a character's combat readiness.

        Returns info about their combat stance.
        """
        self.combat_ready[owner_id] = ready
        inv = self.get_inventory(owner_id)

        result = {
            'owner_id': owner_id,
            'combat_ready': ready,
            'equipped_weapon': None,
            'using_fists': False,
        }

        if inv:
            equipped = [i for i in inv.items if i.is_equipped and i.category == ItemCategory.WEAPON]
            if equipped:
                result['equipped_weapon'] = equipped[0].name
            elif ready:
                result['using_fists'] = True

        return result

    def get_equipped_weapon(self, owner_id: str) -> Optional[Item]:
        """Get the currently equipped weapon for a character."""
        inv = self.get_inventory(owner_id)
        if not inv:
            return None
        weapons = [i for i in inv.items if i.is_equipped and i.category == ItemCategory.WEAPON]
        return weapons[0] if weapons else None

    def has_any_weapon(self, owner_id: str) -> bool:
        """Check if character has any weapon in inventory (equipped or not)."""
        inv = self.get_inventory(owner_id)
        if not inv:
            return False
        return any(i.category == ItemCategory.WEAPON for i in inv.items)

    def unequip_all_weapons(self, owner_id: str) -> List[str]:
        """
        Unequip all weapons and clear combat ready state.
        Returns list of unequipped weapon names.
        """
        self.combat_ready[owner_id] = False
        inv = self.get_inventory(owner_id)
        unequipped = []

        if inv:
            for item in inv.items:
                if item.is_equipped and item.category == ItemCategory.WEAPON:
                    item.is_equipped = False
                    unequipped.append(item.name)
            if unequipped:
                self._save()

        return unequipped

    def get_combat_status(self, owner_id: str) -> dict:
        """
        Get full combat readiness status for a character.

        Used by DM to know if player is ready to fight.
        """
        inv = self.get_inventory(owner_id)
        ready = self.combat_ready.get(owner_id, False)

        status = {
            'owner_id': owner_id,
            'combat_ready': ready,
            'equipped_weapon': None,
            'has_any_weapon': False,
            'using_fists': False,
            'all_weapons': [],
        }

        if inv:
            weapons = [i for i in inv.items if i.category == ItemCategory.WEAPON]
            status['has_any_weapon'] = len(weapons) > 0
            status['all_weapons'] = [w.name for w in weapons]

            equipped = [i for i in weapons if i.is_equipped]
            if equipped:
                status['equipped_weapon'] = equipped[0].name
            elif ready:
                status['using_fists'] = True

        return status

    # =========================================================================
    # Container/Loot System - "Equip to signal intent"
    # =========================================================================

    def get_equipped_container(self, owner_id: str) -> Optional[Item]:
        """Get the currently equipped container for loot mode."""
        inv = self.get_inventory(owner_id)
        if not inv:
            return None
        return inv.get_equipped_container()

    def get_equipped_usable(self, owner_id: str) -> Optional[Item]:
        """
        Get currently equipped consumable or tool (for use intent).
        Returns the item if one is equipped for use.
        """
        inv = self.get_inventory(owner_id)
        if not inv:
            return None
        # Check consumables first, then tools
        item = inv.get_equipped_consumable()
        if item:
            return item
        return inv.get_equipped_tool()

    def unequip_containers(self, owner_id: str) -> List[str]:
        """
        Unequip all containers (after loot action processed).
        Returns list of unequipped container names.
        """
        inv = self.get_inventory(owner_id)
        unequipped = []

        if inv:
            for item in inv.items:
                if item.is_equipped and item.category == ItemCategory.CONTAINER:
                    item.is_equipped = False
                    unequipped.append(item.name)
            if unequipped:
                self._save()

        return unequipped

    def unequip_consumables_and_tools(self, owner_id: str) -> List[str]:
        """
        Unequip all consumables and tools (after use action processed).
        Returns list of unequipped item names.
        """
        inv = self.get_inventory(owner_id)
        unequipped = []

        if inv:
            for item in inv.items:
                if item.is_equipped and item.category in (ItemCategory.CONSUMABLE, ItemCategory.TOOL):
                    item.is_equipped = False
                    unequipped.append(item.name)
            if unequipped:
                self._save()

        return unequipped

    def add_looted_item(
        self,
        owner_id: str,
        item_name: str,
        weight: float = None,
        description: str = ""
    ) -> tuple[Optional[Item], str]:
        """
        Add a looted item to inventory, checking capacity.

        Returns (item, status_message).
        Item is None if it couldn't be added.
        """
        inv = self.get_inventory(owner_id)
        if not inv:
            return None, "No inventory"

        # Estimate weight if not provided
        if weight is None:
            from inventory import estimate_weight, _guess_category
            category = _guess_category(item_name)
            weight = estimate_weight(item_name, category)

        # Check capacity
        can_add, reason = inv.can_add_item(weight)
        if not can_add:
            return None, reason

        # Guess category
        from inventory import _guess_category
        category = _guess_category(item_name)

        # Add the item
        item = inv.add_item(
            name=item_name,
            category=category,
            weight=weight,
            description=description or f"Scavenged {item_name.lower()}",
            acquired_from="Scavenged"
        )
        self._save()

        return item, f"Added {item_name} ({weight:.1f} lbs)"

    def get_loot_status(self, owner_id: str) -> dict:
        """
        Get loot mode status - is container equipped, capacity remaining, etc.
        """
        inv = self.get_inventory(owner_id)
        status = {
            'owner_id': owner_id,
            'loot_mode': False,
            'container': None,
            'capacity': 0.0,
            'current_weight': 0.0,
            'available': 0.0,
        }

        if inv:
            container = inv.get_equipped_container()
            if container:
                status['loot_mode'] = True
                status['container'] = container.name
                status['capacity'] = container.capacity
                status['current_weight'] = inv.get_total_weight()
                status['available'] = inv.get_available_capacity()

        return status


# =============================================================================
# Global instance
# =============================================================================

_tracker: Optional[InventoryTracker] = None


def get_inventory_tracker() -> InventoryTracker:
    """Get the global inventory tracker."""
    global _tracker
    if _tracker is None:
        _tracker = InventoryTracker()
    return _tracker


def init_inventory_tracker(data_dir: Optional[Path] = None) -> InventoryTracker:
    """Initialize the global inventory tracker."""
    global _tracker
    _tracker = InventoryTracker(data_dir)
    return _tracker


# =============================================================================
# Narrative Parsing - Detect inventory changes from text
# =============================================================================

import re

# Patterns for detecting item acquisition
# Using non-greedy matching and stopping at common prepositions/punctuation
ACQUIRE_PATTERNS = [
    r"you (?:pick up|take|grab|pocket|collect|receive|find|obtain|acquire|get) (?:a |an |the |some )?(.+?)(?:\.|,|!|\?|$| and | from | in | on | at | under | behind | near )",
    r"(?:picks up|takes|grabs|pockets|collects|receives|finds|obtains|acquires|gets) (?:a |an |the |some )?(.+?)(?:\.|,|!|\?|$| and | from | in | on | at | under | behind | near )",
    r"(?:add|adds|added) (?:a |an |the |some )?(.+?) to (?:your |the )(?:pack|bag|inventory|backpack|pouch)",
    r"you now have (?:a |an |the |some )?(.+?)(?:\.|,|!|\?|$| in | on )",
    r"hands you (?:a |an |the |some )?(.+?)(?:\.|,|!|\?|$| and )",
    r"gives you (?:a |an |the |some )?(.+?)(?:\.|,|!|\?|$| and )",
]

# Patterns for detecting item loss
LOSE_PATTERNS = [
    r"you (?:drop|lose|give away|hand over|discard|throw away|abandon) (?:a |an |the |your |some )?(.+?)(?:\.|,|!|\?|$| to )",
    r"(?:drops|loses|gives away|hands over|discards|throws away|abandons) (?:a |an |the |some )?(.+?)(?:\.|,|!|\?|$| to )",
    r"you give (?:a |an |the |your |some )?(.+?) to ",
    r"(?:the |your )(.+?) (?:breaks|shatters|crumbles|is destroyed|is lost)",
]

# Patterns for consumable use (item is removed)
CONSUME_PATTERNS = [
    r"you (?:drink|eat|consume|use up|apply) (?:a |an |the |your |some )?(.+?)(?:\.|,|!|\?|$)",
    r"(?:drinks|eats|consumes|uses up|applies) (?:a |an |the |some )?(.+?)(?:\.|,|!|\?|$)",
]

# Words that indicate the match is probably not an item
NON_ITEM_WORDS = {
    # Abstract/temporal
    "moment", "breath", "look", "step", "seat", "rest", "break", "chance",
    "opportunity", "time", "while", "second", "minute", "hour", "day",
    "notice", "glance", "peek", "interest", "action", "turn", "liking",
    "deep breath", "closer look", "quick look", "good look", "better look",
    # Pronouns and possessives (common false positives)
    "yours", "mine", "hers", "his", "theirs", "ours",
    "yours briefly", "it back", "it away", "it down", "it up",
    # Body parts (commonly matched in narrative)
    "sleeve", "arm", "hand", "hands", "leg", "legs", "head", "shoulder",
    "shoulders", "finger", "fingers", "wrist", "elbow", "knee", "foot", "feet",
    "face", "hair", "ear", "ears", "eye", "eyes", "nose", "mouth", "chin",
    "chest", "back", "stomach", "hip", "hips", "neck", "throat", "jaw",
    # Common narrative phrases that get extracted
    "hold", "grip", "charge", "aim", "stance", "position", "cover", "place",
    "lead", "hint", "stock", "measure", "care", "note", "risk", "offense",
    # Verbs that get mismatched (from "gets shoved", "takes effect", etc.)
    "shoved", "pushed", "pulled", "thrown", "knocked", "slammed", "dragged",
    "effect", "effects", "control", "advantage", "initiative", "offense",
    # Environmental/structural (not portable items)
    "rail", "railing", "wall", "floor", "ceiling", "door", "window", "ledge",
    "ground", "path", "road", "street", "corner", "edge", "side",
    # Adverbs/modifiers that slip through
    "briefly", "quickly", "slowly", "carefully", "gently", "firmly",
}

# Category detection based on keywords
CATEGORY_KEYWORDS = {
    ItemCategory.WEAPON: ["sword", "dagger", "knife", "axe", "bow", "arrow", "spear", "staff", "wand", "gun", "pistol", "rifle", "blade", "mace", "hammer", "club"],
    ItemCategory.ARMOR: ["armor", "shield", "helmet", "helm", "gauntlet", "boots", "greaves", "chainmail", "plate", "vest", "jacket"],
    ItemCategory.CONSUMABLE: ["potion", "elixir", "food", "ration", "water", "drink", "bandage", "medicine", "herb", "pill", "ammo", "bullet", "arrow"],
    ItemCategory.TOOL: ["rope", "torch", "lantern", "pickaxe", "shovel", "hammer", "lockpick", "toolkit", "compass", "map", "flint", "tinderbox"],
    ItemCategory.KEY_ITEM: ["key", "letter", "note", "document", "scroll", "artifact", "relic", "amulet", "token", "badge", "pass", "ticket"],
    ItemCategory.TREASURE: ["gold", "coin", "gem", "jewel", "diamond", "ruby", "emerald", "ring", "necklace", "bracelet", "crown", "treasure"],
    ItemCategory.CLOTHING: ["cloak", "robe", "shirt", "pants", "dress", "hat", "gloves", "scarf", "cape", "tunic"],
    ItemCategory.CONTAINER: ["bag", "pouch", "sack", "chest", "box", "backpack", "case", "crate"],
}


def _clean_item_name(raw_name: str) -> Optional[str]:
    """Clean and validate an extracted item name."""
    if not raw_name:
        return None

    # Clean up whitespace and common suffixes
    name = raw_name.strip().lower()

    # Remove trailing punctuation and common non-item continuations
    name = re.sub(r'[,.\-!?;:]+$', '', name)
    name = re.sub(r'\s+(and|or|but|then|before|after|while|when|as|from|into|onto).*$', '', name)

    # Remove leading articles that might have slipped through
    name = re.sub(r'^(a |an |the |some |your |my )', '', name)

    name = name.strip()

    # Reject if too short, too long, or contains non-item words
    if len(name) < 2 or len(name) > 50:
        return None

    # Check exact match against non-item words
    if name in NON_ITEM_WORDS:
        return None

    # Check if ANY word in the name is a non-item word (catches "yours briefly", "shoved hard", etc.)
    words = name.split()
    for word in words:
        if word in NON_ITEM_WORDS:
            return None

    # Reject single-word items that are likely verbs or common words
    if len(words) == 1 and len(name) < 4:
        return None  # Too short to be a real item name

    # Reject if it's just pronouns or common words
    if name in ["it", "them", "this", "that", "something", "anything", "nothing", "everything"]:
        return None

    # Capitalize properly
    return name.title()


def _guess_category(item_name: str) -> ItemCategory:
    """Guess the item category based on keywords."""
    name_lower = item_name.lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in name_lower:
                return category

    return ItemCategory.MISC


# =============================================================================
# Weight Estimation - Realistic weights for items (in lbs)
# =============================================================================

# Specific item weights (name -> lbs)
ITEM_WEIGHTS = {
    # Weapons - melee
    'knife': 0.5, 'dagger': 0.5, 'pocket knife': 0.3, 'hunting knife': 0.8,
    'machete': 1.5, 'sword': 3.0, 'longsword': 3.5, 'katana': 2.5,
    'axe': 4.0, 'hatchet': 1.5, 'fire axe': 5.0, 'battle axe': 6.0,
    'crowbar': 3.0, 'baseball bat': 2.0, 'hammer': 1.5, 'sledgehammer': 10.0,
    'pipe': 2.0, 'spear': 4.0, 'staff': 3.0, 'club': 2.5,
    'cast iron pan': 8.0, 'cleaver': 1.5, 'wrench': 2.0,
    # Weapons - ranged
    'pistol': 2.0, '9mm pistol': 2.0, 'revolver': 2.5, 'service pistol': 2.0,
    'rifle': 7.0, 'hunting rifle': 8.0, 'shotgun': 7.5, 'm4 carbine': 6.5,
    'crossbow': 6.0, 'compound bow': 4.0, 'longbow': 2.5,
    # Ammo
    'bullets': 0.5, 'ammo': 0.5, 'arrows': 1.0, 'quiver': 2.0,
    # Containers
    'backpack': 2.0, 'bag': 1.0, 'pouch': 0.3, 'satchel': 1.5,
    'duffel bag': 2.5, 'messenger bag': 1.5, 'fanny pack': 0.5,
    # Medical
    'first aid kit': 2.0, 'medical kit': 3.0, 'bandages': 0.3,
    'antiseptic': 0.5, 'painkillers': 0.1, 'antibiotics': 0.2,
    'suture kit': 0.5, 'splint': 0.5,
    # Food & Water
    'can of food': 1.0, 'canned food': 1.0, 'rations': 2.0,
    'water bottle': 2.0, 'canteen': 2.5, 'mre': 1.5,
    'protein bar': 0.2, 'jerky': 0.3, 'dried fruit': 0.3,
    # Tools
    'flashlight': 0.5, 'lantern': 2.0, 'lighter': 0.1, 'matches': 0.1,
    'rope': 3.0, 'paracord': 1.0, 'duct tape': 0.5,
    'multi-tool': 0.5, 'lockpicks': 0.2, 'compass': 0.2,
    'binoculars': 1.5, 'radio': 1.0, 'walkie-talkie': 0.8,
    'flare': 0.3, 'flare gun': 1.5,
    # Clothing/Gear
    'jacket': 2.0, 'coat': 3.0, 'boots': 3.0, 'gloves': 0.3,
    'hat': 0.3, 'helmet': 3.0, 'gas mask': 2.0, 'goggles': 0.3,
    'sleeping bag': 4.0, 'blanket': 2.0, 'tarp': 2.0,
    # Misc
    'notebook': 0.5, 'journal': 0.5, 'sketchbook': 0.8,
    'map': 0.1, 'keys': 0.2, 'watch': 0.1,
}

# Category default weights (fallback)
CATEGORY_DEFAULT_WEIGHTS = {
    ItemCategory.WEAPON: 2.0,
    ItemCategory.ARMOR: 5.0,
    ItemCategory.CONSUMABLE: 0.5,
    ItemCategory.TOOL: 1.0,
    ItemCategory.KEY_ITEM: 0.2,
    ItemCategory.TREASURE: 0.5,
    ItemCategory.CLOTHING: 1.0,
    ItemCategory.CONTAINER: 1.5,
    ItemCategory.MISC: 0.5,
}

# Container capacities (name -> lbs they can hold)
CONTAINER_CAPACITIES = {
    'backpack': 25.0,
    'large backpack': 35.0,
    'small backpack': 15.0,
    'duffel bag': 30.0,
    'messenger bag': 15.0,
    'satchel': 10.0,
    'bag': 10.0,
    'pouch': 3.0,
    'belt pouch': 2.0,
    'fanny pack': 5.0,
    'pocket': 1.0,
}


def estimate_weight(item_name: str, category: ItemCategory = None) -> float:
    """
    Estimate the weight of an item based on its name and category.
    Returns weight in lbs.
    """
    name_lower = item_name.lower().strip()

    # Check exact matches first
    if name_lower in ITEM_WEIGHTS:
        return ITEM_WEIGHTS[name_lower]

    # Check partial matches
    for known_item, weight in ITEM_WEIGHTS.items():
        if known_item in name_lower or name_lower in known_item:
            return weight

    # Fall back to category default
    if category:
        return CATEGORY_DEFAULT_WEIGHTS.get(category, 0.5)

    # Guess category and use that default
    guessed_cat = _guess_category(item_name)
    return CATEGORY_DEFAULT_WEIGHTS.get(guessed_cat, 0.5)


def estimate_capacity(container_name: str) -> float:
    """
    Estimate the capacity of a container.
    Returns capacity in lbs.
    """
    name_lower = container_name.lower().strip()

    # Check exact matches
    if name_lower in CONTAINER_CAPACITIES:
        return CONTAINER_CAPACITIES[name_lower]

    # Check partial matches
    for known_container, capacity in CONTAINER_CAPACITIES.items():
        if known_container in name_lower:
            return capacity

    # Default for unknown containers
    return 10.0


def parse_inventory_changes(text: str) -> Dict[str, List[str]]:
    """
    Parse text for inventory changes.

    Returns:
        {
            'acquired': ['Rusty Key', 'Healing Potion'],
            'lost': ['Old Map'],
            'consumed': ['Bandage']
        }
    """
    text_lower = text.lower()
    changes = {
        'acquired': [],
        'lost': [],
        'consumed': [],
    }

    # Check acquisition patterns
    for pattern in ACQUIRE_PATTERNS:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            item = _clean_item_name(match.group(1))
            if item and item not in changes['acquired']:
                changes['acquired'].append(item)

    # Check loss patterns
    for pattern in LOSE_PATTERNS:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            item = _clean_item_name(match.group(1))
            if item and item not in changes['lost']:
                changes['lost'].append(item)

    # Check consume patterns
    for pattern in CONSUME_PATTERNS:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            item = _clean_item_name(match.group(1))
            if item and item not in changes['consumed']:
                changes['consumed'].append(item)

    return changes


def apply_narrative_changes(
    tracker: InventoryTracker,
    owner_id: str,
    owner_name: str,
    text: str,
    source: str = "narrative"
) -> Dict[str, List[str]]:
    """
    Parse narrative text and apply inventory changes.

    Args:
        tracker: The inventory tracker
        owner_id: Character ID to modify
        owner_name: Character name (for creating inventory if needed)
        text: The narrative text to parse
        source: Where this came from (for logging)

    Returns:
        Dict of changes that were applied
    """
    changes = parse_inventory_changes(text)

    # Ensure inventory exists
    tracker.get_or_create_inventory(owner_id, owner_name)

    applied = {
        'acquired': [],
        'lost': [],
        'consumed': [],
    }

    # Add acquired items (but skip if similar item already exists - likely a reference, not acquisition)
    for item_name in changes['acquired']:
        # Check for fuzzy duplicates first
        inv = tracker.get_inventory(owner_id)
        if inv:
            existing = inv.find_item_fuzzy(item_name)
            if existing:
                print(f"[Inventory] Skipping '{item_name}' - similar item '{existing.name}' already exists (reference, not new item)")
                continue

        category = _guess_category(item_name)
        item = tracker.add_item(
            owner_id=owner_id,
            item_name=item_name,
            category=category.value,
            quantity=1,
            acquired_from=source,
        )
        if item:
            applied['acquired'].append(item_name)
            print(f"[Inventory] {owner_name} acquired: {item_name} ({category.value})")

    # Remove lost items
    inv = tracker.get_inventory(owner_id)
    if inv:
        for item_name in changes['lost'] + changes['consumed']:
            # Find the item by name
            existing = inv.find_item_by_name(item_name)
            if existing:
                tracker.remove_item(owner_id, existing.id, 1)
                if item_name in changes['lost']:
                    applied['lost'].append(item_name)
                else:
                    applied['consumed'].append(item_name)
                print(f"[Inventory] {owner_name} lost: {item_name}")

    return applied
