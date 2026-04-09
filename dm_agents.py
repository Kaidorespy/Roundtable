"""
DM Support Agents - Background Ollama-powered systems that assist the DM.

These agents run in the background, monitoring and tracking game state,
providing the DM with information and craft advice without the DM having
to manually track everything.
"""

import json
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime


# =============================================================================
# INVENTORY KEEPER
# =============================================================================
# Tracks who has what. No pulling cans of beans when you should be starving.

@dataclass
class InventoryItem:
    """A single item in someone's inventory."""
    name: str
    quantity: int = 1
    description: str = ""
    acquired_at: str = ""  # ISO timestamp
    acquired_from: str = ""  # How/where they got it


@dataclass
class CharacterInventory:
    """Inventory for a single character."""
    character_id: str
    character_name: str
    items: Dict[str, InventoryItem] = field(default_factory=dict)

    def add_item(self, name: str, quantity: int = 1, description: str = "", source: str = ""):
        """Add or increment an item."""
        key = name.lower().strip()
        if key in self.items:
            self.items[key].quantity += quantity
        else:
            self.items[key] = InventoryItem(
                name=name,
                quantity=quantity,
                description=description,
                acquired_at=datetime.now().isoformat(),
                acquired_from=source
            )

    def remove_item(self, name: str, quantity: int = 1) -> bool:
        """Remove or decrement an item. Returns False if they don't have enough."""
        key = name.lower().strip()
        if key not in self.items:
            return False
        if self.items[key].quantity < quantity:
            return False
        self.items[key].quantity -= quantity
        if self.items[key].quantity <= 0:
            del self.items[key]
        return True

    def has_item(self, name: str, quantity: int = 1) -> bool:
        """Check if character has at least this many of an item."""
        key = name.lower().strip()
        if key not in self.items:
            return False
        return self.items[key].quantity >= quantity

    def list_items(self) -> List[str]:
        """Get a simple list of items."""
        return [f"{item.name} x{item.quantity}" for item in self.items.values()]


class InventoryKeeper:
    """
    Background agent that tracks inventory for all characters in a room.

    Monitors conversation for:
    - Item acquisitions ("I pick up the sword", "She hands me a key")
    - Item usage ("I use the health potion", "I fire my last bullet")
    - Item loss ("I drop my backpack", "The thief steals my wallet")

    The DM can query this at any time to verify player claims.
    """

    def __init__(self, room_id: str):
        self.room_id = room_id
        self.inventories: Dict[str, CharacterInventory] = {}
        self.transaction_log: List[Dict] = []  # Audit trail

    def get_or_create_inventory(self, character_id: str, character_name: str) -> CharacterInventory:
        """Get inventory for a character, creating if needed."""
        if character_id not in self.inventories:
            self.inventories[character_id] = CharacterInventory(
                character_id=character_id,
                character_name=character_name
            )
        return self.inventories[character_id]

    def log_transaction(self, action: str, character_id: str, item: str, quantity: int, context: str):
        """Log an inventory change for audit trail."""
        self.transaction_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,  # "acquire", "use", "drop", "give", "lose"
            "character_id": character_id,
            "item": item,
            "quantity": quantity,
            "context": context  # The message that triggered this
        })

    async def analyze_message(self, message: str, speaker_id: str, speaker_name: str,
                              ollama_generate_func) -> Optional[Dict]:
        """
        Analyze a message for inventory changes using Ollama.

        Returns dict with any detected changes, or None if no inventory action.
        """
        prompt = f"""Analyze this roleplay message for inventory changes.

Message from {speaker_name}: "{message}"

Look for:
1. ACQUIRING items (picking up, receiving, finding, buying, stealing)
2. USING items (consuming, firing, activating - item may be depleted)
3. DROPPING items (leaving behind, giving away, losing)
4. MENTIONING items they claim to have (verify against known inventory)

Respond in JSON format:
{{
    "has_inventory_action": true/false,
    "actions": [
        {{
            "type": "acquire|use|drop|give|mention",
            "item": "item name",
            "quantity": 1,
            "recipient": "character name if giving to someone",
            "depleted": true/false (if used item is consumed/gone),
            "context": "brief description of what happened"
        }}
    ]
}}

If no inventory actions, respond: {{"has_inventory_action": false, "actions": []}}
"""

        try:
            response = await ollama_generate_func(prompt, model="llama3.2")
            # Parse the JSON response
            result = json.loads(response)
            return result
        except Exception as e:
            print(f"[InventoryKeeper] Analysis error: {e}")
            return None

    def apply_changes(self, analysis: Dict, speaker_id: str, speaker_name: str):
        """Apply detected inventory changes."""
        if not analysis or not analysis.get("has_inventory_action"):
            return

        inventory = self.get_or_create_inventory(speaker_id, speaker_name)

        for action in analysis.get("actions", []):
            action_type = action.get("type")
            item = action.get("item", "unknown")
            quantity = action.get("quantity", 1)
            context = action.get("context", "")

            if action_type == "acquire":
                inventory.add_item(item, quantity, source=context)
                self.log_transaction("acquire", speaker_id, item, quantity, context)

            elif action_type == "use":
                if action.get("depleted", False):
                    inventory.remove_item(item, quantity)
                    self.log_transaction("use", speaker_id, item, quantity, context)

            elif action_type == "drop":
                inventory.remove_item(item, quantity)
                self.log_transaction("drop", speaker_id, item, quantity, context)

            elif action_type == "give":
                recipient = action.get("recipient")
                if recipient and inventory.remove_item(item, quantity):
                    self.log_transaction("give", speaker_id, item, quantity, f"to {recipient}: {context}")
                    # Note: recipient's inventory would need to be updated separately

            elif action_type == "mention":
                # Player claims to have something - flag for DM if they don't
                if not inventory.has_item(item, quantity):
                    self.log_transaction("disputed_mention", speaker_id, item, quantity,
                                        f"CLAIMED but not in inventory: {context}")

    def get_summary(self) -> str:
        """Get a summary of all inventories for the DM."""
        if not self.inventories:
            return "No inventory tracked yet."

        lines = ["**Current Inventories:**\n"]
        for inv in self.inventories.values():
            items = inv.list_items()
            if items:
                lines.append(f"**{inv.character_name}:** {', '.join(items)}")
            else:
                lines.append(f"**{inv.character_name}:** (empty)")

        return "\n".join(lines)

    def check_item(self, character_id: str, item_name: str) -> str:
        """DM query: Does this character have this item?"""
        if character_id not in self.inventories:
            return f"No inventory tracked for this character."

        inv = self.inventories[character_id]
        if inv.has_item(item_name):
            item = inv.items.get(item_name.lower().strip())
            return f"Yes, {inv.character_name} has {item.name} x{item.quantity}"
        else:
            return f"No, {inv.character_name} does not have '{item_name}' in their inventory."

    def to_dict(self) -> Dict:
        """Serialize for storage."""
        return {
            "room_id": self.room_id,
            "inventories": {
                char_id: {
                    "character_id": inv.character_id,
                    "character_name": inv.character_name,
                    "items": {k: asdict(v) for k, v in inv.items.items()}
                }
                for char_id, inv in self.inventories.items()
            },
            "transaction_log": self.transaction_log
        }


# =============================================================================
# DRAMATURGE - Literature Consultant
# =============================================================================
# Craft advice, not plot. How did Author X handle this kind of tension?

class Dramaturge:
    """
    The DM's literature consultant.

    Feed it: genre + mood + story beat you're trying to land
    Get back: craft advice based on literary techniques

    This is NOT a plot generator. It's a technique advisor.
    "How does The Martian handle isolation without making it depressing?"
    "What does good survival horror tension feel like?"
    "How do I land a betrayal that feels earned, not cheap?"
    """

    # Reference library - books/films known for specific techniques
    TECHNIQUE_REFERENCES = {
        "isolation": ["The Martian", "Cast Away", "I Am Legend", "Moon", "127 Hours"],
        "survival_horror": ["The Road", "Bird Box", "A Quiet Place", "The Descent"],
        "tension": ["No Country for Old Men", "Sicario", "Prisoners", "Zodiac"],
        "betrayal": ["Gone Girl", "The Departed", "Oldboy", "Atonement"],
        "mystery": ["Gone Girl", "Knives Out", "Chinatown", "Memento"],
        "found_family": ["Guardians of the Galaxy", "The Last of Us", "Firefly"],
        "redemption": ["Shawshank Redemption", "Les Misérables", "A Christmas Carol"],
        "loss": ["Manchester by the Sea", "Ordinary People", "Three Billboards"],
        "romance": ["Pride and Prejudice", "When Harry Met Sally", "Before Sunrise"],
        "cosmic_horror": ["Annihilation", "The Thing", "Event Horizon", "Color Out of Space"],
        "noir": ["Chinatown", "L.A. Confidential", "Blade Runner", "Sin City"],
        "comedy": ["Hot Fuzz", "The Grand Budapest Hotel", "Airplane!", "Blazing Saddles"],
    }

    def __init__(self):
        self.consultation_history: List[Dict] = []

    async def consult(self, genre: str, mood: str, story_beat: str,
                      ollama_generate_func, specific_work: str = None) -> str:
        """
        Get craft advice for a specific storytelling challenge.

        Args:
            genre: The genre context (zombie, noir, fantasy, etc.)
            mood: The emotional texture you're going for
            story_beat: What you're trying to accomplish ("build tension", "land a betrayal")
            specific_work: Optional - a specific book/film to reference
            ollama_generate_func: The Ollama generation function

        Returns:
            Craft advice - technique, not plot
        """

        # Find relevant reference works
        references = []
        for technique, works in self.TECHNIQUE_REFERENCES.items():
            if technique in story_beat.lower() or technique in mood.lower():
                references.extend(works)

        if specific_work:
            references.insert(0, specific_work)

        references = list(dict.fromkeys(references))[:5]  # Dedupe, limit to 5

        prompt = f"""You are a dramaturge - a literary craft consultant for storytellers.

A game master is running a {genre} story with a {mood} mood.
They're trying to: {story_beat}

{"Reference works to consider: " + ", ".join(references) if references else ""}

Provide CRAFT ADVICE, not plot suggestions. Focus on:
- TECHNIQUE: How do skilled authors/filmmakers achieve this effect?
- PACING: When and how to deploy this beat for maximum impact
- SUBTEXT: What should be happening beneath the surface?
- PITFALLS: Common mistakes to avoid
- SPECIFIC TACTICS: Concrete things the GM can do in the next scene

Be concise but insightful. Talk about HOW to do it, not WHAT should happen.
Give the GM tools, not a script.
"""

        try:
            advice = await ollama_generate_func(prompt, model="llama3.2")

            # Log the consultation
            self.consultation_history.append({
                "timestamp": datetime.now().isoformat(),
                "genre": genre,
                "mood": mood,
                "story_beat": story_beat,
                "references": references,
                "advice": advice[:500] + "..." if len(advice) > 500 else advice
            })

            return advice
        except Exception as e:
            return f"Dramaturge consultation failed: {e}"

    async def analyze_technique(self, work: str, technique: str,
                                 ollama_generate_func) -> str:
        """
        Deep dive into how a specific work handles a specific technique.

        "How does The Martian handle isolation without depression?"
        """
        prompt = f"""Analyze how "{work}" handles {technique}.

Focus on CRAFT - the specific techniques used:
- What structural choices support this?
- What does the author/filmmaker do that's counterintuitive?
- What would a lesser work do wrong here?
- What can a GM steal for their own game?

Be specific. Quote or reference actual moments if relevant.
This is craft advice for a game master, not a book report.
"""

        try:
            return await ollama_generate_func(prompt, model="llama3.2")
        except Exception as e:
            return f"Analysis failed: {e}"

    async def suggest_references(self, challenge: str, ollama_generate_func) -> str:
        """
        Suggest books/films that handle a particular challenge well.
        """
        prompt = f"""A game master needs help with: {challenge}

Suggest 3-5 books, films, or TV shows that handle this particularly well.
For each, briefly explain WHY it's a good reference for this specific challenge.

Focus on works that offer LEARNABLE TECHNIQUES, not just good examples.
"""

        try:
            return await ollama_generate_func(prompt, model="llama3.2")
        except Exception as e:
            return f"Suggestion failed: {e}"


# =============================================================================
# ROOM DM STATE
# =============================================================================
# Container for all DM support agents for a room

class RoomDMState:
    """
    Container for all DM support systems for a single room.
    """

    def __init__(self, room_id: str):
        self.room_id = room_id
        self.inventory_keeper = InventoryKeeper(room_id)
        self.dramaturge = Dramaturge()
        self.created_at = datetime.now().isoformat()

        # Future agents will go here:
        # self.time_keeper = TimeKeeper(room_id)
        # self.faction_tracker = FactionTracker(room_id)
        # self.npc_manager = NPCManager(room_id)
        # self.thread_weaver = ThreadWeaver(room_id)  # Through-threads

    def to_dict(self) -> Dict:
        """Serialize the full DM state."""
        return {
            "room_id": self.room_id,
            "created_at": self.created_at,
            "inventory": self.inventory_keeper.to_dict(),
            "dramaturge_history": self.dramaturge.consultation_history
        }


# Global registry of room DM states
_room_dm_states: Dict[str, RoomDMState] = {}


def get_dm_state(room_id: str) -> RoomDMState:
    """Get or create DM state for a room."""
    if room_id not in _room_dm_states:
        _room_dm_states[room_id] = RoomDMState(room_id)
    return _room_dm_states[room_id]


def clear_dm_state(room_id: str):
    """Clear DM state for a room (when room is cleared/deleted)."""
    if room_id in _room_dm_states:
        del _room_dm_states[room_id]
