"""
Relationship Map - Visual representation of who knows what about who.

"We're making a world liver here. Not a god machine."

This surfaces existing relationship data from:
- NPCMemory (trust levels, trading reputation, combat history)
- Grudges (annoyed → nemesis)
- Partner hidden traits (secrets, wounds, wants, fears)

Character-facing only. Shows what YOUR character knows.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import debug_logger as dbg


class RelationshipType(Enum):
    """Types of relationships between entities."""
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    TRUSTED = "trusted"
    ROMANTIC = "romantic"
    HOSTILE = "hostile"
    NEMESIS = "nemesis"
    FAMILY = "family"
    BUSINESS = "business"
    UNKNOWN = "unknown"


@dataclass
class RelationshipNode:
    """A node in the relationship map (a character)."""
    id: str
    name: str
    node_type: str  # "player", "partner", "npc"
    avatar: str = ""
    color: str = "#6b7280"

    # Position hints (for layout)
    x: Optional[float] = None
    y: Optional[float] = None

    # Character info (what the viewer knows)
    known_role: str = ""
    known_traits: List[str] = field(default_factory=list)

    # Hidden from player, visible to DM
    secrets: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RelationshipEdge:
    """An edge connecting two nodes (a relationship)."""
    source_id: str
    target_id: str

    # Relationship from source's perspective
    relationship_type: RelationshipType = RelationshipType.NEUTRAL
    trust_level: float = 0.0  # -1 to 1

    # What's known about this relationship
    label: str = ""  # "allies", "rivals", "partners", etc.
    description: str = ""  # More detail

    # Visual hints
    strength: float = 0.5  # Line thickness (0-1)
    color: str = "#6b7280"
    bidirectional: bool = True  # Does target feel the same?

    # For grudges
    is_grudge: bool = False
    grudge_severity: Optional[str] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["relationship_type"] = self.relationship_type.value
        return d


@dataclass
class RelationshipMap:
    """The complete relationship map for a room/world."""
    room_id: str
    nodes: List[RelationshipNode] = field(default_factory=list)
    edges: List[RelationshipEdge] = field(default_factory=list)

    # Viewer context
    viewer_id: str = ""  # Who is viewing this map
    is_dm_view: bool = False  # DM sees everything

    def to_dict(self) -> Dict:
        return {
            "room_id": self.room_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "viewer_id": self.viewer_id,
            "is_dm_view": self.is_dm_view,
        }


class RelationshipMapBuilder:
    """
    Builds relationship maps from existing data.

    Pulls from:
    - Partners (companions in the room)
    - NPCs (from NPC registry)
    - Player character
    """

    def __init__(self, data_store, npc_registry=None):
        self.data_store = data_store
        self.npc_registry = npc_registry

    def build_map(
        self,
        room_id: str,
        viewer_id: str = "user",
        is_dm_view: bool = False
    ) -> RelationshipMap:
        """Build a relationship map for a room."""
        rel_map = RelationshipMap(
            room_id=room_id,
            viewer_id=viewer_id,
            is_dm_view=is_dm_view,
        )

        room = self.data_store.get_room(room_id)
        if not room:
            return rel_map

        all_partners = self.data_store.get_partners()
        partners_in_room = room.get_partners_in_room(all_partners)

        # Add player node
        from config import settings
        player_node = RelationshipNode(
            id="user",
            name=settings.user_name,
            node_type="player",
            avatar="👤",
            color="#3b82f6",
            known_role="Player Character",
        )
        rel_map.nodes.append(player_node)

        # Add partner nodes
        for partner in partners_in_room:
            node = self._partner_to_node(partner, is_dm_view)
            rel_map.nodes.append(node)

        # Add NPC nodes (if registry available)
        if self.npc_registry:
            npcs = self.npc_registry.get_npcs_in_world(room_id)
            for npc in npcs:
                if npc.is_alive:  # Only living NPCs
                    node = self._npc_to_node(npc, is_dm_view)
                    rel_map.nodes.append(node)

        # Build edges from NPC memories and grudges
        self._build_edges(rel_map, partners_in_room, is_dm_view)

        return rel_map

    def _partner_to_node(self, partner, is_dm_view: bool) -> RelationshipNode:
        """Convert a Partner to a RelationshipNode."""
        known_traits = []
        secrets = []

        # Extract traits from character description
        if partner.get_character():
            # First sentence or so as known trait
            desc = partner.get_character()
            if len(desc) > 100:
                desc = desc[:100] + "..."
            known_traits.append(desc)

        # DM sees hidden traits
        if is_dm_view:
            if partner.secret:
                secrets.append(f"SECRET: {partner.secret}")
            if partner.wound:
                secrets.append(f"WOUND: {partner.wound}")
            if partner.want:
                secrets.append(f"WANT: {partner.want}")
            if partner.fear:
                secrets.append(f"FEAR: {partner.fear}")

        return RelationshipNode(
            id=partner.id,
            name=partner.name,
            node_type="partner",
            avatar=partner.avatar,
            color=partner.color,
            known_role=partner.current_role if hasattr(partner, 'current_role') else "",
            known_traits=known_traits,
            secrets=secrets,
        )

    def _npc_to_node(self, npc, is_dm_view: bool) -> RelationshipNode:
        """Convert an NPC to a RelationshipNode."""
        known_traits = []
        secrets = []

        if npc.personality:
            known_traits.append(npc.personality)
        if npc.current_role:
            known_traits.append(f"Role: {npc.current_role}")

        # DM sees hidden depths
        if is_dm_view:
            if npc.secret:
                secrets.append(f"SECRET: {npc.secret}")
            if npc.wound:
                secrets.append(f"WOUND: {npc.wound}")
            if npc.want:
                secrets.append(f"WANT: {npc.want}")
            if npc.fear:
                secrets.append(f"FEAR: {npc.fear}")

            # Show nemeses
            nemeses = npc.get_nemeses()
            for grudge in nemeses:
                secrets.append(f"NEMESIS: {grudge.target_name} - {grudge.original_reason}")

        # State-based color
        state_colors = {
            "ephemeral": "#6b7280",
            "residue": "#8b5cf6",
            "agency": "#f59e0b",
            "soul": "#ef4444",
            "worldwalker": "#ec4899",
        }
        color = state_colors.get(npc.state.value, "#6b7280")

        return RelationshipNode(
            id=npc.id,
            name=npc.name,
            node_type="npc",
            avatar="🎭",
            color=color,
            known_role=npc.current_role or "",
            known_traits=known_traits,
            secrets=secrets,
        )

    def _build_edges(self, rel_map: RelationshipMap, partners, is_dm_view: bool):
        """Build relationship edges from memories and grudges."""

        # Get all node IDs for validation
        node_ids = {n.id for n in rel_map.nodes}

        # NPC relationships (from memories and grudges)
        if self.npc_registry:
            npcs = self.npc_registry.get_npcs_in_world(rel_map.room_id)

            for npc in npcs:
                if not npc.is_alive:
                    continue

                # Edges from NPC memories
                for player_id, memory in npc.memories.items():
                    if player_id not in node_ids:
                        continue

                    edge = self._memory_to_edge(npc.id, memory)
                    rel_map.edges.append(edge)

                # Edges from grudges (DM view or severe grudges)
                for grudge in npc.grudges:
                    if grudge.target_id not in node_ids:
                        continue

                    # Only show severe grudges to players, all to DM
                    if is_dm_view or grudge.severity.value in ["hostile", "nemesis"]:
                        edge = self._grudge_to_edge(npc.id, grudge)
                        rel_map.edges.append(edge)

        # Partner-to-partner relationships (inferred from room dynamics)
        # For now, just mark them as co-present
        for i, p1 in enumerate(partners):
            for p2 in partners[i+1:]:
                edge = RelationshipEdge(
                    source_id=p1.id,
                    target_id=p2.id,
                    relationship_type=RelationshipType.NEUTRAL,
                    trust_level=0.0,
                    label="acquaintances",
                    description="Present in the same room",
                    strength=0.2,
                    color="#6b7280",
                    bidirectional=True,
                )
                rel_map.edges.append(edge)

    def _memory_to_edge(self, npc_id: str, memory) -> RelationshipEdge:
        """Convert an NPCMemory to a RelationshipEdge."""
        # Determine relationship type from trust level
        if memory.trust_level >= 0.7:
            rel_type = RelationshipType.TRUSTED
            color = "#22c55e"
        elif memory.trust_level >= 0.3:
            rel_type = RelationshipType.FRIENDLY
            color = "#3b82f6"
        elif memory.trust_level <= -0.7:
            rel_type = RelationshipType.HOSTILE
            color = "#ef4444"
        elif memory.trust_level <= -0.3:
            rel_type = RelationshipType.HOSTILE
            color = "#f97316"
        else:
            rel_type = RelationshipType.NEUTRAL
            color = "#6b7280"

        # Build description from notable events
        description = ""
        if memory.notable_events:
            description = "; ".join(memory.notable_events[-3:])
        if memory.trading_reputation:
            description += f" (Trading: {memory.trading_reputation})"

        return RelationshipEdge(
            source_id=npc_id,
            target_id=memory.player_id,
            relationship_type=rel_type,
            trust_level=memory.trust_level,
            label=f"Trust: {memory.trust_level:.1f}",
            description=description,
            strength=abs(memory.trust_level) * 0.5 + 0.3,
            color=color,
            bidirectional=False,  # This is NPC's view of player
        )

    def _grudge_to_edge(self, npc_id: str, grudge) -> RelationshipEdge:
        """Convert a Grudge to a RelationshipEdge."""
        severity_colors = {
            "annoyed": "#fbbf24",
            "resentful": "#f97316",
            "hostile": "#ef4444",
            "nemesis": "#7f1d1d",
        }

        severity_strength = {
            "annoyed": 0.3,
            "resentful": 0.5,
            "hostile": 0.7,
            "nemesis": 1.0,
        }

        return RelationshipEdge(
            source_id=npc_id,
            target_id=grudge.target_id,
            relationship_type=RelationshipType.NEMESIS if grudge.is_nemesis() else RelationshipType.HOSTILE,
            trust_level=-1.0 if grudge.is_nemesis() else -0.7,
            label=grudge.severity.value.upper(),
            description=grudge.reason,
            strength=severity_strength.get(grudge.severity.value, 0.5),
            color=severity_colors.get(grudge.severity.value, "#ef4444"),
            bidirectional=False,
            is_grudge=True,
            grudge_severity=grudge.severity.value,
        )


# =============================================================================
# API Helper Functions
# =============================================================================

def get_relationship_map(
    data_store,
    room_id: str,
    npc_registry=None,
    viewer_id: str = "user",
    is_dm_view: bool = False
) -> Dict:
    """Get a relationship map as a dictionary (for API)."""
    builder = RelationshipMapBuilder(data_store, npc_registry)
    rel_map = builder.build_map(room_id, viewer_id, is_dm_view)
    return rel_map.to_dict()
