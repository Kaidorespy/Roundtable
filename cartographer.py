"""
Cartographer - World map and location tracking.

When a world is created, the Cartographer generates token-level lore:
- Major locations (cities, towns, spaceports - genre-dependent)
- Distances from starting position
- Basic descriptions

Once discovered, locations are PERMANENT. Canon forever.
The Cartographer can expand the map as players venture into unknown territory.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import debug_logger as dbg
from pathlib import Path
from enum import Enum
import json
import uuid
import math


class LocationType(Enum):
    """Types of locations (genre-flexible)."""
    # Universal
    SETTLEMENT = "settlement"       # City, town, village, outpost
    LANDMARK = "landmark"           # Mountain, river, forest, ruins
    DUNGEON = "dungeon"             # Cave, temple, crypt, lair
    ROAD = "road"                   # Path, highway, trade route

    # Fantasy
    CASTLE = "castle"
    TEMPLE = "temple"
    TAVERN = "tavern"
    GUILD = "guild"

    # Sci-Fi
    SPACEPORT = "spaceport"
    STATION = "station"
    COLONY = "colony"
    OUTPOST = "outpost"

    # Modern/Post-Apocalyptic
    SAFEHOUSE = "safehouse"
    CAMP = "camp"
    BUNKER = "bunker"
    RUIN = "ruin"

    # Generic
    POINT_OF_INTEREST = "poi"
    UNKNOWN = "unknown"


class DiscoveryStatus(Enum):
    """How much the player knows about this location."""
    UNKNOWN = "unknown"         # Exists on map but player doesn't know
    RUMORED = "rumored"         # Heard about it, vague details
    KNOWN = "known"             # Knows it exists, basic info
    VISITED = "visited"         # Has been there
    EXPLORED = "explored"       # Knows it well


@dataclass
class Location:
    """A location in the world."""
    id: str
    name: str
    location_type: LocationType

    # Position (relative to starting point)
    # Using simple grid coordinates; (0,0) is the starting area
    x: float = 0.0
    y: float = 0.0
    distance_from_start: float = 0.0  # Calculated

    # Description
    short_description: str = ""      # One line
    full_description: str = ""       # Detailed (revealed on visit)

    # Discovery
    discovery_status: DiscoveryStatus = DiscoveryStatus.UNKNOWN
    discovered_at: Optional[str] = None
    discovered_by: Optional[str] = None

    # Connections
    connected_to: List[str] = field(default_factory=list)  # Location IDs
    travel_time_hours: Dict[str, float] = field(default_factory=dict)  # loc_id -> hours

    # NPCs and features
    notable_npcs: List[str] = field(default_factory=list)  # NPC IDs
    features: List[str] = field(default_factory=list)  # "has_market", "dangerous", etc.

    # Canon status
    is_canon: bool = True  # Once true, cannot be removed

    # Timestamps
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.id:
            self.id = str(uuid.uuid4())[:12]
        # Calculate distance from start
        self.distance_from_start = math.sqrt(self.x**2 + self.y**2)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "location_type": self.location_type.value,
            "x": self.x,
            "y": self.y,
            "distance_from_start": self.distance_from_start,
            "short_description": self.short_description,
            "full_description": self.full_description,
            "discovery_status": self.discovery_status.value,
            "discovered_at": self.discovered_at,
            "discovered_by": self.discovered_by,
            "connected_to": self.connected_to,
            "travel_time_hours": self.travel_time_hours,
            "notable_npcs": self.notable_npcs,
            "features": self.features,
            "is_canon": self.is_canon,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Location":
        data = data.copy()
        if "location_type" in data:
            data["location_type"] = LocationType(data["location_type"])
        if "discovery_status" in data:
            data["discovery_status"] = DiscoveryStatus(data["discovery_status"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class WorldMap:
    """The complete map for a world/room."""
    world_id: str
    genre: str = ""  # "fantasy", "scifi", "zombie", "noir", etc.

    # Starting location
    starting_location_id: Optional[str] = None
    starting_location_name: str = "Starting Area"

    # All locations
    locations: Dict[str, Location] = field(default_factory=dict)  # id -> Location

    # Map bounds (auto-expand as locations added)
    min_x: float = -10.0
    max_x: float = 10.0
    min_y: float = -10.0
    max_y: float = 10.0

    # Generation settings
    world_scale: str = "regional"  # "local", "regional", "continental", "planetary"

    # Metadata
    created_at: str = ""
    last_updated: str = ""

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        self.last_updated = now

    def add_location(self, location: Location) -> Location:
        """Add a location to the map."""
        self.locations[location.id] = location

        # Expand bounds if needed
        padding = 5.0
        if location.x < self.min_x + padding:
            self.min_x = location.x - padding
        if location.x > self.max_x - padding:
            self.max_x = location.x + padding
        if location.y < self.min_y + padding:
            self.min_y = location.y - padding
        if location.y > self.max_y - padding:
            self.max_y = location.y + padding

        self.last_updated = datetime.now().isoformat()
        return location

    def get_location(self, location_id: str) -> Optional[Location]:
        return self.locations.get(location_id)

    def get_discovered_locations(self) -> List[Location]:
        """Get locations the player knows about."""
        return [
            loc for loc in self.locations.values()
            if loc.discovery_status != DiscoveryStatus.UNKNOWN
        ]

    def get_nearby_locations(self, x: float, y: float, radius: float = 5.0) -> List[Location]:
        """Get locations within a radius of a point."""
        nearby = []
        for loc in self.locations.values():
            dist = math.sqrt((loc.x - x)**2 + (loc.y - y)**2)
            if dist <= radius:
                nearby.append(loc)
        return sorted(nearby, key=lambda l: math.sqrt((l.x - x)**2 + (l.y - y)**2))

    def get_travel_time(self, from_loc_id: str, to_loc_id: str) -> Optional[float]:
        """
        Calculate travel time between two locations in hours.

        Uses stored travel times if available, otherwise estimates based on distance.
        Assumes walking speed of ~3 units per hour.
        """
        from_loc = self.get_location(from_loc_id)
        to_loc = self.get_location(to_loc_id)

        if not from_loc or not to_loc:
            return None

        # Check for stored travel time
        if to_loc_id in from_loc.travel_time_hours:
            return from_loc.travel_time_hours[to_loc_id]

        # Calculate based on distance (3 units per hour walking)
        distance = math.sqrt((to_loc.x - from_loc.x)**2 + (to_loc.y - from_loc.y)**2)
        walking_speed = 3.0  # units per hour

        return round(distance / walking_speed, 1)

    def get_travel_context(self, current_loc_id: str = None) -> str:
        """
        Get travel information for DM context.

        Shows nearby destinations and travel times from current location.
        """
        if not self.locations:
            return ""

        discovered = self.get_discovered_locations()
        if not discovered:
            return ""

        lines = ["TRAVEL INFO:"]

        # If we know current location, show travel times from there
        current_loc = self.get_location(current_loc_id) if current_loc_id else None

        if current_loc:
            lines.append(f"Current location: {current_loc.name}")
            lines.append("Nearby destinations:")

            # Sort by distance
            others = [(loc, self.get_travel_time(current_loc_id, loc.id))
                     for loc in discovered if loc.id != current_loc_id]
            others.sort(key=lambda x: x[1] or 999)

            for loc, hours in others[:5]:  # Show 5 closest
                if hours:
                    lines.append(f"  • {loc.name}: ~{hours:.0f} hours walking")
        else:
            # Just list known locations with distances from start
            lines.append("Known locations (distance from start):")
            for loc in sorted(discovered, key=lambda l: l.distance_from_start)[:6]:
                hours = loc.distance_from_start / 3.0
                lines.append(f"  • {loc.name}: ~{hours:.0f} hours from start")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "world_id": self.world_id,
            "genre": self.genre,
            "starting_location_id": self.starting_location_id,
            "starting_location_name": self.starting_location_name,
            "locations": {k: v.to_dict() for k, v in self.locations.items()},
            "min_x": self.min_x,
            "max_x": self.max_x,
            "min_y": self.min_y,
            "max_y": self.max_y,
            "world_scale": self.world_scale,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "WorldMap":
        data = data.copy()
        if "locations" in data:
            data["locations"] = {
                k: Location.from_dict(v) for k, v in data["locations"].items()
            }
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class Cartographer:
    """
    The Cartographer - generates and tracks world geography.

    Responsibilities:
    - Generate initial world lore (major locations) on world creation
    - Track discovered vs unknown locations
    - Expand the map as players explore
    - Provide location info to DM
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".roundtable"
        self.maps_file = self.data_dir / "world_maps.json"
        self.maps: Dict[str, WorldMap] = {}
        self._load()

    def _load(self):
        """Load maps from disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.maps_file.exists():
            try:
                data = json.loads(self.maps_file.read_text())
                for world_id, map_data in data.items():
                    self.maps[world_id] = WorldMap.from_dict(map_data)
            except Exception as e:
                print(f"[Cartographer] Error loading maps: {e}")

    def _save(self):
        """Save maps to disk."""
        try:
            data = {k: v.to_dict() for k, v in self.maps.items()}
            self.maps_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[Cartographer] Error saving maps: {e}")

    def get_or_create_map(self, world_id: str, genre: str = "") -> WorldMap:
        """Get or create a world map."""
        if world_id not in self.maps:
            self.maps[world_id] = WorldMap(world_id=world_id, genre=genre)
            self._save()
        return self.maps[world_id]

    def get_map(self, world_id: str) -> Optional[WorldMap]:
        """Get a world map if it exists."""
        return self.maps.get(world_id)

    async def generate_initial_world(
        self,
        world_id: str,
        genre: str,
        setting_description: str,
        ollama_generate_func,
        num_locations: int = 5
    ) -> WorldMap:
        """
        Generate initial world lore for a new world.

        Creates the starting area and several nearby locations.
        """
        dbg.cartographer(f"▶ Generating world: {world_id} ({genre}) with {num_locations} locations")
        world_map = self.get_or_create_map(world_id, genre)
        world_map.genre = genre

        # Generate starting location
        starting_loc = await self._generate_starting_location(
            genre, setting_description, ollama_generate_func
        )
        world_map.add_location(starting_loc)
        world_map.starting_location_id = starting_loc.id
        world_map.starting_location_name = starting_loc.name
        dbg.cartographer(f"  ├─ Start: {starting_loc.name}")

        # Mark as discovered (player starts here)
        starting_loc.discovery_status = DiscoveryStatus.VISITED

        # Generate nearby locations
        for i in range(num_locations):
            # Place in a rough circle around start
            import random
            import math
            angle = (2 * math.pi * i / num_locations) + random.uniform(-0.3, 0.3)
            distance = random.uniform(3, 8)
            x = math.cos(angle) * distance
            y = math.sin(angle) * distance

            loc = await self._generate_location(
                genre, setting_description, x, y, ollama_generate_func
            )
            world_map.add_location(loc)
            dbg.cartographer(f"  ├─ Location: {loc.name} ({loc.location_type.value})")

            # Connect to starting location
            loc.connected_to.append(starting_loc.id)
            starting_loc.connected_to.append(loc.id)

            # Travel time based on distance
            travel_hours = distance * 0.5  # ~30 min per unit
            loc.travel_time_hours[starting_loc.id] = travel_hours
            starting_loc.travel_time_hours[loc.id] = travel_hours

        self._save()
        dbg.cartographer(f"✓ World generated: {len(world_map.locations)} total locations")
        return world_map

    async def _generate_starting_location(
        self,
        genre: str,
        setting: str,
        ollama_generate_func
    ) -> Location:
        """Generate the starting location."""
        prompt = f"""Generate a starting location for a {genre} story.

Setting context: {setting}

The starting location should be:
- A safe or neutral place where the story begins
- Appropriate for the genre ({genre})
- Memorable but not too grand

Respond in this exact format:
NAME: [location name]
TYPE: [settlement/tavern/camp/safehouse/station/etc]
SHORT: [one sentence description]
FEATURES: [comma-separated list of notable features]"""

        try:
            response = await ollama_generate_func(prompt, )
            return self._parse_location_response(response, 0, 0)
        except Exception as e:
            print(f"[Cartographer] Generation error: {e}")
            return Location(
                id=str(uuid.uuid4())[:12],
                name="Starting Area",
                location_type=LocationType.SETTLEMENT,
                x=0, y=0,
                short_description="Where your journey begins.",
                discovery_status=DiscoveryStatus.VISITED,
            )

    async def _generate_location(
        self,
        genre: str,
        setting: str,
        x: float,
        y: float,
        ollama_generate_func
    ) -> Location:
        """Generate a location at a specific position."""
        distance = math.sqrt(x**2 + y**2)

        # Direction for flavor
        direction = ""
        if y > abs(x): direction = "north"
        elif y < -abs(x): direction = "south"
        elif x > 0: direction = "east"
        else: direction = "west"

        prompt = f"""Generate a location for a {genre} story.

Setting context: {setting}
Direction from start: {direction}
Distance: {"nearby" if distance < 5 else "distant"}

Generate a location that:
- Fits the {genre} genre
- Could be a destination for travelers
- Has potential for adventure or story

Respond in this exact format:
NAME: [location name]
TYPE: [settlement/landmark/dungeon/road/castle/spaceport/camp/ruin/etc]
SHORT: [one sentence description]
FEATURES: [comma-separated list: dangerous, has_market, abandoned, etc]"""

        try:
            response = await ollama_generate_func(prompt, )
            return self._parse_location_response(response, x, y)
        except Exception as e:
            print(f"[Cartographer] Generation error: {e}")
            return Location(
                id=str(uuid.uuid4())[:12],
                name=f"Unknown {direction.title()} Location",
                location_type=LocationType.UNKNOWN,
                x=x, y=y,
                short_description=f"A mysterious place to the {direction}.",
            )

    def _parse_location_response(self, response: str, x: float, y: float) -> Location:
        """Parse LLM response into a Location."""
        lines = response.strip().split('\n')
        data = {}

        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                data[key.strip().upper()] = value.strip()

        name = data.get('NAME', 'Unknown Location')
        type_str = data.get('TYPE', 'poi').lower().replace(' ', '_')

        # Try to match location type
        try:
            loc_type = LocationType(type_str)
        except ValueError:
            # Try common mappings
            type_map = {
                'city': LocationType.SETTLEMENT,
                'town': LocationType.SETTLEMENT,
                'village': LocationType.SETTLEMENT,
                'inn': LocationType.TAVERN,
                'bar': LocationType.TAVERN,
                'forest': LocationType.LANDMARK,
                'mountain': LocationType.LANDMARK,
                'cave': LocationType.DUNGEON,
                'base': LocationType.OUTPOST,
                'fort': LocationType.CASTLE,
            }
            loc_type = type_map.get(type_str, LocationType.POINT_OF_INTEREST)

        features = []
        if 'FEATURES' in data:
            features = [f.strip() for f in data['FEATURES'].split(',') if f.strip()]

        return Location(
            id=str(uuid.uuid4())[:12],
            name=name,
            location_type=loc_type,
            x=x,
            y=y,
            short_description=data.get('SHORT', ''),
            features=features,
        )

    def discover_location(
        self,
        world_id: str,
        location_id: str,
        player_id: str,
        status: DiscoveryStatus = DiscoveryStatus.KNOWN
    ) -> Optional[Location]:
        """Mark a location as discovered by a player."""
        world_map = self.get_map(world_id)
        if not world_map:
            return None

        loc = world_map.get_location(location_id)
        if not loc:
            return None

        # Can only upgrade discovery status, never downgrade
        status_order = [
            DiscoveryStatus.UNKNOWN,
            DiscoveryStatus.RUMORED,
            DiscoveryStatus.KNOWN,
            DiscoveryStatus.VISITED,
            DiscoveryStatus.EXPLORED,
        ]

        current_idx = status_order.index(loc.discovery_status)
        new_idx = status_order.index(status)

        if new_idx > current_idx:
            loc.discovery_status = status
            if not loc.discovered_at:
                loc.discovered_at = datetime.now().isoformat()
                loc.discovered_by = player_id
            self._save()

        return loc

    def discover_from_text(
        self,
        world_id: str,
        text: str,
        player_id: str = "narrative"
    ) -> List[Location]:
        """
        Scan text for location names and mark them as discovered.

        When the DM mentions a location by name, the player now knows about it.
        - First mention = RUMORED (they've heard of it)
        - Detailed description = KNOWN (they know what it is)
        - "You arrive at..." = VISITED

        Returns list of newly discovered locations.
        """
        world_map = self.get_map(world_id)
        if not world_map or not world_map.locations:
            return []

        text_lower = text.lower()
        discovered = []

        for loc_id, loc in world_map.locations.items():
            # Skip already well-known locations
            if loc.discovery_status in [DiscoveryStatus.VISITED, DiscoveryStatus.EXPLORED]:
                continue

            loc_name_lower = loc.name.lower()

            # Check if location name is mentioned
            if loc_name_lower in text_lower:
                # Determine discovery level based on context
                # "arrive at" / "reach" / "enter" = VISITED
                arrival_phrases = ["arrive at", "arrived at", "reach ", "reached ",
                                   "enter ", "entered ", "you are in", "you're in",
                                   "standing in", "inside"]

                if any(phrase in text_lower for phrase in arrival_phrases):
                    status = DiscoveryStatus.VISITED
                # Detailed description suggests KNOWN
                elif len(text) > 100 and loc_name_lower in text_lower:
                    status = DiscoveryStatus.KNOWN
                # Brief mention = RUMORED
                else:
                    status = DiscoveryStatus.RUMORED

                updated = self.discover_location(world_id, loc_id, player_id, status)
                if updated:
                    discovered.append(updated)

        return discovered

    async def expand_map(
        self,
        world_id: str,
        direction: str,  # "north", "south", "east", "west"
        ollama_generate_func,
        num_locations: int = 2
    ) -> List[Location]:
        """
        Expand the map in a direction when players venture into unknown territory.

        Once generated, these locations are CANON FOREVER.
        """
        world_map = self.get_map(world_id)
        if not world_map:
            return []

        import random
        import math

        # Calculate expansion area
        if direction == "north":
            base_x, base_y = 0, world_map.max_y
            angle_range = (math.pi/4, 3*math.pi/4)
        elif direction == "south":
            base_x, base_y = 0, world_map.min_y
            angle_range = (-3*math.pi/4, -math.pi/4)
        elif direction == "east":
            base_x, base_y = world_map.max_x, 0
            angle_range = (-math.pi/4, math.pi/4)
        else:  # west
            base_x, base_y = world_map.min_x, 0
            angle_range = (3*math.pi/4, 5*math.pi/4)

        new_locations = []
        for i in range(num_locations):
            angle = random.uniform(*angle_range)
            distance = random.uniform(3, 7)
            x = base_x + math.cos(angle) * distance
            y = base_y + math.sin(angle) * distance

            loc = await self._generate_location(
                world_map.genre, "", x, y, ollama_generate_func
            )
            loc.is_canon = True  # Permanent
            world_map.add_location(loc)
            new_locations.append(loc)

        self._save()
        return new_locations

    def get_dm_context(self, world_id: str) -> str:
        """Get a context string for the DM about the world map."""
        world_map = self.get_map(world_id)
        if not world_map:
            return "No map data available for this world."

        lines = [f"=== World Map ({world_map.genre or 'unknown genre'}) ===\n"]

        # Starting location
        if world_map.starting_location_id:
            start = world_map.get_location(world_map.starting_location_id)
            if start:
                lines.append(f"Starting Area: {start.name}")
                lines.append(f"  {start.short_description}\n")

        # Discovered locations
        discovered = world_map.get_discovered_locations()
        if discovered:
            lines.append("Known Locations:")
            for loc in sorted(discovered, key=lambda l: l.distance_from_start):
                status_icon = {
                    "visited": "✓",
                    "explored": "★",
                    "known": "○",
                    "rumored": "?",
                }.get(loc.discovery_status.value, " ")
                lines.append(f"  [{status_icon}] {loc.name} ({loc.location_type.value})")
                if loc.short_description:
                    lines.append(f"      {loc.short_description}")

        # Hidden locations (DM only)
        hidden = [l for l in world_map.locations.values()
                  if l.discovery_status == DiscoveryStatus.UNKNOWN]
        if hidden:
            lines.append(f"\nUndiscovered Locations ({len(hidden)}):")
            for loc in hidden[:5]:  # Show first 5
                lines.append(f"  [?] {loc.name} - {loc.short_description[:50]}...")

        return "\n".join(lines)


# =============================================================================
# Global instance
# =============================================================================

_cartographer: Optional[Cartographer] = None


def get_cartographer() -> Cartographer:
    """Get the global cartographer instance."""
    global _cartographer
    if _cartographer is None:
        _cartographer = Cartographer()
    return _cartographer


def init_cartographer(data_dir: Optional[Path] = None) -> Cartographer:
    """Initialize the global cartographer."""
    global _cartographer
    _cartographer = Cartographer(data_dir)
    return _cartographer
