"""
World Texture - The dust in your nose, the silence between words.

"A world isn't real because things happen in it.
It's real because things COULD happen. And don't.
And you notice."

This module generates the ambient reality of a world:
- Sensory details (what does this moment smell/sound/feel like)
- Background murmur (fragments of conversation, signs of life)
- The weight of silence (tracking stillness, building tension)
- Object memory (items remember where they've been)
- Unfinished business (threads left dangling)
- Environmental mood (the quality of light, the texture of air)
"""

import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import debug_logger as dbg


class SenseType(Enum):
    """The five senses plus proprioception."""
    SIGHT = "sight"
    SOUND = "sound"
    SMELL = "smell"
    TOUCH = "touch"
    TASTE = "taste"
    FEELING = "feeling"  # The gut sense, the weight of a moment


@dataclass
class SensoryDetail:
    """A single sensory impression."""
    sense: SenseType
    description: str
    intensity: float = 0.5  # 0-1, how noticeable
    source: str = ""  # What's causing this
    is_absence: bool = False  # Is this about something MISSING?

    def to_dict(self) -> Dict:
        return {
            "sense": self.sense.value,
            "description": self.description,
            "intensity": self.intensity,
            "source": self.source,
            "is_absence": self.is_absence,
        }


@dataclass
class ObjectMemory:
    """
    The history attached to an object.

    Items remember. A knife isn't just a knife -
    it's the knife you pulled from a dead man's hand
    on day three, when the rain wouldn't stop.
    """
    item_name: str
    origin: str  # Where/how you got it
    origin_time: str  # Game time when acquired

    # Moments attached to this object
    moments: List[str] = field(default_factory=list)

    # Who else has touched it
    touched_by: List[str] = field(default_factory=list)

    # Is there blood on it? Literal or metaphorical
    stained_by: Optional[str] = None

    # Last used for what
    last_used: Optional[str] = None
    last_used_time: Optional[str] = None

    def add_moment(self, moment: str, game_time: str = ""):
        """Record a moment with this object."""
        timestamp = game_time or datetime.now().isoformat()
        self.moments.append(f"[{timestamp}] {moment}")
        # Keep last 10 moments
        if len(self.moments) > 10:
            self.moments = self.moments[-10:]

    def get_narrative(self) -> str:
        """Get a narrative description of this object's history."""
        parts = [f"{self.item_name}."]

        if self.origin:
            parts.append(f"You got it {self.origin}.")

        if self.stained_by:
            parts.append(f"There's still {self.stained_by} on it.")

        if self.last_used:
            parts.append(f"Last used to {self.last_used}.")

        if len(self.moments) > 0:
            parts.append(f"It's been through things.")

        return " ".join(parts)

    def to_dict(self) -> Dict:
        return {
            "item_name": self.item_name,
            "origin": self.origin,
            "origin_time": self.origin_time,
            "moments": self.moments,
            "touched_by": self.touched_by,
            "stained_by": self.stained_by,
            "last_used": self.last_used,
            "last_used_time": self.last_used_time,
        }


@dataclass
class UnfinishedThread:
    """
    Something left undone. A promise not kept.
    A conversation interrupted. A door left open.

    These create the weight of reality - the sense that
    things exist beyond what you're paying attention to.
    """
    id: str
    created: str

    thread_type: str  # "promise", "question", "task", "mystery", "relationship"
    description: str

    # Who's involved
    involves: List[str] = field(default_factory=list)

    # How urgent does it feel
    weight: float = 0.5  # 0-1

    # Has it been mentioned since?
    last_mentioned: Optional[str] = None
    times_mentioned: int = 1

    # Is this the kind of thing that resolves, or just... fades?
    can_resolve: bool = True

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "created": self.created,
            "thread_type": self.thread_type,
            "description": self.description,
            "involves": self.involves,
            "weight": self.weight,
            "last_mentioned": self.last_mentioned,
            "times_mentioned": self.times_mentioned,
            "can_resolve": self.can_resolve,
        }


@dataclass
class SilenceTracker:
    """
    Tracks the weight of silence.

    Silence in a zombie world isn't just quiet.
    It's the absence of everything that used to be there.
    Car alarms that stopped. Phones that don't ring.
    The hum of refrigerators in empty houses.
    """
    world_id: str

    # How long since anyone spoke (in-game time)
    silence_duration_hours: float = 0.0

    # What was the last sound?
    last_sound: str = ""
    last_sound_time: str = ""

    # What sounds are MISSING that should be there?
    missing_sounds: List[str] = field(default_factory=list)

    # Tension buildup (silence breeds tension)
    tension: float = 0.0  # 0-1

    def add_silence(self, hours: float):
        """Time passes in silence."""
        self.silence_duration_hours += hours
        # Tension builds with silence, but slowly
        self.tension = min(1.0, self.tension + (hours * 0.05))

    def break_silence(self, sound: str, game_time: str = ""):
        """Something breaks the silence."""
        # The longer the silence, the more jarring the break
        jarring_factor = min(1.0, self.silence_duration_hours / 10.0)

        self.last_sound = sound
        self.last_sound_time = game_time or datetime.now().isoformat()
        self.silence_duration_hours = 0.0
        self.tension = max(0.0, self.tension - 0.2)  # Relief, but not complete

        return jarring_factor

    def get_silence_description(self) -> str:
        """Describe the current quality of silence."""
        if self.silence_duration_hours < 0.5:
            return ""
        elif self.silence_duration_hours < 2:
            return "It's quiet."
        elif self.silence_duration_hours < 6:
            return "The silence has weight to it."
        elif self.silence_duration_hours < 12:
            return "The silence is the loudest thing in the room."
        else:
            return "You've forgotten what noise sounds like."

    def to_dict(self) -> Dict:
        return {
            "world_id": self.world_id,
            "silence_duration_hours": self.silence_duration_hours,
            "last_sound": self.last_sound,
            "last_sound_time": self.last_sound_time,
            "missing_sounds": self.missing_sounds,
            "tension": self.tension,
        }


class AmbientGenerator:
    """
    Generates ambient sensory details for a moment.

    This isn't about events. It's about texture.
    What does RIGHT NOW feel like in your body?
    """

    # Genre-specific sensory palettes
    SENSORY_PALETTES = {
        "zombie_apocalypse": {
            SenseType.SMELL: [
                ("The air smells wrong. Sweet and rotten.", 0.8, "decay"),
                ("Smoke, somewhere distant. Something's still burning.", 0.6, "fire"),
                ("Dust. Everything smells like dust now.", 0.5, "abandonment"),
                ("Your own sweat. When did you last shower?", 0.4, "self"),
                ("Gasoline. Someone was here recently.", 0.7, "recent activity"),
                ("Nothing. The absence of cooking, of life.", 0.5, "absence", True),
            ],
            SenseType.SOUND: [
                ("Wind through broken windows.", 0.4, "environment"),
                ("Your own breathing. Too loud.", 0.5, "self"),
                ("A door, somewhere, creaking in the wind.", 0.6, "building"),
                ("Silence where traffic used to be.", 0.7, "absence", True),
                ("Something dripping. You don't want to know.", 0.5, "unknown"),
                ("A car alarm that finally died.", 0.3, "absence", True),
                ("Birds. There are still birds.", 0.4, "nature"),
            ],
            SenseType.SIGHT: [
                ("Dust motes in a shaft of light.", 0.3, "environment"),
                ("A calendar still showing last month.", 0.5, "time"),
                ("Dark stains on the floor. Don't look.", 0.7, "violence"),
                ("A child's drawing on a refrigerator.", 0.6, "loss"),
                ("Your hands. Dirtier than you've ever seen them.", 0.4, "self"),
                ("Empty shelves where food used to be.", 0.5, "scarcity"),
            ],
            SenseType.TOUCH: [
                ("The grit of unwashed skin.", 0.4, "self"),
                ("Cold metal of a weapon, warming slowly.", 0.5, "weapon"),
                ("The ache in your feet. How far today?", 0.5, "exhaustion"),
                ("Hunger. A fist in your stomach.", 0.6, "hunger"),
                ("The weight of what you're carrying.", 0.4, "burden"),
            ],
            SenseType.FEELING: [
                ("The sense that something is watching.", 0.6, "paranoia"),
                ("Exhaustion that goes deeper than sleep can fix.", 0.5, "fatigue"),
                ("The tight wire of being always ready.", 0.6, "hypervigilance"),
                ("A moment of calm. It won't last.", 0.4, "respite"),
                ("The weight of decisions you've made.", 0.5, "guilt"),
            ],
        },
        "default": {
            SenseType.SMELL: [
                ("The air here has a particular quality.", 0.3, "environment"),
            ],
            SenseType.SOUND: [
                ("Background noise you can't quite identify.", 0.3, "environment"),
            ],
            SenseType.SIGHT: [
                ("Light and shadow.", 0.3, "environment"),
            ],
            SenseType.TOUCH: [
                ("The temperature on your skin.", 0.3, "environment"),
            ],
            SenseType.FEELING: [
                ("A vague sense of something.", 0.3, "intuition"),
            ],
        }
    }

    # Time of day modifiers
    TIME_DETAILS = {
        "dawn": [
            (SenseType.SIGHT, "Gray light seeping in. Another day you didn't die.", 0.5),
            (SenseType.FEELING, "The dread of daylight. Nowhere to hide.", 0.4),
            (SenseType.SOUND, "Birds. They don't know anything's wrong.", 0.4),
        ],
        "day": [
            (SenseType.SIGHT, "Hard sunlight. Everything visible. Everything exposed.", 0.5),
            (SenseType.TOUCH, "The heat building. Or maybe that's just you.", 0.4),
        ],
        "dusk": [
            (SenseType.SIGHT, "The light going gold, then gray. Time to find shelter.", 0.6),
            (SenseType.FEELING, "The day ending. Did you do enough?", 0.5),
        ],
        "night": [
            (SenseType.SIGHT, "Darkness. The world shrinks to what you can touch.", 0.7),
            (SenseType.SOUND, "Every sound louder now. Your heart. Loudest of all.", 0.6),
            (SenseType.FEELING, "The vulnerability of not seeing.", 0.7),
        ],
    }

    # Weather modifiers
    WEATHER_DETAILS = {
        "rain": [
            (SenseType.SOUND, "Rain on the roof. Covering other sounds.", 0.6),
            (SenseType.TOUCH, "Damp that gets into everything.", 0.5),
            (SenseType.SMELL, "Petrichor. The world washing itself.", 0.4),
        ],
        "storm": [
            (SenseType.SOUND, "Thunder. Nature doesn't care about your problems.", 0.7),
            (SenseType.SIGHT, "Lightning freeze-frames the world. What did you see?", 0.8),
            (SenseType.FEELING, "Something primal. Fear older than thought.", 0.6),
        ],
        "fog": [
            (SenseType.SIGHT, "The world ends ten feet away. What's past that?", 0.8),
            (SenseType.SOUND, "Sounds muffled. Direction impossible.", 0.7),
            (SenseType.FEELING, "Anything could be out there.", 0.7),
        ],
        "clear": [
            (SenseType.SIGHT, "Clear sky. You can see too far. They can too.", 0.5),
        ],
    }

    @classmethod
    def generate_moment(
        cls,
        genre: str = "default",
        time_of_day: str = "day",
        weather: str = "clear",
        recent_events: List[str] = None,
        character_state: Dict = None,
    ) -> List[SensoryDetail]:
        """
        Generate sensory details for this exact moment.

        Not what's happening. What it FEELS like.
        """
        details = []

        # Get genre palette
        palette = cls.SENSORY_PALETTES.get(genre, cls.SENSORY_PALETTES["default"])

        # Pick 2-4 sensory details from palette
        all_sensations = []
        for sense_type, sensations in palette.items():
            for sensation in sensations:
                if len(sensation) == 4:
                    desc, intensity, source, is_absence = sensation
                else:
                    desc, intensity, source = sensation
                    is_absence = False
                all_sensations.append(SensoryDetail(
                    sense=sense_type,
                    description=desc,
                    intensity=intensity,
                    source=source,
                    is_absence=is_absence,
                ))

        # Weight by intensity for selection
        weights = [s.intensity for s in all_sensations]
        selected = random.choices(all_sensations, weights=weights, k=random.randint(2, 4))
        details.extend(selected)

        # Add time of day detail
        time_details = cls.TIME_DETAILS.get(time_of_day, [])
        if time_details:
            sense, desc, intensity = random.choice(time_details)
            details.append(SensoryDetail(sense=sense, description=desc, intensity=intensity, source="time"))

        # Add weather detail
        weather_key = weather.replace("_", " ").split()[0]  # "heavy_rain" -> "rain"
        weather_details = cls.WEATHER_DETAILS.get(weather_key, cls.WEATHER_DETAILS.get(weather, []))
        if weather_details:
            sense, desc, intensity = random.choice(weather_details)
            details.append(SensoryDetail(sense=sense, description=desc, intensity=intensity, source="weather"))

        return details

    @classmethod
    def generate_texture_block(
        cls,
        genre: str = "default",
        time_of_day: str = "day",
        weather: str = "clear",
    ) -> str:
        """Generate a prose block of sensory texture for the DM."""
        details = cls.generate_moment(genre, time_of_day, weather)

        lines = ["=== THIS MOMENT ==="]
        for detail in details:
            prefix = "[absence]" if detail.is_absence else ""
            lines.append(f"{prefix} {detail.description}")

        return "\n".join(lines)


class BackgroundMurmur:
    """
    The fragments of life happening around you.

    Conversations overheard. Movement in peripheral vision.
    The sense that the world exists beyond your attention.
    """

    MURMUR_TEMPLATES = {
        "zombie_apocalypse": {
            "overheard": [
                "\"...said we should head north, but after what happened at...\"",
                "\"...three days now. How long can a person...\"",
                "\"...found her like that. Nothing we could...\"",
                "\"...keep your voice down. Sound carries.\"",
                "\"...remember when this was just a gas station?\"",
                "\"...not sleeping again? You need to...\"",
                "\"...trust them? After what...\"",
            ],
            "movement": [
                "Someone shifts position in the dark.",
                "A shadow crosses a window. Probably nothing.",
                "Footsteps upstairs. Pause. Continue.",
                "Someone's checking the barricade. Again.",
            ],
            "activity": [
                "The soft click of someone counting ammunition.",
                "A can being opened. The last of something.",
                "Someone drawing in a journal. Recording. Remembering.",
                "The scratch of a match. A brief flare of light.",
            ],
        },
    }

    @classmethod
    def generate(cls, genre: str = "zombie_apocalypse", count: int = 2) -> List[str]:
        """Generate background murmur fragments."""
        templates = cls.MURMUR_TEMPLATES.get(genre, {})

        fragments = []
        all_murmurs = []
        for category, murmurs in templates.items():
            all_murmurs.extend(murmurs)

        if all_murmurs:
            fragments = random.sample(all_murmurs, min(count, len(all_murmurs)))

        return fragments


class WorldTexture:
    """
    Manages the ambient texture of a world.

    This is what makes a world feel inhabitable.
    Not the events, but the dust between them.
    """

    def __init__(self, world_id: str, genre: str = "default"):
        self.world_id = world_id
        self.genre = genre

        # Object memories
        self.object_memories: Dict[str, ObjectMemory] = {}

        # Unfinished threads
        self.threads: List[UnfinishedThread] = []

        # Silence tracking
        self.silence = SilenceTracker(world_id=world_id)

        # Missing sounds (things that SHOULD be there but aren't)
        self.missing_sounds = [
            "traffic",
            "phones ringing",
            "music from somewhere",
            "children playing",
            "airplanes overhead",
            "the hum of electricity",
        ]
        self.silence.missing_sounds = self.missing_sounds

    def add_object_memory(self, item_name: str, origin: str, game_time: str = ""):
        """Create memory for an object."""
        memory = ObjectMemory(
            item_name=item_name,
            origin=origin,
            origin_time=game_time or datetime.now().isoformat(),
        )
        self.object_memories[item_name.lower()] = memory
        return memory

    def get_object_memory(self, item_name: str) -> Optional[ObjectMemory]:
        """Get memory for an object if it exists."""
        return self.object_memories.get(item_name.lower())

    def add_thread(
        self,
        thread_type: str,
        description: str,
        involves: List[str] = None,
        weight: float = 0.5,
    ) -> UnfinishedThread:
        """Add an unfinished thread."""
        thread = UnfinishedThread(
            id=f"thread_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(100,999)}",
            created=datetime.now().isoformat(),
            thread_type=thread_type,
            description=description,
            involves=involves or [],
            weight=weight,
        )
        self.threads.append(thread)

        # Keep only the 20 most recent/weighted threads
        if len(self.threads) > 20:
            self.threads.sort(key=lambda t: t.weight, reverse=True)
            self.threads = self.threads[:20]

        return thread

    def get_ambient_context(
        self,
        time_of_day: str = "day",
        weather: str = "clear",
    ) -> str:
        """
        Generate the full ambient context for this moment.

        This goes into the DM context to help them paint the world.
        """
        lines = []

        # Sensory texture
        texture = AmbientGenerator.generate_texture_block(
            genre=self.genre,
            time_of_day=time_of_day,
            weather=weather,
        )
        lines.append(texture)

        # Silence weight
        silence_desc = self.silence.get_silence_description()
        if silence_desc:
            lines.append(f"\n{silence_desc}")

        # Background murmur (if people are around)
        murmurs = BackgroundMurmur.generate(self.genre, count=2)
        if murmurs:
            lines.append("\n=== BACKGROUND ===")
            for murmur in murmurs:
                lines.append(murmur)

        # Unfinished threads (pick 1-2 to surface)
        if self.threads:
            heavy_threads = [t for t in self.threads if t.weight > 0.5]
            if heavy_threads:
                lines.append("\n=== UNFINISHED ===")
                for thread in random.sample(heavy_threads, min(2, len(heavy_threads))):
                    lines.append(f"��� {thread.description}")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "world_id": self.world_id,
            "genre": self.genre,
            "object_memories": {k: v.to_dict() for k, v in self.object_memories.items()},
            "threads": [t.to_dict() for t in self.threads],
            "silence": self.silence.to_dict(),
        }


# Global texture storage
_world_textures: Dict[str, WorldTexture] = {}


def get_world_texture(world_id: str, genre: str = "default") -> WorldTexture:
    """Get or create world texture for a world."""
    if world_id not in _world_textures:
        _world_textures[world_id] = WorldTexture(world_id, genre)
    return _world_textures[world_id]
