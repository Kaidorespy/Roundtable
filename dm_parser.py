"""
DM Parser - Message Analysis Layer

Runs on every message. Mostly passive - accumulates information.
Only rarely triggers immediate action (e.g., opening a freezer full of zombies).

See dm_plans.md for full architecture.
"""

import re
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum


class MessageType(Enum):
    """What kind of message is this?"""
    DIALOGUE = "dialogue"           # Character speaking
    ACTION = "action"               # Character doing something
    DISCOVERY = "discovery"         # Looking, searching, inspecting
    MOVEMENT = "movement"           # Going somewhere
    ESTABLISHMENT = "establishment" # Player declaring something exists
    QUESTION = "question"           # Asking about the world
    INTERNAL = "internal"           # Internal thoughts/feelings
    MIXED = "mixed"                 # Multiple types


class NoiseLevel(Enum):
    """How loud is this action?"""
    SILENT = "silent"       # Whispering, sneaking, thinking
    QUIET = "quiet"         # Normal conversation, careful movement
    NORMAL = "normal"       # Walking, talking, basic actions
    LOUD = "loud"           # Shouting, running, breaking things
    EXPLOSIVE = "explosive" # Gunshots, explosions, screams


class MovementType(Enum):
    """Is anyone moving?"""
    NONE = "none"           # Stationary
    LOCAL = "local"         # Moving within current area
    TRAVEL = "travel"       # Leaving current area


@dataclass
class ParsedMessage:
    """Result of parsing a message."""
    raw_content: str
    message_type: MessageType
    noise_level: NoiseLevel
    movement: MovementType

    # Specific detections
    discovery_target: Optional[str] = None      # "the mud", "the tracks", "the freezer"
    movement_destination: Optional[str] = None  # "north", "to the gas station"
    establishment_content: Optional[str] = None # "there's a shed out back"

    # Trigger flags (rare - usually empty)
    triggers: List[str] = field(default_factory=list)  # ["freezer_opened", "gunshot"]

    # Keywords extracted
    keywords: List[str] = field(default_factory=list)  # Notable nouns/actions

    # For Stagehand/Tension Keeper to process later
    needs_stagehand_check: bool = False
    needs_tension_check: bool = False


# === DETECTION PATTERNS ===

DISCOVERY_PATTERNS = [
    r'\b(look|looking)\s+(at|around|for|through|inside|into)\b',
    r'\b(search|searching|searched)\b',
    r'\b(inspect|inspecting|examine|examining)\b',
    r'\b(check|checking)\s+(out|on|the|for)?\b',
    r'\bwhat\s+(is|are|do i see)\b',
    r'\b(find|finding|found)\b',
    r'\b(notice|noticing|noticed)\b',
    r'\b(scan|scanning)\b',
]

MOVEMENT_PATTERNS = [
    r'\b(walk|walking|walked)\s+(to|toward|towards|into|over|north|south|east|west)\b',
    r'\b(go|going|went)\s+(to|toward|towards|into|north|south|east|west)\b',
    r'\b(head|heading|headed)\s+(to|toward|towards|into|north|south|east|west)\b',
    r'\b(move|moving|moved)\s+(to|toward|towards|into)\b',
    r'\b(travel|traveling|travelled)\b',
    r'\b(leave|leaving|left)\s+(the|this|for)\b',
    r'\b(follow|following)\s+(the|a|it)\b',
]

TRAVEL_PATTERNS = [
    r'\b(to the|toward the|towards the)\s+\w+\s*(station|building|house|cabin|store|market|tower)\b',
    r'\b(leave|leaving|left)\s+(the area|this place|here)\b',
    r'\b(head|heading)\s+(out|away)\b',
    r'\btravel\b',
]

ACTION_PATTERNS = [
    r'\bi\s+(open|close|grab|take|drop|throw|push|pull|break|smash|hit|kick)\b',
    r'\b(pick up|put down|set down)\b',
    r'\bi\s+(try|attempt|start|begin)\b',
    r'\b(shoot|fire|swing|stab|attack)\b',
]

LOUD_NOISE_PATTERNS = [
    r'\b(shout|shouting|shouted|yell|yelling|yelled|scream|screaming|screamed)\b',
    r'\b(gunshot|shot|shoot|shooting|fire|fired|firing)\b',
    r'\b(explosion|explode|exploding|bang|crash|smash|smashing)\b',
    r'\b(break|breaking|broke|shatter|shattering|shattered)\s+(the|a|through)\b',
    r'\b(running|run|ran)\s+(away|toward|through)\b',
]

QUIET_PATTERNS = [
    r'\b(whisper|whispering|whispered)\b',
    r'\b(sneak|sneaking|sneaked|crept|creep|creeping)\b',
    r'\b(quietly|silently|carefully|softly)\b',
    r'\b(tiptoe|tiptoeing)\b',
]

ESTABLISHMENT_PATTERNS = [
    r'\bthere\s*(is|was|should be|must be|might be)\s+(a|an|the)\s+',
    r'\bi\s+(remember|recall|notice)\s+(a|an|the)\s+\w+\s+(nearby|here|outside|inside)',
    r'\bmaybe\s+there\'s\s+(a|an)\b',
]


def parse_message(content: str, room_context: Optional[Dict] = None) -> ParsedMessage:
    """
    Parse a message and return annotations.

    This is the main entry point. Called on every message.

    Args:
        content: The raw message text
        room_context: Optional dict with room state (location, tension state, etc.)

    Returns:
        ParsedMessage with all annotations
    """
    content_lower = content.lower()

    # Detect message type
    message_type = _detect_message_type(content_lower)

    # Detect noise level
    noise_level = _detect_noise_level(content_lower)

    # Detect movement
    movement, destination = _detect_movement(content_lower)

    # Extract discovery target if applicable
    discovery_target = None
    if message_type in [MessageType.DISCOVERY, MessageType.MIXED]:
        discovery_target = _extract_discovery_target(content_lower)

    # Check for establishments
    establishment = _detect_establishment(content)

    # Extract keywords
    keywords = _extract_keywords(content)

    # Build result
    result = ParsedMessage(
        raw_content=content,
        message_type=message_type,
        noise_level=noise_level,
        movement=movement,
        discovery_target=discovery_target,
        movement_destination=destination,
        establishment_content=establishment,
        keywords=keywords,
    )

    # Flag if needs follow-up processing
    if message_type == MessageType.DISCOVERY:
        result.needs_stagehand_check = True

    if noise_level in [NoiseLevel.LOUD, NoiseLevel.EXPLOSIVE]:
        result.needs_tension_check = True

    if movement == MovementType.TRAVEL:
        result.needs_stagehand_check = True
        result.needs_tension_check = True

    # Check for immediate triggers (TODO: integrate with Tension Keeper)
    # For now, just flag obvious dangerous actions
    result.triggers = _detect_triggers(content_lower, room_context)

    return result


def _detect_message_type(content: str) -> MessageType:
    """Classify the message type."""
    types_found = []

    # Check for discovery
    for pattern in DISCOVERY_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            types_found.append(MessageType.DISCOVERY)
            break

    # Check for movement
    for pattern in MOVEMENT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            types_found.append(MessageType.MOVEMENT)
            break

    # Check for action
    for pattern in ACTION_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            types_found.append(MessageType.ACTION)
            break

    # Check for establishment
    for pattern in ESTABLISHMENT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            types_found.append(MessageType.ESTABLISHMENT)
            break

    # Check for question (to the world/DM, not to another character)
    if re.search(r'\?$', content.strip()) and re.search(r'\b(what|where|when|how|is there|are there|can i|do i)\b', content, re.IGNORECASE):
        types_found.append(MessageType.QUESTION)

    # Determine final type
    if len(types_found) == 0:
        # Default to dialogue or internal based on content
        if re.search(r'^["\'"]|["\'"]$|^i\s+(say|tell|ask)\b', content, re.IGNORECASE):
            return MessageType.DIALOGUE
        elif re.search(r'^i\s+(feel|think|wonder|realize|notice)\b', content, re.IGNORECASE):
            return MessageType.INTERNAL
        else:
            return MessageType.DIALOGUE  # Default
    elif len(types_found) == 1:
        return types_found[0]
    else:
        return MessageType.MIXED


def _detect_noise_level(content: str) -> NoiseLevel:
    """Detect how loud this action is."""
    # Check for quiet first (explicit quietness overrides)
    for pattern in QUIET_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return NoiseLevel.QUIET

    # Check for loud
    for pattern in LOUD_NOISE_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            # Distinguish between loud and explosive
            if re.search(r'\b(gunshot|shot|shoot|explosion|explode|bang)\b', content, re.IGNORECASE):
                return NoiseLevel.EXPLOSIVE
            return NoiseLevel.LOUD

    # Default based on action type
    if re.search(r'\b(whisper|silent|quiet|careful)\b', content, re.IGNORECASE):
        return NoiseLevel.SILENT

    return NoiseLevel.NORMAL


def _detect_movement(content: str) -> tuple[MovementType, Optional[str]]:
    """Detect if there's movement and where to."""
    # Check for travel (leaving area)
    for pattern in TRAVEL_PATTERNS:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            # Try to extract destination
            dest_match = re.search(r'to(?:ward|wards)?\s+(?:the\s+)?(\w+(?:\s+\w+)?)', content, re.IGNORECASE)
            destination = dest_match.group(1) if dest_match else None
            return MovementType.TRAVEL, destination

    # Check for local movement
    for pattern in MOVEMENT_PATTERNS:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            # Extract destination
            dest_match = re.search(r'(?:to|toward|towards|into)\s+(?:the\s+)?(\w+(?:\s+\w+)?)', content, re.IGNORECASE)
            destination = dest_match.group(1) if dest_match else None
            return MovementType.LOCAL, destination

    return MovementType.NONE, None


def _extract_discovery_target(content: str) -> Optional[str]:
    """Extract what the player is trying to discover/examine."""
    # Pattern: look at/for/around THE X, search THE X, inspect THE X
    patterns = [
        r'(?:look|search|inspect|examine|check)\s+(?:at|for|around|through|inside|into)?\s*(?:the\s+)?(\w+(?:\s+\w+)?)',
        r'(?:what|where)\s+(?:is|are)\s+(?:the\s+)?(\w+)',
        r'(?:find|notice|see)\s+(?:a|an|the|any)?\s*(\w+(?:\s+\w+)?)',
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def _detect_establishment(content: str) -> Optional[str]:
    """Detect if player is establishing something exists in the world."""
    for pattern in ESTABLISHMENT_PATTERNS:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            # Get the rest of the sentence after the match
            start = match.end()
            rest = content[start:start+50]  # Get next 50 chars
            # Clean up
            rest = re.sub(r'[.!?,;].*', '', rest).strip()
            if rest:
                return match.group(0) + rest
    return None


def _extract_keywords(content: str) -> List[str]:
    """Extract notable nouns and actions for indexing."""
    # Simple extraction - notable nouns
    # Skip common words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'i', 'you', 'we', 'they',
                  'it', 'my', 'your', 'our', 'this', 'that', 'and', 'or', 'but', 'in',
                  'on', 'at', 'to', 'for', 'of', 'with', 'by'}

    # Find all words
    words = re.findall(r'\b[a-z]+\b', content.lower())

    # Filter
    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    # Dedupe while preserving order
    seen = set()
    unique = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            unique.append(w)

    return unique[:10]  # Limit to 10


def _detect_triggers(content: str, room_context: Optional[Dict]) -> List[str]:
    """
    Detect immediate triggers that need response NOW.

    This is where integration with Tension Keeper will happen.
    For now, just detect obvious dangerous patterns.
    """
    triggers = []

    # TODO: Check room_context for known danger zones and their triggers
    # For now, just pattern match obvious things

    # Gunshot
    if re.search(r'\b(shoot|fire|shot|pull(?:ed)?\s+(?:the\s+)?trigger)\b', content, re.IGNORECASE):
        triggers.append('gunshot')

    # Opening containers (potential zombie traps)
    if re.search(r'\b(open|opened|opening)\s+(?:the\s+)?(freezer|fridge|door|trunk|container|closet|locker)\b', content, re.IGNORECASE):
        triggers.append('container_opened')

    # Breaking things (noise)
    if re.search(r'\b(break|smash|shatter|kick\s+(?:down|in|open))\b', content, re.IGNORECASE):
        triggers.append('loud_destruction')

    return triggers


# === UTILITY ===

def format_parsed_message(parsed: ParsedMessage) -> str:
    """Format a ParsedMessage for logging/debugging."""
    lines = [
        f"Type: {parsed.message_type.value}",
        f"Noise: {parsed.noise_level.value}",
        f"Movement: {parsed.movement.value}",
    ]

    if parsed.discovery_target:
        lines.append(f"Discovery target: {parsed.discovery_target}")
    if parsed.movement_destination:
        lines.append(f"Destination: {parsed.movement_destination}")
    if parsed.establishment_content:
        lines.append(f"Establishes: {parsed.establishment_content}")
    if parsed.triggers:
        lines.append(f"TRIGGERS: {parsed.triggers}")
    if parsed.keywords:
        lines.append(f"Keywords: {', '.join(parsed.keywords[:5])}")

    return " | ".join(lines)
