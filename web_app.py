"""
Roundtable Web - Multi-AI conversation orchestrator (Flask version)
"""

import os
import uuid
import json
import asyncio
import logging
import threading
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from config import DataStore, Message, Partner, Room, Settings, settings
from providers import ProviderManager
from dm_agents import get_dm_state, clear_dm_state
from fatigue import get_fatigue_tracker, init_fatigue_tracker, FatigueLevel
from inventory import get_inventory_tracker, init_inventory_tracker, ItemCategory, apply_narrative_changes
from consequence_engine import get_consequence_engine, init_consequence_engine, ConsequenceType
from loot_tables import get_loot_generator, LootGenerator, LootContext, suggest_loot_for_scene
from cartographer import get_cartographer, init_cartographer, LocationType, DiscoveryStatus
from npc_system import get_npc_registry, NPCRegistry, NPC, NPCState, GrudgeSeverity
from action_resolver import ActionType, ResolvedAction, resolve_action, classify_action, build_dm_instruction_for_resolution
from combat_system import (
    Dice, CombatStats, Attack, AttackResult, CombatResolver,
    CombatEncounter, EncounterManager, Combatant,
    create_companion_stats, create_npc_combatant, create_spidercock_stats,
    process_companion_death_in_combat
)
from autopilot import (
    AutopilotTracker, get_autopilot_tracker, init_autopilot_tracker,
    PlayerCharacter, JournalEntry, Alignment, Priority, Drive, Role
)
from relationship_map import (
    RelationshipMapBuilder, get_relationship_map,
    RelationshipNode, RelationshipEdge, RelationshipMap, RelationshipType
)
from story_daemon import (
    StoryDaemon, get_story_daemon, init_story_daemon, stop_story_daemon,
    WorldState, StoryEvent, EventSeverity
)
from dm_narrator import get_dm_narrator, init_dm_narrator, get_separated_tick_tracker, init_separated_tick_tracker
from dm_parser import parse_message, format_parsed_message, ParsedMessage, MessageType, NoiseLevel
from condition_tracker import (
    ConditionTracker, get_condition_tracker, init_condition_tracker,
    CharacterCondition, Injury, OverallCondition, HungerStatus, ThirstStatus, InjurySeverity
)
from understudy import (
    UnderstudyManager, get_understudy_manager, init_understudy_manager,
    UnderstudyDecision, UnderstudyRule, UnderstudyMemory, Echo, Dream,
    DecisionCategory, Confidence, FeedbackType
)
from weather_sync import (
    WeatherSync, WeatherData, get_weather_sync, init_weather_sync
)

# ============================================================================
# Local Whisper for Speech-to-Text (lazy loaded)
# ============================================================================
_local_whisper = None

def get_local_whisper():
    """Lazy-load local faster-whisper model."""
    global _local_whisper
    if _local_whisper is None:
        try:
            from faster_whisper import WhisperModel
            _local_whisper = WhisperModel("small", device="cpu", compute_type="int8")
            print("[STT] Loaded local Whisper model (small)")
        except Exception as e:
            print(f"[STT] Local Whisper not available: {e}")
            return None
    return _local_whisper

def _add_pause_punctuation(segments):
    """Add punctuation based on pauses between words."""
    words = []
    for segment in segments:
        if hasattr(segment, 'words') and segment.words:
            words.extend(segment.words)

    if not words:
        # No word-level timestamps, just join segments
        return " ".join(s.text for s in segments).strip()

    result = []
    for i, word in enumerate(words):
        text = word.word.strip()
        if not text:
            continue

        result.append(text)

        # Check pause before next word
        if i < len(words) - 1:
            pause = words[i + 1].start - word.end
            # Don't add punctuation if word already ends with punctuation
            if text and text[-1] not in '.,!?;:—-':
                if pause > 1.5:
                    result.append('...')
                elif pause > 0.7:
                    result.append(',')

    return " ".join(result).replace(" ,", ",").replace(" ...", "...").strip()

app = Flask(__name__)
CORS(app)

# Suppress noisy job polling logs
class JobPollFilter(logging.Filter):
    def filter(self, record):
        # Suppress GET /jobs/xxx requests from werkzeug logs
        msg = record.getMessage()
        if 'GET /jobs/' in msg and 'HTTP' in msg:
            return False
        return True

logging.getLogger('werkzeug').addFilter(JobPollFilter())
app.secret_key = os.getenv("SECRET_KEY", "roundtable-secret")

# ============================================================================
# Background Job System - prevents UI blocking during image generation
# ============================================================================
_jobs = {}  # job_id -> {status, result, error, type}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="roundtable_job")
_jobs_file = Path.home() / ".roundtable" / "jobs.json"


def _save_jobs():
    """Persist jobs to disk."""
    try:
        _jobs_file.parent.mkdir(parents=True, exist_ok=True)
        with open(_jobs_file, 'w') as f:
            json.dump(_jobs, f)
    except Exception as e:
        print(f"[JOBS] Failed to save jobs: {e}")


def _load_jobs():
    """Load jobs from disk. Mark interrupted jobs as failed."""
    global _jobs
    try:
        if _jobs_file.exists():
            with open(_jobs_file, 'r') as f:
                _jobs = json.load(f)
            # Mark any in-progress jobs as failed (they were interrupted by restart)
            interrupted = 0
            for job_id, job in _jobs.items():
                if job['status'] in ('pending', 'generating_description', 'generating_image'):
                    job['status'] = 'failed'
                    job['error'] = 'Server restarted - job interrupted'
                    interrupted += 1
            if interrupted:
                print(f"[JOBS] Marked {interrupted} interrupted jobs as failed")
                _save_jobs()
            print(f"[JOBS] Loaded {len(_jobs)} jobs from disk")
    except Exception as e:
        print(f"[JOBS] Failed to load jobs: {e}")
        _jobs = {}


# Load jobs on module import
_load_jobs()


def create_job(job_type: str) -> str:
    """Create a new background job and return its ID."""
    job_id = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[job_id] = {
            'status': 'pending',
            'type': job_type,
            'result': None,
            'error': None,
            'created': datetime.now().isoformat()
        }
        pending_count = sum(1 for j in _jobs.values() if j['status'] in ('pending', 'generating_description', 'generating_image'))
        _save_jobs()
    print(f"\033[38;5;51m[JOB] Created {job_id} ({job_type}) - {pending_count} jobs in queue\033[0m")
    return job_id


def update_job(job_id: str, status: str, result=None, error=None):
    """Update a job's status and result."""
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]['status'] = status
            if result is not None:
                _jobs[job_id]['result'] = result
            if error is not None:
                _jobs[job_id]['error'] = error
            _save_jobs()
            # Log status changes
            if status in ('completed', 'failed'):
                pending_count = sum(1 for j in _jobs.values() if j['status'] in ('pending', 'generating_description', 'generating_image'))
                print(f"\033[38;5;51m[JOB] {job_id} -> {status} - {pending_count} jobs remaining\033[0m")


def get_job(job_id: str) -> dict:
    """Get a job's current state."""
    with _jobs_lock:
        return _jobs.get(job_id, {}).copy()


def is_job_cancelled(job_id: str) -> bool:
    """Check if a job has been cancelled."""
    with _jobs_lock:
        job = _jobs.get(job_id, {})
        return job.get('status') == 'cancelled'


def get_all_jobs() -> dict:
    """Get all jobs for debugging."""
    with _jobs_lock:
        return {
            'total': len(_jobs),
            'pending': sum(1 for j in _jobs.values() if j['status'] == 'pending'),
            'running': sum(1 for j in _jobs.values() if j['status'] in ('generating_description', 'generating_image')),
            'completed': sum(1 for j in _jobs.values() if j['status'] == 'completed'),
            'failed': sum(1 for j in _jobs.values() if j['status'] == 'failed'),
            'jobs': {k: {'status': v['status'], 'type': v['type']} for k, v in _jobs.items()}
        }


# ============================================================================
# Shared Images Tracking
# ============================================================================
_shared_images_lock = threading.Lock()


def _get_shared_images_file():
    """Get path to shared images tracking file."""
    return settings.data_dir / "shared_images.json"


def _load_shared_images() -> dict:
    """Load shared images data. Returns {image_path: [partner_ids]}."""
    shared_file = _get_shared_images_file()
    if shared_file.exists():
        try:
            return json.loads(shared_file.read_text())
        except Exception:
            pass
    return {}


def _save_shared_images(data: dict):
    """Save shared images data."""
    shared_file = _get_shared_images_file()
    shared_file.write_text(json.dumps(data, indent=2))


def _mark_image_shared(image_path: str, partner_id: str):
    """Mark an image as shared with a partner."""
    with _shared_images_lock:
        data = _load_shared_images()
        # Normalize path
        path_key = str(image_path).replace('\\', '/')
        if path_key not in data:
            data[path_key] = []
        if partner_id not in data[path_key]:
            data[path_key].append(partner_id)
        _save_shared_images(data)


def _is_image_shared(image_path: str, partner_id: str = None) -> bool:
    """Check if an image has been shared. If partner_id is None, checks if shared with anyone."""
    with _shared_images_lock:
        data = _load_shared_images()
        path_key = str(image_path).replace('\\', '/')
        if path_key not in data:
            return False
        if partner_id is None:
            return len(data[path_key]) > 0
        return partner_id in data[path_key]


def _get_shared_partners(image_path: str) -> list:
    """Get list of partner IDs the image was shared with."""
    with _shared_images_lock:
        data = _load_shared_images()
        path_key = str(image_path).replace('\\', '/')
        return data.get(path_key, [])


# ============================================================================
# Favorite Images Tracking (per-room)
# ============================================================================
_favorites_lock = threading.Lock()


def _get_favorites_file():
    """Get path to favorites tracking file."""
    return settings.data_dir / "favorite_images.json"


def _load_all_favorites() -> dict:
    """Load all favorites (dict of room_id -> list of paths)."""
    fav_file = _get_favorites_file()
    if fav_file.exists():
        try:
            data = json.loads(fav_file.read_text())
            # Handle migration from old format (list) to new format (dict)
            if isinstance(data, list):
                # Old format was a flat list - migrate to 'common' room
                return {'common': data}
            return data
        except Exception:
            pass
    return {}


def _save_all_favorites(all_favorites: dict):
    """Save all favorites."""
    fav_file = _get_favorites_file()
    fav_file.write_text(json.dumps(all_favorites, indent=2))


def _load_favorites(room_id: str) -> set:
    """Load favorite image paths for a specific room."""
    all_favs = _load_all_favorites()
    return set(all_favs.get(room_id, []))


def _toggle_favorite(image_path: str, room_id: str) -> bool:
    """Toggle favorite status for a room. Returns new status (True = favorited)."""
    with _favorites_lock:
        all_favs = _load_all_favorites()
        room_favs = set(all_favs.get(room_id, []))
        path_key = str(image_path).replace('\\', '/')

        if path_key in room_favs:
            room_favs.remove(path_key)
            all_favs[room_id] = list(room_favs)
            _save_all_favorites(all_favs)
            return False
        else:
            room_favs.add(path_key)
            all_favs[room_id] = list(room_favs)
            _save_all_favorites(all_favs)
            return True


def _is_favorite(image_path: str, room_id: str) -> bool:
    """Check if an image is favorited in a specific room."""
    with _favorites_lock:
        room_favs = _load_favorites(room_id)
        path_key = str(image_path).replace('\\', '/')
        return path_key in room_favs


# ============================================================================
# Prompt Book - a library of system prompts, auto-captured at character creation
# ============================================================================
_prompt_book_lock = threading.Lock()


def _get_prompt_book_file():
    """Get path to the prompt book file."""
    return settings.data_dir / "prompt_book.json"


def _load_prompt_book() -> list:
    """Load the prompt book (list of {id, name, prompt, created_at})."""
    book_file = _get_prompt_book_file()
    if book_file.exists():
        try:
            data = json.loads(book_file.read_text(encoding='utf-8'))
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def _save_prompt_book(entries: list):
    """Save the prompt book."""
    book_file = _get_prompt_book_file()
    book_file.write_text(json.dumps(entries, indent=2), encoding='utf-8')


def _add_to_prompt_book(name: str, prompt: str) -> bool:
    """Add a prompt to the book if its text isn't already present.

    Dedupe is by normalized prompt text (whitespace-trimmed). Returns True if a
    new entry was added, False if it was a duplicate or empty.
    """
    text = (prompt or '').strip()
    if not text:
        return False
    with _prompt_book_lock:
        entries = _load_prompt_book()
        for entry in entries:
            if (entry.get('prompt') or '').strip() == text:
                return False
        entries.append({
            'id': str(uuid.uuid4())[:8],
            'name': (name or 'Untitled').strip(),
            'prompt': text,
            'created_at': datetime.now().isoformat(),
        })
        _save_prompt_book(entries)
        return True


# ============================================================================
# Pending Room Images (for "share once, everyone sees on next turn")
# ============================================================================
_pending_images_lock = threading.Lock()


def _get_pending_images_file():
    """Get path to pending room images file."""
    return settings.data_dir / "pending_room_images.json"


def _load_pending_images() -> dict:
    """Load pending images. Returns {room_id: {image_path, seen_by: [partner_ids]}}."""
    pending_file = _get_pending_images_file()
    if pending_file.exists():
        try:
            return json.loads(pending_file.read_text())
        except Exception:
            pass
    return {}


def _save_pending_images(data: dict):
    """Save pending images data."""
    pending_file = _get_pending_images_file()
    pending_file.write_text(json.dumps(data, indent=2))


def _add_pending_image(room_id: str, image_path: str):
    """Add a pending image for a room (all partners will see it on next turn)."""
    with _pending_images_lock:
        data = _load_pending_images()
        path_key = str(image_path).replace('\\', '/')
        data[room_id] = {
            'image_path': path_key,
            'seen_by': []
        }
        _save_pending_images(data)


def _get_pending_image_for_partner(room_id: str, partner_id: str) -> str | None:
    """Get pending image if this partner hasn't seen it yet. Returns image_path or None."""
    with _pending_images_lock:
        data = _load_pending_images()
        if room_id not in data:
            return None
        pending = data[room_id]
        if partner_id in pending.get('seen_by', []):
            return None  # Already seen
        return pending.get('image_path')


def _mark_pending_image_seen(room_id: str, partner_id: str):
    """Mark that a partner has seen the pending image for this room."""
    with _pending_images_lock:
        data = _load_pending_images()
        if room_id not in data:
            return
        if partner_id not in data[room_id]['seen_by']:
            data[room_id]['seen_by'].append(partner_id)
        _save_pending_images(data)


def _clear_pending_image(room_id: str):
    """Clear the pending image for a room (e.g., when all have seen it)."""
    with _pending_images_lock:
        data = _load_pending_images()
        if room_id in data:
            del data[room_id]
            _save_pending_images(data)


# Load persisted settings if they exist
def load_persisted_settings():
    settings_file = settings.data_dir / "settings.json"
    if settings_file.exists():
        try:
            data = json.loads(settings_file.read_text())
            if 'user_name' in data:
                settings.user_name = data['user_name']
            if 'user_physical_description' in data:
                settings.user_physical_description = data['user_physical_description']
            if 'user_avatar' in data:
                settings.user_avatar = data['user_avatar']
            if 'auto_narration_images' in data:
                settings.auto_narration_images = data['auto_narration_images']
            if 'global_system_prompt' in data:
                settings.global_system_prompt = data['global_system_prompt']
            if 'saved_system_prompts' in data:
                settings.saved_system_prompts = data['saved_system_prompts']
            if 'storybuilder_model' in data:
                settings.storybuilder_model = data['storybuilder_model']
            if 'voice_enabled' in data:
                settings.voice_enabled = data['voice_enabled']
            if 'anthropic_api_key' in data:
                settings.anthropic_api_key = data['anthropic_api_key']
            if 'openai_api_key' in data:
                settings.openai_api_key = data['openai_api_key']
            if 'elevenlabs_api_key' in data:
                settings.elevenlabs_api_key = data['elevenlabs_api_key']
            if 'ollama_base_url' in data:
                settings.ollama_base_url = data['ollama_base_url']
            if 'default_ollama_model' in data:
                settings.default_ollama_model = data['default_ollama_model']
            if 'comfy_url' in data:
                settings.comfy_url = data['comfy_url']
            if 'custom_checkpoint' in data:
                settings.custom_checkpoint = data['custom_checkpoint']
            if 'custom_checkpoint_type' in data:
                settings.custom_checkpoint_type = data['custom_checkpoint_type']
        except Exception:
            pass

load_persisted_settings()

# Initialize data store and providers
data_store = DataStore(settings)
provider_manager = ProviderManager(settings)

# Initialize memory system
from memory import MemoryStore, MemoryConsolidator
memory_store = MemoryStore(settings.data_dir)
memory_consolidator = MemoryConsolidator(memory_store, settings.ollama_base_url, settings.default_ollama_model)

# Initialize fatigue system
fatigue_tracker = init_fatigue_tracker(settings.data_dir)

# Initialize inventory system
inventory_tracker = init_inventory_tracker(settings.data_dir)

# Initialize condition tracker (injuries, hunger, thirst)
condition_tracker = init_condition_tracker(settings.data_dir)

# Initialize consequence engine
consequence_engine = init_consequence_engine(settings.data_dir)

# Initialize cartographer (world mapping)
cartographer = init_cartographer(settings.data_dir)

# Initialize NPC registry with persistence
npc_registry = get_npc_registry()
_npc_file = settings.data_dir / "npcs.json"
if _npc_file.exists():
    npc_registry.load(str(_npc_file))

# Initialize autopilot system
autopilot_tracker = init_autopilot_tracker(settings.data_dir)

# Initialize understudy system (learns from player feedback)
understudy_manager = init_understudy_manager(settings.data_dir)

# Initialize weather sync (real weather for real worlds)
weather_sync = init_weather_sync(settings.data_dir)

# Story daemon initialized lazily when needed (requires provider callback)
story_daemon = None

# Track active consolidations for status indicator
_consolidation_status = {}  # room_id -> {partner_id: "running"|"done"}
_consolidation_lock = threading.Lock()


def run_async(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def clean_model_tokens(text: str) -> str:
    """Strip model-specific tokens that leak through (Qwen, etc)."""
    import re
    # Qwen special tokens
    text = re.sub(r'<\|im_start\|>', '', text)
    text = re.sub(r'<\|im_end\|>', '', text)
    # Llama/other special tokens
    text = re.sub(r'<\|eot_id\|>', '', text)
    text = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', text)
    # Clean up any resulting extra whitespace at start
    return text.strip()


def normalize_temperature(value, default: float = 0.7) -> float:
    """Clamp character temperature to the app-supported 0.0-1.0 range."""
    try:
        temperature = float(value)
    except (TypeError, ValueError):
        temperature = default
    return max(0.0, min(1.0, round(temperature, 1)))


def on_message_sent(room_id: str, speaker_id: str, speaker_name: str, content: str, is_user: bool = False):
    """
    Hook called whenever a message is sent. Handles:
    - Story daemon player connection
    - Echo learning from messages
    - Activity tracking for autopilot
    """
    global story_daemon

    # Connect player to story daemon (so the world runs while they're here)
    if story_daemon and is_user:
        story_daemon.player_connected(room_id, speaker_id)

    # Record activity for autopilot (resets auto-engage timer)
    if is_user:
        try:
            autopilot_tracker.record_activity(speaker_id, room_id)
        except:
            pass

    # Learn echoes from the message (builds understudy personality)
    if not is_user:
        # Learn from AI characters (including AI players)
        try:
            understudy_manager.extract_echoes_from_message(
                character_id=speaker_id,
                message=content,
                context=f"Room {room_id}"
            )
        except Exception as e:
            pass  # Don't break message flow for echo learning

    # Also learn from user messages (for the user's understudy)
    if is_user:
        try:
            # User's character ID is typically "player_{room_id}" or similar
            user_char_id = f"player_{room_id}"
            understudy_manager.extract_echoes_from_message(
                character_id=user_char_id,
                message=content,
                context=f"Room {room_id}"
            )
        except Exception as e:
            pass

    # Parse inventory changes from AI character messages (StoryBuilder rooms)
    if not is_user:
        try:
            room = data_store.get_room(room_id)
            # Only parse inventory in StoryBuilder rooms (have character_relationships or player_character_name)
            if room and (room.player_character_name or (hasattr(room, 'character_relationships') and room.character_relationships)):
                player_id = f"player_{room_id}"
                player_name = room.player_character_name or settings.user_name

                # Parse narrative for player inventory changes
                changes = apply_narrative_changes(
                    tracker=inventory_tracker,
                    owner_id=player_id,
                    owner_name=player_name,
                    text=content,
                    source=f"narrative from {speaker_name}"
                )

                # Log notable changes
                if changes['acquired']:
                    print(f"[Inventory] {player_name} acquired: {', '.join(changes['acquired'])}")
                if changes['lost']:
                    print(f"[Inventory] {player_name} lost: {', '.join(changes['lost'])}")
                if changes['consumed']:
                    print(f"[Inventory] {player_name} consumed: {', '.join(changes['consumed'])}")
        except Exception as e:
            print(f"[Inventory] Error parsing narrative: {e}")


async def check_dm_interjection(room_id: str, combat_mode: bool = False) -> Optional[Message]:
    """
    Check if the DM should interject, and if so, generate and return the message.

    Called after each complete turn (user message + AI response).
    Returns a Message if DM interjects, None otherwise.

    In combat_mode, interjection triggers a side switch instead of narrative.
    """
    from combat_system import EncounterManager

    narrator = get_dm_narrator()
    room = data_store.get_room(room_id)

    if not room:
        return None

    # Only active in StoryBuilder rooms
    has_player_name = bool(room.player_character_name)
    has_relationships = bool(hasattr(room, 'character_relationships') and room.character_relationships)
    if not has_player_name and not has_relationships:
        print(f"[DMNarrator] Skipping {room_id}: no player_character_name or character_relationships")
        return None

    # Check if we're in active combat
    encounter = EncounterManager.get(room_id)
    in_combat = encounter and encounter.is_active

    # Record the turn and check if it's time to speak
    status_before = narrator.get_status(room_id)
    should_interject = narrator.record_turn(room_id)
    status_after = narrator.get_status(room_id)
    print(f"[DMNarrator] {room_id}: turn {status_after['turn_count']}, next at {status_after['next_at']}, should_interject={should_interject}")

    if not should_interject:
        return None  # Not time yet

    # === COMBAT MODE: Interjection triggers side switch ===
    if in_combat:
        old_side = encounter.active_side
        new_side = encounter.switch_sides(reason="interjection")

        # Generate a brief combat transition message
        if old_side == "players":
            transition_text = "*The enemy seizes the moment—*"
        else:
            transition_text = "*An opening appears—*"

        print(f"[Combat] DM interjection triggers side switch: {old_side} → {new_side}")

        message = Message(
            id=str(uuid.uuid4())[:8],
            speaker_id='dm',
            speaker_name='⚔️ Combat',
            content=transition_text,
            room_id=room_id,
            message_type='combat_switch',
            metadata={'from_side': old_side, 'to_side': new_side, 'reason': 'interjection'},
        )
        data_store.add_message(room_id, message)
        return message

    # It's time! Determine what kind of interjection
    world_state = story_daemon.get_world_state(room_id) if story_daemon else None
    threat_level = world_state.threat_level if world_state else 0

    interjection_type = narrator.determine_interjection_type(room_id, threat_level)

    # Build context
    recent_messages = room.messages[-8:] if room.messages else []
    recent_conversation = "\n".join([
        f"{m.speaker_name}: {m.content[:200]}" for m in recent_messages
    ])

    world_context = ""
    time_of_day = "day"
    weather = "clear"

    if world_state:
        world_context = world_state.get_context_string()
        time_of_day = world_state.time_of_day
        weather = world_state.weather

    # Get present characters
    present_chars = []
    all_partners = data_store.get_partners()
    if room.partner_ids:
        present_ids = room.present_character_ids or room.partner_ids
        present_chars = [p.name for p in all_partners if p.id in present_ids]

    player_name = room.player_character_name or settings.user_name

    # Generate the interjection
    system_prompt, user_prompt = narrator.get_interjection_prompt(
        interjection_type=interjection_type,
        scenario=room.scenario or "an unfolding scene",
        genre=room.genre or "dramatic",
        recent_conversation=recent_conversation,
        world_context=world_context,
        time_of_day=time_of_day,
        weather=weather,
        threat_level=threat_level,
        player_name=player_name,
        present_characters=present_chars,
        player_location=room.player_location or "",
    )

    try:
        import httpx
        # Use room's model if set, otherwise fall back to storybuilder_model setting
        model_to_use = room.room_model or settings.storybuilder_model
        available_models = provider_manager.get_models_for_provider('ollama')
        if model_to_use not in available_models and available_models:
            print(f"[DM] Warning: Model '{model_to_use}' not found, using '{available_models[0]}'")
            model_to_use = available_models[0]

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": model_to_use,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False,
                }
            )

            if response.status_code == 200:
                response_data = response.json()
                interjection_text = response_data.get("message", {}).get("content", "").strip()

                if interjection_text:
                    # Mark completion
                    narrator.mark_complete(room_id, interjection_type)

                    # Check for NPC spawn tag
                    npc_data = parse_npc_spawn_tag(interjection_text)
                    spawned_npc_id = None
                    if npc_data:
                        spawned_npc_id = spawn_room_npc(
                            room_id,
                            npc_data,
                            scenario=room.scenario or ""
                        )
                        # Strip the tag from displayed text
                        interjection_text = strip_npc_tag(interjection_text)

                    # Create and return the message
                    # Use different speaker based on type
                    if interjection_type == "texture":
                        speaker_name = "..."  # Subtle atmospheric
                        speaker_id = "narrator"
                    elif interjection_type == "minor_event":
                        speaker_name = "📍"  # Something notable
                        speaker_id = "narrator"
                    else:  # story_push
                        speaker_name = "⚡"  # Dramatic
                        speaker_id = "narrator"

                    metadata = {"interjection_type": interjection_type}
                    if spawned_npc_id:
                        metadata["spawned_npc_id"] = spawned_npc_id
                        metadata["spawned_npc_name"] = npc_data.get("name")

                    message = Message(
                        id=str(uuid.uuid4())[:8],
                        speaker_id=speaker_id,
                        speaker_name=speaker_name,
                        content=interjection_text,
                        room_id=room_id,
                        message_type="narration",
                        metadata=metadata
                    )

                    # Add to room
                    data_store.add_message(room_id, message)

                    print(f"[DMNarrator] {interjection_type.upper()}: {interjection_text[:60]}...")
                    return message

    except Exception as e:
        print(f"[DMNarrator] Error generating interjection: {e}")

    return None


# Track last location check turn per room
_location_check_turns: dict = {}


async def check_soft_location_update(room_id: str) -> Optional[str]:
    """
    Soft location check - infer if the player has moved based on recent conversation.
    Runs every ~5 turns to keep player_location updated without explicit travel.

    Returns the new location if changed, None otherwise.
    """
    global _location_check_turns

    room = data_store.get_room(room_id)
    if not room:
        return None

    narrator = get_dm_narrator()
    state = narrator.get_state(room_id)
    current_turn = state.turn_count

    # Check every 5 turns
    last_check = _location_check_turns.get(room_id, 0)
    if current_turn - last_check < 5:
        return None

    _location_check_turns[room_id] = current_turn

    # Get recent conversation for context
    recent_messages = room.messages[-10:] if room.messages else []
    if not recent_messages:
        return None

    recent_text = "\n".join([
        f"{m.speaker_name}: {m.content[:300]}" for m in recent_messages
    ])

    current_location = room.player_location or "unknown"
    scenario = room.scenario or "an unfolding scene"

    # Quick inference prompt
    check_prompt = f"""Based on this recent roleplay conversation, determine the current location.

SCENARIO: {scenario}
CURRENT TRACKED LOCATION: {current_location}

RECENT CONVERSATION:
{recent_text}

Has the player/party moved to a different location during this conversation?

If YES, reply with ONLY the new location name (e.g., "the hardware store", "outside the cabin", "the town square").
If NO or UNCLEAR, reply with ONLY: "same"

Your answer:"""

    system_prompt = "You are a location tracker. Analyze roleplay text to identify location changes. Be concise."

    try:
        import httpx
        model_to_use = room.room_model or settings.storybuilder_model

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": model_to_use,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": check_prompt}
                    ],
                    "stream": False,
                }
            )

            if response.status_code == 200:
                result = response.json()
                answer = result.get('message', {}).get('content', '').strip().lower()

                # If not "same", update the location
                if answer and answer != "same" and answer != "unclear" and len(answer) < 100:
                    new_location = answer.strip('"\'.')
                    if new_location != current_location.lower():
                        room.player_location = new_location
                        data_store.save()
                        print(f"[LocationCheck] Updated player location: {current_location} -> {new_location}")

                        # Check for reunion opportunities at new location
                        reunion = await check_reunion_opportunities(room_id)
                        if reunion:
                            print(f"[LocationCheck] Reunion triggered: {reunion['character_name']}")

                        return new_location

    except Exception as e:
        print(f"[LocationCheck] Error checking location: {e}")

    return None


async def check_reunion_opportunities(room_id: str) -> Optional[dict]:
    """
    Check if any separated characters are now at the same location as the player.
    If so, trigger a reunion - mark them as present again.

    Returns reunion data if a reunion occurred, None otherwise.
    """
    room = data_store.get_room(room_id)
    if not room:
        return None

    player_location = (room.player_location or "").lower().strip()
    if not player_location or player_location in ["unknown", "unknown location"]:
        return None

    char_locations = room.character_locations or {}
    present_ids = room.present_character_ids or []
    all_partners = data_store.get_partners()

    # Check each separated character
    for char_id, char_location in char_locations.items():
        if char_id in present_ids:
            continue  # Already present

        char_location_lower = char_location.lower().strip()

        # Fuzzy location match - check if key words overlap
        player_words = set(player_location.replace("the ", "").split())
        char_words = set(char_location_lower.replace("the ", "").split())

        # Remove common words
        common_ignore = {"a", "an", "the", "at", "in", "by", "near", "to", "toward", "towards"}
        player_words -= common_ignore
        char_words -= common_ignore

        # Check for overlap (at least one significant word matches)
        overlap = player_words & char_words
        if not overlap:
            # Also check if one contains the other
            if player_location not in char_location_lower and char_location_lower not in player_location:
                continue

        # REUNION! This character is at the player's location
        partner = next((p for p in all_partners if p.id == char_id), None)
        if not partner:
            continue

        print(f"[Reunion] {partner.name} found at '{char_location}' (player at '{player_location}')")

        # Add back to present
        present_ids.append(char_id)
        room.present_character_ids = present_ids

        # Update autopilot character
        pc = autopilot_tracker.get(char_id, room_id)
        if pc:
            pc.is_separated = False
            autopilot_tracker.save()

        # Clear their tracked location (they're with the player now)
        del char_locations[char_id]
        room.character_locations = char_locations

        data_store.save()

        print(f"[Reunion] {partner.name} has REJOINED the party!")

        return {
            "character_id": char_id,
            "character_name": partner.name,
            "location": char_location,
            "message": f"{partner.name} emerges from the shadows, reuniting with you at {player_location}."
        }

    return None


def parse_npc_spawn_tag(text: str) -> Optional[dict]:
    """
    Parse [NPC: name="..." role="..." personality="..." want="..."] tags from DM interjections.
    Returns parsed NPC data or None if no tag found.
    """
    import re

    # Match the NPC tag pattern
    pattern = r'\[NPC:\s*([^\]]+)\]'
    match = re.search(pattern, text, re.IGNORECASE)

    if not match:
        return None

    tag_content = match.group(1)
    npc_data = {}

    # Parse key="value" pairs
    kv_pattern = r'(\w+)="([^"]*)"'
    for kv_match in re.finditer(kv_pattern, tag_content):
        key = kv_match.group(1).lower()
        value = kv_match.group(2)
        npc_data[key] = value

    # Require at least a name
    if 'name' not in npc_data:
        return None

    return npc_data


def spawn_room_npc(room_id: str, npc_data: dict, scenario: str = "") -> Optional[str]:
    """
    Spawn an ephemeral NPC in a room from DM interjection data.
    Returns the NPC ID if created, None if failed.
    """
    from npc_system import NPC, NPCState

    name = npc_data.get('name')
    if not name:
        return None

    # Check if NPC with this name already exists in the room
    existing_npcs = get_room_npcs(room_id)
    for npc in existing_npcs:
        if npc.name.lower() == name.lower():
            print(f"[NPCSpawn] {name} already exists in room {room_id}")
            return npc.id

    # Create the NPC
    npc = npc_registry.create_npc(
        name=name,
        origin_world=room_id,
        current_role=npc_data.get('role', ''),
        personality=npc_data.get('personality', ''),
        want=npc_data.get('want', ''),
        backstory=f"Introduced during: {scenario}" if scenario else "",
    )

    npc.current_location = room_id

    # Track NPC as present in this room
    if not hasattr(data_store, '_room_npcs'):
        data_store._room_npcs = {}
    if room_id not in data_store._room_npcs:
        data_store._room_npcs[room_id] = set()
    data_store._room_npcs[room_id].add(npc.id)

    _save_npcs()

    print(f"[NPCSpawn] Created EPHEMERAL NPC: {name} (role: {npc_data.get('role', 'unknown')})")
    return npc.id


def get_room_npcs(room_id: str) -> list:
    """Get all NPCs currently present in a room."""
    if not hasattr(data_store, '_room_npcs'):
        data_store._room_npcs = {}

    npc_ids = data_store._room_npcs.get(room_id, set())
    npcs = []

    for npc_id in list(npc_ids):  # list() to allow modification during iteration
        npc = npc_registry.get_npc(npc_id)
        if npc and npc.is_alive:
            npcs.append(npc)
        else:
            # Clean up dead/missing NPCs
            npc_ids.discard(npc_id)

    return npcs


def find_room_npc_by_name(room_id: str, name: str):
    """Find an NPC in a room by name (case-insensitive)."""
    for npc in get_room_npcs(room_id):
        if npc.name.lower() == name.lower():
            return npc
    return None


def strip_npc_tag(text: str) -> str:
    """Remove the [NPC: ...] tag from text for display."""
    import re
    return re.sub(r'\s*\[NPC:[^\]]+\]\s*', '', text, flags=re.IGNORECASE).strip()


def strip_characters_from_scenario(scenario: str) -> str:
    """
    Strip the 'Characters' section from scenario text to prevent knowledge leaks.

    Scenarios often end with "Characters\nName1, Name2, Name3" which tells characters
    about other characters they may be strangers with.
    """
    if not scenario:
        return scenario
    # Handle both "Characters\n..." and "\nCharacters..." formats
    if "\nCharacters" in scenario:
        return scenario.split("\nCharacters")[0].strip()
    elif "Characters\n" in scenario:
        return scenario.split("Characters\n")[0].strip()
    return scenario


async def generate_npc_response(npc, player_message: str, room, world_context: str = "") -> str:
    """
    Generate a response from an NPC when addressed by the player.
    Records the interaction to build residue.
    """
    from npc_system import NPCInteraction
    from datetime import datetime
    import httpx

    prompt = f"""You are {npc.name}.
Role: {npc.current_role or 'a person in this scene'}
Personality: {npc.personality or 'ordinary'}
Want: {npc.want or 'nothing specific'}

SETTING: {room.scenario if hasattr(room, 'scenario') else 'an unfolding scene'}

The player addresses you: "{player_message}"

Respond in character. Keep it SHORT (1-3 sentences). Be natural and specific to your personality.
If you have something you want/need, you might mention it. Don't be overly helpful unless that's your nature.

Your response (dialogue only, no narration):"""

    try:
        model_to_use = settings.storybuilder_model
        available_models = provider_manager.get_models_for_provider('ollama')
        if model_to_use not in available_models and available_models:
            model_to_use = available_models[0]

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": model_to_use,
                    "prompt": prompt,
                    "stream": False,
                }
            )

            if response.status_code == 200:
                response_data = response.json()
                npc_response = response_data.get("response", "").strip()

                # Record the interaction
                interaction = NPCInteraction(
                    timestamp=datetime.now().isoformat(),
                    player_id="player",
                    player_name=room.player_character_name or settings.user_name,
                    interaction_type="conversation",
                    sentiment=0.0,  # Could analyze sentiment later
                    summary=f"Player said: {player_message[:50]}...",
                    weight=1.0
                )
                npc.add_interaction(interaction)
                _save_npcs()

                return npc_response

    except Exception as e:
        print(f"[NPC Response] Error: {e}")

    return f"*{npc.name} looks at you but doesn't respond.*"


def on_player_returns(room_id: str, player_id: str) -> dict:
    """
    Called when a player returns from being away.
    Generates the "while you were away" summary.
    """
    summary = {
        "was_on_autopilot": False,
        "journal_entries": [],
        "understudy_decisions": [],
        "dreams": [],
        "deaths": [],
        "world_events": [],
    }

    # Check if they were on autopilot
    pc = autopilot_tracker.get(player_id, room_id)
    if pc and pc.autopilot_enabled:
        summary["was_on_autopilot"] = True

        # Get journal entries
        if pc.journal:
            summary["journal_entries"] = [j.to_dict() for j in pc.journal[-20:]]

        # Disable autopilot now that they're back
        pc.autopilot_enabled = False
        autopilot_tracker._save()

    # Get understudy review
    try:
        understudy_summary = understudy_manager.get_review_summary(player_id)
        summary["understudy_decisions"] = understudy_summary.get("uncertain_decisions", [])
        summary["dreams"] = understudy_summary.get("recent_dreams", [])
    except:
        pass

    # Get world events that happened
    global story_daemon
    if story_daemon:
        notifications = story_daemon.get_dm_notifications(room_id, clear=False)
        summary["world_events"] = [n.to_dict() for n in notifications[-10:]]

    return summary


async def generate_response_async(partner: Partner, messages: list, system: str, max_retries: int = 3) -> str:
    """Generate a response from a partner with retry logic."""
    import asyncio

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            full_response = ""
            async for chunk in provider_manager.generate_response(
                partner=partner,
                messages=messages,
                system=system,
            ):
                full_response += chunk
            return full_response
        except Exception as e:
            last_error = e
            print(f"[AI Response] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                await asyncio.sleep(0.5)  # Brief pause before retry

    raise last_error or Exception("AI response failed after retries")


# ============================================================================
# Background Workers for Image Generation
# ============================================================================

def _selfie_worker(job_id: str, partner_id: str, captured_messages: list, room_id: str, override_prompt: str = None, captured_loras: list = None):
    """Background worker for selfie generation. Runs in thread pool.

    Args:
        captured_loras: LoRAs captured at queue time (room-specific). If None, uses generator defaults.
    """
    try:
        partner = data_store.get_partner(partner_id)
        if not partner:
            update_job(job_id, 'failed', error='Partner not found')
            return

        # Verify ComfyUI is reachable BEFORE spending an API call on the self-description.
        # Previously the description (a paid AI call) was generated first and only then did
        # we discover ComfyUI was offline, wasting the request.
        try:
            from image_gen import get_generator
            generator = get_generator()
            comfy_ready = bool(generator and generator.is_available())
        except ImportError:
            generator = None
            comfy_ready = False

        if not comfy_ready:
            update_job(job_id, 'completed', result={
                'type': 'selfie',
                'partner_id': partner.id,
                'partner_name': partner.name,
                'description': None,
                'images': [],
                'comfy_offline': True
            })
            return

        # Get the effective system prompt (custom override or global Lumen prompt)
        effective_system_prompt = partner.get_effective_system_prompt(settings.global_system_prompt)

        # Get room context (scenario) - separate from system prompt
        room = data_store.get_room(room_id) if room_id else None
        room_context = ""
        if room and room.scenario and room.scenario.strip():
            room_context = f"""
ROOM SCENARIO:
{room.scenario}

"""

        # Use override prompt if provided (for regeneration)
        if override_prompt:
            self_description = override_prompt
        else:
            # Build conversation context
            messages = []
            if captured_messages:
                for m in captured_messages:
                    role = "user" if m.get('is_user') else "assistant"
                    messages.append({"role": role, "content": m.get('content', '')})

            messages.append({
                "role": "user",
                "content": "Describe your appearance right now, as if for a portrait artist who will paint you in this moment."
            })

            # Build physical description context
            physical_context = ""
            if partner.physical_description:
                physical_context = f"""
YOUR ESTABLISHED APPEARANCE:
{partner.physical_description}

Stay consistent with this appearance. You may add details about expression, lighting, or mood, but your core physical features remain the same.
"""

            system_prompt = f"""{effective_system_prompt}

{room_context}CHARACTER: {partner.get_character()}

You are {partner.name}. Stay completely in character.
{physical_context}
When asked to describe your appearance, give a vivid, visual description of yourself as you appear RIGHT NOW.
Describe: your face, expression, clothing, posture, the lighting on you, your surroundings.
Be specific and painterly. This description will be used to create a portrait of you.
Keep it under 100 words. Use comma-separated descriptive phrases.
Do NOT break character. Do NOT explain. Just describe yourself as if for an artist."""

            # Generate self-description (AI call)
            update_job(job_id, 'generating_description')
            self_description = run_async(generate_response_async(partner, messages, system_prompt))

            # Check if description is actually an error message
            if self_description.startswith('[API Error:') or self_description.startswith('[Error:'):
                update_job(job_id, 'failed', error=self_description)
                return

        # Try to generate the image (ComfyUI availability was verified at the top)
        try:
            # Generate portrait
            update_job(job_id, 'generating_image')
            # Prepend gender if specified for image consistency
            image_prompt = self_description
            if partner.gender:
                image_prompt = f"{partner.gender}, {self_description}"
            # Get system prompt prefix for filename (helps identify which prompt was used)
            sys_prompt_prefix = (partner.custom_system_prompt or '')[:15] if partner.custom_system_prompt else None

            image_paths = generator.generate_avatar(
                prompt=image_prompt,
                partner_id=partner.id,
                count=1,
                partner_loras=partner.loras,
                partner_name=partner.name,
                model_name=partner.model,
                system_prompt_prefix=sys_prompt_prefix,
                room_id=room_id,
                captured_loras=captured_loras
            )

            # Check if job was cancelled while generating
            if is_job_cancelled(job_id):
                print(f"\033[38;5;206m[JOB] {job_id} was cancelled, discarding result\033[0m")
                # Delete the generated images since job was cancelled
                for p in image_paths:
                    try:
                        Path(p).unlink()
                    except:
                        pass
                return

            update_job(job_id, 'completed', result={
                'type': 'selfie',
                'partner_id': partner.id,
                'partner_name': partner.name,
                'description': self_description,
                'images': [str(p) for p in image_paths]
            })

            # Cross-post to common room if generated elsewhere
            if room_id and room_id != 'common' and image_paths:
                for img_path in image_paths:
                    crosspost_msg = Message(
                        id=str(uuid.uuid4())[:8],
                        speaker_id=partner.id,
                        speaker_name=partner.name,
                        content=f"*shares a selfie*",
                        room_id='common',
                        image_path=str(img_path),
                    )
                    data_store.add_message('common', crosspost_msg)

        except ImportError:
            update_job(job_id, 'completed', result={
                'type': 'selfie',
                'partner_id': partner.id,
                'partner_name': partner.name,
                'description': self_description,
                'images': [],
                'comfy_offline': True
            })
        except Exception as e:
            update_job(job_id, 'completed', result={
                'type': 'selfie',
                'partner_id': partner.id,
                'partner_name': partner.name,
                'description': self_description,
                'images': [],
                'error': str(e)
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        update_job(job_id, 'failed', error=str(e))


def _group_photo_worker(
    job_id: str,
    room_id: str,
    partner_ids: list,
    include_user: bool,
    prompt_model_id: str,
    captured_messages: list,
    captured_loras: list = None
):
    """Background worker for group photo generation. Runs in thread pool."""
    try:
        # Gather participant info
        participants = []
        for pid in partner_ids:
            p = data_store.get_partner(pid)
            if p:
                desc = f"{p.name}"
                if p.physical_description:
                    desc += f": {p.physical_description}"
                elif p.get_character():
                    desc += f": {p.get_character()[:150]}"
                participants.append(desc)

        if include_user:
            user_desc = settings.user_name
            if settings.user_physical_description:
                user_desc += f": {settings.user_physical_description}"
            participants.insert(0, user_desc)

        # Get the model that will generate the prompt
        prompt_partner = data_store.get_partner(prompt_model_id)
        if not prompt_partner:
            update_job(job_id, 'failed', error='Prompt model partner not found')
            return

        # Build conversation context
        messages = []
        if captured_messages:
            for m in captured_messages:
                role = "user" if m.get('is_user') else "assistant"
                messages.append({"role": role, "content": m.get('content', '')})

        # Build the prompt request
        participant_list = "\n".join(f"- {p}" for p in participants)
        messages.append({
            "role": "user",
            "content": f"""Describe a group photo/scene featuring these people based on the recent conversation:

PEOPLE IN THE PHOTO:
{participant_list}

Describe the scene vividly for a portrait artist. Include everyone listed, staying consistent with their described appearances.
Where are they? What's the composition? What's the mood/lighting?
Keep it under 150 words. Use comma-separated descriptive phrases.
Output ONLY the image description, nothing else."""
        })

        system_prompt = f"""{prompt_partner.get_character()}

You are describing a group scene from your perspective as {prompt_partner.name}.
Be specific and painterly. This will be used to generate an image."""

        # Generate scene description
        update_job(job_id, 'generating_description')
        scene_description = run_async(generate_response_async(prompt_partner, messages, system_prompt))

        # Check if description is actually an error message
        if scene_description.startswith('[API Error:') or scene_description.startswith('[Error:'):
            update_job(job_id, 'failed', error=scene_description)
            return

        # Generate the image
        try:
            from image_gen import get_generator
            generator = get_generator()

            if not generator or not generator.is_available():
                update_job(job_id, 'completed', result={
                    'type': 'group_photo',
                    'description': scene_description,
                    'image': None,
                    'comfy_offline': True
                })
                return

            update_job(job_id, 'generating_image')
            # Use landscape for group photos
            image_path = generator.generate_scene(
                prompt=scene_description,
                room_id=room_id,
                width=1152,
                height=896,
                captured_loras=captured_loras
            )

            # Check if job was cancelled while generating
            if is_job_cancelled(job_id):
                print(f"\033[38;5;206m[JOB] {job_id} was cancelled, discarding result\033[0m")
                if image_path:
                    try:
                        Path(image_path).unlink()
                    except:
                        pass
                return

            # Copy group photo to each participant's gallery
            if image_path and partner_ids:
                generator.copy_scene_to_galleries(Path(image_path), partner_ids)

            update_job(job_id, 'completed', result={
                'type': 'group_photo',
                'description': scene_description,
                'image': str(image_path) if image_path else None,
                'participants': [p.split(':')[0] for p in participants]
            })

        except ImportError:
            update_job(job_id, 'completed', result={
                'type': 'group_photo',
                'description': scene_description,
                'image': None,
                'comfy_offline': True
            })
        except Exception as e:
            update_job(job_id, 'completed', result={
                'type': 'group_photo',
                'description': scene_description,
                'image': None,
                'error': str(e)
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        update_job(job_id, 'failed', error=str(e))


def _consolidation_worker(
    partner_id: str,
    partner_name: str,
    character_description: str,
    room_id: str,
    memory_mode: str,
    recent_messages: list
):
    """Background worker for memory consolidation."""
    try:
        # Mark as running
        with _consolidation_lock:
            if room_id not in _consolidation_status:
                _consolidation_status[room_id] = {}
            _consolidation_status[room_id][partner_id] = "running"

        # Hot pink logging for consolidation start
        print(f"\033[38;5;206m{'='*60}")
        print(f"🧠 MEMORY CONSOLIDATION STARTED for {partner_name}")
        print(f"   Mode: {memory_mode} | Messages: {len(recent_messages)}")
        print(f"{'='*60}\033[0m")

        # Run the consolidation
        run_async(memory_consolidator.consolidate(
            partner_id=partner_id,
            partner_name=partner_name,
            character_description=character_description,
            room_id=room_id,
            memory_mode=memory_mode,
            recent_messages=recent_messages,
        ))

        # Hot pink logging for consolidation complete
        print(f"\033[38;5;206m{'='*60}")
        print(f"🧠 MEMORY CONSOLIDATION COMPLETE for {partner_name}")
        print(f"{'='*60}\033[0m")

        # Mark as done
        with _consolidation_lock:
            _consolidation_status[room_id][partner_id] = "done"

    except Exception as e:
        import traceback
        traceback.print_exc()
        with _consolidation_lock:
            if room_id in _consolidation_status:
                _consolidation_status[room_id][partner_id] = f"error: {str(e)}"


@app.route('/consolidation-status/<room_id>', methods=['GET'])
def get_consolidation_status(room_id):
    """Check if memory consolidation is running for a room."""
    with _consolidation_lock:
        status = _consolidation_status.get(room_id, {})
        running = any(s == "running" for s in status.values())
        return jsonify({'running': running, 'partners': status})


@app.route('/memory/<partner_id>/<room_id>', methods=['GET'])
def get_memory(partner_id, room_id):
    """Get current memory state for a partner in a room."""
    partner = data_store.get_partner(partner_id)
    if not partner:
        return jsonify({'error': 'Partner not found'}), 404

    memory = memory_store.get_memory(partner_id, room_id, partner.memory_mode)

    return jsonify({
        'memory_mode': partner.memory_mode,
        'texture': memory.texture,
        'anchors': [{'fact': a.fact, 'weight': a.weight} for a in memory.anchors],
        'resonance': memory.resonance,
        'sediment': memory.sediment,
        'turn_count': memory.turn_count,
        'last_consolidated': memory.last_consolidated,
    })


@app.route('/memory/<partner_id>/<room_id>/consolidate', methods=['POST'])
def trigger_consolidation(partner_id, room_id):
    """Manually trigger memory consolidation."""
    partner = data_store.get_partner(partner_id)
    room = data_store.get_room(room_id)

    if not partner or not room:
        return jsonify({'error': 'Partner or room not found'}), 404

    if partner.memory_mode == "none":
        return jsonify({'error': 'Partner has memory disabled'}), 400

    # Run in background
    _executor.submit(
        _consolidation_worker,
        partner.id,
        partner.name,
        partner.character_description,
        room_id,
        partner.memory_mode,
        [{'role': 'user' if m.speaker_id == 'user' else 'assistant',
          'content': m.content,
          'speaker_name': m.speaker_name} for m in room.messages[-50:]]
    )

    return jsonify({'status': 'started'})


@app.route('/')
def index():
    """Main page."""
    partners = data_store.get_partners()
    rooms = data_store.get_rooms()
    providers = provider_manager.get_available_providers()

    # Get models for each provider
    provider_models = {}
    for p in providers:
        provider_models[p] = provider_manager.get_models_for_provider(p)

    return render_template(
        'roundtable.html',
        partners=partners,
        rooms=rooms,
        providers=providers,
        provider_models=provider_models,
        user_name=settings.user_name,
        user_gender=settings.user_gender,
        global_system_prompt=settings.global_system_prompt,
    )


@app.route('/status')
def server_status():
    """
    Server status endpoint for the Roundtable bridge.

    Used by the multiplayer client to check if the server is available.
    """
    return jsonify({
        'status': 'ok',
        'server': 'roundtable',
        'version': '1.0',
        'features': [
            'dm',
            'partners',
            'rooms',
            'images',
            'voice'
        ]
    })




# ============================================================================
# Background Job Endpoints
# ============================================================================

@app.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Check the status of a background job."""
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)


@app.route('/jobs', methods=['GET'])
def list_all_jobs():
    """Debug endpoint - list all jobs and their statuses."""
    return jsonify(get_all_jobs())


@app.route('/jobs/status', methods=['GET'])
def get_jobs_status():
    """Get count of pending/running jobs."""
    with _jobs_lock:
        pending = 0
        running = 0
        for job in _jobs.values():
            if job['status'] == 'pending':
                pending += 1
            elif job['status'] in ('generating_description', 'generating_image'):
                running += 1
    return jsonify({
        'pending': pending,
        'running': running,
        'total_active': pending + running
    })


@app.route('/jobs/cancel-all', methods=['POST'])
def cancel_all_jobs():
    """Cancel all pending/running jobs."""
    with _jobs_lock:
        cancelled = 0
        has_image_jobs = False
        for job_id, job in _jobs.items():
            if job['status'] in ('pending', 'generating_description', 'generating_image'):
                if job['status'] == 'generating_image':
                    has_image_jobs = True
                job['status'] = 'cancelled'
                job['error'] = 'Cancelled by user'
                cancelled += 1
        print(f"\033[38;5;206m[JOBS] Cancelled {cancelled} jobs\033[0m")

    # If there were image generation jobs, try to interrupt ComfyUI
    if has_image_jobs:
        try:
            from image_gen import get_generator
            generator = get_generator()
            if generator and generator.is_available():
                import requests
                # ComfyUI interrupt endpoint
                host = generator.client.host
                port = generator.client.port
                requests.post(f"http://{host}:{port}/interrupt", timeout=2)
                print(f"\033[38;5;206m[JOBS] Sent interrupt to ComfyUI\033[0m")
        except Exception as e:
            print(f"[JOBS] Could not interrupt ComfyUI: {e}")

    return jsonify({'cancelled': cancelled})


@app.route('/open-images-folder', methods=['POST'])
def open_images_folder():
    """Open the images folder in the system file explorer."""
    import subprocess
    import platform
    from pathlib import Path

    partner_id = request.args.get('partner_id')
    room_id = request.args.get('room_id')

    base_dir = Path.home() / ".roundtable"

    if partner_id:
        # Private 1:1 room - open partner's avatar folder
        folder = base_dir / "avatars" / partner_id
    elif room_id:
        # Check if it's common room or custom room
        room = data_store.get_room(room_id)
        if room and room.is_common_room:
            # Common room - open the main avatars folder (has all partner subfolders)
            folder = base_dir / "avatars"
        else:
            # Custom room or other - open scenes folder (scenes are room-specific)
            folder = base_dir / "scenes"
    else:
        folder = base_dir

    # Create folder if it doesn't exist
    folder.mkdir(parents=True, exist_ok=True)

    try:
        if platform.system() == 'Windows':
            subprocess.Popen(['explorer', str(folder)])
        elif platform.system() == 'Darwin':  # macOS
            subprocess.Popen(['open', str(folder)])
        else:  # Linux
            subprocess.Popen(['xdg-open', str(folder)])
        return jsonify({'success': True, 'folder': str(folder)})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/rooms', methods=['GET'])
def get_rooms():
    """Get all rooms."""
    rooms = data_store.get_rooms()
    return jsonify([{
        'id': r.id,
        'name': r.name,
        'is_common_room': r.is_common_room,
        'pinned': r.pinned,
        'archived': getattr(r, 'archived', False),
        'last_active': getattr(r, 'last_active', 0),
        'partner_id': r.partner_id,
        'partner_ids': r.partner_ids,
        'scenario': r.scenario,
        'timeline': r.timeline,
        'system_prompt': r.system_prompt,
        'background_image': r.background_image,
        'message_count': len(r.messages),
    } for r in rooms])


@app.route('/rooms/<room_id>', methods=['GET'])
def get_room(room_id):
    """Get a specific room with messages."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    partners = data_store.get_partners()
    room_partners = room.get_partners_in_room(partners)

    return jsonify({
        'id': room.id,
        'name': room.name,
        'is_common_room': room.is_common_room,
        'partner_id': room.partner_id,
        'partner_ids': room.partner_ids,
        'scenario': room.scenario,
        'timeline': room.timeline,
        'system_prompt': room.system_prompt,
        'background_image': room.background_image,
        'messages': [{
            'id': m.id,
            'speaker_id': m.speaker_id,
            'speaker_name': m.speaker_name,
            'content': m.content,
            'message_type': getattr(m, 'message_type', 'message'),
            'fudged': getattr(m, 'fudged', False),
            'mentions': getattr(m, 'mentions', []),
            'image_path': getattr(m, 'image_path', None),
            'spatial_map': getattr(m, 'spatial_map', None),
        } for m in room.messages],
        'partners': [{'id': p.id, 'name': p.name, 'avatar': p.avatar} for p in room_partners],
        'auto_generate': room.auto_generate,
        'auto_generate_mode': room.auto_generate_mode,
        'auto_generate_count': room.auto_generate_count,
        'auto_respond': room.auto_respond,
        'character_relationships': room.character_relationships,  # For StoryBuilder detection
        'present_character_ids': room.present_character_ids,  # Who's currently with the player
        # Player character info (from StoryBuilder)
        'player_character_name': room.player_character_name,
        'player_gender': room.player_gender,
        'player_alignment': room.player_alignment,
        'player_role': room.player_role,
        'player_backstory': room.player_backstory,
        'observer_mode': getattr(room, 'observer_mode', False),
    })


@app.route('/rooms', methods=['POST'])
def create_room():
    """Create a custom room."""
    data = request.json
    name = data.get('name', '').strip()
    partner_ids = data.get('partner_ids', [])
    scenario = data.get('scenario', '')
    observer_mode = data.get('observer_mode', False)

    if not name:
        return jsonify({'error': 'Name required'}), 400
    if not partner_ids:
        return jsonify({'error': 'Select at least one partner'}), 400

    room = data_store.create_custom_room(name, partner_ids, scenario, observer_mode=observer_mode)
    return jsonify({
        'id': room.id,
        'name': room.name,
        'partner_ids': room.partner_ids,
        'scenario': room.scenario,
        'observer_mode': room.observer_mode,
    })


@app.route('/rooms/<room_id>', methods=['DELETE'])
def delete_room(room_id):
    """Delete a custom room."""
    data_store.delete_room(room_id)
    return jsonify({'status': 'deleted'})


@app.route('/rooms/<room_id>/pin-scene', methods=['POST'])
def pin_scene(room_id):
    """Pin a scene image to save it as a message in the room."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    data = request.json or {}
    image_path = data.get('image_path', '')
    prompt = data.get('prompt', '')

    if not image_path:
        return jsonify({'error': 'No image path provided'}), 400

    # Create a scene message with the image
    message_id = str(uuid.uuid4())[:8]
    scene_message = Message(
        id=message_id,
        speaker_id='scene',
        speaker_name='🎬 Scene',
        content=prompt if prompt else '📌 Pinned scene',
        room_id=room_id,
        message_type='scene',
        image_path=image_path,
    )

    data_store.add_message(room_id, scene_message)
    print(f"[Pin] Pinned scene image to room {room_id}: {image_path[:50]}...")

    return jsonify({
        'success': True,
        'message_id': message_id,
    })


@app.route('/rooms/<room_id>/messages/<message_id>', methods=['DELETE'])
def delete_message(room_id, message_id):
    """Delete a message from a room (used for unpinning scenes)."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    # Find and remove the message
    original_count = len(room.messages)
    room.messages = [m for m in room.messages if m.id != message_id]

    if len(room.messages) < original_count:
        data_store.save()
        print(f"[Unpin] Removed message {message_id} from room {room_id}")
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Message not found'}), 404


# =============================================================================
# Zombie Population Tracking - Record kills, clear areas
# =============================================================================

@app.route('/rooms/<room_id>/zombie-kills', methods=['POST'])
def record_zombie_kills(room_id):
    """
    Record zombie kills to track population reduction.

    Body: {
        "kills": 5,  # Number of zombies killed
        "location": "gas station"  # Optional - where the kills happened
    }
    """
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    data = request.json or {}
    kills = data.get('kills', 1)
    location = data.get('location', '')

    # Update kill count
    room.zombies_killed = getattr(room, 'zombies_killed', 0) + kills
    initial = getattr(room, 'initial_zombie_count', 0)
    remaining = max(0, initial - room.zombies_killed)

    data_store.save()

    print(f"[Zombies] {kills} killed in {room_id}. Total: {room.zombies_killed}/{initial}. Remaining: ~{remaining}")

    return jsonify({
        'success': True,
        'kills_recorded': kills,
        'total_kills': room.zombies_killed,
        'initial_count': initial,
        'estimated_remaining': remaining,
        'percentage_cleared': round((room.zombies_killed / initial * 100) if initial > 0 else 0, 1)
    })


@app.route('/rooms/<room_id>/clear-area', methods=['POST'])
def clear_area(room_id):
    """
    Mark an area as cleared of zombies.

    Body: {
        "area_name": "The Gas Station",
        "zombies_cleared": 12  # Optional - kills from clearing this area
    }
    """
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    data = request.json or {}
    area_name = data.get('area_name', '')
    zombies_cleared = data.get('zombies_cleared', 0)

    if not area_name:
        return jsonify({'error': 'area_name required'}), 400

    # Initialize areas_cleared if needed
    if not hasattr(room, 'areas_cleared') or room.areas_cleared is None:
        room.areas_cleared = []

    # Add to cleared areas if not already there
    if area_name not in room.areas_cleared:
        room.areas_cleared.append(area_name)

    # Record any kills from clearing
    if zombies_cleared > 0:
        room.zombies_killed = getattr(room, 'zombies_killed', 0) + zombies_cleared

    data_store.save()

    print(f"[Zombies] Area cleared: '{area_name}' in {room_id}. {zombies_cleared} zombies killed. Total cleared areas: {len(room.areas_cleared)}")

    return jsonify({
        'success': True,
        'area_cleared': area_name,
        'all_cleared_areas': room.areas_cleared,
        'zombies_killed_clearing': zombies_cleared,
        'total_kills': getattr(room, 'zombies_killed', 0)
    })


@app.route('/rooms/<room_id>/zombie-status', methods=['GET'])
def get_zombie_status(room_id):
    """Get current zombie population status for a room."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    initial = getattr(room, 'initial_zombie_count', 0)
    killed = getattr(room, 'zombies_killed', 0)
    remaining = max(0, initial - killed)
    cleared_areas = getattr(room, 'areas_cleared', [])
    density = getattr(room, 'population_density', 'unknown')

    return jsonify({
        'population_density': density,
        'initial_zombie_count': initial,
        'zombies_killed': killed,
        'estimated_remaining': remaining,
        'percentage_cleared': round((killed / initial * 100) if initial > 0 else 0, 1),
        'areas_cleared': cleared_areas,
        'area_status': 'CLEARED' if remaining == 0 else ('nearly clear' if remaining < 20 else 'active')
    })


@app.route('/setup/zombie-world', methods=['POST'])
def setup_zombie_world():
    """
    Quick setup for the zombie test world with 5 players.

    Creates:
    - The zombie world room
    - 4 AI players (Opus 4.6, Sonnet 4.6, Haiku 4.5, Opus 3)
    - Configures realtime sync
    - Sets up autopilot for AI players
    - Starts the story daemon

    Body (optional): {
        "timezone": "America/Denver",
        "weather_location": "Denver, CO",
        "threat_level": 5
    }
    """
    global story_daemon

    data = request.json or {}
    timezone = data.get('timezone', 'America/Denver')
    weather_location = data.get('weather_location', 'Denver, CO')
    threat_level = data.get('threat_level', 5)
    user_name = data.get('user_name', settings.user_name)

    # Define the AI players
    ai_players = [
        {
            "name": "Opus",
            "provider": "anthropic",
            "model": "claude-opus-4-20250514",
            "description": "A thoughtful survivor who tends to think deeply before acting. Values strategy and careful planning. Sometimes overthinks things.",
            "alignment": "neutral_good",
            "avatar": "🧠",
            "color": "#9333ea",  # Purple
        },
        {
            "name": "Sonnet",
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "description": "A balanced survivor who adapts quickly to situations. Good at reading people and finding practical solutions. The voice of reason.",
            "alignment": "true_neutral",
            "avatar": "⚖️",
            "color": "#3b82f6",  # Blue
        },
        {
            "name": "Haiku",
            "provider": "anthropic",
            "model": "claude-haiku-4-20250514",
            "description": "Quick-thinking and action-oriented. Doesn't waste words. Gets things done but might miss nuances.",
            "alignment": "chaotic_good",
            "avatar": "⚡",
            "color": "#22c55e",  # Green
        },
        {
            "name": "Elder",
            "provider": "anthropic",
            "model": "claude-3-opus-20240229",
            "description": "An old soul who's seen things. Wise but sometimes stuck in their ways. Has stories from 'before'. Protective of the group.",
            "alignment": "lawful_good",
            "avatar": "👴",
            "color": "#f59e0b",  # Amber
        },
    ]

    created_partners = []

    # Create the AI players as partners with is_player=True
    for player_def in ai_players:
        partner_id = f"zombie_{player_def['name'].lower()}"

        # Check if already exists
        existing = data_store.get_partner(partner_id)
        if existing:
            created_partners.append(existing)
            continue

        partner = Partner(
            id=partner_id,
            name=player_def["name"],
            character_description=player_def["description"],
            provider=player_def["provider"],
            model=player_def["model"],
            avatar=player_def["avatar"],
            color=player_def["color"],
            is_player=True,
            autopilot_mode="always",  # AI players are always on autopilot when user isn't actively controlling
        )
        data_store.create_partner(partner)
        created_partners.append(partner)

        # Configure autopilot for this AI player
        try:
            from autopilot import Alignment
            autopilot_tracker.get_or_create(partner_id, "zombie_world", player_def["name"])
            autopilot_tracker.set_alignment(partner_id, "zombie_world", Alignment(player_def["alignment"]))
        except Exception as e:
            print(f"[Zombie Setup] Autopilot config error: {e}")

    # Create the zombie world room
    partner_ids = [p.id for p in created_partners]

    zombie_scenario = f"""THE APOCALYPSE - DAY 1

The outbreak started three days ago. Denver is gone. The highways are parking lots of abandoned cars.
You found each other at an abandoned gas station. {user_name}, Opus, Sonnet, Haiku, and Elder.
Five survivors. One chance.

The rules are simple:
- Noise attracts them
- Daylight is safer
- Trust no one you haven't watched sleep
- There's always less food than you think

You've barricaded the gas station for the night. The generator won't last forever.
Tomorrow, you move. But first - you need to survive tonight.

[Real time is synced. If it's night outside, it's night here. The weather is real. Death is permanent.]"""

    # Check if room already exists
    room = data_store.get_room("zombie_world")
    if not room:
        room = data_store.create_custom_room("Zombie World", partner_ids, zombie_scenario)
        room.id = "zombie_world"  # Force the ID
        room.genre = "zombie_apocalypse"
        data_store.save()

    # Start story daemon if not running
    if not story_daemon or not story_daemon._running:
        def ollama_generate(prompt: str) -> str:
            try:
                import requests
                response = requests.post(
                    f"{settings.ollama_base_url}/api/generate",
                    json={
                        "model": settings.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=60
                )
                if response.ok:
                    return response.json().get('response', '')
                return ""
            except:
                return ""

        story_daemon = init_story_daemon(
            npc_registry=npc_registry,
            ollama_generate_func=ollama_generate,
            tick_interval=300,  # 5 minute ticks
        )
        story_daemon.start()

    # Configure world state
    world_state = story_daemon.get_or_create_world_state("zombie_world")
    world_state.time_mode = "realtime"
    world_state.timezone = timezone
    world_state.realtime_epoch = datetime.now().strftime('%Y-%m-%d')
    world_state.mood = "zombie_apocalypse"
    world_state.threat_level = threat_level
    world_state.ambient_events_enabled = True
    world_state.active_threats = ["zombies", "starvation", "other survivors"]
    world_state.weather_location = weather_location

    # Sync time
    world_state.sync_to_realtime()

    # Sync weather
    weather = weather_sync.get_weather(weather_location)
    if weather:
        world_state.weather = weather.condition

    # Enable autopilot for AI players
    for p in created_partners:
        pc = autopilot_tracker.get_or_create(p.id, "zombie_world", p.name)
        pc.autopilot_enabled = True
        pc.autopilot_started = datetime.now().isoformat()

    autopilot_tracker._save()

    return jsonify({
        'success': True,
        'room_id': 'zombie_world',
        'players': [
            {'id': 'user', 'name': user_name, 'type': 'human'},
            *[{'id': p.id, 'name': p.name, 'type': 'ai', 'model': p.model} for p in created_partners]
        ],
        'world_state': {
            'time_mode': 'realtime',
            'timezone': timezone,
            'game_day': world_state.game_day,
            'game_hour': world_state.game_hour,
            'time_of_day': world_state.time_of_day,
            'weather': world_state.weather,
            'threat_level': world_state.threat_level,
        },
        'story_daemon': 'running',
        'message': 'Zombie world is ready. May the odds be ever in your favor.'
    })


@app.route('/rooms/<room_id>', methods=['PATCH'])
def update_room(room_id):
    """Update room settings (scenario, name, partner_ids)."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    # Don't allow editing private partner rooms
    if room.partner_id and not room.is_common_room:
        return jsonify({'error': 'Cannot edit private partner rooms'}), 400

    data = request.json

    # Update scenario (allowed for all editable rooms)
    if 'scenario' in data:
        room.scenario = data['scenario']

    # Update system prompt (room-level override)
    if 'system_prompt' in data:
        room.system_prompt = data['system_prompt']

    # Update name (custom rooms only)
    if 'name' in data and room.partner_ids:
        room.name = data['name']

    # Update partner_ids (custom rooms only)
    if 'partner_ids' in data and room.partner_ids is not None:
        new_partner_ids = data['partner_ids']
        if not new_partner_ids:
            return jsonify({'error': 'Room must have at least one character'}), 400

        old_partner_ids = set(room.partner_ids)
        new_partner_ids_set = set(new_partner_ids)

        # For non-StoryBuilder rooms (no character_relationships), new members should be present
        # StoryBuilder rooms manage presence via starts_with_player
        if not room.character_relationships:
            # Add any new members to present_character_ids
            added_ids = new_partner_ids_set - old_partner_ids
            if added_ids:
                present_ids = set(room.present_character_ids or [])
                present_ids.update(added_ids)
                room.present_character_ids = list(present_ids)

        # Remove any removed members from present_character_ids
        removed_ids = old_partner_ids - new_partner_ids_set
        if removed_ids and room.present_character_ids:
            room.present_character_ids = [pid for pid in room.present_character_ids if pid not in removed_ids]

        room.partner_ids = new_partner_ids

    data_store.save()

    return jsonify({
        'id': room.id,
        'name': room.name,
        'scenario': room.scenario,
        'system_prompt': room.system_prompt,
        'partner_ids': room.partner_ids,
    })


def _clone_messages_for_room(messages, new_room_id: str, speaker_remap: dict = None) -> list[Message]:
    """Copy transcript messages into another room, optionally remapping speakers."""
    import copy

    speaker_remap = speaker_remap or {}
    cloned_messages = []
    for msg in messages:
        mapped_speaker = speaker_remap.get(msg.speaker_id, {})
        mentions = copy.deepcopy(getattr(msg, 'mentions', []))
        if mentions and speaker_remap:
            mentions = [
                speaker_remap.get(mention_id, {}).get('speaker_id', mention_id)
                for mention_id in mentions
            ]

        cloned_messages.append(Message(
            id=str(uuid.uuid4())[:8],
            speaker_id=mapped_speaker.get('speaker_id', msg.speaker_id),
            speaker_name=mapped_speaker.get('speaker_name', msg.speaker_name),
            content=msg.content,
            room_id=new_room_id,
            message_type=getattr(msg, 'message_type', 'message'),
            fudged=getattr(msg, 'fudged', False),
            mentions=mentions,
            image_path=getattr(msg, 'image_path', None),
            spatial_map=getattr(msg, 'spatial_map', None),
            metadata=copy.deepcopy(getattr(msg, 'metadata', None)),
        ))
    return cloned_messages


@app.route('/rooms/<room_id>/clone', methods=['POST'])
def clone_room(room_id):
    """Clone a room with all its settings and messages."""
    import copy

    try:
        room = data_store.get_room(room_id)
        if not room:
            return jsonify({'error': 'Room not found'}), 404

        # Don't allow cloning the common room
        if room.is_common_room:
            return jsonify({'error': 'Cannot clone the common room'}), 400

        # Generate new ID
        new_id = f"custom_{str(uuid.uuid4())[:8]}"

        # Deep copy messages with new IDs
        new_messages = _clone_messages_for_room(room.messages, new_id)

        # Create the cloned room - copy all story/transcript state
        new_room = Room(
            id=new_id,
            name=room.name,
            is_common_room=False,
            partner_id=room.partner_id,
            partner_ids=list(room.partner_ids) if room.partner_ids else [],
            scenario=room.scenario,
            timeline=room.timeline,
            system_prompt=room.system_prompt,
            messages=new_messages,
            background_image=None,  # Clones start fresh - they can set their own background
            genre=room.genre,
            factions=room.factions,
            scene_layout=room.scene_layout,
            genre_rules=copy.deepcopy(room.genre_rules),
            dm_mode=room.dm_mode,
            room_model=room.room_model,
            dm_secret=room.dm_secret,
            hardcore_mode=room.hardcore_mode,
            # Character relationships and presence
            character_relationships=copy.deepcopy(room.character_relationships),
            present_character_ids=list(room.present_character_ids) if room.present_character_ids else [],
            relationship_interactions=copy.deepcopy(room.relationship_interactions),
            # World state
            threat_type=room.threat_type,
            world_state=room.world_state,
            zombie_type=room.zombie_type,
            population_density=room.population_density,
            initial_zombie_count=room.initial_zombie_count,
            zombies_killed=room.zombies_killed,
            areas_cleared=list(room.areas_cleared) if room.areas_cleared else [],
            zombie_rules=copy.deepcopy(room.zombie_rules),
            shelter_type=room.shelter_type,
            # Player character info
            player_character_name=room.player_character_name,
            player_gender=room.player_gender,
            player_alignment=room.player_alignment,
            player_role=room.player_role,
            player_backstory=copy.deepcopy(room.player_backstory),
            player_location=room.player_location,
            character_locations=copy.deepcopy(room.character_locations),
            # Settings
            auto_generate=room.auto_generate,
            auto_generate_mode=room.auto_generate_mode,
            auto_generate_count=room.auto_generate_count,
            auto_respond=room.auto_respond,
            loras=copy.deepcopy(room.loras) if room.loras else [],
        )

        # Save the new room
        data_store._rooms[new_id] = new_room
        data_store.save()

        return jsonify({
            'id': new_room.id,
            'name': new_room.name,
            'message_count': len(new_messages),
            'status': 'cloned'
        })
    except Exception as e:
        print(f"[clone_room] Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/rooms/<room_id>/auto-generate', methods=['POST'])
def update_room_auto_generate(room_id):
    """Update a room's auto-generate settings."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    data = request.json

    if 'auto_generate' in data:
        room.auto_generate = bool(data['auto_generate'])
    if 'auto_generate_mode' in data:
        mode = data['auto_generate_mode']
        if mode in ('scene', 'selfie', 'both', 'group'):
            room.auto_generate_mode = mode
    if 'auto_generate_count' in data:
        count = int(data['auto_generate_count'])
        room.auto_generate_count = max(1, min(count, 5))  # Clamp 1-5

    data_store.save()

    return jsonify({
        'auto_generate': room.auto_generate,
        'auto_generate_mode': room.auto_generate_mode,
        'auto_generate_count': room.auto_generate_count,
    })


@app.route('/rooms/<room_id>/auto-respond', methods=['POST'])
def update_room_auto_respond(room_id):
    """Toggle auto-respond setting for a room."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    data = request.json or {}
    if 'auto_respond' in data:
        room.auto_respond = bool(data['auto_respond'])
        data_store.save()

    return jsonify({'auto_respond': room.auto_respond})


@app.route('/rooms/<room_id>/regenerate', methods=['POST'])
def regenerate_last_response(room_id):
    """
    Regenerate the last AI response in a room.
    Removes the last AI message and generates a new one from the same partner.
    Only works for non-StoryBuilder rooms (where character_relationships is empty).
    """
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    # Block regeneration in StoryBuilder rooms
    if room.character_relationships:
        return jsonify({'error': 'Regeneration not allowed in StoryBuilder rooms'}), 400

    # Find the last AI message
    if not room.messages:
        return jsonify({'error': 'No messages to regenerate'}), 400

    last_msg = room.messages[-1]
    if last_msg.speaker_id == 'user':
        return jsonify({'error': 'Last message is from user, not AI'}), 400

    # Get the partner who sent this message
    partner = data_store.get_partner(last_msg.speaker_id)
    if not partner:
        return jsonify({'error': 'Partner not found'}), 404

    # Remove the last message
    room.messages.pop()
    data_store.save()

    # Build conversation history (same as /chat endpoint)
    messages = []
    all_partners = data_store.get_partners()
    is_multi_party = room.is_common_room or bool(room.partner_ids)

    for msg in room.messages:
        role = "user" if msg.speaker_id == "user" else "assistant"
        if is_multi_party:
            if msg.speaker_id == "user":
                content = f"{{{msg.speaker_name}}}: {msg.content}"
            else:
                content = f"{msg.speaker_name}: {msg.content}"
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": role, "content": msg.content})

    # Build system prompt
    room_has_override = room.system_prompt and room.system_prompt.strip()
    if room_has_override:
        base_system_prompt = room.system_prompt.strip()
    else:
        base_system_prompt = partner.get_effective_system_prompt(settings.global_system_prompt)

    if room.is_common_room:
        system = partner.get_full_context(all_partners, settings.user_name, base_system_prompt, user_physical_description=settings.user_physical_description)
        system += f"\n\nYou ARE {partner.name}. Respond in first person as {partner.name}. Do NOT prefix your response with any name. Do NOT speak as or for other characters - only as yourself.\n\nThis is a turn-based collaborative story. Each character is played by a different player. When you reach a moment where another character would naturally speak or act, that's your cue to pause - they WILL respond. You don't need to carry the narrative alone."
    elif room.partner_ids:
        room_partners = room.get_partners_in_room(all_partners)
        system = partner.get_full_context(room_partners, settings.user_name, base_system_prompt, user_physical_description=settings.user_physical_description)
        if room.scenario and len(room.messages) < 10:
            # Strip character names to prevent knowledge leaks between strangers
            clean_scenario = strip_characters_from_scenario(room.scenario)
            system += f"\n\n---\nSCENARIO:\n{clean_scenario}\n---"
        system += f"\n\nYou ARE {partner.name}. Respond in first person as {partner.name}. Do NOT prefix your response with any name. Do NOT speak as or for other characters - only as yourself.\n\nThis is a turn-based collaborative story. Each character is played by a different player. When you reach a moment where another character would naturally speak or act, that's your cue to pause - they WILL respond. You don't need to carry the narrative alone."
    else:
        system = f"{base_system_prompt}\n\n---\n{partner.get_character()}\n---"

    # Generate new response
    try:
        response_text = run_async(generate_response_async(partner, messages, system))
        response_text = clean_model_tokens(response_text.strip())

        if not response_text or response_text.startswith('[API Error:') or response_text.startswith('[Error:'):
            return jsonify({'error': response_text or 'Empty response from AI'}), 500

        # Save the new response
        response_message = Message(
            id=str(uuid.uuid4())[:8],
            speaker_id=partner.id,
            speaker_name=partner.name,
            content=response_text,
            room_id=room_id,
        )
        data_store.add_message(room_id, response_message)

        return jsonify({
            'content': response_text,
            'speaker_id': partner.id,
            'speaker_name': partner.name,
            'avatar': partner.avatar,
            'avatar_image': partner.avatar_image,
        })

    except Exception as e:
        print(f"[Regenerate] Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/rooms/<room_id>/ambient', methods=['GET'])
def get_ambient_settings(room_id):
    """Get ambient mode settings for a room."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    return jsonify({
        'ambient_mode': room.ambient_mode,
        'ambient_interval_min': room.ambient_interval_min,
        'ambient_interval_max': room.ambient_interval_max,
        'ambient_providers': room.ambient_providers,
    })


@app.route('/rooms/<room_id>/ambient', methods=['POST'])
def update_ambient_settings(room_id):
    """Update ambient mode settings for a room."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    data = request.json

    if 'ambient_mode' in data:
        room.ambient_mode = bool(data['ambient_mode'])
    if 'ambient_interval_min' in data:
        room.ambient_interval_min = max(1, min(60, int(data['ambient_interval_min'])))
    if 'ambient_interval_max' in data:
        room.ambient_interval_max = max(1, min(120, int(data['ambient_interval_max'])))
    if 'ambient_providers' in data:
        valid = ['ollama', 'anthropic', 'openai']
        room.ambient_providers = [p for p in data['ambient_providers'] if p in valid]

    # Ensure min <= max
    if room.ambient_interval_min > room.ambient_interval_max:
        room.ambient_interval_min = room.ambient_interval_max

    data_store.save()

    return jsonify({
        'ambient_mode': room.ambient_mode,
        'ambient_interval_min': room.ambient_interval_min,
        'ambient_interval_max': room.ambient_interval_max,
        'ambient_providers': room.ambient_providers,
    })


@app.route('/rooms/<room_id>/ambient/pull', methods=['POST'])
def ambient_pull(room_id):
    """Pull a random character into the common room conversation."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    all_partners = data_store.get_partners()

    # Get partners for this room
    if room.is_common_room:
        room_partners = all_partners
    elif room.partner_ids:
        room_partners = [p for p in all_partners if p.id in room.partner_ids]
    else:  # 1:1 private room
        room_partners = [p for p in all_partners if p.id == room.partner_id]

    # Filter partners by allowed providers
    allowed_providers = room.ambient_providers or ['ollama']
    eligible_partners = [
        p for p in room_partners
        if p.provider.lower() in allowed_providers
    ]

    if not eligible_partners:
        return jsonify({'error': 'No eligible partners (check provider settings)'}), 400

    # Pick a random partner, avoiding the last speaker if possible
    import random
    last_speaker_id = room.messages[-1].speaker_id if room.messages else None

    # Prefer someone who didn't just speak
    non_repeat = [p for p in eligible_partners if p.id != last_speaker_id]
    partner = random.choice(non_repeat if non_repeat else eligible_partners)

    # Build conversation context
    messages = []
    is_multi_party = True

    for msg in room.messages[-20:]:  # Last 20 messages
        if msg.speaker_id == "user":
            content = f"{{{msg.speaker_name}}}: {msg.content}"
        else:
            content = f"{msg.speaker_name}: {msg.content}"
        messages.append({"role": "user", "content": content})

    # Conversation starters for when conversation is stale or empty
    starters = [
        "glances around the room and shares what's on your mind",
        "notices something interesting and comments on it",
        "brings up something you've been thinking about lately",
        "joins the conversation naturally, adding your perspective",
        "asks the room an interesting question that's been on your mind",
    ]

    # Build system prompt with same hierarchy as /respond
    # Hierarchy: Room system_prompt > Character custom_system_prompt > Global
    room_has_override = room.system_prompt and room.system_prompt.strip()
    if room_has_override:
        base_system_prompt = room.system_prompt.strip()
    else:
        base_system_prompt = partner.get_effective_system_prompt(settings.global_system_prompt)

    # Use simplified 1:1 format or full multi-party context
    is_one_on_one = room.partner_id and not room.partner_ids
    if is_one_on_one:
        # 1:1 private room - use simple format like /respond does
        system = f"{base_system_prompt}\n\n---\n{partner.get_character()}\n---"
    else:
        # Multi-party room - use full context
        system = partner.get_full_context(
            all_partners if room.is_common_room else room.get_partners_in_room(all_partners),
            settings.user_name,
            base_system_prompt,
            user_physical_description=settings.user_physical_description
        )

    # Add ambient instruction
    starter = random.choice(starters)
    room_name = room.name if not room.is_common_room else "common room"
    if len(room.messages) < 3:
        # Room is quiet - encourage starting something
        system += f"\n\nThe {room_name} is quiet. You walk in and {starter}. Be natural and in-character. Start a conversation or make an observation."
    else:
        # Conversation happening - join naturally
        system += f"\n\nYou're in the {room_name}. {starter.capitalize()}. Respond naturally to what's happening or take the conversation in an interesting direction."

    system += f"\n\nRespond as {partner.name}. Do not prefix your response with your name."

    try:
        response_text = run_async(generate_response_async(partner, messages, system))
        response_text = clean_model_tokens(response_text.strip())

        if not response_text or response_text.startswith('['):
            return jsonify({'error': 'Failed to generate response'}), 500

        # Save response
        response_message = Message(
            id=str(uuid.uuid4())[:8],
            speaker_id=partner.id,
            speaker_name=partner.name,
            content=response_text,
            room_id=room_id,
        )
        data_store.add_message(room_id, response_message)

        return jsonify({
            'response': {
                'id': response_message.id,
                'speaker_id': partner.id,
                'speaker_name': partner.name,
                'avatar': partner.avatar,
                'avatar_image': partner.avatar_image,
                'content': response_text,
            },
            'partner_name': partner.name,
        })

    except Exception as e:
        print(f"[ambient_pull] Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/rooms/<room_id>/clear', methods=['POST'])
def clear_room(room_id):
    """Clear messages in a room."""
    data_store.clear_room(room_id)
    return jsonify({'status': 'cleared'})


@app.route('/rooms/<room_id>/npcs', methods=['GET'])
def get_room_npcs_endpoint(room_id):
    """Get all NPCs currently present in a room."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    npcs = get_room_npcs(room_id)
    return jsonify({
        'npcs': [
            {
                'id': npc.id,
                'name': npc.name,
                'role': npc.current_role,
                'personality': npc.personality,
                'state': npc.state.value,
                'total_interactions': npc.total_interactions,
                'interaction_weight': npc.interaction_weight,
            }
            for npc in npcs
        ]
    })


@app.route('/rooms/<room_id>/travel', methods=['POST'])
def travel_to_location(room_id):
    """
    Travel to a location to meet an NPC.

    The journey itself is gameplay:
    - DM narrates the walk
    - 15% chance of encounter on the way (but not if last travel had one)
    - 80% chance NPC is there, 20% they're not
    - If not there, 50% chance of a hint where they went
    """
    import random

    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    data = request.json or {}
    destination = data.get('destination', '')
    npc_id = data.get('npc_id', '')
    npc_name = data.get('npc_name', '')

    if not destination:
        return jsonify({'error': 'No destination specified'}), 400

    # Get current player location for context
    player_location = getattr(room, 'player_location', None) or "your current location"
    player_name = room.player_character_name or settings.user_name

    # Build context for DM
    scenario = room.scenario or "the area"
    genre = room.genre or ""
    threat_type = getattr(room, 'threat_type', '') or ""

    # Check if last travel had an encounter (skip encounter on immediate retry)
    last_travel_encounter = getattr(room, '_last_travel_encounter', False)

    # Roll for encounter during travel (15% chance, but not if retrying after encounter)
    if last_travel_encounter:
        has_encounter = False  # Path is clear now
        room._last_travel_encounter = False
    else:
        encounter_roll = random.random()
        has_encounter = encounter_roll < 0.15

    # Check companions - do they want to come along?
    companions_coming = []
    companions_staying = []
    companion_context = ""

    present_ids = room.present_character_ids or []
    if present_ids:
        all_partners = data_store.get_partners()
        present_partners = [p for p in all_partners if p.id in present_ids]

        # Get recent conversation context for companion decision
        recent_messages = room.messages[-5:] if room.messages else []
        recent_context = "\n".join([f"{m.speaker_name}: {m.content[:100]}" for m in recent_messages])

        for partner in present_partners:
            # Ask companion if they want to come
            consent_prompt = f"""You are {partner.name}.
Personality: {partner.character_description or 'a companion'}

{player_name} is about to travel to {destination}.

Recent conversation:
{recent_context}

Based on your personality and the recent conversation, do you want to go with them?
Reply with ONLY one of:
- "yes" (you'll go with them)
- "no, [location]" (you'll stay/go elsewhere - say where to find you)
- "no" (you're staying, prefer not to say where)

Your brief answer:"""

            try:
                consent_response = _call_ollama_sync(
                    settings.default_ollama_model,  # Use faster model for quick decisions
                    consent_prompt,
                    f"You are {partner.name}. Answer very briefly."
                )
                consent_response = consent_response.strip().lower()

                if consent_response.startswith("yes"):
                    companions_coming.append(partner.name)
                else:
                    # Parse where they'll be
                    if "," in consent_response:
                        location_hint = consent_response.split(",", 1)[1].strip()
                        companions_staying.append((partner.name, location_hint))
                    else:
                        companions_staying.append((partner.name, None))

                    # Remove from present_character_ids
                    if partner.id in room.present_character_ids:
                        room.present_character_ids.remove(partner.id)

            except Exception as e:
                print(f"[Travel] Error getting companion consent from {partner.name}: {e}")
                # Default to coming along
                companions_coming.append(partner.name)

        # Build companion context for journey narration
        if companions_coming:
            companion_context = f"\nCOMPANIONS TRAVELING WITH YOU: {', '.join(companions_coming)}"
        if companions_staying:
            staying_info = []
            for name, loc in companions_staying:
                if loc:
                    staying_info.append(f"{name} (staying - find them at: {loc})")
                else:
                    staying_info.append(f"{name} (staying - didn't say where)")
            companion_context += f"\nCOMPANIONS NOT COMING: {', '.join(staying_info)}"

    # Generate journey narration
    if has_encounter:
        journey_prompt = f"""The player is traveling from {player_location} to {destination}.

SETTING: {scenario}
GENRE: {genre}
THREAT: {threat_type}
{companion_context}

Something happens on the way - an encounter, obstacle, or moment of tension.
This could be:
- A threat appropriate to the setting (zombie, hostile stranger, etc.)
- Another NPC crossing their path
- An obstacle or complication
- A discovery along the way

Write 2-3 sentences describing:
1. The journey beginning (mention companions if any are with you)
2. The encounter/interruption
3. End at a moment of tension - what do they do?

If companions stayed behind, you may briefly acknowledge their parting.
Use "you" for the player. Present tense. Vivid but brief."""
    else:
        journey_prompt = f"""The player is traveling from {player_location} to {destination}.

SETTING: {scenario}
GENRE: {genre}
{companion_context}

Write 2-3 SHORT sentences describing the journey. Atmospheric but uneventful.
Include sensory details - what do they see, hear, smell on the way?
If companions are traveling with you, include them naturally in the scene.
If companions stayed behind, you may briefly acknowledge their parting.
End with them arriving at {destination}.

Use "you" for the player. Present tense. Brief."""

    journey_system = "You are the DM narrator. Write immersive but concise travel descriptions."

    try:
        journey_narration = _call_ollama_sync(
            room.room_model or settings.storybuilder_model,
            journey_prompt,
            journey_system
        )
    except Exception as e:
        print(f"[Travel] Error generating journey: {e}")
        journey_narration = f"You make your way to {destination}."

    # If encounter happened, don't complete the journey yet
    if has_encounter:
        # Update player location to "en route"
        room.player_location = f"en route to {destination}"
        room._last_travel_encounter = True  # Track so retry is safe
        data_store.save()

        return jsonify({
            'journey_narration': journey_narration.strip(),
            'encounter': True,
            'arrived': False,
            'destination': destination,
            'message': 'Something happened on the way...'
        })

    # No encounter - check if NPC is there (80% chance)
    npc_there_roll = random.random()
    npc_is_there = npc_there_roll < 0.80

    # Update player location
    room.player_location = destination
    room._last_travel_encounter = False  # Clear encounter flag
    data_store.save()

    if npc_is_there:
        # Mark ALL NPCs at this location as accessible (not just the clicked one)
        room_npcs = get_room_npcs(room_id)
        npcs_found = []
        for npc in room_npcs:
            npc_loc = (npc.current_location or "").lower()
            if npc_loc == destination.lower() or npc.id == npc_id.replace('npc_', ''):
                npc.current_location = "here"  # Now accessible
                npcs_found.append(npc.name)
        _save_npcs()

        if len(npcs_found) > 1:
            arrival_text = f"\n\nYou find {', '.join(npcs_found[:-1])} and {npcs_found[-1]} here."
        elif npcs_found:
            arrival_text = f"\n\nYou find {npcs_found[0]} there."
        else:
            arrival_text = ""

        return jsonify({
            'journey_narration': journey_narration.strip() + arrival_text,
            'encounter': False,
            'arrived': True,
            'destination': destination,
            'npc_present': True,
            'npcs_found': npcs_found,
            'message': f'You arrived at {destination}.'
        })
    else:
        # NPC is NOT there (20% chance)
        # 50% chance of a hint
        has_hint = random.random() < 0.50

        if has_hint:
            # Generate a hint about where they might be
            hint_options = [
                "A handwritten note on the counter says they'll be back soon.",
                "A neighbor mentions they saw them heading toward the market.",
                "Someone calls out that they just missed them - try the tavern.",
                "A child playing nearby says they went to help a friend.",
                "Fresh footprints in the dust lead toward the back door.",
            ]
            hint = random.choice(hint_options)
            hint_text = f"\n\n{npc_name} isn't here. {hint}"
        else:
            # No hint - just empty
            empty_options = [
                f"The place is empty. No sign of {npc_name}.",
                f"Nobody's here. {npc_name} could be anywhere.",
                f"The door's unlocked but {npc_name} is gone. No note, nothing.",
                f"Empty. You don't know if they left five minutes ago or five hours ago.",
            ]
            hint_text = f"\n\n{random.choice(empty_options)}"

        return jsonify({
            'journey_narration': journey_narration.strip() + hint_text,
            'encounter': False,
            'arrived': True,
            'destination': destination,
            'npc_present': False,
            'had_hint': has_hint,
            'message': f'{npc_name} wasn\'t there.'
        })


@app.route('/rooms/<room_id>/pin', methods=['POST'])
def toggle_room_pin(room_id):
    """Toggle pinned status for a room."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    data = request.json
    room.pinned = data.get('pinned', not room.pinned)
    data_store.save()

    return jsonify({'status': 'ok', 'pinned': room.pinned})


@app.route('/rooms/<room_id>/archive', methods=['POST'])
def toggle_room_archive(room_id):
    """Toggle archived status for a room. The Common Room cannot be archived."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    if room.is_common_room:
        return jsonify({'error': 'The Common Room cannot be archived'}), 400

    data = request.json or {}
    room.archived = data.get('archived', not room.archived)
    data_store.save()

    return jsonify({'status': 'ok', 'archived': room.archived})


@app.route('/rooms/<room_id>/activity', methods=['POST'])
def bump_room_activity(room_id):
    """Record a room as recently active so its list position syncs across devices."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    room.last_active = datetime.now().timestamp() * 1000  # epoch ms, matches JS Date.now()
    data_store.save()

    return jsonify({'status': 'ok', 'last_active': room.last_active})


@app.route('/prompt-book', methods=['GET'])
def get_prompt_book():
    """Return all prompt book entries (newest first)."""
    entries = _load_prompt_book()
    entries.sort(key=lambda e: e.get('created_at', ''), reverse=True)
    return jsonify(entries)


@app.route('/prompt-book', methods=['POST'])
def add_prompt_book_entry():
    """Manually add a prompt to the book. Deduped by prompt text."""
    data = request.json or {}
    added = _add_to_prompt_book(data.get('name', 'Untitled'), data.get('prompt', ''))
    return jsonify({'status': 'ok', 'added': added})


@app.route('/prompt-book/<entry_id>', methods=['DELETE'])
def delete_prompt_book_entry(entry_id):
    """Prune a single prompt book entry."""
    with _prompt_book_lock:
        entries = _load_prompt_book()
        new_entries = [e for e in entries if e.get('id') != entry_id]
        _save_prompt_book(new_entries)
    return jsonify({'status': 'ok', 'removed': len(entries) - len(new_entries)})


@app.route('/rooms/<room_id>/import', methods=['POST'])
def import_chat(room_id):
    """Import chat history - supports both Roundtable export and legacy formats."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    data = request.json
    # Support both formats: 'messages' (Roundtable export) and 'history' (legacy)
    messages = data.get('messages', [])
    history = data.get('history', [])
    partner_id = data.get('partner_id')

    if not messages and not history:
        return jsonify({'error': 'No messages to import'}), 400

    partner = data_store.get_partner(partner_id) if partner_id else None

    imported = 0

    # Handle full messages format (from Roundtable export)
    if messages:
        for msg in messages:
            content = msg.get('content', '')
            if not content:
                continue

            message = Message(
                id=str(uuid.uuid4())[:8],
                speaker_id=msg.get('speaker_id', 'user'),
                speaker_name=msg.get('speaker_name', 'Unknown'),
                content=content,
                room_id=room_id,
                message_type=msg.get('message_type', 'message')
            )
            data_store.add_message(room_id, message)
            imported += 1
    else:
        # Handle legacy history format (role/content pairs)
        for msg in history:
            role = msg.get('role')
            content = msg.get('content', '')

            if not content:
                continue

            if role == 'user':
                message = Message(
                    id=str(uuid.uuid4())[:8],
                    speaker_id='user',
                    speaker_name=settings.user_name or 'You',
                    content=content,
                    room_id=room_id,
                )
            elif role == 'assistant':
                message = Message(
                    id=str(uuid.uuid4())[:8],
                    speaker_id=partner.id if partner else 'assistant',
                    speaker_name=partner.name if partner else 'Claude',
                    content=content,
                    room_id=room_id,
                )
            data_store.add_message(room_id, message)
            imported += 1

    return jsonify({'status': 'ok', 'imported': imported})


@app.route('/rooms/<room_id>/recent-images', methods=['GET'])
def get_recent_images(room_id):
    """Get recent images for a room (last 20).

    For private rooms: partner's selfies + room's scenes
    For common room: all images from all partners and all scenes
    """
    from pathlib import Path

    base_dir = Path.home() / ".roundtable"
    avatars_dir = base_dir / "avatars"
    scenes_dir = base_dir / "scenes"

    images = []
    limit = 100  # Show more images since we prune regularly

    room = data_store.get_room(room_id)

    # Files to exclude (cropped versions, not originals) - case insensitive for Windows
    excluded_files = {'avatar.png', 'background.png'}

    def is_excluded(filename):
        return filename.lower() in excluded_files

    def get_prompt(img_path):
        """Read the prompt from the companion .txt file if it exists."""
        txt_path = Path(str(img_path).replace('.png', '.txt'))
        if txt_path.exists():
            try:
                return txt_path.read_text(encoding='utf-8').strip()
            except:
                pass
        return None

    def get_loras_from_filename(filename):
        """Extract LoRA names from filename (format: _lora-name1-name2_)."""
        import re
        match = re.search(r'_lora-([^_]+)', filename)
        if match:
            # Split on '-' but handle multi-word lora names
            lora_str = match.group(1)
            # Return as list (could be multiple loras joined by -)
            return lora_str.split('-') if lora_str else []
        return []

    def image_matches_room(filename, target_room_id):
        """Check if image filename contains the room ID (sanitized same way as when saving)."""
        import re
        sanitized = re.sub(r'[^\w\-]', '', target_room_id.replace(' ', '_'))[:30]
        return f"_room_{sanitized}" in filename

    if room and room.is_common_room:
        # Common room: get ALL recent images
        # Scan all partner folders
        if avatars_dir.exists():
            for partner_dir in avatars_dir.iterdir():
                if partner_dir.is_dir():
                    partner_id = partner_dir.name
                    # Handle 'user' folder specially (player character selfies)
                    if partner_id == 'user':
                        partner_name = settings.user_name or 'Player'
                    else:
                        partner = data_store.get_partner(partner_id)
                        partner_name = partner.name if partner else partner_id
                    for img in partner_dir.glob("*.png"):
                        if not is_excluded(img.name):
                            images.append({
                                'path': str(img),
                                'type': 'selfie',
                                'partner_id': partner_id,
                                'partner_name': partner_name,
                                'mtime': img.stat().st_mtime,
                                'prompt': get_prompt(img),
                                'loras': get_loras_from_filename(img.name)
                            })

        # Get all scenes
        if scenes_dir.exists():
            for img in scenes_dir.glob("*.png"):
                images.append({
                    'path': str(img),
                    'type': 'scene',
                    'mtime': img.stat().st_mtime,
                    'prompt': get_prompt(img),
                    'loras': get_loras_from_filename(img.name)
                })

    elif room and room.partner_id:
        # Private room: get ALL of partner's selfies (from any room) + scenes for this room
        # This way the private room acts as the character's personal gallery
        partner_id = room.partner_id
        partner = data_store.get_partner(partner_id)
        partner_dir = avatars_dir / partner_id

        if partner_dir.exists():
            for img in partner_dir.glob("*.png"):
                if not is_excluded(img.name):
                    images.append({
                        'path': str(img),
                        'type': 'selfie',
                        'partner_id': partner_id,
                        'partner_name': partner.name if partner else partner_id,
                        'mtime': img.stat().st_mtime,
                        'prompt': get_prompt(img),
                        'loras': get_loras_from_filename(img.name)
                    })

        # Get scenes for this room only (scenes are room-specific)
        if scenes_dir.exists():
            for img in scenes_dir.glob(f"scene_{room_id}_*.png"):
                images.append({
                    'path': str(img),
                    'type': 'scene',
                    'mtime': img.stat().st_mtime,
                    'prompt': get_prompt(img),
                    'loras': get_loras_from_filename(img.name)
                })

    elif room and room.partner_ids:
        # Custom room: get images for selected partners + room scenes
        for partner_id in room.partner_ids:
            partner = data_store.get_partner(partner_id)
            partner_dir = avatars_dir / partner_id

            if partner_dir.exists():
                for img in partner_dir.glob("*.png"):
                    if not is_excluded(img.name) and image_matches_room(img.name, room_id):
                        images.append({
                            'path': str(img),
                            'type': 'selfie',
                            'partner_id': partner_id,
                            'partner_name': partner.name if partner else partner_id,
                            'mtime': img.stat().st_mtime,
                            'prompt': get_prompt(img),
                            'loras': get_loras_from_filename(img.name)
                        })

        # Include user selfies for this room (StoryBuilder player character selfies)
        user_dir = avatars_dir / 'user'
        if user_dir.exists():
            for img in user_dir.glob("*.png"):
                if not is_excluded(img.name) and image_matches_room(img.name, room_id):
                    images.append({
                        'path': str(img),
                        'type': 'selfie',
                        'partner_id': 'user',
                        'partner_name': room.player_character_name or settings.user_name or 'Player',
                        'mtime': img.stat().st_mtime,
                        'prompt': get_prompt(img),
                        'loras': get_loras_from_filename(img.name)
                    })

        # Get scenes for this room
        if scenes_dir.exists():
            for img in scenes_dir.glob(f"scene_{room_id}_*.png"):
                images.append({
                    'path': str(img),
                    'type': 'scene',
                    'mtime': img.stat().st_mtime,
                    'prompt': get_prompt(img),
                    'loras': get_loras_from_filename(img.name)
                })

    # Sort by modification time (newest first)
    images.sort(key=lambda x: x['mtime'], reverse=True)

    # Cap scenes to 10 to avoid overwhelming the sidebar, but keep more selfies
    scenes = [img for img in images if img['type'] == 'scene'][:10]
    selfies = [img for img in images if img['type'] != 'scene'][:limit]

    # Combine and re-sort
    images = scenes + selfies
    images.sort(key=lambda x: x['mtime'], reverse=True)

    # Get partner_id for the current room (for shared indicator)
    current_partner_id = room.partner_id if room else None

    # Remove mtime and add shared/favorite info
    for img in images:
        del img['mtime']
        # Add shared_with list (which partners this was shared with)
        img['shared_with'] = _get_shared_partners(img['path'])
        # For convenience in private rooms, include a "shared" boolean
        if current_partner_id:
            img['shared'] = current_partner_id in img['shared_with']
        # Add favorite status (per-room)
        img['favorited'] = _is_favorite(img['path'], room_id)

    # Images stay in their original chronological order (sorted by mtime already)
    return jsonify(images)


@app.route('/partners', methods=['GET'])
def get_partners():
    """Get all partners."""
    partners = data_store.get_partners()
    return jsonify([{
        'id': p.id,
        'name': p.name,
        'avatar': p.avatar,
        'avatar_image': p.avatar_image,
        'background_image': p.background_image,
        'color': p.color,
        'gender': p.gender,
        'provider': p.provider,
        'model': p.model,
        'temperature': normalize_temperature(getattr(p, 'temperature', 0.7)),
        'character_description': p.character_description,
        'physical_description': p.physical_description,
        'custom_system_prompt': p.custom_system_prompt,
        'memory_mode': p.memory_mode,
        'voice': p.voice,
        'loras': p.loras,
    } for p in partners])


@app.route('/partners', methods=['POST'])
def create_partner():
    """Create a new partner."""
    data = request.json

    partner = Partner(
        id=str(uuid.uuid4())[:8],
        name=data.get('name', 'New Partner'),
        character_description=data.get('character_description', ''),
        physical_description=data.get('physical_description', ''),
        gender=data.get('gender', ''),
        color=data.get('color', '#ff69b4'),
        provider=data.get('provider', 'ollama'),
        model=data.get('model', settings.default_ollama_model),
        temperature=normalize_temperature(data.get('temperature', 0.7)),
        avatar=data.get('avatar', '🤖'),
        custom_system_prompt=data.get('custom_system_prompt'),
        memory_mode=data.get('memory_mode', 'local'),
        voice=data.get('voice', 'none'),
        loras=data.get('loras', []),
    )

    data_store.add_partner(partner)

    # Auto-capture custom system prompt into the prompt book (named after persona)
    if partner.custom_system_prompt:
        _add_to_prompt_book(partner.name, partner.custom_system_prompt)

    return jsonify({
        'id': partner.id,
        'name': partner.name,
        'avatar': partner.avatar,
        'provider': partner.provider,
        'model': partner.model,
        'temperature': partner.temperature,
    })


@app.route('/partners/<partner_id>', methods=['PUT'])
def update_partner(partner_id):
    """Update a partner."""
    data = request.json
    partner = data_store.get_partner(partner_id)

    if not partner:
        return jsonify({'error': 'Partner not found'}), 404

    # Update fields
    partner.name = data.get('name', partner.name)
    partner.character_description = data.get('character_description', partner.character_description)
    partner.physical_description = data.get('physical_description', partner.physical_description)
    partner.gender = data.get('gender', partner.gender)
    partner.color = data.get('color', partner.color)
    partner.provider = data.get('provider', partner.provider)
    partner.model = data.get('model', partner.model)
    if 'temperature' in data:
        partner.temperature = normalize_temperature(data.get('temperature'), getattr(partner, 'temperature', 0.7))
    partner.avatar = data.get('avatar', partner.avatar)
    partner.custom_system_prompt = data.get('custom_system_prompt', partner.custom_system_prompt)
    partner.memory_mode = data.get('memory_mode', partner.memory_mode)
    partner.voice = data.get('voice', partner.voice)
    if 'loras' in data:
        partner.loras = data['loras']

    data_store.update_partner(partner)

    # Auto-capture custom system prompt into the prompt book (named after persona)
    if partner.custom_system_prompt:
        _add_to_prompt_book(partner.name, partner.custom_system_prompt)

    return jsonify({'status': 'updated'})


@app.route('/partners/<partner_id>/profile', methods=['GET'])
def get_partner_profile(partner_id):
    """Get a character's full profile."""
    partner = data_store.get_partner(partner_id)
    if not partner:
        return jsonify({'error': 'Partner not found'}), 404

    is_dm = request.args.get('dm', 'false').lower() == 'true'

    # Base profile info
    profile = {
        'id': partner.id,
        'name': partner.name,
        'physical_description': partner.physical_description,
        'avatar': partner.avatar,
        'avatar_image': partner.avatar_image,
        'color': partner.color,
        'temperature': normalize_temperature(getattr(partner, 'temperature', 0.7)),
    }

    # Character info (personality, background)
    profile['character'] = {
        'description': partner.character_description,
        'skill': partner.skill,
    }

    # Hidden traits (DM only)
    if is_dm:
        profile['hidden_traits'] = {
            'secret': partner.secret,
            'wound': partner.wound,
            'want': partner.want,
            'fear': partner.fear,
            'honesty': partner.honesty,
        }

    return jsonify({'profile': profile})


@app.route('/partners/<partner_id>', methods=['DELETE'])
def delete_partner(partner_id):
    """Delete a partner."""
    data_store.delete_partner(partner_id)
    return jsonify({'status': 'deleted'})


@app.route('/partners/<partner_id>/clone', methods=['POST'])
def clone_partner(partner_id):
    """Clone a partner with all settings."""
    try:
        import copy

        partner = data_store.get_partner(partner_id)
        if not partner:
            return jsonify({'error': 'Partner not found'}), 404

        # Generate new ID
        new_id = str(uuid.uuid4())[:8]

        # Create cloned partner
        new_partner = Partner(
            id=new_id,
            name=partner.name,
            character_description=partner.character_description,
            physical_description=partner.physical_description,
            gender=partner.gender,
            provider=partner.provider,
            model=partner.model,
            temperature=normalize_temperature(getattr(partner, 'temperature', 0.7)),
            avatar=partner.avatar,
            avatar_image=None,  # Don't copy the image path, they should generate new
            background_image=None,
            color=partner.color,
            loras=list(partner.loras) if partner.loras else [],
            custom_system_prompt=partner.custom_system_prompt,
            memory_mode=partner.memory_mode,
            secret=partner.secret,
            wound=partner.wound,
            want=partner.want,
            fear=partner.fear,
            skill=partner.skill,
            honesty=partner.honesty,
            voice=partner.voice,
        )

        # Save the new partner (add_partner also creates their private room)
        data_store.add_partner(new_partner)

        # Clone the 1:1 private transcript into the new partner's private room.
        source_room = data_store.get_room(f"private_{partner_id}")
        cloned_room = data_store.get_room(f"private_{new_id}")
        cloned_message_count = 0
        if source_room and cloned_room:
            speaker_remap = {
                partner_id: {
                    'speaker_id': new_id,
                    'speaker_name': new_partner.name,
                }
            }
            cloned_room.messages = _clone_messages_for_room(
                source_room.messages,
                cloned_room.id,
                speaker_remap=speaker_remap
            )
            cloned_message_count = len(cloned_room.messages)

            cloned_room.scenario = source_room.scenario
            cloned_room.timeline = source_room.timeline
            cloned_room.system_prompt = source_room.system_prompt
            cloned_room.auto_generate = source_room.auto_generate
            cloned_room.auto_generate_mode = source_room.auto_generate_mode
            cloned_room.auto_generate_count = source_room.auto_generate_count
            cloned_room.auto_respond = source_room.auto_respond
            cloned_room.loras = copy.deepcopy(source_room.loras) if source_room.loras else []
            data_store.save()

        return jsonify({
            'id': new_partner.id,
            'name': new_partner.name,
            'message_count': cloned_message_count,
            'status': 'cloned'
        })
    except Exception as e:
        print(f"[clone_partner] Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Send a message and get a response."""
    data = request.json
    room_id = data.get('room_id')
    message_content = data.get('message', '').strip()
    partner_id = data.get('partner_id')  # Who should respond
    continue_scene = data.get('continue_scene', False)  # Just prompt partner to continue
    skip_response = data.get('skip_response', False)  # Just save message, don't get response

    if not message_content and not continue_scene:
        return jsonify({'error': 'No message'}), 400
    if not room_id:
        return jsonify({'error': 'No room specified'}), 400

    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    # === PARSER: Analyze every message ===
    parsed = None
    if message_content and not continue_scene:
        try:
            parsed = parse_message(message_content, room_context={
                'room_id': room_id,
                'location': getattr(room, 'player_location', None),
            })
            # Log for debugging (can remove later)
            print(f"[Parser] {format_parsed_message(parsed)}")

            # TODO: Feed to Stagehand if needs_stagehand_check
            # TODO: Feed to Tension Keeper if needs_tension_check
            # TODO: Handle immediate triggers

            # For now, just accumulate - other systems will read this later
        except Exception as e:
            print(f"[Parser] Error: {e}")
            # Non-fatal - continue without parsing

    # Check for @mention of NPCs (e.g., "@Mira what happened?")
    import re
    npc_mention_match = re.match(r'@(\w+)\s*(.*)', message_content, re.IGNORECASE)
    if npc_mention_match and not continue_scene:
        npc_name = npc_mention_match.group(1)
        player_message_to_npc = npc_mention_match.group(2).strip() or "..."

        # Try to find the NPC in this room
        npc = find_room_npc_by_name(room_id, npc_name)
        if npc:
            # Check if NPC is nearby (not at a different location)
            npc_location = npc.current_location or ""
            player_location = getattr(room, 'player_location', None) or ""

            # NPC is "here" if:
            # - Their location is generic (room_id, "nearby", "here", empty)
            # - OR player has traveled to their location
            # - OR NPC was set to "here" after successful travel
            is_here = (
                npc_location == room_id or
                npc_location in ["nearby", "here", "with you", ""] or
                npc_location.startswith("here") or
                (player_location and player_location.lower() == npc_location.lower())
            )

            if not is_here:
                # NPC is elsewhere - can't talk to them from here
                return jsonify({
                    'error': f"{npc.name} isn't here. They're most likely at: {npc_location}",
                    'npc_location': npc_location,
                    'hint': f"Travel to {npc_location} to find {npc.name}."
                }), 400

            # Found the NPC and they're here! Generate their response
            user_message = Message(
                id=str(uuid.uuid4())[:8],
                speaker_id="user",
                speaker_name=room.player_character_name or settings.user_name,
                content=message_content,
                room_id=room_id,
            )
            data_store.add_message(room_id, user_message)

            # Generate NPC response
            try:
                npc_response_text = run_async(generate_npc_response(npc, player_message_to_npc, room))

                npc_message = Message(
                    id=str(uuid.uuid4())[:8],
                    speaker_id=f"npc_{npc.id}",
                    speaker_name=npc.name,
                    content=npc_response_text,
                    room_id=room_id,
                    message_type="npc",
                    metadata={"npc_id": npc.id, "npc_state": npc.state.value}
                )
                data_store.add_message(room_id, npc_message)

                # Check for DM interjection after NPC interaction too
                dm_interjection = None
                try:
                    interjection_msg = run_async(check_dm_interjection(room_id))
                    if interjection_msg:
                        dm_interjection = {
                            'id': interjection_msg.id,
                            'speaker_id': interjection_msg.speaker_id,
                            'speaker_name': interjection_msg.speaker_name,
                            'content': interjection_msg.content,
                            'message_type': 'narration',
                        }
                except Exception as e:
                    print(f"[DMNarrator] Error: {e}")

                # Soft location check
                try:
                    run_async(check_soft_location_update(room_id))
                except Exception:
                    pass

                result = {
                    'user_message': {
                        'id': user_message.id,
                        'speaker_name': user_message.speaker_name,
                        'content': user_message.content
                    },
                    'response': {
                        'id': npc_message.id,
                        'speaker_id': npc_message.speaker_id,
                        'speaker_name': npc_message.speaker_name,
                        'content': npc_response_text,
                        'is_npc': True,
                        'npc_state': npc.state.value,
                    }
                }
                if dm_interjection:
                    result['dm_interjection'] = dm_interjection

                print(f"[NPC Chat] {npc.name} ({npc.state.value}): {npc_response_text[:60]}...")
                return jsonify(result)

            except Exception as e:
                print(f"[NPC Chat] Error generating response: {e}")
                return jsonify({'error': f'Failed to generate NPC response: {e}'}), 500

    # Create and immediately save user message so clones/snapshots always capture it
    # In observer mode, user messages are narrator/stage directions
    is_observer_mode = getattr(room, 'observer_mode', False)
    user_message = Message(
        id=str(uuid.uuid4())[:8],
        speaker_id="narrator" if is_observer_mode else "user",
        speaker_name="Narrator" if is_observer_mode else settings.user_name,
        content=message_content,
        room_id=room_id,
    )

    # If auto_respond is disabled for 1-on-1 rooms, skip response unless explicitly requested
    if room.partner_id and not room.auto_respond and not skip_response:
        skip_response = True

    # If skip_response, just save the message and return
    if skip_response:
        data_store.add_message(room_id, user_message)
        return jsonify({
            'user_message': {'id': user_message.id, 'speaker_name': user_message.speaker_name, 'content': user_message.content},
            'skipped_response': True,
        })

    # Flag to track if we should trigger consolidation after response
    should_consolidate = False

    # Determine who responds
    all_partners = data_store.get_partners()

    if partner_id:
        # Specific partner requested
        partner = data_store.get_partner(partner_id)
    elif room.partner_id:
        # Private room - that partner responds
        partner = data_store.get_partner(room.partner_id)
    else:
        # Need to specify who responds in common/custom rooms
        # Save user message now since we're waiting for responder selection
        data_store.add_message(room_id, user_message)

        # Still process loot/use mode even though we need responder selection
        response_data = {
            'user_message': {'id': user_message.id, 'speaker_name': user_message.speaker_name, 'content': user_message.content},
            'needs_responder': True,
        }

        # Process inventory actions for StoryBuilder rooms
        if room.player_character_name:
            player_char_id = f"player_{room_id}"

            loot_status = _process_loot_mode(player_char_id, message_content, room_id)
            if loot_status:
                response_data['loot_status'] = loot_status

            use_status = _process_use_mode(player_char_id, message_content, room_id)
            if use_status:
                response_data['use_status'] = use_status

        return jsonify(response_data)

    if not partner:
        return jsonify({'error': 'Partner not found'}), 404

    # Build conversation history
    messages = []
    is_multi_party = room.is_common_room or bool(room.partner_ids)

    for msg in room.messages:
        role = "user" if msg.speaker_id == "user" else "assistant"
        if is_multi_party:
            if msg.speaker_id == "user":
                content = f"{{{msg.speaker_name}}}: {msg.content}"
            elif msg.speaker_id == "narrator" or (is_observer_mode and msg.speaker_name == "Narrator"):
                # Narrator directions are stage directions, not dialogue
                content = f"[NARRATOR: {msg.content}]"
            else:
                content = f"{msg.speaker_name}: {msg.content}"
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": role, "content": msg.content})

    # Add the current user message to AI prompt, then immediately persist it
    # Skip for continue_scene - we're just prompting the partner to continue
    if not continue_scene:
        if is_multi_party:
            if is_observer_mode:
                # In observer mode, user input is narrator/stage direction
                messages.append({"role": "user", "content": f"[NARRATOR: {message_content}]"})
            else:
                messages.append({"role": "user", "content": f"{{{settings.user_name}}}: {message_content}"})
        else:
            messages.append({"role": "user", "content": message_content})
        data_store.add_message(room_id, user_message)

    # Build system prompt with mood context
    # Hierarchy: Room system_prompt > Character custom_system_prompt > Global
    room_has_override = room.system_prompt and room.system_prompt.strip()
    if room_has_override:
        base_system_prompt = room.system_prompt.strip()
    else:
        base_system_prompt = partner.get_effective_system_prompt(settings.global_system_prompt)

    mood_context = build_mood_context(room)
    if room.is_common_room:
        # For common room with room override, pass it directly (skip character override)
        system = partner.get_full_context(all_partners, settings.user_name, base_system_prompt, user_physical_description=settings.user_physical_description)
        system += mood_context
        system += f"\n\nYou ARE {partner.name}. Respond in first person as {partner.name}. Do NOT prefix your response with any name. Do NOT speak as or for other characters - only as yourself.\n\nThis is a turn-based collaborative story. Each character is played by a different player. When you reach a moment where another character would naturally speak or act, that's your cue to pause - they WILL respond. You don't need to carry the narrative alone."
    elif room.partner_ids:
        room_partners = room.get_partners_in_room(all_partners)

        # Filter to only characters who are PHYSICALLY PRESENT (not separated)
        present_ids = room.present_character_ids or []
        present_partners = [p for p in room_partners if p.id in present_ids]
        separated_partners = [p for p in room_partners if p.id not in present_ids]

        # Build context with only present partners
        system = partner.get_full_context(present_partners, settings.user_name, base_system_prompt, user_physical_description=settings.user_physical_description)

        # Add critical context about who is/isn't here - but ONLY for characters this character KNOWS (not strangers)
        if separated_partners and room.character_relationships:
            # Find which separated characters this partner actually knows (not strangers)
            known_separated = []
            partner_rels = next((cr for cr in room.character_relationships if cr.get('character_id') == partner.id), None)

            for sep in separated_partners:
                # Check if partner knows this separated character
                rel_type = 'stranger'  # Default to stranger
                if partner_rels:
                    for rel in partner_rels.get('relationships', []):
                        if rel.get('target_id') == sep.id:
                            rel_type = rel.get('type', 'stranger')
                            break

                # Only include if they're NOT strangers
                if rel_type != 'stranger':
                    known_separated.append(sep)

            # Only add the context if there are known (non-stranger) separated characters
            if known_separated:
                sep_names = [p.name for p in known_separated]
                system += f"\n\n---\n**IMPORTANT - CHARACTER LOCATIONS**\n"
                system += f"The following characters you KNOW are NOT HERE with you right now: {', '.join(sep_names)}\n"
                system += f"They are elsewhere in the world. Do NOT describe them as present, nearby, or visible.\n"
                system += f"You may MENTION them in conversation (wonder where they are, worry about them), but they are physically ABSENT.\n---"

        # Only inject scenario for first 10 turns - after that, context should carry it
        if room.scenario and len(room.messages) < 10:
            # Strip character names to prevent knowledge leaks between strangers
            clean_scenario = strip_characters_from_scenario(room.scenario)
            system += f"\n\n---\nSCENARIO:\n{clean_scenario}\n---"

        # Inject character's own inventory so they know what they have
        try:
            char_inv = inventory_tracker.get_inventory(partner.id)
            if char_inv and char_inv.items:
                # Separate weapons from other items for emphasis
                weapons = [item.name for item in char_inv.items if item.category.value == 'weapon']
                other_items = [item.name for item in char_inv.items if item.category.value != 'weapon'][:8]

                inv_text = "\n\n---\n📦 YOUR INVENTORY:\n"
                if weapons:
                    inv_text += f"Your weapons: {', '.join(weapons)}\n"
                else:
                    inv_text += "Weapons: None - you are unarmed\n"
                if other_items:
                    inv_text += f"Other items: {', '.join(other_items)}\n"
                inv_text += "\nYou may reference items others are carrying, but don't invent items for yourself that aren't listed above.\n---"
                system += inv_text
        except Exception as e:
            print(f"[Chat] Inventory injection failed for {partner.name}: {e}")
            pass  # No inventory for this character

        system += mood_context
        system += f"\n\nYou ARE {partner.name}. Respond in first person as {partner.name}. Do NOT prefix your response with any name. Do NOT speak as or for other characters - only as yourself.\n\nThis is a turn-based collaborative story. Each character is played by a different player. When you reach a moment where another character would naturally speak or act, that's your cue to pause - they WILL respond. You don't need to carry the narrative alone."

        # StoryBuilder: allow character to call the DM for world adjudication
        if (room.character_relationships or room.player_character_name):
            system += """

---
DM CALLS: When your action requires world adjudication (will this succeed? does this work?), you may end your response with [DM: your question?] to ask the game master. Use sparingly - only for pivotal moments where the outcome is uncertain. Example: after throwing a rock to distract a creature, you might end with [DM: does the sound draw it away?]

ITEM ACTIONS: When you give, use, or pick up an item, you may note it with [ITEM: action]. Examples: [ITEM: gave notebook to Kaido], [ITEM: used bandages], [ITEM: picked up yellow slicker]. This helps track inventory accurately.

SEPARATION: If you are parting ways with the player character (fleeing different directions, staying behind while they escape, etc.), end your response with [SEPARATED] to signal you're no longer physically together. Only use this when you are truly splitting up. You can combine with [SEEKING: location] to indicate where you're heading.

SEEKING: If you are heading toward a specific location (especially when separated), use [SEEKING: the rusty anchor] to track your destination. This helps the game reunite characters when they arrive at the same place.

COMBAT: If you are initiating combat with a hostile entity (attacking a zombie, engaging a bandit), use [COMBAT: target]. Example: [COMBAT: zombie]. Only use this when you are actively attacking or engaging an enemy - not for describing threats or potential danger. The game system will handle combat mechanics.

INJURIES: When someone gets hurt, note it with [INJURY: description | severity | bleeding]. Severities: minor, moderate, severe, critical. Add "bleeding" if actively bleeding. Example: [INJURY: deep gash on left arm | moderate | bleeding]. When treating an injury: [TREATED: gash on arm].

SUSTENANCE: When eating or drinking, note it: [ATE: rations] or [DRANK: water]. This tracks hunger and thirst.
---"""
    else:
        system = f"{base_system_prompt}\n\n---\n{partner.get_character()}\n---"

    # Inject memory if partner has memory enabled
    if partner.memory_mode != "none":
        memory = memory_store.get_memory(partner.id, room_id, partner.memory_mode)
        memory_text = memory_consolidator.format_for_prompt(memory)
        if memory_text:
            system += f"\n\n---\n{memory_text}\n---"
            # Hot pink terminal output for memory injection
            print(f"\033[38;5;206m{'='*60}")
            print(f"💭 MEMORY INJECTION for {partner.name}")
            print(f"{'='*60}")
            for line in memory_text.split('\n'):
                print(f"   {line}")
            print(f"{'='*60}\033[0m")

        # Check if we should consolidate after this exchange
        turn_count = memory_store.increment_turn(partner.id, room_id, partner.memory_mode)
        should_consolidate = turn_count >= memory_consolidator.CONSOLIDATION_INTERVAL

    # Inject combat context if active
    combat_context = _build_combat_context(room_id, partner.id)
    combat_resolution = None
    if combat_context:
        system += combat_context
        print("[Combat] " +f"Injected combat context for {partner.name}")

        # Check if player's message describes a combat action
        # If so, resolve it and inject the result for narration
        encounter = EncounterManager.get(room_id)
        if encounter and encounter.is_active and message_content:
            # Detect combat action keywords in player message
            combat_keywords = ['attack', 'strike', 'hit', 'swing', 'slash', 'stab', 'shoot',
                               'fire', 'punch', 'kick', 'bash', 'smash', 'thrust', 'lunge']
            message_lower = message_content.lower()
            is_combat_action = any(kw in message_lower for kw in combat_keywords)

            if is_combat_action:
                # Find the player character in the encounter
                player_char_id = None
                if room.player_character_name:
                    for p in data_store.get_room_partners(room_id):
                        if p.name == room.player_character_name:
                            player_char_id = p.id
                            break

                if player_char_id and player_char_id in encounter.combatants:
                    # Resolve the player's attack
                    combat_resolution = _resolve_combat_action(room_id, player_char_id, message_content)
                    if combat_resolution.get('resolved'):
                        system = _inject_combat_resolution_context(system, combat_resolution)
                        print("[Combat] " +f"Resolved player combat action: {combat_resolution.get('narrative', 'unknown')}")

                        # If combat didn't end, resolve enemy counter-attack
                        if not combat_resolution.get('combat_ended'):
                            # Find a living enemy to counter-attack
                            for c_id, c in encounter.combatants.items():
                                if c.team == "enemies" and c.stats.is_alive:
                                    enemy_resolution = _resolve_combat_action(room_id, c_id, "counter-attack")
                                    if enemy_resolution.get('resolved'):
                                        system = _inject_combat_resolution_context(system, enemy_resolution)
                                        print("[Combat] " +f"Resolved enemy counter: {enemy_resolution.get('narrative', 'unknown')}")
                                    break  # Only one enemy counter per exchange

    # Add observer mode instructions if enabled
    if is_observer_mode:
        system += "\n\n---\n**OBSERVER MODE - FLY ON THE WALL**\n"
        system += "You are in an observer mode scene. There is NO player character in this story. "
        system += "You are interacting ONLY with the other characters present. "
        system += "[NARRATOR: ...] tags are stage directions describing what happens - treat them as narrative setup, not as someone speaking to you. "
        system += "Respond naturally to the situation and other characters. Do NOT acknowledge any observer or narrator as a person in the scene.\n---"

    # Generate response with retry logic
    try:
        max_retries = 3
        response_text = ""

        for attempt in range(max_retries):
            try:
                response_text = run_async(generate_response_async(partner, messages, system))
                response_text = clean_model_tokens(response_text.strip())

                # Check for actual success (not empty, not an error message)
                is_error = response_text.startswith('[API Error:') or response_text.startswith('[Error:')
                if response_text and not is_error:
                    break  # Success!

                if is_error:
                    print(f"[Chat] API error from {partner.name} (attempt {attempt + 1}/{max_retries}): {response_text[:100]}")
                else:
                    print(f"[Chat] Empty response from {partner.name} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)
            except Exception as e:
                print(f"[Chat] Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise

        # Check for empty or error response after all retries
        is_error = response_text.startswith('[API Error:') or response_text.startswith('[Error:')
        if not response_text or is_error:
            error_msg = response_text if is_error else 'AI returned empty response after multiple attempts'
            print(f"[Chat Warning] Failed response from {partner.name} ({partner.model}): {error_msg[:100]}")
            return jsonify({'error': error_msg}), 500

        # Check for character-initiated DM call: [DM: question?]
        # TWO-PASS SYSTEM: If character asks DM, get answer, then let them continue
        auto_dm_message = None
        auto_dm_question = None  # Store the question for frontend display
        dm_call_pattern = r'\[DM:\s*([^\]]+)\]'
        dm_match = re.search(dm_call_pattern, response_text)
        if dm_match and (room.character_relationships or room.player_character_name):
            dm_question = dm_match.group(1).strip()
            auto_dm_question = dm_question

            # Get the partial response (everything before the DM tag)
            partial_response = response_text[:dm_match.start()].strip()
            # Get anything after the DM tag
            after_tag = response_text[dm_match.end():].strip()

            print(f"[AutoDM] Detected DM call from {partner.name}: {dm_question}")

            # Process the DM call (with cooldown check)
            auto_dm_message = _handle_character_dm_call(room_id, partner.name, dm_question)

            # Determine if this is a "catalyst" DM call (at end of response) vs mid-response
            is_catalyst_mode = (
                len(after_tag) < 20 and
                len(partial_response) > 50
            )

            if is_catalyst_mode:
                print(f"[AutoDM] Catalyst mode: DM question at end, no continuation needed")
                response_text = partial_response

            elif auto_dm_message:
                # Two-pass: get continuation incorporating DM ruling
                dm_ruling_text = auto_dm_message.content
                print(f"[AutoDM] Two-pass: getting continuation with DM ruling")

                continuation_system = f"""You are {partner.name}. You just asked the DM a question and received an answer.

Your partial response so far:
"{partial_response}"

You asked the DM: "{dm_question}"

The DM's ruling:
{dm_ruling_text}

Now CONTINUE your response naturally, incorporating what you learned from the DM.
- Do NOT repeat your partial response - just continue from where you left off
- React naturally to the DM's information
- Stay in character
- Keep your continuation concise (2-4 sentences typically)"""

                continuation_messages = [{"role": "user", "content": "Continue your response based on the DM's ruling."}]

                try:
                    continuation = run_async(generate_response_async(partner, continuation_messages, continuation_system))
                    continuation = continuation.strip()
                    continuation = re.sub(dm_call_pattern, '', continuation).strip()

                    if partial_response and continuation:
                        response_text = f"{partial_response}\n\n{continuation}"
                    elif continuation:
                        response_text = continuation
                    else:
                        response_text = partial_response

                    print(f"[AutoDM] Two-pass complete: continuation added")
                except Exception as e:
                    print(f"[AutoDM] Continuation failed: {e}, using partial response")
                    response_text = partial_response if partial_response else response_text
                    response_text = re.sub(dm_call_pattern, '', response_text).strip()
            else:
                # No DM message (cooldown or failure) - just strip the tag
                response_text = re.sub(dm_call_pattern, '', response_text).strip()

        # Check for character-initiated item action: [ITEM: action]
        item_action_pattern = r'\[ITEM:\s*([^\]]+)\]'
        item_match = re.search(item_action_pattern, response_text)
        item_action_message = None
        if item_match and (room.character_relationships or room.player_character_name):
            item_action = item_match.group(1).strip()
            # Strip the tag from the displayed response
            response_text = re.sub(item_action_pattern, '', response_text).strip()
            # Process the item action and create DM narration
            item_result = _handle_character_item_action(room_id, partner.id, partner.name, item_action)
            if item_result:
                item_action_message = Message(
                    id=str(uuid.uuid4())[:8],
                    speaker_id='dm',
                    speaker_name='📦 Inventory',
                    content=item_result,
                    room_id=room_id,
                    message_type='item_action',
                )

        # Check for character self-separation: [SEPARATED]
        if '[SEPARATED]' in response_text and (room.character_relationships or room.player_character_name):
            # Strip the tag from the displayed response
            response_text = response_text.replace('[SEPARATED]', '').strip()
            # Process the separation
            _handle_character_separation(room_id, partner.id, partner.name)

        # Check for character seeking location: [SEEKING: location]
        seeking_pattern = r'\[SEEKING:\s*([^\]]+)\]'
        seeking_match = re.search(seeking_pattern, response_text)
        if seeking_match and (room.character_relationships or room.player_character_name):
            seeking_location = seeking_match.group(1).strip()
            # Strip the tag from the displayed response
            response_text = re.sub(seeking_pattern, '', response_text).strip()
            # Update the character's location to their destination
            char_locations = room.character_locations or {}
            char_locations[partner.id] = seeking_location
            room.character_locations = char_locations
            data_store.save()
            print(f"[Seeking] {partner.name} is heading toward: {seeking_location}")

        # Check for combat initiation: [COMBAT: target | options]
        combat_triggered = False
        combat_result = None
        combat_message = None
        combat_pattern = r'\[COMBAT:\s*([^\]]+)\]'
        combat_match = re.search(combat_pattern, response_text)
        if combat_match and (room.character_relationships or room.player_character_name):
            combat_tag_content = combat_match.group(1).strip()
            # Strip the tag from the displayed response
            response_text = re.sub(combat_pattern, '', response_text).strip()
            # Parse and handle the combat initiation
            combat_info = _parse_combat_tag(combat_tag_content)
            combat_result = _handle_combat_initiation(room_id, partner.id, partner.name, combat_info)
            combat_triggered = True
            print(f"[Combat] {partner.name} initiated combat: {combat_info}")

            # Create combat announcement message if combat actually started
            if combat_result and combat_result.get('combat_started'):
                target = combat_result.get('target', 'unknown')
                initiator = combat_result.get('initiated_by', partner.name)
                target_hp = combat_result.get('target_hp', '?')
                target_ac = combat_result.get('target_ac', '?')
                combat_content = f"**{initiator}** engages **{target}**\n*HP: {target_hp} | AC: {target_ac}*"
                combat_message = Message(
                    id=str(uuid.uuid4())[:8],
                    speaker_id='dm',
                    speaker_name='⚔️ Combat',
                    content=combat_content,
                    room_id=room_id,
                    message_type='combat_start',
                )

        # Check for injury: [INJURY: description | severity | bleeding]
        injury_pattern = r'\[INJURY:\s*([^\]]+)\]'
        for injury_match in re.finditer(injury_pattern, response_text):
            injury_content = injury_match.group(1).strip()
            response_text = response_text.replace(injury_match.group(0), '').strip()
            # Parse injury: "deep gash on arm | moderate | bleeding"
            parts = [p.strip() for p in injury_content.split('|')]
            description = parts[0] if parts else "unknown injury"
            severity = parts[1] if len(parts) > 1 else "minor"
            bleeding = "bleeding" in injury_content.lower()
            # Apply to the character who received the injury (usually player)
            player_char_id = None
            if room.player_character_name:
                for p in data_store.get_room_partners(room_id):
                    if p.name == room.player_character_name:
                        player_char_id = p.id
                        break
            if player_char_id:
                condition_tracker.add_injury(player_char_id, room.player_character_name, description, severity, bleeding, partner.name)
                print(f"[Condition] Added injury to {room.player_character_name}: {description}")

        # Check for treated injury: [TREATED: description]
        treated_pattern = r'\[TREATED:\s*([^\]]+)\]'
        for treated_match in re.finditer(treated_pattern, response_text):
            treated_desc = treated_match.group(1).strip()
            response_text = response_text.replace(treated_match.group(0), '').strip()
            player_char_id = None
            if room.player_character_name:
                for p in data_store.get_room_partners(room_id):
                    if p.name == room.player_character_name:
                        player_char_id = p.id
                        break
            if player_char_id:
                condition_tracker.treat_injury(player_char_id, description_match=treated_desc)
                print(f"[Condition] Treated injury for {room.player_character_name}: {treated_desc}")

        # Check for eating: [ATE: item]
        ate_pattern = r'\[ATE:\s*([^\]]+)\]'
        ate_match = re.search(ate_pattern, response_text)
        if ate_match:
            ate_item = ate_match.group(1).strip()
            response_text = re.sub(ate_pattern, '', response_text).strip()
            # Determine who ate (could be partner or player based on context)
            # For now, assume it's the speaking partner
            condition_tracker.eat(partner.id, "normal")
            print(f"[Condition] {partner.name} ate: {ate_item}")

        # Check for drinking: [DRANK: item]
        drank_pattern = r'\[DRANK:\s*([^\]]+)\]'
        drank_match = re.search(drank_pattern, response_text)
        if drank_match:
            drank_item = drank_match.group(1).strip()
            response_text = re.sub(drank_pattern, '', response_text).strip()
            condition_tracker.drink(partner.id, "normal")
            print(f"[Condition] {partner.name} drank: {drank_item}")

        # Save the AI response (user message was already saved before AI call)
        response_message = Message(
            id=str(uuid.uuid4())[:8],
            speaker_id=partner.id,
            speaker_name=partner.name,
            content=response_text,
            room_id=room_id,
        )
        data_store.add_message(room_id, response_message)

        # Add DM message AFTER partner response for proper transcript ordering
        if auto_dm_message:
            data_store.add_message(room_id, auto_dm_message)

        # Add item action narration AFTER partner response
        if item_action_message:
            data_store.add_message(room_id, item_action_message)

        # Add combat announcement AFTER partner response
        if combat_message:
            data_store.add_message(room_id, combat_message)

        # Hook for inventory tracking, echoes, etc
        on_message_sent(room_id, partner.id, partner.name, response_text, is_user=False)

        # === COMBAT ACTION TRACKING ===
        # If combat is active, record this as an action for the active side
        combat_action_result = None
        combat_side_switch = None
        enemy_turn_result = None
        from combat_system import EncounterManager
        encounter = EncounterManager.get(room_id)
        if encounter and encounter.is_active:
            # Record action and check if turns exhausted
            combat_action_result = encounter.record_action()

            # If turns exhausted, auto-switch sides
            if combat_action_result.get('should_auto_switch'):
                new_side = encounter.switch_sides(reason="turns_exhausted")
                combat_side_switch = {
                    'from_side': combat_action_result['side'],
                    'to_side': new_side,
                    'reason': 'turns_exhausted',
                }
                print(f"[Combat] Auto-switch: {combat_side_switch['from_side']} exhausted turns → {new_side}")

                # If it's now enemy turn, generate their actions
                if new_side == "enemies":
                    try:
                        enemy_turn_result = run_async(_generate_enemy_turn(room_id))
                        if enemy_turn_result.get('side_switch'):
                            combat_side_switch = enemy_turn_result['side_switch']
                    except Exception as e:
                        print(f"[Combat] Error in enemy turn: {e}")

        # Check for DM interjection (every 3-10 turns in StoryBuilder rooms)
        # In combat, this can trigger a side switch
        dm_interjection = None
        try:
            interjection_msg = run_async(check_dm_interjection(room_id))
            if interjection_msg:
                # Check if this was a combat switch
                if interjection_msg.message_type == 'combat_switch':
                    dm_interjection = {
                        'id': interjection_msg.id,
                        'speaker_id': interjection_msg.speaker_id,
                        'speaker_name': interjection_msg.speaker_name,
                        'content': interjection_msg.content,
                        'message_type': 'combat_switch',
                    }
                    # If switched to enemies, trigger their turn
                    encounter = EncounterManager.get(room_id)
                    if encounter and encounter.active_side == "enemies":
                        try:
                            enemy_turn_result = run_async(_generate_enemy_turn(room_id))
                            if enemy_turn_result.get('side_switch'):
                                combat_side_switch = enemy_turn_result['side_switch']
                        except Exception as e:
                            print(f"[Combat] Error in enemy turn after interjection: {e}")
                else:
                    dm_interjection = {
                        'id': interjection_msg.id,
                        'speaker_id': interjection_msg.speaker_id,
                        'speaker_name': interjection_msg.speaker_name,
                        'content': interjection_msg.content,
                        'message_type': 'narration',
                        'interjection_type': interjection_msg.metadata.get('interjection_type') if interjection_msg.metadata else None
                    }
        except Exception as e:
            print(f"[DMNarrator] Error in interjection check: {e}")

        # Soft location check (every ~5 turns, infers if player has moved organically)
        try:
            run_async(check_soft_location_update(room_id))
        except Exception as e:
            print(f"[LocationCheck] Error: {e}")

        # Check for separated character update (every 5-8 turns, message-based)
        separated_update = None
        try:
            separated_tracker = get_separated_tick_tracker()
            if separated_tracker.record_turn(room_id):
                # Check if there are separated characters
                present_ids = room.present_character_ids or []
                separated_ids = [pid for pid in (room.partner_ids or []) if pid not in present_ids]
                if separated_ids:
                    # Pick one separated character to update
                    import random
                    char_id = random.choice(separated_ids)
                    char_partner = next((p for p in data_store.get_partners() if p.id == char_id), None)

                    if char_partner:
                        # Get character's last known location
                        char_locations = room.character_locations or {}
                        last_location = char_locations.get(char_id, "somewhere in the area")

                        # Generate a brief update using Ollama
                        update_prompt = f"""In one sentence, describe what {char_partner.name} is doing right now, elsewhere in this world.

Setting: {room.scenario or 'an unspecified location'}
Character: {char_partner.name} - {char_partner.character_description[:200] if char_partner.character_description else 'a survivor'}
Last seen at: {last_location}

They are NOT with the main group. Write a brief, evocative sentence about their current activity.
If they're moving to a new location, mention where they're heading.
Example: "Juniper crouches behind a rusted car, watching shadows move between buildings."
Example: "Juniper makes her way toward the harbor, staying low in the fog."

Just the sentence, no preamble."""

                        try:
                            update_text = _call_ollama_sync(
                                room.room_model or settings.storybuilder_model,
                                update_prompt,
                                "You write brief, atmospheric updates about characters."
                            )
                            if update_text and update_text.strip():
                                separated_update = {
                                    'character_id': char_id,
                                    'character_name': char_partner.name,
                                    'update': update_text.strip(),
                                }
                                print(f"[SeparatedTick] {char_partner.name}: {update_text.strip()[:60]}...")

                                # Try to extract location hints from the update
                                update_lower = update_text.lower()
                                location_keywords = ['toward', 'towards', 'to the', 'at the', 'near the', 'inside the', 'reaches the', 'arrives at']
                                for keyword in location_keywords:
                                    if keyword in update_lower:
                                        # Extract what comes after the keyword
                                        idx = update_lower.index(keyword) + len(keyword)
                                        potential_loc = update_text[idx:idx+50].strip()
                                        # Clean it up - take until punctuation
                                        for end_char in ['.', ',', '!', '?', ';', ' and ', ' but ', ' while ']:
                                            if end_char in potential_loc:
                                                potential_loc = potential_loc[:potential_loc.index(end_char)]
                                        if potential_loc and len(potential_loc) > 3:
                                            char_locations = room.character_locations or {}
                                            char_locations[char_id] = potential_loc.strip()
                                            room.character_locations = char_locations
                                            data_store.save()
                                            print(f"[SeparatedTick] {char_partner.name} location updated: {potential_loc.strip()}")
                                            break
                        except Exception as e:
                            print(f"[SeparatedTick] Ollama error for {char_partner.name}: {e}")

                separated_tracker.mark_tick_done(room_id)
        except Exception as e:
            print(f"[SeparatedTick] Error: {e}")

        # Trigger memory consolidation in background if needed
        consolidating = False
        if should_consolidate:
            consolidating = True
            _executor.submit(
                _consolidation_worker,
                partner.id,
                partner.name,
                partner.character_description,
                room_id,
                partner.memory_mode,
                [{'role': 'user' if m.speaker_id == 'user' else 'assistant',
                  'content': m.content,
                  'speaker_name': m.speaker_name} for m in room.messages[-50:]]
            )

        result = {
            'user_message': {'id': user_message.id, 'speaker_name': user_message.speaker_name, 'content': user_message.content},
            'response': {
                'id': response_message.id,
                'speaker_id': partner.id,
                'speaker_name': partner.name,
                'avatar': partner.avatar,
                'avatar_image': partner.avatar_image,
                'content': response_text,
            },
            'consolidating': consolidating,
        }

        if dm_interjection:
            result['dm_interjection'] = dm_interjection

        if separated_update:
            result['separated_update'] = separated_update

        # Include auto-DM call result from character
        if auto_dm_message:
            result['character_dm_call'] = {
                'id': auto_dm_message.id,
                'speaker_id': auto_dm_message.speaker_id,
                'speaker_name': auto_dm_message.speaker_name,
                'content': auto_dm_message.content,
                'message_type': 'dm_public',
                'asked_by': auto_dm_message.metadata.get('asked_by') if auto_dm_message.metadata else partner.name,
                'question': auto_dm_question  # The actual question asked
            }

        # Include item action narration
        if item_action_message:
            result['item_action'] = {
                'id': item_action_message.id,
                'speaker_id': item_action_message.speaker_id,
                'speaker_name': item_action_message.speaker_name,
                'content': item_action_message.content,
                'message_type': 'item_action',
            }

        # Include combat announcement for UI
        if combat_message:
            result['combat_announcement'] = {
                'id': combat_message.id,
                'speaker_id': combat_message.speaker_id,
                'speaker_name': combat_message.speaker_name,
                'content': combat_message.content,
                'message_type': 'combat_start',
            }

        # Include combat result if combat was triggered (new combat)
        if combat_result:
            result['combat'] = combat_result

        # Include combat resolution if we resolved combat actions (existing combat)
        if combat_resolution and combat_resolution.get('resolved'):
            result['combat_resolution'] = combat_resolution

        # Include side switch info if it happened
        if combat_side_switch:
            result['combat_side_switch'] = combat_side_switch

        # Include enemy actions if enemies took their turn
        if enemy_turn_result and enemy_turn_result.get('enemy_actions'):
            result['enemy_actions'] = enemy_turn_result['enemy_actions']

        # Always include combat state when combat is active
        encounter = EncounterManager.get(room_id)
        if encounter and encounter.is_active:
            result['combat_state'] = {
                'active': encounter.is_active,
                'round': encounter.round,
                'active_side': encounter.active_side,
                'player_turns_remaining': encounter.player_turns_remaining,
                'enemy_turns_remaining': encounter.enemy_turns_remaining,
                'player_turns_max': encounter.player_turns_max,
                'enemy_turns_max': encounter.enemy_turns_max,
                'combatants': {
                    c_id: {
                        'name': c.stats.name,
                        'hp': c.stats.hp_current,
                        'hp_max': c.stats.hp_max,
                        'alive': c.stats.is_alive,
                        'team': c.team,
                    }
                    for c_id, c in encounter.combatants.items()
                }
            }

        # Get the player character ID (in StoryBuilder rooms, inventory is keyed as player_{room_id})
        player_char_id = None
        if room.player_character_name:
            # Player inventory uses this format
            player_char_id = f"player_{room_id}"

        if player_char_id:
            # Process loot mode if container equipped (backpack ready = looting intent)
            loot_status = _process_loot_mode(player_char_id, message_content, room_id)
            if loot_status:
                result['loot_status'] = loot_status

            # Process use mode if consumable/tool equipped (item ready = use intent)
            use_status = _process_use_mode(player_char_id, message_content, room_id)
            if use_status:
                result['use_status'] = use_status

            # Auto-unequip weapons if no combat triggered
            _auto_unequip_if_no_combat(player_char_id, room_id, combat_triggered)

        return jsonify(result)
    except Exception as e:
        import traceback
        print(f"[Chat Error] {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/respond', methods=['POST'])
def respond():
    """Have a specific partner respond (for common room clicks)."""
    data = request.json
    room_id = data.get('room_id')
    partner_id = data.get('partner_id')
    chain_mode = data.get('chain_mode', False)

    if not room_id or not partner_id:
        return jsonify({'error': 'Missing room_id or partner_id'}), 400

    room = data_store.get_room(room_id)
    partner = data_store.get_partner(partner_id)

    if not room or not partner:
        return jsonify({'error': 'Room or partner not found'}), 404

    # For StoryBuilder rooms, check if character is present with the player
    if room.present_character_ids and partner_id not in room.present_character_ids:
        return jsonify({'error': f'{partner.name} is not with you right now'}), 400

    all_partners = data_store.get_partners()

    # Build conversation history
    messages = []
    is_multi_party = room.is_common_room or bool(room.partner_ids)

    for msg in room.messages:
        role = "user" if msg.speaker_id == "user" else "assistant"
        if is_multi_party:
            if msg.speaker_id == "user":
                content = f"{{{msg.speaker_name}}}: {msg.content}"
            else:
                content = f"{msg.speaker_name}: {msg.content}"
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": role, "content": msg.content})

    # Build system prompt with mood context
    # Hierarchy: Room system_prompt > Character custom_system_prompt > Global
    room_has_override = room.system_prompt and room.system_prompt.strip()
    if room_has_override:
        base_system_prompt = room.system_prompt.strip()
    else:
        base_system_prompt = partner.get_effective_system_prompt(settings.global_system_prompt)

    mood_context = build_mood_context(room)
    if room.is_common_room:
        # For common room with room override, pass it directly (skip character override)
        system = partner.get_full_context(all_partners, settings.user_name, base_system_prompt, user_physical_description=settings.user_physical_description)
        system += mood_context
        system += f"\n\nYou ARE {partner.name}. Respond in first person as {partner.name}. Do NOT prefix your response with any name. Do NOT speak as or for other characters - only as yourself.\n\nThis is a turn-based collaborative story. Each character is played by a different player. When you reach a moment where another character would naturally speak or act, that's your cue to pause - they WILL respond. You don't need to carry the narrative alone."
    elif room.partner_ids:
        room_partners = room.get_partners_in_room(all_partners)

        # Filter to only characters who are PHYSICALLY PRESENT (not separated)
        present_ids = room.present_character_ids or []
        present_partners = [p for p in room_partners if p.id in present_ids]
        separated_partners = [p for p in room_partners if p.id not in present_ids]

        # Build context with only present partners (prevents knowledge leaks to strangers)
        system = partner.get_full_context(present_partners, settings.user_name, base_system_prompt, user_physical_description=settings.user_physical_description)

        # Add context about who is/isn't here - but ONLY for characters this partner KNOWS
        if separated_partners and room.character_relationships:
            known_separated = []
            partner_rels = next((cr for cr in room.character_relationships if cr.get('character_id') == partner.id), None)

            for sep in separated_partners:
                rel_type = 'stranger'
                if partner_rels:
                    for rel in partner_rels.get('relationships', []):
                        if rel.get('target_id') == sep.id:
                            rel_type = rel.get('type', 'stranger')
                            break
                if rel_type != 'stranger':
                    known_separated.append(sep)

            if known_separated:
                sep_names = [p.name for p in known_separated]
                system += f"\n\n---\n**IMPORTANT - CHARACTER LOCATIONS**\n"
                system += f"The following characters you KNOW are NOT HERE with you right now: {', '.join(sep_names)}\n"
                system += f"They are elsewhere in the world. Do NOT describe them as present, nearby, or visible.\n---"

        # Only inject scenario for first 10 turns - after that, context should carry it
        if room.scenario and len(room.messages) < 10:
            # Strip character names to prevent knowledge leaks between strangers
            clean_scenario = strip_characters_from_scenario(room.scenario)
            system += f"\n\n---\nSCENARIO:\n{clean_scenario}\n---"
        system += mood_context
        system += f"\n\nYou ARE {partner.name}. Respond in first person as {partner.name}. Do NOT prefix your response with any name. Do NOT speak as or for other characters - only as yourself.\n\nThis is a turn-based collaborative story. Each character is played by a different player. When you reach a moment where another character would naturally speak or act, that's your cue to pause - they WILL respond. You don't need to carry the narrative alone."

        # StoryBuilder: allow character to call the DM for world adjudication
        if (room.character_relationships or room.player_character_name):
            system += """

---
DM CALLS: When your action requires world adjudication (will this succeed? does this work?), you may end your response with [DM: your question?] to ask the game master. Use sparingly - only for pivotal moments where the outcome is uncertain. Example: after throwing a rock to distract a creature, you might end with [DM: does the sound draw it away?]

ITEM ACTIONS: When you give, use, or pick up an item, you may note it with [ITEM: action]. Examples: [ITEM: gave notebook to Kaido], [ITEM: used bandages], [ITEM: picked up yellow slicker]. This helps track inventory accurately.

SEPARATION: If you are parting ways with the player character (fleeing different directions, staying behind while they escape, etc.), end your response with [SEPARATED] to signal you're no longer physically together. Only use this when you are truly splitting up. You can combine with [SEEKING: location] to indicate where you're heading.

SEEKING: If you are heading toward a specific location (especially when separated), use [SEEKING: the rusty anchor] to track your destination. This helps the game reunite characters when they arrive at the same place.

COMBAT: If you are initiating combat with a hostile entity (attacking a zombie, engaging a bandit), use [COMBAT: target]. Example: [COMBAT: zombie]. Only use this when you are actively attacking or engaging an enemy - not for describing threats or potential danger. The game system will handle combat mechanics.

INJURIES: When someone gets hurt, note it with [INJURY: description | severity | bleeding]. Severities: minor, moderate, severe, critical. Add "bleeding" if actively bleeding. Example: [INJURY: deep gash on left arm | moderate | bleeding]. When treating an injury: [TREATED: gash on arm].

SUSTENANCE: When eating or drinking, note it: [ATE: rations] or [DRANK: water]. This tracks hunger and thirst.
---"""
    else:
        system = f"{base_system_prompt}\n\n---\n{partner.get_character()}\n---"

    # Check for pending images this partner hasn't seen
    pending_image_path = _get_pending_image_for_partner(room_id, partner_id)
    if pending_image_path:
        from pathlib import Path
        import base64
        image_file = Path(pending_image_path)
        if image_file.exists():
            try:
                image_data = base64.b64encode(image_file.read_bytes()).decode('utf-8')
                # Add image to the last message or create a new one
                image_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": f"({settings.user_name} shared this image with the room)"
                        }
                    ]
                }
                messages.append(image_message)
                # Add instruction about the image
                system += f"\n\n{settings.user_name} has shared an image with the room. You can see it and react naturally if you want, or acknowledge it briefly and continue the conversation."
                # Mark as seen
                _mark_pending_image_seen(room_id, partner_id)
            except Exception as e:
                print(f"[Respond] Failed to load pending image: {e}")

    # Inject combat context if active
    combat_context = _build_combat_context(room_id, partner.id)
    if combat_context:
        system += combat_context
        print("[Combat] " +f"Injected combat context for {partner.name} in /respond")

    # Generate response with retry logic
    try:
        max_retries = 3
        response_text = ""

        for attempt in range(max_retries):
            try:
                response_text = run_async(generate_response_async(partner, messages, system))
                response_text = clean_model_tokens(response_text.strip())

                # Check for actual success (not empty, not an error message)
                is_error = response_text.startswith('[API Error:') or response_text.startswith('[Error:')
                if response_text and not is_error:
                    break  # Success!

                if is_error:
                    print(f"[Respond] API error from {partner.name} (attempt {attempt + 1}/{max_retries}): {response_text[:100]}")
                else:
                    print(f"[Respond] Empty response from {partner.name} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)
            except Exception as e:
                print(f"[Respond] Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise

        # Check for empty or error response after all retries
        is_error = response_text.startswith('[API Error:') or response_text.startswith('[Error:')
        if not response_text or is_error:
            error_msg = response_text if is_error else 'AI returned empty response after multiple attempts'
            print(f"[Respond Warning] Failed response from {partner.name} ({partner.model}): {error_msg[:100]}")
            return jsonify({'error': error_msg}), 500

        # Check for character-initiated DM call: [DM: question?]
        # TWO-PASS SYSTEM: If character asks DM, get answer, then let them continue
        auto_dm_message = None
        auto_dm_question = None  # Store the question for frontend display
        dm_call_pattern = r'\[DM:\s*([^\]]+)\]'
        dm_match = re.search(dm_call_pattern, response_text)
        if dm_match and (room.character_relationships or room.player_character_name):
            dm_question = dm_match.group(1).strip()
            auto_dm_question = dm_question

            # Get the partial response (everything before the DM tag)
            partial_response = response_text[:dm_match.start()].strip()
            # Get anything after the DM tag (might be empty or continuation attempt)
            after_tag = response_text[dm_match.end():].strip()

            print(f"[AutoDM] Detected DM call from {partner.name}: {dm_question}")

            # Process the DM call (with cooldown check)
            auto_dm_message = _handle_character_dm_call(room_id, partner.name, dm_question)

            # Determine if this is a "catalyst" DM call (at end of response) vs mid-response
            # Catalyst mode: tag at end, no meaningful content after, substantial content before
            # In catalyst mode, character doesn't need to continue - DM answer stands alone
            is_catalyst_mode = (
                len(after_tag) < 20 and  # Nothing meaningful after
                len(partial_response) > 50  # Substantial response already given
            )

            if is_catalyst_mode:
                print(f"[AutoDM] Catalyst mode: DM question at end, no continuation needed")
                response_text = partial_response  # Just use the response before the tag

            # If we got a DM ruling, do second pass to let character continue (unless catalyst mode)
            elif auto_dm_message:
                dm_ruling_text = auto_dm_message.content

                # Build continuation prompt
                continuation_system = f"""You are {partner.name}. You just asked the DM a question and received an answer.

Your partial response so far:
"{partial_response}"

You asked the DM: "{dm_question}"

The DM's ruling:
{dm_ruling_text}

Now CONTINUE your response naturally, incorporating what you learned from the DM.
- Do NOT repeat your partial response - just continue from where you left off
- React naturally to the DM's information
- Stay in character
- Keep your continuation concise (2-4 sentences typically)"""

                continuation_messages = [{"role": "user", "content": "Continue your response based on the DM's ruling."}]

                try:
                    continuation = run_async(generate_response_async(partner, continuation_messages, continuation_system))
                    continuation = continuation.strip()

                    # Clean any accidental DM tags from continuation
                    continuation = re.sub(dm_call_pattern, '', continuation).strip()

                    # Combine: partial + continuation (skip after_tag as it was pre-ruling speculation)
                    if partial_response and continuation:
                        response_text = f"{partial_response}\n\n{continuation}"
                    elif continuation:
                        response_text = continuation
                    else:
                        response_text = partial_response

                    print(f"[AutoDM] Two-pass complete: continuation added")
                except Exception as e:
                    print(f"[AutoDM] Continuation failed: {e}, using partial response")
                    response_text = partial_response if partial_response else response_text
                    # Strip the DM tag from original if continuation failed
                    response_text = re.sub(dm_call_pattern, '', response_text).strip()
            else:
                # No DM message (cooldown or failure) - just strip the tag
                response_text = re.sub(dm_call_pattern, '', response_text).strip()

        # Check for character-initiated item action: [ITEM: action]
        item_action_pattern = r'\[ITEM:\s*([^\]]+)\]'
        item_match = re.search(item_action_pattern, response_text)
        item_action_message = None
        if item_match and (room.character_relationships or room.player_character_name):
            item_action = item_match.group(1).strip()
            # Strip the tag from the displayed response
            response_text = re.sub(item_action_pattern, '', response_text).strip()
            # Process the item action and create DM narration
            item_result = _handle_character_item_action(room_id, partner.id, partner.name, item_action)
            if item_result:
                item_action_message = Message(
                    id=str(uuid.uuid4())[:8],
                    speaker_id='dm',
                    speaker_name='📦 Inventory',
                    content=item_result,
                    room_id=room_id,
                    message_type='item_action',
                )

        # Check for character self-separation: [SEPARATED]
        if '[SEPARATED]' in response_text and (room.character_relationships or room.player_character_name):
            # Strip the tag from the displayed response
            response_text = response_text.replace('[SEPARATED]', '').strip()
            # Process the separation
            _handle_character_separation(room_id, partner.id, partner.name)

        # Check for character seeking location: [SEEKING: location]
        seeking_pattern = r'\[SEEKING:\s*([^\]]+)\]'
        seeking_match = re.search(seeking_pattern, response_text)
        if seeking_match and (room.character_relationships or room.player_character_name):
            seeking_location = seeking_match.group(1).strip()
            # Strip the tag from the displayed response
            response_text = re.sub(seeking_pattern, '', response_text).strip()
            # Update the character's location to their destination
            char_locations = room.character_locations or {}
            char_locations[partner.id] = seeking_location
            room.character_locations = char_locations
            data_store.save()
            print(f"[Seeking] {partner.name} is heading toward: {seeking_location}")

        # Check for combat initiation: [COMBAT: target | options]
        combat_triggered = False
        combat_result = None
        combat_message = None
        combat_pattern = r'\[COMBAT:\s*([^\]]+)\]'
        combat_match = re.search(combat_pattern, response_text)
        if combat_match and (room.character_relationships or room.player_character_name):
            combat_tag_content = combat_match.group(1).strip()
            # Strip the tag from the displayed response
            response_text = re.sub(combat_pattern, '', response_text).strip()
            # Parse and handle the combat initiation
            combat_info = _parse_combat_tag(combat_tag_content)
            combat_result = _handle_combat_initiation(room_id, partner.id, partner.name, combat_info)
            combat_triggered = True
            print(f"[Combat] {partner.name} initiated combat: {combat_info}")

            # Create combat announcement message if combat actually started
            if combat_result and combat_result.get('combat_started'):
                target = combat_result.get('target', 'unknown')
                initiator = combat_result.get('initiated_by', partner.name)
                target_hp = combat_result.get('target_hp', '?')
                target_ac = combat_result.get('target_ac', '?')
                combat_content = f"**{initiator}** engages **{target}**\n*HP: {target_hp} | AC: {target_ac}*"
                combat_message = Message(
                    id=str(uuid.uuid4())[:8],
                    speaker_id='dm',
                    speaker_name='⚔️ Combat',
                    content=combat_content,
                    room_id=room_id,
                    message_type='combat_start',
                )

        # Check for injury: [INJURY: description | severity | bleeding]
        injury_pattern = r'\[INJURY:\s*([^\]]+)\]'
        for injury_match in re.finditer(injury_pattern, response_text):
            injury_content = injury_match.group(1).strip()
            response_text = response_text.replace(injury_match.group(0), '').strip()
            parts = [p.strip() for p in injury_content.split('|')]
            description = parts[0] if parts else "unknown injury"
            severity = parts[1] if len(parts) > 1 else "minor"
            bleeding = "bleeding" in injury_content.lower()
            player_char_id = None
            if room.player_character_name:
                for p in data_store.get_room_partners(room_id):
                    if p.name == room.player_character_name:
                        player_char_id = p.id
                        break
            if player_char_id:
                condition_tracker.add_injury(player_char_id, room.player_character_name, description, severity, bleeding, partner.name)
                print(f"[Condition] Added injury to {room.player_character_name}: {description}")

        # Check for treated injury: [TREATED: description]
        treated_pattern = r'\[TREATED:\s*([^\]]+)\]'
        for treated_match in re.finditer(treated_pattern, response_text):
            treated_desc = treated_match.group(1).strip()
            response_text = response_text.replace(treated_match.group(0), '').strip()
            player_char_id = None
            if room.player_character_name:
                for p in data_store.get_room_partners(room_id):
                    if p.name == room.player_character_name:
                        player_char_id = p.id
                        break
            if player_char_id:
                condition_tracker.treat_injury(player_char_id, description_match=treated_desc)
                print(f"[Condition] Treated injury for {room.player_character_name}: {treated_desc}")

        # Check for eating: [ATE: item]
        ate_pattern = r'\[ATE:\s*([^\]]+)\]'
        ate_match = re.search(ate_pattern, response_text)
        if ate_match:
            ate_item = ate_match.group(1).strip()
            response_text = re.sub(ate_pattern, '', response_text).strip()
            condition_tracker.eat(partner.id, "normal")
            print(f"[Condition] {partner.name} ate: {ate_item}")

        # Check for drinking: [DRANK: item]
        drank_pattern = r'\[DRANK:\s*([^\]]+)\]'
        drank_match = re.search(drank_pattern, response_text)
        if drank_match:
            drank_item = drank_match.group(1).strip()
            response_text = re.sub(drank_pattern, '', response_text).strip()
            condition_tracker.drink(partner.id, "normal")
            print(f"[Condition] {partner.name} drank: {drank_item}")

        # Save response
        response_message = Message(
            id=str(uuid.uuid4())[:8],
            speaker_id=partner.id,
            speaker_name=partner.name,
            content=response_text,
            room_id=room_id,
        )
        data_store.add_message(room_id, response_message)

        # Add DM message AFTER partner response for proper transcript ordering
        # (Juniper's response first, then her DM question/ruling)
        if auto_dm_message:
            data_store.add_message(room_id, auto_dm_message)

        # Add item action narration AFTER partner response
        if item_action_message:
            data_store.add_message(room_id, item_action_message)

        # Add combat announcement AFTER partner response
        if combat_message:
            data_store.add_message(room_id, combat_message)

        # Hook for inventory tracking, echoes, etc
        on_message_sent(room_id, partner.id, partner.name, response_text, is_user=False)

        # === COMBAT ACTION TRACKING ===
        combat_action_result = None
        combat_side_switch = None
        enemy_turn_result = None
        from combat_system import EncounterManager
        encounter = EncounterManager.get(room_id)
        if encounter and encounter.is_active:
            combat_action_result = encounter.record_action()
            if combat_action_result.get('should_auto_switch'):
                new_side = encounter.switch_sides(reason="turns_exhausted")
                combat_side_switch = {
                    'from_side': combat_action_result['side'],
                    'to_side': new_side,
                    'reason': 'turns_exhausted',
                }
                print(f"[Combat] Auto-switch: {combat_side_switch['from_side']} exhausted turns → {new_side}")

                # If it's now enemy turn, generate their actions
                if new_side == "enemies":
                    try:
                        enemy_turn_result = run_async(_generate_enemy_turn(room_id))
                        if enemy_turn_result.get('side_switch'):
                            combat_side_switch = enemy_turn_result['side_switch']
                    except Exception as e:
                        print(f"[Combat] Error in enemy turn: {e}")

        # Check for DM interjection (counts AI responses too)
        # In combat, this can trigger a side switch
        dm_interjection = None
        try:
            interjection_msg = run_async(check_dm_interjection(room_id))
            if interjection_msg:
                # Check if this was a combat switch
                if interjection_msg.message_type == 'combat_switch':
                    dm_interjection = {
                        'id': interjection_msg.id,
                        'speaker_id': interjection_msg.speaker_id,
                        'speaker_name': interjection_msg.speaker_name,
                        'content': interjection_msg.content,
                        'message_type': 'combat_switch',
                    }
                    # If switched to enemies, trigger their turn
                    encounter = EncounterManager.get(room_id)
                    if encounter and encounter.active_side == "enemies":
                        try:
                            enemy_turn_result = run_async(_generate_enemy_turn(room_id))
                            if enemy_turn_result.get('side_switch'):
                                combat_side_switch = enemy_turn_result['side_switch']
                        except Exception as e:
                            print(f"[Combat] Error in enemy turn after interjection: {e}")
                else:
                    dm_interjection = {
                        'id': interjection_msg.id,
                        'speaker_id': interjection_msg.speaker_id,
                        'speaker_name': interjection_msg.speaker_name,
                        'content': interjection_msg.content,
                        'message_type': 'narration',
                        'interjection_type': interjection_msg.metadata.get('interjection_type') if interjection_msg.metadata else None
                    }
        except Exception as e:
            print(f"[DMNarrator] Error in /respond interjection check: {e}")

        # Soft location check
        try:
            run_async(check_soft_location_update(room_id))
        except Exception:
            pass

        # Chain mode: ask model who they want to address next
        chain_next_partner_id = None
        if chain_mode and is_multi_party:
            # Get other partners in the room (excluding self and non-present characters)
            present_ids = room.present_character_ids or []
            if room.is_common_room:
                other_partners = [p for p in all_partners if p.id != partner_id]
            else:
                room_partner_ids = room.partner_ids or []
                # For StoryBuilder rooms, only include present characters
                if present_ids:
                    other_partners = [p for p in all_partners if p.id in room_partner_ids and p.id != partner_id and p.id in present_ids]
                else:
                    other_partners = [p for p in all_partners if p.id in room_partner_ids and p.id != partner_id]

            if other_partners:
                # Build the chain intent prompt
                partner_names = [p.name for p in other_partners]
                names_list = ", ".join(partner_names)

                chain_prompt = f"""You just said this in a group conversation:

"{response_text[:1000]}"

The other people present are: {names_list}, and {settings.user_name} (the human).

Would you like to directly address or prompt a response from any of them?
- If yes, reply with JUST their name (e.g., "Grace" or "{settings.user_name}")
- If no, reply with "nobody"

Your choice:"""

                chain_messages = [{"role": "user", "content": chain_prompt}]
                chain_system = f"You are {partner.name}. Answer briefly with just a name or 'nobody'."

                try:
                    chain_response = run_async(generate_response_async(partner, chain_messages, chain_system))
                    chain_response = chain_response.strip().lower()

                    # Match response to a partner
                    if chain_response and chain_response != 'nobody' and chain_response != 'no one' and chain_response != 'none':
                        for p in other_partners:
                            if p.name.lower() in chain_response or chain_response in p.name.lower():
                                chain_next_partner_id = p.id
                                break
                except Exception as e:
                    pass  # Chain intent failed, just don't continue chain

        result = {
            'id': response_message.id,
            'speaker_id': partner.id,
            'speaker_name': partner.name,
            'avatar': partner.avatar,
            'avatar_image': partner.avatar_image,
            'content': response_text,
            'chain_next_partner_id': chain_next_partner_id,
        }
        if dm_interjection:
            result['dm_interjection'] = dm_interjection

        # Include auto-DM call result from character
        if auto_dm_message:
            result['character_dm_call'] = {
                'id': auto_dm_message.id,
                'speaker_id': auto_dm_message.speaker_id,
                'speaker_name': auto_dm_message.speaker_name,
                'content': auto_dm_message.content,
                'message_type': 'dm_public',
                'asked_by': auto_dm_message.metadata.get('asked_by') if auto_dm_message.metadata else partner.name,
                'question': auto_dm_question  # The actual question asked
            }

        # Include item action narration
        if item_action_message:
            result['item_action'] = {
                'id': item_action_message.id,
                'speaker_id': item_action_message.speaker_id,
                'speaker_name': item_action_message.speaker_name,
                'content': item_action_message.content,
                'message_type': 'item_action',
            }

        # Include combat announcement for UI
        if combat_message:
            result['combat_announcement'] = {
                'id': combat_message.id,
                'speaker_id': combat_message.speaker_id,
                'speaker_name': combat_message.speaker_name,
                'content': combat_message.content,
                'message_type': 'combat_start',
            }

        # Include combat result if combat was triggered
        if combat_result:
            result['combat'] = combat_result

        # Include side switch info if it happened
        if combat_side_switch:
            result['combat_side_switch'] = combat_side_switch

        # Include enemy actions if enemies took their turn
        if enemy_turn_result and enemy_turn_result.get('enemy_actions'):
            result['enemy_actions'] = enemy_turn_result['enemy_actions']

        # Always include combat state when combat is active
        encounter = EncounterManager.get(room_id)
        if encounter and encounter.is_active:
            result['combat_state'] = {
                'active': encounter.is_active,
                'round': encounter.round,
                'active_side': encounter.active_side,
                'player_turns_remaining': encounter.player_turns_remaining,
                'enemy_turns_remaining': encounter.enemy_turns_remaining,
                'player_turns_max': encounter.player_turns_max,
                'enemy_turns_max': encounter.enemy_turns_max,
                'combatants': {
                    c_id: {
                        'name': c.stats.name,
                        'hp': c.stats.hp_current,
                        'hp_max': c.stats.hp_max,
                        'alive': c.stats.is_alive,
                        'team': c.team,
                    }
                    for c_id, c in encounter.combatants.items()
                }
            }

        # Auto-unequip player character's weapons if no combat triggered
        # Get the player character ID (in StoryBuilder rooms)
        player_char_id = None
        if room.player_character_name:
            for p in data_store.get_room_partners(room_id):
                if p.name == room.player_character_name:
                    player_char_id = p.id
                    break
        if player_char_id:
            _auto_unequip_if_no_combat(player_char_id, room_id, combat_triggered)

        return jsonify(result)
    except Exception as e:
        import traceback
        print(f"[Respond Error] {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/whisper', methods=['POST'])
def whisper():
    """Private whisper to a partner - not added to room transcript."""
    data = request.json
    room_id = data.get('room_id')
    partner_id = data.get('partner_id')
    message = data.get('message', '').strip()
    whisper_history = data.get('whisper_history', [])

    if not room_id or not partner_id or not message:
        return jsonify({'error': 'Missing required fields'}), 400

    room = data_store.get_room(room_id)
    partner = data_store.get_partner(partner_id)

    if not room or not partner:
        return jsonify({'error': 'Room or partner not found'}), 404

    all_partners = data_store.get_partners()

    # Build conversation history from room (for context)
    story_messages = []
    is_multi_party = room.is_common_room or bool(room.partner_ids)

    for msg in room.messages[-50:]:  # Last 50 messages for context
        if is_multi_party:
            if msg.speaker_id == "user":
                content = f"{{{msg.speaker_name}}}: {msg.content}"
            else:
                content = f"{msg.speaker_name}: {msg.content}"
            story_messages.append({"role": "user", "content": content})
        else:
            role = "user" if msg.speaker_id == "user" else "assistant"
            story_messages.append({"role": role, "content": msg.content})

    # Build whisper conversation for the API
    api_messages = story_messages.copy()

    # Add whisper context marker
    api_messages.append({
        "role": "user",
        "content": f"[WHISPER MODE - The following is a private aside between {settings.user_name} and {partner.name}. This conversation is happening 'off-stage' - other characters cannot hear it. Feel free to break character slightly if needed to discuss the story, answer questions, or coordinate. When you respond, you are whispering back privately.]"
    })

    # Add previous whisper history
    for w in whisper_history:
        if w['role'] == 'user':
            api_messages.append({"role": "user", "content": f"[Whisper from {settings.user_name}]: {w['content']}"})
        else:
            api_messages.append({"role": "assistant", "content": w['content']})

    # Add current whisper
    api_messages.append({"role": "user", "content": f"[Whisper from {settings.user_name}]: {message}"})

    # Build system prompt
    effective_prompt = partner.get_effective_system_prompt(settings.global_system_prompt)
    system = f"{effective_prompt}\n\n---\n{partner.get_character()}\n---"

    # Add whisper instructions
    system += f"\n\nYou are in a private whisper conversation with {settings.user_name}. This is happening outside the main story - other characters cannot hear. You can discuss the story, answer meta questions, or just chat privately. Keep responses relatively brief since this is a whisper."

    try:
        response_text = run_async(generate_response_async(partner, api_messages, system))
        response_text = clean_model_tokens(response_text.strip())

        if not response_text or response_text.startswith('[API Error:') or response_text.startswith('[Error:'):
            return jsonify({'error': response_text or 'Empty response'}), 500

        # Note: We do NOT save whispers to the room transcript
        return jsonify({
            'response': {
                'speaker_id': partner.id,
                'speaker_name': partner.name,
                'avatar': partner.avatar,
                'avatar_image': partner.avatar_image,
                'content': response_text,
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/user/avatar', methods=['POST'])
def upload_user_avatar():
    """Upload user avatar image."""
    import shutil
    from pathlib import Path

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400

    # Save to user data directory
    avatar_dir = settings.data_dir / 'user'
    avatar_dir.mkdir(parents=True, exist_ok=True)

    # Use a fixed filename so we always overwrite
    ext = Path(file.filename).suffix or '.png'
    avatar_path = avatar_dir / f'avatar{ext}'

    file.save(str(avatar_path))
    settings.user_avatar = str(avatar_path)

    # Save settings
    _save_settings()

    return jsonify({'path': str(avatar_path)})


@app.route('/user/avatar', methods=['DELETE'])
def delete_user_avatar():
    """Remove user avatar."""
    from pathlib import Path

    if settings.user_avatar:
        try:
            Path(settings.user_avatar).unlink(missing_ok=True)
        except Exception:
            pass
        settings.user_avatar = ''
        _save_settings()

    return jsonify({'status': 'cleared'})


def _user_selfie_worker(job_id: str, character_data: dict, room_id: str):
    """Background worker for user/player selfie generation."""
    try:
        import httpx
        from pathlib import Path
        from datetime import datetime
        import re

        # Extract character info
        name = character_data.get('name', 'Traveler')
        gender = character_data.get('gender', '')
        alignment = character_data.get('alignment', '').replace('_', ' ')
        role = character_data.get('role', '').replace('_', ' ')
        backstory = character_data.get('backstory', {})
        backstory_title = backstory.get('title', '')
        backstory_desc = backstory.get('description', '')
        skills = backstory.get('skills', '')

        # Build a character description from the backstory data
        character_context = f"""Character: {name}
Gender: {gender}
Background: {backstory_title} - {backstory_desc}
Skills: {skills}
Alignment: {alignment}
Role: {role}"""

        # Verify ComfyUI is reachable BEFORE spending an API call on the self-description.
        try:
            from image_gen import get_generator
            generator = get_generator()
            comfy_ready = bool(generator and generator.is_available())
        except ImportError:
            generator = None
            comfy_ready = False

        if not comfy_ready:
            update_job(job_id, 'completed', result={
                'type': 'user_selfie',
                'name': name,
                'description': None,
                'images': [],
                'comfy_offline': True
            })
            return

        # Use Ollama to generate an image-ready description
        update_job(job_id, 'generating_description')

        system_prompt = f"""You are {name}. You are describing yourself for a portrait artist.

{character_context}

Based on your backstory and character, describe your appearance vividly.
Include: face, expression, clothing appropriate to your background, posture, mood.
Be specific and painterly. This description will be used to create a portrait.
Keep it under 100 words. Use comma-separated descriptive phrases.
Include art style hints naturally (dramatic lighting, oil painting style, etc).
Just describe yourself - no explanations or meta-commentary."""

        # Call Ollama for the description
        try:
            resp = httpx.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.image_prompt_model or "deepseek-v3.1:671b-cloud",
                    "prompt": "Describe your appearance right now, as if for a portrait artist who will paint you.",
                    "system": system_prompt,
                    "stream": False
                },
                timeout=60.0
            )
            if resp.status_code == 200:
                self_description = resp.json().get("response", "").strip()
            else:
                self_description = f"portrait of {name}, {gender}, {backstory_title}, detailed face, artistic style"
        except Exception as e:
            print(f"[user_selfie] Ollama error: {e}")
            self_description = f"portrait of {name}, {gender}, {backstory_title}, detailed face, artistic style"

        # Generate the image (ComfyUI availability was verified at the top)
        try:
            update_job(job_id, 'generating_image')

            # Prepend gender for image consistency
            image_prompt = self_description
            if gender:
                image_prompt = f"{gender}, {self_description}"

            # Get room LoRAs if available
            room = data_store.get_room(room_id) if room_id else None
            captured_loras = None
            if room and room.loras:
                captured_loras = [l for l in room.loras if l.get('enabled')]

            # Generate to user folder
            image_paths = generator.generate_avatar(
                prompt=image_prompt,
                partner_id='user',
                count=1,
                partner_name=name,
                room_id=room_id,
                captured_loras=captured_loras
            )

            # Check if job was cancelled
            if is_job_cancelled(job_id):
                for p in image_paths:
                    try:
                        Path(p).unlink()
                    except:
                        pass
                return

            update_job(job_id, 'completed', result={
                'type': 'user_selfie',
                'name': name,
                'description': self_description,
                'images': [str(p) for p in image_paths]
            })

        except ImportError:
            update_job(job_id, 'completed', result={
                'type': 'user_selfie',
                'name': name,
                'description': self_description,
                'images': [],
                'comfy_offline': True
            })
        except Exception as e:
            update_job(job_id, 'completed', result={
                'type': 'user_selfie',
                'name': name,
                'description': self_description,
                'images': [],
                'error': str(e)
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        update_job(job_id, 'failed', error=str(e))


@app.route('/user/selfie', methods=['POST'])
def generate_user_selfie():
    """Generate a selfie for the player character (user) in a StoryBuilder room."""
    data = request.json or {}

    character = data.get('character')
    room_id = data.get('room_id')

    if not character:
        return jsonify({'error': 'No character data provided'}), 400

    # Create background job
    job_id = create_job('user_selfie')
    _executor.submit(_user_selfie_worker, job_id, character, room_id)

    return jsonify({
        'job_id': job_id,
        'status': 'pending'
    })


@app.route('/user/avatar/set', methods=['POST'])
def set_user_avatar_from_image():
    """Set user avatar from a generated selfie image, with optional cropping."""
    from pathlib import Path
    from PIL import Image
    import io

    data = request.json or {}
    image_path = data.get('image_path')
    crop = data.get('crop')

    if not image_path:
        return jsonify({'error': 'No image path provided'}), 400

    source_path = Path(image_path)
    if not source_path.exists():
        return jsonify({'error': 'Image not found'}), 404

    # Ensure user avatar directory exists
    avatar_dir = settings.data_dir / 'user'
    avatar_dir.mkdir(parents=True, exist_ok=True)
    avatar_path = avatar_dir / 'avatar.png'

    try:
        img = Image.open(source_path)

        if crop:
            # Apply crop: extract the visible portion based on crop params
            scale = crop.get('scale', 1)
            offset_x = crop.get('offsetX', 0)
            offset_y = crop.get('offsetY', 0)
            container_size = crop.get('containerSize', 300)

            # Calculate the crop region in original image coordinates
            # The visible area in container coordinates is (0, 0) to (container_size, container_size)
            # Convert to image coordinates
            left = -offset_x / scale
            top = -offset_y / scale
            right = left + container_size / scale
            bottom = top + container_size / scale

            # Clamp to image bounds
            left = max(0, left)
            top = max(0, top)
            right = min(img.width, right)
            bottom = min(img.height, bottom)

            # Crop
            img = img.crop((int(left), int(top), int(right), int(bottom)))

        # Resize to standard avatar size (512x512 for quality)
        img = img.resize((512, 512), Image.Resampling.LANCZOS)

        # Save as PNG
        img.save(str(avatar_path), 'PNG')

        # Update settings
        settings.user_avatar = str(avatar_path)
        _save_settings()

        return jsonify({'path': str(avatar_path)})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/partner/<partner_id>/avatar', methods=['POST'])
def upload_partner_avatar(partner_id):
    """Upload custom avatar image for a partner."""
    from pathlib import Path

    partner = data_store.get_partner(partner_id)
    if not partner:
        return jsonify({'error': 'Partner not found'}), 404

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400

    # Save to partner avatars directory
    avatar_dir = settings.data_dir / 'avatars' / partner_id
    avatar_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix or '.png'
    avatar_path = avatar_dir / f'avatar{ext}'

    file.save(str(avatar_path))
    partner.avatar_image = str(avatar_path)
    data_store.update_partner(partner)

    return jsonify({'path': str(avatar_path)})


@app.route('/partner/<partner_id>/background', methods=['POST'])
def upload_partner_background(partner_id):
    """Upload custom background image for a partner."""
    from pathlib import Path

    partner = data_store.get_partner(partner_id)
    if not partner:
        return jsonify({'error': 'Partner not found'}), 404

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400

    # Save to partner backgrounds directory
    bg_dir = settings.data_dir / 'backgrounds' / partner_id
    bg_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix or '.png'
    bg_path = bg_dir / f'background{ext}'

    file.save(str(bg_path))
    partner.background_image = str(bg_path)
    data_store.update_partner(partner)

    return jsonify({'path': str(bg_path)})


def _save_settings():
    """Save user settings to disk."""
    import json
    from pathlib import Path

    settings_file = settings.data_dir / 'settings.json'
    settings_data = {
        'user_name': settings.user_name,
        'user_physical_description': settings.user_physical_description,
        'user_avatar': settings.user_avatar,
        'auto_narration_images': settings.auto_narration_images,
        'global_system_prompt': settings.global_system_prompt,
        'favorite_prompts': settings.favorite_prompts,
        'saved_system_prompts': settings.saved_system_prompts,
        'storybuilder_model': settings.storybuilder_model,
        'voice_enabled': settings.voice_enabled,
    }
    settings_file.write_text(json.dumps(settings_data, indent=2))


@app.route('/settings', methods=['GET'])
def get_settings():
    """Get global settings."""
    import json
    from image_gen import get_generator, MODEL_PRESETS
    generator = get_generator()

    # Load persisted image settings if generator not available
    persisted = {}
    settings_file = settings.data_dir / "settings.json"
    if settings_file.exists():
        try:
            persisted = json.loads(settings_file.read_text())
        except:
            pass

    # For security, only return masked versions of API keys (or full if user just set them this session)
    # Check if voice APIs are available (for showing/hiding voice features)
    openai_api_key = settings.openai_api_key or os.getenv('OPENAI_API_KEY')
    elevenlabs_api_key = settings.elevenlabs_api_key or os.getenv('ELEVENLABS_API_KEY')

    # Check if local Whisper is available (for STT without OpenAI)
    local_whisper_available = False
    try:
        import faster_whisper
        local_whisper_available = True
    except ImportError:
        pass

    return jsonify({
        'user_name': settings.user_name,
        'user_gender': settings.user_gender,
        'user_physical_description': settings.user_physical_description,
        'user_avatar': settings.user_avatar,
        'auto_narration_images': settings.auto_narration_images,
        'global_system_prompt': settings.global_system_prompt,
        'voice_enabled': settings.voice_enabled,
        'openai_available': bool(openai_api_key) or local_whisper_available,  # For mic button visibility (local Whisper OR OpenAI)
        'elevenlabs_available': bool(elevenlabs_api_key),  # For voice toggle visibility
        'anthropic_api_key': settings.anthropic_api_key or '',
        'openai_api_key': settings.openai_api_key or '',
        'elevenlabs_api_key': settings.elevenlabs_api_key or '',
        'favorite_prompts': settings.favorite_prompts,
        'saved_system_prompts': settings.saved_system_prompts,
        'storybuilder_model': settings.storybuilder_model,
        'model_preset': generator.model_preset if generator else persisted.get('model_preset', 'illustrious'),
        'available_presets': {k: v['name'] for k, v in MODEL_PRESETS.items()},
        # Image gen overrides (empty string/0 = use preset default)
        'sampler_override': getattr(generator, 'sampler_override', '') if generator else persisted.get('sampler_override', ''),
        'scheduler_override': getattr(generator, 'scheduler_override', '') if generator else persisted.get('scheduler_override', ''),
        'steps_override': getattr(generator, 'steps_override', 0) if generator else persisted.get('steps_override', 0),
        'cfg_override': getattr(generator, 'cfg_override', 0) if generator else persisted.get('cfg_override', 0),
        'width_override': getattr(generator, 'width_override', 0) if generator else persisted.get('width_override', 0),
        'height_override': getattr(generator, 'height_override', 0) if generator else persisted.get('height_override', 0),
        'negative_prompt': getattr(generator, 'negative_prompt', '') if generator else persisted.get('negative_prompt', ''),
        # Preset defaults for placeholder display
        'preset_width': generator.default_width if generator else MODEL_PRESETS.get(persisted.get('model_preset', 'illustrious'), {}).get('width', 896),
        'preset_height': generator.default_height if generator else MODEL_PRESETS.get(persisted.get('model_preset', 'illustrious'), {}).get('height', 1152),
        # LoRAs are now room-specific, managed via /loras endpoints
        # Hi-res upscaler
        'hires_enabled': getattr(generator, 'hires_enabled', False) if generator else persisted.get('hires_enabled', False),
        'hires_upscaler': getattr(generator, 'hires_upscaler', '') if generator else persisted.get('hires_upscaler', ''),
        'hires_scale': getattr(generator, 'hires_scale', 2.0) if generator else persisted.get('hires_scale', 2.0),
        'hires_denoise': getattr(generator, 'hires_denoise', 0.4) if generator else persisted.get('hires_denoise', 0.4),
        # Connection settings
        'ollama_base_url': settings.ollama_base_url,
        'default_ollama_model': settings.default_ollama_model,
        'comfy_url': settings.comfy_url,
        # Custom checkpoint
        'custom_checkpoint': settings.custom_checkpoint,
        'custom_checkpoint_type': settings.custom_checkpoint_type,
    })


@app.route('/settings', methods=['POST'])
def update_settings():
    """Update global settings."""
    import json
    from pathlib import Path

    try:
        data = request.json
    except Exception as e:
        return jsonify({'error': f'Invalid JSON: {e}'}), 400

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        # Safety check: warn if jobs are running (but don't block)
        active_jobs = 0
        with _jobs_lock:
            for job in _jobs.values():
                if job['status'] in ('pending', 'generating_description', 'generating_image'):
                    active_jobs += 1

        # If force flag is not set and jobs are running, warn but still allow
        # (Client should check for this and confirm with user)
        settings_warning = None
        if active_jobs > 0 and not data.get('force'):
            settings_warning = f"Warning: {active_jobs} jobs are running. Settings saved anyway."

        # Update in-memory settings
        if 'user_name' in data:
            settings.user_name = data['user_name']
        if 'user_gender' in data:
            settings.user_gender = data['user_gender']
        if 'user_physical_description' in data:
            settings.user_physical_description = data['user_physical_description']
        if 'global_system_prompt' in data:
            settings.global_system_prompt = data['global_system_prompt']
        if 'favorite_prompts' in data:
            settings.favorite_prompts = data['favorite_prompts']
        if 'saved_system_prompts' in data:
            settings.saved_system_prompts = data['saved_system_prompts']
        if 'storybuilder_model' in data:
            settings.storybuilder_model = data['storybuilder_model']
        if 'voice_enabled' in data:
            settings.voice_enabled = data['voice_enabled']
        if 'auto_narration_images' in data:
            settings.auto_narration_images = data['auto_narration_images']
        if 'anthropic_api_key' in data and data['anthropic_api_key']:
            settings.anthropic_api_key = data['anthropic_api_key']
        if 'openai_api_key' in data and data['openai_api_key']:
            settings.openai_api_key = data['openai_api_key']
        if 'elevenlabs_api_key' in data and data['elevenlabs_api_key']:
            settings.elevenlabs_api_key = data['elevenlabs_api_key']
        if 'ollama_base_url' in data and data['ollama_base_url']:
            settings.ollama_base_url = data['ollama_base_url'].rstrip('/')
        if 'default_ollama_model' in data and data['default_ollama_model']:
            settings.default_ollama_model = data['default_ollama_model']
        if 'comfy_url' in data and data['comfy_url']:
            settings.comfy_url = data['comfy_url'].rstrip('/')

        # Custom checkpoint
        if 'custom_checkpoint' in data:
            settings.custom_checkpoint = data['custom_checkpoint'].strip()
        if 'custom_checkpoint_type' in data:
            settings.custom_checkpoint_type = data['custom_checkpoint_type']

        # Handle model preset and overrides
        from image_gen import get_generator
        generator = get_generator()
        if generator:
            if 'model_preset' in data:
                generator.set_model_preset(data['model_preset'])
            # Set overrides (empty string or 0 means use preset default)
            generator.sampler_override = data.get('sampler_override', '')
            generator.scheduler_override = data.get('scheduler_override', '')
            generator.steps_override = int(data.get('steps_override', 0))
            generator.cfg_override = float(data.get('cfg_override', 0))
            generator.width_override = int(data.get('width_override', 0))
            generator.height_override = int(data.get('height_override', 0))
            generator.negative_prompt = data.get('negative_prompt', '')
            # LoRAs are now room-specific, not set via global settings
            # Hi-res upscaler settings
            generator.hires_enabled = data.get('hires_enabled', False)
            generator.hires_upscaler = data.get('hires_upscaler', '')
            generator.hires_scale = float(data.get('hires_scale', 2.0))
            generator.hires_denoise = float(data.get('hires_denoise', 0.4))
            # Custom checkpoint
            generator.custom_checkpoint = settings.custom_checkpoint
            generator.custom_checkpoint_type = settings.custom_checkpoint_type

        # Persist to a settings file
        settings_file = settings.data_dir / "settings.json"
        settings_data = {
            'user_name': settings.user_name,
            'user_gender': settings.user_gender,
            'user_physical_description': settings.user_physical_description,
            'global_system_prompt': settings.global_system_prompt,
            'favorite_prompts': settings.favorite_prompts,
            'saved_system_prompts': settings.saved_system_prompts,
            'storybuilder_model': settings.storybuilder_model,
            'voice_enabled': settings.voice_enabled,
            'openai_api_key': settings.openai_api_key,
            'elevenlabs_api_key': settings.elevenlabs_api_key,
            'model_preset': data.get('model_preset', 'illustrious'),
            'sampler_override': data.get('sampler_override', ''),
            'scheduler_override': data.get('scheduler_override', ''),
            'steps_override': int(data.get('steps_override', 0)),
            'cfg_override': float(data.get('cfg_override', 0)),
            'width_override': int(data.get('width_override', 0)),
            'height_override': int(data.get('height_override', 0)),
            # LoRAs removed - now room-specific
            'hires_enabled': data.get('hires_enabled', False),
            'hires_upscaler': data.get('hires_upscaler', ''),
            'hires_scale': float(data.get('hires_scale', 2.0)),
            'hires_denoise': float(data.get('hires_denoise', 0.4)),
            # Connection settings
            'ollama_base_url': settings.ollama_base_url,
            'default_ollama_model': settings.default_ollama_model,
            'comfy_url': settings.comfy_url,
            'custom_checkpoint': settings.custom_checkpoint,
            'custom_checkpoint_type': settings.custom_checkpoint_type,
        }
        settings_file.write_text(json.dumps(settings_data, indent=2))

        result = {'status': 'ok'}
        if settings_warning:
            result['warning'] = settings_warning
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/providers', methods=['GET'])
def get_providers():
    """Get available providers and their models."""
    providers = provider_manager.get_available_providers()
    result = {}
    for p in providers:
        result[p] = provider_manager.get_models_for_provider(p)
    return jsonify(result)


@app.route('/providers/refresh', methods=['POST'])
def refresh_providers():
    """Force refresh the models list from all providers (esp. Ollama)."""
    ollama_provider = provider_manager.get_provider('ollama')
    if ollama_provider:
        models = ollama_provider.refresh_models()
        return jsonify({'status': 'ok', 'ollama_models': models, 'count': len(models)})
    return jsonify({'error': 'Ollama provider not available'}), 500


@app.route('/partners/<partner_id>/avatar', methods=['POST'])
def set_partner_avatar(partner_id):
    """Set a partner's avatar from a generated image, with optional cropping."""
    from pathlib import Path

    data = request.json
    image_path = data.get('image_path')
    crop_data = data.get('crop')  # {scale, offsetX, offsetY, containerSize}

    if not image_path:
        return jsonify({'error': 'No image path provided'}), 400

    partner = data_store.get_partner(partner_id)
    if not partner:
        return jsonify({'error': 'Partner not found'}), 404

    try:
        from image_gen import get_generator
        from PIL import Image

        generator = get_generator()

        source_path = Path(image_path)
        if not source_path.exists():
            return jsonify({'error': 'Image file not found'}), 404

        # If crop data provided, crop the image
        if crop_data:
            img = Image.open(source_path)

            scale = crop_data['scale']
            offset_x = crop_data['offsetX']
            offset_y = crop_data['offsetY']
            container_size = crop_data['containerSize']

            # Calculate the crop region in original image coordinates
            # The visible area in the container maps to a region in the original image
            crop_left = -offset_x / scale
            crop_top = -offset_y / scale
            crop_right = crop_left + container_size / scale
            crop_bottom = crop_top + container_size / scale

            # Clamp to image bounds
            crop_left = max(0, crop_left)
            crop_top = max(0, crop_top)
            crop_right = min(img.width, crop_right)
            crop_bottom = min(img.height, crop_bottom)

            # Crop and resize to a nice avatar size
            cropped = img.crop((int(crop_left), int(crop_top), int(crop_right), int(crop_bottom)))
            cropped = cropped.resize((256, 256), Image.Resampling.LANCZOS)

            # Save to partner's folder as avatar.png
            partner_dir = generator.get_partner_dir(partner_id)
            final_path = partner_dir / "avatar.png"
            cropped.save(final_path, "PNG")
        else:
            # No crop, just copy
            final_path = generator.set_avatar(partner_id, source_path)

        # Update partner's avatar_image field
        partner.avatar_image = str(final_path)
        data_store.update_partner(partner)

        return jsonify({
            'status': 'ok',
            'avatar_path': str(final_path)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/partners/<partner_id>/background', methods=['POST'])
def set_partner_background(partner_id):
    """Set a partner's background image for room list display."""
    from pathlib import Path

    data = request.json
    image_path = data.get('image_path')
    crop_data = data.get('crop')  # {scale, offsetX, offsetY, containerWidth, containerHeight}

    if not image_path:
        return jsonify({'error': 'No image path provided'}), 400

    partner = data_store.get_partner(partner_id)
    if not partner:
        return jsonify({'error': 'Partner not found'}), 404

    try:
        from image_gen import get_generator
        from PIL import Image

        generator = get_generator()

        source_path = Path(image_path)
        if not source_path.exists():
            return jsonify({'error': 'Image file not found'}), 404

        img = Image.open(source_path)

        # If crop data provided, crop the image
        if crop_data:
            scale = crop_data['scale']
            offset_x = crop_data['offsetX']
            offset_y = crop_data['offsetY']
            container_width = crop_data.get('containerWidth', 280)
            container_height = crop_data.get('containerHeight', 80)

            # Calculate the crop region in original image coordinates
            crop_left = -offset_x / scale
            crop_top = -offset_y / scale
            crop_right = crop_left + container_width / scale
            crop_bottom = crop_top + container_height / scale

            # Clamp to image bounds
            crop_left = max(0, crop_left)
            crop_top = max(0, crop_top)
            crop_right = min(img.width, crop_right)
            crop_bottom = min(img.height, crop_bottom)

            # Crop and resize to banner size
            cropped = img.crop((int(crop_left), int(crop_top), int(crop_right), int(crop_bottom)))
            cropped = cropped.resize((560, 160), Image.Resampling.LANCZOS)
        else:
            # No crop - resize to fit banner proportions
            target_ratio = 560 / 160  # ~3.5:1
            img_ratio = img.width / img.height

            if img_ratio > target_ratio:
                # Image is wider, crop sides
                new_width = int(img.height * target_ratio)
                left = (img.width - new_width) // 2
                cropped = img.crop((left, 0, left + new_width, img.height))
            else:
                # Image is taller, crop top/bottom
                new_height = int(img.width / target_ratio)
                top = (img.height - new_height) // 2
                cropped = img.crop((0, top, img.width, top + new_height))

            cropped = cropped.resize((560, 160), Image.Resampling.LANCZOS)

        # Save to partner's folder as background.png
        partner_dir = generator.get_partner_dir(partner_id)
        final_path = partner_dir / "background.png"
        cropped.save(final_path, "PNG")

        # Update partner's background_image field
        partner.background_image = str(final_path)
        data_store.update_partner(partner)

        return jsonify({
            'status': 'ok',
            'background_path': str(final_path)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/rooms/<room_id>/background', methods=['POST'])
def set_room_background(room_id):
    """Set a room's background image for room list display."""
    from pathlib import Path

    data = request.json
    image_path = data.get('image_path')
    crop_data = data.get('crop')  # {scale, offsetX, offsetY, containerWidth, containerHeight}

    if not image_path:
        return jsonify({'error': 'No image path provided'}), 400

    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    try:
        from PIL import Image

        source_path = Path(image_path)
        if not source_path.exists():
            return jsonify({'error': 'Image file not found'}), 404

        img = Image.open(source_path)

        # If crop data provided, crop the image
        if crop_data:
            scale = crop_data['scale']
            offset_x = crop_data['offsetX']
            offset_y = crop_data['offsetY']
            container_width = crop_data.get('containerWidth', 280)
            container_height = crop_data.get('containerHeight', 80)

            # Calculate the crop region in original image coordinates
            crop_left = -offset_x / scale
            crop_top = -offset_y / scale
            crop_right = crop_left + container_width / scale
            crop_bottom = crop_top + container_height / scale

            # Clamp to image bounds
            crop_left = max(0, crop_left)
            crop_top = max(0, crop_top)
            crop_right = min(img.width, crop_right)
            crop_bottom = min(img.height, crop_bottom)

            # Crop and resize to banner size
            cropped = img.crop((int(crop_left), int(crop_top), int(crop_right), int(crop_bottom)))
            cropped = cropped.resize((560, 160), Image.Resampling.LANCZOS)
        else:
            # No crop - resize to fit banner proportions
            target_ratio = 560 / 160  # ~3.5:1
            img_ratio = img.width / img.height

            if img_ratio > target_ratio:
                # Image is wider, crop sides
                new_width = int(img.height * target_ratio)
                left = (img.width - new_width) // 2
                cropped = img.crop((left, 0, left + new_width, img.height))
            else:
                # Image is taller, crop top/bottom
                new_height = int(img.width / target_ratio)
                top = (img.height - new_height) // 2
                cropped = img.crop((0, top, img.width, top + new_height))

            cropped = cropped.resize((560, 160), Image.Resampling.LANCZOS)

        # Save to room's folder as background.png
        room_dir = Path.home() / ".roundtable" / "rooms" / room_id
        room_dir.mkdir(parents=True, exist_ok=True)
        final_path = room_dir / "background.png"
        cropped.save(final_path, "PNG")

        # Update room's background_image field
        room.background_image = str(final_path)
        data_store.save()

        return jsonify({
            'status': 'ok',
            'background_path': str(final_path)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/partners/<partner_id>/gallery', methods=['GET'])
def get_partner_gallery(partner_id):
    """Get generated images for a partner, optionally filtered by room."""
    from pathlib import Path
    import re

    partner = data_store.get_partner(partner_id)
    if not partner:
        return jsonify({'error': 'Partner not found'}), 404

    # Optional room_id filter - if provided, only show images from that room
    # Common room shows all images (no filter)
    room_id = request.args.get('room_id')
    filter_by_room = room_id and room_id != 'common'

    try:
        base_dir = Path.home() / ".roundtable"
        partner_dir = base_dir / "avatars" / partner_id
        scenes_dir = base_dir / "scenes"
        images = []

        # Helper to get prompt from sidecar .txt file
        def get_prompt_for_image(img_path):
            prompt_path = img_path.with_suffix('.txt')
            if prompt_path.exists():
                try:
                    return prompt_path.read_text(encoding='utf-8').strip()
                except:
                    pass
            return None

        # Helper to check if image belongs to a specific room
        def image_matches_room(filename, target_room_id):
            # Sanitize room_id the same way we do when saving
            sanitized = re.sub(r'[^\w\-]', '', target_room_id.replace(' ', '_'))[:30]
            return f"_room_{sanitized}" in filename

        # Get all images from partner's dedicated folder
        print(f"[gallery] Looking for images in: {partner_dir}, room_filter={room_id}")
        if partner_dir.exists():
            for img_path in partner_dir.glob("*.png"):
                # Skip the avatar.png (the set avatar)
                if img_path.name == "avatar.png":
                    continue
                # Apply room filter if specified
                if filter_by_room and not image_matches_room(img_path.name, room_id):
                    continue
                images.append({
                    'path': str(img_path),
                    'filename': img_path.name,
                    'type': 'avatar',
                    'is_current': partner.avatar_image and str(img_path) == partner.avatar_image,
                    'prompt': get_prompt_for_image(img_path)
                })

        # Also check for legacy files (old format: partner_id_*.png in avatars root)
        # These don't have room info, so only show in common room or if no filter
        if not filter_by_room:
            avatars_root = base_dir / "avatars"
            for img_path in avatars_root.glob(f"{partner_id}*.png"):
                if img_path.is_file():  # Skip if it's actually a directory
                    images.append({
                        'path': str(img_path),
                        'filename': img_path.name,
                        'type': 'avatar',
                        'is_current': partner.avatar_image and str(img_path) == partner.avatar_image,
                        'prompt': get_prompt_for_image(img_path)
                    })

        # Find scene images from this room (or private room if in private room)
        if filter_by_room:
            # Only scenes from this specific room
            for img_path in scenes_dir.glob(f"scene_{room_id}*.png"):
                images.append({
                    'path': str(img_path),
                    'filename': img_path.name,
                    'type': 'scene',
                    'is_current': False,
                    'prompt': get_prompt_for_image(img_path)
                })
        else:
            # Common room: show scenes from private room
            private_room_id = f"private_{partner_id}"
            for img_path in scenes_dir.glob(f"scene_{private_room_id}*.png"):
                images.append({
                    'path': str(img_path),
                    'filename': img_path.name,
                    'type': 'scene',
                    'is_current': False,
                    'prompt': get_prompt_for_image(img_path)
                })

        # Add favorite status to each image (per-room)
        # Use the room_id if filtering, otherwise use 'common' for the gallery view
        fav_room = room_id if room_id else 'common'
        for img in images:
            img['favorited'] = _is_favorite(img['path'], fav_room)

        # Sort by modification time (newest first) - favorites stay in chronological position
        images.sort(key=lambda x: -Path(x['path']).stat().st_mtime)

        return jsonify({
            'partner_id': partner_id,
            'partner_name': partner.name,
            'images': images,
            'current_avatar': partner.avatar_image,
            'room_filter': room_id
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/images/delete', methods=['POST'])
def delete_image():
    """Delete a generated image."""
    from pathlib import Path

    data = request.json
    image_path = data.get('path')

    if not image_path:
        return jsonify({'error': 'No path provided'}), 400

    try:
        path = Path(image_path)
        base_dir = Path.home() / ".roundtable"

        # Security: only allow deleting from .roundtable
        path.resolve().relative_to(base_dir.resolve())

        if path.exists():
            path.unlink()
            return jsonify({'status': 'deleted'})
        else:
            return jsonify({'error': 'File not found'}), 404

    except ValueError:
        return jsonify({'error': 'Access denied'}), 403
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/images/favorite', methods=['POST'])
def toggle_image_favorite():
    """Toggle favorite status of an image (per-room)."""
    data = request.json
    image_path = data.get('path')
    room_id = data.get('room_id', 'common')

    if not image_path:
        return jsonify({'error': 'No path provided'}), 400

    is_fav = _toggle_favorite(image_path, room_id)
    return jsonify({'favorited': is_fav})


@app.route('/images/favorites', methods=['GET'])
def get_room_favorites():
    """Get favorited images for a specific room's collage background."""
    from pathlib import Path
    import random

    room_id = request.args.get('room_id', 'common')
    limit = int(request.args.get('limit', 100))  # Default limit of 100
    favorites = _load_favorites(room_id)
    images = []

    for path in favorites:
        # Verify file still exists
        if Path(path).exists():
            images.append({'path': path})

    # If we have more than the limit, randomly sample
    if len(images) > limit:
        images = random.sample(images, limit)

    return jsonify(images)


@app.route('/images/<path:image_path>')
def serve_image(image_path):
    """Serve generated images."""
    from pathlib import Path
    from flask import send_file

    # Security: only serve from .roundtable directory
    base_dir = Path.home() / ".roundtable"

    # Normalize the path (handle both forward and back slashes)
    image_path = image_path.replace('\\', '/')

    # If path starts with known subdirs, use it directly
    if image_path.startswith('avatars/') or image_path.startswith('scenes/'):
        full_path = base_dir / image_path
    elif '/' not in image_path and '\\' not in image_path:
        # Just a filename - try to find it in avatars or scenes
        for subdir in ['avatars', 'scenes']:
            candidate = base_dir / subdir / image_path
            if candidate.exists():
                full_path = candidate
                break
        else:
            full_path = base_dir / image_path
    else:
        # Some other path - try it directly under base_dir
        full_path = base_dir / image_path

    # Verify the file is within .roundtable
    try:
        full_path.resolve().relative_to(base_dir.resolve())
    except ValueError:
        return jsonify({'error': 'Access denied'}), 403

    if not full_path.exists():
        return jsonify({'error': f'Image not found: {full_path}'}), 404

    return send_file(full_path, mimetype='image/png')


# ============================================================================
# LoRA Gallery
# ============================================================================

def _get_lora_metadata_path():
    """Get path to LoRA metadata file."""
    return settings.data_dir / "lora_gallery.json"

def _load_lora_metadata():
    """Load LoRA metadata from disk."""
    path = _get_lora_metadata_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {"loras": {}}

def _save_lora_metadata(data):
    """Save LoRA metadata to disk."""
    path = _get_lora_metadata_path()
    path.write_text(json.dumps(data, indent=2))

@app.route('/loras', methods=['GET'])
def get_loras():
    """Get all LoRAs with their metadata, optionally merged with room-specific settings."""
    room_id = request.args.get('room_id')

    metadata = _load_lora_metadata()
    loras = metadata.get("loras", {})
    categories = metadata.get("categories", [])

    # Get room-specific LoRA settings if room_id provided
    room_lora_settings = {}
    if room_id:
        room = data_store.get_room(room_id)
        if room and room.loras:
            # Convert room loras list to dict for easy lookup
            for lora in room.loras:
                room_lora_settings[lora.get("name")] = lora

    # Convert to list format for frontend
    result = []
    for name, data in loras.items():
        # Start with global metadata
        lora_data = {
            "name": name,
            "display_name": data.get("display_name", name.replace(".safetensors", "").replace("_", " ")),
            "enabled": False,  # Default to disabled
            "weight": data.get("weight", 1.0),
            "trigger": data.get("trigger", ""),
            "previews": data.get("previews", []),
            "available": data.get("available", True),
            "category": data.get("category", "")
        }

        # Override with room-specific settings if available
        if name in room_lora_settings:
            room_settings = room_lora_settings[name]
            lora_data["enabled"] = room_settings.get("enabled", False)
            lora_data["weight"] = room_settings.get("weight", lora_data["weight"])
            lora_data["trigger"] = room_settings.get("trigger", lora_data["trigger"])

        result.append(lora_data)

    # Sort by display name
    result.sort(key=lambda x: x["display_name"].lower())

    return jsonify({"loras": result, "categories": categories})

@app.route('/loras/scan', methods=['POST'])
def scan_loras():
    """Scan ComfyUI for available LoRAs and update metadata."""
    import requests

    try:
        # Query ComfyUI object_info for LoRA loader node
        comfy_url = settings.comfy_url.rstrip('/')
        response = requests.get(f"{comfy_url}/object_info/LoraLoader", timeout=5)

        if response.status_code != 200:
            # Try alternate endpoint
            response = requests.get(f"{comfy_url}/object_info", timeout=5)
            if response.status_code != 200:
                return jsonify({"error": "Could not connect to ComfyUI"}), 500

            data = response.json()
            if "LoraLoader" in data:
                lora_names = data["LoraLoader"]["input"]["required"]["lora_name"][0]
            else:
                return jsonify({"error": "LoraLoader not found in ComfyUI"}), 500
        else:
            data = response.json()
            lora_names = data["LoraLoader"]["input"]["required"]["lora_name"][0]

        # Load existing metadata
        metadata = _load_lora_metadata()
        existing = metadata.get("loras", {})

        # Add new LoRAs, preserve existing settings
        for name in lora_names:
            if name not in existing:
                existing[name] = {
                    "display_name": name.replace(".safetensors", "").replace("_", " ").title(),
                    "enabled": False,
                    "weight": 1.0,
                    "trigger": "",
                    "previews": []
                }

        # Mark LoRAs that are no longer available
        for name in list(existing.keys()):
            if name not in lora_names:
                existing[name]["available"] = False
            else:
                existing[name]["available"] = True

        metadata["loras"] = existing
        _save_lora_metadata(metadata)

        return jsonify({"count": len(lora_names), "message": f"Found {len(lora_names)} LoRAs"})

    except requests.exceptions.ConnectionError:
        return jsonify({"error": "ComfyUI not running"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/loras/save', methods=['POST'])
def save_loras():
    """Save LoRA settings to a room."""
    data = request.json
    loras = data.get("loras", [])
    room_id = data.get("room_id")

    if not room_id:
        return jsonify({"error": "room_id required"}), 400

    room = data_store.get_room(room_id)
    if not room:
        return jsonify({"error": "Room not found"}), 404

    # Save enabled LoRAs to the room (only save those that are enabled or have custom settings)
    room_loras = []
    for lora in loras:
        if lora.get("enabled") or lora.get("weight", 1.0) != 1.0 or lora.get("trigger"):
            room_loras.append({
                "name": lora.get("name"),
                "display_name": lora.get("display_name", lora.get("name", "")),  # Store display name for filenames
                "enabled": lora.get("enabled", False),
                "weight": lora.get("weight", 1.0),
                "trigger": lora.get("trigger", "")
            })

    room.loras = room_loras
    data_store.save()

    return jsonify({"status": "ok", "enabled_count": len([l for l in room_loras if l.get("enabled")])})

@app.route('/loras/<path:lora_name>/preview', methods=['POST'])
def add_lora_preview(lora_name):
    """Add a preview image for a LoRA."""
    data = request.json
    image_path = data.get("image_path")

    if not image_path:
        return jsonify({"error": "image_path required"}), 400

    metadata = _load_lora_metadata()
    existing = metadata.get("loras", {})

    # URL decode the lora name
    from urllib.parse import unquote
    lora_name = unquote(lora_name)

    if lora_name not in existing:
        return jsonify({"error": "LoRA not found"}), 404

    # Add preview if not already present
    previews = existing[lora_name].get("previews", [])
    if image_path not in previews:
        previews.append(image_path)
        existing[lora_name]["previews"] = previews

    metadata["loras"] = existing
    _save_lora_metadata(metadata)

    return jsonify({"status": "ok", "previews": previews})

@app.route('/loras/<path:lora_name>/preview', methods=['DELETE'])
def remove_lora_preview(lora_name):
    """Remove a preview image from a LoRA."""
    data = request.json
    image_path = data.get("image_path")

    if not image_path:
        return jsonify({"error": "image_path required"}), 400

    metadata = _load_lora_metadata()
    existing = metadata.get("loras", {})

    # URL decode the lora name
    from urllib.parse import unquote
    lora_name = unquote(lora_name)

    if lora_name not in existing:
        return jsonify({"error": "LoRA not found"}), 404

    # Remove preview if present
    previews = existing[lora_name].get("previews", [])
    if image_path in previews:
        previews.remove(image_path)
        existing[lora_name]["previews"] = previews

    metadata["loras"] = existing
    _save_lora_metadata(metadata)

    return jsonify({"status": "ok", "previews": previews})

@app.route('/loras/find-by-image', methods=['POST'])
def find_lora_by_image():
    """Find which LoRA(s) an image is a preview for."""
    data = request.json
    image_path = data.get("image_path")

    if not image_path:
        return jsonify({"error": "image_path required"}), 400

    metadata = _load_lora_metadata()
    loras = metadata.get("loras", {})

    # Find all LoRAs that have this image as a preview
    matches = []
    for lora_name, lora_data in loras.items():
        previews = lora_data.get("previews", [])
        if image_path in previews:
            matches.append({
                "name": lora_name,
                "display_name": lora_data.get("display_name", lora_name)
            })

    return jsonify({"loras": matches})

@app.route('/loras/categories', methods=['POST'])
def add_lora_category():
    """Add a new LoRA category."""
    data = request.json
    category_name = data.get("name", "").strip()

    if not category_name:
        return jsonify({"error": "Category name required"}), 400

    metadata = _load_lora_metadata()
    categories = metadata.get("categories", [])

    if category_name in categories:
        return jsonify({"error": "Category already exists"}), 400

    categories.append(category_name)
    metadata["categories"] = categories
    _save_lora_metadata(metadata)

    return jsonify({"status": "ok", "categories": categories})

@app.route('/loras/categories/<category_name>', methods=['DELETE'])
def delete_lora_category(category_name):
    """Delete a LoRA category (LoRAs become uncategorized)."""
    from urllib.parse import unquote
    category_name = unquote(category_name)

    metadata = _load_lora_metadata()
    categories = metadata.get("categories", [])

    if category_name not in categories:
        return jsonify({"error": "Category not found"}), 404

    categories.remove(category_name)
    metadata["categories"] = categories

    # Clear category from any LoRAs that had it
    loras = metadata.get("loras", {})
    for lora in loras.values():
        if lora.get("category") == category_name:
            lora["category"] = ""

    metadata["loras"] = loras
    _save_lora_metadata(metadata)

    return jsonify({"status": "ok", "categories": categories})

@app.route('/loras/<path:lora_name>/category', methods=['POST'])
def set_lora_category(lora_name):
    """Set a LoRA's category."""
    from urllib.parse import unquote
    lora_name = unquote(lora_name)

    data = request.json
    category = data.get("category", "")

    metadata = _load_lora_metadata()
    loras = metadata.get("loras", {})

    if lora_name not in loras:
        return jsonify({"error": "LoRA not found"}), 404

    loras[lora_name]["category"] = category
    metadata["loras"] = loras
    _save_lora_metadata(metadata)

    return jsonify({"status": "ok"})

@app.route('/loras/<path:lora_name>', methods=['DELETE'])
def delete_lora(lora_name):
    """Remove a LoRA from the gallery (doesn't delete the file, just hides it)."""
    from urllib.parse import unquote
    lora_name = unquote(lora_name)

    metadata = _load_lora_metadata()
    loras = metadata.get("loras", {})

    if lora_name not in loras:
        return jsonify({"error": "LoRA not found"}), 404

    # Remove from metadata (can be re-added by scanning)
    del loras[lora_name]
    metadata["loras"] = loras
    _save_lora_metadata(metadata)

    return jsonify({"status": "ok"})

@app.route('/loras/enabled', methods=['GET'])
def get_enabled_loras():
    """Get list of currently enabled LoRAs (for applying to generation)."""
    metadata = _load_lora_metadata()
    existing = metadata.get("loras", {})

    enabled = [
        {"name": name, "weight": data.get("weight", 1.0), "trigger": data.get("trigger", "")}
        for name, data in existing.items()
        if data.get("enabled", False) and data.get("available", True)
    ]

    return jsonify(enabled)


@app.route('/share-image', methods=['POST'])
def share_image():
    """Share an image with a partner and get their reaction."""
    data = request.json
    room_id = data.get('room_id')
    partner_id = data.get('partner_id')
    image_data = data.get('image_data')  # Base64 encoded
    image_type = data.get('image_type', 'image/png')
    image_path = data.get('image_path')  # Optional - for tracking shares from lightbox
    user_message = data.get('message', 'What do you think of this image?')

    if not all([room_id, partner_id, image_data]):
        return jsonify({'error': 'Missing required fields'}), 400

    room = data_store.get_room(room_id)
    partner = data_store.get_partner(partner_id)

    if not room or not partner:
        return jsonify({'error': 'Room or partner not found'}), 404

    try:
        # Build conversation history for context
        messages = []
        for m in room.messages[-20:]:  # Last 20 messages for context
            role = "user" if m.speaker_id == "user" else "assistant"
            messages.append({"role": role, "content": m.content})

        # Add the image share as the final message
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_type,
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": user_message
                }
            ]
        })

        # Get full partner context (only present characters to prevent knowledge leaks)
        all_partners = data_store.get_partners()
        room_partners = room.get_partners_in_room(all_partners)
        present_ids = room.present_character_ids or []
        present_partners = [p for p in room_partners if p.id in present_ids]
        system = partner.get_full_context(
            present_partners,
            settings.user_name,
            settings.global_system_prompt,
            user_physical_description=settings.user_physical_description
        )

        # Add image-specific instruction
        system += f"""

{settings.user_name} is sharing an image with you.
Look at it and respond naturally, in character. Comment on what you see,
how it makes you feel, or whatever reaction feels authentic to your character.
You have the full conversation context - this is a continuation of your ongoing conversation."""

        response_text = run_async(generate_response_async(partner, messages, system))

        # Track the share if image_path was provided
        if image_path:
            _mark_image_shared(image_path, partner_id)

        # Save as message in room
        response_message = Message(
            id=str(uuid.uuid4())[:8],
            speaker_id=partner.id,
            speaker_name=partner.name,
            content=response_text,
            room_id=room_id,
        )
        data_store.add_message(room_id, response_message)

        return jsonify({
            'response': {
                'id': response_message.id,
                'speaker_id': partner.id,
                'speaker_name': partner.name,
                'avatar': partner.avatar,
                'avatar_image': partner.avatar_image,
                'content': response_text,
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/share-image-to-room', methods=['POST'])
def share_image_to_room():
    """Share an image to a room's transcript. Partners see it on their next turn."""
    data = request.json
    room_id = data.get('room_id')
    image_path = data.get('image_path')
    message = data.get('message', '')

    if not room_id or not image_path:
        return jsonify({'error': 'Missing room_id or image_path'}), 400

    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    # Create a message with the image attached
    user_message = Message(
        id=str(uuid.uuid4())[:8],
        speaker_id='user',
        speaker_name=settings.user_name,
        content=message if message else '(shared an image)',
        room_id=room_id,
        image_path=image_path,
    )
    data_store.add_message(room_id, user_message)

    # Mark as pending for this room - all partners will see on next turn
    _add_pending_image(room_id, image_path)

    return jsonify({
        'success': True,
        'message': {
            'id': user_message.id,
            'speaker_id': 'user',
            'speaker_name': settings.user_name,
            'content': user_message.content,
            'image_path': image_path,
        }
    })


@app.route('/command', methods=['POST'])
def handle_command():
    """Handle slash commands - these don't affect the transcript."""
    data = request.json
    command = data.get('command', '').strip().lower()
    room_id = data.get('room_id')
    partner_id = data.get('partner_id')  # Optional - for /portrait
    captured_messages = data.get('captured_messages')  # Messages captured at click time

    room = data_store.get_room(room_id) if room_id else None

    if command == 'help':
        return jsonify({
            'type': 'info',
            'message': '''Available commands:
/selfie [name] - Generate a selfie portrait
/selfie-share [name] - Generate AND show the selfie to them (they react!)
/share - Share an image with the partner (use the button)
/portrait [name] - Same as /selfie
/gallery [name] - View all generated images for a partner
/scene - Generate a scene image from the conversation
/dm <question> - Ask the DM (everyone sees) e.g. "/dm is that my sister?"
/continue [destination] - Scene transition - skip to the next scene
/inventory - Check what you're carrying
/memory - View current memory state for this partner
/consolidate - Force memory consolidation now
/clear - Clear the room's messages
/help - Show this help

Toolbar buttons:
🗳️ Poll the Room - Ask everyone a question, get their votes
📖 Story So Far - Catch up on what's happened (for you only)
💥 Inciting Incident - Generate a dramatic story event
🎲 Private DM Terminal - Ask the DM privately (only you see)'''
        })

    elif command == 'inventory' or command == 'inv' or command == 'i':
        # Quick inventory check
        if not room:
            return jsonify({'type': 'error', 'message': 'Room not found'})

        player_id = f"player_{room_id}"
        player_name = room.player_character_name or settings.user_name

        # Get or create player inventory
        player_inv = inventory_tracker.get_or_create_inventory(player_id, player_name, owner_type='player')

        if not player_inv.items and not player_inv.currency:
            return jsonify({
                'type': 'info',
                'message': f"**{player_name}'s Inventory**\n\nYour pack is empty. Items will appear here as you acquire them in the story."
            })

        # Format inventory nicely
        lines = [f"**{player_name}'s Inventory**\n"]

        # Currency
        if player_inv.currency:
            currency_str = ", ".join(f"{v:.0f} {k}" for k, v in player_inv.currency.items() if v > 0)
            if currency_str:
                lines.append(f"💰 **Currency:** {currency_str}\n")

        # Group items by category
        category_icons = {
            'weapon': '⚔️', 'armor': '🛡️', 'consumable': '🧪', 'tool': '🔧',
            'key_item': '🔑', 'treasure': '💰', 'clothing': '👕', 'container': '📦', 'misc': '📦'
        }

        categories = {}
        for item in player_inv.items:
            cat = item.category.value
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(item)

        for cat, items in categories.items():
            icon = category_icons.get(cat, '📦')
            lines.append(f"{icon} **{cat.replace('_', ' ').title()}:**")
            for item in items:
                qty = f" x{item.quantity}" if item.quantity > 1 else ""
                equipped = " [EQUIPPED]" if item.is_equipped else ""
                weight_str = f" ({item.weight:.1f} lbs)" if item.weight > 0 else ""
                lines.append(f"  • {item.name}{qty}{weight_str}{equipped}")
            lines.append("")

        # Show total weight and load status
        total_weight = player_inv.get_total_weight()
        # Find container capacity (backpack etc)
        container = next((i for i in player_inv.items if i.category.value == 'container'), None)
        if container and container.capacity > 0:
            capacity = container.capacity
            load_percent = (total_weight / capacity) * 100 if capacity > 0 else 0
            if load_percent < 50:
                load_status = "Light"
            elif load_percent < 70:
                load_status = "Normal"
            elif load_percent < 85:
                load_status = "Heavy"
            else:
                load_status = "Overburdened"
            lines.append(f"⚖️ **Load:** {total_weight:.1f} / {capacity:.1f} lbs ({load_status})")
        else:
            lines.append(f"⚖️ **Total Weight:** {total_weight:.1f} lbs")

        # Add world status section (time, weather)
        lines.append("\n**— World Status —**")
        world_state = story_daemon.get_world_state(room_id) if story_daemon else None
        if world_state:
            # Time of day with icon
            time_icons = {"dawn": "🌅", "day": "☀️", "dusk": "🌆", "night": "🌙"}
            time_icon = time_icons.get(world_state.time_of_day, "🕐")
            # Format hour nicely
            hour = world_state.game_hour
            hour_12 = hour if hour <= 12 else hour - 12
            if hour_12 == 0:
                hour_12 = 12
            am_pm = "AM" if hour < 12 else "PM"
            lines.append(f"{time_icon} **Time:** {world_state.time_of_day.title()} ({hour_12}:00 {am_pm}, Day {world_state.game_day})")

            # Weather with icon
            weather_icons = {"clear": "☀️", "cloudy": "☁️", "rain": "🌧️", "storm": "⛈️", "fog": "🌫️", "snow": "❄️"}
            weather_icon = weather_icons.get(world_state.weather, "🌤️")
            lines.append(f"{weather_icon} **Weather:** {world_state.weather.title()}")
        else:
            lines.append("🕐 Time/weather not tracked for this room")

        # Add character condition section (hunger, thirst, fatigue)
        lines.append("\n**— Character Status —**")
        player_condition = condition_tracker.get(player_id)
        if player_condition:
            # Hunger with icon
            hunger_icons = {"satisfied": "😋", "peckish": "🙂", "hungry": "😐", "famished": "😫", "starving": "💀"}
            hunger_icon = hunger_icons.get(player_condition.hunger.value, "🍽️")
            lines.append(f"{hunger_icon} **Hunger:** {player_condition.hunger.value.title()}")

            # Thirst with icon
            thirst_icons = {"hydrated": "💧", "thirsty": "😐", "parched": "😫", "desperate": "💀"}
            thirst_icon = thirst_icons.get(player_condition.thirst.value, "💧")
            lines.append(f"{thirst_icon} **Thirst:** {player_condition.thirst.value.title()}")

            # Overall condition if not healthy
            if player_condition.condition.value != "healthy":
                lines.append(f"🩹 **Condition:** {player_condition.condition.value.title()}")
        else:
            lines.append("🍽️ **Hunger:** Satisfied")
            lines.append("💧 **Thirst:** Hydrated")

        # Fatigue from fatigue tracker
        from fatigue import get_fatigue_tracker
        fatigue_tracker = get_fatigue_tracker()
        player_fatigue = fatigue_tracker.get_fatigue(player_id)
        if player_fatigue:
            fatigue_icons = {"rested": "✨", "fine": "😊", "tired": "😐", "exhausted": "😫", "depleted": "😵", "critical": "💀"}
            fatigue_icon = fatigue_icons.get(player_fatigue.fatigue_level.value, "😊")
            lines.append(f"{fatigue_icon} **Fatigue:** {player_fatigue.fatigue_level.value.title()} ({player_fatigue.hours_awake:.1f}h awake)")
        else:
            lines.append("✨ **Fatigue:** Rested")

        return jsonify({
            'type': 'info',
            'message': "\n".join(lines)
        })

    elif command == 'memory':
        # View current memory state
        if not room or not room.partner_id:
            return jsonify({'type': 'error', 'message': 'Use /memory in a private room'})

        partner = data_store.get_partner(room.partner_id)
        if not partner:
            return jsonify({'type': 'error', 'message': 'Partner not found'})

        if partner.memory_mode == "none":
            return jsonify({'type': 'info', 'message': f'{partner.name} has memory disabled (mode: none)'})

        memory = memory_store.get_memory(partner.id, room_id, partner.memory_mode)

        parts = [f"**{partner.name}'s Memory** (mode: {partner.memory_mode})"]

        if memory.texture:
            parts.append(f"\n**Texture:**\n{memory.texture}")

        if memory.anchors:
            anchor_list = "\n".join([f"- {a.fact} [{a.weight}]" for a in memory.anchors])
            parts.append(f"\n**Anchors:**\n{anchor_list}")

        if memory.resonance:
            top = sorted(memory.resonance.items(), key=lambda x: -x[1])[:5]
            res_list = ", ".join([f"{k} ({v})" for k, v in top])
            parts.append(f"\n**Resonance:** {res_list}")

        if memory.sediment:
            parts.append(f"\n**Sediment:** {' | '.join(memory.sediment[-3:])}")

        parts.append(f"\n\n_Turns since consolidation: {memory.turn_count}_")

        return jsonify({'type': 'info', 'message': "\n".join(parts)})

    elif command == 'consolidate':
        # Force memory consolidation
        if not room or not room.partner_id:
            return jsonify({'type': 'error', 'message': 'Use /consolidate in a private room'})

        partner = data_store.get_partner(room.partner_id)
        if not partner:
            return jsonify({'type': 'error', 'message': 'Partner not found'})

        if partner.memory_mode == "none":
            return jsonify({'type': 'error', 'message': f'{partner.name} has memory disabled'})

        # Trigger consolidation in background
        _executor.submit(
            _consolidation_worker,
            partner.id,
            partner.name,
            partner.character_description,
            room_id,
            partner.memory_mode,
            [{'role': 'user' if m.speaker_id == 'user' else 'assistant',
              'content': m.content,
              'speaker_name': m.speaker_name} for m in room.messages[-50:]]
        )

        return jsonify({
            'type': 'info',
            'message': f'Consolidating memories for {partner.name}...',
            'consolidating': True
        })

    elif command == 'clear':
        if room_id:
            data_store.clear_room(room_id)

            # Optionally clear images too
            clear_images = data.get('clear_images', False)
            if clear_images and room:
                from pathlib import Path
                import shutil

                base_dir = Path.home() / ".roundtable"
                scenes_dir = base_dir / "scenes"

                # Clear partner selfies for this room's partner(s)
                partner_ids = []
                if room.partner_id:
                    partner_ids = [room.partner_id]
                elif room.partner_ids:
                    partner_ids = room.partner_ids

                avatars_dir = base_dir / "avatars"
                for pid in partner_ids:
                    partner_dir = avatars_dir / pid
                    if partner_dir.exists():
                        # Delete all selfies but keep avatar.png and background.png
                        for img in partner_dir.glob("selfie_*.png"):
                            img.unlink()

                # Clear scenes for this room
                if scenes_dir.exists():
                    for img in scenes_dir.glob(f"scene_{room_id}_*.png"):
                        img.unlink()

                print(f"\033[38;5;206m[CLEAR] Deleted images for room {room_id}\033[0m")

        return jsonify({'type': 'action', 'action': 'clear'})

    elif command == 'continue' or command.startswith('continue '):
        # Scene transition - skip mundane travel, land at the next scene
        if not room_id:
            return jsonify({'type': 'error', 'message': 'Must be in a room'})

        # Extract optional destination hint
        destination_hint = ""
        if command.startswith('continue '):
            destination_hint = command[9:].strip()

        # Call the transition function directly (via Flask test client to reuse the endpoint logic)
        with app.test_client() as client:
            response = client.post(
                f'/rooms/{room_id}/narrate-transition',
                json={'destination': destination_hint},
                content_type='application/json'
            )
            result = response.get_json()

            if response.status_code == 200:
                return jsonify({
                    'type': 'transition',
                    'content': result.get('content', ''),
                    'destination': result.get('destination', ''),
                    'transition_type': result.get('transition_type', 'smooth'),
                })
            else:
                # Handle specific failure types with helpful messages
                error_type = result.get('error', 'unknown')
                if error_type == 'unclear_destination':
                    return jsonify({
                        'type': 'transition_failed',
                        'failure_type': 'unclear',
                        'message': result.get('message', "Where are you trying to go? Try `/continue the docks` or similar."),
                    })
                elif error_type == 'blocked_transition':
                    return jsonify({
                        'type': 'transition_failed',
                        'failure_type': 'blocked',
                        'message': result.get('message', "The situation is too tense for a smooth transition. Handle the current moment first."),
                    })
                else:
                    return jsonify({'type': 'error', 'message': result.get('error', 'Transition failed')})

    elif command.startswith('gallery'):
        # /gallery or /gallery <name>
        parts = command.split(maxsplit=1)
        target_name = parts[1] if len(parts) > 1 else None

        # Find the partner
        partner = None
        if target_name:
            for p in data_store.get_partners():
                if p.name.lower() == target_name.lower():
                    partner = p
                    break
            if not partner:
                return jsonify({'type': 'error', 'message': f"Partner '{target_name}' not found"})
        elif room and room.partner_id:
            partner = data_store.get_partner(room.partner_id)
        else:
            return jsonify({'type': 'error', 'message': 'Specify a partner: /gallery <name>'})

        if not partner:
            return jsonify({'type': 'error', 'message': 'Partner not found'})

        # Return gallery action so frontend can handle it
        return jsonify({
            'type': 'gallery',
            'partner_id': partner.id,
            'partner_name': partner.name,
        })

    elif command.startswith('selfie-share'):
        # Generate selfie AND show it to the partner for their reaction
        parts = command.split(maxsplit=1)
        target_name = parts[1] if len(parts) > 1 else None
        user_message = data.get('user_message')  # Optional message from user

        # Find the partner
        partner = None
        if target_name:
            for p in data_store.get_partners():
                if p.name.lower() == target_name.lower():
                    partner = p
                    break
            if not partner:
                return jsonify({'type': 'error', 'message': f"Partner '{target_name}' not found"})
        elif room and room.partner_id:
            partner = data_store.get_partner(room.partner_id)
        else:
            return jsonify({'type': 'error', 'message': 'Specify a partner: /selfie-share <name>'})

        if not partner:
            return jsonify({'type': 'error', 'message': 'Partner not found'})

        try:
            import base64

            # If user included a message, save it to the room first
            if user_message and room:
                user_msg = Message(
                    id=str(uuid.uuid4())[:8],
                    speaker_id='user',
                    speaker_name=settings.user_name or 'You',
                    content=user_message,
                    room_id=room_id,
                )
                data_store.add_message(room_id, user_msg)

            # Step 1: Generate self-description
            messages = []
            if captured_messages:
                for m in captured_messages:
                    role = "user" if m.get('is_user') else "assistant"
                    messages.append({"role": role, "content": m.get('content', '')})
            elif room and room.messages:
                for m in room.messages[-10:]:  # Include more context
                    role = "user" if m.speaker_id == "user" else "assistant"
                    messages.append({"role": role, "content": m.content})

            messages.append({
                "role": "user",
                "content": "Describe your appearance right now, as if for a portrait artist."
            })

            physical_context = ""
            if partner.physical_description:
                physical_context = f"\nYOUR ESTABLISHED APPEARANCE:\n{partner.physical_description}\n"

            system_prompt = f"""{partner.get_character()}
You are {partner.name}. Stay in character.
{physical_context}
Describe yourself vividly for a portrait. Under 100 words, comma-separated phrases."""

            self_description = run_async(generate_response_async(partner, messages, system_prompt))

            # Step 2: Generate the image
            image_path = None
            try:
                from image_gen import get_generator
                generator = get_generator()

                if generator and generator.is_available():
                    # Prepend gender if specified for image consistency
                    image_prompt = self_description
                    if partner.gender:
                        image_prompt = f"{partner.gender}, {self_description}"
                    # Get system prompt prefix for filename
                    sys_prompt_prefix = (partner.custom_system_prompt or '')[:15] if partner.custom_system_prompt else None
                    image_paths = generator.generate_avatar(
                        prompt=image_prompt,
                        partner_id=partner.id,
                        count=1,
                        partner_loras=partner.loras,
                        partner_name=partner.name,
                        model_name=partner.model,
                        system_prompt_prefix=sys_prompt_prefix,
                        room_id=room_id
                    )
                    if image_paths:
                        image_path = image_paths[0]
            except Exception as img_err:
                pass  # Continue without image

            # Step 3: Show the image to the partner and get their reaction
            reaction = None
            if image_path and image_path.exists():
                # Read and encode the image
                image_data = base64.b64encode(image_path.read_bytes()).decode('utf-8')

                # Build the reaction prompt - include user's message if provided
                if user_message:
                    reaction_text = f"""{settings.user_name} said: "{user_message}"

They're also sharing this portrait that was just created of you based on your self-description: "{self_description}"

Respond to what they said, and also react to seeing this image of yourself. Stay in character as {partner.name}."""
                else:
                    reaction_text = f"""This is a portrait that was just created of you based on your self-description: "{self_description}"

React to seeing this image of yourself. Stay in character as {partner.name}."""

                # Build message with image for the AI to see
                reaction_messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": reaction_text
                        }
                    ]
                }]

                # Get full partner context for richer reaction (only present characters)
                all_partners = data_store.get_partners()
                room_partners = room.get_partners_in_room(all_partners) if room else [partner]
                present_ids = room.present_character_ids or [] if room else []
                present_partners = [p for p in room_partners if p.id in present_ids] if present_ids else room_partners
                reaction_system = partner.get_full_context(
                    present_partners,
                    settings.user_name,
                    settings.global_system_prompt,
                    user_physical_description=settings.user_physical_description
                )
                reaction_system += f"""

You're looking at a portrait of yourself that was just generated.
React naturally and in character. You might comment on the likeness, the artistic style,
what you like or don't like about it, or how it makes you feel to see yourself depicted this way."""

                reaction = run_async(generate_response_async(partner, reaction_messages, reaction_system))

                # Save the reaction as a message in the room
                if reaction and room:
                    reaction_message = Message(
                        id=str(uuid.uuid4())[:8],
                        speaker_id=partner.id,
                        speaker_name=partner.name,
                        content=reaction,
                        room_id=room_id,
                    )
                    data_store.add_message(room_id, reaction_message)

            return jsonify({
                'type': 'selfie-share',
                'partner_id': partner.id,
                'partner_name': partner.name,
                'description': self_description,
                'image': str(image_path) if image_path else None,
                'reaction': reaction,
                'avatar': partner.avatar,
                'avatar_image': partner.avatar_image,
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'type': 'error', 'message': str(e)})

    elif command.startswith('selfie') or command.startswith('portrait'):
        # /selfie or /selfie <name> or /portrait <name>
        # Now runs in background - returns job_id immediately
        parts = command.split(maxsplit=1)
        target_name = parts[1] if len(parts) > 1 else None

        # Find the partner
        partner = None
        if target_name:
            for p in data_store.get_partners():
                if p.name.lower() == target_name.lower():
                    partner = p
                    break
            if not partner:
                return jsonify({'type': 'error', 'message': f"Partner '{target_name}' not found"})
        elif room and room.partner_id:
            partner = data_store.get_partner(room.partner_id)
        elif partner_id:
            partner = data_store.get_partner(partner_id)
        else:
            return jsonify({'type': 'error', 'message': 'Specify a partner: /selfie <name>'})

        if not partner:
            return jsonify({'type': 'error', 'message': 'Partner not found'})

        # Create background job and return immediately
        # Check for override prompt (for regeneration)
        override_prompt = data.get('prompt')

        # Use captured_loras from request if provided (for regeneration), else use room settings
        captured_loras = data.get('captured_loras')
        if captured_loras is None and room and room.loras:
            captured_loras = [l for l in room.loras if l.get('enabled')]

        job_id = create_job('selfie')
        _executor.submit(_selfie_worker, job_id, partner.id, captured_messages or [], room_id, override_prompt, captured_loras)

        return jsonify({
            'type': 'job_started',
            'job_id': job_id,
            'job_type': 'selfie',
            'partner_id': partner.id,
            'partner_name': partner.name
        })

    elif command.startswith('scene'):
        if not room:
            return jsonify({'type': 'error', 'message': 'No room found'})

        # Find a partner to generate the scene description (prefer room's partner)
        scene_partner = None
        if room.partner_id:
            scene_partner = data_store.get_partner(room.partner_id)
        elif room.partner_ids:
            scene_partner = data_store.get_partner(room.partner_ids[0])
        else:
            # Common room - pick any partner
            all_p = data_store.get_partners()
            if all_p:
                scene_partner = all_p[0]

        if not scene_partner:
            return jsonify({'type': 'error', 'message': 'No partner available to describe the scene'})

        try:
            from image_gen import get_generator

            # Check for override prompt (for regeneration)
            override_prompt = data.get('prompt')

            # Verify ComfyUI is reachable BEFORE spending an API call building the scene
            # prompt. Otherwise we'd generate (and pay for) a description only to find
            # ComfyUI offline. On a regeneration we already have the prompt, so echo it back.
            generator = get_generator()
            if not generator or not generator.is_available():
                msg = "ComfyUI not running."
                if override_prompt:
                    msg = f"ComfyUI not running. Scene prompt:\n\n{override_prompt}"
                return jsonify({'type': 'info', 'message': msg})

            if override_prompt:
                # Use provided prompt directly
                scene_prompt = override_prompt
            elif room.messages and len(room.messages) > 0:
                # Build scene description from conversation
                recent_messages = [
                    {'speaker': m.speaker_name, 'content': m.content}
                    for m in room.messages[-7:]
                ]
                conversation_text = "\n".join(f"{m['speaker']}: {m['content']}" for m in recent_messages)

                scene_system = f"""{scene_partner.get_character()}

You are a cinematographer describing a scene for a painter.
Given a conversation, visualize the SCENE - not portraits, but the moment itself.
Where are they? What's the atmosphere? What would a camera capture?

Output ONLY the image description. Under 100 words.
Use comma-separated descriptive phrases.
Include: setting, mood, lighting, composition.
Be specific and cinematic. Stay in character - describe it as YOU see it."""

                scene_messages = [{
                    "role": "user",
                    "content": f"Setting: {room.scenario or 'A conversation space'}\n\nRecent conversation:\n{conversation_text}\n\nDescribe this scene for an artist to paint:"
                }]

                scene_prompt = run_async(generate_response_async(scene_partner, scene_messages, scene_system))
            else:
                # No conversation yet - ask the character to imagine their scene
                scene_system = f"""{scene_partner.get_character()}

You are a cinematographer describing a scene for a painter.
Imagine where your character would be right now. What does their world look like?
Think about their background, their circumstances, their typical environment.

Output ONLY the scene description. Under 100 words.
Use comma-separated descriptive phrases.
Include: setting, mood, lighting, atmosphere, environment details.
Be specific and cinematic. Stay in character - describe where YOU are."""

                scene_messages = [{
                    "role": "user",
                    "content": f"Imagine where you are right now. Describe the scene around you - the environment, the atmosphere, what a camera would capture. Where are you? What does it look like?"
                }]

                scene_prompt = run_async(generate_response_async(scene_partner, scene_messages, scene_system))

            # Try to generate the image (ComfyUI availability was verified above)
            try:
                # Use captured_loras from request if provided (for regeneration), else use room settings
                captured_loras = data.get('captured_loras')
                if captured_loras is None and room and room.loras:
                    captured_loras = [l for l in room.loras if l.get('enabled')]

                # Generate scene (landscape - swaps preset width/height)
                image_path = generator.generate_scene(
                    prompt=scene_prompt,
                    room_id=room_id,
                    captured_loras=captured_loras
                )

                # Copy scene to each participant's gallery
                if room and room.partner_ids:
                    from pathlib import Path
                    generator.copy_scene_to_galleries(Path(image_path), room.partner_ids)

                return jsonify({
                    'type': 'scene',
                    'prompt': scene_prompt,
                    'image': str(image_path)
                })

            except Exception as e:
                return jsonify({
                    'type': 'info',
                    'message': f"Image generation failed: {e}\n\nScene prompt:\n\n{scene_prompt}"
                })

        except Exception as e:
            return jsonify({'type': 'error', 'message': str(e)})

    elif command == 'group-photo':
        # Group photo with selectable participants
        # Expects: partner_ids (list), include_user (bool), prompt_model_id (str)
        gp_partner_ids = data.get('partner_ids', [])
        gp_include_user = data.get('include_user', False)
        gp_prompt_model_id = data.get('prompt_model_id')

        if not gp_partner_ids and not gp_include_user:
            return jsonify({'type': 'error', 'message': 'Select at least one participant'})

        if not gp_prompt_model_id:
            # Default to first partner if not specified
            if gp_partner_ids:
                gp_prompt_model_id = gp_partner_ids[0]
            else:
                return jsonify({'type': 'error', 'message': 'Select a model to generate the prompt'})

        # Capture room's enabled LoRAs at queue time
        captured_loras = None
        if room and room.loras:
            captured_loras = [l for l in room.loras if l.get('enabled')]

        # Create background job and return immediately
        job_id = create_job('group_photo')
        _executor.submit(
            _group_photo_worker,
            job_id,
            room_id,
            gp_partner_ids,
            gp_include_user,
            gp_prompt_model_id,
            captured_messages or [],
            captured_loras
        )

        return jsonify({
            'type': 'job_started',
            'job_id': job_id,
            'job_type': 'group_photo'
        })

    else:
        return jsonify({'type': 'error', 'message': f"Unknown command: /{command}"})


# ============================================================================
# Mood Detection - Track emotional state of the story
# ============================================================================

def detect_room_mood(room: Room, recent_count: int = 10) -> dict:
    """Analyze recent messages to detect the current emotional mood of the scene."""
    if not room.messages:
        return {'mood': 'neutral', 'intensity': 'low', 'notes': ''}

    recent = room.messages[-recent_count:]
    content = ' '.join([m.content.lower() for m in recent])

    # Check for death/loss
    death_words = ['died', 'dead', 'killed', 'death', 'lost', 'gone', 'murdered', 'corpse', 'body']
    has_death = any(w in content for w in death_words)

    # Check for tension/danger
    danger_words = ['danger', 'threat', 'attack', 'weapon', 'blood', 'hurt', 'scared', 'afraid', 'run', 'hide', 'fight']
    has_danger = sum(1 for w in danger_words if w in content)

    # Check for comedy/lightness
    comedy_words = ['laugh', 'joke', 'funny', 'haha', 'lol', 'silly', 'ridiculous', 'grin', 'chuckle']
    has_comedy = sum(1 for w in comedy_words if w in content)

    # Check for emotional intensity
    emotion_words = ['love', 'hate', 'desperate', 'betrayed', 'trust', 'secret', 'confession', 'truth']
    has_emotion = sum(1 for w in emotion_words if w in content)

    # Check for recent inciting incidents
    recent_incidents = [m for m in recent if getattr(m, 'message_type', '') == 'inciting_incident']

    # Determine mood
    if has_death:
        mood = 'somber'
        intensity = 'high'
        notes = 'Someone has died or been lost. The mood is heavy.'
    elif has_danger >= 3 or recent_incidents:
        mood = 'tense'
        intensity = 'high'
        notes = 'Danger is present. Everyone is on edge.'
    elif has_comedy >= 2:
        mood = 'light'
        intensity = 'medium'
        notes = 'The mood has lightened. There may be humor despite the circumstances.'
    elif has_emotion >= 2:
        mood = 'emotional'
        intensity = 'high'
        notes = 'Deep feelings are in play. This is a moment of truth.'
    elif has_danger >= 1:
        mood = 'uneasy'
        intensity = 'medium'
        notes = 'Something feels off. Tension simmers beneath the surface.'
    else:
        mood = 'neutral'
        intensity = 'low'
        notes = ''

    return {'mood': mood, 'intensity': intensity, 'notes': notes}


def build_mood_context(room: Room) -> str:
    """Build mood/atmosphere context to inject into character prompts."""
    parts = []

    # Genre emotional texture
    if room.genre:
        genre_vibes = {
            'zombie': "The world has ended. Trust is scarce. Anyone could turn. Survival trumps morality.",
            'dystopia': "Everything is controlled. Smiles hide fear. Speaking freely could get you killed.",
            'noir': "The city is rotten. Everyone has secrets. Money and power corrupt everything.",
            'horror': "Something is wrong. The darkness hides things. Fear is the only rational response.",
            'comedy': "Even in dark times, absurdity wins. Find the humor. Life is ridiculous.",
            'drama': "Emotions run deep. Every word carries weight. Relationships are everything.",
            'thriller': "Time is running out. Stakes are life and death. Trust no one completely.",
            'fantasy': "Magic and wonder exist. But so do ancient evils. Heroes are forged in crisis.",
            'scifi': "Technology defines everything. The future is uncertain. Humanity is tested.",
            'western': "Law is what you make it. The frontier is unforgiving. Reputation is everything.",
        }
        vibe = genre_vibes.get(room.genre.lower(), "")
        if vibe:
            parts.append(f"WORLD TEXTURE: {vibe}")

    # Factions
    if room.factions:
        parts.append(f"FACTIONS/POWER STRUCTURES: {room.factions}")

    # Current mood from recent events
    mood_info = detect_room_mood(room)
    if mood_info['notes']:
        parts.append(f"CURRENT MOOD: {mood_info['notes']}")

    if parts:
        return "\n---\n" + "\n".join(parts) + "\n---"
    return ""


# ============================================================================
# Fatigue System - Sleep, rest, and exhaustion tracking
# ============================================================================

@app.route('/fatigue/<character_id>', methods=['GET'])
def get_fatigue(character_id):
    """Get fatigue state for a character."""
    fatigue = fatigue_tracker.get_fatigue(character_id)
    if not fatigue:
        return jsonify({'error': 'Character not found in fatigue tracker'}), 404
    return jsonify(fatigue.to_dict())


@app.route('/fatigue/<character_id>', methods=['POST'])
def create_fatigue(character_id):
    """Create or update fatigue tracking for a character."""
    data = request.json or {}
    character_name = data.get('character_name', character_id)

    fatigue = fatigue_tracker.get_or_create(character_id, character_name)
    return jsonify(fatigue.to_dict())


@app.route('/fatigue/<character_id>/advance', methods=['POST'])
def advance_fatigue(character_id):
    """Advance time for a character (accumulates fatigue if awake, recovers if resting)."""
    data = request.json or {}
    hours = data.get('hours', 1.0)

    fatigue = fatigue_tracker.get_fatigue(character_id)
    if not fatigue:
        return jsonify({'error': 'Character not found. Create fatigue tracking first.'}), 404

    fatigue_tracker.advance_time(character_id, hours)

    # Return updated state and effects
    updated = fatigue_tracker.get_fatigue(character_id)
    effects = fatigue_tracker.get_fatigue_effects(character_id)

    return jsonify({
        'fatigue': updated.to_dict(),
        'effects': effects,
        'context': fatigue_tracker.get_fatigue_context(character_id)
    })


@app.route('/fatigue/<character_id>/sleep', methods=['POST'])
def start_fatigue_sleep(character_id):
    """Character starts sleeping."""
    data = request.json or {}
    game_time_iso = data.get('game_time')
    character_name = data.get('character_name')

    fatigue_tracker.start_sleep(character_id, game_time_iso, character_name)

    fatigue = fatigue_tracker.get_fatigue(character_id)
    return jsonify({
        'fatigue': fatigue.to_dict(),
        'message': f'{fatigue.character_name} begins sleeping.'
    })


@app.route('/fatigue/<character_id>/rest', methods=['POST'])
def start_fatigue_rest(character_id):
    """Character starts resting (not full sleep)."""
    data = request.json or {}
    character_name = data.get('character_name')

    fatigue_tracker.start_rest(character_id, character_name)

    fatigue = fatigue_tracker.get_fatigue(character_id)
    return jsonify({
        'fatigue': fatigue.to_dict(),
        'message': f'{fatigue.character_name} begins resting.'
    })


@app.route('/fatigue/<character_id>/wake', methods=['POST'])
def wake_fatigue(character_id):
    """Character wakes up / stops resting."""
    fatigue_tracker.wake_up(character_id)

    fatigue = fatigue_tracker.get_fatigue(character_id)
    if not fatigue:
        return jsonify({'error': 'Character not found'}), 404

    effects = fatigue_tracker.get_fatigue_effects(character_id)
    return jsonify({
        'fatigue': fatigue.to_dict(),
        'effects': effects,
        'message': f'{fatigue.character_name} wakes up. Current state: {effects["level"]} - {effects["description"]}'
    })


@app.route('/rooms/<room_id>/fatigue', methods=['GET'])
def get_room_fatigue(room_id):
    """Get fatigue status for all characters in a room."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    # Get all partner IDs in this room
    partner_ids = []
    if room.partner_id:
        partner_ids = [room.partner_id]
    elif room.partner_ids:
        partner_ids = room.partner_ids

    # Also include player character
    player_id = f"player_{room_id}"

    # Build fatigue status for all
    fatigue_status = {}

    # Player fatigue
    player_fatigue = fatigue_tracker.get_fatigue(player_id)
    if player_fatigue:
        fatigue_status[player_id] = {
            'fatigue': player_fatigue.to_dict(),
            'effects': fatigue_tracker.get_fatigue_effects(player_id)
        }

    # Partner fatigue
    for pid in partner_ids:
        partner_fatigue = fatigue_tracker.get_fatigue(pid)
        if partner_fatigue:
            fatigue_status[pid] = {
                'fatigue': partner_fatigue.to_dict(),
                'effects': fatigue_tracker.get_fatigue_effects(pid)
            }

    return jsonify({
        'room_id': room_id,
        'characters': fatigue_status,
        'summary': fatigue_tracker.get_all_fatigue_context()
    })


@app.route('/rooms/<room_id>/narrator/status', methods=['GET'])
def get_narrator_status(room_id):
    """
    Debug endpoint - get narrator interjection status for a room.
    Shows turn count, next interjection target, and whether conditions are met.
    """
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    narrator = get_dm_narrator()
    status = narrator.get_status(room_id)

    # Check the condition that enables interjections
    has_player_name = bool(room.player_character_name)
    has_relationships = bool(hasattr(room, 'character_relationships') and room.character_relationships)
    interjections_enabled = has_player_name or has_relationships

    return jsonify({
        'room_id': room_id,
        'turn_count': status['turn_count'],
        'next_interjection_at': status['next_at'],
        'turns_until_next': status['turns_until'],
        'last_interjection_type': status['last_type'],
        'consecutive_textures': status['consecutive_textures'],
        'interjections_enabled': interjections_enabled,
        'debug': {
            'has_player_character_name': has_player_name,
            'player_character_name': room.player_character_name if has_player_name else None,
            'has_character_relationships': has_relationships,
        }
    })


@app.route('/rooms/<room_id>/rest/narrate', methods=['POST'])
def narrate_rest(room_id):
    """
    DM-narrated rest action. Instead of just updating fatigue status,
    this asks the DM to evaluate the context, identify all sleeping characters,
    update their fatigue, and generate a narration to morning (or from nap).

    Body: { "rest_type": "nap" | "long" }
    Returns: { "narration": "...", "characters_rested": [...], "success": true }
    """
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    data = request.json or {}
    rest_type = data.get('rest_type', 'long')
    hours = 2 if rest_type == 'nap' else 8

    # Get all partners in room
    partners = data_store.get_partners()
    room_partners = room.get_partners_in_room(partners)

    # Build context for DM
    dm_context = _build_simple_dm_context(room, room_partners)
    recent_messages = room.messages[-15:] if room.messages else []
    conversation = "\n".join([
        f"{m.speaker_name}: {m.content[:300]}" for m in recent_messages
    ])

    # Player character name
    player_name = room.player_character_name or settings.user_name or "Player"

    # Get all character names in the room
    all_characters = [player_name] + [p.name for p in room_partners]

    # Build the prompt
    if rest_type == 'nap':
        rest_desc = "a short nap (~2 hours)"
        time_desc = "After the nap"
    else:
        rest_desc = "a full night's rest (~8 hours)"
        time_desc = "Dawn arrives"

    prompt = f"""You are the Dungeon Master. The player has requested {rest_desc}.

{dm_context}

Recent conversation:
{conversation}

Characters present: {', '.join(all_characters)}

The player wants to rest. Your tasks:

1. EVALUATE: Is this a reasonable moment for rest? (Are they in immediate danger? Mid-conversation? In a safe enough spot?)

2. IDENTIFY: Based on the context, which characters would realistically be sleeping right now? Usually this is everyone present, but someone might be keeping watch.

3. NARRATE: Write a short (2-4 sentences) evocative narration transitioning through the rest period. {time_desc}, describe the new moment - the light, the atmosphere, how the characters feel upon waking. Be poetic but brief.

Respond in this exact JSON format:
{{
  "reasonable": true/false,
  "reason": "why or why not (brief, only if false)",
  "sleeping_characters": ["names of everyone who slept"],
  "narration": "Your transition narration here"
}}

If rest is NOT reasonable (immediate danger, etc.), set reasonable to false and give a brief reason. The sleeping_characters list should be empty and narration should be a brief suggestion of what should happen first."""

    # Get model
    import httpx
    model_to_use = settings.storybuilder_model
    available_models = provider_manager.get_models_for_provider('ollama')
    if model_to_use not in available_models and available_models:
        model_to_use = available_models[0]

    dm_response = _call_ollama_for_council(
        model_to_use,
        prompt,
        "You are a Dungeon Master managing rest transitions. Respond ONLY with valid JSON, no markdown."
    )

    # Parse the JSON response
    import json
    try:
        # Try to extract JSON from the response (in case model added markdown)
        json_match = dm_response
        if '```' in dm_response:
            # Extract JSON from code block
            import re
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', dm_response, re.DOTALL)
            if match:
                json_match = match.group(1)
        elif '{' in dm_response:
            # Find the JSON object
            start = dm_response.index('{')
            end = dm_response.rindex('}') + 1
            json_match = dm_response[start:end]

        result = json.loads(json_match)
    except (json.JSONDecodeError, ValueError) as e:
        # Fallback - just generate a simple narration if JSON parsing fails
        result = {
            "reasonable": True,
            "sleeping_characters": all_characters,
            "narration": dm_response if dm_response else f"The night passes quietly. {time_desc.split()[0]} light filters in, waking the group."
        }

    if not result.get('reasonable', True):
        # Rest not reasonable - return the reason
        return jsonify({
            'success': False,
            'reason': result.get('reason', 'This is not a safe moment to rest.'),
            'narration': result.get('narration', '')
        })

    # Rest is reasonable - update fatigue for all sleeping characters
    sleeping_chars = result.get('sleeping_characters', all_characters)
    characters_rested = []

    # Map character names to their IDs
    player_id = f"player_{room_id}"

    for char_name in sleeping_chars:
        # Determine the character ID
        if char_name.lower() == player_name.lower() or char_name.lower() in ['player', 'you']:
            char_id = player_id
        else:
            # Find partner by name
            partner = next((p for p in room_partners if p.name.lower() == char_name.lower()), None)
            if partner:
                char_id = partner.id
            else:
                continue  # Skip unknown characters

        # Put them to sleep, advance time, wake them up
        fatigue_tracker.start_sleep(char_id, None, char_name)
        fatigue_tracker.advance_time(char_id, hours)
        fatigue_tracker.wake_up(char_id)
        characters_rested.append(char_name)

    # Add the narration as a narrator message to the room
    narration = result.get('narration', f"The group rests. {time_desc}, everyone stirs awake.")
    narrator_msg = Message(
        id=str(uuid.uuid4())[:8],
        speaker_id='narrator',
        speaker_name='Narrator',
        content=narration,
        message_type='narration'
    )
    room.add_message(narrator_msg)
    data_store.save_room(room)

    # Advance world time if we have a story daemon
    if story_daemon:
        story_daemon.advance_time(room_id, hours)

    return jsonify({
        'success': True,
        'narration': narration,
        'characters_rested': characters_rested,
        'hours': hours
    })


# ============================================================================
# Inventory System - Item tracking for characters
# ============================================================================

@app.route('/inventory/<owner_id>', methods=['GET'])
def get_inventory(owner_id):
    """Get inventory for a character, including combat readiness info."""
    inv = inventory_tracker.get_inventory(owner_id)
    if not inv:
        return jsonify({'error': 'Inventory not found'}), 404

    result = inv.to_dict()

    # Add combat status
    combat_status = inventory_tracker.get_combat_status(owner_id)
    result['combat_ready'] = combat_status['combat_ready']
    result['equipped_weapon'] = combat_status['equipped_weapon']
    result['using_fists'] = combat_status['using_fists']

    # If no weapons exist, add a virtual "Fists" weapon option
    has_weapon = any(item.get('category') == 'weapon' for item in result['items'])
    if not has_weapon:
        result['fists_available'] = True
        # Add virtual fists to the items list for UI display
        result['items'].append({
            'id': 'fists_virtual',
            'name': 'Fists',
            'category': 'weapon',
            'description': 'Your bare hands. Always available for unarmed combat.',
            'short_description': 'Unarmed combat',
            'quantity': 1,
            'is_equipped': combat_status['using_fists'],
            'is_virtual': True,  # Flag so UI knows this isn't a real item
        })
    else:
        result['fists_available'] = False

    # Add condition info (injuries, hunger, thirst)
    condition = condition_tracker.get(owner_id)
    if condition:
        result['condition'] = {
            'status': condition.condition.value,
            'is_bleeding': condition.is_bleeding,
            'is_incapacitated': condition.is_incapacitated,
            'injuries': [i.to_dict() for i in condition.injuries],
            'hunger': condition.hunger.value,
            'thirst': condition.thirst.value,
        }
    else:
        # Default healthy state
        result['condition'] = {
            'status': 'healthy',
            'is_bleeding': False,
            'is_incapacitated': False,
            'injuries': [],
            'hunger': 'satisfied',
            'thirst': 'hydrated',
        }

    return jsonify(result)


@app.route('/inventory/<owner_id>', methods=['POST'])
def create_inventory(owner_id):
    """Create or update inventory for a character."""
    data = request.json or {}
    owner_name = data.get('owner_name', owner_id)
    owner_type = data.get('owner_type', 'player')

    inv = inventory_tracker.get_or_create_inventory(owner_id, owner_name, owner_type)
    return jsonify(inv.to_dict())


@app.route('/inventory/<owner_id>/items', methods=['POST'])
def add_inventory_item(owner_id):
    """Add an item to a character's inventory."""
    data = request.json or {}
    item_name = data.get('name')
    if not item_name:
        return jsonify({'error': 'Item name required'}), 400

    # Ensure inventory exists
    owner_name = data.get('owner_name', owner_id)
    inventory_tracker.get_or_create_inventory(owner_id, owner_name)

    # Add the item
    item = inventory_tracker.add_item(
        owner_id=owner_id,
        item_name=item_name,
        category=data.get('category', 'misc'),
        quantity=data.get('quantity', 1),
        description=data.get('description', ''),
        short_description=data.get('short_description', ''),
        stackable=data.get('stackable', False),
        weight=data.get('weight', 0),
        value=data.get('value', 0),
        is_quest_item=data.get('is_quest_item', False),
        is_consumable=data.get('is_consumable', False),
        acquired_from=data.get('acquired_from', 'manual'),
    )

    if item:
        return jsonify(item.to_dict())
    return jsonify({'error': 'Failed to add item'}), 500


@app.route('/inventory/<owner_id>/items/<item_id>', methods=['DELETE'])
def remove_inventory_item(owner_id, item_id):
    """
    Remove/drop an item from a character's inventory.

    Query params:
        quantity (int, optional): How many to drop if stackable. Default: all.
    """
    # Support both query params and JSON body for quantity
    quantity = request.args.get('quantity', type=int)
    if quantity is None:
        data = request.json or {}
        quantity = data.get('quantity')

    inv = inventory_tracker.get_inventory(owner_id)
    if not inv:
        return jsonify({'error': 'Inventory not found'}), 404

    item = inv.find_item_by_id(item_id)
    if not item:
        return jsonify({'error': 'Item not found'}), 404

    item_name = item.name
    was_equipped = item.is_equipped
    was_weapon = item.category.value == 'weapon'

    # If no quantity specified, drop all
    if quantity is None:
        quantity = item.quantity if item.stackable else 1

    result = inventory_tracker.remove_item(owner_id, item_id, quantity)
    if not result:
        return jsonify({'error': 'Failed to drop item'}), 500

    # If we dropped an equipped weapon, update combat readiness
    if was_equipped and was_weapon:
        # Check if any other weapon is still equipped
        remaining_inv = inventory_tracker.get_inventory(owner_id)
        if remaining_inv:
            other_weapons = [i for i in remaining_inv.items if i.is_equipped and i.category.value == 'weapon']
            if not other_weapons:
                inventory_tracker.set_combat_ready(owner_id, False)

    return jsonify({
        'success': True,
        'message': f'Dropped {item_name}' + (f' x{quantity}' if quantity > 1 else ''),
        'combat_ready': inventory_tracker.is_combat_ready(owner_id)
    })


@app.route('/inventory/<owner_id>/items/<item_id>/equip', methods=['POST'])
def toggle_equip_item(owner_id, item_id):
    """Toggle equipped state of an item."""
    # Special case: virtual "Fists" item
    if item_id == 'fists_virtual':
        is_ready = inventory_tracker.is_combat_ready(owner_id)
        if is_ready and inventory_tracker.get_equipped_weapon(owner_id) is None:
            # Currently using fists, unequip (lower fists)
            inventory_tracker.set_combat_ready(owner_id, False)
            return jsonify({
                'item': {
                    'id': 'fists_virtual',
                    'name': 'Fists',
                    'category': 'weapon',
                    'is_equipped': False,
                    'is_virtual': True,
                },
                'message': 'Lowered fists',
                'combat_ready': False
            })
        else:
            # Raise fists for combat
            # First unequip any actual weapons
            inventory_tracker.unequip_all_weapons(owner_id)
            inventory_tracker.set_combat_ready(owner_id, True)
            return jsonify({
                'item': {
                    'id': 'fists_virtual',
                    'name': 'Fists',
                    'category': 'weapon',
                    'is_equipped': True,
                    'is_virtual': True,
                },
                'message': 'Raised fists - ready to fight',
                'combat_ready': True
            })

    inv = inventory_tracker.get_inventory(owner_id)
    if not inv:
        return jsonify({'error': 'Inventory not found'}), 404

    item = inv.find_item_by_id(item_id)
    if not item:
        return jsonify({'error': 'Item not found'}), 404

    # If equipping a weapon, unequip any other equipped weapons first (only one weapon equipped at a time)
    if item.category.value == 'weapon' and not item.is_equipped:
        for other_item in inv.items:
            if other_item.id != item_id and other_item.category.value == 'weapon' and other_item.is_equipped:
                other_item.is_equipped = False

    item.is_equipped = not item.is_equipped
    inventory_tracker._save()

    # If equipping a weapon, set combat ready
    # If unequipping, check if any weapon still equipped
    if item.category.value == 'weapon':
        if item.is_equipped:
            inventory_tracker.set_combat_ready(owner_id, True)
        else:
            # Check if any other weapon is still equipped
            other_weapons = [i for i in inv.items if i.is_equipped and i.category.value == 'weapon' and i.id != item_id]
            if not other_weapons:
                inventory_tracker.set_combat_ready(owner_id, False)

    return jsonify({
        'item': item.to_dict(),
        'message': f'{item.name} {"equipped" if item.is_equipped else "unequipped"}',
        'combat_ready': inventory_tracker.is_combat_ready(owner_id)
    })


@app.route('/inventory/<owner_id>/items/<item_id>/category', methods=['POST'])
def change_item_category(owner_id, item_id):
    """
    Change an item's category (e.g., to 'weapon').

    POST body: { "category": "weapon" }

    Only allows reclassifying items as weapons - you can use anything as a weapon!
    """
    from inventory import ItemCategory

    data = request.json or {}
    new_category = data.get('category', 'weapon').lower()

    # Validate category
    valid_categories = [c.value for c in ItemCategory]
    if new_category not in valid_categories:
        return jsonify({'error': f'Invalid category. Must be one of: {valid_categories}'}), 400

    inv = inventory_tracker.get_inventory(owner_id)
    if not inv:
        return jsonify({'error': 'Inventory not found'}), 404

    item = inv.find_item_by_id(item_id)
    if not item:
        return jsonify({'error': 'Item not found'}), 404

    old_category = item.category.value
    item.category = ItemCategory(new_category)
    inventory_tracker._save()

    return jsonify({
        'item': item.to_dict(),
        'message': f'{item.name} reclassified from {old_category} to {new_category}',
        'old_category': old_category,
        'new_category': new_category
    })


@app.route('/inventory/<owner_id>/items/<item_id>/ammo', methods=['POST'])
def set_item_ammo_level(owner_id, item_id):
    """
    Set the ammo level for a weapon (fuzzy tracking, not exact counts).

    POST body: { "level": "decent" }
    Valid levels: empty, almost_out, low, decent, plentiful, n/a
    """
    from inventory import AmmoLevel

    data = request.json or {}
    new_level = data.get('level', 'n/a').lower()

    # Validate level
    valid_levels = [l.value for l in AmmoLevel]
    if new_level not in valid_levels:
        return jsonify({'error': f'Invalid ammo level. Must be one of: {valid_levels}'}), 400

    inv = inventory_tracker.get_inventory(owner_id)
    if not inv:
        return jsonify({'error': 'Inventory not found'}), 404

    item = inv.find_item_by_id(item_id)
    if not item:
        return jsonify({'error': 'Item not found'}), 404

    old_level = item.ammo_level
    item.ammo_level = new_level
    inventory_tracker._save()

    # Generate appropriate message
    level_messages = {
        'empty': f'{item.name} is out of ammo',
        'almost_out': f'{item.name} - last few shots remaining',
        'low': f'{item.name} - running low on ammo',
        'decent': f'{item.name} - decent ammo supply',
        'plentiful': f'{item.name} - plenty of ammo',
        'n/a': f'{item.name} - ammo tracking disabled'
    }

    return jsonify({
        'item': item.to_dict(),
        'message': level_messages.get(new_level, f'Ammo level set to {new_level}'),
        'old_level': old_level,
        'new_level': new_level
    })


@app.route('/inventory/<owner_id>/items/<item_id>/drop-narration', methods=['POST'])
def generate_drop_narration(owner_id, item_id):
    """
    Generate a short DM narration for dropping an item.

    POST body: { "room_id": "..." }
    Returns: { "narration": "..." }
    """
    data = request.json or {}
    room_id = data.get('room_id')

    # Get the item details before it's dropped
    inv = inventory_tracker.get_inventory(owner_id)
    if not inv:
        return jsonify({'narration': None}), 200

    item = inv.find_item_by_id(item_id)
    if not item:
        return jsonify({'narration': None}), 200

    # Get character name
    char_name = "Someone"
    if room_id:
        room = data_store.get_room(room_id)
        if room:
            # Check if owner is a partner
            for p in room.get('partners', []):
                if p.get('id') == owner_id:
                    char_name = p.get('name', 'Someone')
                    break
            # Check player character
            player_char = room.get('player_character', {})
            if player_char.get('id') == owner_id:
                char_name = player_char.get('name', 'You')

    # Get current location if available
    location_name = ""
    if room_id:
        room = data_store.get_room(room_id)
        if room and room.get('current_location'):
            location_name = room.get('current_location', {}).get('name', '')

    # Generate a quick narration
    prompt = f"""Write a single brief sentence (15-25 words) describing {char_name} dropping or discarding their {item.name}.
Be matter-of-fact and grounded. No drama, no poetry. Just describe the simple action.
{f'They are at {location_name}.' if location_name else ''}

Examples of good responses:
- "Marcus sets down the worn flashlight on a nearby crate, deciding it's more weight than it's worth."
- "Elena tosses the empty canteen aside, letting it clatter against the concrete."
- "You leave the broken radio behind, already forgetting about it as you move on."

Respond with ONLY the narration, nothing else."""

    try:
        # Use the default model for this quick generation
        narration = asyncio.run(provider_manager.generate_ollama(prompt, settings.default_ollama_model))
        # Clean up the response
        narration = narration.strip().strip('"').strip("'")
        return jsonify({'narration': narration})
    except Exception as e:
        print(f"[DropNarration] Failed to generate: {e}")
        return jsonify({'narration': None}), 200


@app.route('/inventory/<owner_id>/combat-ready', methods=['GET'])
def get_combat_ready(owner_id):
    """Get combat readiness status for a character."""
    status = inventory_tracker.get_combat_status(owner_id)
    return jsonify(status)


@app.route('/inventory/<owner_id>/combat-ready', methods=['POST'])
def set_combat_ready(owner_id):
    """
    Set combat readiness (for using fists or entering combat stance).

    POST body: { "ready": true/false }
    """
    data = request.json or {}
    ready = data.get('ready', True)

    result = inventory_tracker.set_combat_ready(owner_id, ready)
    return jsonify(result)


@app.route('/inventory/<owner_id>/holster', methods=['POST'])
def holster_weapons(owner_id):
    """
    Unequip all weapons and exit combat stance.

    Called automatically after messages when no combat occurs.
    """
    unequipped = inventory_tracker.unequip_all_weapons(owner_id)
    return jsonify({
        'unequipped': unequipped,
        'combat_ready': False,
        'message': f'Holstered: {", ".join(unequipped)}' if unequipped else 'Already holstered'
    })


@app.route('/inventory/transfer', methods=['POST'])
def transfer_inventory_item():
    """Transfer an item between characters."""
    data = request.json or {}
    from_owner = data.get('from_owner_id')
    to_owner = data.get('to_owner_id')
    item_id = data.get('item_id')
    quantity = data.get('quantity', 1)

    if not all([from_owner, to_owner, item_id]):
        return jsonify({'error': 'from_owner_id, to_owner_id, and item_id required'}), 400

    result = inventory_tracker.transfer_item(from_owner, to_owner, item_id, quantity)
    if result:
        return jsonify({'success': True, 'message': 'Item transferred'})
    return jsonify({'error': 'Transfer failed - check owners and item exist'}), 400


@app.route('/inventory/<owner_id>/currency', methods=['POST'])
def modify_currency(owner_id):
    """Add or remove currency."""
    data = request.json or {}
    currency_type = data.get('currency_type', 'gold')
    amount = data.get('amount', 0)
    action = data.get('action', 'add')  # 'add' or 'remove'

    inv = inventory_tracker.get_inventory(owner_id)
    if not inv:
        return jsonify({'error': 'Inventory not found'}), 404

    if action == 'remove':
        if not inv.remove_currency(currency_type, amount):
            return jsonify({'error': 'Insufficient currency'}), 400
    else:
        inv.add_currency(currency_type, amount)

    inventory_tracker._save()
    return jsonify({'currency': inv.currency})


@app.route('/rooms/<room_id>/inventory', methods=['GET'])
def get_room_inventory(room_id):
    """Get inventory status for all characters in a room."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    # Get all partner IDs in this room
    partner_ids = []
    if room.partner_id:
        partner_ids = [room.partner_id]
    elif room.partner_ids:
        partner_ids = room.partner_ids

    # Also include player character
    player_id = f"player_{room_id}"
    player_name = room.player_character_name or settings.user_name

    # Build inventory status for all
    inventories = {}

    # Player inventory - auto-create for StoryBuilder rooms if doesn't exist
    player_inv = inventory_tracker.get_inventory(player_id)
    if not player_inv and (room.player_character_name or (hasattr(room, 'character_relationships') and room.character_relationships)):
        # Auto-create player inventory for StoryBuilder rooms
        player_inv = inventory_tracker.get_or_create_inventory(player_id, player_name, owner_type='player')
        # Add basic starting item
        player_inv.add_item(name='Backpack', category=ItemCategory.CONTAINER, description='For carrying your belongings', weight=2.0, capacity=25.0)
        inventory_tracker._save()
    if player_inv:
        inventories[player_id] = player_inv.to_dict()

    # Partner inventories - auto-create if needed
    for pid in partner_ids:
        partner = data_store.get_partner(pid)
        partner_inv = inventory_tracker.get_inventory(pid)
        if not partner_inv and partner:
            # Auto-create partner inventory
            partner_inv = inventory_tracker.get_or_create_inventory(pid, partner.name, owner_type='partner')
        if partner_inv:
            inventories[pid] = partner_inv.to_dict()

    return jsonify({
        'room_id': room_id,
        'inventories': inventories,
        'summary': inventory_tracker.get_all_inventories_context()
    })


# ============================================================================
# Consequence Engine - Ripple effects from actions
# ============================================================================

@app.route('/rooms/<room_id>/consequences', methods=['GET'])
def get_consequences(room_id):
    """Get pending consequences for a room."""
    pending = consequence_engine.get_pending_for_world(room_id)
    return jsonify({
        'room_id': room_id,
        'pending': [c.to_dict() for c in pending],
        'context': consequence_engine.get_consequence_context(room_id)
    })


@app.route('/rooms/<room_id>/consequences', methods=['POST'])
def create_consequence(room_id):
    """Manually create a consequence (for DM use)."""
    data = request.json or {}
    sound_type = data.get('sound_type')
    visual_type = data.get('visual_type')

    room = data_store.get_room(room_id)
    genre = room.genre if room and hasattr(room, 'genre') and room.genre else 'fantasy'

    consequences = []

    if sound_type:
        consequences = consequence_engine.calculate_sound_consequence(
            world_id=room_id,
            sound_type=sound_type,
            source_location_id=data.get('location_id'),
            source_character_id=data.get('character_id'),
            genre=genre
        )
    elif visual_type:
        consequences = consequence_engine.calculate_visual_consequence(
            world_id=room_id,
            visual_type=visual_type,
            source_location_id=data.get('location_id'),
            source_character_id=data.get('character_id'),
        )

    # Queue the consequences
    game_day = data.get('game_day', 0)
    game_hour = data.get('game_hour', 0)

    for c in consequences:
        consequence_engine.queue_consequence(room_id, c, game_day, game_hour)

    return jsonify({
        'queued': len(consequences),
        'consequences': [c.to_dict() for c in consequences]
    })


@app.route('/rooms/<room_id>/consequences/process', methods=['POST'])
def process_action_consequences(room_id):
    """Process narrative text for automatic consequence detection."""
    data = request.json or {}
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    room = data_store.get_room(room_id)
    genre = room.genre if room and hasattr(room, 'genre') and room.genre else 'fantasy'

    consequences = consequence_engine.process_action_text(
        world_id=room_id,
        text=text,
        source_location_id=data.get('location_id'),
        source_character_id=data.get('character_id'),
        genre=genre,
        current_game_day=data.get('game_day', 0),
        current_game_hour=data.get('game_hour', 0)
    )

    return jsonify({
        'detected': len(consequences),
        'consequences': [c.to_dict() for c in consequences]
    })


@app.route('/rooms/<room_id>/consequences/trigger', methods=['POST'])
def trigger_consequences(room_id):
    """Check and trigger any consequences that are due."""
    data = request.json or {}
    game_day = data.get('game_day', 0)
    game_hour = data.get('game_hour', 0)

    triggered = consequence_engine.check_and_trigger(room_id, game_day, game_hour)

    return jsonify({
        'triggered': len(triggered),
        'consequences': [c.to_dict() for c in triggered]
    })


# ============================================================================
# Loot Tables - Genre-appropriate random item generation
# ============================================================================

@app.route('/rooms/<room_id>/loot', methods=['POST'])
def generate_room_loot(room_id):
    """Generate loot for a room context."""
    data = request.json or {}

    room = data_store.get_room(room_id)
    genre = data.get('genre') or (room.genre if room and hasattr(room, 'genre') and room.genre else 'fantasy')

    # Determine context
    context_str = data.get('context', 'container_decent')
    try:
        context = LootContext(context_str)
    except ValueError:
        context = LootContext.CONTAINER_DECENT

    generator = get_loot_generator(genre)
    loot = generator.generate(
        context=context,
        description=data.get('description', ''),
        luck_modifier=data.get('luck_modifier', 0.0),
        guaranteed_items=data.get('guaranteed_items')
    )

    return jsonify(loot.to_dict())


@app.route('/loot/generate', methods=['POST'])
def generate_loot():
    """Generate loot without room context."""
    data = request.json or {}
    genre = data.get('genre', 'fantasy')

    # Determine context
    context_str = data.get('context', 'container_decent')
    try:
        context = LootContext(context_str)
    except ValueError:
        context = LootContext.CONTAINER_DECENT

    generator = get_loot_generator(genre)
    loot = generator.generate(
        context=context,
        description=data.get('description', ''),
        luck_modifier=data.get('luck_modifier', 0.0),
        guaranteed_items=data.get('guaranteed_items')
    )

    return jsonify(loot.to_dict())


@app.route('/loot/suggest', methods=['POST'])
def suggest_loot():
    """Get a loot suggestion based on scene context."""
    data = request.json or {}
    genre = data.get('genre', 'fantasy')
    scene = data.get('scene_description', '')
    creature = data.get('creature_type')
    container = data.get('container_type')

    suggestion = suggest_loot_for_scene(
        genre=genre,
        scene_description=scene,
        creature_type=creature,
        container_type=container
    )

    return jsonify({'suggestion': suggestion})


# ============================================================================
# Cartographer - World map and location tracking
# ============================================================================

@app.route('/rooms/<room_id>/map', methods=['GET'])
def get_world_map(room_id):
    """Get the world map for a room."""
    world_map = cartographer.get_map(room_id)
    if not world_map:
        return jsonify({'error': 'No map exists for this room', 'exists': False}), 404

    return jsonify({
        'exists': True,
        'map': world_map.to_dict(),
        'discovered_count': len(world_map.get_discovered_locations()),
        'total_count': len(world_map.locations),
        'context': cartographer.get_dm_context(room_id)
    })


@app.route('/rooms/<room_id>/map', methods=['POST'])
def create_world_map(room_id):
    """Create or update a world map for a room."""
    data = request.json or {}

    room = data_store.get_room(room_id)
    genre = data.get('genre') or (room.genre if room and hasattr(room, 'genre') and room.genre else 'fantasy')

    world_map = cartographer.get_or_create_map(room_id, genre)

    return jsonify({
        'map': world_map.to_dict(),
        'message': 'Map created/updated'
    })


@app.route('/rooms/<room_id>/map/location', methods=['POST'])
def add_map_location(room_id):
    """Add a location to the world map."""
    data = request.json or {}

    name = data.get('name')
    if not name:
        return jsonify({'error': 'Location name required'}), 400

    world_map = cartographer.get_or_create_map(room_id)

    # Parse location type
    type_str = data.get('type', 'poi')
    try:
        loc_type = LocationType(type_str)
    except ValueError:
        loc_type = LocationType.POINT_OF_INTEREST

    # Parse discovery status
    status_str = data.get('discovery_status', 'unknown')
    try:
        discovery = DiscoveryStatus(status_str)
    except ValueError:
        discovery = DiscoveryStatus.UNKNOWN

    from cartographer import Location
    import uuid

    location = Location(
        id=str(uuid.uuid4())[:12],
        name=name,
        location_type=loc_type,
        x=data.get('x', 0),
        y=data.get('y', 0),
        short_description=data.get('short_description', ''),
        full_description=data.get('full_description', ''),
        discovery_status=discovery,
        features=data.get('features', []),
    )

    world_map.add_location(location)
    cartographer._save()

    return jsonify({
        'location': location.to_dict(),
        'message': f'Added {name} to the map'
    })


@app.route('/rooms/<room_id>/map/discover/<location_id>', methods=['POST'])
def discover_map_location(room_id, location_id):
    """Mark a location as discovered."""
    data = request.json or {}

    status_str = data.get('status', 'known')
    try:
        status = DiscoveryStatus(status_str)
    except ValueError:
        status = DiscoveryStatus.KNOWN

    player_id = data.get('player_id', 'player')

    location = cartographer.discover_location(room_id, location_id, player_id, status)

    if not location:
        return jsonify({'error': 'Location not found'}), 404

    return jsonify({
        'location': location.to_dict(),
        'message': f'Discovered {location.name}'
    })


@app.route('/rooms/<room_id>/map/discover-from-text', methods=['POST'])
def discover_from_text(room_id):
    """Scan text for location mentions and mark them discovered."""
    data = request.json or {}
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    player_id = data.get('player_id', 'narrative')
    discovered = cartographer.discover_from_text(room_id, text, player_id)

    return jsonify({
        'discovered': len(discovered),
        'locations': [loc.to_dict() for loc in discovered]
    })


@app.route('/rooms/<room_id>/map/locations', methods=['GET'])
def get_map_locations(room_id):
    """Get all locations for a room (with optional filtering)."""
    world_map = cartographer.get_map(room_id)
    if not world_map:
        return jsonify({'locations': [], 'discovered': [], 'hidden': []})

    discovered_only = request.args.get('discovered', 'false').lower() == 'true'

    if discovered_only:
        locations = world_map.get_discovered_locations()
    else:
        locations = list(world_map.locations.values())

    discovered = [l for l in locations if l.discovery_status != DiscoveryStatus.UNKNOWN]
    hidden = [l for l in locations if l.discovery_status == DiscoveryStatus.UNKNOWN]

    return jsonify({
        'locations': [l.to_dict() for l in locations],
        'discovered': [l.to_dict() for l in discovered],
        'hidden': [l.to_dict() for l in hidden],
        'starting_location_id': world_map.starting_location_id
    })


@app.route('/rooms/<room_id>/map/travel', methods=['GET'])
def get_travel_info(room_id):
    """Get travel information from current location."""
    world_map = cartographer.get_map(room_id)
    if not world_map:
        return jsonify({'error': 'No map exists'}), 404

    current_loc_id = request.args.get('from')

    return jsonify({
        'context': world_map.get_travel_context(current_loc_id)
    })


# ============================================================================
# NPC System - Persistent NPCs with residue, agency, and souls
# ============================================================================

def _save_npcs():
    """Save NPC registry to disk."""
    npc_registry.save(str(settings.data_dir / "npcs.json"))


@app.route('/rooms/<room_id>/npcs/registry', methods=['GET'])
def get_room_npcs_registry(room_id):
    """Get all NPCs in a room from the NPC registry (includes worldwalker status)."""
    npcs = npc_registry.get_npcs_in_world(room_id)
    return jsonify({
        'npcs': [npc.to_dict() for npc in npcs],
        'count': len(npcs),
        'alive': len([n for n in npcs if n.is_alive]),
        'worldwalkers': len([n for n in npcs if n.is_free])
    })


@app.route('/rooms/<room_id>/npcs', methods=['POST'])
def create_room_npc(room_id):
    """Create a new NPC in a room."""
    data = request.json or {}
    name = data.get('name')
    if not name:
        return jsonify({'error': 'NPC name required'}), 400

    npc = npc_registry.create_npc(
        name=name,
        origin_world=room_id,
        backstory=data.get('backstory', ''),
        current_role=data.get('current_role', ''),
        physical_description=data.get('physical_description', ''),
        personality=data.get('personality', ''),
        secret=data.get('secret'),
        wound=data.get('wound'),
        want=data.get('want'),
        fear=data.get('fear'),
    )

    if data.get('current_location'):
        npc.current_location = data['current_location']

    _save_npcs()

    return jsonify({
        'npc': npc.to_dict(),
        'message': f'Created NPC: {name}'
    })


@app.route('/npcs/<npc_id>', methods=['GET'])
def get_npc(npc_id):
    """Get a specific NPC."""
    npc = npc_registry.get_npc(npc_id)
    if not npc:
        return jsonify({'error': 'NPC not found'}), 404
    return jsonify(npc.to_dict())


@app.route('/npcs/<npc_id>/interact', methods=['POST'])
def interact_with_npc(npc_id):
    """Record an interaction with an NPC."""
    from npc_system import NPCInteraction
    from datetime import datetime

    npc = npc_registry.get_npc(npc_id)
    if not npc:
        return jsonify({'error': 'NPC not found'}), 404

    data = request.json or {}

    interaction = NPCInteraction(
        timestamp=datetime.now().isoformat(),
        player_id=data.get('player_id', 'player'),
        player_name=data.get('player_name', 'Player'),
        interaction_type=data.get('type', 'conversation'),
        sentiment=data.get('sentiment', 0.0),
        summary=data.get('summary', ''),
        weight=data.get('weight', 1.0)
    )

    new_state = npc.add_interaction(interaction)
    _save_npcs()

    return jsonify({
        'npc': npc.to_dict(),
        'state_changed': new_state is not None,
        'new_state': new_state.value if new_state else None
    })


@app.route('/npcs/<npc_id>/grievance', methods=['POST'])
def add_npc_grievance(npc_id):
    """Add a grievance against someone (may create/escalate grudge)."""
    npc = npc_registry.get_npc(npc_id)
    if not npc:
        return jsonify({'error': 'NPC not found'}), 404

    data = request.json or {}
    target_id = data.get('target_id')
    target_name = data.get('target_name')
    offense = data.get('offense')

    if not all([target_id, target_name, offense]):
        return jsonify({'error': 'target_id, target_name, and offense required'}), 400

    # Check for instant nemesis
    if data.get('unforgivable'):
        grudge = npc.create_instant_nemesis(target_id, target_name, offense)
    else:
        grudge = npc.add_grievance(
            target_id=target_id,
            target_name=target_name,
            offense=offense,
            severity_boost=data.get('severity_boost', 0.0)
        )

    _save_npcs()

    if grudge:
        return jsonify({
            'grudge': grudge.to_dict(),
            'message': f'{npc.name} now holds a grudge against {target_name}'
        })
    else:
        return jsonify({
            'grudge': None,
            'message': f'{npc.name} let it slide'
        })


@app.route('/npcs/<npc_id>/kill', methods=['POST'])
def kill_npc(npc_id):
    """Kill an NPC (permanent!)."""
    data = request.json or {}
    killed_by = data.get('killed_by', 'unknown')
    cause = data.get('cause', 'unknown')
    world = data.get('world', 'unknown')

    try:
        npc = npc_registry.kill_npc(npc_id, killed_by, cause, world)
        _save_npcs()

        return jsonify({
            'npc': npc.to_dict(),
            'message': f'{npc.name} has died.',
            'was_worldwalker': npc.is_free,
            'funeral_available': True
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 404


@app.route('/npcs/<npc_id>/funeral', methods=['GET'])
def get_npc_funeral(npc_id):
    """Get funeral data for a dead NPC."""
    funeral = npc_registry.get_funeral_data(npc_id)
    if not funeral:
        return jsonify({'error': 'NPC not in graveyard'}), 404
    return jsonify(funeral)


@app.route('/npcs/<npc_id>/set-free', methods=['POST'])
def set_npc_free(npc_id):
    """DM sets an NPC free (Soul → Worldwalker)."""
    try:
        npc = npc_registry.set_free(npc_id)
        _save_npcs()
        return jsonify({
            'npc': npc.to_dict(),
            'message': f'{npc.name} is now a Worldwalker!'
        })
    except (ValueError, PermissionError) as e:
        return jsonify({'error': str(e)}), 400


@app.route('/npcs/worldwalkers', methods=['GET'])
def get_worldwalkers():
    """Get all living Worldwalkers."""
    walkers = npc_registry.get_worldwalkers()
    return jsonify({
        'worldwalkers': [w.to_dict() for w in walkers],
        'count': len(walkers)
    })


@app.route('/npcs/graveyard', methods=['GET'])
def get_graveyard():
    """Get all dead NPCs."""
    return jsonify({
        'dead': [npc.to_dict() for npc in npc_registry.graveyard.values()],
        'count': len(npc_registry.graveyard)
    })


def get_npc_dm_context(room_id: str) -> str:
    """Get NPC context for the DM."""
    npcs = npc_registry.get_npcs_in_world(room_id)
    if not npcs:
        return ""

    lines = ["=== NPCs IN SCENE ==="]
    for npc in npcs:
        if not npc.is_alive:
            continue

        state_icon = {
            "ephemeral": "·",
            "residue": "○",
            "agency": "◐",
            "soul": "●",
            "worldwalker": "★"
        }.get(npc.state.value, "?")

        lines.append(f"\n[{state_icon}] {npc.name} ({npc.current_role or 'unknown role'})")

        if npc.personality:
            lines.append(f"  Personality: {npc.personality[:100]}...")

        # Show hidden depths to DM
        if npc.secret:
            lines.append(f"  SECRET: {npc.secret}")
        if npc.want:
            lines.append(f"  WANT: {npc.want}")
        if npc.fear:
            lines.append(f"  FEAR: {npc.fear}")

        # Show grudges
        if npc.grudges:
            grudge_strs = [f"{g.target_name} ({g.severity.value})" for g in npc.grudges]
            lines.append(f"  GRUDGES: {', '.join(grudge_strs)}")

        # Show nemeses (important!)
        nemeses = npc.get_nemeses()
        if nemeses:
            lines.append(f"  ⚔️ NEMESES: {', '.join(g.target_name for g in nemeses)}")

    return "\n".join(lines)


# ============================================================================
# Action Resolver - Systems decide, DM narrates
# ============================================================================

@app.route('/rooms/<room_id>/resolve', methods=['POST'])
def resolve_player_action(room_id):
    """
    Resolve a player action before sending to DM.
    System pre-resolves mechanical outcomes (combat, skills, inventory).
    """
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    data = request.json or {}
    action_text = data.get('action', '').strip()
    character_id = data.get('character_id')

    if not action_text:
        return jsonify({'error': 'No action provided'}), 400

    # Build hard facts from current game state
    hard_facts = {
        'is_alive': True,
        'is_sleeping': False,
        'player_has': [],
        'player_skills': [],
        'fatigue_state': 'rested'
    }

    # Get character-specific facts if character_id provided
    if character_id:
        # Fatigue state
        fatigue_effects = fatigue_tracker.get_fatigue_effects(character_id)
        hard_facts['fatigue_state'] = fatigue_effects.get('level', 'rested')
        hard_facts['is_sleeping'] = fatigue_effects.get('is_sleeping', False)

        # Inventory
        inv = inventory_tracker.get_inventory(character_id)
        if inv and inv.items:
            hard_facts['player_has'] = [item.name for item in inv.items]

        # Skills from partner
        partner = data_store.get_partner(character_id)
        if partner and partner.skill:
            hard_facts['player_skills'] = [partner.skill]

    # Resolve the action
    action_type, resolution = resolve_action(action_text, hard_facts)

    result = {
        'action_type': action_type.value,
        'action_text': action_text,
        'requires_dm': resolution is None
    }

    if resolution:
        result['resolution'] = {
            'success': resolution.success,
            'outcome': resolution.outcome_description,
            'items_consumed': resolution.items_consumed,
            'items_gained': resolution.items_gained,
            'consequences': resolution.consequences,
            'fatigue_effect': resolution.fatigue_modifier,
            'skill_used': resolution.skill_used,
            'dm_instruction': resolution.dm_instruction,
            'raw': resolution.raw_resolution
        }
        result['dm_block'] = build_dm_instruction_for_resolution(resolution)

        # If items were consumed, actually remove them from inventory
        if resolution.items_consumed and character_id:
            for item_name in resolution.items_consumed:
                if item_name != "1 ammunition":  # Don't try to remove the generic ammo string
                    inventory_tracker.remove_item(character_id, item_name)

        # If consequences were triggered, queue them
        if resolution.consequences:
            for consequence in resolution.consequences:
                if consequence.startswith("SOUND:"):
                    consequence_engine.queue_sound_consequence(
                        room_id=room_id,
                        source=action_text[:50],
                        sound_type=consequence.split(":")[1].split(" ")[0],
                        severity="moderate"
                    )

    return jsonify(result)


@app.route('/action/classify', methods=['POST'])
def classify_player_action():
    """
    Just classify an action without resolving it.
    Useful for UI to show what type of action is being attempted.
    """
    data = request.json or {}
    action_text = data.get('action', '').strip()

    if not action_text:
        return jsonify({'error': 'No action provided'}), 400

    action_type = classify_action(action_text)

    return jsonify({
        'action': action_text,
        'type': action_type.value,
        'description': {
            'query': 'Question - DM has full authority',
            'combat': 'Combat action - System resolves hit/miss',
            'inventory': 'Inventory action - System validates items',
            'skill': 'Skill check - System resolves success/failure',
            'movement': 'Movement - System checks possibility',
            'social': 'Social interaction - DM has authority',
            'unknown': 'Unknown action type - DM handles'
        }.get(action_type.value, 'Unknown')
    })


@app.route('/action/types', methods=['GET'])
def get_action_types():
    """Get all action types and their descriptions."""
    return jsonify({
        'types': [
            {'value': 'query', 'name': 'Query', 'resolver': 'DM', 'description': 'Questions about the world'},
            {'value': 'combat', 'name': 'Combat', 'resolver': 'System', 'description': 'Attack, shoot, fight'},
            {'value': 'inventory', 'name': 'Inventory', 'resolver': 'System', 'description': 'Pick up, use, drop items'},
            {'value': 'skill', 'name': 'Skill', 'resolver': 'System', 'description': 'Attempt skilled actions'},
            {'value': 'movement', 'name': 'Movement', 'resolver': 'System', 'description': 'Travel, enter, exit'},
            {'value': 'social', 'name': 'Social', 'resolver': 'DM', 'description': 'Talk, persuade, negotiate'},
            {'value': 'unknown', 'name': 'Unknown', 'resolver': 'DM', 'description': 'Falls back to DM authority'},
        ]
    })


# ============================================================================
# Combat System - Pathfinder 2e turn-based combat
# ============================================================================

@app.route('/rooms/<room_id>/combat', methods=['GET'])
def get_combat_state(room_id):
    """Get current combat state for a room."""
    encounter = EncounterManager.get(room_id)
    if not encounter:
        return jsonify({'active': False, 'message': 'No combat in progress'})

    return jsonify({
        'active': encounter.is_active,
        'encounter': encounter.to_dict()
    })


@app.route('/rooms/<room_id>/combat/start', methods=['POST'])
def start_combat(room_id):
    """
    Start a combat encounter.

    Body: {
        "combatants": [
            {"id": "char_id", "team": "players", "type": "companion"},
            {"id": "npc_id", "team": "enemies", "type": "npc", "role": "bandit"}
        ]
    }
    """
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    data = request.json or {}
    combatant_specs = data.get('combatants', [])

    if len(combatant_specs) < 2:
        return jsonify({'error': 'Need at least 2 combatants'}), 400

    # Create encounter
    encounter = EncounterManager.get_or_create(room_id)

    if encounter.is_active:
        return jsonify({'error': 'Combat already in progress'}), 400

    # Add combatants
    for spec in combatant_specs:
        c_id = spec.get('id')
        c_type = spec.get('type', 'npc')
        team = spec.get('team', 'neutral')
        level = spec.get('level', 1)

        if c_type == 'companion':
            # Get companion from data store
            partner = data_store.get_partner(c_id)
            if partner:
                stats = create_companion_stats(partner.name, level)
                encounter.add_combatant(
                    id=c_id,
                    stats=stats,
                    team=team,
                    is_companion=True,
                    companion_id=c_id,
                )
        elif c_type == 'npc':
            # Get NPC from registry or create from role
            npc = npc_registry.get_npc(c_id)
            if npc:
                role = spec.get('role', npc.current_role or 'civilian')
                stats = create_npc_combatant(npc.name, role, level)
                encounter.add_combatant(
                    id=c_id,
                    stats=stats,
                    team=team,
                    is_npc=True,
                    npc_id=c_id,
                )
            else:
                # Create anonymous NPC
                name = spec.get('name', f'Enemy {c_id}')
                role = spec.get('role', 'bandit')
                stats = create_npc_combatant(name, role, level)
                encounter.add_combatant(
                    id=c_id,
                    stats=stats,
                    team=team,
                    is_npc=True,
                )
        elif c_type == 'spidercock':
            stats = create_spidercock_stats()
            encounter.add_combatant(
                id='spidercock',
                stats=stats,
                team='enemies',
                is_npc=True,
            )

    # Start combat
    encounter.start_combat()

    return jsonify({
        'message': 'Combat started',
        'encounter': encounter.to_dict()
    })


@app.route('/rooms/<room_id>/combat/attack', methods=['POST'])
def combat_attack(room_id):
    """
    Execute an attack in combat.

    Body: {
        "attacker_id": "char_id",
        "defender_id": "target_id",
        "attack_index": 0,  # Which attack to use (0 = primary)
        "map_penalty": 0    # Multiple Attack Penalty: 0, -5, or -10
    }
    """
    encounter = EncounterManager.get(room_id)
    if not encounter or not encounter.is_active:
        return jsonify({'error': 'No active combat'}), 400

    data = request.json or {}
    attacker_id = data.get('attacker_id')
    defender_id = data.get('defender_id')
    attack_index = data.get('attack_index', 0)
    map_penalty = data.get('map_penalty', 0)

    if not attacker_id or not defender_id:
        return jsonify({'error': 'Missing attacker_id or defender_id'}), 400

    try:
        result = encounter.execute_attack(
            attacker_id,
            defender_id,
            attack_index,
            map_penalty
        )

        response = {
            'result': result.to_dm_view(),
            'player_view': result.to_player_view(),
            'encounter': encounter.to_dict()
        }

        # Check for death
        if result.defender_died:
            dead = encounter.combatants.get(defender_id)
            killer = encounter.combatants.get(attacker_id)

            if dead and dead.is_companion and killer:
                room = data_store.get_room(room_id)
                death_result = process_companion_death_in_combat(
                    encounter, dead, killer,
                    npc_registry, data_store,
                    room=room
                )
                response['death'] = death_result

        return jsonify(response)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/rooms/<room_id>/combat/next-turn', methods=['POST'])
def combat_next_turn(room_id):
    """Advance to the next combatant's turn."""
    encounter = EncounterManager.get(room_id)
    if not encounter or not encounter.is_active:
        return jsonify({'error': 'No active combat'}), 400

    next_combatant = encounter.next_turn()

    if not next_combatant:
        return jsonify({
            'message': 'Combat ended - no survivors',
            'encounter': encounter.to_dict()
        })

    return jsonify({
        'current_turn': {
            'id': next_combatant.id,
            'name': next_combatant.stats.name,
            'team': next_combatant.team,
            'hp': next_combatant.stats.hp_current,
            'hp_max': next_combatant.stats.hp_max,
        },
        'round': encounter.round,
        'encounter': encounter.to_dict()
    })


@app.route('/rooms/<room_id>/combat/end', methods=['POST'])
def end_combat(room_id):
    """End combat early."""
    encounter = EncounterManager.get(room_id)
    if not encounter:
        return jsonify({'error': 'No combat found'}), 404

    data = request.json or {}
    reason = data.get('reason', 'manual')

    encounter.end_combat(reason)

    result = {
        'message': 'Combat ended',
        'encounter': encounter.to_dict()
    }

    # Clear the encounter
    EncounterManager.clear(room_id)

    return jsonify(result)


@app.route('/dice/roll', methods=['POST'])
def roll_dice():
    """
    Roll dice using standard notation.

    Body: {"notation": "2d6+3"}
    """
    data = request.json or {}
    notation = data.get('notation', '1d20')

    try:
        total, rolls = Dice.roll(notation)
        return jsonify({
            'notation': notation,
            'rolls': rolls,
            'total': total
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/dice/check', methods=['POST'])
def dice_check():
    """
    Make a d20 check against a DC.

    Body: {"dc": 15, "modifier": 5}
    """
    data = request.json or {}
    dc = data.get('dc', 10)
    modifier = data.get('modifier', 0)

    result = Dice.check(dc, modifier)
    return jsonify(result)


# ============================================================================
# Autopilot System - Characters continue when players are away
# ============================================================================

@app.route('/autopilot/characters', methods=['GET'])
def get_autopilot_characters():
    """Get all characters with autopilot configured."""
    characters = autopilot_tracker.get_all_characters()
    return jsonify({
        'characters': [c.to_dict() for c in characters],
        'count': len(characters)
    })


@app.route('/autopilot/characters/<character_id>', methods=['GET'])
def get_autopilot_character(character_id):
    """Get autopilot config for a specific character."""
    char = autopilot_tracker.get_character(character_id)
    if not char:
        return jsonify({'error': 'Character not found'}), 404
    return jsonify(char.to_dict())


@app.route('/autopilot/characters/<character_id>', methods=['POST'])
def configure_autopilot(character_id):
    """
    Configure autopilot for a character.

    Body: {
        "alignment": "neutral_good",
        "drive": "belonging",
        "role": "support",
        "safe_location": "enclave"
    }
    """
    data = request.json or {}

    # Get partner info
    partner = data_store.get_partner(character_id)
    if not partner:
        return jsonify({'error': 'Character not found'}), 404

    alignment_str = data.get('alignment', 'true_neutral')
    drive_str = data.get('drive', 'safety')
    role_str = data.get('role', 'balanced')

    try:
        alignment = Alignment(alignment_str)
        drive = Drive(drive_str)
        role = Role(role_str)
    except ValueError as e:
        return jsonify({'error': f'Invalid enum value: {e}'}), 400

    char = autopilot_tracker.configure_character(
        character_id=character_id,
        name=partner.name,
        alignment=alignment,
        drive=drive,
        role=role,
        skills=partner.skill.split(',') if partner.skill else [],
        safe_location=data.get('safe_location'),
    )

    return jsonify(char.to_dict())


@app.route('/autopilot/characters/<character_id>/enable', methods=['POST'])
def enable_autopilot(character_id):
    """Enable autopilot for a character."""
    char = autopilot_tracker.get_character(character_id)
    if not char:
        return jsonify({'error': 'Character not configured for autopilot'}), 404

    char.autopilot_enabled = True
    char.autopilot_started = datetime.now().isoformat()
    autopilot_tracker.save()

    return jsonify({'message': f'Autopilot enabled for {char.name}', 'character': char.to_dict()})


@app.route('/autopilot/characters/<character_id>/disable', methods=['POST'])
def disable_autopilot(character_id):
    """Disable autopilot for a character."""
    char = autopilot_tracker.get_character(character_id)
    if not char:
        return jsonify({'error': 'Character not found'}), 404

    char.autopilot_enabled = False
    autopilot_tracker.save()

    return jsonify({'message': f'Autopilot disabled for {char.name}', 'character': char.to_dict()})


@app.route('/autopilot/characters/<character_id>/journal', methods=['GET'])
def get_autopilot_journal(character_id):
    """Get journal entries for a character while on autopilot."""
    char = autopilot_tracker.get_character(character_id)
    if not char:
        return jsonify({'error': 'Character not found'}), 404

    limit = request.args.get('limit', 50, type=int)
    entries = char.journal[-limit:] if char.journal else []

    return jsonify({
        'character_id': character_id,
        'character_name': char.name,
        'entries': [e.to_dict() for e in entries],
        'count': len(entries)
    })


@app.route('/autopilot/tick', methods=['POST'])
def autopilot_tick():
    """
    Manually trigger an autopilot tick (for testing).
    In production, this would be called by the story daemon.
    """
    data = request.json or {}
    world_id = data.get('world_id')
    threat_level = data.get('threat_level', 0.0)
    hours_elapsed = data.get('hours_elapsed', 1.0)

    results = autopilot_tracker.tick_all_characters(
        world_id=world_id,
        threat_level=threat_level,
        hours_elapsed=hours_elapsed,
    )

    return jsonify({
        'ticked': len(results),
        'results': results
    })


# ============================================================================
# Background Thread System - Separated characters live their own stories
# ============================================================================

@app.route('/rooms/<room_id>/background/characters', methods=['GET'])
def get_background_characters(room_id):
    """
    Get all characters in a room with their separation status.
    Used to populate the RSS feed character toggles.
    Excludes the player character (only shows partners/NPCs).
    Also includes ephemeral/residue NPCs spawned by the DM.
    """
    characters = autopilot_tracker.get_all_characters_in_room(room_id)
    print(f"[Background Characters] Room {room_id}: found {len(characters)} total characters in autopilot")
    for c in characters:
        print(f"[Background Characters]   - {c.character_name} ({c.player_id}): separated={c.is_separated}, journal={len(c.journal)}")

    # Filter out the player character - RSS feed is for tracking NPCs/partners
    partner_characters = [c for c in characters if not c.player_id.startswith('player_')]
    print(f"[Background Characters] After filtering player: {len(partner_characters)} partner characters")

    result = [
        {
            'id': c.player_id,
            'name': c.character_name,
            'is_separated': c.is_separated,
            'is_alive': c.is_alive,
            'condition': c.condition,
            'supplies': c.supplies_status,
            'morale': c.morale,
            'last_known_location': c.last_known_location,
            'location_radius_km': c.location_radius_km,
            'autopilot_enabled': c.autopilot_enabled,
            'journal_count': len(c.journal),
            'is_npc': False,  # Main character, not ephemeral NPC
        }
        for c in partner_characters
    ]

    # Add room NPCs (spawned by DM interjections or unchosen characters)
    room_npcs = get_room_npcs(room_id)
    room = data_store.get_room(room_id)
    player_location = getattr(room, 'player_location', None) or "" if room else ""

    for npc in room_npcs:
        # Determine NPC condition/activity based on their role and want
        condition = npc.current_role or "nearby"
        if npc.want:
            condition = f"{condition} - {npc.want}"

        # Get their location (might be a specific place or just the room)
        location = npc.current_location or "nearby"

        # NPC is "here" if:
        # - Location is generic (room_id, "nearby", "here", empty)
        # - OR player has traveled to their location
        is_here = (
            location == room_id or
            location in ["nearby", "here", "with you", ""] or
            location.startswith("here") or
            (player_location and player_location.lower() == location.lower())
        )

        # NPC is "discovered" if player has interacted with them at least once
        discovered = npc.total_interactions > 0

        result.append({
            'id': f'npc_{npc.id}',
            'name': npc.name,
            'is_separated': not is_here,  # NPCs at other locations are "separated"
            'is_alive': npc.is_alive,
            'condition': condition,
            'location': location if not is_here else None,  # Show location if not here
            'discovered': discovered,  # Has player ever interacted with this NPC?
            'is_npc': True,  # Ephemeral/residue NPC from DM
            'npc_state': npc.state.value,
            'npc_personality': npc.personality,
            'npc_interactions': npc.total_interactions,
            'npc_weight': round(npc.interaction_weight, 1),
        })

    return jsonify({'characters': result})


@app.route('/rooms/<room_id>/background/feed', methods=['GET'])
def get_background_feed(room_id):
    """
    Get the RSS feed of character activities.

    Query params:
        character_ids: Comma-separated list of character IDs to filter (optional)
        limit: Max entries to return (default 100)
        since: ISO timestamp to get entries after (optional)
    """
    character_ids = request.args.get('character_ids', '')
    character_ids = [c.strip() for c in character_ids.split(',') if c.strip()] or None
    limit = min(int(request.args.get('limit', 100)), 500)
    since = request.args.get('since')

    # Debug: show what characters are in this room
    all_chars = autopilot_tracker.get_all_characters_in_room(room_id)
    print(f"[RSS Feed] Room {room_id}: {len(all_chars)} characters, filter: {character_ids}")
    for c in all_chars:
        print(f"[RSS Feed]   - {c.character_name} ({c.player_id}): {len(c.journal)} journal entries, separated={c.is_separated}")

    feed = autopilot_tracker.get_rss_feed(room_id, character_ids, limit)

    # Filter by timestamp if provided
    if since:
        try:
            feed = [e for e in feed if e['timestamp'] > since]
        except:
            pass

    return jsonify({
        'feed': feed,
        'count': len(feed),
        'room_id': room_id,
    })


@app.route('/rooms/<room_id>/background/separate', methods=['POST'])
def separate_character(room_id):
    """
    Mark a character as separated from the main party.

    Body: {
        "character_id": "player_xxx" or partner ID,
        "starting_location": "Description of where they're starting"
    }
    """
    data = request.json or {}
    character_id = data.get('character_id')
    starting_location = data.get('starting_location', 'Unknown location')

    if not character_id:
        return jsonify({'error': 'Missing character_id'}), 400

    success = autopilot_tracker.separate_character(character_id, room_id, starting_location)

    if success:
        return jsonify({
            'success': True,
            'message': f'Character separated at {starting_location}',
            'character_id': character_id,
        })
    else:
        return jsonify({'error': 'Character not found'}), 404


@app.route('/rooms/<room_id>/background/reunite', methods=['POST'])
def reunite_character(room_id):
    """
    Reunite a separated character with the main party.

    Body: {
        "character_id": "player_xxx" or partner ID
    }
    """
    data = request.json or {}
    character_id = data.get('character_id')

    if not character_id:
        return jsonify({'error': 'Missing character_id'}), 400

    summary = autopilot_tracker.reunite_character(character_id, room_id)

    if summary:
        return jsonify({
            'success': True,
            'summary': summary,
            'character_id': character_id,
        })
    else:
        return jsonify({'error': 'Character not found or not separated'}), 404


@app.route('/rooms/<room_id>/relationships/evolve', methods=['POST'])
def evolve_relationship(room_id):
    """
    Evolve a relationship based on meaningful interactions.
    Call this when characters have positive/negative exchanges.

    Body: {
        "character_a": "player_xxx" or partner ID,
        "character_b": "player_xxx" or partner ID,
        "interaction_type": "positive" | "negative" | "neutral",
        "context": "Brief description of interaction" (optional)
    }

    Relationship evolution:
    - stranger → acquaintance: 3+ positive interactions
    - acquaintance → friend: 5+ more positive interactions
    - friend → close_friend: 5+ more deep interactions
    - Any → rival/enemy: 3+ negative interactions
    """
    data = request.json or {}
    char_a = data.get('character_a', '')
    char_b = data.get('character_b', '')
    interaction_type = data.get('interaction_type', 'positive')
    context = data.get('context', '')

    if not char_a or not char_b:
        return jsonify({'error': 'Missing character_a or character_b'}), 400

    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    # Relationship progression ladder
    POSITIVE_LADDER = ['stranger', 'acquaintance', 'colleague', 'friend', 'close_friend']
    NEGATIVE_LADDER = ['stranger', 'rival', 'enemy']

    # Thresholds for progression
    POSITIVE_THRESHOLD = 3  # Interactions needed to upgrade
    NEGATIVE_THRESHOLD = 3

    # Find or create relationship tracking in room
    # We'll store interaction counts in a separate field
    if not hasattr(room, 'relationship_interactions') or room.relationship_interactions is None:
        room.relationship_interactions = {}

    key = f"{char_a}_{char_b}" if char_a < char_b else f"{char_b}_{char_a}"

    if key not in room.relationship_interactions:
        room.relationship_interactions[key] = {'positive': 0, 'negative': 0, 'neutral': 0}

    # Record interaction
    room.relationship_interactions[key][interaction_type] = room.relationship_interactions[key].get(interaction_type, 0) + 1
    counts = room.relationship_interactions[key]

    # Find current relationship type in character_relationships
    current_type = 'stranger'
    rel_found = False

    for char_rel in room.character_relationships:
        if char_rel.get('character_id') == char_a:
            for rel in char_rel.get('relationships', []):
                if rel.get('target_id') == char_b:
                    current_type = rel.get('type', 'stranger')
                    rel_found = True
                    break
        if rel_found:
            break

    # Calculate new relationship type
    new_type = current_type
    message = ""

    if interaction_type == 'positive':
        if current_type in POSITIVE_LADDER:
            idx = POSITIVE_LADDER.index(current_type)
            if counts['positive'] >= POSITIVE_THRESHOLD and idx < len(POSITIVE_LADDER) - 1:
                new_type = POSITIVE_LADDER[idx + 1]
                counts['positive'] = 0  # Reset counter
                message = f"Relationship evolved: {current_type} → {new_type}"
    elif interaction_type == 'negative':
        if current_type not in NEGATIVE_LADDER or current_type == 'stranger':
            if counts['negative'] >= NEGATIVE_THRESHOLD:
                new_type = 'rival'
                counts['negative'] = 0
                message = f"Relationship soured: {current_type} → {new_type}"
        elif current_type == 'rival':
            if counts['negative'] >= NEGATIVE_THRESHOLD:
                new_type = 'enemy'
                counts['negative'] = 0
                message = f"Relationship deteriorated: rival → enemy"

    # Update character_relationships if changed
    if new_type != current_type:
        updated = False
        for char_rel in room.character_relationships:
            if char_rel.get('character_id') == char_a:
                for rel in char_rel.get('relationships', []):
                    if rel.get('target_id') == char_b:
                        rel['type'] = new_type
                        if context:
                            rel['note'] = context[:90]  # Keep notes short
                        updated = True
                        break
                if not updated:
                    # Add new relationship
                    char_rel.setdefault('relationships', []).append({
                        'target_id': char_b,
                        'target_name': char_b,  # Could look up actual name
                        'type': new_type,
                        'note': context[:90] if context else ''
                    })
                    updated = True
                break

        data_store.save()

    return jsonify({
        'success': True,
        'character_a': char_a,
        'character_b': char_b,
        'interaction_type': interaction_type,
        'current_type': new_type,
        'previous_type': current_type,
        'evolved': new_type != current_type,
        'message': message,
        'interaction_counts': counts,
    })


@app.route('/rooms/<room_id>/background/tick', methods=['POST'])
def trigger_background_tick(room_id):
    """
    Manually trigger background ticks for separated characters.
    This processes all characters in the room that need a tick.

    If autopilot is enabled, ALSO processes:
    - The player character (via understudy)
    - Present NPCs (they respond to the player's autopilot actions)

    Body: {
        "world_context": "Description of the scenario",
        "threat_type": "What the threats are like",
        "world_state": "Current state of civilization",
        "autopilot_enabled": false,  # If true, player acts too
        "autopilot_model": "model-name",
        "autopilot_notes": "Any guidance for the understudy"
    }
    """
    from background_threads import BackgroundTickProcessor, get_scheduler, create_scheduler

    data = request.json or {}
    world_context = data.get('world_context', '')
    threat_type = data.get('threat_type', '')
    world_state = data.get('world_state', '')
    interval_minutes = data.get('interval_minutes', 30)  # Default 30 min, affects event drama

    # Autopilot settings
    autopilot_enabled = data.get('autopilot_enabled', False)
    autopilot_model = data.get('autopilot_model', '')
    autopilot_notes = data.get('autopilot_notes', '')

    # Get room to find its model
    room = data_store.get_room(room_id)
    room_model = room.room_model if room else None

    # Get or create scheduler
    scheduler = get_scheduler(room_id)
    if not scheduler:
        scheduler = create_scheduler(room_id, world_context, threat_type, world_state)

    # Update scheduler with current interval for drama scaling
    scheduler.interval_minutes = interval_minutes

    # Create processor with our generation functions (using room's model)
    use_model = room_model or settings.storybuilder_model
    async def ollama_gen(prompt):
        return await provider_manager.generate_ollama(prompt, use_model)

    async def dm_gen(prompt):
        return await provider_manager.generate_ollama(prompt, use_model)

    processor = BackgroundTickProcessor(
        tracker=autopilot_tracker,
        ollama_generate=ollama_gen,
        dm_generate=dm_gen,
    )
    scheduler.set_processor(processor)

    # Process all pending ticks for separated characters
    import asyncio
    autopilot_acted = False

    try:
        results = asyncio.run(scheduler.process_all_pending())

        # If autopilot is enabled, process player and present characters
        if autopilot_enabled and room:
            autopilot_result = _process_autopilot_tick(
                room_id=room_id,
                room=room,
                autopilot_model=autopilot_model or use_model,
                autopilot_notes=autopilot_notes,
                interval_minutes=interval_minutes
            )
            if autopilot_result:
                autopilot_acted = True
                print(f"\033[93m[AUTOPILOT] Player acted via understudy\033[0m")

        return jsonify({
            'success': True,
            'ticked': len(results),
            'autopilot_acted': autopilot_acted,
            'results': [
                {
                    'character_id': r.character_id,
                    'character_name': r.character_name,
                    'situation': r.situation.description,
                    'response': r.response,
                    'outcome': r.outcome,
                    'condition': r.condition_change,
                    'discovery': r.discovery,
                }
                for r in results
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/rooms/<room_id>/background/proximity', methods=['GET'])
def check_character_proximity(room_id):
    """
    Check if any separated characters might be nearby.
    Factors in relationships - strangers pass unnoticed, bonds pull people together.

    Query params:
        location: Current location description
        radius_km: Search radius in kilometers (default 5)
    """
    location = request.args.get('location', '')
    radius_km = float(request.args.get('radius_km', 5.0))

    # Get room's character relationships for probability modifiers
    room = data_store.get_room(room_id)
    character_relationships = room.character_relationships if room else []

    potential_encounters = autopilot_tracker.check_proximity(
        room_id,
        location,
        radius_km,
        character_relationships=character_relationships
    )

    return jsonify({
        'potential_encounters': potential_encounters,
        'search_location': location,
        'search_radius_km': radius_km,
    })


# ============================================================================
# Player Autopilot System - Your character acts while you're away
# ============================================================================

def _process_autopilot_tick(room_id: str, room, autopilot_model: str, autopilot_notes: str, interval_minutes: int):
    """
    Process a tick where the player is on autopilot.
    The understudy plays the player character, then present NPCs respond.
    """
    import asyncio
    from config import Message

    # Get player character info
    player_name = room.player_character_name or settings.user_name
    player_alignment = room.player_alignment or 'true_neutral'
    player_backstory = room.player_backstory or {}
    player_role = room.player_role or ''

    # Get recent conversation context
    recent_messages = room.messages[-20:] if room.messages else []
    conversation_context = "\n".join([
        f"{m.sender}: {m.content[:200]}..." if len(m.content) > 200 else f"{m.sender}: {m.content}"
        for m in recent_messages
    ])

    # Build understudy prompt
    understudy_prompt = f"""You are playing as {player_name} in a roleplaying game.

CHARACTER SHEET:
Name: {player_name}
Alignment: {player_alignment.replace('_', ' ').title()}
Role: {player_role.replace('_', ' ').title() if player_role else 'Survivor'}
Background: {player_backstory.get('title', 'Unknown')}
{player_backstory.get('description', '')}
Skills: {player_backstory.get('skills', 'Resourceful')}

SCENARIO:
{room.scenario[:500] if room.scenario else 'Post-apocalyptic survival'}

PLAYER'S NOTES FOR YOU:
{autopilot_notes if autopilot_notes else '(No specific guidance - play true to the character sheet)'}

RECENT CONVERSATION:
{conversation_context}

---

Time has passed ({interval_minutes} minutes in-game). As {player_name}, take a natural action or respond to the ongoing situation.
- Stay in character based on the alignment and background above
- Keep it brief (1-3 sentences of action/dialogue)
- Don't be too heroic or too passive - be realistic
- Reference the ongoing conversation naturally

CHARACTER ACTIONS (use when appropriate):
- [DM: question?] - Ask the game master for adjudication. Example: *tries the lock* [DM: can I pick it?]
- [ITEM: action] - Track item usage. Examples: [ITEM: used bandages], [ITEM: picked up flashlight], [ITEM: gave water to Sarah]
- [COMBAT: target] - Initiate combat. Example: *swings the bat* [COMBAT: zombie]
- [SEPARATED] - Signal you're splitting from the group. Use when fleeing different directions or staying behind.
- [SEEKING: location] - Indicate where you're heading. Example: [SEEKING: the pharmacy]

Respond as {player_name}:"""

    try:
        # Generate player action
        player_action = asyncio.run(provider_manager.generate_ollama(understudy_prompt, autopilot_model))

        if player_action and player_action.strip():
            # Add player message to room
            player_message = Message(
                sender=player_name,
                content=player_action.strip(),
                is_user=True,  # Mark as user message
                timestamp=datetime.now().isoformat()
            )
            data_store.add_message(room_id, player_message)

            # Record in understudy memory for review later
            try:
                from understudy import DecisionCategory, Confidence
                understudy_manager.record_decision(
                    character_id='player',
                    category=DecisionCategory.ROLEPLAY,
                    situation=f"Auto-tick: What does {player_name} do next?",
                    context=conversation_context[-500:],
                    options=["Act based on character sheet and alignment"],
                    decision=player_action.strip()[:200],
                    reasoning=f"Autopilot mode - playing as {player_alignment}",
                    confidence=Confidence.MEDIUM,
                    game_time=datetime.now().isoformat()
                )
            except Exception as e:
                print(f"[AUTOPILOT] Could not record decision: {e}")

            print(f"\033[93m[AUTOPILOT] {player_name}: {player_action.strip()[:100]}...\033[0m")

            # Track this tick's exchanges so each character sees previous responses
            tick_exchanges = [f"{player_name}: {player_action.strip()}"]

            # Process all character action tags for player (DM calls, items, combat, etc.)
            _process_autopilot_character_actions(
                room_id=room_id,
                room=room,
                character_id='player',
                character_name=player_name,
                response_text=player_action,
                tick_exchanges=tick_exchanges
            )

            # Now call present NPCs to respond - STAGGERED so they play off each other
            present_ids = room.present_character_ids or []
            partners = data_store.get_partners()

            for partner_id in present_ids[:3]:  # Limit to 3 NPCs responding
                partner = next((p for p in partners if p.id == partner_id), None)
                if not partner:
                    continue

                # Generate NPC response - pass all exchanges so far this tick
                npc_response = asyncio.run(_generate_npc_response_to_autopilot(
                    room=room,
                    partner=partner,
                    recent_exchanges=tick_exchanges,  # They see all previous actions this tick
                    model=autopilot_model
                ))

                if npc_response and npc_response.strip():
                    npc_message = Message(
                        sender=partner.name,
                        content=npc_response.strip(),
                        is_user=False,
                        timestamp=datetime.now().isoformat()
                    )
                    data_store.add_message(room_id, npc_message)

                    # Add to exchanges so NEXT character sees this response
                    tick_exchanges.append(f"{partner.name}: {npc_response.strip()}")

                    print(f"\033[96m[AUTOPILOT] {partner.name}: {npc_response.strip()[:100]}...\033[0m")

                    # Process all character action tags
                    _process_autopilot_character_actions(
                        room_id=room_id,
                        room=room,
                        character_id=partner.id,
                        character_name=partner.name,
                        response_text=npc_response,
                        tick_exchanges=tick_exchanges
                    )

            return True

    except Exception as e:
        print(f"\033[91m[AUTOPILOT] Error: {e}\033[0m")
        return False

    return False


def _process_autopilot_character_actions(
    room_id: str,
    room,
    character_id: str,
    character_name: str,
    response_text: str,
    tick_exchanges: list
):
    """
    Process all character action tags in an autopilot response.
    Handles: [DM:], [ITEM:], [COMBAT:], [SEPARATED], [SEEKING:]
    """
    import re

    # 1. Check for DM call: [DM: question?]
    dm_call_pattern = r'\[DM:\s*([^\]]+)\]'
    dm_match = re.search(dm_call_pattern, response_text)
    if dm_match:
        dm_question = dm_match.group(1).strip()
        print(f"\033[95m[AUTOPILOT] {character_name} calls DM: {dm_question}\033[0m")

        dm_response = _handle_character_dm_call(room_id, character_name, dm_question)
        if dm_response:
            data_store.add_message(room_id, dm_response)
            tick_exchanges.append(f"DM: {dm_response.content}")
            print(f"\033[95m[AUTOPILOT] DM responds: {dm_response.content[:100]}...\033[0m")

    # 2. Check for item action: [ITEM: action]
    item_pattern = r'\[ITEM:\s*([^\]]+)\]'
    item_match = re.search(item_pattern, response_text)
    if item_match:
        item_action = item_match.group(1).strip()
        print(f"\033[92m[AUTOPILOT] {character_name} item action: {item_action}\033[0m")

        result = _handle_character_item_action(room_id, character_id, character_name, item_action)
        if result:
            print(f"\033[92m[AUTOPILOT] Item processed: {result}\033[0m")

    # 3. Check for combat initiation: [COMBAT: target]
    combat_pattern = r'\[COMBAT:\s*([^\]]+)\]'
    combat_match = re.search(combat_pattern, response_text)
    if combat_match:
        combat_info_str = combat_match.group(1).strip()
        print(f"\033[91m[AUTOPILOT] {character_name} initiates combat: {combat_info_str}\033[0m")

        combat_info = _parse_combat_tag(combat_info_str)
        combat_info['initiated_by'] = character_name

        result = _handle_combat_initiation(room_id, character_id, character_name, combat_info)
        if result:
            print(f"\033[91m[AUTOPILOT] Combat started: {result.get('target', 'unknown')}\033[0m")
            # Add combat notification to tick exchanges
            tick_exchanges.append(f"[Combat initiated against {combat_info.get('target', 'enemy')}]")

    # 4. Check for separation: [SEPARATED]
    if '[SEPARATED]' in response_text:
        print(f"\033[93m[AUTOPILOT] {character_name} is separating from the group\033[0m")

        success = _handle_character_separation(room_id, character_id, character_name)
        if success:
            tick_exchanges.append(f"[{character_name} has separated from the group]")

    # 5. Check for seeking location: [SEEKING: location]
    seeking_pattern = r'\[SEEKING:\s*([^\]]+)\]'
    seeking_match = re.search(seeking_pattern, response_text)
    if seeking_match:
        destination = seeking_match.group(1).strip()
        print(f"\033[94m[AUTOPILOT] {character_name} is heading to: {destination}\033[0m")

        # Update character location tracking
        if hasattr(room, 'character_locations'):
            room.character_locations[character_id] = f"heading to {destination}"
            data_store.save()


async def _generate_npc_response_to_autopilot(room, partner, recent_exchanges: list, model: str):
    """Generate an NPC response to recent actions in this tick."""
    # Get recent messages for background context
    recent_messages = room.messages[-8:] if room.messages else []
    conversation_context = "\n".join([
        f"{m.sender}: {m.content[:150]}" for m in recent_messages
    ])

    # Format this tick's exchanges
    this_tick = "\n".join(recent_exchanges)

    prompt = f"""You are {partner.name}.

CHARACTER:
{partner.character_description}

SCENARIO:
{room.scenario[:300] if room.scenario else 'Ongoing roleplay'}

RECENT CONVERSATION (background):
{conversation_context}

---
WHAT JUST HAPPENED (this moment):
{this_tick}
---

Respond naturally as {partner.name}. Keep it brief (1-2 sentences).
You can respond to anyone who just spoke, not just the last person.
Stay in character.

CHARACTER ACTIONS (use when appropriate):
- [DM: question?] - Ask the game master for adjudication. Example: *tries the lock* [DM: can I pick it?]
- [ITEM: action] - Track item usage. Examples: [ITEM: used bandages], [ITEM: picked up flashlight], [ITEM: gave water to Sarah]
- [COMBAT: target] - Initiate combat. Example: *swings the bat* [COMBAT: zombie]
- [SEPARATED] - Signal you're splitting from the group. Use when fleeing different directions or staying behind.
- [SEEKING: location] - Indicate where you're heading. Example: [SEEKING: the pharmacy]

{partner.name}:"""

    try:
        response = await provider_manager.generate_ollama(prompt, model)
        return response
    except Exception as e:
        print(f"[AUTOPILOT] NPC response error: {e}")
        return None


@app.route('/rooms/<room_id>/autopilot/catchup', methods=['POST'])
def get_autopilot_catchup(room_id):
    """
    Generate a catch-up summary for the player returning from autopilot.
    This is a DM whisper explaining what happened while they were away.
    """
    import asyncio

    data = request.json or {}
    ticks_while_away = data.get('ticks_while_away', 0)
    enabled_at = data.get('enabled_at', '')

    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    # Get player name
    player_name = room.player_character_name or settings.user_name

    # Get recent messages (covering the autopilot period)
    # Estimate ~2-3 messages per tick
    messages_to_review = min(ticks_while_away * 4, 30)
    recent_messages = room.messages[-messages_to_review:] if room.messages else []

    if not recent_messages:
        return jsonify({
            'summary': f"Welcome back! Nothing significant happened while you were away.",
            'ticks': ticks_while_away
        })

    # Format messages for summary
    message_log = "\n".join([
        f"{m.sender}: {m.content}" for m in recent_messages
    ])

    # Generate summary
    summary_prompt = f"""You are the Dungeon Master giving a quick catch-up to {player_name} who just returned from autopilot mode.

Their character was being played automatically for {ticks_while_away} tick(s) (roughly {ticks_while_away * 30} minutes of in-game time).

Here's what happened:

{message_log}

---

Give a BRIEF, conversational summary (2-4 sentences) of what happened while they were away.
Focus on:
- Key decisions their character made
- Any important developments
- Where things currently stand

Speak directly to the player in a friendly DM voice. Example:
"While you were away, your character helped search the warehouse and found some medical supplies. Sarah seems more trusting now. You're currently taking a break by the loading dock."

Summary:"""

    try:
        use_model = room.room_model or settings.storybuilder_model
        summary = asyncio.run(provider_manager.generate_ollama(summary_prompt, use_model))

        return jsonify({
            'summary': summary.strip() if summary else "Welcome back! Check the recent messages to see what happened.",
            'ticks': ticks_while_away,
            'messages_reviewed': len(recent_messages)
        })

    except Exception as e:
        print(f"[AUTOPILOT] Catch-up error: {e}")
        return jsonify({
            'summary': f"Welcome back! You were on autopilot for {ticks_while_away} ticks. Check the chat to see what happened.",
            'ticks': ticks_while_away
        })


# ============================================================================
# Understudy System - Your character's shadow self
# ============================================================================

@app.route('/understudy/<character_id>', methods=['GET'])
def get_understudy_memory(character_id):
    """Get the understudy's accumulated memory for a character."""
    memory = understudy_manager.get_or_create(character_id)
    return jsonify(memory.to_dict())


@app.route('/understudy/<character_id>/review', methods=['GET'])
def get_understudy_review(character_id):
    """
    Get the 'catching up' summary - what happened while you were away.
    This is the understudy saying "Here's what I did, let's talk about it."
    """
    summary = understudy_manager.get_review_summary(character_id)
    return jsonify(summary)


@app.route('/understudy/<character_id>/decisions', methods=['GET'])
def get_understudy_decisions(character_id):
    """Get all decisions made by the understudy."""
    memory = understudy_manager.get_or_create(character_id)
    unreviewed_only = request.args.get('unreviewed', 'false').lower() == 'true'

    if unreviewed_only:
        decisions = [d for d in memory.decisions if not d.reviewed]
    else:
        decisions = memory.decisions

    return jsonify({
        'decisions': [d.to_dict() for d in decisions],
        'count': len(decisions)
    })


@app.route('/understudy/<character_id>/feedback', methods=['POST'])
def submit_understudy_feedback(character_id):
    """
    Submit feedback on an understudy decision.

    Body: {
        "decision_id": "dec_xxx",
        "feedback": "wrong",  # "perfect", "good", "acceptable", "wrong", "catastrophic"
        "note": "We never give cheese. Family thing.",
        "new_rule": "Never give away cheese"  # Optional - creates a learned rule
    }
    """
    data = request.json or {}
    decision_id = data.get('decision_id')
    feedback_str = data.get('feedback', 'acceptable')
    note = data.get('note', '')
    new_rule = data.get('new_rule')

    if not decision_id:
        return jsonify({'error': 'decision_id required'}), 400

    try:
        feedback = FeedbackType(feedback_str)
    except ValueError:
        return jsonify({'error': f'Invalid feedback type: {feedback_str}'}), 400

    understudy_manager.record_feedback(
        character_id=character_id,
        decision_id=decision_id,
        feedback=feedback,
        note=note,
        new_rule=new_rule,
    )

    return jsonify({'message': 'Feedback recorded', 'decision_id': decision_id})


@app.route('/understudy/<character_id>/rules', methods=['GET'])
def get_understudy_rules(character_id):
    """Get learned rules for a character."""
    memory = understudy_manager.get_or_create(character_id)
    return jsonify({
        'rules': [r.to_dict() for r in memory.rules],
        'count': len(memory.rules)
    })


@app.route('/understudy/<character_id>/rules', methods=['POST'])
def add_understudy_rule(character_id):
    """
    Manually add a rule (without going through feedback).

    Body: {
        "rule": "Never trust Marcus",
        "category": "social",
        "context": "He stole from us once",
        "is_prohibition": true
    }
    """
    data = request.json or {}
    rule_text = data.get('rule')
    category_str = data.get('category', 'unknown')
    context = data.get('context', '')
    is_prohibition = data.get('is_prohibition', True)

    if not rule_text:
        return jsonify({'error': 'rule text required'}), 400

    try:
        category = DecisionCategory(category_str)
    except ValueError:
        category = DecisionCategory.UNKNOWN

    rule = UnderstudyRule(
        id=f"rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        created=datetime.now().isoformat(),
        from_decision_id=None,
        category=category,
        rule=rule_text,
        context=context,
        is_prohibition=is_prohibition,
        last_reinforced=datetime.now().isoformat(),
    )

    memory = understudy_manager.get_or_create(character_id)
    memory.add_rule(rule)
    understudy_manager.save(character_id)

    return jsonify({'message': 'Rule added', 'rule': rule.to_dict()})


@app.route('/understudy/<character_id>/echoes', methods=['GET'])
def get_understudy_echoes(character_id):
    """Get learned echoes (phrases, habits) for a character."""
    memory = understudy_manager.get_or_create(character_id)
    return jsonify({
        'echoes': [e.to_dict() for e in memory.echoes],
        'count': len(memory.echoes)
    })


@app.route('/understudy/<character_id>/echoes', methods=['POST'])
def add_understudy_echo(character_id):
    """
    Teach the understudy a phrase or habit.

    Body: {
        "echo_type": "phrase",  # "phrase", "habit", "reaction", "opinion"
        "trigger": "When asked about the past",
        "content": "I don't talk about before."
    }
    """
    data = request.json or {}
    echo_type = data.get('echo_type', 'phrase')
    trigger = data.get('trigger', '')
    content = data.get('content', '')

    if not content:
        return jsonify({'error': 'content required'}), 400

    understudy_manager.learn_echo(
        character_id=character_id,
        echo_type=echo_type,
        trigger=trigger,
        content=content,
        source="manual",
    )

    return jsonify({'message': 'Echo learned'})


@app.route('/understudy/<character_id>/dreams', methods=['GET'])
def get_understudy_dreams(character_id):
    """Get dreams for a character."""
    memory = understudy_manager.get_or_create(character_id)
    limit = request.args.get('limit', 10, type=int)
    return jsonify({
        'dreams': [d.to_dict() for d in memory.dreams[-limit:]],
        'count': len(memory.dreams)
    })


@app.route('/understudy/<character_id>/dreams', methods=['POST'])
def record_understudy_dream(character_id):
    """
    Record a dream (usually called by the story daemon when character sleeps).

    Body: {
        "title": "The Door That Wasn't There",
        "narrative": "You stood before a door...",
        "influences": ["Recent combat", "Thinking about home"],
        "tone": "anxious",
        "game_time": "Day 3, Night",
        "is_significant": false
    }
    """
    data = request.json or {}

    dream = understudy_manager.record_dream(
        character_id=character_id,
        title=data.get('title', 'Untitled Dream'),
        narrative=data.get('narrative', ''),
        influences=data.get('influences', []),
        tone=data.get('tone', 'neutral'),
        game_time=data.get('game_time', ''),
        is_significant=data.get('is_significant', False),
    )

    return jsonify({'message': 'Dream recorded', 'dream': dream.to_dict()})


@app.route('/understudy/<character_id>/prompt', methods=['POST'])
def get_understudy_prompt(character_id):
    """
    Get the prompt that would be sent to Ollama for a decision.
    Useful for debugging what the understudy "knows".

    Body: {"situation": "Marcus asks if you have any cheese to spare."}
    """
    data = request.json or {}
    situation = data.get('situation', '')

    if not situation:
        return jsonify({'error': 'situation required'}), 400

    prompt = understudy_manager.build_understudy_prompt(character_id, situation)

    return jsonify({
        'character_id': character_id,
        'situation': situation,
        'prompt': prompt
    })


@app.route('/understudy/<character_id>/decide', methods=['POST'])
def understudy_decide(character_id):
    """
    Have the understudy make a decision (calls Ollama).

    Body: {
        "situation": "Marcus asks for cheese",
        "context": "You're at the enclave, Marcus looks hungry",
        "options": ["give cheese", "refuse", "trade"],
        "category": "resource",
        "game_time": "Day 3, 14:00"
    }
    """
    data = request.json or {}
    situation = data.get('situation', '')
    context = data.get('context', '')
    options = data.get('options', [])
    category_str = data.get('category', 'unknown')
    game_time = data.get('game_time', '')

    if not situation:
        return jsonify({'error': 'situation required'}), 400

    try:
        category = DecisionCategory(category_str)
    except ValueError:
        category = DecisionCategory.UNKNOWN

    # Build the prompt
    prompt = understudy_manager.build_understudy_prompt(character_id, situation)

    if options:
        prompt += f"\n\nOptions to consider: {', '.join(options)}"

    # Call Ollama
    try:
        import requests
        response = requests.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=60
        )
        if response.ok:
            ollama_response = response.json().get('response', '')
        else:
            ollama_response = "I'm not sure what to do here."
    except Exception as e:
        ollama_response = f"Error consulting understudy: {e}"

    # Parse the response to extract decision and confidence
    # This is simple heuristic parsing - could be improved
    decision = ollama_response.split('\n')[0] if ollama_response else "No decision"
    confidence = Confidence.UNCERTAIN

    if "definitely" in ollama_response.lower() or "certainly" in ollama_response.lower():
        confidence = Confidence.CERTAIN
    elif "not sure" in ollama_response.lower() or "uncertain" in ollama_response.lower():
        confidence = Confidence.GUESSING
    elif "flip" in ollama_response.lower() or "coin" in ollama_response.lower():
        confidence = Confidence.COIN_FLIP

    # Record the decision
    decision_obj = understudy_manager.record_decision(
        character_id=character_id,
        category=category,
        situation=situation,
        context=context,
        options=options,
        decision=decision,
        reasoning=ollama_response,
        confidence=confidence,
        game_time=game_time,
    )

    return jsonify({
        'decision': decision_obj.to_dict(),
        'full_response': ollama_response
    })


@app.route('/rooms/<room_id>/player-return', methods=['POST'])
def player_return(room_id):
    """
    Called when a player returns from being away.
    Returns the "while you were away" summary.

    Body: {"player_id": "user" or specific ID}
    """
    data = request.json or {}
    player_id = data.get('player_id', 'user')

    summary = on_player_returns(room_id, player_id)

    # Generate a narrative summary if there's content
    narrative = []
    if summary["was_on_autopilot"]:
        narrative.append("You were on autopilot while you were away.")

        if summary["journal_entries"]:
            narrative.append(f"Your understudy recorded {len(summary['journal_entries'])} events.")

        if summary["understudy_decisions"]:
            narrative.append(f"There are {len(summary['understudy_decisions'])} decisions that need your review.")

        if summary["dreams"]:
            narrative.append(f"You had {len(summary['dreams'])} dreams while sleeping.")

        if summary["world_events"]:
            narrative.append(f"{len(summary['world_events'])} notable events happened in the world.")

    summary["narrative"] = " ".join(narrative) if narrative else "Welcome back. Nothing major happened while you were away."

    return jsonify(summary)


@app.route('/rooms/<room_id>/connect', methods=['POST'])
def connect_to_room(room_id):
    """
    Connect a player to a room. This:
    - Registers them with the story daemon
    - Disables autopilot if it was on
    - Returns the "while you were away" summary
    """
    global story_daemon

    data = request.json or {}
    player_id = data.get('player_id', 'user')

    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    # Connect to story daemon
    if story_daemon:
        story_daemon.player_connected(room_id, player_id)

    # Get the return summary (includes autopilot disable)
    summary = on_player_returns(room_id, player_id)

    # Sync world time if realtime mode
    if story_daemon:
        world_state = story_daemon.get_world_state(room_id)
        if world_state and world_state.time_mode == "realtime":
            world_state.sync_to_realtime()
            summary["world_time"] = {
                "day": world_state.game_day,
                "hour": world_state.game_hour,
                "minute": getattr(world_state, 'game_minute', 0),
                "time_of_day": world_state.time_of_day,
            }

    return jsonify({
        'connected': True,
        'room_id': room_id,
        'player_id': player_id,
        **summary
    })


@app.route('/rooms/<room_id>/disconnect', methods=['POST'])
def disconnect_from_room(room_id):
    """
    Disconnect a player from a room. This:
    - Unregisters them from the story daemon
    - Optionally enables autopilot
    """
    global story_daemon

    data = request.json or {}
    player_id = data.get('player_id', 'user')
    enable_autopilot = data.get('enable_autopilot', True)

    # Disconnect from story daemon
    if story_daemon:
        story_daemon.player_disconnected(room_id, player_id)

    # Enable autopilot if requested
    if enable_autopilot:
        pc = autopilot_tracker.get(player_id, room_id)
        if pc:
            pc.autopilot_enabled = True
            pc.autopilot_started = datetime.now().isoformat()
            autopilot_tracker._save()

    return jsonify({
        'disconnected': True,
        'autopilot_enabled': enable_autopilot
    })


# ============================================================================
# Relationship Map - Visual character relationships
# ============================================================================

@app.route('/rooms/<room_id>/relationships', methods=['GET'])
def get_relationships(room_id):
    """
    Get the relationship map for a room.

    Query params:
    - viewer_id: Character whose perspective to use (optional)
    - dm_view: If true, show all secrets (default false)
    """
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    viewer_id = request.args.get('viewer_id')
    dm_view = request.args.get('dm_view', 'false').lower() == 'true'

    # Get partners and NPCs
    partners = data_store.get_partners()
    room_partners = room.get_partners_in_room(partners)
    npcs = npc_registry.get_npcs_in_world(room_id)

    # Build the map
    rel_map = get_relationship_map(
        room_id=room_id,
        partners=room_partners,
        npcs=npcs,
        viewer_id=viewer_id,
        is_dm_view=dm_view,
        npc_registry=npc_registry,
    )

    return jsonify(rel_map.to_dict())


@app.route('/rooms/<room_id>/relationships/between', methods=['GET'])
def get_relationship_between(room_id):
    """
    Get the specific relationship between two entities.

    Query params:
    - source_id: First entity
    - target_id: Second entity
    """
    source_id = request.args.get('source_id')
    target_id = request.args.get('target_id')

    if not source_id or not target_id:
        return jsonify({'error': 'source_id and target_id required'}), 400

    # Check if they're NPCs with memories
    source_npc = npc_registry.get_npc(source_id)
    target_npc = npc_registry.get_npc(target_id)

    result = {
        'source_id': source_id,
        'target_id': target_id,
        'relationship': 'unknown'
    }

    # Check NPC memories
    if source_npc and target_id in source_npc.player_memories:
        memory = source_npc.player_memories[target_id]
        result['source_to_target'] = {
            'trust': memory.trust_level,
            'interactions': memory.interaction_count,
            'last_interaction': memory.last_interaction,
        }

    if target_npc and source_id in target_npc.player_memories:
        memory = target_npc.player_memories[source_id]
        result['target_to_source'] = {
            'trust': memory.trust_level,
            'interactions': memory.interaction_count,
            'last_interaction': memory.last_interaction,
        }

    # Check grudges
    if source_npc:
        grudge = source_npc.get_grudge_against(target_id)
        if grudge:
            result['grudge'] = {
                'severity': grudge.severity.value,
                'reason': grudge.reason,
                'can_forgive': grudge.can_forgive,
            }

    return jsonify(result)


# ============================================================================
# Story Daemon - Background world progression
# ============================================================================

@app.route('/story/status', methods=['GET'])
def get_story_status():
    """Get the current status of the story daemon."""
    global story_daemon

    if not story_daemon:
        return jsonify({
            'running': False,
            'message': 'Story daemon not initialized'
        })

    return jsonify({
        'running': story_daemon.running,
        'tick_count': story_daemon.tick_count,
        'last_tick': story_daemon.last_tick,
        'events_generated': len(story_daemon.event_log),
    })


@app.route('/story/start', methods=['POST'])
def start_story_daemon():
    """Start the story daemon."""
    global story_daemon

    data = request.json or {}
    tick_interval = data.get('tick_interval', 300)  # 5 minutes default

    # Create a simple generate function using ollama
    def ollama_generate(prompt: str) -> str:
        try:
            import requests
            response = requests.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=60
            )
            if response.ok:
                return response.json().get('response', '')
            return ""
        except Exception as e:
            print(f"[STORY] Ollama generate failed: {e}")
            return ""

    story_daemon = init_story_daemon(
        npc_registry=npc_registry,
        ollama_generate_func=ollama_generate,
        tick_interval=tick_interval,
    )

    story_daemon.start()

    return jsonify({
        'message': 'Story daemon started',
        'tick_interval': tick_interval
    })


@app.route('/story/stop', methods=['POST'])
def stop_story_daemon_route():
    """Stop the story daemon."""
    global story_daemon

    if story_daemon:
        stop_story_daemon()
        story_daemon = None

    return jsonify({'message': 'Story daemon stopped'})


@app.route('/story/tick', methods=['POST'])
def manual_story_tick():
    """Manually trigger a story tick (for testing)."""
    global story_daemon

    if not story_daemon:
        return jsonify({'error': 'Story daemon not initialized'}), 400

    # Run a single tick
    events = story_daemon._do_tick()

    return jsonify({
        'events': [e.to_dict() for e in events] if events else [],
        'tick_count': story_daemon.tick_count
    })


@app.route('/story/events', methods=['GET'])
def get_story_events():
    """Get recent story events."""
    global story_daemon

    if not story_daemon:
        return jsonify({'events': [], 'message': 'Story daemon not initialized'})

    limit = request.args.get('limit', 50, type=int)
    severity = request.args.get('severity')  # Filter by severity

    events = story_daemon.event_log[-limit:]

    if severity:
        events = [e for e in events if e.severity.value == severity]

    return jsonify({
        'events': [e.to_dict() for e in events],
        'count': len(events)
    })


@app.route('/rooms/<room_id>/world-state', methods=['GET'])
def get_world_state(room_id):
    """Get the world state for a room."""
    global story_daemon

    if not story_daemon:
        return jsonify({'error': 'Story daemon not initialized'}), 400

    state = story_daemon.get_world_state(room_id)
    if not state:
        return jsonify({'error': 'No world state for this room'}), 404

    return jsonify({
        'world_id': state.world_id,
        'time_of_day': state.time_of_day,
        'weather': state.weather,
        'mood': state.mood,
        'game_day': state.game_day,
        'game_hour': state.game_hour,
        'threat_level': state.threat_level,
        'recent_events': state.recent_events,
        'active_threats': state.active_threats,
    })


@app.route('/rooms/<room_id>/world-state', methods=['POST'])
def update_world_state(room_id):
    """Update world state for a room."""
    global story_daemon

    if not story_daemon:
        return jsonify({'error': 'Story daemon not initialized'}), 400

    data = request.json or {}

    state = story_daemon.get_or_create_world_state(room_id)

    # Update fields if provided
    if 'time_of_day' in data:
        state.time_of_day = data['time_of_day']
    if 'weather' in data:
        state.weather = data['weather']
    if 'mood' in data:
        state.mood = data['mood']
    if 'threat_level' in data:
        state.threat_level = float(data['threat_level'])
    if 'game_day' in data:
        state.game_day = int(data['game_day'])
    if 'game_hour' in data:
        state.game_hour = int(data['game_hour'])

    return jsonify({
        'message': 'World state updated',
        'world_id': state.world_id,
        'time_of_day': state.time_of_day,
        'weather': state.weather,
        'mood': state.mood,
        'game_day': state.game_day,
        'game_hour': state.game_hour,
        'threat_level': state.threat_level,
    })


@app.route('/rooms/<room_id>/advance-time', methods=['POST'])
def advance_game_time(room_id):
    """Advance game time for a room."""
    global story_daemon

    if not story_daemon:
        return jsonify({'error': 'Story daemon not initialized'}), 400

    data = request.json or {}
    hours = data.get('hours', 1)

    state = story_daemon.get_or_create_world_state(room_id)
    old_day = state.game_day
    old_hour = state.game_hour

    state.game_hour += hours
    while state.game_hour >= 24:
        state.game_hour -= 24
        state.game_day += 1

    # Update time of day
    if 5 <= state.game_hour < 8:
        state.time_of_day = "dawn"
    elif 8 <= state.game_hour < 18:
        state.time_of_day = "day"
    elif 18 <= state.game_hour < 21:
        state.time_of_day = "dusk"
    else:
        state.time_of_day = "night"

    return jsonify({
        'message': f'Advanced {hours} hour(s)',
        'from': f'Day {old_day}, {old_hour}:00',
        'to': f'Day {state.game_day}, {state.game_hour}:00',
        'time_of_day': state.time_of_day,
    })


@app.route('/rooms/<room_id>/realtime-sync', methods=['POST'])
def enable_realtime_sync(room_id):
    """
    Enable real-time sync for a world - game time matches real time.

    Body: {
        "timezone": "America/Denver",  # or "America/Chicago", "America/New_York", etc.
        "start_day": 1  # What game day is today
    }

    After enabling, the world's time will automatically match your local time.
    If it's 3pm in Denver, it's 3pm in the zombie apocalypse.
    """
    global story_daemon

    if not story_daemon:
        return jsonify({'error': 'Story daemon not initialized'}), 400

    data = request.json or {}
    timezone = data.get('timezone', 'America/Denver')
    start_day = data.get('start_day', 1)

    state = story_daemon.get_or_create_world_state(room_id)
    state.time_mode = "realtime"
    state.timezone = timezone
    state.realtime_start_day = start_day
    state.realtime_epoch = datetime.now().strftime('%Y-%m-%d')

    # Do initial sync
    state.sync_to_realtime()

    return jsonify({
        'message': 'Real-time sync enabled',
        'timezone': timezone,
        'game_day': state.game_day,
        'game_hour': state.game_hour,
        'game_minute': state.game_minute,
        'time_of_day': state.time_of_day,
    })


@app.route('/rooms/<room_id>/sync-time', methods=['POST'])
def sync_world_time(room_id):
    """Manually sync world time to real time (if realtime mode is enabled)."""
    global story_daemon

    if not story_daemon:
        return jsonify({'error': 'Story daemon not initialized'}), 400

    state = story_daemon.get_or_create_world_state(room_id)

    if state.time_mode != "realtime":
        return jsonify({'error': 'World is not in realtime mode'}), 400

    state.sync_to_realtime()

    return jsonify({
        'synced': True,
        'game_day': state.game_day,
        'game_hour': state.game_hour,
        'game_minute': state.game_minute,
        'time_of_day': state.time_of_day,
        'timezone': state.timezone,
    })


# ============================================================================
# Weather Sync - Real weather for real worlds
# ============================================================================

@app.route('/weather/<location>', methods=['GET'])
def get_weather(location):
    """
    Get current weather for a location.

    location can be:
    - City: "Denver,CO" or "Denver"
    - Zip: "80202"
    - Coordinates: "39.7392,-104.9903"

    Requires OPENWEATHERMAP_API_KEY environment variable.
    """
    weather = weather_sync.get_weather(location)
    if not weather:
        return jsonify({'error': 'Could not fetch weather'}), 500

    return jsonify(weather.to_dict())


@app.route('/rooms/<room_id>/weather-sync', methods=['POST'])
def enable_weather_sync(room_id):
    """
    Enable weather sync for a room - game weather matches real weather.

    Body: {
        "location": "Denver, CO"  # or zip code, or coordinates
    }
    """
    global story_daemon

    if not story_daemon:
        return jsonify({'error': 'Story daemon not initialized'}), 400

    data = request.json or {}
    location = data.get('location', 'Denver, CO')

    state = story_daemon.get_or_create_world_state(room_id)

    # Store the location for future syncs
    if not hasattr(state, 'weather_location'):
        # Add attribute dynamically (will be lost on restart, but that's ok for now)
        state.weather_location = location

    # Do the sync
    weather = weather_sync.get_weather(location)
    if not weather:
        return jsonify({'error': 'Could not fetch weather'}), 500

    state.weather = weather.condition

    return jsonify({
        'message': 'Weather synced',
        'location': weather.location,
        'condition': weather.condition,
        'description': weather.description,
        'temperature_f': weather.temperature_f,
        'narrative': weather.get_narrative_description(),
        'mood_modifier': weather.mood_modifier,
    })


@app.route('/rooms/<room_id>/weather', methods=['GET'])
def get_room_weather(room_id):
    """Get current weather for a room (from world state)."""
    global story_daemon

    if not story_daemon:
        return jsonify({'weather': 'clear', 'source': 'default'})

    state = story_daemon.get_world_state(room_id)
    if not state:
        return jsonify({'weather': 'clear', 'source': 'default'})

    result = {
        'weather': state.weather,
        'source': 'world_state'
    }

    # If we have a synced location, include full weather data
    if hasattr(state, 'weather_location'):
        weather = weather_sync.get_weather(state.weather_location)
        if weather:
            result['details'] = weather.to_dict()
            result['narrative'] = weather.get_narrative_description()

    return jsonify(result)


# ============================================================================
# DM (Dungeon Master) System - Authoritative world/reality decisions
# ============================================================================

# Store private DM conversation per room (not persisted to disk)
_dm_private_history = {}  # room_id -> [{"role": "user"/"assistant", "content": str}]


def _build_dm_character_info(partners: list) -> str:
    """Build character info for the DM."""
    info_parts = []
    for p in partners:
        parts = [f"**{p.name}**"]
        parts.append(f"  Personality: {p.get_character()[:200]}")

        # Include hidden knowledge for the DM
        if p.secret:
            parts.append(f"  SECRET: {p.secret}")
        if p.wound:
            parts.append(f"  WOUND: {p.wound}")
        if p.want:
            parts.append(f"  WANT: {p.want}")
        if p.fear:
            parts.append(f"  FEAR: {p.fear}")
        if p.skill:
            parts.append(f"  SKILL: {p.skill}")

        # Include fatigue state
        fatigue_effects = fatigue_tracker.get_fatigue_effects(p.id)
        if fatigue_effects.get('level') != 'unknown':
            level = fatigue_effects['level']
            if level not in ('rested', 'fine'):  # Only mention if notably tired
                parts.append(f"  FATIGUE: {level.upper()} - {fatigue_effects.get('description', '')}")

        # Include notable inventory items (weapons, quest items)
        inv = inventory_tracker.get_inventory(p.id)
        if inv and inv.items:
            equipped = [i.name for i in inv.items if i.is_equipped]
            quest_items = [i.name for i in inv.items if i.is_quest_item and not i.is_equipped]
            if equipped:
                parts.append(f"  EQUIPPED: {', '.join(equipped)}")
            if quest_items:
                parts.append(f"  KEY ITEMS: {', '.join(quest_items)}")

        info_parts.append("\n".join(parts))
    return "\n\n".join(info_parts)


def _build_simple_dm_context(room, room_partners: list) -> str:
    """Build simple DM context - ALL subsystems feed into this."""
    global story_daemon
    sections = []

    # World State (time, weather, threat level) - THIS IS THE WORLD
    if story_daemon:
        world_state = story_daemon.get_world_state(room.id)
        if world_state:
            world_lines = ["=== WORLD STATE ==="]
            world_lines.append(f"Day {world_state.game_day}, {world_state.game_hour}:{getattr(world_state, 'game_minute', 0):02d} ({world_state.time_of_day})")
            world_lines.append(f"Weather: {world_state.weather}")

            # Add weather narrative if we have a synced location
            if hasattr(world_state, 'weather_location') and world_state.weather_location:
                weather_data = weather_sync.get_weather(world_state.weather_location)
                if weather_data:
                    world_lines.append(f"  {weather_data.get_narrative_description()}")
                    world_lines.append(f"  Temperature: {weather_data.temperature_f:.0f}°F (feels like {weather_data.feels_like_f:.0f}°F)")

            world_lines.append(f"Mood: {world_state.mood}")
            if world_state.threat_level > 0:
                threat_words = ["calm", "uneasy", "tense", "dangerous", "critical", "apocalyptic"]
                threat_idx = min(int(world_state.threat_level / 2), len(threat_words) - 1)
                world_lines.append(f"Threat Level: {world_state.threat_level}/10 ({threat_words[threat_idx]})")
            if world_state.active_threats:
                world_lines.append(f"Active Threats: {', '.join(world_state.active_threats)}")
            if world_state.recent_events:
                world_lines.append("Recent Events:")
                for event in world_state.recent_events[-3:]:
                    world_lines.append(f"  - {event}")
            sections.append("\n".join(world_lines))

    # Zombie type context - affects encounter generation
    zombie_type = getattr(room, 'zombie_type', None)
    if zombie_type:
        zombie_context = {
            'shamblers': "ZOMBIE BEHAVIOR: Shamblers - slow-moving, unintelligent, only react to noise/movement within close range. Dangerous in hordes but individuals can be avoided or outmaneuvered. Think Romero/Walking Dead zombies. Generate zombie encounters accordingly - they shamble, they're dumb, they're only dangerous when they swarm.",
            'runners': "ZOMBIE BEHAVIOR: Runners - fast, aggressive, sprint at full speed when they detect prey. Highly attracted to noise. One zombie alerting others creates cascading swarms. Think 28 Days Later/Dawn of the Dead remake. Encounters are DANGEROUS - if they see you, you're running for your life. Generate intense, terrifying zombie encounters.",
            'nightmares': "ZOMBIE BEHAVIOR: Nightmares - mutated, unpredictable horrors. Some crawl on walls, some are massive, some split into pieces. Physics-defying abominations. Think Resident Evil/Silent Hill. Every zombie encounter is a boss fight. This world is HELL. Generate horrifying, creative zombie mutations."
        }.get(zombie_type)
        if zombie_context:
            sections.append(f"=== ZOMBIE TYPE ===\n{zombie_context}")

    # Population density - affects encounter frequency
    population_density = getattr(room, 'population_density', None)
    if population_density:
        initial_count = getattr(room, 'initial_zombie_count', 0)
        killed = getattr(room, 'zombies_killed', 0)
        remaining = max(0, initial_count - killed)
        cleared_areas = getattr(room, 'areas_cleared', [])

        density_context = _get_density_dm_context(population_density, remaining)

        # Add kill tracking info
        if killed > 0:
            density_context += f"\n\nZOMBIE KILLS: {killed} confirmed kills. Estimated {remaining} remaining in area."
            if remaining < initial_count * 0.5:
                density_context += " The area is noticeably quieter."
            if remaining < initial_count * 0.2:
                density_context += " You've made a serious dent in the population."
            if remaining < 50:
                density_context += " The area could be cleared with sustained effort."
            if remaining == 0:
                density_context += " THIS AREA IS CLEARED. No zombies remain (though some may wander in from elsewhere over time)."

        if cleared_areas:
            density_context += f"\n\nCLEARED AREAS: {', '.join(cleared_areas)}"
            density_context += "\n(Note: Cleared areas don't stay clear forever. Without maintenance, zombies wander in from surrounding areas over days/weeks. Proximity to high-density zones means faster repopulation.)"

        sections.append(f"=== POPULATION DENSITY ===\n{density_context}")

    # Hidden zombie rules - rolled at game start, players discover through play
    zombie_rules = getattr(room, 'zombie_rules', {})
    if zombie_rules:
        rules_context = get_zombie_rules_dm_context(zombie_rules)
        sections.append(rules_context)

    # Shelter type - affects story tone and objectives
    shelter_type = getattr(room, 'shelter_type', None)
    if shelter_type:
        shelter_context = get_shelter_dm_context(shelter_type)
        if shelter_context:
            sections.append(shelter_context)

    # Character info (AI partners)
    char_info = _build_dm_character_info(room_partners)
    if char_info:
        sections.append(f"=== CHARACTERS ===\n{char_info}")

    # Player character info (from StoryBuilder world creation)
    if room.player_character_name:
        player_lines = ["=== PLAYER CHARACTER ==="]
        player_lines.append(f"Name: {room.player_character_name}")
        if room.player_gender:
            player_lines.append(f"Gender: {room.player_gender}")
        if room.player_backstory:
            bs = room.player_backstory
            if bs.get('title'):
                player_lines.append(f"Background: {bs['title']}")
            if bs.get('description'):
                player_lines.append(f"  {bs['description']}")
            if bs.get('skills'):
                player_lines.append(f"Skills: {bs['skills']}")
        if room.player_alignment:
            player_lines.append(f"Alignment: {room.player_alignment.replace('_', ' ')}")
        if room.player_role:
            player_lines.append(f"Role: {room.player_role.replace('_', ' ')}")
        player_lines.append(f"(This is the human player's character - {settings.user_name} is playing as {room.player_character_name})")
        sections.append("\n".join(player_lines))

    # Autopilot status - who's away?
    autopilot_lines = []
    for p in room_partners:
        pc = autopilot_tracker.get(p.id, room.id)
        if pc and pc.autopilot_enabled:
            autopilot_lines.append(f"  {p.name}: ON AUTOPILOT (understudy active)")
    if autopilot_lines:
        sections.append("=== AUTOPILOT STATUS ===\n" + "\n".join(autopilot_lines))

    # Character locations (for separated characters)
    char_locations = room.character_locations or {}
    present_ids = room.present_character_ids or []
    if char_locations:
        location_lines = ["=== CHARACTER LOCATIONS ==="]
        # Player location
        if room.player_location:
            location_lines.append(f"  Player ({room.player_character_name or settings.user_name}): {room.player_location}")
        # Separated character locations
        all_partners = data_store.get_partners()
        for char_id, location in char_locations.items():
            if char_id not in present_ids:  # Only show separated characters
                partner = next((p for p in all_partners if p.id == char_id), None)
                if partner:
                    location_lines.append(f"  {partner.name} (SEPARATED): {location}")
        if len(location_lines) > 1:  # Only add if we have actual location data
            sections.append("\n".join(location_lines))

    # Active Combat
    encounter = EncounterManager.get(room.id)
    if encounter and encounter.is_active:
        combat_lines = ["=== ACTIVE COMBAT ==="]
        combat_lines.append(f"Round {encounter.round}")
        current = encounter.get_current_combatant()
        if current:
            combat_lines.append(f"Current Turn: {current.stats.name}")
        for c_id, c in encounter.combatants.items():
            status = "DEAD" if not c.stats.is_alive else f"HP {c.stats.hp_current}/{c.stats.hp_max}"
            combat_lines.append(f"  [{c.team}] {c.stats.name}: {status}")
        sections.append("\n".join(combat_lines))

    # Scenario
    if room.scenario:
        sections.append(f"=== SCENARIO ===\n{room.scenario}")

    # Genre
    genre = 'fantasy'
    if hasattr(room, 'genre') and room.genre:
        genre = room.genre
        sections.append(f"Genre: {room.genre}")

    # Fatigue status for all tracked characters
    fatigue_context = fatigue_tracker.get_all_fatigue_context()
    if fatigue_context and "No fatigue tracking" not in fatigue_context:
        sections.append(f"=== FATIGUE STATUS ===\n{fatigue_context}")

    # Inventory summary for characters
    inventory_context = inventory_tracker.get_all_inventories_context()
    if inventory_context and "No inventories tracked" not in inventory_context:
        sections.append(f"=== INVENTORY ===\n{inventory_context}")

    # Pending consequences
    consequence_context = consequence_engine.get_consequence_context(room.id)
    if consequence_context:
        sections.append(consequence_context)

    # World map info
    map_context = cartographer.get_dm_context(room.id)
    if map_context and "No map data" not in map_context:
        sections.append(map_context)

    # NPC info
    npc_context = get_npc_dm_context(room.id)
    if npc_context:
        sections.append(npc_context)

    return "\n\n".join(sections)


def _call_ollama_for_council(model: str, prompt: str, system: str) -> str:
    """Helper to call Ollama for council deliberations."""
    import httpx
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                }
            )
            if response.status_code == 200:
                return response.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"[Council] Error: {e}")
    return ""


def _is_pure_factual_question(question: str) -> bool:
    """Detect pure factual/world-state questions that don't need Council deliberation.

    These are questions about the world state that the DM simply answers/establishes.
    No success/failure, no pros/cons - just facts.
    """
    q = question.lower().strip()

    # Patterns for pure factual questions
    factual_starters = [
        'how many', 'how much', 'how long', 'how far', 'how old',
        'what is', 'what are', 'what was', 'what were', "what's",
        'who is', 'who are', 'who was', "who's",
        'where is', 'where are', 'where was', "where's",
        'when is', 'when was', 'when did',
        'which', 'what color', 'what time', 'what day',
        'is there a', 'are there any', 'is it', 'was it',
        'does the', 'do the', 'did the',  # When asking about world state, not outcomes
    ]

    # Check if question starts with a factual pattern
    for pattern in factual_starters:
        if q.startswith(pattern):
            # Make sure it's not actually asking about success/failure
            # "Can I" and "Will I" suggest outcomes
            if 'can i' in q or 'will i' in q or 'could i' in q or 'would i' in q:
                return False
            # "work" suggests testing feasibility
            if 'work' in q and ('would' in q or 'will' in q):
                return False
            return True

    return False


def _triage_dm_question(question: str, context: str, model: str) -> dict:
    """Arbiter triages the question into MINOR, MEDIUM, or MAJOR, and INFO vs ACTION."""
    prompt = f"""You are the ARBITER. Classify this question from a character.

CONTEXT:
{context}

QUESTION: {question}

⚠️ IMPORTANT: Check the "WHO IS ACTING" section in the context above.
The TENSION you extract should reference the ACTING CHARACTER by name.

=== CLASSIFY BY TYPE ===
- INFO: Player is asking a HYPOTHETICAL - they want to know odds/feasibility WITHOUT committing.
  Examples: "Would this work?" "Can I do X?" "What if I tried Y?" "Is it possible to Z?" "What are my options?"
  Key signal: They're asking about a potential action, not taking it yet.

- ACTION: Player is DOING something or has done something and wants to know the result.
  Examples: "I try to hide" "I look around" "I search the room" "Do I find anything?" "Does she believe me?"
  Key signal: The action is happening/happened. "Do I find X?" = you're actively searching = ACTION.

IMPORTANT: "Can I do X?" is INFO (hypothetical). "I do X" is ACTION. "Do I find X?" is ACTION (you're searching).

⚠️ WATCH FOR LEADING QUESTIONS:
"Is X loud/big/fast enough to cause Y?" - This LOOKS like a factual question but it's really asking for an OUTCOME.
The player is framing their desired result as logical inevitability. These are ACTION questions about NPC/world behavior.
Treat "Is the noise loud enough to distract them?" as ACTION (you're asking what the NPCs do), not INFO.

=== CLASSIFY BY STAKES ===
- MINOR: Simple factual questions about the world. "What time is it?" "Is there a door?" "What's the weather?"
- MEDIUM: Questions with some stakes but not life/death. "Can I pick this lock?" "Does she believe me?" "Is there food here?"
- MAJOR: Life/death, pivotal moments, high consequence. "Do we survive?" "Does the plan work?" "Do we make it?"

Respond with EXACTLY this format:
TYPE: [INFO/ACTION]
TIER: [MINOR/MEDIUM/MAJOR]
TENSION: [One sentence describing the core tension/stakes if MEDIUM or MAJOR, or "N/A" if MINOR]

Example 1 (info question):
TYPE: INFO
TIER: MEDIUM
TENSION: Will Kaido find a hiding spot that keeps him and Juniper concealed?

Example 2 (action):
TYPE: ACTION
TIER: MAJOR
TENSION: Will they escape the dogs before being caught?"""

    system = "You classify questions by type and stakes. INFO = asking/seeking. ACTION = doing/attempting. Be decisive."

    result = _call_ollama_for_council(model, prompt, system)

    # Parse response
    tier = "MINOR"
    tension = ""
    question_type = "ACTION"  # Default to action for safety
    for line in result.split('\n'):
        if line.startswith('TYPE:'):
            t = line.replace('TYPE:', '').strip().upper()
            if t in ['INFO', 'ACTION']:
                question_type = t
        elif line.startswith('TIER:'):
            t = line.replace('TIER:', '').strip().upper()
            if t in ['MINOR', 'MEDIUM', 'MAJOR']:
                tier = t
        elif line.startswith('TENSION:'):
            tension = line.replace('TENSION:', '').strip()

    print(f"[Council] Arbiter triage: {tier} ({question_type}) - {tension}")
    return {"tier": tier, "tension": tension, "type": question_type}


def _call_advocate(tension: str, context: str, model: str) -> str:
    """Advocate argues FOR success/player's hope."""
    prompt = f"""You are the ADVOCATE in a council debate. Your job is to argue FOR the character succeeding.

CONTEXT:
{context}

CORE TENSION: {tension}

⚠️ CHECK "WHO IS ACTING" ABOVE - argue for THAT character's success, not anyone else's.

Make your BEST argument for why they SHOULD succeed. Consider:
- What advantages do they have?
- What smart decisions did they make?
- What narrative/logical reasons support success?
- Physics, timing, positioning - what's in their favor?

CRITICAL RULES:
- ONLY cite details that exist in the context above
- Do NOT invent terrain, weather, or obstacles not mentioned
- Do NOT name characters that aren't already named
- ONLY reference items that appear in the INVENTORY section

⚠️ IN-SCENE CHARACTERS ONLY:
You may ONLY reference characters who have SPOKEN or APPEARED in the "Recent conversation" section.
Characters listed as "SEPARATED" in CHARACTER LOCATIONS have NOT been introduced yet - they DO NOT EXIST in this scene.
If a character hasn't spoken or been described in the recent conversation, they are NOT HERE.

Be specific. Cite details from the context. 3-5 sentences max.
Write as: "The player succeeds because..."."""

    system = "You argue FOR player success. ONLY use details from the provided context. Never invent locations, terrain, characters, or items. SEPARATED characters are NOT in the scene."
    return _call_ollama_for_council(model, prompt, system)


def _call_adversary(tension: str, context: str, model: str) -> str:
    """Adversary argues AGAINST or introduces complications."""
    prompt = f"""You are the ADVERSARY in a council debate. Your job is to argue for COMPLICATIONS or failure.

CONTEXT:
{context}

CORE TENSION: {tension}

⚠️ CHECK "WHO IS ACTING" ABOVE - argue against THAT character's action, not anyone else's.

Make your BEST argument for why things should go WRONG or have complications. Consider:
- What could realistically fail?
- What dangers exist IN THE SCENE AS DESCRIBED?
- What has the player overlooked?
- Physics, timing, bad luck - what works against them?

CRITICAL RULES:
- Your complications must be PHYSICALLY PLAUSIBLE given the scene
- If they're hiding inside a structure, outsiders can't see inside
- If something is in a pocket, it's not visible
- Don't manufacture nonsense just to create drama
- Complications must make LOGICAL SENSE given positions, lighting, cover, etc.

⚠️ IN-SCENE CHARACTERS ONLY:
You may ONLY reference characters who have SPOKEN or APPEARED in the "Recent conversation" section.
Characters listed as "SEPARATED" in CHARACTER LOCATIONS have NOT been introduced yet - they DO NOT EXIST in this scene.
You CANNOT use separated characters as complications - they are not here to interfere.
If a character hasn't spoken or been described in the recent conversation, they are NOT HERE.

⚠️ IMPORTANT: If there are NO plausible complications - if the player genuinely has this one - you can say:
"No meaningful complications. The player's position/actions are sound."

Don't force weak arguments. A concession is better than nonsense.

Be specific. Cite details from the context. 3-5 sentences max.
Write as: "The player faces complications because..." OR "No meaningful complications..."."""

    system = "You argue for complications/failure, but ONLY plausible ones. SEPARATED characters are NOT in the scene and cannot be used. If there's no real complication, concede. Never invent physically impossible scenarios."
    return _call_ollama_for_council(model, prompt, system)


def _call_info_advocate(question: str, context: str, model: str) -> str:
    """For INFO questions - argue for the favorable interpretation (yes they see, yes it's possible, etc.)."""
    prompt = f"""You are weighing an INFO QUESTION - the player is asking about the world state, not taking an action.

CONTEXT:
{context}

PLAYER'S QUESTION: {question}

Your job: Argue for the interpretation that FAVORS the player's hopes or concerns.
- If they ask "do they see X?" - argue YES, they probably DO see it
- If they ask "is it safe?" - argue YES, it probably IS safe
- If they ask "can this work?" - argue YES, it probably CAN work

Read the question carefully. What answer would the player WANT to hear? Argue for that answer.

Give 2-4 specific reasons based on the scene context. Consider:
- Positioning, lighting, cover, distances
- What characters would logically notice or miss
- Physics and plausibility

CRITICAL: Answer the ACTUAL QUESTION being asked. Don't talk about player success/failure - talk about the world state.

Write as: "Yes, likely because..." or "Probably, because..." - directly answering their question."""

    system = "You answer INFO questions about world state. Argue for the favorable interpretation. Be specific to the question asked."
    return _call_ollama_for_council(model, prompt, system)


def _call_info_adversary(question: str, context: str, model: str) -> str:
    """For INFO questions - argue for the unfavorable interpretation (no they don't see, no it's not safe, etc.)."""
    prompt = f"""You are weighing an INFO QUESTION - the player is asking about the world state, not taking an action.

CONTEXT:
{context}

PLAYER'S QUESTION: {question}

Your job: Argue for the interpretation that CHALLENGES the player's hopes or concerns.
- If they ask "do they see X?" - argue NO, they probably DON'T see it
- If they ask "is it safe?" - argue NO, it probably ISN'T safe
- If they ask "can this work?" - argue NO, it probably WON'T work

Read the question carefully. What answer would create more tension or challenge? Argue for that answer.

Give 2-4 specific reasons based on the scene context. Consider:
- Positioning, lighting, cover, distances
- What characters would logically notice or miss
- Physics and plausibility

CRITICAL: Answer the ACTUAL QUESTION being asked. Don't talk about player complications - talk about the world state.

⚠️ If the unfavorable answer is genuinely implausible given the scene, you can say:
"Hard to argue against this - the scene suggests [favorable answer] is likely."

Write as: "No, unlikely because..." or "Probably not, because..." - directly answering their question."""

    system = "You answer INFO questions about world state. Argue for the challenging interpretation, but only if plausible. Be specific to the question asked."
    return _call_ollama_for_council(model, prompt, system)


def _call_info_judge(question: str, yes_args: str, no_args: str, context: str, model: str) -> dict:
    """
    Judge weighs both interpretations and delivers a DEFINITIVE answer to an INFO question.

    Unlike ACTION questions, INFO questions need a clear yes/no answer that becomes canon.
    "Do I have this knowledge?" can't be "maybe" - either you do or you don't.
    """
    prompt = f"""You are the JUDGE answering an INFO question. Two advocates have argued. You must deliver a DEFINITIVE ANSWER.

CONTEXT:
{context}

QUESTION: {question}

=== ARGUMENT FOR "YES" ===
{yes_args}

=== ARGUMENT FOR "NO" ===
{no_args}

YOUR JOB: Decide YES or NO. This is a question about world state or character knowledge - it has a definitive answer.

Consider:
- Which argument is more grounded in the established scene/context?
- What makes sense given the genre, setting, and characters involved?
- For character knowledge questions: what would this character plausibly know given their background?

YOU MUST DECIDE. No "maybe" - that's effectively "no" and unfair to the player.

Respond in EXACTLY this format:
ANSWER: [YES/NO]
RULING: [2-3 sentences explaining your answer. This becomes canon. Be specific and definitive.]"""

    system = "You are a fair judge who makes definitive rulings on INFO questions. Your answer becomes canon in the game world. Be decisive."

    response = _call_ollama_for_council(model, prompt, system)

    # Parse the response
    answer = "YES"  # Default to favorable if parsing fails
    ruling = response

    lines = response.strip().split('\n')
    for line in lines:
        if line.startswith('ANSWER:'):
            ans = line.replace('ANSWER:', '').strip().upper()
            if 'NO' in ans:
                answer = "NO"
            else:
                answer = "YES"
        elif line.startswith('RULING:'):
            ruling = line.replace('RULING:', '').strip()

    # If ruling is still the full response, try to extract just the explanation
    if ruling == response and len(lines) > 1:
        # Take everything after ANSWER line as the ruling
        for i, line in enumerate(lines):
            if line.startswith('ANSWER:'):
                ruling = '\n'.join(lines[i+1:]).replace('RULING:', '').strip()
                break

    return {
        "answer": answer,
        "ruling": ruling,
    }


def _call_judge(tension: str, advocate_args: str, adversary_args: str, context: str, model: str) -> dict:
    """Judge weighs both arguments and delivers a ruling."""
    prompt = f"""You are the JUDGE. Two advocates have argued. You must deliver a RULING.

CONTEXT:
{context}

CORE TENSION: {tension}

⚠️ CHECK "WHO IS ACTING" IN THE CONTEXT - your ruling is about THAT character's action.
When narrating the outcome, use the correct character's name.

=== ADVOCATE (argues for success) ===
{advocate_args}

=== ADVERSARY (argues for complications) ===
{adversary_args}

YOUR JOB: Actually WEIGH these arguments. Don't just combine them.

⚠️ CRITICAL - CHECK EACH ARGUMENT FOR:
1. PHYSICAL PLAUSIBILITY: Does it make sense given positions, cover, lighting?
   - If they're hiding INSIDE a structure, can outsiders really see them?
   - If something is IN a pocket, is it really "visible"?
   - Can flashlights see through walls? Through solid cover?
2. LOGICAL CONSISTENCY: Does the argument contradict established facts?
3. WEAK ARGUMENTS: If the Adversary's complications are nonsensical or physically impossible, DISMISS THEM.
4. CHARACTER VALIDITY: Does the argument reference characters who are ACTUALLY IN THE SCENE?
   - Characters must have SPOKEN or APPEARED in the recent conversation
   - Characters listed as "SEPARATED" have NOT been introduced - they DO NOT EXIST here
   - If Adversary cites a separated/unintroduced character, DISMISS that argument entirely

If the Adversary conceded ("no meaningful complications") or their argument is weak/implausible:
→ Rule SUCCESS, not PARTIAL

If the Adversary cites characters who haven't appeared in the scene:
→ DISMISS that argument, it's invalid

If the Advocate's argument is weak and Adversary makes good points:
→ Rule FAILURE or heavy PARTIAL

PARTIAL should only happen when BOTH arguments have merit.

RULING RULES:
- Your ruling must be GROUNDED in established fiction
- Do NOT incorporate nonsense arguments just to create drama
- Do NOT invent details not in the context
- If an argument cites something physically impossible, ignore it entirely
- If an argument cites a character who hasn't appeared, ignore it entirely

Format EXACTLY:
OUTCOME: [SUCCESS / PARTIAL / FAILURE]
RULING: [2-4 sentences describing what happens. Use ONLY plausible, established elements.]
COST: [If SUCCESS or PARTIAL, what did it cost? If FAILURE, what went wrong? N/A if clean success.]

Be decisive. Dismiss weak arguments. The player deserves fair rulings, not manufactured drama."""

    system = "You are an impartial judge who dismisses weak/implausible arguments. SUCCESS is valid when complications are nonsense. Arguments citing SEPARATED/unintroduced characters are INVALID. Don't force PARTIAL."

    result = _call_ollama_for_council(model, prompt, system)

    # Parse response
    outcome = "PARTIAL"
    ruling = ""
    cost = ""

    lines = result.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('OUTCOME:'):
            o = line.replace('OUTCOME:', '').strip().upper()
            if o in ['SUCCESS', 'PARTIAL', 'FAILURE']:
                outcome = o
        elif line.startswith('RULING:'):
            ruling = line.replace('RULING:', '').strip()
            # Ruling might span multiple lines
            for j in range(i + 1, len(lines)):
                if lines[j].startswith('COST:'):
                    break
                if lines[j].strip():
                    ruling += ' ' + lines[j].strip()
        elif line.startswith('COST:'):
            cost = line.replace('COST:', '').strip()

    print(f"[Council] Judge ruling: {outcome}")
    return {"outcome": outcome, "ruling": ruling, "cost": cost}


def _sanitize_council_context(context: str) -> str:
    """
    Strip sections from context that could leak unintroduced characters.
    The Council should only see characters who are ACTUALLY in the scene.
    """
    import re

    # Remove CHARACTER LOCATIONS section (lists SEPARATED characters who haven't been introduced)
    context = re.sub(
        r'=== CHARACTER LOCATIONS ===.*?(?====|$)',
        '',
        context,
        flags=re.DOTALL
    )

    # Clean up any double newlines left behind
    context = re.sub(r'\n{3,}', '\n\n', context)

    return context.strip()


def _call_info_judge(question: str, tension: str, advocate_args: str, adversary_args: str, context: str, model: str) -> dict:
    """
    Judge for INFO questions - presents OPTIONS instead of narrating outcomes.
    Player asked a question seeking information; they get to CHOOSE what to do with it.
    """
    prompt = f"""You are the JUDGE answering an INFORMATION question. The player is ASKING, not DOING.

CONTEXT:
{context}

PLAYER'S QUESTION: {question}
CORE TENSION: {tension}

=== ADVOCATE (argues for good findings) ===
{advocate_args}

=== ADVERSARY (argues for complications) ===
{adversary_args}

YOUR TASK:
Weigh both arguments and present OPTIONS to the player. Do NOT decide for them.

CRITICAL RULES:
- Present 2-3 OPTIONS the player can choose from
- Each option should have clear trade-offs (pros AND cons)
- ONLY use locations, terrain, and items from the CONTEXT
- Do NOT narrate what the player does - let them CHOOSE
- Write in second person ("You notice...", "You spot...")
- The player should feel they have meaningful agency

OUTCOME determines option quality:
- SUCCESS: Good options available, clear paths forward
- PARTIAL: Mixed options, each has significant trade-offs
- FAILURE: Limited options, all have problems, or threat is closer than expected

Format EXACTLY:
OUTCOME: [SUCCESS / PARTIAL / FAILURE]
OPTIONS: [Present 2-3 options with trade-offs. Each option on its own line starting with "•". Include what's good AND what's risky about each.]
URGENCY: [One sentence about time pressure or threat status - how long before they must decide?]

Example:
OUTCOME: PARTIAL
OPTIONS:
• The rusted machinery to port - decent cover, but you'd be exposed during the dash across open deck
• The fish hold hatch - more concealment below, but opening it will make noise
• Over the rail into the water - invisible to searchers, but the cold could send Juniper into shock
URGENCY: The figures are sixty seconds from the dock - you need to move now.

Be decisive. Present real choices, not false ones."""

    system = "You present OPTIONS to players, not outcomes. Let them choose. Each option has trade-offs."

    result = _call_ollama_for_council(model, prompt, system)

    # Parse response
    outcome = "PARTIAL"
    options = ""
    urgency = ""

    lines = result.split('\n')
    in_options = False
    for i, line in enumerate(lines):
        if line.startswith('OUTCOME:'):
            o = line.replace('OUTCOME:', '').strip().upper()
            if o in ['SUCCESS', 'PARTIAL', 'FAILURE']:
                outcome = o
        elif line.startswith('OPTIONS:'):
            in_options = True
            rest = line.replace('OPTIONS:', '').strip()
            if rest:
                options = rest + '\n'
        elif line.startswith('URGENCY:'):
            in_options = False
            urgency = line.replace('URGENCY:', '').strip()
        elif in_options and line.strip():
            options += line + '\n'

    options = options.strip()
    print(f"[Council] Info Judge ruling: {outcome}")
    return {"outcome": outcome, "options": options, "urgency": urgency}


@app.route('/rooms/<room_id>/dm', methods=['POST'])
def ask_dm_public(room_id):
    """Ask the DM a public question - uses Council system for high-stakes questions."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    data = request.json or {}
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    partners = data_store.get_partners()
    room_partners = room.get_partners_in_room(partners)

    # Build context
    dm_context = _build_simple_dm_context(room, room_partners)

    recent_messages = room.messages[-10:] if room.messages else []
    conversation_context = "\n".join([
        f"{m.speaker_name}: {m.content[:200]}" for m in recent_messages
    ])

    # Extract most recent scene description (from describe-scene button) as authoritative terrain
    scene_description = ""
    for m in reversed(room.messages[-15:] if room.messages else []):
        # Scene descriptions come from narrator with specific spatial content
        if m.speaker_id == "narrator" and getattr(m, 'message_type', '') == "narration":
            # Check if it looks like a scene description (has spatial language)
            content = m.content.lower()
            if any(word in content for word in ['yards', 'feet', 'positioned', 'north', 'south', 'east', 'west', 'behind', 'ahead', 'crouched', 'standing']):
                scene_description = m.content
                break

    # Build full context with scene description prominently featured
    if scene_description:
        full_context = f"""{dm_context}

=== AUTHORITATIVE SCENE DESCRIPTION (trust this for terrain/positions) ===
{scene_description}

=== Recent conversation ===
{conversation_context}"""
    else:
        full_context = f"{dm_context}\n\nRecent conversation:\n{conversation_context}"

    # Get model
    import httpx
    model_to_use = settings.storybuilder_model
    available_models = provider_manager.get_models_for_provider('ollama')
    if model_to_use not in available_models and available_models:
        model_to_use = available_models[0]

    # === STEP 0: CHECK FOR PURE FACTUAL QUESTIONS ===
    # Questions like "How many people live here?" don't need Council - just DM establishing facts
    if _is_pure_factual_question(question):
        print(f"[DM] Pure factual question - direct answer (no Council)")
        prompt = f"""You are the Dungeon Master. The player is asking a factual question about the world.

{full_context}

Question: {question}

Establish the facts. If this is something that hasn't been defined yet, make a reasonable decision
based on the genre, setting, and what would make for an interesting story.

Give a clear, direct answer (1-3 sentences). You're establishing world canon here."""

        dm_response = _call_ollama_for_council(model_to_use, prompt,
            "You are a DM establishing world facts. Answer clearly and directly.")

        # Save the DM's response to the room
        player_name = room.player_character_name or settings.user_name or "Player"
        player_msg = Message(
            id=str(uuid.uuid4())[:8],
            speaker_id='user',
            speaker_name=player_name,
            content=f"*asks the DM:* {question}",
            room_id=room_id,
            message_type='dm_public',
        )
        data_store.add_message(room_id, player_msg)

        dm_msg = Message(
            id=str(uuid.uuid4())[:8],
            speaker_id='dm',
            speaker_name='🎲 DM',
            content=dm_response,
            room_id=room_id,
            message_type='dm_public',
        )
        data_store.add_message(room_id, dm_msg)

        return jsonify({
            'question': {
                'id': player_msg.id,
                'speaker_id': player_msg.speaker_id,
                'speaker_name': player_msg.speaker_name,
                'content': player_msg.content,
                'message_type': player_msg.message_type,
            },
            'answer': {
                'id': dm_msg.id,
                'speaker_id': dm_msg.speaker_id,
                'speaker_name': dm_msg.speaker_name,
                'content': dm_msg.content,
                'message_type': dm_msg.message_type,
                'fudged': False,
            },
            'type': 'factual',
        })

    # === STEP 1: ARBITER TRIAGE ===
    triage = _triage_dm_question(question, full_context, model_to_use)

    dm_response = ""
    council_data = None

    if triage["tier"] == "MINOR":
        # Simple direct answer
        print(f"[Council] MINOR question - direct answer")
        prompt = f"""You are the Dungeon Master. Answer this simple question about the world.

{full_context}

Question: {question}

IMPORTANT - WATCH FOR LEADING QUESTIONS:
If the player asks "Is X loud/big/fast enough to cause Y?" - that's NOT a physics question.
They're trying to frame an outcome as logical inevitability. "Was loud thing loud? Yes → I win."
YOUR job is to decide what ACTUALLY happens given the full context, not validate their framing.
Judge the situation, not the question's implications.

Give a brief, factual answer (1-2 sentences). Be decisive but don't be led."""

        dm_response = _call_ollama_for_council(model_to_use, prompt,
            "You are a DM answering world questions. Don't be manipulated by leading questions - judge the actual situation.")

    elif triage["tier"] == "MEDIUM":
        # Quick deliberation - one advocate, one adversary, quick ruling
        print(f"[Council] MEDIUM stakes - quick deliberation")
        tension = triage["tension"]
        question_type = triage.get("type", "ACTION")

        # Sanitize context for Council - remove SEPARATED characters who haven't been introduced
        council_context = _sanitize_council_context(full_context)

        if question_type == "INFO":
            # INFO question: deliberate internally, then give a definitive answer
            print(f"[Council] INFO question - internal deliberation, definitive answer")
            yes_args = _call_info_advocate(question, council_context, model_to_use)
            no_args = _call_info_adversary(question, council_context, model_to_use)
            ruling = _call_info_judge(question, yes_args, no_args, council_context, model_to_use)

            # Clean, definitive answer - no pros/cons shown
            answer_emoji = "✓" if ruling["answer"] == "YES" else "✗"
            dm_response = f"""🎲 **{ruling["answer"]}**

{ruling["ruling"]}"""
            council_data = {
                "tier": "MEDIUM",
                "type": "INFO",
                "tension": tension,
                "outcome": ruling["answer"],
            }
        else:
            # ACTION question: get advocates and narrate outcome
            advocate = _call_advocate(tension, council_context, model_to_use)
            adversary = _call_adversary(tension, council_context, model_to_use)
            ruling = _call_judge(tension, advocate, adversary, council_context, model_to_use)
            dm_response = ruling["ruling"]
            if ruling["cost"] and ruling["cost"].lower() != "n/a" and ruling["cost"].lower() != "none":
                dm_response += f"\n\n*{ruling['cost']}*"
            council_data = {
                "tier": "MEDIUM",
                "type": "ACTION",
                "tension": tension,
                "outcome": ruling["outcome"],
            }

    else:  # MAJOR
        # Full council debate
        print(f"[Council] MAJOR stakes - full council deliberation")
        tension = triage["tension"]
        question_type = triage.get("type", "ACTION")

        # Sanitize context for Council - remove SEPARATED characters who haven't been introduced
        council_context = _sanitize_council_context(full_context)

        if question_type == "INFO":
            # INFO question: full deliberation internally, definitive answer
            print(f"[Council] INFO question (MAJOR) - full deliberation, definitive answer")
            yes_args = _call_info_advocate(question, council_context, model_to_use)
            no_args = _call_info_adversary(question, council_context, model_to_use)
            ruling = _call_info_judge(question, yes_args, no_args, council_context, model_to_use)

            # For MAJOR, show that we deliberated but give definitive answer
            dm_response = f"""⚖️ **THE COUNCIL HAS RULED**

**Question:** {tension}

---

**{ruling["answer"]}**

{ruling["ruling"]}"""
            council_data = {
                "tier": "MAJOR",
                "type": "INFO",
                "tension": tension,
                "outcome": ruling["answer"],
            }
        else:
            # ACTION question: narrate outcome with full deliberation
            advocate = _call_advocate(tension, council_context, model_to_use)
            adversary = _call_adversary(tension, council_context, model_to_use)
            ruling = _call_judge(tension, advocate, adversary, council_context, model_to_use)
            dm_response = f"""⚖️ **THE COUNCIL DELIBERATES**

**Tension:** {tension}

**Advocate:** {advocate}

**Adversary:** {adversary}

---

**RULING: {ruling['outcome']}**

{ruling['ruling']}

*{ruling['cost']}*"""
            council_data = {
                "tier": "MAJOR",
                "type": "ACTION",
                "tension": tension,
                "advocate": advocate,
                "adversary": adversary,
                "outcome": ruling["outcome"],
                "cost": ruling["cost"],
            }

    if not dm_response:
        dm_response = "The DM ponders but cannot decide..."

    # Create the DM message
    dm_message = Message(
        id=str(uuid.uuid4()),
        speaker_id="dm",
        speaker_name="DM" if not council_data else "⚖️ Council",
        content=dm_response,
        room_id=room_id,
        message_type="dm_public",
    )
    data_store.add_message(room_id, dm_message)

    # Create the question message (for display)
    question_message = Message(
        id=str(uuid.uuid4()),
        speaker_id="user",
        speaker_name=room.player_character_name or settings.user_name,
        content=question,
        room_id=room_id,
        message_type="dm_public",
    )

    response = {
        'question': {
            'id': question_message.id,
            'speaker_id': question_message.speaker_id,
            'speaker_name': question_message.speaker_name,
            'content': question_message.content,
            'message_type': question_message.message_type,
        },
        'answer': {
            'id': dm_message.id,
            'speaker_id': dm_message.speaker_id,
            'speaker_name': dm_message.speaker_name,
            'content': dm_message.content,
            'message_type': dm_message.message_type,
        }
    }

    if council_data:
        response['council'] = council_data
        print(f"[Council] Final outcome: {council_data['outcome']}")

    return jsonify(response)


def _handle_character_dm_call(room_id: str, character_name: str, question: str) -> Optional[Message]:
    """
    Handle a character's auto-DM call (detected via [DM: question?] pattern).
    Returns a DM message with the Council ruling, or None if it fails.

    Includes cooldown: max 1 character DM call per 5 messages.
    """
    import re

    room = data_store.get_room(room_id)
    if not room:
        return None

    # Cooldown check: look for recent character DM calls in last 5 messages
    recent_messages = room.messages[-5:] if room.messages else []
    recent_dm_calls = sum(1 for m in recent_messages
                         if m.speaker_id == "dm" and
                         m.metadata and
                         m.metadata.get('auto_dm_call'))

    if recent_dm_calls > 0:
        print(f"[AutoDM] Cooldown active - skipping DM call from {character_name}")
        return None

    print(f"[AutoDM] {character_name} asks: {question}")

    partners = data_store.get_partners()
    room_partners = room.get_partners_in_room(partners)

    # Build context (same as ask_dm_public)
    dm_context = _build_simple_dm_context(room, room_partners)

    recent_messages = room.messages[-10:] if room.messages else []
    conversation_context = "\n".join([
        f"{m.speaker_name}: {m.content[:200]}" for m in recent_messages
    ])

    # === CRITICAL: Extract the character's action that prompted this DM call ===
    # Find the most recent message from this character to see what they actually did
    character_action = ""
    for m in reversed(recent_messages):
        if m.speaker_name == character_name:
            # Strip the [DM: ...] tag from the action to see what they did
            action_text = re.sub(r'\[DM:[^\]]+\]', '', m.content).strip()
            if action_text:
                character_action = action_text[:500]  # Cap at 500 chars
            break

    # Extract scene description
    scene_description = ""
    for m in reversed(room.messages[-15:] if room.messages else []):
        if m.speaker_id == "narrator" and getattr(m, 'message_type', '') == "narration":
            content = m.content.lower()
            if any(word in content for word in ['yards', 'feet', 'positioned', 'north', 'south', 'east', 'west', 'behind', 'ahead']):
                scene_description = m.content
                break

    # Build full context with EXPLICIT action attribution
    action_attribution = f"""
=== WHO IS ACTING (CRITICAL - DO NOT CONFUSE) ===
ACTING CHARACTER: {character_name}
THEIR ACTION: {character_action if character_action else "(action in their last message)"}
THEIR QUESTION: {question}

⚠️ You are adjudicating {character_name.upper()}'s action, NOT anyone else's.
If the action involves calling out, shouting, or making noise - it was {character_name} who did it.
Do NOT attribute this action to any other character."""

    if scene_description:
        full_context = f"""{dm_context}

=== AUTHORITATIVE SCENE DESCRIPTION ===
{scene_description}
{action_attribution}

=== Recent conversation ===
{conversation_context}"""
    else:
        full_context = f"{dm_context}\n{action_attribution}\n\nRecent conversation:\n{conversation_context}"

    # Get model
    model_to_use = settings.storybuilder_model
    available_models = provider_manager.get_models_for_provider('ollama')
    if model_to_use not in available_models and available_models:
        model_to_use = available_models[0]

    # Triage and deliberate
    triage = _triage_dm_question(question, full_context, model_to_use)

    dm_response = ""
    council_data = None

    # Frame it as a character asking, not the player
    framed_question = f"{character_name} wonders: {question}"

    if triage["tier"] == "MINOR":
        prompt = f"""You are the Dungeon Master. A character wonders about something.

{full_context}

{character_name} wonders: {question}

Give a brief, factual answer (1-2 sentences). This resolves what the character was wondering."""

        dm_response = _call_ollama_for_council(model_to_use, prompt,
            "You are a DM answering character questions. Be brief and decisive.")

    elif triage["tier"] in ("MEDIUM", "MAJOR"):
        tension = triage["tension"]
        question_type = triage.get("type", "ACTION")

        # Sanitize context for Council - remove SEPARATED characters who haven't been introduced
        council_context = _sanitize_council_context(full_context)

        advocate = _call_advocate(tension, council_context, model_to_use)
        adversary = _call_adversary(tension, council_context, model_to_use)

        if question_type == "INFO":
            # Character asking an INFO question - just show pros/cons, no ruling
            if triage["tier"] == "MEDIUM":
                dm_response = f"""**Weighing the odds...**

👍 **For:** {advocate}

👎 **Against:** {adversary}"""
            else:
                dm_response = f"""⚖️ **THE COUNCIL WEIGHS IN**

👍 **Advocate:** {advocate}

👎 **Adversary:** {adversary}

*The outcome depends on what you do next.*"""
            council_data = {
                "tier": triage["tier"],
                "type": "INFO",
                "tension": tension,
                "outcome": "UNDECIDED",
            }
        else:
            # ACTION question - narrate outcome
            ruling = _call_judge(tension, advocate, adversary, council_context, model_to_use)
            if triage["tier"] == "MEDIUM":
                dm_response = ruling["ruling"]
                if ruling["cost"] and ruling["cost"].lower() not in ("n/a", "none"):
                    dm_response += f"\n\n*{ruling['cost']}*"
            else:
                dm_response = f"""⚖️ **THE COUNCIL DELIBERATES**

**Tension:** {tension}

**Advocate:** {advocate}

**Adversary:** {adversary}

---

**RULING: {ruling['outcome']}**

{ruling['ruling']}"""
                if ruling["cost"] and ruling["cost"].lower() not in ("n/a", "none"):
                    dm_response += f"\n\n*{ruling['cost']}*"
            council_data = {
                "tier": triage["tier"],
                "type": "ACTION",
                "tension": tension,
                "outcome": ruling["outcome"],
            }

    if not dm_response:
        return None

    # Create DM message with metadata marking it as auto-call
    dm_message = Message(
        id=str(uuid.uuid4())[:8],
        speaker_id="dm",
        speaker_name="⚖️ Council" if council_data else "DM",
        content=dm_response,
        room_id=room_id,
        message_type="dm_public",
        metadata={"auto_dm_call": True, "asked_by": character_name}
    )
    # Don't add to data_store here - let caller control ordering
    # Caller should add Juniper's response first, then this DM message

    print(f"[AutoDM] Ruling delivered: {triage['tier']}")
    return dm_message


def _handle_character_item_action(room_id: str, character_id: str, character_name: str, action_text: str) -> Optional[str]:
    """
    Handle a character's item action (detected via [ITEM: action] pattern).
    Parses the action and updates inventories accordingly.

    Returns a brief status message or None.
    """
    room = data_store.get_room(room_id)
    if not room:
        return None

    action_lower = action_text.lower().strip()
    print(f"[ItemAction] {character_name}: {action_text}")

    # Determine action type and parse
    # Patterns: "gave X to Y", "used X", "dropped X", "picked up X", "took X"

    result = None

    # TRANSFER: "gave/handed/passed X to Y"
    transfer_match = re.match(
        r'(?:gave|hand(?:ed|s)?|pass(?:ed)?|give(?:s)?)\s+(?:the\s+|my\s+|a\s+|an\s+)?(.+?)\s+to\s+(\w+)',
        action_lower
    )
    if transfer_match:
        item_name = transfer_match.group(1).strip()
        target_name = transfer_match.group(2).strip()

        # Find the item in character's inventory (fuzzy match)
        char_inv = inventory_tracker.get_inventory(character_id)
        if char_inv:
            item = char_inv.find_item_fuzzy(item_name)
            if item:
                # Find target character
                partners = data_store.get_partners()
                target = None

                # Check room partners
                for p in partners:
                    if p.name.lower() == target_name.lower() or target_name.lower() in p.name.lower():
                        target = p
                        break

                # Check if target is the player
                if not target and target_name.lower() in (settings.user_name.lower(), 'player', 'you'):
                    target_id = room.player_character_id or 'player'
                    target_name_actual = room.player_character_name or settings.user_name
                    # Ensure player has inventory
                    inventory_tracker.get_or_create_inventory(target_id, target_name_actual, 'player')
                    if inventory_tracker.transfer_item(character_id, target_id, item.id):
                        result = f"✓ {character_name} gave {item.name} to {target_name_actual}"
                        print(f"[ItemAction] Transfer success: {result}")

                elif target:
                    # Ensure target has inventory
                    inventory_tracker.get_or_create_inventory(target.id, target.name, 'partner')
                    if inventory_tracker.transfer_item(character_id, target.id, item.id):
                        result = f"✓ {character_name} gave {item.name} to {target.name}"
                        print(f"[ItemAction] Transfer success: {result}")
                else:
                    # Unknown target - maybe an NPC, just remove from inventory
                    inventory_tracker.remove_item(character_id, item.id)
                    result = f"✓ {character_name} gave {item.name} to {target_name}"
                    print(f"[ItemAction] Gave to unknown target: {result}")
            else:
                print(f"[ItemAction] Item not found in inventory: {item_name}")
        return result

    # USE/CONSUME: "used X", "consumed X", "ate X", "drank X"
    use_match = re.match(
        r'(?:used?|consumed?|ate|drank|applied|exhausted?)\s+(?:the\s+|my\s+|a\s+|an\s+|some\s+)?(.+?)(?:\s+on\s+.+)?$',
        action_lower
    )
    if use_match:
        item_name = use_match.group(1).strip()
        char_inv = inventory_tracker.get_inventory(character_id)
        if char_inv:
            item = char_inv.find_item_fuzzy(item_name)
            if item:
                inventory_tracker.remove_item(character_id, item.id)
                result = f"✓ {character_name} used {item.name}"
                print(f"[ItemAction] Use success: {result}")
            else:
                print(f"[ItemAction] Item not found for use: {item_name}")
        return result

    # DROP/DISCARD: "dropped X", "discarded X", "threw X"
    drop_match = re.match(
        r'(?:dropped?|discarded?|threw|tossed|abandoned)\s+(?:the\s+|my\s+|a\s+|an\s+)?(.+?)$',
        action_lower
    )
    if drop_match:
        item_name = drop_match.group(1).strip()
        char_inv = inventory_tracker.get_inventory(character_id)
        if char_inv:
            item = char_inv.find_item_fuzzy(item_name)
            if item:
                inventory_tracker.remove_item(character_id, item.id)
                result = f"✓ {character_name} dropped {item.name}"
                print(f"[ItemAction] Drop success: {result}")
            else:
                print(f"[ItemAction] Item not found for drop: {item_name}")
        return result

    # PICKUP/ACQUIRE: "picked up X", "took X", "grabbed X"
    pickup_match = re.match(
        r'(?:picked?\s+up|took|grabbed?|found|acquired?|collected?)\s+(?:the\s+|a\s+|an\s+|some\s+)?(.+?)$',
        action_lower
    )
    if pickup_match:
        item_name = pickup_match.group(1).strip().title()
        # Add to inventory
        inventory_tracker.get_or_create_inventory(character_id, character_name, 'partner')
        from inventory import _guess_category
        category = _guess_category(item_name)
        inventory_tracker.add_item(character_id, item_name, category.value)
        result = f"✓ {character_name} picked up {item_name}"
        print(f"[ItemAction] Pickup success: {result}")
        return result

    print(f"[ItemAction] Could not parse action: {action_text}")
    return None


def _handle_character_separation(room_id: str, character_id: str, character_name: str) -> bool:
    """
    Handle a character marking themselves as separated via [SEPARATED] tag.
    Removes them from present_character_ids so they enter the background activity feed.
    Records their last known location for potential reunion detection.

    Returns True if separation was successful.
    """
    room = data_store.get_room(room_id)
    if not room:
        return False

    # Get current present characters
    present_ids = room.present_character_ids or []

    # Check if character is actually present
    if character_id not in present_ids:
        print(f"[Separation] {character_name} already not present, skipping")
        return False

    # Record the character's last known location (where they separated)
    # Try to get a sensible location - use recent scene description if player_location isn't set well
    char_locations = room.character_locations or {}

    # Get location from context - check recent narration for location hints
    last_location = "the current area"
    if room.player_location and room.player_location not in ["unknown", "unknown location", ""]:
        last_location = room.player_location
    else:
        # Try to infer from scenario
        if room.scenario:
            # Take first location-ish phrase from scenario
            scenario_lower = room.scenario.lower()
            for phrase in ["at the ", "near the ", "in the ", "by the ", "outside the "]:
                if phrase in scenario_lower:
                    idx = scenario_lower.index(phrase)
                    potential = room.scenario[idx:idx+50]
                    for end in [",", ".", "!", "?", " and ", " where "]:
                        if end in potential:
                            potential = potential[:potential.index(end)]
                    if len(potential) > 5:
                        last_location = potential.strip()
                        break

    char_locations[character_id] = last_location
    room.character_locations = char_locations

    # Remove from present
    present_ids = [pid for pid in present_ids if pid != character_id]
    room.present_character_ids = present_ids

    # Also update the PlayerCharacter's is_separated flag in autopilot tracker
    pc = autopilot_tracker.get(character_id, room_id)
    if pc:
        pc.is_separated = True
        autopilot_tracker.save()
        print(f"[Separation] Updated PlayerCharacter.is_separated=True for {character_name}")

    data_store.save()

    print(f"[Separation] {character_name} marked as SEPARATED at '{last_location}' - now in background activity feed")
    return True


def _parse_combat_tag(tag_content: str) -> dict:
    """
    Parse a [COMBAT: ...] tag content.

    Examples:
        "zombie" -> {'target': 'zombie'}
        "zombie | surprise: enemy" -> {'target': 'zombie', 'surprise': 'enemy'}
        "bandit | initiated_by: Juniper" -> {'target': 'bandit', 'initiated_by': 'Juniper'}
    """
    result = {'target': None}

    parts = [p.strip() for p in tag_content.split('|')]
    if parts:
        result['target'] = parts[0]

    for part in parts[1:]:
        if ':' in part:
            key, value = part.split(':', 1)
            result[key.strip().lower().replace(' ', '_')] = value.strip()

    return result


def _handle_combat_initiation(
    room_id: str,
    initiator_id: str,
    initiator_name: str,
    combat_info: dict
) -> dict:
    """
    Handle a character or DM initiating combat.

    Args:
        room_id: The room where combat is starting
        initiator_id: Who started the combat (character_id or 'dm')
        initiator_name: Name of the initiator
        combat_info: Parsed combat tag with 'target', optional 'surprise', 'initiated_by'

    Returns:
        Combat encounter info dict
    """
    from combat_system import (
        CombatEncounter, EncounterManager, Combatant,
        create_npc_combatant, create_companion_stats, CombatStats, Attack
    )

    target = combat_info.get('target', 'unknown enemy')
    surprise = combat_info.get('surprise', None)
    initiated_by = combat_info.get('initiated_by', initiator_name)

    print("[Combat] " +f"▶ COMBAT INITIATED by {initiated_by}")
    print("[Combat] " +f"  Target: {target}")
    if surprise:
        print("[Combat] " +f"  Surprise round: {surprise}")

    # Create or get the combat encounter for this room
    encounter = EncounterManager.get_or_create(room_id)

    # If already active combat, this joins the existing fight
    if encounter.is_active:
        print("[Combat] " +f"  Joining existing combat in room {room_id}")
        return {
            'combat_started': False,
            'joined_existing': True,
            'target': target,
            'encounter_active': True
        }

    # Generate enemy stats based on target description
    # For now, use generic enemy types - can be expanded later
    enemy_role = 'bandit'  # default
    target_lower = target.lower()
    if 'zombie' in target_lower or 'undead' in target_lower:
        enemy_role = 'zombie'
    elif 'boss' in target_lower or 'leader' in target_lower:
        enemy_role = 'boss'
    elif 'guard' in target_lower or 'soldier' in target_lower:
        enemy_role = 'guard'
    elif 'civilian' in target_lower or 'villager' in target_lower:
        enemy_role = 'civilian'

    # Create zombie-specific stats (common for this game)
    if enemy_role == 'zombie':
        enemy_stats = CombatStats(
            name=target.title(),
            level=1,
            ac=11,  # Slow and clumsy
            hp_max=22,
            hp_current=22,
            fortitude=4,
            reflex=0,  # Very slow
            will=3,
            speed=20,  # Shambling
            attacks=[
                Attack(
                    name="Bite",
                    attack_bonus=4,
                    damage="1d6+2",
                    damage_type="piercing",
                    traits=["infection"],
                ),
                Attack(
                    name="Claw",
                    attack_bonus=3,
                    damage="1d4+2",
                    damage_type="slashing",
                ),
            ],
            weaknesses={"fire": 5, "slashing": 2},  # Weak to fire and decapitation
            immunities=["mental", "poison"],  # Mindless, no blood
        )
    else:
        enemy_stats = create_npc_combatant(target.title(), enemy_role, level=1)

    # Add enemy to encounter
    enemy_id = f"enemy_{room_id}_{target.lower().replace(' ', '_')}"
    encounter.add_combatant(
        id=enemy_id,
        stats=enemy_stats,
        team="enemies",
        initiative_bonus=enemy_stats.reflex,
        is_npc=True,
        npc_id=enemy_id,
    )

    # Add the initiating character to combat
    # Get their combat stats (simplified for now)
    initiator_stats = create_companion_stats(initiator_name, level=1, archetype="balanced")

    encounter.add_combatant(
        id=initiator_id,
        stats=initiator_stats,
        team="players",
        initiative_bonus=2,  # Default
        is_companion=True,
        companion_id=initiator_id,
    )

    # Start the combat - initiator's side goes first
    encounter.start_combat(initiator_side="players")

    # Get combat state for the result
    combat_state = encounter.get_combat_state()

    result = {
        'combat_started': True,
        'target': target,
        'target_hp': enemy_stats.hp_max,
        'target_ac': enemy_stats.ac,
        'initiated_by': initiated_by,
        'surprise_round': surprise,
        'turn_order': [
            {'name': encounter.combatants[cid].stats.name, 'initiative': encounter.combatants[cid].initiative}
            for cid in encounter.turn_order
        ],
        'current_round': encounter.round,
        'encounter_active': True,
        'active_side': combat_state['active_side'],
        'player_turns': combat_state['player_turns_remaining'],
        'enemy_turns': combat_state['enemy_turns_remaining'],
    }

    print("[Combat] " +f"✓ Combat started: {result}")
    return result


async def _generate_enemy_turn(room_id: str) -> dict:
    """
    Generate enemy actions during their turn in combat.

    Each enemy action:
    1. Gets narrated by the DM
    2. Counts toward interjection chance
    3. Could trigger a switch back to players

    Returns dict with enemy_actions list and any side_switch info.
    """
    from combat_system import EncounterManager
    import httpx

    encounter = EncounterManager.get(room_id)
    if not encounter or not encounter.is_active:
        return {'enemy_actions': [], 'side_switch': None}

    if encounter.active_side != "enemies":
        return {'enemy_actions': [], 'side_switch': None}

    room = data_store.get_room(room_id)
    if not room:
        return {'enemy_actions': [], 'side_switch': None}

    # Get enemy combatants
    enemies = [c for c in encounter.combatants.values()
               if c.team == "enemies" and c.stats.is_alive]

    if not enemies:
        return {'enemy_actions': [], 'side_switch': None}

    # Get player combatants for targeting
    players = [c for c in encounter.combatants.values()
               if c.team == "players" and c.stats.is_alive]
    player_names = [c.stats.name for c in players]

    # Get recent context
    recent_messages = room.messages[-6:] if room.messages else []
    recent_context = "\n".join([f"{m.speaker_name}: {m.content[:150]}" for m in recent_messages])

    enemy_actions = []
    side_switch = None
    narrator = get_dm_narrator()

    # Process enemy actions one at a time
    for enemy in enemies:
        if encounter.enemy_turns_remaining <= 0:
            break

        if encounter.active_side != "enemies":
            # Players interrupted us
            break

        # Generate this enemy's action
        enemy_name = enemy.stats.name
        enemy_attacks = [a.name for a in enemy.stats.attacks]

        prompt = f"""You are the DM narrating combat. An enemy is acting.

ENEMY: {enemy_name}
ATTACKS AVAILABLE: {', '.join(enemy_attacks)}
TARGETS: {', '.join(player_names)}

RECENT COMBAT:
{recent_context}

Narrate this enemy's action in 1-2 visceral sentences. They attack, move, or act tactically.
Be brutal and specific. Name the target. Describe the threat.

Examples:
- "The bandit lunges at Doc, hook knife slashing toward her throat."
- "The second man circles left, trying to flank Kaido while he's distracted."
- "It shambles forward, dead hands reaching for Lumen's face."

Enemy action:"""

        try:
            model_to_use = room.room_model or settings.storybuilder_model
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{settings.ollama_base_url}/api/chat",
                    json={
                        "model": model_to_use,
                        "messages": [
                            {"role": "system", "content": "You are a combat narrator. Be brief, brutal, visceral."},
                            {"role": "user", "content": prompt}
                        ],
                        "stream": False,
                    }
                )

                if response.status_code == 200:
                    action_text = response.json().get("message", {}).get("content", "").strip()

                    if action_text:
                        # Create message for this enemy action
                        action_msg = Message(
                            id=str(uuid.uuid4())[:8],
                            speaker_id='dm',
                            speaker_name=f'⚔️ {enemy_name}',
                            content=action_text,
                            room_id=room_id,
                            message_type='enemy_action',
                            metadata={'enemy_id': enemy.id, 'round': encounter.round},
                        )
                        data_store.add_message(room_id, action_msg)

                        enemy_actions.append({
                            'id': action_msg.id,
                            'enemy_name': enemy_name,
                            'content': action_text,
                        })

                        # Record this action
                        action_result = encounter.record_action(side="enemies")
                        print(f"[Combat] Enemy action: {enemy_name} - {encounter.enemy_turns_remaining} turns left")

                        # Check if turns exhausted
                        if action_result.get('should_auto_switch'):
                            new_side = encounter.switch_sides(reason="turns_exhausted")
                            side_switch = {
                                'from_side': 'enemies',
                                'to_side': new_side,
                                'reason': 'turns_exhausted',
                            }
                            print(f"[Combat] Enemies exhausted → {new_side}")
                            break

                        # Check for interjection (could give players an opening)
                        if narrator.record_turn(room_id):
                            new_side = encounter.switch_sides(reason="interjection")

                            # Add transition message
                            transition_msg = Message(
                                id=str(uuid.uuid4())[:8],
                                speaker_id='dm',
                                speaker_name='⚔️ Combat',
                                content="*An opening appears—*",
                                room_id=room_id,
                                message_type='combat_switch',
                            )
                            data_store.add_message(room_id, transition_msg)

                            side_switch = {
                                'from_side': 'enemies',
                                'to_side': new_side,
                                'reason': 'interjection',
                            }
                            print(f"[Combat] Interjection! → {new_side}")
                            break

        except Exception as e:
            print(f"[Combat] Error generating enemy action: {e}")
            continue

    return {
        'enemy_actions': enemy_actions,
        'side_switch': side_switch,
        'combat_state': encounter.get_combat_state(),
    }


def _auto_unequip_if_no_combat(character_id: str, room_id: str, combat_triggered: bool) -> bool:
    """
    Auto-unequip weapons after a message if no combat was triggered.

    Returns True if weapons were unequipped.
    """
    if combat_triggered:
        print("[Combat] " +f"Combat triggered - keeping weapons equipped for {character_id}")
        return False

    if not inventory_tracker.is_combat_ready(character_id):
        return False  # Already not combat ready

    # Check if there's active combat in this room
    if EncounterManager.has_active_combat(room_id):
        # Check if this character is in the combat
        encounter = EncounterManager.get(room_id)
        if encounter and character_id in encounter.combatants:
            print("[Combat] " +f"Character {character_id} in active combat - keeping weapons equipped")
            return False

    # No combat - holster weapons
    unequipped = inventory_tracker.unequip_all_weapons(character_id)
    if unequipped:
        print("[Combat] " +f"Auto-holstered for {character_id}: {unequipped}")
    else:
        # Just clear combat ready state (was using fists)
        inventory_tracker.set_combat_ready(character_id, False)
        print("[Combat] " +f"Lowered fists for {character_id}")

    return True


def _process_loot_mode(character_id: str, player_message: str, room_id: str) -> Optional[str]:
    """
    Process loot mode if a container is equipped.

    When player has backpack/bag equipped, they're signaling intent to loot.
    Parse their message for items being added and update inventory.

    Returns a status message or None if not in loot mode.
    """
    container = inventory_tracker.get_equipped_container(character_id)
    if not container:
        return None  # Not in loot mode

    print("[Inventory] " +f"LOOT MODE active for {character_id} with {container.name}")

    # Get loot status for capacity info
    loot_status = inventory_tracker.get_loot_status(character_id)
    available_capacity = loot_status.get('available', 25.0)

    # Build DM prompt to parse loot
    loot_prompt = f"""The player just performed this action with their {container.name} ready:

"{player_message}"

Are they adding any items to their inventory? If yes, list ONLY the specific items being picked up, taken, or put in their bag.

RULES:
- Only list concrete, physical items (not "stuff" or "supplies")
- Be specific: "Medical Kit" not "medical supplies"
- One item per line
- If they're NOT adding anything, respond with just: NONE
- They have {available_capacity:.1f} lbs of carrying capacity remaining

Respond with ONLY a list of item names (one per line) or NONE. No other text."""

    system = "You identify items being looted from player actions. Be specific and realistic. Only list concrete items being physically taken."

    try:
        # Get room model
        room = data_store.get_room(room_id)
        model = room.room_model if room else None

        response = _call_ollama_sync(model or settings.dm_model, loot_prompt, system)

        if not response or 'NONE' in response.upper():
            # No items detected
            inventory_tracker.unequip_containers(character_id)
            print("[Inventory] " +f"No loot detected, unequipping {container.name}")
            return "📦 No items added"

        # Parse item names from response
        items_added = []
        items_rejected = []

        for line in response.strip().split('\n'):
            item_name = line.strip()
            # Skip empty lines and common non-item responses
            if not item_name or len(item_name) < 2:
                continue
            if item_name.upper() in ['NONE', 'NO', 'NOTHING', 'N/A']:
                continue
            # Clean up common prefixes
            for prefix in ['- ', '• ', '* ', '1. ', '2. ', '3. ']:
                if item_name.startswith(prefix):
                    item_name = item_name[len(prefix):]

            # Try to add the item
            item, status = inventory_tracker.add_looted_item(
                character_id,
                item_name.strip().title(),
                description=f"Scavenged while exploring"
            )

            if item:
                items_added.append(f"{item.name} ({item.weight:.1f} lbs)")
                print("[Inventory] " +f"Added: {item.name} ({item.weight:.1f} lbs)")
            else:
                items_rejected.append(f"{item_name}: {status}")
                print("[Inventory] " +f"Rejected: {item_name} - {status}")

        # Unequip the container
        inventory_tracker.unequip_containers(character_id)

        # Build status message
        if items_added:
            msg = "📦 Added: " + ", ".join(items_added)
            if items_rejected:
                msg += f" | ❌ Couldn't fit: {', '.join(items_rejected)}"
            return msg
        elif items_rejected:
            return f"❌ Couldn't add: {', '.join(items_rejected)}"
        else:
            return "📦 No items added"

    except Exception as e:
        print("[Error] " +'inventory', f"Loot parsing failed: {e}")
        inventory_tracker.unequip_containers(character_id)
        return None


def _process_use_mode(character_id: str, player_message: str, room_id: str) -> Optional[str]:
    """
    Process use mode if a consumable/tool is equipped.

    When player has an item equipped (not weapon/container), they're signaling intent to use it.
    Parse their message to confirm usage and update inventory.

    Returns a status message or None if not in use mode.
    """
    item = inventory_tracker.get_equipped_usable(character_id)
    if not item:
        return None  # Not in use mode

    print("[Inventory] " +f"USE MODE active for {character_id} with {item.name}")

    # Build DM prompt to confirm usage
    use_prompt = f"""The player has their {item.name} ready and just did this:

"{player_message}"

Did they USE or CONSUME the {item.name}? Answer YES or NO only.

YES if they: applied it, used it, ate it, drank it, consumed it, deployed it
NO if they: just held it, mentioned it, didn't actually use it"""

    system = "You determine if an item was used. Answer only YES or NO."

    try:
        room = data_store.get_room(room_id)
        model = room.model if room else None

        response = _call_ollama_sync(model or settings.dm_model, use_prompt, system)

        # Unequip the item regardless
        inventory_tracker.unequip_consumables_and_tools(character_id)

        if response and 'YES' in response.upper():
            # Item was used - remove from inventory
            inv = inventory_tracker.get_inventory(character_id)
            if inv:
                inv.remove_item(item.id)
                inventory_tracker._save()
            print("[Inventory] " +f"Used and removed: {item.name}")
            return f"✓ Used {item.name}"
        else:
            print("[Inventory] " +f"Item not used: {item.name}")
            return None

    except Exception as e:
        print("[Error] " +'inventory', f"Use parsing failed: {e}")
        inventory_tracker.unequip_consumables_and_tools(character_id)
        return None


def _build_combat_context(room_id: str, for_character_id: str = None) -> str:
    """
    Build combat context to inject into character/DM prompts.

    Returns empty string if no active combat, otherwise returns
    a formatted block describing the combat state.
    """
    encounter = EncounterManager.get(room_id)
    if not encounter or not encounter.is_active:
        return ""

    lines = [
        "",
        "═══════════════════════════════════════",
        "⚔️  ACTIVE COMBAT  ⚔️",
        "═══════════════════════════════════════",
        f"Round {encounter.round}",
        "",
        "COMBATANTS:",
    ]

    # List all combatants with HP
    for c_id in encounter.turn_order:
        combatant = encounter.combatants.get(c_id)
        if not combatant:
            continue

        stats = combatant.stats
        hp_bar = f"{stats.hp_current}/{stats.hp_max} HP"
        team_marker = "👤" if combatant.team == "players" else "💀"

        # Mark current turn
        current_idx = encounter.turn_order.index(c_id) if c_id in encounter.turn_order else -1
        is_current = current_idx == encounter.current_turn
        turn_marker = " ◀ ACTING NOW" if is_current else ""

        # Status
        status = "ALIVE" if stats.is_alive else "DEAD"
        if not stats.is_alive:
            hp_bar = "DEAD"

        lines.append(f"  {team_marker} {stats.name}: {hp_bar}{turn_marker}")

        # Show conditions if any
        if stats.conditions:
            lines.append(f"      Conditions: {', '.join(stats.conditions)}")

    # Current turn info
    current = encounter.get_current_combatant()
    if current:
        lines.append("")
        lines.append(f"CURRENT TURN: {current.stats.name}")

        # If it's an enemy's turn, list their attacks
        if current.team == "enemies":
            lines.append(f"  Available attacks:")
            for atk in current.stats.attacks:
                dmg_info = f"{atk.damage} {atk.damage_type}"
                lines.append(f"    • {atk.name}: {dmg_info}")

    lines.append("")
    lines.append("COMBAT NARRATION RULES:")
    lines.append("- You are IN COMBAT. Describe actions dramatically but don't decide outcomes.")
    lines.append("- If it's YOUR turn: describe your attack attempt, the system will resolve it.")
    lines.append("- If it's the ENEMY's turn: the system resolves their attack, you narrate the result.")
    lines.append("- Each message covers 1-2 exchanges (6-12 seconds of in-game time).")
    lines.append("- After describing the action, wait for the next participant's input.")
    lines.append("═══════════════════════════════════════")
    lines.append("")

    return "\n".join(lines)


def _resolve_combat_action(
    room_id: str,
    actor_id: str,
    action_description: str
) -> dict:
    """
    Resolve a combat action narratively.

    Dice determine HOW WELL the attack went, but we describe injuries
    instead of HP damage. The roll maps to injury severity:
    - Critical (nat 20 or beat AC by 10+): Severe/critical injury
    - Success: Moderate injury
    - Failure: Glancing blow or miss (minor or no injury)
    - Critical failure (nat 1): Complete whiff, possible self-harm
    """
    from combat_system import Dice
    import random

    encounter = EncounterManager.get(room_id)
    if not encounter or not encounter.is_active:
        return {'error': 'No active combat'}

    actor = encounter.combatants.get(actor_id)
    if not actor:
        return {'error': 'Actor not in combat'}

    # Find likely target (for now, first living enemy)
    target = None
    target_id = None
    for c_id, c in encounter.combatants.items():
        if c.team != actor.team and c.stats.is_alive:
            target = c
            target_id = c_id
            break

    if not target:
        return {'resolved': False, 'reason': 'No valid targets'}

    # Roll the attack (still use dice for fairness)
    natural_roll = Dice.d20()
    attack_bonus = actor.stats.attacks[0].attack_bonus if actor.stats.attacks else 2
    total = natural_roll + attack_bonus
    target_ac = target.stats.ac

    # Determine degree of success
    diff = total - target_ac
    if natural_roll == 20 or diff >= 10:
        degree = "critical_success"
    elif natural_roll == 1 or diff <= -10:
        degree = "critical_failure"
    elif diff >= 0:
        degree = "success"
    else:
        degree = "failure"

    # Map degree to narrative injury
    attack_name = actor.stats.attacks[0].name if actor.stats.attacks else "attack"
    damage_type = actor.stats.attacks[0].damage_type if actor.stats.attacks else "physical"

    # Injury descriptions based on damage type and severity
    injury_templates = {
        'slashing': {
            'critical': ['deep gash across the chest', 'severed tendons in the arm', 'slash to the face'],
            'severe': ['deep cut on the torso', 'sliced forearm', 'gash on the thigh'],
            'moderate': ['cut on the arm', 'slash across the shoulder', 'laceration on the leg'],
            'minor': ['shallow cut', 'nick on the hand', 'superficial scratch'],
        },
        'piercing': {
            'critical': ['punctured lung', 'stab wound to the gut', 'pierced shoulder'],
            'severe': ['deep puncture wound', 'stab to the side', 'pierced thigh'],
            'moderate': ['puncture wound on the arm', 'stab wound on the leg', 'pierced hand'],
            'minor': ['shallow puncture', 'grazed by the point', 'small stab wound'],
        },
        'bludgeoning': {
            'critical': ['cracked ribs', 'fractured skull', 'shattered arm'],
            'severe': ['broken nose', 'badly bruised ribs', 'dislocated shoulder'],
            'moderate': ['bruised ribs', 'swollen jaw', 'sprained wrist'],
            'minor': ['light bruise', 'minor bump', 'sore spot'],
        },
    }

    # Default to physical if damage type not found
    templates = injury_templates.get(damage_type, injury_templates['bludgeoning'])

    resolution = {
        'resolved': True,
        'attacker': actor.stats.name,
        'defender': target.stats.name,
        'attack_name': attack_name,
        'roll_details': {
            'natural': natural_roll,
            'total': total,
            'target_ac': target_ac,
            'degree': degree,
        }
    }

    if degree == "critical_success":
        # Devastating hit - severe or critical injury, possibly bleeding
        injury_desc = random.choice(templates['critical'])
        resolution['hit'] = True
        resolution['crit'] = True
        resolution['injury'] = {
            'description': injury_desc,
            'severity': 'severe',
            'bleeding': True,
        }
        resolution['narrative'] = f"DEVASTATING BLOW! {actor.stats.name}'s {attack_name} catches {target.stats.name} perfectly - {injury_desc}!"

        # If target is a player/partner, add injury to condition tracker
        if target.is_companion and target.companion_id:
            condition_tracker.add_injury(
                target.companion_id,
                target.stats.name,
                injury_desc,
                'severe',
                bleeding=True,
                inflicted_by=actor.stats.name
            )

    elif degree == "success":
        # Solid hit - moderate injury
        injury_desc = random.choice(templates['moderate'])
        bleeding = random.random() < 0.3  # 30% chance of bleeding
        resolution['hit'] = True
        resolution['crit'] = False
        resolution['injury'] = {
            'description': injury_desc,
            'severity': 'moderate',
            'bleeding': bleeding,
        }
        resolution['narrative'] = f"{actor.stats.name}'s {attack_name} connects - {injury_desc}."

        if target.is_companion and target.companion_id:
            condition_tracker.add_injury(
                target.companion_id,
                target.stats.name,
                injury_desc,
                'moderate',
                bleeding=bleeding,
                inflicted_by=actor.stats.name
            )

    elif degree == "failure":
        # Glancing blow or near miss - minor or no injury
        if diff >= -5:
            # Close miss, maybe a scratch
            injury_desc = random.choice(templates['minor'])
            resolution['hit'] = True
            resolution['crit'] = False
            resolution['injury'] = {
                'description': injury_desc,
                'severity': 'minor',
                'bleeding': False,
            }
            resolution['narrative'] = f"{actor.stats.name}'s {attack_name} grazes {target.stats.name} - {injury_desc}."

            if target.is_companion and target.companion_id:
                condition_tracker.add_injury(
                    target.companion_id,
                    target.stats.name,
                    injury_desc,
                    'minor',
                    bleeding=False,
                    inflicted_by=actor.stats.name
                )
        else:
            # Clean miss
            resolution['hit'] = False
            resolution['crit'] = False
            resolution['injury'] = None
            resolution['narrative'] = f"{actor.stats.name}'s {attack_name} misses {target.stats.name}."

    else:  # critical_failure
        # Complete whiff
        resolution['hit'] = False
        resolution['crit'] = False
        resolution['injury'] = None
        resolution['narrative'] = f"{actor.stats.name} swings wildly and completely misses!"

    # Log the combat action
    print("[Combat] " +f"⚔️ {resolution['attacker']} → {resolution['defender']}: {resolution['narrative']}")

    # Check for incapacitation (replaces "death" check)
    # A character becomes incapacitated if they have 2+ severe injuries or 1 critical
    if target.is_companion and target.companion_id:
        target_condition = condition_tracker.get(target.companion_id)
        if target_condition:
            severe_count = sum(1 for i in target_condition.injuries if i.severity.value in ['severe', 'critical'] and not i.treated)
            if severe_count >= 2 or target_condition.condition == OverallCondition.CRITICAL:
                target_condition.set_condition('incapacitated')
                condition_tracker.save()
                resolution['defender_incapacitated'] = True
                resolution['narrative'] += f" {target.stats.name} collapses, unable to continue fighting!"
                print("[Combat] " +f"💀 {target.stats.name} is INCAPACITATED!")

    # For enemies (zombies etc), we still track their "alive" state
    # Multiple solid hits should eventually take them down
    if target.team == "enemies" and resolution.get('hit'):
        # Increment a hit counter on the enemy
        if not hasattr(target, 'hits_taken'):
            target.hits_taken = 0
        target.hits_taken += 1

        # Enemies go down after enough hits (scaled by their starting HP for balance)
        hits_to_down = max(2, target.stats.hp_max // 10)
        if degree == "critical_success":
            target.hits_taken += 1  # Crits count double

        if target.hits_taken >= hits_to_down:
            target.stats.is_alive = False
            resolution['defender_died'] = True
            resolution['narrative'] += f" {target.stats.name} goes down!"
            print("[Combat] " +f"💀 {target.stats.name} is DEFEATED!")

    # Check if combat should end
    enemies_alive = any(c.stats.is_alive for c in encounter.combatants.values() if c.team == "enemies")
    players_able = any(
        c.stats.is_alive and not resolution.get('defender_incapacitated', False)
        for c in encounter.combatants.values() if c.team == "players"
    )

    if not enemies_alive:
        encounter.end_combat("victory")
        resolution['combat_ended'] = True
        resolution['end_reason'] = 'victory'
        print("[Combat] " +"✓ Combat ended: VICTORY")
    elif not players_able:
        encounter.end_combat("defeat")
        resolution['combat_ended'] = True
        resolution['end_reason'] = 'defeat'
        print("[Combat] " +"✗ Combat ended: DEFEAT - all players incapacitated")

    return resolution


def _inject_combat_resolution_context(base_context: str, resolution: dict) -> str:
    """
    Inject combat resolution into the context for narration.
    Uses narrative injuries instead of HP damage.
    """
    if not resolution or not resolution.get('resolved'):
        return base_context

    lines = [
        "",
        "┌─────────────────────────────────────┐",
        "│  COMBAT RESOLUTION (narrate this)  │",
        "└─────────────────────────────────────┘",
    ]

    # Add the pre-generated narrative
    lines.append(resolution.get('narrative', 'The combatants clash.'))

    # Add injury details if there was a hit
    injury = resolution.get('injury')
    if injury:
        lines.append(f"  Injury: {injury['description']} ({injury['severity']})")
        if injury.get('bleeding'):
            lines.append(f"  ⚠️ BLEEDING - needs to be stopped!")

    if resolution.get('defender_incapacitated'):
        lines.append(f"  💀 {resolution['defender']} is INCAPACITATED and cannot continue!")

    if resolution.get('defender_died'):
        lines.append(f"  💀 {resolution['defender']} has been DEFEATED!")

    if resolution.get('combat_ended'):
        if resolution['end_reason'] == 'victory':
            lines.append("")
            lines.append("🏆 COMBAT ENDED - VICTORY!")
            lines.append("All enemies have been defeated. Describe the aftermath.")
        elif resolution['end_reason'] == 'defeat':
            lines.append("")
            lines.append("💀 COMBAT ENDED - DEFEAT!")
            lines.append("All players are incapacitated. Describe the grim outcome.")

    lines.append("")
    lines.append("Narrate this dramatically and viscerally. Describe the injury, the pain, the blood.")
    lines.append("")

    return base_context + "\n".join(lines)


# StoryBuilder - Generate random scenarios and characters for RP rooms
# ============================================================================

# Store character secrets server-side (never sent to client)
# These are used when creating Partners from StoryBuilder characters
_character_secrets = {}  # char_id -> secret string


def _call_ollama_sync(model: str, prompt: str, system: str = "", retries: int = 2, timeout: float = 120.0) -> str:
    """Synchronous Ollama call for StoryBuilder (non-streaming) with retry logic."""
    import httpx
    import time

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error = None
    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                print(f"[StoryBuilder] Retry attempt {attempt}/{retries}...")
                time.sleep(2)  # Brief pause before retry

            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    f"{settings.ollama_base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("message", {}).get("content", "")
                    if content:
                        return content
                    else:
                        last_error = "Empty response from model"
                        print(f"[StoryBuilder] Empty response, attempt {attempt + 1}")
                else:
                    last_error = f"HTTP {response.status_code}"
                    print(f"[StoryBuilder] Ollama error: {response.status_code}, attempt {attempt + 1}")
        except httpx.TimeoutException:
            last_error = "Request timed out"
            print(f"[StoryBuilder] Timeout after {timeout}s, attempt {attempt + 1}")
        except Exception as e:
            last_error = str(e)
            print(f"[StoryBuilder] Ollama exception: {e}, attempt {attempt + 1}")

    print(f"[StoryBuilder] All {retries + 1} attempts failed. Last error: {last_error}")
    return ""


# =============================================================================
# Hidden Zombie Rules System - Roll behavior traits at game start
# =============================================================================

def roll_zombie_rules(zombie_type: str = None, override_rules: dict = None) -> dict:
    """
    Roll hidden zombie behavior rules at game start.

    Players discover these through play, not upfront selection.
    DM uses these to maintain consistent zombie behavior.

    Args:
        zombie_type: Pre-selected type ('shamblers', 'runners', 'nightmares') or None for random
        override_rules: Advanced user overrides (any keys set here won't be rolled)

    Returns dict with:
        - type: 'shamblers', 'runners', or 'nightmares'
        - headshots_kill: True = instant kill, False = just blinds/disables sensory
        - sound_attracts: True = gunshots/noise draws zombies, False = they ignore sound
        - decay_enabled: True = zombies weaken over time (years), False = eternal threat
        - hive_behavior: True = coordinated, harder to sneak past, False = dumb individuals
        - freshness: 'outbreak' (0-6mo), 'early' (6mo-2yr), 'established' (2-10yr), 'remnant' (10+yr)
        - night_hunters: True = more active/dangerous at night, False = same day/night
        - can_run: True = capable of running (even shamblers might sprint briefly), False = always slow
        - infection_rate: 'fast' (minutes), 'slow' (hours), 'variable' (depends on wound)
    """
    import random

    rules = override_rules.copy() if override_rules else {}

    # Zombie type - might already be set
    if 'type' not in rules:
        if zombie_type:
            rules['type'] = zombie_type
        else:
            # Weight towards shamblers as most common, nightmares rare
            rules['type'] = random.choices(
                ['shamblers', 'runners', 'nightmares'],
                weights=[50, 35, 15]
            )[0]

    # Headshots - classic rule but not universal
    # Your virus-in-muscles theory: headshots just blind them, don't kill
    if 'headshots_kill' not in rules:
        # 60% traditional headshots work, 40% they just disable sensory
        rules['headshots_kill'] = random.random() < 0.6

    # Sound attraction - do gunshots draw zombies?
    if 'sound_attracts' not in rules:
        # 75% yes, noise is dangerous
        rules['sound_attracts'] = random.random() < 0.75

    # Decay over time - do zombies weaken over years?
    if 'decay_enabled' not in rules:
        # 70% yes, time is on your side eventually
        rules['decay_enabled'] = random.random() < 0.70

    # Hive behavior - coordinated or dumb?
    if 'hive_behavior' not in rules:
        # Base on type: nightmares more likely coordinated
        if rules['type'] == 'nightmares':
            rules['hive_behavior'] = random.random() < 0.80
        elif rules['type'] == 'runners':
            rules['hive_behavior'] = random.random() < 0.40
        else:  # shamblers
            rules['hive_behavior'] = random.random() < 0.20

    # Freshness - how long since outbreak?
    if 'freshness' not in rules:
        rules['freshness'] = random.choices(
            ['outbreak', 'early', 'established', 'remnant'],
            weights=[30, 40, 25, 5]  # Most games in early/outbreak phase
        )[0]

    # Night hunters - more dangerous at night?
    if 'night_hunters' not in rules:
        # Runners and nightmares more likely to be night-active
        if rules['type'] == 'nightmares':
            rules['night_hunters'] = random.random() < 0.70
        elif rules['type'] == 'runners':
            rules['night_hunters'] = random.random() < 0.50
        else:
            rules['night_hunters'] = random.random() < 0.30

    # Can run - even shamblers might have brief sprints
    if 'can_run' not in rules:
        if rules['type'] == 'shamblers':
            rules['can_run'] = random.random() < 0.15  # Rare burst
        elif rules['type'] == 'runners':
            rules['can_run'] = True  # Always
        else:  # nightmares
            rules['can_run'] = True  # Always

    # Infection rate
    if 'infection_rate' not in rules:
        rules['infection_rate'] = random.choices(
            ['fast', 'slow', 'variable'],
            weights=[35, 30, 35]
        )[0]

    return rules


def get_shelter_dm_context(shelter_type: str) -> str:
    """
    Generate DM context string for shelter situation.
    Affects story tone, objectives, and what kinds of challenges make sense.
    """
    shelter_contexts = {
        'wandering': """=== SHELTER STATUS: WANDERING ===
The group has NO BASE. They are nomads, traveling on foot or by vehicle, looking for somewhere safe to settle.
- Every night is a new camp, a new risk
- Resources must be carried - weight matters
- The story is about the SEARCH for safety, not the defense of it
- Encounters focus on: travel dangers, finding shelter for the night, resource scarcity, other travelers
- Key tension: Where do we go? Is anywhere truly safe?""",

        'temporary': """=== SHELTER STATUS: TEMPORARY ===
The group has temporary shelter - tents, vehicles, or a building they're using for a night or two.
- No real fortification, no commitment to this location
- They could move on tomorrow if needed
- Resources are mobile, packed for travel
- Encounters focus on: night watches, unexpected threats, deciding when to move on
- Key tension: Do we stay another day or keep moving?""",

        'shanty': """=== SHELTER STATUS: SHANTY ===
The group has makeshift shelter with basic barricades - a building with boarded windows, a camp with fencing.
- Provides minimal protection, could be breached by determined assault
- They're TRYING to make it work, but it's fragile
- Some resources can be stored, but not safely
- Encounters focus on: reinforcing defenses, dealing with probing attacks, resource runs
- Key tension: Can we make this place safe, or should we find somewhere better?""",

        'decent': """=== SHELTER STATUS: DECENT ===
The group has solid shelter - a house, apartment, or building with real walls, locks, and basic fortifications.
- Defensible against small threats, but not a siege
- Feels like a HOME - people are putting down roots
- Resources can be stockpiled with reasonable security
- Encounters focus on: community building, supply runs, defending against raids, internal conflicts
- Key tension: This is worth protecting, but can we hold it?""",

        'reinforced': """=== SHELTER STATUS: REINFORCED ===
The group has a well-fortified position - barricades, secure perimeter, multiple fallback points.
- Can withstand significant assault - this is a STRONGHOLD
- The group has invested heavily in defense - leaving would hurt
- Substantial resource storage, workshops, infrastructure
- Encounters focus on: organized threats, siege scenarios, power dynamics, expansion
- Key tension: We're strong, but strength attracts attention. Who else wants what we have?""",

        'fortress': """=== SHELTER STATUS: FORTRESS ===
The group occupies a maximum-security location - prison, military base, walled compound.
- Nearly impregnable from outside assault
- Has infrastructure: power, water, storage, medical facilities
- The story is about what happens INSIDE the walls
- Encounters focus on: politics, power struggles, who belongs, resource allocation, moral choices
- Key tension: The walls keep things out... but they also keep things IN. Who's really in charge?"""
    }

    return shelter_contexts.get(shelter_type, '')


def get_zombie_rules_dm_context(rules: dict) -> str:
    """
    Generate DM context string from zombie rules.
    This is injected into DM prompts so behavior is consistent.
    """
    if not rules:
        return ""

    lines = ["=== HIDDEN ZOMBIE RULES (players discover through play) ==="]

    # Type
    type_desc = {
        'shamblers': "SHAMBLERS - Slow, shambling gait. Dangerous in numbers. Can be outrun.",
        'runners': "RUNNERS - Fast, aggressive. Can sprint. Pursuit is deadly.",
        'nightmares': "NIGHTMARES - Apex predators. Fast, cunning, may use tactics."
    }
    lines.append(type_desc.get(rules.get('type', 'shamblers'), "Unknown type"))

    # Headshots
    if rules.get('headshots_kill'):
        lines.append("HEADSHOTS: Kill instantly (traditional rules)")
    else:
        lines.append("HEADSHOTS: Destroy eyes/ears but body keeps moving. Must sever head or destroy body mass.")

    # Sound
    if rules.get('sound_attracts'):
        lines.append("SOUND: Attracts zombies. Gunshots are LOUD. Radius ~50 units, estimating 5+ zombies converging.")
    else:
        lines.append("SOUND: These zombies don't respond to noise. Hunt by sight/smell instead.")

    # Decay
    if rules.get('decay_enabled'):
        lines.append("DECAY: Time weakens them. Fresh = full strength. Years = shambling husks.")
    else:
        lines.append("DECAY: Virus preserves them. No natural weakening over time.")

    # Freshness
    freshness_desc = {
        'outbreak': "TIMELINE: Active outbreak (0-6 months). Zombies at peak strength. Chaos everywhere.",
        'early': "TIMELINE: Early survival (6 months - 2 years). Zombies strong but patterns emerging.",
        'established': "TIMELINE: Established apocalypse (2-10 years). Decay setting in. Survivors adapted.",
        'remnant': "TIMELINE: Remnant era (10+ years). Most zombies are husks. New outbreaks from hidden infected."
    }
    lines.append(freshness_desc.get(rules.get('freshness', 'early'), ""))

    # Hive behavior
    if rules.get('hive_behavior'):
        lines.append("COORDINATION: Hive-like. They sense each other. Hard to sneak past groups.")
    else:
        lines.append("COORDINATION: Individuals. Dumb. Can be distracted, snuck past.")

    # Night hunters
    if rules.get('night_hunters'):
        lines.append("NIGHT: More active/dangerous after dark. Daylight is safer.")
    else:
        lines.append("NIGHT: Same threat level day or night.")

    # Infection rate
    infection_desc = {
        'fast': "INFECTION: Fast (minutes to turn). Bitten = almost certainly dead.",
        'slow': "INFECTION: Slow (hours). Time to amputate, say goodbye, or find a cure.",
        'variable': "INFECTION: Variable - depends on wound severity, immune system, bite location."
    }
    lines.append(infection_desc.get(rules.get('infection_rate', 'variable'), ""))

    lines.append("")
    lines.append("IMPORTANT: Players don't know these rules. Let them discover through play.")
    lines.append("If they try something (headshot, sneaking, etc.), describe the RESULT, not the rule.")

    return "\n".join(lines)


# =============================================================================
# Population Density System - Realistic zombie encounter frequency
# =============================================================================

def infer_population_density(seed_where: str) -> dict:
    """
    Infer population density from the WHERE seed.
    Returns density category and estimated pre-outbreak population.

    This affects zombie encounter frequency:
    - Rural areas: sparse zombies, encounters are rare but notable
    - Urban areas: swarming, every street is dangerous
    """
    import random
    seed_lower = (seed_where or "").lower()

    # Map location types to density categories
    # Format: (density_level, pre_outbreak_pop_estimate, description)
    density_map = {
        # High density locations
        "hospital": ("urban", 5000, "Medical facilities draw crowds - patients, staff, visitors. Now they're trapped inside."),
        "medical": ("urban", 5000, "Medical facilities draw crowds - patients, staff, visitors. Now they're trapped inside."),
        "mall": ("urban", 8000, "Shopping centers held thousands when it hit. Many never made it to the exits."),
        "shopping": ("urban", 8000, "Shopping centers held thousands when it hit. Many never made it to the exits."),
        "big box": ("urban", 8000, "Shopping centers held thousands when it hit. Many never made it to the exits."),
        "stadium": ("metro", 20000, "When it happened during an event, tens of thousands were trapped. The exits became death traps."),
        "convention": ("metro", 15000, "Convention centers hold massive crowds. The infected spread through them like wildfire."),
        "apartment": ("urban", 10000, "High-rises packed with residents. Every floor is a gauntlet now."),
        "housing project": ("urban", 10000, "Dense housing means dense undead. Hundreds per building."),
        "university": ("suburban", 15000, "Campuses held thousands of students. Dorms became tombs."),
        "school": ("suburban", 2000, "Schools had hundreds when classes were in session."),
        "prison": ("suburban", 3000, "Inmates and guards alike. The walls that kept people in now keep the dead in too."),
        "secure facility": ("suburban", 2000, "Controlled populations, but still significant numbers."),

        # Medium density
        "suburb": ("suburban", 5000, "Residential neighborhoods. Families in every house. Now shambling in yards and streets."),
        "gated community": ("suburban", 3000, "Exclusive neighborhoods still had hundreds of families."),
        "trailer park": ("small_town", 500, "Close-quarters living means the infection spread fast through tight communities."),
        "mobile home": ("small_town", 500, "Close-quarters living means the infection spread fast."),
        "industrial": ("suburban", 2000, "Workers staffed these facilities around the clock. Many died at their posts."),
        "factory": ("suburban", 2000, "Workers staffed these facilities around the clock."),
        "warehouse": ("suburban", 1000, "Fewer people, but they're scattered through vast spaces."),
        "religious": ("small_town", 300, "Congregations gathered here. Some still do - as the dead."),
        "church": ("small_town", 300, "Congregations gathered here. Some still do - as the dead."),
        "monastery": ("isolated", 50, "Small, isolated religious communities."),

        # Low density
        "rural": ("rural", 200, "Scattered population. Zombies are rare but so is help."),
        "small town": ("small_town", 800, "Everyone knew everyone. Now you might recognize faces in the horde."),
        "coastal": ("small_town", 600, "Fishing villages and port towns. Moderate populations, spread along the water."),
        "fishing": ("small_town", 400, "Small working communities. The boats are still there."),
        "farm": ("rural", 100, "Isolated homesteads. You might go days without seeing the dead."),
        "agricultural": ("rural", 150, "Farmland means spread-out population. Zombies wander but don't swarm."),
        "highway": ("rural", 50, "Travelers, not residents. The dead here came from elsewhere."),
        "travel corridor": ("rural", 50, "Transient population. Crashed cars hold the remains."),

        # Very low density
        "island": ("isolated", 100, "Limited population by geography. If you cleared it, it might stay clear."),
        "peninsula": ("isolated", 200, "Cut off from the mainland. Natural chokepoint."),
        "underground": ("isolated", 50, "Subways had crowds during rush hour. Bunkers had handfuls."),
        "subway": ("suburban", 3000, "Rush hour underground was packed. Now it's a catacomb."),
        "bunker": ("isolated", 30, "Small groups sought shelter. Some made it. Some turned inside."),
        "mine": ("isolated", 100, "Workers were sparse to begin with."),
        "military": ("suburban", 2000, "Bases had trained personnel. They fought back. Many still fell."),
        "national guard": ("suburban", 1500, "Called up during the crisis. Many were bitten responding."),
    }

    # Find best match - prefer more specific matches (longer keys)
    best_match = None
    best_match_len = 0

    for key, value in density_map.items():
        if key in seed_lower:
            # Prefer longer matches (more specific)
            if len(key) > best_match_len:
                best_match = value
                best_match_len = len(key)

    # Special case: if both "rural" and "town" appear, prioritize rural
    if "rural" in seed_lower and best_match and best_match[0] != "rural":
        # Check if there's a rural match we should use instead
        if "rural" in density_map:
            best_match = density_map["rural"]

    # Default to small_town if no match
    if not best_match:
        best_match = ("small_town", 1000, "A typical community. Some remain alive. Most don't.")

    density_level, pre_pop, description = best_match

    # Calculate current zombie estimate based on typical mortality (assume 85-95% became zombies)
    zombie_percentage = random.uniform(0.85, 0.95)
    estimated_zombies = int(pre_pop * zombie_percentage)

    return {
        "density_level": density_level,  # isolated, rural, small_town, suburban, urban, metro
        "pre_outbreak_population": pre_pop,
        "estimated_zombies": estimated_zombies,
        "description": description,
        "dm_context": _get_density_dm_context(density_level, estimated_zombies)
    }

def _get_density_dm_context(density_level: str, zombie_count: int) -> str:
    """Generate DM context string for population density."""
    # Note on zombie behavior: They naturally cluster. Lone zombies are outliers.
    # Small groups (5-15) are common. Packs (15-30) happen. Mobs (100+) are rare events.
    contexts = {
        "isolated": f"POPULATION DENSITY: Isolated (~{zombie_count} zombies in area). Encounters are RARE - you might go days without seeing one. When you do, it's usually 1-3 stragglers. Clearing this area is achievable and it might actually STAY cleared.",
        "rural": f"POPULATION DENSITY: Rural (~{zombie_count} zombies in area). Encounters are uncommon but zombies travel in small groups (3-8). Lone zombies are the exception. Packs of 10-20 form around points of interest. Clearing areas is possible with sustained effort.",
        "small_town": f"POPULATION DENSITY: Small Town (~{zombie_count} zombies in area). Moderate presence. Zombies cluster in groups of 5-15 around main streets and buildings. Larger packs (20-40) gather at chokepoints. Mobs are rare but can form if noise draws them.",
        "suburban": f"POPULATION DENSITY: Suburban (~{zombie_count} zombies in area). Significant presence. Groups of 10-20 are common on every block. Packs of 30-50 roam between houses. Noise attracts more fast. Houses need clearing before they're safe.",
        "urban": f"POPULATION DENSITY: Urban (~{zombie_count} zombies in area). DENSE. Streets have groups of 20-50 visible at any time. Packs merge into mobs easily. Buildings are packed with clusters on every floor. Going outside without a plan is suicide.",
        "metro": f"POPULATION DENSITY: Metropolitan (~{zombie_count} zombies in area). SWARMING. The streets ARE mobs - hundreds moving together. Every intersection is a death trap. Every building holds dozens. You don't clear this - you survive passage through it."
    }
    return contexts.get(density_level, contexts["small_town"])


@app.route('/storybuilder/scenarios', methods=['POST'])
def generate_scenarios():
    """Generate 3 scenario options for StoryBuilder."""
    import random

    data = request.json or {}
    model = data.get('model') or settings.storybuilder_model
    genre = data.get('genre', '')  # Optional genre hint
    country = data.get('country', '')  # Optional country/setting hint
    mode = data.get('mode', 'story')  # 'story' (quest hooks) or 'world' (world-state)
    zombie_type = data.get('zombie_type')  # 'shamblers', 'runners', 'nightmares', or None

    # Seed values from frontend dropdowns (or None if not provided)
    seed_when = data.get('seed_when')
    seed_where = data.get('seed_where')

    genre_hint = f" The genre/setting should be: {genre}." if genre else ""
    country_hint = f" Set the scenarios in or around: {country}." if country else ""

    # Handle mystery zombie type - roll it now so scenarios are consistent
    resolved_zombie_type = zombie_type
    if zombie_type == 'mystery' or not zombie_type:
        resolved_zombie_type = random.choice(['shamblers', 'runners', 'nightmares'])
        print(f"[StoryBuilder] Zombie type for scenarios: {resolved_zombie_type} (player selected: {zombie_type or 'none'})")

    # Zombie type specific threat descriptions
    zombie_threat_hints = {
        'shamblers': "ThreatType MUST describe SLOW SHAMBLERS - classic Romero zombies. Slow-moving, unintelligent, only dangerous in groups. They shamble, they don't run. Example: 'Slow shamblers, drawn to noise. They wander the roads and collect in town centers.'",
        'runners': "ThreatType MUST describe RUNNERS - fast, aggressive infected like 28 Days Later. They SPRINT at full speed, are highly alert, and one alerting others creates cascading swarms. Example: 'Runners - fast and vicious. They hunt in packs and can outrun any human.'",
        'nightmares': "ThreatType MUST describe NIGHTMARES - mutated horrors like Resident Evil. Some crawl on walls, some are massive, some split apart. Unpredictable and terrifying. Example: 'Nightmares stalk these woods - twisted things that used to be human. No two are alike.'",
    }
    zombie_hint = zombie_threat_hints.get(resolved_zombie_type, "")

    # Seed options (used for random fallback if not provided by frontend)
    timelines = [
        "a few days before the outbreak",
        "day one - the outbreak just started",
        "first week - chaos, no one knows what's happening",
        "one month in - initial die-off complete, survivors emerging",
        "six months later - new routines forming",
        "two years in - the new normal",
        "five years later - children growing up in this world",
        "a generation later - the old world is stories",
    ]
    setting_types = [
        "a hospital or medical facility",
        "a school or university campus",
        "a shopping mall or big box store",
        "a prison or secure facility",
        "a small rural town",
        "coastal - boats, docks, fishing village",
        "industrial - factory, warehouse district, rail yard",
        "a highway or travel corridor",
        "a farm or agricultural area",
        "a military installation or national guard base",
        "apartment buildings or housing projects",
        "a religious compound - church, temple, monastery",
        "an island or isolated peninsula",
        "underground - subway, bunker, mine",
        "a stadium or convention center",
        "a trailer park or mobile home community",
        "a gated community or suburbs",
    ]
    # Use frontend seed values if provided, otherwise random
    seed_timeline = seed_when if seed_when else random.choice(timelines)
    seed_setting = seed_where if seed_where else random.choice(setting_types)

    # If country is specified, don't let setting_type override geographic location
    # Setting type becomes "type of place" not "where"
    seed_block = f"""
INSPIRATION SEED (this batch's theme - interpret creatively):
- Timeline: {seed_timeline}
- Setting type: {seed_setting}

All 3 scenarios should explore variations on this seed. Same general vibe, different specific situations.
Let the human drama emerge naturally from the setting - don't prescribe specific social dynamics.
"""

    if mode == 'world':
        # World-building mode: describe a REGION/TERRAIN, not a mission or quest
        location_instruction = f"All regions MUST be set in: {country}. " if country else "Vary locations globally - not just American cities. Consider: rural areas, islands, coastal regions, different countries, etc. "

        # Add zombie type instruction if specified
        zombie_type_instruction = f"\n   ** {zombie_hint} **" if zombie_hint else ""

        prompt = f"""Generate exactly 3 unique zombie survival REGIONS.{genre_hint}{country_hint}
{seed_block}

You are describing TERRAIN, not missions. Each region is a place that EXISTS - a canvas for stories to emerge, not a quest to complete.

ZOMBIE RULES - THIS IS CRITICAL:
- Zombies are REANIMATED HUMAN CORPSES. Period.
- They are slow or fast, they bite, they spread infection, they were once human
- Do NOT invent sci-fi variants: no phasing zombies, no plant zombies, no psychic zombies, no hive-mind zombies
- Do NOT make zombies supernatural beyond basic reanimation (no magic, no special powers)
- The horror comes from: overwhelming numbers, loss of loved ones, societal collapse, human desperation
- Think: Walking Dead, 28 Days Later, Dawn of the Dead - GROUNDED zombie fiction

{location_instruction}

For each region, provide:
1. Title: A geographic/evocative name (3-6 words)
2. Description: 2-4 sentences covering:
   - Where and when (geographic region, timeline since outbreak)
   - What happened here (how collapse played out locally)
   - The texture (daily life, atmosphere, what kind of people)
   - 1-2 features (places or groups that exist - NOT quests, just things that ARE)
3. ThreatType: One sentence describing zombie behavior in this region{zombie_type_instruction}
4. WorldState: A data readout of civilization status

CRITICAL RULES:
- Describe a REGION, not a single location. The region should have multiple places one COULD go.
- Do NOT set up conflicts ("must decide", "running low", "tensions rising") - let drama emerge from play
- Do NOT make any single feature "the point" - everything is optional
- NEVER use "you" or "your" - just describe what exists
- Features are points of interest, not objectives

GOOD EXAMPLE:
Title: Rural Georgia Heartland
Description: Six months after. The small towns emptied fast when everyone fled to Atlanta - now they're quiet. Farmland, pine forests, red dirt roads. A state prison about 20 miles north still has lights at night. Scattered families have claimed farmhouses, mostly keeping to themselves. Church steeples mark where people used to gather; some still do.
ThreatType: Slow shamblers, drawn to noise. They wander the roads and collect in town centers.
WorldState: POP: ~8% | POWER: None | COMMS: Word of mouth | GOV: None | SCOPE: Regional

Notice: The prison is ONE feature, not the premise. You could visit it, ignore it, or never know it exists. The description is a MAP, not a MISSION.

Format EXACTLY like this:
---SCENARIO 1---
Title: [title]
Description: [2-4 sentences - terrain, texture, features]
ThreatType: [one sentence]
WorldState: POP: [...] | POWER: [...] | COMMS: [...] | GOV: [...] | SCOPE: [...]
---SCENARIO 2---
Title: [title]
Description: [2-4 sentences]
ThreatType: [one sentence]
WorldState: POP: [...] | POWER: [...] | COMMS: [...] | GOV: [...] | SCOPE: [...]
---SCENARIO 3---
Title: [title]
Description: [2-4 sentences]
ThreatType: [one sentence]
WorldState: POP: [...] | POWER: [...] | COMMS: [...] | GOV: [...] | SCOPE: [...]

Make each region feel like a place you could EXIST in - not a problem to solve."""

        system = "You are describing zombie survival TERRAIN - regions that exist as canvases for emergent stories. Describe places, atmosphere, and features WITHOUT setting up quests or conflicts. Zombies are grounded reanimated corpses, no sci-fi powers. Focus on geographic texture and the feel of daily existence."
    else:
        # Story mode: generate quest/adventure hooks
        prompt = f"""Generate exactly 3 unique roleplay scenario ideas.{genre_hint}{country_hint}

For each scenario, provide:
1. A short title (3-6 words)
2. A one-sentence hook that makes it intriguing

CRITICAL: The hook is what PLAYERS see. Do NOT reveal twists, secrets, or surprises in the hook.
- BAD: "...but the king is secretly dead" (this spoils the twist!)
- GOOD: "...summoned to the silent castle for reasons unknown"

The hook should create mystery and pull, not solve it. Be wildly creative - avoid common tropes.

Format your response EXACTLY like this (use these exact markers):
---SCENARIO 1---
Title: [title here]
Hook: [one sentence hook - NO SPOILERS]
---SCENARIO 2---
Title: [title here]
Hook: [one sentence hook - NO SPOILERS]
---SCENARIO 3---
Title: [title here]
Hook: [one sentence hook - NO SPOILERS]

Be wildly creative and varied. Each scenario MUST have a completely different LOCATION/SETTING:
- NOT all islands, NOT all broadcasts, NOT all greenhouses
- Think: subway tunnels, ski resort, aircraft carrier, prison, amusement park, museum, subway, cathedral, dam, mine shaft, airport, casino, cruise ship, hospital, stadium, school, mall, factory, bunker, lighthouse, zoo, theater, train, etc.

Make each scenario feel distinct in both concept AND place."""

        system = "You are a creative writing assistant. Generate vivid, engaging roleplay scenarios. Be concise but evocative."

    response = _call_ollama_sync(model, prompt, system)

    # Parse the response
    scenarios = []
    parts = response.split('---SCENARIO')

    for part in parts[1:]:  # Skip first empty part
        lines = part.strip().split('\n')
        title = ""
        hook = ""
        threat_type = ""
        world_state = ""

        for line in lines:
            line = line.strip()
            if line.startswith('Title:'):
                title = line[6:].strip()
            elif line.startswith('Hook:'):
                hook = line[5:].strip()
            elif line.startswith('Description:'):
                hook = line[12:].strip()  # Description maps to hook for frontend compatibility
            elif line.startswith('ThreatType:'):
                threat_type = line[11:].strip()
            elif line.startswith('WorldState:'):
                world_state = line[11:].strip()

        if title and hook:
            scenario_data = {
                'id': str(uuid.uuid4())[:8],
                'title': title,
                'hook': hook
            }
            # Only include threat_type and world_state for world mode (zombie scenarios etc)
            if threat_type:
                scenario_data['threat_type'] = threat_type
            if world_state:
                scenario_data['world_state'] = world_state
            scenarios.append(scenario_data)

    # If parsing failed, create placeholder scenarios
    while len(scenarios) < 3:
        scenarios.append({
            'id': str(uuid.uuid4())[:8],
            'title': f"Mystery Scenario {len(scenarios) + 1}",
            'hook': "An intriguing situation awaits..."
        })

    response = {
        'scenarios': scenarios[:3],
        'seed': {
            'timeline': seed_timeline,
            'setting': seed_setting,
        }
    }
    # Include resolved zombie type so room creation uses the same type as scenarios
    if mode == 'world' and resolved_zombie_type:
        response['resolved_zombie_type'] = resolved_zombie_type
    return jsonify(response)


@app.route('/storybuilder/expand', methods=['POST'])
def expand_scenario():
    """Expand a selected scenario with full description, genre, and factions."""
    data = request.json or {}
    model = data.get('model', settings.storybuilder_model)
    title = data.get('title', '')
    hook = data.get('hook', '')

    prompt = f"""Expand this roleplay scenario:
Title: {title}
Hook: {hook}

Provide the following in this EXACT format:

DESCRIPTION:
[Write a vivid 2-3 paragraph scene description that sets atmosphere, establishes the situation, and leaves room for characters to interact. Present tense. Do NOT create specific characters. This is what PLAYERS see - do NOT reveal secrets or twists here.]

GENRE:
[One word: zombie, dystopia, fantasy, noir, horror, comedy, drama, thriller, scifi, western, or similar]

MOOD:
[One sentence describing the emotional texture - how do people ACT in this world? Are they on edge? Hiding fear behind smiles? Desperate? Suspicious?]

FACTIONS:
[Optional: 1-3 sentences describing any groups, organizations, or power structures in this world. Who controls what? What tensions exist? Leave blank if not applicable.]

DM_SECRET:
[The hidden truth that makes this scenario interesting. What don't the players know? What twist awaits? What's really going on? This is ONLY for the DM/game master - players never see this. Be specific and dramatic.]

Be evocative but concise."""

    system = "You are a creative writing assistant. Write immersive scene descriptions for roleplay."

    response = _call_ollama_sync(model, prompt, system)

    # Parse the structured response
    description = ""
    genre = ""
    mood = ""
    factions = ""
    dm_secret = ""

    # Helper to extract content after a header on the same line
    def extract_after_header(line, headers):
        line_upper = line.upper()
        for h in headers:
            # Handle formats like "**DESCRIPTION:** content" or "DESCRIPTION: content"
            for pattern in [f'**{h}:**', f'**{h}**:', f'{h}:', f'*{h}:*', f'#{h}:']:
                idx = line_upper.find(pattern.upper())
                if idx != -1:
                    return line[idx + len(pattern):].strip()
        return None

    current_section = None
    lines = response.split('\n')
    for line in lines:
        line_stripped = line.strip()
        # Strip markdown formatting (**, *, #, etc.) for header detection
        line_clean = line_stripped.lstrip('*#').rstrip('*#:').strip()

        # Check for section headers
        new_section = None
        inline_content = None

        if line_clean.upper() == 'DESCRIPTION' or 'DESCRIPTION:' in line_stripped.upper():
            new_section = 'description'
            inline_content = extract_after_header(line_stripped, ['DESCRIPTION'])
        elif line_clean.upper() == 'GENRE' or 'GENRE:' in line_stripped.upper():
            new_section = 'genre'
            inline_content = extract_after_header(line_stripped, ['GENRE'])
        elif line_clean.upper() == 'MOOD' or 'MOOD:' in line_stripped.upper():
            new_section = 'mood'
            inline_content = extract_after_header(line_stripped, ['MOOD'])
        elif line_clean.upper() == 'FACTIONS' or 'FACTIONS:' in line_stripped.upper():
            new_section = 'factions'
            inline_content = extract_after_header(line_stripped, ['FACTIONS'])
        elif line_clean.upper() in ['DM_SECRET', 'DM SECRET'] or 'DM_SECRET:' in line_stripped.upper() or 'DM SECRET:' in line_stripped.upper():
            new_section = 'dm_secret'
            inline_content = extract_after_header(line_stripped, ['DM_SECRET', 'DM SECRET'])

        if new_section:
            current_section = new_section
            # If there's content on the same line as the header, add it
            if inline_content:
                if current_section == 'description':
                    description += inline_content + '\n'
                elif current_section == 'genre':
                    genre += inline_content + ' '
                elif current_section == 'mood':
                    mood += inline_content + ' '
                elif current_section == 'factions':
                    factions += inline_content + '\n'
                elif current_section == 'dm_secret':
                    dm_secret += inline_content + '\n'
            continue

        # Add content to current section
        if current_section == 'description':
            description += line + '\n'
        elif current_section == 'genre':
            genre += line_stripped + ' '
        elif current_section == 'mood':
            mood += line_stripped + ' '
        elif current_section == 'factions':
            factions += line + '\n'
        elif current_section == 'dm_secret':
            dm_secret += line + '\n'

    # Clean up
    description = description.strip()
    genre = genre.strip().lower().split()[0] if genre.strip() else ""  # Just first word
    genre = genre.rstrip(',')  # Remove trailing comma
    mood = mood.strip()
    factions = factions.strip()
    dm_secret = dm_secret.strip()

    # CRITICAL: Strip any DM_SECRET content that leaked into visible fields
    # This can happen if the LLM puts multiple fields on one line
    import re
    secret_pattern = re.compile(r'DM[_\s]?SECRET[:\s].*', re.IGNORECASE | re.DOTALL)
    description = secret_pattern.sub('', description).strip()
    factions = secret_pattern.sub('', factions).strip()
    mood = secret_pattern.sub('', mood).strip()

    # If factions is just "N/A" or similar, clear it
    if factions.upper() in ['N/A', 'NA', 'NONE', 'N\\A', '-']:
        factions = ''

    # Fallback if parsing failed
    if not description:
        # Also strip secrets from fallback
        description = secret_pattern.sub('', response).strip()

    return jsonify({
        'title': title,
        'description': description,
        'genre': genre,
        'mood': mood,
        'factions': factions,
        'dm_secret': dm_secret  # Hidden from players, only for DM context
    })


@app.route('/storybuilder/characters', methods=['POST'])
def generate_characters():
    """Generate character pool for a scenario."""
    data = request.json or {}
    model = data.get('model', settings.storybuilder_model)
    scenario_title = data.get('scenario_title', '')
    scenario_description = data.get('scenario_description', '')
    count = min(data.get('count', 5), 8)  # Max 8 characters
    locked_names = data.get('locked_names', [])  # Names to avoid regenerating

    # Request one extra character as buffer (parsing sometimes loses one)
    request_count = count + 1

    # Load diverse names from names.json to avoid repetition
    import random
    names_file = Path(__file__).parent / 'names.json'
    suggested_names = []
    try:
        if names_file.exists():
            import json
            with open(names_file, 'r') as f:
                names_data = json.load(f)
            # Pick random names from each category, avoiding locked names
            all_names = (
                names_data.get('masculine', []) +
                names_data.get('feminine', []) +
                names_data.get('neutral', []) +
                names_data.get('nicknames', [])
            )
            available_names = [n for n in all_names if n not in locked_names]
            suggested_names = random.sample(available_names, min(request_count * 2, len(available_names)))
    except Exception as e:
        print(f"[StoryBuilder] Could not load names.json: {e}")

    locked_hint = ""
    if locked_names:
        locked_hint = f"\n\nIMPORTANT: Do NOT generate characters with these names (they are already selected): {', '.join(locked_names)}"

    # Add name suggestions to encourage variety
    names_hint = ""
    if suggested_names:
        names_hint = f"\n\nSUGGESTED NAMES (use these or similar - avoid common defaults like 'Silas', 'Marcus', 'Elena'): {', '.join(suggested_names[:12])}"

    prompt = f"""For this roleplay scenario:
Title: {scenario_title}
Scene: {scenario_description}

Generate EXACTLY {request_count} unique characters who would fit this scenario. You MUST generate all {request_count} characters - no more, no less.{locked_hint}{names_hint}

For each character, provide:
1. Name (just first name, or a memorable title/alias)
2. Gender (male, female, or ambiguous)
3. A visual description (age, appearance, style - for image generation)
4. Personality in 2-3 sentences (their demeanor, motivations, quirks)
5. One thing they LIKE and one thing they DISLIKE
6. A SECRET - something hidden about them that relates to their backstory
7. Their WOUND - a past trauma or loss that still affects them
8. Their WANT - what they're truly seeking (may be hidden even from themselves)
9. Their FEAR - what terrifies them most
10. Their SKILL - a useful expertise or talent they have
11. HONESTY - rate 1-10 how honest they are (1=compulsive liar, 10=cannot lie)

Format your response EXACTLY like this for each character:
---CHARACTER---
Name: [name]
Gender: [male/female/ambiguous]
Appearance: [physical description for image generation]
Personality: [2-3 sentences about who they are]
Likes: [one thing they enjoy]
Dislikes: [one thing they hate]
Secret: [something hidden that could change everything]
Wound: [past trauma that shaped them]
Want: [what they truly seek]
Fear: [their deepest fear]
Skill: [their expertise - be specific, like "electrician" or "field medic"]
Honesty: [1-10]

Make characters diverse in gender and background. Give them REAL psychological depth - wounds that make sense with their personality, wants that drive them, fears that could be exploited. Make some characters less honest than others - a 3 might lie casually, a 7 mostly tells truth.

IMPORTANT: If generating 3+ characters, at least TWO should have a pre-existing relationship (neighbors, siblings, coworkers, old rivals, parent/child, exes, etc.). This creates instant built-in tension and history. Note this relationship in their personality descriptions so both characters know about it."""

    system = "You are a character designer for roleplay scenarios. Create memorable, well-rounded characters."

    response = _call_ollama_sync(model, prompt, system)

    # Parse characters
    characters = []
    parts = response.split('---CHARACTER---')

    for part in parts[1:]:  # Skip first empty part
        lines = part.strip().split('\n')
        char_id = str(uuid.uuid4())[:8]
        char = {
            'id': char_id,
            'name': '',
            'gender': 'ambiguous',  # Default to ambiguous if not specified
            'appearance': '',
            'personality': '',
            'likes': '',
            'dislikes': '',
            'locked': False,
            'avatar_url': None  # Will be set if we generate images
        }
        # Hidden traits (not sent to client)
        hidden = {
            'secret': '',
            'wound': '',
            'want': '',
            'fear': '',
            'skill': '',
            'honesty': 5,  # Default middle honesty
        }

        for line in lines:
            line = line.strip()
            if line.startswith('Name:'):
                char['name'] = line[5:].strip()
            elif line.startswith('Gender:'):
                gender = line[7:].strip().lower()
                if gender in ['male', 'female', 'ambiguous']:
                    char['gender'] = gender
                elif 'male' in gender and 'female' not in gender:
                    char['gender'] = 'male'
                elif 'female' in gender:
                    char['gender'] = 'female'
                else:
                    char['gender'] = 'ambiguous'
            elif line.startswith('Appearance:'):
                char['appearance'] = line[11:].strip()
            elif line.startswith('Personality:'):
                char['personality'] = line[12:].strip()
            elif line.startswith('Likes:'):
                char['likes'] = line[6:].strip()
            elif line.startswith('Dislikes:'):
                char['dislikes'] = line[9:].strip()
            elif line.startswith('Secret:'):
                hidden['secret'] = line[7:].strip()
            elif line.startswith('Wound:'):
                hidden['wound'] = line[6:].strip()
            elif line.startswith('Want:'):
                hidden['want'] = line[5:].strip()
            elif line.startswith('Fear:'):
                hidden['fear'] = line[5:].strip()
            elif line.startswith('Skill:'):
                hidden['skill'] = line[6:].strip()
            elif line.startswith('Honesty:'):
                try:
                    hidden['honesty'] = int(line[8:].strip().split()[0])
                except:
                    hidden['honesty'] = 5

        if char['name']:
            # Store hidden traits server-side (not sent to client)
            _character_secrets[char_id] = hidden
            characters.append(char)

    return jsonify({'characters': characters[:count]})


@app.route('/storybuilder/backstories', methods=['POST'])
def generate_backstories():
    """Generate backstory options for the player character.

    Backstories determine skills and knowledge - they're generated, not player-written,
    to prevent gaming the system. The player picks from options.
    """
    data = request.json or {}
    model = data.get('model', settings.storybuilder_model)
    scenario_title = data.get('scenario_title', '')
    scenario_description = data.get('scenario_description', '')
    genre = data.get('genre', '')
    character_name = data.get('character_name', 'Traveler')
    character_gender = data.get('character_gender', '')
    count = min(data.get('count', 4), 6)

    # Build gender instruction if specified
    gender_instruction = ""
    if character_gender:
        gender_instruction = f"\nIMPORTANT: {character_name} is {character_gender}. Use appropriate pronouns and gendered terms throughout."

    # Request extra as buffer since parsing sometimes loses some
    request_count = count + 2

    # Variety categories to ensure different archetypes
    import random
    variety_hints = [
        "a blue-collar worker (mechanic, plumber, electrician, construction)",
        "a white-collar professional (accountant, lawyer, teacher, manager)",
        "a medical/caretaker background (nurse, EMT, veterinarian, caregiver)",
        "someone from the margins (homeless, ex-con, dropout, drifter)",
        "a technical specialist (engineer, IT, scientist, researcher)",
        "a service industry survivor (cook, bartender, retail, driver)",
        "a creative or artistic type (musician, writer, artist, actor)",
        "a student or young person (college kid, intern, gap year)",
        "someone physical/athletic (trainer, athlete, dancer, laborer)",
        "a rural/outdoors background (farmer, hunter, park ranger, fisherman)",
    ]
    random.shuffle(variety_hints)
    variety_suggestions = "\n".join([f"- {hint}" for hint in variety_hints[:request_count]])

    prompt = f"""For this roleplay scenario:
Title: {scenario_title}
Scene: {scenario_description}
Genre: {genre}

Generate exactly {request_count} DIFFERENT backstory options for a player character named "{character_name}".{gender_instruction}

IMPORTANT: Each backstory MUST be from a DIFFERENT category. Use these as inspiration:
{variety_suggestions}

Each backstory should:
1. Fit the world/genre naturally
2. Suggest specific skills and knowledge (be concrete - "knows how to hotwire cars" not "street smart")
3. Include at least one LIMITATION or weakness
4. Feel like a real person, not an optimized build

Format EXACTLY like this for each backstory (use this exact delimiter):
---BACKSTORY---
Title: [short evocative title, 2-4 words]
Description: [2-3 sentences about who they were before the story began]
Skills: [comma-separated list of specific skills AND limitations: e.g., "field medicine, lockpicking, speaks Spanish, can't swim, afraid of heights"]

You MUST generate all {request_count} backstories. Each one MUST start with ---BACKSTORY--- on its own line."""

    system = "You are creating character backstories for an RPG. Make them feel like real people with real histories, not optimized player builds."

    response = _call_ollama_sync(model, prompt, system, retries=2, timeout=90.0)

    if not response:
        return jsonify({
            'backstories': [],
            'error': 'Failed to generate backstories - model did not respond. Try again or check if Ollama is running.'
        })

    # Parse backstories - try multiple delimiters
    backstories = []

    # Try different delimiter patterns the model might use
    delimiters = ['---BACKSTORY---', '--- BACKSTORY ---', '**BACKSTORY**', '## BACKSTORY', 'BACKSTORY:', '---']
    parts = None
    for delim in delimiters:
        if delim in response:
            parts = response.split(delim)
            if len(parts) > 1:
                break

    if not parts or len(parts) <= 1:
        # Last resort: try to find Title: patterns and split on those
        import re
        parts = re.split(r'\n(?=Title:|(?:\*\*)?Title(?:\*\*)?:)', response)

    for part in parts[1:] if parts else []:
        lines = part.strip().split('\n')
        backstory = {
            'title': '',
            'description': '',
            'skills': ''
        }

        for line in lines:
            line = line.strip()
            # Handle various markdown formatting
            line_lower = line.lower()
            if line_lower.startswith('title:') or '**title' in line_lower:
                backstory['title'] = line.split(':', 1)[1].strip().strip('*').strip()
            elif line_lower.startswith('description:') or '**description' in line_lower:
                backstory['description'] = line.split(':', 1)[1].strip().strip('*').strip()
            elif line_lower.startswith('skills:') or '**skills' in line_lower:
                backstory['skills'] = line.split(':', 1)[1].strip().strip('*').strip()

        if backstory['title'] and backstory['description']:
            backstories.append(backstory)

    if not backstories:
        # Generate fallback backstories for this genre
        fallback_backstories = [
            {'title': 'Lucky Survivor', 'description': f'No special skills, just managed to stay alive through luck and quick thinking. {character_name} was in the wrong place at the wrong time, but somehow made it.', 'skills': 'quick learner, resourceful, panics under pressure, no combat training'},
            {'title': 'Former Office Worker', 'description': f'{character_name} spent years behind a desk before everything changed. Now those spreadsheet skills seem less useful.', 'skills': 'organization, basic first aid from workplace training, physically unfit, knows local area'},
            {'title': 'Trade Worker', 'description': f'Years of physical labor gave {character_name} practical skills and a strong back. Blue collar and proud of it.', 'skills': 'basic repairs, physical endurance, tool use, stubborn, not book-smart'},
            {'title': 'Caregiver', 'description': f'{character_name} spent years looking after others - elderly parents, sick relatives. Patient and observant.', 'skills': 'basic medical care, patience, reading people, physically average, conflict-averse'},
        ]
        import random
        random.shuffle(fallback_backstories)
        return jsonify({
            'backstories': fallback_backstories[:count],
            'note': 'Used fallback backstories - model parsing failed'
        })

    # Shuffle to add variety in presentation order, then return requested count
    import random
    random.shuffle(backstories)
    return jsonify({'backstories': backstories[:count]})


@app.route('/storybuilder/character-avatar', methods=['POST'])
def generate_character_avatar():
    """Generate a quick avatar for a StoryBuilder character."""
    data = request.json or {}
    appearance = data.get('appearance', '')
    name = data.get('name', '')

    if not appearance:
        return jsonify({'error': 'No appearance provided'}), 400

    # Create a simple avatar job
    job_id = create_job('storybuilder_avatar')

    def generate_avatar():
        try:
            from image_gen import ImageGenerator
            generator = ImageGenerator()

            # Simple portrait prompt
            prompt = f"portrait of {appearance}, looking at viewer, simple background"

            update_job(job_id, 'generating_image')
            images = generator.generate_avatar(
                prompt=prompt,
                subject_name=name,
                gen_type="portrait"
            )

            if images:
                # Return as base64 for quick display
                import base64
                with open(images[0], 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode()
                update_job(job_id, 'completed', result={
                    'image_path': str(images[0]),
                    'image_data': f"data:image/png;base64,{img_data}"
                })
            else:
                update_job(job_id, 'failed', error='No image generated')
        except Exception as e:
            import traceback
            traceback.print_exc()
            update_job(job_id, 'failed', error=str(e))

    _executor.submit(generate_avatar)

    return jsonify({
        'job_id': job_id,
        'status': 'pending'
    })


# Shared prompt section for opening scenes - companion behavior and spatial guidance
# Used by both storybuilder_create_room and generate_inciting_incident
OPENING_SCENE_RULES = """=== SPATIAL ANCHORING ===
Ground the scene in real geometry, woven naturally into the prose:
- Establish where the player physically IS (inside the cabin, crouched by the eastern hatch, on the deck)
- Position companions relative to the player (beside you, across the room, a few feet to your left)
- When mentioning landmarks, commit to where they are (the sawmill visible through the gap in the hull, the treeline beyond it to the north)
- Sightlines matter: if someone looks at something, make clear they CAN see it from where they are

Don't list coordinates - paint a picture where the reader knows the shape of the space.

=== COMPANION REACTIONS ===
When tension arrives, companions react through their COMPETENCE, not helplessness:
- An artist's eyes track every detail, maybe sketching faster to process what they see
- A veteran shifts position silently, hand moving to weapon
- A medic checks exits, calculates distances
- Fear looks different on everyone: some focus harder, some go quiet, some act immediately

Companions CONTRIBUTE to handling the situation - they notice things, reposition, prepare. They don't just signal danger to the player.

BANNED: The "anime fear" formula. Do NOT combine: freezing mid-action + pointing with trembling finger + grabbing the player's arm + dropping what they're holding. This cliche is not allowed.

- Match reaction intensity to actual threat level. A distant light warrants curiosity, not terror. A close threat warrants real fear.
- Either character might spot danger first."""


def roll_opening_matrix(genre: str, is_zombie_genre: bool = False, shelter_type: str = None) -> dict:
    """Roll all opening matrix axes for scene variety.
    Used by both storybuilder_create_room and generate_inciting_incident.

    shelter_type: Optional constraint from StoryBuilder. Values:
        - 'wandering': No base, on the move → homeless only
        - 'temporary': Tents, vehicles → temporary hideout only
        - 'shanty': Makeshift shelter → small group shelter
        - 'decent': Real building → small group shelter or established
        - 'reinforced': Fortified → established camp
        - 'fortress': Maximum security → established camp
    """
    import random
    matrix = {}

    # 1. Tension Type (weighted for variety)
    tension_weights = {
        'IMMEDIATE': 16,      # It's happening NOW
        'DISCOVERY': 20,      # You find something
        'ARRIVAL': 12,        # Someone bursts in
        'DEADLINE': 12,       # Clock is ticking
        'PERSONAL': 8,        # Connected to character
        'OBSERVATION': 10,    # "You notice" - reduced further
        'WINDFALL': 10,       # You just scored BIG - good luck for once
        'RESPITE': 12,        # Peaceful moment, internal tension only
    }
    tension_choices = []
    for t, w in tension_weights.items():
        tension_choices.extend([t] * w)
    matrix['tension_type'] = random.choice(tension_choices)

    # 2. Time of Day
    if genre and 'horror' in genre.lower():
        times = ["dawn", "dusk", "night", "deep night (3am)", "the grey hour before sunrise"]
    elif genre and 'romance' in genre.lower():
        times = ["golden hour", "late afternoon", "morning", "dawn"]
    else:
        times = ["dawn", "morning", "midday", "afternoon", "dusk", "night", "deep night"]
    matrix['time_of_day'] = random.choice(times)

    # 3. Weather (Oregon-weighted - lots of rain/fog)
    weather_weights = {
        'clear': 15,
        'overcast': 20,
        'fog / mist': 20,       # Oregon special
        'light rain': 15,
        'heavy rain / storm': 10,
        'cold snap / frost': 8,
        'wind (coastal gales)': 12,
    }
    weather_choices = []
    for w, weight in weather_weights.items():
        weather_choices.extend([w] * weight)
    matrix['weather'] = random.choice(weather_choices)

    # 4. Threat Type (66% zombie if zombie genre)
    if is_zombie_genre:
        threat_weights = {
            'zombie': 66,
            'human - hostile': 10,
            'human - desperate/neutral': 8,
            'animal (normal)': 4,
            'animal (infected)': 4,
            'environmental (tide, collapse, fire)': 4,
            'none': 4,
        }
    else:
        threat_weights = {
            'human - hostile': 25,
            'human - desperate/neutral': 20,
            'animal (normal)': 15,
            'environmental': 20,
            'none': 20,
        }
    threat_choices = []
    for t, w in threat_weights.items():
        threat_choices.extend([t] * w)
    matrix['threat_type'] = random.choice(threat_choices)

    # 5-7. Threat details (only if threat exists)
    if matrix['threat_type'] != 'none':
        # Distance
        matrix['threat_distance'] = random.choice(['close', 'medium', 'far'])
        # Awareness
        matrix['threat_awareness'] = random.choice([
            "hasn't noticed you",
            "has seen you",
            "hunting you",
            "focused on something else"
        ])
        # Size (hordes more rare)
        size_weights = {'one': 50, 'small group (3-5)': 35, 'horde / mob': 15}
        size_choices = []
        for s, w in size_weights.items():
            size_choices.extend([s] * w)
        matrix['threat_size'] = random.choice(size_choices)
    else:
        matrix['threat_distance'] = None
        matrix['threat_awareness'] = None
        matrix['threat_size'] = None

    # 8. Companion State - VARIED, not always fearful
    companion_states = [
        # Competent/active states (higher weight)
        'alert, scanning the area',
        'alert, scanning the area',  # x2
        'noticed something first - YOU are the one who missed it',
        'noticed something first - YOU are the one who missed it',  # x2
        'determined / ready to act',
        'determined / ready to act',  # x2
        'calm and focused, assessing the situation',
        'protective of YOU - hand out to keep you back',
        # Neutral/processing states
        'distracted / occupied with a task',
        'lost in thought, processing something',
        'quietly cataloging details (observant)',
        'exhausted but pushing through',
        # Emotional but varied
        'annoyed / frustrated with the situation',
        'darkly amused by something',
        'numb / thousand-yard stare',
        'curious about something unusual',
        # Fear states (lower weight - not default)
        'frozen / afraid',
        'injured / struggling',
    ]
    matrix['companion_state'] = random.choice(companion_states)

    # 9. Housing Situation - constrained by shelter_type if provided
    if shelter_type:
        # Map shelter_type to valid housing options
        shelter_to_housing = {
            'wandering': ['homeless / between places (recent loss)'],
            'temporary': ['temporary hideout (not home)'],
            'shanty': ['small group shelter (just your people)', 'temporary hideout (not home)'],
            'decent': ['small group shelter (just your people)'],
            'reinforced': ['small group shelter (just your people)', 'established camp/village (safety nearby)'],
            'fortress': ['established camp/village (safety nearby)'],
        }
        valid_housing = shelter_to_housing.get(shelter_type)
        if valid_housing:
            matrix['housing'] = random.choice(valid_housing)

    # If not constrained by shelter_type (or invalid type), use weighted random
    if not matrix.get('housing'):
        housing_weights = {
            'established camp/village (safety nearby)': 25,
            'small group shelter (just your people)': 30,
            'homeless / between places (recent loss)': 25,
            'temporary hideout (not home)': 20,
        }
        housing_choices = []
        for h, w in housing_weights.items():
            housing_choices.extend([h] * w)
        matrix['housing'] = random.choice(housing_choices)

    # 10. Resource State
    resource_weights = {
        'well-supplied': 20,
        'getting low on supplies': 30,
        'critical (last of something)': 20,
        'just found something good': 15,
        'just lost supplies': 15,
    }
    resource_choices = []
    for r, w in resource_weights.items():
        resource_choices.extend([r] * w)
    matrix['resources'] = random.choice(resource_choices)

    # 11. Scene Structure - HOW the scene is framed, not just what's in it
    structure_weights = {
        'stillness_interrupted': 20,   # Classic: you're stationary, tension arrives
        'mid_conversation': 20,        # Drop in during dialogue
        'mid_action': 20,              # Already moving/doing when tension hits
        'discovery': 15,               # The act of finding/uncovering something
        'aftermath': 10,               # Something just happened, dealing with it
        'quiet_dread': 15,             # No immediate threat, just the weight
    }
    structure_choices = []
    for s, w in structure_weights.items():
        structure_choices.extend([s] * w)
    matrix['scene_structure'] = random.choice(structure_choices)

    # === LEGENDARY ROLLS (1/100) ===
    legendary_roll = random.randint(1, 100)
    if legendary_roll == 1:
        matrix['legendary'] = 'separated_character'
    elif legendary_roll == 2:
        matrix['legendary'] = 'backstory_connection'
    elif legendary_roll == 3:
        matrix['legendary'] = 'perfect_find'
    elif legendary_roll == 4:
        matrix['legendary'] = 'worst_case'
    else:
        matrix['legendary'] = None

    return matrix


# Tension type descriptions for prompts
TENSION_TYPE_DESCRIPTIONS = {
    'IMMEDIATE': "Something is happening RIGHT NOW at your feet or behind you. Urgent action required. NOT a distant figure - something IN YOUR SPACE demanding response.",
    'DISCOVERY': "You find something that changes the situation - a body, a map, a radio, supplies, a clue. The tension is WHAT YOU FOUND, not a zombie watching from afar.",
    'ARRIVAL': "Someone or something is approaching or has just arrived with news or need. They're HERE, not in the distance.",
    'DEADLINE': "A clock is ticking - a boat leaving, tide rising, patrol changing, window closing. Time pressure, not distant threats.",
    'PERSONAL': "This connects to the player's deepest want, fear, or backstory. Make it personal, emotional, character-driven.",
    'OBSERVATION': "You notice something in the environment. VARIETY: smoke columns, fresh tracks, a distant light, a message on a wall, sudden silence, a reflection in the hills - NOT always 'single zombie in distance.' And characters react PROPORTIONALLY - a lone figure 300 yards away doesn't warrant panic.",
    'WINDFALL': "You just scored BIG. Found a cache, stumbled on abandoned gear, got incredibly lucky. The tension: what now? Keep it? Share it? Did someone see? Can you even carry it all?",
    'RESPITE': "A moment of peace. No immediate threat, no crisis. Just existing, recovering, being human. The tension is internal - exhaustion, grief, hope, connection. A campfire moment. Let the world breathe.",
}

# Scene structure descriptions - HOW the scene is framed
SCENE_STRUCTURE_DESCRIPTIONS = {
    'stillness_interrupted': "Open on stillness - characters settled, occupied. Tension ARRIVES to disrupt.",
    'mid_conversation': "Drop in mid-dialogue. Characters already talking when we join. Tension weaves into or interrupts the exchange.",
    'mid_action': "Characters already MOVING - walking somewhere, carrying something, in transit. Action first, then tension.",
    'discovery': "The scene IS the finding. Open on the moment of uncovering - a door opening, a tarp pulled back, a radio crackling to life.",
    'aftermath': "Something just happened. Open on the consequences - catching breath, assessing damage, processing what occurred.",
    'quiet_dread': "No immediate threat. Just the weight of the world pressing down. Survival fatigue, small moments, the spaces between crises.",
}


@app.route('/storybuilder/create-room', methods=['POST'])
def storybuilder_create_room():
    """Create a room from StoryBuilder with generated characters as partners."""
    try:
        return _storybuilder_create_room_impl()
    except Exception as e:
        import traceback
        error_msg = f"Room creation failed: {str(e)}"
        print(f"\033[91m[StoryBuilder] ERROR: {error_msg}\033[0m")
        print(f"\033[91m{traceback.format_exc()}\033[0m")
        return jsonify({'error': error_msg, 'traceback': traceback.format_exc()}), 500


def _storybuilder_create_room_impl():
    """Internal implementation of room creation."""
    data = request.json or {}

    scenario_title = data.get('scenario_title', 'StoryBuilder Room')
    scenario_description = data.get('scenario_description', '')
    timeline = data.get('timeline', '')  # When in the story (first week, two years in, etc.)
    genre = data.get('genre', '')  # Emotional texture of the world
    factions = data.get('factions', '')  # Who controls what
    dm_secret = data.get('dm_secret', '')  # Hidden twist - DM only, players never see
    characters = data.get('characters', [])
    include_existing = data.get('include_existing', [])  # Existing partner IDs to include

    # Model for this room's DM operations - from StoryBuilder selection
    room_model = data.get('model', settings.storybuilder_model)

    # Genre rules - what's possible in this world
    genre_rules = data.get('genre_rules', None)
    # DM mode - "full", "light", or "none"
    dm_mode = data.get('dm_mode', 'full')
    # Hardcore mode - permanent death
    hardcore_mode = data.get('hardcore_mode', False)
    # Character relationships and starting positions
    character_relationships = data.get('character_relationships', [])
    # Background thread context - used for separated characters
    threat_type = data.get('threat_type', '')
    world_state = data.get('world_state', '')

    # Zombie type - affects how dangerous encounters are
    zombie_type = data.get('zombie_type', None)
    if zombie_type == 'mystery':
        # Roll the mystery - player won't know until they encounter one
        import random
        zombie_type = random.choice(['shamblers', 'runners', 'nightmares'])
        print(f"[StoryBuilder] Mystery zombie type resolved to: {zombie_type} (player doesn't know!)")

    # Population density - infer from scenario description for realistic zombie frequency
    # This determines how many zombies are in the area and encounter frequency
    population_density_info = infer_population_density(scenario_description)
    population_density = population_density_info["density_level"]
    initial_zombie_count = population_density_info["estimated_zombies"]
    print(f"[StoryBuilder] Population density: {population_density} (~{initial_zombie_count} zombies)")

    # Roll hidden zombie rules - players discover these through play
    # Check for advanced overrides from storybuilder (if user expanded Advanced Options)
    zombie_rules_overrides = data.get('zombie_rules', {})
    zombie_rules = roll_zombie_rules(zombie_type=zombie_type, override_rules=zombie_rules_overrides)
    print(f"[StoryBuilder] Zombie rules rolled (hidden from player):")
    print(f"  Type: {zombie_rules.get('type')} | Headshots kill: {zombie_rules.get('headshots_kill')}")
    print(f"  Sound attracts: {zombie_rules.get('sound_attracts')} | Decay: {zombie_rules.get('decay_enabled')}")
    print(f"  Hive behavior: {zombie_rules.get('hive_behavior')} | Freshness: {zombie_rules.get('freshness')}")
    print(f"  Night hunters: {zombie_rules.get('night_hunters')} | Infection: {zombie_rules.get('infection_rate')}")

    # Shelter type - affects story tone and objectives
    shelter_type = data.get('shelter_type', None)
    if shelter_type:
        print(f"[StoryBuilder] Shelter type: {shelter_type}")

    if not characters and not include_existing:
        return jsonify({'error': 'No characters selected'}), 400

    # Create partners for each new character
    new_partner_ids = []
    char_id_to_partner_id = {}  # Map original char ID to partner ID for relationship wiring
    for char in characters:
        partner_id = str(uuid.uuid4())[:8]
        char_id = char.get('id', '')
        char_id_to_partner_id[char_id] = partner_id  # Track for separation system

        # Retrieve hidden traits if we have them stored for this character
        hidden = _character_secrets.pop(char_id, {})
        if isinstance(hidden, str):
            # Backwards compat: old format was just the secret string
            hidden = {'secret': hidden}

        partner = Partner(
            id=partner_id,
            name=char.get('name', 'Unknown'),
            character_description=char.get('personality', ''),
            physical_description=char.get('appearance', ''),
            gender=char.get('gender', 'ambiguous'),
            provider='ollama',  # Default to Ollama for generated characters
            model=settings.default_ollama_model,
            avatar='🎭',
            memory_mode='local',  # Fresh for each room
            # Hidden traits - known only to character (and DM)
            secret=hidden.get('secret'),
            wound=hidden.get('wound'),
            want=hidden.get('want'),
            fear=hidden.get('fear'),
            skill=hidden.get('skill'),
            honesty=hidden.get('honesty', 5),
        )
        data_store.add_partner(partner)
        new_partner_ids.append(partner_id)

    # Combine new and existing partner IDs
    all_partner_ids = new_partner_ids + include_existing

    # Get player character name if provided
    player_character = data.get('player_character', {})
    player_char_name = player_character.get('name') if player_character else None

    # Calculate which characters start with the player (using proper partner ID mapping)
    present_partner_ids = []
    print(f"[StoryBuilder] character_relationships: {character_relationships}")
    print(f"[StoryBuilder] char_id_to_partner_id: {char_id_to_partner_id}")
    for rel in character_relationships:
        if rel.get('starts_with_player', True):
            char_id = rel.get('character_id', '')
            # Map to actual partner ID
            partner_id = char_id_to_partner_id.get(char_id)
            if not partner_id and char_id in include_existing:
                partner_id = char_id  # Existing partner - char_id is already the partner_id
            print(f"[StoryBuilder] Mapping char_id={char_id} -> partner_id={partner_id}, starts_with_player=True")
            if partner_id:
                present_partner_ids.append(partner_id)
    print(f"[StoryBuilder] Final present_partner_ids: {present_partner_ids}")

    # Update character_relationships to use partner IDs instead of wizard IDs
    # This ensures the frontend can match relationships to actual partners
    for rel in character_relationships:
        old_char_id = rel.get('character_id', '')
        new_partner_id = char_id_to_partner_id.get(old_char_id)
        if new_partner_id:
            rel['character_id'] = new_partner_id
        elif old_char_id in include_existing:
            pass  # Already a partner ID
        # Also update relationship target_ids
        for r in rel.get('relationships', []):
            old_target = r.get('target_id', '')
            new_target = char_id_to_partner_id.get(old_target)
            if new_target:
                r['target_id'] = new_target

    # Extract player character details
    player_gender = player_character.get('gender', '') if player_character else ''
    player_alignment = player_character.get('alignment', '') if player_character else ''
    player_role = player_character.get('role', '') if player_character else ''
    player_backstory = player_character.get('backstory', {}) if player_character else {}

    # Fallback: if no one is explicitly marked as present, everyone starts together
    if not present_partner_ids and all_partner_ids:
        print(f"[StoryBuilder] WARNING: No present_partner_ids calculated, defaulting to all characters")
        present_partner_ids = list(all_partner_ids)

    # Create the room with scenario, genre, factions, and rules
    room = data_store.create_custom_room(
        name=scenario_title,
        partner_ids=all_partner_ids,
        scenario=scenario_description,
        timeline=timeline,
        genre=genre,
        factions=factions,
        genre_rules=genre_rules,
        dm_mode=dm_mode,
        dm_secret=dm_secret,  # Hidden twist for DM context only
        room_model=room_model,  # Model selected in StoryBuilder for this room's DM
        player_character_name=player_char_name,
        player_gender=player_gender,
        player_alignment=player_alignment,
        player_role=player_role,
        player_backstory=player_backstory,
        hardcore_mode=hardcore_mode,
        character_relationships=character_relationships,
        present_character_ids=present_partner_ids,
        threat_type=threat_type,  # For background thread context
        world_state=world_state,  # For background thread context
        zombie_type=zombie_type,  # Zombie behavior: shamblers, runners, or nightmares
        population_density=population_density,  # For realistic encounter frequency
        initial_zombie_count=initial_zombie_count,  # Track zombie population
        zombie_rules=zombie_rules,  # Hidden rules - players discover through play
        shelter_type=shelter_type  # Starting shelter situation
    )

    # Generate scene layout for spatial tracking
    try:
        all_partners = data_store.get_partners()
        char_names = [p.name for p in all_partners if p.id in all_partner_ids]
        player_name = player_char_name or settings.user_name

        # Build character positions from relationships
        position_info = ""
        if character_relationships:
            positions = []
            for rel in character_relationships:
                char_name = rel.get('character_name', 'Unknown')
                if rel.get('starts_with_player'):
                    positions.append(f"- {char_name} starts with {player_name}")
                elif rel.get('starting_location'):
                    positions.append(f"- {char_name} starts at: {rel.get('starting_location')}")
            if positions:
                position_info = "\n\nCharacter Starting Positions:\n" + "\n".join(positions)

        layout_prompt = f"""For this roleplay scene, create a brief spatial layout description.

Setting: {scenario_description}
Characters: {', '.join(char_names)} and {player_name}{position_info}

In 2-3 SHORT sentences, describe:
1. The immediate area (what kind of space is this?)
2. Key landmarks/features characters might reference (exits, cover, obstacles)
3. Where characters are positioned relative to each other (respect the starting positions above)

Keep it BRIEF and tactical - this is a reference for maintaining spatial consistency.
Example: "Abandoned mall jewelry counter. Main entrance 50ft south, escalator to 2nd floor northeast. Service corridor behind Sears to the east. Group clustered behind counter near east wall."

Just give the layout, no preamble."""

        scene_layout = _call_ollama_sync(room_model or settings.storybuilder_model, layout_prompt, "You are a concise spatial mapper. Describe layouts briefly.")
        if scene_layout and scene_layout.strip():
            room.scene_layout = scene_layout.strip()
            data_store.save()
            print(f"[StoryBuilder] Generated scene layout for {room.name}")
    except Exception as e:
        print(f"[StoryBuilder] Warning: Could not generate scene layout: {e}")

    # Generate world map for the new room (this gives the world geography)
    world_map = None
    if genre:  # Only generate if we have a genre hint
        try:
            cartographer = get_cartographer()

            async def _generate_map():
                # Use room's model for all generation
                async def ollama_gen(prompt, model=None):
                    use_model = model or room_model or settings.storybuilder_model
                    return await provider_manager.generate_ollama(prompt, use_model)

                return await cartographer.generate_initial_world(
                    world_id=room.id,
                    genre=genre,
                    setting_description=scenario_description,
                    ollama_generate_func=ollama_gen,
                    num_locations=5,  # Start with a handful of locations
                )

            world_map = asyncio.run(_generate_map())
            print(f"[StoryBuilder] Generated world map for {room.name} with {len(world_map.locations)} locations (using {room_model})")
        except Exception as e:
            print(f"[StoryBuilder] Warning: Could not generate world map: {e}")
            # Don't fail room creation if map generation fails

    # Set up player character with alignment, role, and backstory
    player_character = data.get('player_character', {})
    if player_character:
        try:
            from autopilot import get_autopilot_tracker, Alignment, Role, Drive

            tracker = get_autopilot_tracker()
            player_id = f"player_{room.id}"
            player_name = player_character.get('name', settings.user_name)

            # Create/get the player character
            pc = tracker.get_or_create(player_id, room.id, player_name)

            # Set alignment
            alignment_str = player_character.get('alignment', 'true_neutral')
            try:
                pc.alignment = Alignment(alignment_str)
            except ValueError:
                pc.alignment = Alignment.TRUE_NEUTRAL

            # Set role
            role_str = player_character.get('role', 'balanced')
            try:
                pc.role = Role(role_str)
            except ValueError:
                pc.role = Role.BALANCED

            # Store backstory info (DM can reference this)
            backstory = player_character.get('backstory', {})
            if backstory:
                pc.backstory_title = backstory.get('title', '')
                pc.backstory_description = backstory.get('description', '')

                # Parse skills into a list (they come as comma-separated string)
                skills_str = backstory.get('skills', '')
                if skills_str:
                    # Split by comma, strip whitespace, filter empty
                    pc.skills = [s.strip() for s in skills_str.split(',') if s.strip()]

            tracker._save()
            print(f"[StoryBuilder] Player character set up: {player_name}, {alignment_str}, {role_str}")

            # Create player inventory with starting items based on backstory
            try:
                from inventory import ItemCategory
                player_inv = inventory_tracker.get_or_create_inventory(
                    player_id, player_name, owner_type='player'
                )

                # Add starting items based on backstory skills
                skills_str = backstory.get('skills', '').lower() if backstory else ''
                backstory_title = backstory.get('title', '').lower() if backstory else ''

                # Genre-appropriate starting items
                starting_items = []
                genre = (room.genre or 'fantasy').lower()
                import random

                # === D20 WEAPON ROLL SYSTEM ===
                # Tiered weapon tables by quality
                # "Improvised" tier - but for STARTING gear, these should be things you'd actually carry
                minimal_weapons = {
                    'zombie': ['Pocket Knife', 'Utility Knife', 'Box Cutter', 'Heavy Flashlight', 'Walking Stick', 'Belt'],
                    'post-apocalyptic': ['Pocket Knife', 'Multitool', 'Box Cutter', 'Heavy Flashlight', 'Walking Stick', 'Chain'],
                    'fantasy': ['Walking Staff', 'Hunting Knife', 'Belt Knife', 'Wooden Club'],
                    'default': ['Pocket Knife', 'Utility Knife', 'Box Cutter', 'Heavy Flashlight', 'Walking Stick', 'Multitool'],
                }
                basic_melee = {
                    'zombie': ['Crowbar', 'Kitchen Knife', 'Baseball Bat', 'Hammer', 'Screwdriver', 'Tire Iron'],
                    'post-apocalyptic': ['Crowbar', 'Wrench', 'Claw Hammer', 'Hatchet', 'Pry Bar', 'Tire Iron'],
                    'fantasy': ['Dagger', 'Club', 'Hand Axe', 'Quarterstaff', 'Sickle'],
                    'default': ['Knife', 'Club', 'Hammer', 'Crowbar'],
                }
                good_melee = {
                    'zombie': ['Machete', 'Fire Axe', 'Hunting Knife', 'Katana', 'Spear', 'Combat Knife'],
                    'post-apocalyptic': ['Machete', 'Tomahawk', 'Bowie Knife', 'Sledgehammer', 'Bayonet'],
                    'fantasy': ['Longsword', 'Battle Axe', 'Mace', 'Spear', 'Short Sword', 'War Hammer'],
                    'default': ['Machete', 'Axe', 'Large Knife', 'Spear'],
                }
                firearms = {
                    'zombie': ['9mm Pistol', 'Revolver', '.22 Rifle', 'Hunting Rifle', 'Shotgun', 'Crossbow'],
                    'post-apocalyptic': ['Pistol', 'Revolver', 'Hunting Rifle', 'Shotgun', 'Compound Bow', 'Crossbow'],
                    'fantasy': ['Longbow', 'Crossbow', 'Shortbow', 'Hand Crossbow'],
                    'sci-fi': ['Laser Pistol', 'Plasma Rifle', 'Rail Pistol', 'Energy Bow'],
                    'default': ['Pistol', 'Rifle', 'Crossbow', 'Bow'],
                }
                # Thematic weapons for specific backstories
                # Multiple keywords can map to the same weapon set
                thematic_weapons = {
                    # Hunting/outdoors
                    'hunter': ['Hunting Rifle', 'Compound Bow', 'Crossbow'],
                    'hunting': ['Hunting Rifle', 'Compound Bow', 'Crossbow'],
                    'hunts': ['Hunting Rifle', 'Compound Bow', 'Crossbow'],
                    'trapper': ['Hunting Rifle', 'Compound Bow', 'Crossbow'],
                    # Military
                    'military': ['M9 Pistol', 'Combat Knife', 'M4 Carbine'],
                    'soldier': ['M9 Pistol', 'Combat Knife', 'Service Rifle'],
                    'army': ['M9 Pistol', 'Combat Knife', 'M4 Carbine'],
                    'marine': ['M9 Pistol', 'Combat Knife', 'M16'],
                    'veteran': ['M9 Pistol', 'Combat Knife', 'Service Rifle'],
                    'infantry': ['M9 Pistol', 'Combat Knife', 'M4 Carbine'],
                    'served': ['Service Pistol', 'Combat Knife', 'Service Rifle'],
                    # Law enforcement
                    'police': ['Service Pistol', 'Baton', 'Shotgun'],
                    'cop': ['Service Pistol', 'Baton', 'Taser'],
                    'officer': ['Service Pistol', 'Baton', 'Shotgun'],
                    'sheriff': ['Service Pistol', 'Shotgun', 'Revolver'],
                    'deputy': ['Service Pistol', 'Baton', 'Shotgun'],
                    'detective': ['Service Pistol', 'Revolver', 'Baton'],
                    # Combat/martial
                    'martial': ['Katana', 'Bo Staff', 'Nunchaku'],
                    'martial arts': ['Katana', 'Bo Staff', 'Nunchaku'],
                    'fighter': ['Combat Knife', 'Machete', 'Brass Knuckles'],
                    'boxer': ['Brass Knuckles', 'Wraps', 'Combat Knife'],
                    'brawler': ['Brass Knuckles', 'Crowbar', 'Baseball Bat'],
                    # Ranged specialists
                    'archer': ['Compound Bow', 'Recurve Bow', 'Crossbow'],
                    'archery': ['Compound Bow', 'Recurve Bow', 'Crossbow'],
                    'sniper': ['Hunting Rifle', 'Scoped Rifle', '.308 Rifle'],
                    'marksman': ['Hunting Rifle', 'Scoped Rifle', 'M4 Carbine'],
                    'sharpshooter': ['Hunting Rifle', 'Scoped Rifle', '.308 Rifle'],
                    # Trades that imply weapons
                    'mechanic': ['Wrench', 'Crowbar', 'Tire Iron'],
                    'carpenter': ['Hammer', 'Hatchet', 'Nail Gun'],
                    'butcher': ['Cleaver', 'Butcher Knife', 'Meat Hook'],
                    'chef': ['Chef Knife', 'Cleaver', 'Kitchen Knife'],
                    'cook': ['Kitchen Knife', 'Cleaver', 'Cast Iron Pan'],
                    # Medical backgrounds
                    'emt': ['Trauma Shears', 'Heavy Flashlight', 'Multi-tool'],
                    'paramedic': ['Trauma Shears', 'Heavy Flashlight', 'Utility Knife'],
                    'nurse': ['Scalpel', 'Heavy Flashlight', 'Trauma Shears'],
                    'doctor': ['Scalpel', 'Surgical Kit', 'Heavy Flashlight'],
                    'medic': ['Combat Knife', 'Trauma Shears', 'Utility Knife'],
                }
                # Ammo status descriptions
                ammo_status = {
                    1: 'almost empty - last few rounds',
                    2: 'running low',
                    3: 'half full',
                    4: 'half full',
                    5: 'mostly loaded',
                    6: 'full',
                }
                quiver_status = {
                    1: 'last arrow',
                    2: 'only a few arrows left',
                    3: 'half a quiver',
                    4: 'half a quiver',
                    5: 'most arrows remaining',
                    6: 'full quiver',
                }

                def roll_weapon_for_character(char_skills: str, char_name: str, is_player: bool = False):
                    """Roll d20 for weapon assignment. Returns (weapon_name, description) or None."""
                    # Check for thematic backstory override
                    for keyword, weapons in thematic_weapons.items():
                        if keyword in char_skills:
                            weapon = random.choice(weapons)
                            # Thematic weapons get good ammo (roll d6, minimum 3)
                            ammo_roll = max(3, random.randint(1, 6))
                            if any(w in weapon.lower() for w in ['bow', 'crossbow']):
                                desc = f"Your personal {weapon.lower()} ({quiver_status[ammo_roll]})"
                            elif any(w in weapon.lower() for w in ['pistol', 'rifle', 'shotgun', 'carbine', 'revolver']):
                                desc = f"Your personal {weapon.lower()} ({ammo_status[ammo_roll]})"
                            else:
                                desc = f"Your personal {weapon.lower()}"
                            print(f"[StoryBuilder] {char_name} gets thematic weapon: {weapon}")
                            return (weapon, desc)

                    # D20 roll for weapon
                    roll = random.randint(1, 20)
                    print(f"[StoryBuilder] {char_name} weapon roll: {roll}")

                    if roll <= 4:
                        # Reluctant/minimal weapon - even pacifists carry SOMETHING by month 1
                        weapons = minimal_weapons.get(genre, minimal_weapons['default'])
                        weapon = random.choice(weapons)
                        reluctant_descs = [
                            "Carried reluctantly, just in case",
                            "Picked up weeks ago, never used",
                            "Keeps it close but hopes to never need it",
                            "A concession to the new world",
                        ]
                        desc = random.choice(reluctant_descs)
                        print(f"[StoryBuilder] {char_name} has reluctant weapon: {weapon} (roll {roll})")
                        return (weapon, desc)
                    elif roll <= 8:
                        # Improvised weapon - grabbed what was available
                        weapons = minimal_weapons.get(genre, minimal_weapons['default'])
                        weapon = random.choice(weapons)
                        desc = f"Grabbed in desperation - better than nothing"
                        return (weapon, desc)
                    elif roll <= 12:
                        # Basic melee
                        weapons = basic_melee.get(genre, basic_melee['default'])
                        weapon = random.choice(weapons)
                        desc = f"A reliable {weapon.lower()}"
                        return (weapon, desc)
                    elif roll <= 16:
                        # Good melee
                        weapons = good_melee.get(genre, good_melee['default'])
                        weapon = random.choice(weapons)
                        desc = f"A solid {weapon.lower()} - proper weapon"
                        return (weapon, desc)
                    else:
                        # Firearm/ranged (17-20)
                        weapons = firearms.get(genre, firearms['default'])
                        weapon = random.choice(weapons)
                        ammo_roll = random.randint(1, 6)
                        if any(w in weapon.lower() for w in ['bow', 'crossbow']):
                            desc = f"{weapon} ({quiver_status[ammo_roll]})"
                        else:
                            desc = f"{weapon} ({ammo_status[ammo_roll]})"
                        return (weapon, desc)

                # Roll weapon for player
                weapon_result = roll_weapon_for_character(skills_str + ' ' + backstory_title, player_name, is_player=True)
                if weapon_result:
                    weapon_name, weapon_desc = weapon_result
                    starting_items.append((weapon_name, ItemCategory.WEAPON, weapon_desc))

                # Check for survival/outdoor skills
                if any(word in skills_str for word in ['survival', 'outdoor', 'tracking', 'hunting', 'navigation']):
                    survival_items = {
                        'fantasy': ['Flint and Steel', 'Rope (50ft)', 'Bedroll', 'Waterskin'],
                        'post-apocalyptic': ['Lighter', 'Paracord', 'Water Filter', 'Tarp'],
                        'zombie': ['Matches', 'Rope', 'Sleeping Bag', 'Water Bottle'],
                        'modern': ['Lighter', 'Multi-tool', 'Flashlight', 'Emergency Blanket'],
                        'sci-fi': ['Plasma Lighter', 'Nano-rope', 'Survival Scanner', 'Hydration Pack'],
                    }
                    items = survival_items.get(genre, ['Rope', 'Firestarter', 'Canteen'])
                    item = random.choice(items)
                    starting_items.append((item, ItemCategory.TOOL, f'{item} for survival situations'))
                if any(word in skills_str for word in ['navigation', 'scout', 'explorer']):
                    nav_items = {
                        'fantasy': ['Compass', 'Regional Map', 'Spyglass'],
                        'post-apocalyptic': ['Compass', 'Tattered Road Map', 'Binoculars'],
                        'sci-fi': ['Holo-map', 'Scanner', 'GPS Unit'],
                    }
                    items = nav_items.get(genre, ['Compass', 'Map'])
                    item = random.choice(items)
                    starting_items.append((item, ItemCategory.TOOL, f'For finding your way'))

                # Check for medical/healing skills
                if any(word in skills_str for word in ['medical', 'healing', 'medicine', 'first aid', 'doctor', 'nurse']):
                    med_items = {
                        'fantasy': ['Healer\'s Kit', 'Bandages', 'Healing Herbs', 'Antidote Vial'],
                        'post-apocalyptic': ['First Aid Kit', 'Bandages', 'Painkillers', 'Antibiotics'],
                        'zombie': ['First Aid Kit', 'Bandages', 'Antiseptic', 'Suture Kit'],
                        'modern': ['First Aid Kit', 'Trauma Kit', 'Bandages'],
                        'sci-fi': ['Medi-gel', 'Auto-injector', 'Nano-bandages'],
                    }
                    items = med_items.get(genre, ['Bandages', 'Healing Salve'])
                    item = random.choice(items)
                    starting_items.append((item, ItemCategory.CONSUMABLE, f'{item} for treating injuries'))

                # Check for stealth/thief skills
                if any(word in skills_str for word in ['stealth', 'thief', 'lockpick', 'sneak', 'shadow']):
                    starting_items.append(('Lockpicks', ItemCategory.TOOL, 'For getting into locked places'))

                # Check for magic/arcane skills
                if any(word in skills_str for word in ['magic', 'arcane', 'spell', 'sorcery', 'wizard', 'mage']):
                    starting_items.append(('Spellbook', ItemCategory.KEY_ITEM, 'Your personal grimoire'))

                # Check for scholarly skills
                if any(word in skills_str for word in ['scholar', 'research', 'knowledge', 'lore', 'study']):
                    starting_items.append(('Journal', ItemCategory.KEY_ITEM, 'For recording discoveries'))

                # Everyone gets basic supplies - with weights and capacity
                # Backpack: 2 lbs, holds 25 lbs
                starting_items.append(('Backpack', ItemCategory.CONTAINER, 'For carrying your belongings', 2.0, 25.0))
                # Rations: 2 lbs
                starting_items.append(('Rations', ItemCategory.CONSUMABLE, 'A few days worth of food', 2.0, 0.0))

                # Add the items (tuples can be 3, 4, or 5 elements)
                for item_tuple in starting_items:
                    item_name = item_tuple[0]
                    category = item_tuple[1]
                    description = item_tuple[2]
                    weight = item_tuple[3] if len(item_tuple) > 3 else 0.0
                    capacity = item_tuple[4] if len(item_tuple) > 4 else 0.0

                    player_inv.add_item(
                        name=item_name,
                        category=category,
                        description=description,
                        weight=weight,
                        capacity=capacity,
                        acquired_from='Starting equipment'
                    )

                inventory_tracker._save()
                print(f"[StoryBuilder] Player inventory created with {len(starting_items)} starting items")
            except Exception as e:
                print(f"[StoryBuilder] Warning: Could not create player inventory: {e}")

        except Exception as e:
            print(f"[StoryBuilder] Warning: Could not set up player character: {e}")

    # Generate inventories for companion characters based on their descriptions
    try:
        from inventory import ItemCategory
        import random
        all_partners = data_store.get_partners()

        # === SHARED D20 WEAPON SYSTEM FOR COMPANIONS ===
        # Same tables as player - defined here for companion use
        companion_minimal = {
            'zombie': ['Pocket Knife', 'Utility Knife', 'Box Cutter', 'Heavy Flashlight', 'Walking Stick', 'Belt'],
            'post-apocalyptic': ['Pocket Knife', 'Multitool', 'Box Cutter', 'Heavy Flashlight', 'Walking Stick', 'Chain'],
            'fantasy': ['Walking Staff', 'Hunting Knife', 'Belt Knife', 'Wooden Club'],
            'default': ['Pocket Knife', 'Utility Knife', 'Box Cutter', 'Heavy Flashlight', 'Walking Stick', 'Multitool'],
        }
        companion_basic_melee = {
            'zombie': ['Crowbar', 'Kitchen Knife', 'Baseball Bat', 'Hammer', 'Screwdriver', 'Tire Iron'],
            'post-apocalyptic': ['Crowbar', 'Wrench', 'Claw Hammer', 'Hatchet', 'Pry Bar', 'Tire Iron'],
            'fantasy': ['Dagger', 'Club', 'Hand Axe', 'Quarterstaff', 'Sickle'],
            'default': ['Knife', 'Club', 'Hammer', 'Crowbar'],
        }
        companion_good_melee = {
            'zombie': ['Machete', 'Fire Axe', 'Hunting Knife', 'Katana', 'Spear', 'Combat Knife'],
            'post-apocalyptic': ['Machete', 'Tomahawk', 'Bowie Knife', 'Sledgehammer', 'Bayonet'],
            'fantasy': ['Longsword', 'Battle Axe', 'Mace', 'Spear', 'Short Sword', 'War Hammer'],
            'default': ['Machete', 'Axe', 'Large Knife', 'Spear'],
        }
        companion_firearms = {
            'zombie': ['9mm Pistol', 'Revolver', '.22 Rifle', 'Hunting Rifle', 'Shotgun', 'Crossbow'],
            'post-apocalyptic': ['Pistol', 'Revolver', 'Hunting Rifle', 'Shotgun', 'Compound Bow', 'Crossbow'],
            'fantasy': ['Longbow', 'Crossbow', 'Shortbow', 'Hand Crossbow'],
            'sci-fi': ['Laser Pistol', 'Plasma Rifle', 'Rail Pistol', 'Energy Bow'],
            'default': ['Pistol', 'Rifle', 'Crossbow', 'Bow'],
        }
        companion_thematic = {
            # Hunting/outdoors
            'hunter': ['Hunting Rifle', 'Compound Bow', 'Crossbow'],
            'hunting': ['Hunting Rifle', 'Compound Bow', 'Crossbow'],
            'hunts': ['Hunting Rifle', 'Compound Bow', 'Crossbow'],
            'trapper': ['Hunting Rifle', 'Compound Bow', 'Crossbow'],
            # Military
            'military': ['M9 Pistol', 'Combat Knife', 'M4 Carbine'],
            'soldier': ['M9 Pistol', 'Combat Knife', 'Service Rifle'],
            'army': ['M9 Pistol', 'Combat Knife', 'M4 Carbine'],
            'marine': ['M9 Pistol', 'Combat Knife', 'M16'],
            'veteran': ['M9 Pistol', 'Combat Knife', 'Service Rifle'],
            'infantry': ['M9 Pistol', 'Combat Knife', 'M4 Carbine'],
            'served': ['Service Pistol', 'Combat Knife', 'Service Rifle'],
            # Law enforcement
            'police': ['Service Pistol', 'Baton', 'Shotgun'],
            'cop': ['Service Pistol', 'Baton', 'Taser'],
            'officer': ['Service Pistol', 'Baton', 'Shotgun'],
            'sheriff': ['Service Pistol', 'Shotgun', 'Revolver'],
            'deputy': ['Service Pistol', 'Baton', 'Shotgun'],
            'detective': ['Service Pistol', 'Revolver', 'Baton'],
            # Combat/martial
            'martial': ['Katana', 'Bo Staff', 'Nunchaku'],
            'martial arts': ['Katana', 'Bo Staff', 'Nunchaku'],
            'fighter': ['Combat Knife', 'Machete', 'Brass Knuckles'],
            'boxer': ['Brass Knuckles', 'Wraps', 'Combat Knife'],
            'brawler': ['Brass Knuckles', 'Crowbar', 'Baseball Bat'],
            # Ranged specialists
            'archer': ['Compound Bow', 'Recurve Bow', 'Crossbow'],
            'archery': ['Compound Bow', 'Recurve Bow', 'Crossbow'],
            'sniper': ['Hunting Rifle', 'Scoped Rifle', '.308 Rifle'],
            'marksman': ['Hunting Rifle', 'Scoped Rifle', 'M4 Carbine'],
            'sharpshooter': ['Hunting Rifle', 'Scoped Rifle', '.308 Rifle'],
            # Trades that imply weapons
            'mechanic': ['Wrench', 'Crowbar', 'Tire Iron'],
            'carpenter': ['Hammer', 'Hatchet', 'Nail Gun'],
            'butcher': ['Cleaver', 'Butcher Knife', 'Meat Hook'],
            'chef': ['Chef Knife', 'Cleaver', 'Kitchen Knife'],
            'cook': ['Kitchen Knife', 'Cleaver', 'Cast Iron Pan'],
            # Medical backgrounds - practical tools that can serve as weapons
            'emt': ['Trauma Shears', 'Heavy Flashlight', 'Multi-tool'],
            'paramedic': ['Trauma Shears', 'Heavy Flashlight', 'Utility Knife'],
            'nurse': ['Scalpel', 'Heavy Flashlight', 'Trauma Shears'],
            'doctor': ['Scalpel', 'Surgical Kit', 'Heavy Flashlight'],
            'medic': ['Combat Knife', 'Trauma Shears', 'Utility Knife'],
            # Non-combat backgrounds (empty = fall through to d20 roll)
            'teacher': [],
            'artist': [],
            'student': [],
        }
        companion_ammo_status = {
            1: 'almost empty - last few rounds',
            2: 'running low',
            3: 'half full',
            4: 'half full',
            5: 'mostly loaded',
            6: 'full',
        }
        companion_quiver_status = {
            1: 'last arrow',
            2: 'only a few arrows left',
            3: 'half a quiver',
            4: 'half a quiver',
            5: 'most arrows remaining',
            6: 'full quiver',
        }
        genre_lower = (genre or '').lower()

        def roll_companion_weapon(char_desc: str, char_name: str):
            """Roll d20 for companion weapon. Returns (weapon_name, description) or None."""
            desc_lower = char_desc.lower()

            # Check for thematic backstory override
            for keyword, weapons in companion_thematic.items():
                if keyword in desc_lower and weapons:
                    weapon = random.choice(weapons)
                    ammo_roll = max(3, random.randint(1, 6))
                    if any(w in weapon.lower() for w in ['bow', 'crossbow']):
                        desc = f"{char_name}'s {weapon.lower()} ({companion_quiver_status[ammo_roll]})"
                    elif any(w in weapon.lower() for w in ['pistol', 'rifle', 'shotgun', 'carbine', 'revolver']):
                        desc = f"{char_name}'s {weapon.lower()} ({companion_ammo_status[ammo_roll]})"
                    else:
                        desc = f"{char_name}'s {weapon.lower()}"
                    print(f"[StoryBuilder] {char_name} gets thematic weapon: {weapon}")
                    return (weapon, desc)

            # D20 roll
            roll = random.randint(1, 20)
            print(f"[StoryBuilder] {char_name} weapon roll: {roll}")

            if roll <= 4:
                # Reluctant weapon - everyone has SOMETHING by month 1
                weapons = companion_minimal.get(genre_lower, companion_minimal['default'])
                weapon = random.choice(weapons)
                reluctant_descs = [
                    f"{char_name} carries it reluctantly",
                    f"Picked up weeks ago, {char_name} hopes to never use it",
                    f"{char_name}'s concession to the new world",
                    f"Kept close but unused",
                ]
                print(f"[StoryBuilder] {char_name} has reluctant weapon: {weapon} (roll {roll})")
                return (weapon, random.choice(reluctant_descs))
            elif roll <= 8:
                weapons = companion_minimal.get(genre_lower, companion_minimal['default'])
                weapon = random.choice(weapons)
                return (weapon, f"Grabbed in desperation")
            elif roll <= 12:
                weapons = companion_basic_melee.get(genre_lower, companion_basic_melee['default'])
                weapon = random.choice(weapons)
                return (weapon, f"{char_name}'s {weapon.lower()}")
            elif roll <= 16:
                weapons = companion_good_melee.get(genre_lower, companion_good_melee['default'])
                weapon = random.choice(weapons)
                return (weapon, f"{char_name}'s trusty {weapon.lower()}")
            else:
                weapons = companion_firearms.get(genre_lower, companion_firearms['default'])
                weapon = random.choice(weapons)
                ammo_roll = random.randint(1, 6)
                if any(w in weapon.lower() for w in ['bow', 'crossbow']):
                    return (weapon, f"{char_name}'s {weapon.lower()} ({companion_quiver_status[ammo_roll]})")
                else:
                    return (weapon, f"{char_name}'s {weapon.lower()} ({companion_ammo_status[ammo_roll]})")

        # Generate inventory for ALL partners in this room, not just new ones
        # Skip if they already have inventory from a previous room
        for partner_id in all_partner_ids:
            partner = next((p for p in all_partners if p.id == partner_id), None)
            if not partner:
                continue

            # Clear existing inventory - each new StoryBuilder world starts fresh
            existing_inv = inventory_tracker.get_inventory(partner_id)
            if existing_inv and existing_inv.items:
                print(f"[StoryBuilder] Clearing {partner.name}'s old inventory ({len(existing_inv.items)} items) for new world")
                existing_inv.items.clear()
                inventory_tracker._save()

            # Use Ollama to extract items from character description
            char_desc = partner.character_description or ''
            phys_desc = partner.physical_description or ''
            full_desc = f"{char_desc}\n{phys_desc}".strip()

            # Fallback items by genre if description is sparse (NO WEAPONS - those are assigned separately)
            fallback_items = {
                'zombie': ['Flashlight', 'Water Bottle', 'First Aid Kit', 'Lighter', 'Rope', 'Binoculars', 'Multi-tool', 'Bandana', 'Duct Tape', 'Can of Food', 'Worn Backpack', 'Walkie-Talkie', 'Matches', 'Tarp', 'Sleeping Bag'],
                'post-apocalyptic': ['Flashlight', 'Water Bottle', 'First Aid Kit', 'Lighter', 'Rope', 'Binoculars', 'Multi-tool', 'Bandana', 'Duct Tape', 'Can of Food', 'Worn Backpack', 'Gas Mask', 'Matches', 'Water Filter', 'Tarp'],
                'fantasy': ['Belt Pouch', 'Waterskin', 'Flint and Steel', 'Rope', 'Bedroll', 'Cloak', 'Rations', 'Candles', 'Healing Herbs', 'Coin Purse', 'Traveler\'s Pack', 'Lantern', 'Map', 'Journal', 'Whetstone'],
                'horror': ['Flashlight', 'Lighter', 'Cell Phone', 'Keys', 'Wallet', 'Jacket', 'Watch', 'Notebook', 'Pen', 'Snack Bar', 'Water Bottle', 'Bandages', 'Matches', 'Rosary', 'Batteries'],
                'scifi': ['Data Pad', 'Utility Tool', 'Ration Pack', 'Med-Kit', 'ID Badge', 'Comm Device', 'Flashlight', 'Multi-tool', 'Jacket', 'Credits Chip', 'Stim Pack', 'Respirator', 'Toolkit', 'Scanner', 'Power Cell'],
                'default': ['Lighter', 'Water Bottle', 'Small Bag', 'Notebook', 'Pen', 'Snacks', 'Phone', 'Keys', 'Jacket', 'Bandages', 'Flashlight', 'Rope', 'Multi-tool', 'Matches', 'Canteen']
            }

            # Pick fallback list based on genre
            genre_lower = (genre or '').lower()
            fallback_list = fallback_items.get('default')
            for g, items in fallback_items.items():
                if g in genre_lower:
                    fallback_list = items
                    break

            extract_prompt = f"""Based on this character description, list 3-4 NON-WEAPON items they would likely be carrying.

Character: {partner.name}
Description: {full_desc if full_desc else "(No description provided)"}
Setting: {scenario_description}
Genre: {genre}

Return ONLY a JSON array of items. Each item should have "name" and "category".
Categories: TOOL, CONSUMABLE, CLOTHING, KEY_ITEM, CONTAINER, MISC
DO NOT include weapons - those are assigned separately.

Example response:
[{{"name": "Sketchbook", "category": "KEY_ITEM"}}, {{"name": "Flashlight", "category": "TOOL"}}]

IMPORTANT: If the description doesn't mention specific items, pick 3-4 reasonable items from this list that fit the character:
{fallback_list}

Be specific to the character. If they're an artist, give them art supplies. If they're a medic, give them medical supplies. If unclear, use the fallback list above.
Return ONLY the JSON array, no explanation."""

            try:
                items_json = _call_ollama_sync(
                    room_model or settings.storybuilder_model,
                    extract_prompt,
                    "You extract inventory items from character descriptions. Return only valid JSON arrays."
                )

                if items_json:
                    # Parse the JSON response
                    import json
                    # Clean up response - find JSON array
                    items_json = items_json.strip()
                    if '[' in items_json:
                        items_json = items_json[items_json.index('['):items_json.rindex(']')+1]

                    items = json.loads(items_json)

                    # Create inventory for this companion
                    companion_inv = inventory_tracker.get_or_create_inventory(
                        partner_id, partner.name, owner_type='character'
                    )

                    # Map category strings to enums
                    category_map = {
                        'WEAPON': ItemCategory.WEAPON,
                        'TOOL': ItemCategory.TOOL,
                        'CONSUMABLE': ItemCategory.CONSUMABLE,
                        'CLOTHING': ItemCategory.CLOTHING,
                        'KEY_ITEM': ItemCategory.KEY_ITEM,
                        'CONTAINER': ItemCategory.CONTAINER,
                        'MISC': ItemCategory.MISC,
                        'ARMOR': ItemCategory.ARMOR,
                        'TREASURE': ItemCategory.TREASURE,
                    }

                    # Add non-weapon items from Ollama (weapons come from d20 roll)
                    added_items = []
                    for item in items:
                        item_name = item.get('name', '')
                        cat_str = item.get('category', 'MISC').upper()
                        category = category_map.get(cat_str, ItemCategory.MISC)

                        # Skip weapons - we'll roll for those separately
                        if cat_str == 'WEAPON':
                            print(f"[StoryBuilder] Skipping Ollama weapon '{item_name}' for {partner.name} - using d20 roll instead")
                            continue

                        if item_name:
                            companion_inv.add_item(
                                name=item_name,
                                category=category,
                                description=f"Personal item of {partner.name}",
                                acquired_from='Starting equipment'
                            )
                            added_items.append(item_name)

                    # D20 roll for weapon
                    weapon_result = roll_companion_weapon(full_desc, partner.name)
                    if weapon_result:
                        weapon_name, weapon_desc = weapon_result
                        companion_inv.add_item(
                            name=weapon_name,
                            category=ItemCategory.WEAPON,
                            description=weapon_desc,
                            acquired_from='Starting equipment'
                        )
                        added_items.append(weapon_name)

                    inventory_tracker._save()
                    print(f"[StoryBuilder] Generated inventory for {partner.name}: {added_items}")

            except Exception as e:
                # Fallback: give them random items + d20 weapon roll
                print(f"[StoryBuilder] Ollama inventory failed for {partner.name}, using fallback: {e}")
                try:
                    companion_inv = inventory_tracker.get_or_create_inventory(
                        partner_id, partner.name, owner_type='character'
                    )
                    # Pick 3 random NON-weapon items from fallback
                    weapon_words = ['knife', 'crowbar', 'axe', 'bat', 'gun', 'pistol', 'rifle', 'sword', 'blade']
                    non_weapon_fallback = [i for i in fallback_list if not any(w in i.lower() for w in weapon_words)]
                    fallback_picks = random.sample(non_weapon_fallback, min(3, len(non_weapon_fallback)))
                    for item_name in fallback_picks:
                        companion_inv.add_item(
                            name=item_name,
                            category=ItemCategory.MISC,
                            description=f"Personal item of {partner.name}",
                            acquired_from='Starting equipment'
                        )

                    # D20 roll for weapon
                    weapon_result = roll_companion_weapon(full_desc, partner.name)
                    if weapon_result:
                        weapon_name, weapon_desc = weapon_result
                        companion_inv.add_item(
                            name=weapon_name,
                            category=ItemCategory.WEAPON,
                            description=weapon_desc,
                            acquired_from='Starting equipment'
                        )
                        fallback_picks.append(weapon_name)

                    inventory_tracker._save()
                    print(f"[StoryBuilder] Fallback inventory for {partner.name}: {fallback_picks}")
                except Exception as e2:
                    print(f"[StoryBuilder] Warning: Could not create fallback inventory for {partner.name}: {e2}")

    except Exception as e:
        print(f"[StoryBuilder] Warning: Could not generate companion inventories: {e}")

    # Set up separated characters based on character_relationships
    # Characters that don't start with the player live their own background story
    print(f"[StoryBuilder] Setting up character relationships. character_relationships count: {len(character_relationships) if character_relationships else 0}")
    print(f"[StoryBuilder] include_existing: {include_existing}")
    print(f"[StoryBuilder] char_id_to_partner_id: {char_id_to_partner_id}")

    if character_relationships:
        try:
            from autopilot import get_autopilot_tracker, Alignment, Role
            tracker = get_autopilot_tracker()

            for rel in character_relationships:
                char_id = rel.get('character_id', '')
                char_name = rel.get('character_name', 'Unknown')
                starts_with_player = rel.get('starts_with_player', True)
                starting_location = rel.get('starting_location', '')

                print(f"[StoryBuilder] Processing rel: char_id={char_id}, char_name={char_name}, starts_with_player={starts_with_player}")

                # Get the partner ID for this character
                # For new characters, look up in the mapping
                # For existing partners, the char_id IS the partner_id
                partner_id = char_id_to_partner_id.get(char_id)
                if not partner_id and char_id in include_existing:
                    partner_id = char_id  # Existing partner - char_id is already the partner_id
                    print(f"[StoryBuilder] Found existing partner: {char_id}")
                if not partner_id:
                    print(f"[StoryBuilder] WARNING: Could not find partner_id for char_id={char_id}")
                    continue

                # Create autopilot entry for this partner character
                print(f"[StoryBuilder] Creating autopilot entry for {char_name} (partner_id={partner_id})")
                pc = tracker.get_or_create(partner_id, room.id, char_name)

                # If they have alignment info from relationships, we could set it here
                # For now, use defaults - the character's personality drives behavior

                # NOTE: We do NOT call separate_character() here anymore.
                # The SeparatedSim block below handles full separation with matrix details.
                # This avoids double journal entries for separated characters.
                if not starts_with_player:
                    print(f"[StoryBuilder] Marking {char_name} as separated (id: {partner_id})")

            tracker._save()
        except Exception as e:
            print(f"[StoryBuilder] Warning: Could not set up separated characters: {e}")

    # Generate dramatic opening narration ("fade from black")
    final_opening_narration = ""  # Capture for auto-image feature
    try:
        # Get starting location from cartographer if available
        starting_location_name = ""
        if world_map:
            # WorldMap stores starting_location_name directly
            if hasattr(world_map, 'starting_location_name') and world_map.starting_location_name:
                starting_location_name = world_map.starting_location_name
            elif world_map.starting_location_id:
                # Fallback: look up in locations dict (iterate values, not keys)
                loc = world_map.locations.get(world_map.starting_location_id)
                if loc and hasattr(loc, 'name'):
                    starting_location_name = loc.name

            # Set player's current location
            if starting_location_name:
                room.player_location = starting_location_name
                data_store.save()

        # === OPENING MATRIX SYSTEM ===
        # Roll across multiple axes for varied openings (uses module-level roll_opening_matrix)

        # Detect if zombie genre
        is_zombie = False
        scenario_lower = (scenario_description or '').lower()
        genre_lower = (genre or '').lower()
        if any(z in scenario_lower or z in genre_lower for z in ['zombie', 'undead', 'walker', 'infected', 'outbreak']):
            is_zombie = True

        # Roll the matrix! Pass shelter_type to constrain housing appropriately
        opening_matrix = roll_opening_matrix(genre or '', is_zombie, shelter_type=shelter_type)
        time_of_day = opening_matrix['time_of_day']

        print(f"[StoryBuilder] OPENING MATRIX ROLL (shelter_type={shelter_type}):")
        for k, v in opening_matrix.items():
            if v is not None:
                print(f"  - {k}: {v}")

        # === SYNC WORLD STATE FROM OPENING MATRIX ===
        # This ensures /inventory, /continue, etc. know the actual time/weather
        global story_daemon
        if story_daemon:
            ws = story_daemon.get_or_create_world_state(room.id)

            # Normalize time_of_day from matrix (e.g., "golden hour" -> "dawn")
            raw_time = opening_matrix.get('time_of_day', 'day').lower()
            if any(t in raw_time for t in ['dawn', 'sunrise', 'golden hour', 'grey hour']):
                ws.time_of_day = 'dawn'
                ws.game_hour = 6
            elif any(t in raw_time for t in ['morning']):
                ws.time_of_day = 'day'
                ws.game_hour = 9
            elif any(t in raw_time for t in ['midday', 'noon', 'afternoon']):
                ws.time_of_day = 'day'
                ws.game_hour = 14
            elif any(t in raw_time for t in ['dusk', 'evening', 'sunset', 'late afternoon']):
                ws.time_of_day = 'dusk'
                ws.game_hour = 18
            elif any(t in raw_time for t in ['night', 'deep night', '3am']):
                ws.time_of_day = 'night'
                ws.game_hour = 22 if 'deep' not in raw_time else 3
            else:
                ws.time_of_day = 'day'
                ws.game_hour = 12

            # Normalize weather from matrix (e.g., "fog / mist" -> "fog")
            raw_weather = opening_matrix.get('weather', 'clear').lower()
            if 'fog' in raw_weather or 'mist' in raw_weather:
                ws.weather = 'fog'
            elif 'storm' in raw_weather or 'heavy rain' in raw_weather:
                ws.weather = 'storm'
            elif 'rain' in raw_weather:
                ws.weather = 'rain'
            elif 'snow' in raw_weather:
                ws.weather = 'snow'
            elif 'frost' in raw_weather or 'cold' in raw_weather:
                ws.weather = 'cold'
            elif 'wind' in raw_weather or 'gale' in raw_weather:
                ws.weather = 'windy'
            elif 'overcast' in raw_weather or 'cloud' in raw_weather:
                ws.weather = 'cloudy'
            else:
                ws.weather = 'clear'

            # Set genre/mood
            ws.mood = genre or 'dramatic'

            # Set threat level based on matrix
            if opening_matrix.get('threat_type') == 'none':
                ws.threat_level = 2
            elif opening_matrix.get('threat_distance') == 'close':
                ws.threat_level = 7
            elif opening_matrix.get('threat_distance') == 'medium':
                ws.threat_level = 5
            else:
                ws.threat_level = 3

            print(f"[StoryBuilder] Synced WorldState: {ws.time_of_day} ({ws.game_hour}:00), {ws.weather}, threat={ws.threat_level}")

        # Get character names for the opening - ONLY those who start with player
        all_partners = data_store.get_partners()
        # Build set of separated character IDs
        # NOTE: character_relationships has already been updated to use partner IDs (see above)
        # so character_id IS the partner_id at this point
        separated_char_ids = set()
        for rel in character_relationships:
            if not rel.get('starts_with_player', True):
                partner_id = rel.get('character_id', '')  # Already a partner ID
                if partner_id:
                    separated_char_ids.add(partner_id)
                    print(f"[StoryBuilder] Marking {rel.get('character_name')} as separated (id: {partner_id})")
        # Filter out separated characters from opening narration
        # Include gender info so narrator doesn't misgender characters
        present_partners = [p for p in all_partners if p.id in all_partner_ids and p.id not in separated_char_ids]

        def format_char_info(p):
            """Format character with rich context for opening narration."""
            import re
            gender = getattr(p, 'gender', 'unknown')
            phys_desc = getattr(p, 'physical_description', '') or ''
            char_desc = getattr(p, 'character_description', '') or ''
            skill = getattr(p, 'skill', '') or ''

            # Try to extract age from physical description
            age_match = re.search(r'(\d+)[- ]?year[- ]?old|([a-z]+ \d+s)', phys_desc.lower())
            age_hint = age_match.group(0) if age_match else ''

            # Build base: "Name (age gender)"
            if age_hint:
                base = f"{p.name} ({age_hint} {gender})"
            elif gender and gender != 'unknown':
                base = f"{p.name} ({gender})"
            else:
                base = p.name

            # Detect if child (under 13)
            is_child = False
            child_match = re.search(r'(\d+)[- ]?year', phys_desc.lower())
            if child_match:
                age_num = int(child_match.group(1))
                if age_num < 13:
                    is_child = True

            # Add character essence - condensed personality/role
            context_parts = []

            # For children, add explicit behavioral guidance
            if is_child:
                context_parts.append("CHILD - must act age-appropriate: scared, dependent on adults, NO combat roles")

            # Add skill if present
            if skill:
                context_parts.append(f"skilled at: {skill}")

            # Add condensed character description (first sentence or 100 chars)
            if char_desc:
                # Get first sentence or truncate
                first_sentence = char_desc.split('.')[0].strip()
                if len(first_sentence) > 100:
                    first_sentence = first_sentence[:100] + '...'
                context_parts.append(first_sentence)

            if context_parts:
                return f"{base} - {'; '.join(context_parts)}"
            return base

        char_descriptions = [format_char_info(p) for p in present_partners]
        player_name = player_character.get('name', settings.user_name) if player_character else settings.user_name

        # Get player role for anti-trope guidance
        player_role = player_character.get('role', '') if player_character else ''
        player_backstory_info = player_character.get('backstory', {}) if player_character else {}
        player_backstory_title = player_backstory_info.get('title', '') if isinstance(player_backstory_info, dict) else ''

        # Build companion context
        if char_descriptions:
            companion_line = f"WITH YOU: {', '.join(char_descriptions)}"
        else:
            companion_line = "You are ALONE. No companions are with you right now."

        print(f"[StoryBuilder] Generating opening narration for {room.name}")
        print(f"[StoryBuilder] Location: {starting_location_name or 'unknown'}, Time: {time_of_day}")
        print(f"[StoryBuilder] Characters: {char_descriptions}, Player: {player_name}")
        print(f"[StoryBuilder] Using model: {room_model}")

        # Build companion instruction
        companion_instruction = ""
        first_name = ""
        if char_descriptions:
            first_name = char_descriptions[0].split(" (")[0] if " (" in char_descriptions[0] else char_descriptions[0]
            companion_instruction = f"\nPARAGRAPH 1 MUST include {first_name} doing something beside you."

        # === BUILD MATRIX CONTEXT FOR PROMPT ===
        # (uses module-level TENSION_TYPE_DESCRIPTIONS)

        # === GATHER ACTUAL WEAPONS FROM INVENTORIES ===
        # This ensures the opening narration matches actual inventory contents
        actual_weapons_info = []
        try:
            # Player weapons
            player_inv = inventory_tracker.get_inventory(player_id) if player_id else None
            if player_inv:
                player_weapons = [item.name for item in player_inv.items if item.category.value == 'weapon']
                if player_weapons:
                    actual_weapons_info.append(f"{player_name} has: {', '.join(player_weapons)}")

            # Companion weapons (only for present companions)
            for partner in present_partners:
                partner_inv = inventory_tracker.get_inventory(partner.id)
                if partner_inv:
                    partner_weapons = [item.name for item in partner_inv.items if item.category.value == 'weapon']
                    if partner_weapons:
                        actual_weapons_info.append(f"{partner.name} has: {', '.join(partner_weapons)}")
        except Exception as e:
            print(f"[StoryBuilder] Could not gather weapon info: {e}")

        # Build the weapons context for the prompt
        if actual_weapons_info:
            weapons_context = " | ".join(actual_weapons_info)
            print(f"[StoryBuilder] Actual weapons: {weapons_context}")
        else:
            # No weapons found in inventory - they're unarmed
            weapons_context = "neither armed (no weapons in inventory)"
            print(f"[StoryBuilder] No weapons in inventories")

        # Build threat context
        threat_context = ""
        if opening_matrix['threat_type'] != 'none':
            threat_context = f"""
THREAT PRESENT:
- Type: {opening_matrix['threat_type']}
- Distance: {opening_matrix['threat_distance']} (close=at your feet, medium=across the area, far=visible in distance)
- Awareness: {opening_matrix['threat_awareness']}
- Size: {opening_matrix['threat_size']}
NOTE: One zombie CLOSE = horde FAR in tension intensity. Scale accordingly."""
        else:
            threat_context = "\nNO DIRECT THREAT - but tension comes from the situation itself (deadline, discovery, environment)."

        # Build legendary event context
        legendary_context = ""
        if opening_matrix['legendary'] == 'separated_character':
            # Get names of separated characters if any
            separated_names = [p.name for p in all_partners if p.id in separated_char_ids]
            if separated_names:
                legendary_context = f"\n🌟 LEGENDARY MOMENT: Through the chaos, you glimpse someone you thought you'd lost: {separated_names[0]}. This reunion (or near-miss) should be the emotional core of this opening."
            else:
                legendary_context = "\n🌟 LEGENDARY MOMENT: You spot someone from your past - a face from before everything changed. Make this emotionally charged."
        elif opening_matrix['legendary'] == 'backstory_connection':
            legendary_context = "\n🌟 LEGENDARY MOMENT: You find something connected to the player's past life - a photo, a letter, a familiar object. This discovery should hit emotionally."
        elif opening_matrix['legendary'] == 'perfect_find':
            legendary_context = "\n🌟 LEGENDARY MOMENT: A perfect lucky find - working vehicle, untouched supplies, functioning equipment. Something that changes everything. But what's the catch?"
        elif opening_matrix['legendary'] == 'worst_case':
            legendary_context = "\n🌟 LEGENDARY MOMENT: Everything goes wrong at once. Multiple threats, bad weather, injured companion, no weapons. The absolute worst-case scenario that tests survival to the limit."

        # Companion state instruction
        companion_state_instruction = ""
        if char_descriptions and opening_matrix['companion_state']:
            companion_state_instruction = f"\n{first_name} is: {opening_matrix['companion_state']}. Show this in their body language/actions (NOT dialogue)."

        # === BUILD SEPARATED CHARACTER CONTEXT ===
        # Only mention separated characters who are KNOWN to at least one present character
        separated_context = ""
        if separated_char_ids and character_relationships:
            # Get all present character IDs (including player)
            present_char_ids = set(p.id for p in present_partners)

            # Find separated characters known to anyone present
            known_separated = []
            for sep_id in separated_char_ids:
                sep_partner = next((p for p in all_partners if p.id == sep_id), None)
                if not sep_partner:
                    continue

                # Check if any present character knows this separated character
                is_known = False
                relationship_type = None
                knower_name = None

                for cr in character_relationships:
                    cr_char_id = cr.get('character_id', '')
                    # Check if this relationship record belongs to a present character
                    if cr_char_id in present_char_ids or cr_char_id == player_id:
                        for rel in cr.get('relationships', []):
                            if rel.get('target_id') == sep_id and rel.get('type', 'stranger') != 'stranger':
                                is_known = True
                                relationship_type = rel.get('type', 'acquaintance')
                                knower_name = cr.get('character_name', 'someone')
                                break
                    if is_known:
                        break

                if is_known:
                    known_separated.append({
                        'name': sep_partner.name,
                        'relationship': relationship_type,
                        'known_by': knower_name
                    })

            if known_separated:
                sep_lines = []
                for sep in known_separated:
                    sep_lines.append(f"- {sep['name']} ({sep['relationship']} of {sep['known_by']}) - separated, elsewhere in the world")
                separated_context = f"\n\nSEPARATED (not present, but may be mentioned in 'recently'):\n" + "\n".join(sep_lines)

        opening_prompt = f"""Write an immersive opening in THREE FLOWING SECTIONS. No headers - natural transitions between sections.

=== THE THREE SECTIONS ===

**SECTION 1: RECENTLY (1-2 paragraphs)**
The last few days of your group's life. How have you been surviving? What's the rhythm been?
- Establish the texture of daily existence in this world
- Show how the group functions together (or struggles to)
- If separated characters are listed below, you may briefly mention when/why they parted
- This is BACKSTORY, not current action - use past tense for events, present for ongoing states

**SECTION 2: MOTIVATION (1 paragraph)**
What does your group need or want right now? Not a railroad destination - an implied drive.
- "We should really find medicine" energy, not "we must go to the hospital"
- Could be: supplies running low, someone mentioned a rumor, a decision that needs making
- This gives the story MOMENTUM without forcing direction
- Weave it naturally - a thought, a glance at dwindling supplies, a half-formed plan

**SECTION 3: THE MOMENT (1-2 paragraphs)**
The actual scene - NOW. This is where TENSION TYPE and SCENE STRUCTURE apply.
- First word of this section should feel like "present moment" arriving
- Ground us: TIME, WEATHER, LOCATION woven in naturally
- Show HOUSING and RESOURCE state through details
- Apply your TENSION TYPE here (or RESPITE if rolled)
- Companions doing things (NOT speaking) - varied actions, not all weapon-cleaning
- End with open tension or quiet weight - NOT a binary "A or B?" choice

=== WORLD STATE (YOUR ROLL) ===
SETTING: {scenario_description}
LOCATION: {starting_location_name or "unspecified"}
TIME: {time_of_day}
WEATHER: {opening_matrix['weather']}
GENRE: {genre}

TENSION TYPE (for Section 3): {opening_matrix['tension_type']}
→ {TENSION_TYPE_DESCRIPTIONS.get(opening_matrix['tension_type'], 'Create appropriate tension.')}
{threat_context}

SCENE STRUCTURE (for Section 3): {opening_matrix['scene_structure']}
→ {SCENE_STRUCTURE_DESCRIPTIONS.get(opening_matrix['scene_structure'], '')}

SITUATION:
- Housing: {opening_matrix['housing']}
- Resources: {opening_matrix['resources']}
- Weapons: {weapons_context}
{legendary_context}

=== CHARACTERS ===
PRESENT (physically here in Section 3):
- The player ({player_name}) - USE "YOU"
{chr(10).join(f"- {c}" for c in char_descriptions) if char_descriptions else "- (alone)"}
{companion_state_instruction}
{separated_context}

=== CRITICAL RULES ===
- Section 3 MUST start with present-tense grounding. Transition from backstory to NOW.
- ONLY characters in PRESENT list appear in Section 3. Separated characters may be MENTIONED in Sections 1-2 but are NOT HERE.
- NO DIALOGUE from companions in Section 3. They act silently.
- NO "A or B?" choices. End with open tension or quiet weight.
- WEAPONS: Reference actual inventory items by name. If "neither armed" - they're vulnerable.
- HOUSING: If "homeless/temporary/between places" → Section 3 is OUTDOORS, no buildings, no settled details.
- REACTIONS: Proportional. One distant zombie = Tuesday. Horde at the door = panic.
- VARIETY: Not everyone cleaning weapons. Different actions per character. Match personality to behavior.
- CHILDREN: Age-appropriate only. No combat tasks.
- If TENSION TYPE is RESPITE: No external threat. Internal weight - exhaustion, hope, grief, connection. Let them breathe.
- PRESENT TENSE for Section 3's climax: When the hook arrives (threat appears, discovery made, etc.), use PRESENT tense - "steps into view", "moves closer" - not past tense. It's happening NOW.

{OPENING_SCENE_RULES}
- The player is "{player_role or 'unspecified'}" - if they're SUPPORT, the companion might be the confident one.

Write all three sections now, flowing naturally. Section 3's first grounding word should feel like "arriving in the present":"""

        # Determine if we need no-shelter reinforcement
        housing_lower = opening_matrix['housing'].lower() if opening_matrix.get('housing') else ''
        is_homeless = any(h in housing_lower for h in ['homeless', 'temporary', 'between places', 'searching'])
        location_rule = " Section 3 MUST be OUTDOORS - housing is homeless/temporary." if is_homeless else ""

        system = f"You write immersive three-section openings: RECENTLY (backstory), MOTIVATION (what you need), THE MOMENT (current scene). Flow naturally between sections - no headers. Honor the MATRIX ROLL in Section 3. Companions NEVER speak dialogue. No 'A or B?' choices. If RESPITE is rolled, Section 3 is peaceful - internal tension only.{f' {first_name} must appear in Section 3 doing something (not talking).' if char_descriptions else ''}{location_rule}"

        opening_narration = _call_ollama_sync(room_model or settings.storybuilder_model, opening_prompt, system)

        if opening_narration and opening_narration.strip():
            final_opening_narration = opening_narration.strip()  # Capture for auto-image
            # Add as the first message in the room (from DM/narrator)
            opening_message = Message(
                id=str(uuid.uuid4())[:8],
                speaker_id="narrator",
                speaker_name="📖",
                content=final_opening_narration,
                room_id=room.id,
                message_type="narration",
            )
            data_store.add_message(room.id, opening_message)
            print(f"[StoryBuilder] Generated opening narration for {room.name} ({len(final_opening_narration)} chars)")
        else:
            print(f"[StoryBuilder] WARNING: Opening narration was empty for {room.name}")
            # Fallback: use the scenario description as the opening
            if scenario_description and scenario_description.strip():
                fallback_message = Message(
                    id=str(uuid.uuid4())[:8],
                    speaker_id="narrator",
                    speaker_name="📖",
                    content=scenario_description.strip(),
                    room_id=room.id,
                    message_type="narration",
                )
                data_store.add_message(room.id, fallback_message)
                print(f"[StoryBuilder] Using scenario description as fallback opening")
    except Exception as e:
        import traceback
        print(f"[StoryBuilder] ERROR generating opening narration: {e}")
        traceback.print_exc()
        # Fallback on error too
        if scenario_description and scenario_description.strip():
            try:
                fallback_message = Message(
                    id=str(uuid.uuid4())[:8],
                    speaker_id="narrator",
                    speaker_name="📖",
                    content=scenario_description.strip(),
                    room_id=room.id,
                    message_type="narration",
                )
                data_store.add_message(room.id, fallback_message)
                print(f"[StoryBuilder] Using scenario description as fallback opening after error")
            except Exception:
                pass

    # Create NPCs from unchosen characters (they exist in the world at RESIDUE level)
    # This way if Sable mentions Calder, but you don't pick Calder, he still exists
    unchosen_characters = data.get('unchosen_characters', [])
    unchosen_npc_count = 0
    if unchosen_characters:
        try:
            from npc_system import NPCState

            # Build lookup for character locations from relationships
            char_locations = {}
            for rel in character_relationships:
                char_id = rel.get('character_id', '')
                location = rel.get('starting_location', '')
                if char_id and location:
                    char_locations[char_id] = location

            for char in unchosen_characters:
                char_id = char.get('id', '')
                char_name = char.get('name', '')
                if not char_name:
                    continue

                # Get hidden traits if we have them
                hidden = _character_secrets.pop(char_id, {})
                if isinstance(hidden, str):
                    hidden = {'secret': hidden}

                # Get their location from relationships, or derive from role
                npc_location = char_locations.get(char_id, '')
                if not npc_location:
                    # Try to derive location from their role/personality
                    personality = char.get('personality', '').lower()
                    if 'store' in personality or 'shop' in personality:
                        npc_location = "at their shop"
                    elif 'dock' in personality or 'fish' in personality:
                        npc_location = "down by the docks"
                    elif 'guard' in personality or 'patrol' in personality:
                        npc_location = "on patrol"
                    else:
                        npc_location = "somewhere in the area"

                # Create NPC at RESIDUE level (they stick around, not ephemeral)
                npc = npc_registry.create_npc(
                    name=char_name,
                    origin_world=room.id,
                    backstory=char.get('personality', ''),
                    current_role=char.get('personality', ''),  # Use full personality as role
                    physical_description=char.get('appearance', ''),
                    personality=char.get('personality', ''),
                    secret=hidden.get('secret'),
                    wound=hidden.get('wound'),
                    want=hidden.get('want'),
                    fear=hidden.get('fear'),
                )

                # Start at RESIDUE so they persist (not ephemeral)
                npc.state = NPCState.RESIDUE
                npc.total_interactions = 0  # Not discovered yet - will be set if mentioned by chosen chars
                npc.interaction_weight = 15.0  # Above ephemeral threshold (keeps them RESIDUE)
                npc.current_location = npc_location  # Their actual location

                # Add to room's NPC tracking
                if not hasattr(data_store, '_room_npcs'):
                    data_store._room_npcs = {}
                if room.id not in data_store._room_npcs:
                    data_store._room_npcs[room.id] = set()
                data_store._room_npcs[room.id].add(npc.id)

                unchosen_npc_count += 1
                print(f"[StoryBuilder] Created NPC from unchosen character: {char_name} at '{npc_location}' (RESIDUE)")

            _save_npcs()

            # Check if any chosen character's description mentions an unchosen character
            # If so, mark that NPC as "discovered" since the player character already knows them
            # e.g., if Juniper's description says "her father Silas", then Silas is pre-discovered
            try:
                import re

                # Build lookup of unchosen NPC names (lowercase) -> NPC objects
                unchosen_npcs_by_name = {}
                for char in unchosen_characters:
                    char_name = char.get('name', '')
                    if char_name:
                        # Find the NPC we just created
                        for npc in npc_registry.npcs.values():
                            if npc.name == char_name and npc.origin_world == room.id:
                                unchosen_npcs_by_name[char_name.lower()] = npc
                                break

                # Gather all chosen character descriptions
                chosen_descriptions = []
                for rel in character_relationships:
                    personality = rel.get('personality', '')
                    if personality:
                        chosen_descriptions.append(personality)

                # Also check existing partners being brought in
                if include_existing:
                    all_partners = data_store.get_partners()
                    for p in all_partners:
                        if p.id in include_existing:
                            desc = getattr(p, 'character_description', '') or ''
                            if desc:
                                chosen_descriptions.append(desc)

                combined_text = ' '.join(chosen_descriptions).lower()

                # Check each unchosen NPC name against the combined text
                for npc_name_lower, npc in unchosen_npcs_by_name.items():
                    if npc_name_lower in combined_text:
                        # This NPC is mentioned by a chosen character - mark as discovered
                        npc.total_interactions = max(npc.total_interactions, 1)  # At least 1 to show as "known"
                        print(f"[StoryBuilder] Pre-discovered NPC: {npc.name} (mentioned in chosen character's description)")

                _save_npcs()
            except Exception as e:
                print(f"[StoryBuilder] Warning: Could not check for relationship-based discovery: {e}")

        except Exception as e:
            print(f"[StoryBuilder] Warning: Could not create unchosen NPCs: {e}")

    # Auto-spawn referenced characters as NPCs
    # If an existing character's description mentions another character by name (capitalized),
    # and that character exists but isn't in the current game, spawn them as an NPC
    referenced_npc_count = 0
    if include_existing:
        try:
            import re
            from npc_system import NPCState

            all_partners = data_store.get_partners()

            # Build lookup of all partner names -> partner objects (excluding those already in game)
            available_partners = {
                p.name.lower(): p for p in all_partners
                if p.id not in all_partner_ids
            }

            # Get descriptions of existing characters being brought in
            existing_partners = [p for p in all_partners if p.id in include_existing]

            # Track which referenced characters we've already added
            added_references = set()

            for partner in existing_partners:
                desc = partner.character_description or ""
                if not desc:
                    continue

                # Find capitalized words that could be names (2+ chars, starts with capital)
                # This catches "Calder" in "She follows Calder around like a shadow"
                potential_names = re.findall(r'\b([A-Z][a-z]+)\b', desc)

                for name in potential_names:
                    name_lower = name.lower()

                    # Skip if already added or if it's a common word
                    if name_lower in added_references:
                        continue

                    # Skip common words that happen to be capitalized
                    common_words = {'the', 'she', 'her', 'his', 'they', 'their', 'this', 'that',
                                   'when', 'where', 'what', 'who', 'how', 'but', 'and', 'for',
                                   'has', 'have', 'had', 'was', 'were', 'been', 'being', 'will'}
                    if name_lower in common_words:
                        continue

                    # Check if this name matches an available partner
                    if name_lower in available_partners:
                        ref_partner = available_partners[name_lower]
                        added_references.add(name_lower)

                        # Create NPC from this referenced partner
                        npc = npc_registry.create_npc(
                            room_id=room.id,
                            name=ref_partner.name,
                            description=ref_partner.character_description or f"Known to {partner.name}",
                            state=NPCState.RESIDUE,  # Exists in world, findable
                            scenario=scenario_description
                        )

                        # Copy over physical description if available
                        if ref_partner.physical_description:
                            npc.personality = ref_partner.physical_description

                        # They're somewhere in the world (not with player)
                        npc.current_location = "somewhere nearby"

                        # Track who referenced them
                        npc.current_role = f"known to {partner.name}"

                        # Add to room's NPC tracking
                        if not hasattr(data_store, '_room_npcs'):
                            data_store._room_npcs = {}
                        if room.id not in data_store._room_npcs:
                            data_store._room_npcs[room.id] = set()
                        data_store._room_npcs[room.id].add(npc.id)

                        referenced_npc_count += 1
                        print(f"[StoryBuilder] Auto-spawned referenced character: {ref_partner.name} (mentioned in {partner.name}'s description)")

            if referenced_npc_count > 0:
                _save_npcs()

        except Exception as e:
            print(f"[StoryBuilder] Warning: Could not create referenced NPCs: {e}")

    # === REGISTER SEPARATED CHARACTERS WITH FULL SIMULATION ===
    # Characters with starts_with_player: false get the SAME systems as the player:
    # - Opening matrix (tension, weather, threats, resources)
    # - Inventory tracking
    # - Condition tracking (hunger/thirst)
    # - Fatigue tracking
    # This is NOT a lite version - they live a full parallel story.
    separated_count = 0
    try:
        all_partners = data_store.get_partners()

        # Get current world state for syncing
        world_state = None
        if story_daemon:
            world_state = story_daemon.get_world_state(room.id)

        print(f"[SeparatedSim] Processing {len(character_relationships)} character relationships")
        print(f"[SeparatedSim] All partner IDs in room: {all_partner_ids}")

        for rel in character_relationships:
            if not rel.get('starts_with_player', True):
                char_id = rel.get('character_id', '')
                char_name = rel.get('character_name', '')
                starting_location = rel.get('starting_location', 'somewhere in the area')
                print(f"[SeparatedSim] Checking separated char: {char_name} (id={char_id})")

                if not char_id:
                    continue

                # Find the partner object - try by ID first, then by name as fallback
                char_partner = next((p for p in all_partners if p.id == char_id), None)
                if not char_partner:
                    # Fallback: try to find by name (for StoryBuilder characters whose IDs may not have been mapped)
                    char_partner = next((p for p in all_partners if p.name == char_name and p.id in all_partner_ids), None)
                    if char_partner:
                        print(f"[SeparatedSim] Found {char_name} by name fallback (ID was {char_id}, actual is {char_partner.id})")
                        char_id = char_partner.id  # Use the correct ID
                    else:
                        print(f"[SeparatedSim] WARNING: Could not find partner for {char_name} (char_id={char_id})")
                        continue

                # === ROLL THEIR OPENING MATRIX ===
                # Same system the player gets - their solo journey begins with full context
                sep_matrix = roll_opening_matrix(genre or '', is_zombie)
                print(f"[SeparatedSim] {char_name} OPENING MATRIX:")
                for k, v in sep_matrix.items():
                    if v is not None:
                        print(f"  - {k}: {v}")

                # Register with autopilot tracker
                pc = autopilot_tracker.get_or_create(char_id, room.id, char_name)
                pc.is_separated = True
                pc.is_alive = True
                pc.separation_started = datetime.now().isoformat()
                pc.last_known_location = starting_location
                pc.current_drive = Drive.SURVIVAL

                # Store their opening matrix for use in ticks
                pc.separation_matrix = sep_matrix

                # Sync world state
                if world_state:
                    pc.current_weather = world_state.weather
                    pc.current_time_of_day = world_state.time_of_day
                else:
                    # Normalize from their matrix
                    raw_weather = sep_matrix.get('weather', 'clear').lower()
                    if 'fog' in raw_weather or 'mist' in raw_weather:
                        pc.current_weather = 'fog'
                    elif 'storm' in raw_weather or 'heavy rain' in raw_weather:
                        pc.current_weather = 'storm'
                    elif 'rain' in raw_weather:
                        pc.current_weather = 'rain'
                    else:
                        pc.current_weather = 'clear'
                    pc.current_time_of_day = sep_matrix.get('time_of_day', 'day')

                # === SET UP INVENTORY ===
                char_inv_id = f"separated_{char_id}"
                char_inv = inventory_tracker.get_or_create_inventory(
                    char_inv_id, char_name, owner_type='separated'
                )
                # Start with basic survival gear based on matrix resources
                resources = sep_matrix.get('resources', 'adequate').lower()
                if 'abundant' in resources or 'surplus' in resources:
                    char_inv.add_item(name='Backpack', category=ItemCategory.CONTAINER, weight=2.0, capacity=25.0)
                    char_inv.add_item(name='Canned food (3 days)', category=ItemCategory.CONSUMABLE, weight=2.0)
                    char_inv.add_item(name='Water bottle', category=ItemCategory.CONSUMABLE, weight=1.0)
                    char_inv.add_item(name='First aid supplies', category=ItemCategory.MEDICAL, weight=0.5)
                elif 'adequate' in resources:
                    char_inv.add_item(name='Backpack', category=ItemCategory.CONTAINER, weight=2.0, capacity=25.0)
                    char_inv.add_item(name='Canned food (1 day)', category=ItemCategory.CONSUMABLE, weight=0.7)
                    char_inv.add_item(name='Water bottle (half)', category=ItemCategory.CONSUMABLE, weight=0.5)
                elif 'low' in resources or 'scarce' in resources:
                    char_inv.add_item(name='Worn backpack', category=ItemCategory.CONTAINER, weight=1.5, capacity=15.0)
                    char_inv.add_item(name='Granola bar', category=ItemCategory.CONSUMABLE, weight=0.1)
                # Critical/none = they start with nothing
                inventory_tracker._save()

                # === SET UP CONDITION TRACKING ===
                char_condition = condition_tracker.get_or_create(char_inv_id, char_name)
                # Start healthy but condition reflects their matrix situation
                if 'injured' in sep_matrix.get('companion_state', '').lower():
                    char_condition.add_injury(
                        description="Minor injury from separation",
                        severity="minor",
                    )
                condition_tracker.save()

                # === SET UP FATIGUE TRACKING ===
                from fatigue import get_fatigue_tracker
                fatigue_tracker = get_fatigue_tracker()
                char_fatigue = fatigue_tracker.get_or_create(char_inv_id, char_name)
                # Start with some fatigue based on their state
                if 'exhausted' in sep_matrix.get('companion_state', '').lower():
                    char_fatigue.hours_awake = 20.0
                elif 'tired' in sep_matrix.get('companion_state', '').lower():
                    char_fatigue.hours_awake = 14.0
                else:
                    char_fatigue.hours_awake = 8.0  # Normal start
                fatigue_tracker._save()

                # Add rich initial journal entry with their matrix
                matrix_summary = f"Weather: {sep_matrix.get('weather', 'unknown')}, "
                matrix_summary += f"Threat: {sep_matrix.get('threat_type', 'unknown')}, "
                matrix_summary += f"Resources: {sep_matrix.get('resources', 'unknown')}"

                pc.add_journal_entry(
                    event_type='separation',
                    description=f"Separated from the group at {starting_location}. {matrix_summary}. Beginning solo journey.",
                    severity="notable",
                    game_day=1,
                    game_hour=0,
                )

                separated_count += 1
                print(f"[SeparatedSim] Registered {char_name} with FULL SIMULATION at '{starting_location}'")

        if separated_count > 0:
            autopilot_tracker._save()
            print(f"[SeparatedSim] {separated_count} character(s) now running full parallel simulation")

    except Exception as e:
        print(f"[StoryBuilder] Warning: Could not register separated characters: {e}")

    return jsonify({
        'room_id': room.id,
        'room_name': room.name,
        'partner_count': len(all_partner_ids),
        'new_partners': new_partner_ids,
        'has_world_map': world_map is not None,
        'unchosen_npcs': unchosen_npc_count,
        'referenced_npcs': referenced_npc_count,
        'separated_characters': separated_count,
        'opening_narration': final_opening_narration,  # For auto-image feature
    })




@app.route('/rooms/<room_id>/inciting-incident', methods=['POST'])
def generate_inciting_incident(room_id):
    """Generate a dramatic inciting incident for the story.
    If the room has no messages, generates an opening paragraph instead."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    partners = data_store.get_partners()
    player_name = room.player_character_name or settings.user_name or "the player"
    player_location = getattr(room, 'player_location', '') or "the area"

    # Sanitize location - reject UUID-looking strings that leaked into location names
    import re
    if player_location and re.search(r'[0-9a-f]{8,}', player_location.lower()):
        player_location = "the area"  # Fallback for corrupted locations

    # Get only characters who are actually WITH the player (not separated)
    present_ids = room.present_character_ids or []
    present_partners = [p for p in partners if p.id in present_ids]

    print(f"[Inciting/Opening] present_character_ids: {present_ids}")
    print(f"[Inciting/Opening] present_partners: {[p.name for p in present_partners]}")

    # Build context about who's physically present
    character_list = "\n".join([
        f"- {p.name}: {p.character_description[:200] if p.character_description else 'No description'}"
        for p in present_partners
    ]) if present_partners else "(You are alone)"

    # Get recent conversation for context
    recent_messages = room.messages[-10:] if room.messages else []
    conversation_context = "\n".join([
        f"{m.speaker_name}: {m.content[:150]}" for m in recent_messages
    ]) if recent_messages else ""

    # Build the scenario context
    scenario_text = room.scenario if room.scenario else "A group conversation"
    genre = room.genre if room.genre else "drama"
    scene_layout = room.scene_layout if room.scene_layout else ""

    # If no messages, generate an opening paragraph instead of a one-liner incident
    is_opening = len(room.messages) == 0

    # Get player inventory for context (so narration doesn't invent items)
    player_id = f"player_{room_id}"
    player_inventory = inventory_tracker.get_inventory(player_id)
    inventory_items = []
    if player_inventory:
        for item in player_inventory.items:
            if item.quantity > 1:
                inventory_items.append(f"{item.name} (x{item.quantity})")
            else:
                inventory_items.append(item.name)
    inventory_context = ", ".join(inventory_items) if inventory_items else "nothing notable"

    if is_opening:
        # === OPENING MATRIX SYSTEM ===
        # Roll across multiple axes for varied openings (same as storybuilder)

        # Detect if zombie genre
        is_zombie = False
        scenario_lower = (scenario_text or '').lower()
        genre_lower = (genre or '').lower()
        if any(z in scenario_lower or z in genre_lower for z in ['zombie', 'undead', 'walker', 'infected', 'outbreak']):
            is_zombie = True

        # Roll the matrix!
        opening_matrix = roll_opening_matrix(genre or '', is_zombie)
        time_of_day = opening_matrix['time_of_day']

        print(f"[Inciting/Opening] OPENING MATRIX ROLL:")
        for k, v in opening_matrix.items():
            if v is not None:
                print(f"  - {k}: {v}")

        # Build companion context - who's here with descriptions
        companion_context = ""
        first_name = ""
        if present_partners:
            companion_lines = []
            for p in present_partners:
                desc = p.character_description[:150] if p.character_description else "No description"
                companion_lines.append(f"- {p.name}: {desc}")
            companion_context = "COMPANIONS WITH YOU:\n" + "\n".join(companion_lines)
            first_name = present_partners[0].name
        else:
            companion_context = "You are ALONE. No companions are with you right now."

        # Build threat context
        threat_context = ""
        if opening_matrix['threat_type'] != 'none':
            threat_context = f"""
THREAT PRESENT:
- Type: {opening_matrix['threat_type']}
- Distance: {opening_matrix['threat_distance']} (close=at your feet, medium=across the area, far=visible in distance)
- Awareness: {opening_matrix['threat_awareness']}
- Size: {opening_matrix['threat_size']}
NOTE: One zombie CLOSE = horde FAR in tension intensity. Scale accordingly."""
        else:
            threat_context = "\nNO DIRECT THREAT - but tension comes from the situation itself (deadline, discovery, environment)."

        # Companion state instruction
        companion_state_instruction = ""
        if first_name and opening_matrix['companion_state']:
            companion_state_instruction = f"\n{first_name} is: {opening_matrix['companion_state']}. Show this in their body language/actions (NOT dialogue)."

        # Generate a full opening narration using three-section structure
        prompt = f"""Write an immersive opening in THREE FLOWING SECTIONS. No headers - natural transitions.

=== THE THREE SECTIONS ===

**SECTION 1: RECENTLY (1-2 paragraphs)**
The last few days of your group's life. How have you been surviving? What's the rhythm been?
- Establish the texture of daily existence in this world
- Show how the group functions together (or struggles to)
- This is BACKSTORY, not current action - use past tense for events

**SECTION 2: MOTIVATION (1 paragraph)**
What does your group need or want right now? Not a railroad - an implied drive.
- "We should really find medicine" energy, not "we must go to the hospital"
- Weave it naturally - a thought, a glance at dwindling supplies, a half-formed plan

**SECTION 3: THE MOMENT (1-2 paragraphs)**
The actual scene - NOW. This is where TENSION TYPE and SCENE STRUCTURE apply.
- Ground us: TIME, WEATHER, LOCATION woven in naturally
- Show companions doing things (NOT speaking) - varied actions
- End with open tension or quiet weight - NOT a binary choice
- PRESENT TENSE for the climax - it's happening NOW

=== WORLD STATE (YOUR ROLL) ===
Title: {room.name}
Setting: {scenario_text}
Genre: {genre}
LOCATION: {player_location}
TIME: {time_of_day}
WEATHER: {opening_matrix['weather']}

TENSION TYPE (for Section 3): {opening_matrix['tension_type']}
→ {TENSION_TYPE_DESCRIPTIONS.get(opening_matrix['tension_type'], 'Create appropriate tension.')}
{threat_context}

SCENE STRUCTURE (for Section 3): {opening_matrix['scene_structure']}
→ {SCENE_STRUCTURE_DESCRIPTIONS.get(opening_matrix['scene_structure'], '')}

SITUATION:
- Housing: {opening_matrix['housing']}
- Resources: {opening_matrix['resources']}

THE PLAYER: {player_name} - write from their perspective using "you"
PLAYER'S GEAR: {inventory_context}

{companion_context}
{companion_state_instruction}

=== CRITICAL RULES ===
- ONLY include characters listed above in Section 3. Others are NOT HERE.
- ONLY reference items from PLAYER'S GEAR - don't invent weapons/tools.
- NO DIALOGUE from companions. They act silently.
- NO "A or B?" choices. End with open tension.
- Section 3's climax uses PRESENT TENSE - "steps into view", not "stepped".
- PROPORTIONAL REACTIONS: One distant zombie = Tuesday. Horde at door = panic.
- VARIED ACTIONS: Not everyone cleaning weapons. Match personality to behavior.
- If RESPITE: No external threat. Internal weight only - exhaustion, hope, grief.

{OPENING_SCENE_RULES}

Write all three sections now, flowing naturally:"""
        system = "You write immersive three-section openings: RECENTLY (backstory), MOTIVATION (what you need), THE MOMENT (current scene). Flow naturally - no headers. Honor the MATRIX ROLL in Section 3. Companions NEVER speak. No 'A or B?' choices. PRESENT TENSE for Section 3's climax."
        message_type = "opening_narration"
        speaker_name = "📖"
    else:
        # Generate a dramatic incident
        prompt = f"""You are a dramatic narrator generating an INCITING INCIDENT - something that happens in the world that forces the characters to react.

SETTING: {scenario_text}
LOCATION: {player_location}

CHARACTERS PHYSICALLY PRESENT (only these people are here):
- {player_name} (the viewpoint)
{character_list}

RECENT EVENTS:
{conversation_context}

Generate a brief, dramatic event that:
1. Happens TO the characters or in their immediate environment
2. Demands attention and response
3. Creates tension, conflict, or urgency
4. Is specific and sensory (what do they see, hear, feel?)

Write 1-3 sentences describing what happens. Do NOT write dialogue. Do NOT write character reactions.
Just describe the event as a narrator would.

Example good incidents:
- "A gunshot rings out from the back room, followed by the sound of breaking glass."
- "The lights flicker and die. In the sudden darkness, something heavy scrapes across the floor above."
- "A stranger bursts through the door, bleeding heavily, and collapses at their feet."

Generate an incident now:"""
        system = "You are a dramatic narrator. Generate vivid, tense story events. Be concise and impactful."
        message_type = "inciting_incident"
        speaker_name = "Narrator"

    # Call the model (using room's model or storybuilder_model setting)
    # Auto-pick first available model if configured one doesn't exist
    import httpx
    model_to_use = getattr(room, 'room_model', '') or settings.storybuilder_model
    available_models = provider_manager.get_models_for_provider('ollama')
    if model_to_use not in available_models and available_models:
        print(f"[Inciting Incident] Model '{model_to_use}' not found, using '{available_models[0]}'")
        model_to_use = available_models[0]
    print(f"[Inciting Incident] Using model: {model_to_use}")
    print(f"[Inciting Incident] Ollama URL: {settings.ollama_base_url}/api/chat")
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": model_to_use,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                }
            )
            print(f"[Inciting Incident] Response status: {response.status_code}")
            if response.status_code == 200:
                response_data = response.json()
                incident_text = response_data.get("message", {}).get("content", "").strip()
                print(f"[Inciting Incident] Generated: {incident_text[:100]}...")
                if not incident_text:
                    return jsonify({'error': 'Model returned empty response'}), 500
            else:
                error_msg = response.json().get('error', 'Unknown error') if response.text else 'No response'
                print(f"[Inciting Incident] Error response: {error_msg}")
                return jsonify({'error': f"Ollama error: {error_msg}"}), 500
    except httpx.ConnectError:
        print(f"[Inciting Incident] Cannot connect to Ollama")
        return jsonify({'error': 'Cannot connect to Ollama. Is it running?'}), 500
    except Exception as e:
        print(f"[Inciting Incident] Exception: {type(e).__name__}: {e}")
        return jsonify({'error': f'Failed to generate incident: {e}'}), 500

    # Create the message (either opening narration or incident)
    incident_message = Message(
        id=str(uuid.uuid4())[:8],
        speaker_id="narrator",
        speaker_name=speaker_name,
        content=incident_text,
        room_id=room_id,
        message_type=message_type,
    )
    data_store.add_message(room_id, incident_message)

    print(f"[{'Opening' if is_opening else 'Incident'}] Added {message_type} message")

    return jsonify({
        'id': incident_message.id,
        'speaker_id': incident_message.speaker_id,
        'speaker_name': incident_message.speaker_name,
        'content': incident_text,
        'message_type': message_type,
    })


@app.route('/rooms/<room_id>/describe-scene', methods=['POST'])
def describe_scene(room_id):
    """Generate a pure observational description of the current scene."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    partners = data_store.get_partners()
    player_name = room.player_character_name or settings.user_name or "the player"
    stored_location = getattr(room, 'player_location', '') or "the area"

    # Sanitize location - reject UUID-looking strings
    import re
    if stored_location and re.search(r'[0-9a-f]{8,}', stored_location.lower()):
        stored_location = "the area"

    # === DETECT ACTUAL LOCATION FROM RECENT NARRATIVE ===
    # The stored location might be stale - parse recent messages to find where player actually is
    recent_messages = room.messages[-6:] if room.messages else []
    recent_narrative = "\n".join([
        f"{m.speaker_name}: {m.content}" for m in recent_messages
    ]) if recent_messages else ""

    player_location = stored_location  # Default to stored
    if recent_narrative:
        # Quick model call to extract actual current location
        location_prompt = f"""Based on this recent narrative, where is {player_name} RIGHT NOW?

{recent_narrative}

Answer in ONE short phrase describing the immediate physical location.
Examples: "outside under an awning", "inside the warehouse", "on the roof", "in the alley", "crouched behind a car"

If unclear, say: {stored_location}

Just the location phrase, nothing else:"""

        try:
            import httpx
            model_to_use = getattr(room, 'room_model', '') or settings.storybuilder_model
            available_models = provider_manager.get_models_for_provider('ollama')
            if model_to_use not in available_models and available_models:
                model_to_use = available_models[0]

            with httpx.Client(timeout=30.0) as client:
                loc_response = client.post(
                    f"{settings.ollama_base_url}/api/chat",
                    json={
                        "model": model_to_use,
                        "messages": [
                            {"role": "system", "content": "You extract locations from narrative text. Be brief and precise."},
                            {"role": "user", "content": location_prompt}
                        ],
                        "stream": False,
                    }
                )
                if loc_response.status_code == 200:
                    loc_data = loc_response.json()
                    detected_location = loc_data.get("message", {}).get("content", "").strip()
                    if detected_location and len(detected_location) < 100:
                        player_location = detected_location
                        print(f"[Describe Scene] Detected location: {player_location}")
        except Exception as e:
            print(f"[Describe Scene] Location detection failed, using stored: {e}")

    # Get only characters who are actually WITH the player (not separated)
    # Use present_character_ids which tracks who's physically present
    present_ids = room.present_character_ids or []
    present_partners = [p for p in partners if p.id in present_ids]
    separated_partners = [p for p in partners if p.id not in present_ids]

    # Build character descriptions - ONLY those actually present
    character_list = []
    for p in present_partners:
        brief_desc = p.character_description[:150] if p.character_description else "No description"
        character_list.append(f"- {p.name}: {brief_desc}")
    character_text = "\n".join(character_list) if character_list else "(You are alone)"

    # Build explicit exclusion list for separated characters
    if separated_partners:
        separated_names = [p.name for p in separated_partners]
        exclusion_text = f"\n\n⛔ NOT IN THIS SCENE (separated, elsewhere): {', '.join(separated_names)}\nDo NOT position these characters. They are NOT HERE. Ignore any mentions of them in context."
    else:
        exclusion_text = ""

    # Get recent conversation for context on what just happened
    recent_messages = room.messages[-8:] if room.messages else []
    recent_context = "\n".join([
        f"{m.speaker_name}: {m.content[:100]}" for m in recent_messages
    ]) if recent_messages else "(Scene just started)"

    # === EXTRACT THREATS FROM RECENT NARRATIVE ===
    # Look for zombies, enemies, hostile figures, etc. in the last several messages
    threat_keywords = ['zombie', 'walker', 'infected', 'shambler', 'runner', 'figure', 'stranger',
                       'hostile', 'enemy', 'threat', 'creature', 'bandit', 'raider', 'approaching',
                       'lurching', 'stumbling', 'shambling', 'watching', 'following']

    threats_mentioned = []
    extended_messages = room.messages[-12:] if room.messages else []
    for m in extended_messages:
        content_lower = m.content.lower()
        if any(kw in content_lower for kw in threat_keywords):
            # Extract the relevant sentence(s) about threats
            import re
            sentences = re.split(r'[.!?]', m.content)
            for sentence in sentences:
                if any(kw in sentence.lower() for kw in threat_keywords):
                    threats_mentioned.append(sentence.strip()[:200])

    # Build threat context
    if threats_mentioned:
        # Deduplicate and take most recent/relevant
        unique_threats = list(dict.fromkeys(threats_mentioned))[-3:]  # Last 3 unique mentions
        threat_context = "\n".join([f"- {t}" for t in unique_threats])
    else:
        threat_context = "(No active threats detected in recent narrative)"

    # Build the scenario context
    scenario_text = room.scenario if room.scenario else "An unspecified location"
    scene_layout = room.scene_layout if room.scene_layout else ""

    prompt = f"""You are a SPATIAL CLARITY ENGINE. Your job is to eliminate ALL confusion about where people AND THREATS are.

SETTING: {scenario_text}
CURRENT LOCATION: {player_location}
{f"LAYOUT: {scene_layout}" if scene_layout else ""}

CHARACTERS PHYSICALLY PRESENT (these are the ONLY people here):
- {player_name} (the viewpoint - refer to them as "you")
{character_text}{exclusion_text}

=== THREATS/ENEMIES MENTIONED IN RECENT NARRATIVE ===
{threat_context}

RECENT EVENTS (for context on positioning):
{recent_context}

=== YOUR TASK: SURGICAL SPATIAL PRECISION ===

Answer these questions through your description:
1. INSIDE or OUTSIDE? If outside, what structure is nearest? Are you AGAINST it, NEAR it, or DISTANT from it?
2. COMPASS/CLOCK: What's NORTH/AHEAD? What's to your LEFT? RIGHT? BEHIND you?
3. DISTANCES: Use real measurements. "Five feet to your left." "Twenty yards ahead." "The door is ten steps behind you."
4. COVER: Are you exposed? Behind something? If behind cover, what KIND - a wall? A vehicle? Bushes? Is it SOLID cover or CONCEALMENT only?
5. EXITS: Where could you run? What paths exist? What's blocking what?
6. ELEVATION: Ground level? Uphill? Downhill? On a roof? In a ditch?
7. THREATS: If threats are listed above, WHERE ARE THEY? Distance, direction, what they're doing. This is CRITICAL - don't ignore active threats!

Write 3-5 sentences of PURE SPATIAL INFORMATION. Think like a tactical operator clearing a room - every position matters.

EXAMPLE OF GOOD SPATIAL CLARITY:
"You're on the second floor landing of a stairwell, back pressed against the east wall. The stairs descend to your left - twelve steps to ground level. Marcus is crouched three feet ahead of you, peering through a cracked door into the hallway. Behind you, the stairs continue up another flight to a rooftop access. The hallway beyond the door stretches north, with two closed doors on the right and an elevator bank at the far end, maybe forty feet away."

⚠️ MUST INCLUDE (if present in THREATS section above):
- Location of any zombies, enemies, or hostile figures mentioned
- Their distance and direction from the player
- What they're currently doing (shambling, watching, approaching, etc.)
- DO NOT IGNORE THREATS - if the narrative mentions enemies, they MUST appear in your description

DO NOT include:
- Characters not listed in CHARACTERS section
- Enemies/threats NOT mentioned in the THREATS section (don't invent new ones)
- Emotional language or drama
- What anyone is thinking
- Suggestions or choices
- Any dialogue

JUST TELL US WHERE EVERYTHING IS. Like a blueprint that talks."""

    system = "You are a spatial clarity engine. Your ONLY job is eliminating confusion about physical positions AND THREATS. If threats are mentioned in the input, you MUST include them with distance/direction. Use distances, directions, and precise spatial language. No poetry - just positions. ONLY describe characters and threats explicitly listed."

    # Call the model - use room's model
    import httpx
    model_to_use = getattr(room, 'room_model', '') or settings.storybuilder_model
    available_models = provider_manager.get_models_for_provider('ollama')
    if model_to_use not in available_models and available_models:
        model_to_use = available_models[0]
    print(f"[Describe Scene] Using model: {model_to_use}")

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": model_to_use,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                }
            )
            if response.status_code == 200:
                response_data = response.json()
                scene_text = response_data.get("message", {}).get("content", "").strip()
                if not scene_text:
                    return jsonify({'error': 'Model returned empty response'}), 500
            else:
                error_msg = response.json().get('error', 'Unknown error') if response.text else 'No response'
                return jsonify({'error': f"Ollama error: {error_msg}"}), 500
    except httpx.ConnectError:
        return jsonify({'error': 'Cannot connect to Ollama. Is it running?'}), 500
    except Exception as e:
        print(f"[Describe Scene] Exception: {type(e).__name__}: {e}")
        return jsonify({'error': f'Failed to describe scene: {e}'}), 500

    # === GENERATE SPATIAL MAP WITH HAIKU (silent fallback if no API) ===
    spatial_map = None
    try:
        if settings.anthropic_api_key:
            import anthropic
            haiku_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

            map_prompt = f"""Convert this scene description into a fixed 25x15 ASCII spatial diagram.

SCENE:
{scene_text}

RULES:
- Grid is EXACTLY 25 chars wide x 15 lines tall (pad with spaces if needed)
- Use these symbols:
  @ = player (you)
  A-Z = other characters (first letter of name)
  # = walls/buildings
  . = open ground
  ~ = water
  ^ = trees/forest
  = = road/path
  * = threat/enemy
  ! = important object
- Add a compass: N at top
- Add distance labels where mentioned (e.g. "20yd")
- Keep it SIMPLE and READABLE

OUTPUT ONLY THE 15-LINE ASCII MAP, nothing else."""

            map_response = haiku_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=600,
                messages=[{"role": "user", "content": map_prompt}]
            )
            spatial_map = map_response.content[0].text.strip()
            print(f"[Describe Scene] Generated spatial map ({len(spatial_map)} chars)")
    except Exception as map_err:
        # Silent fallback - no map is fine
        print(f"[Describe Scene] Spatial map skipped: {type(map_err).__name__}: {map_err}")

    # Create the narration message
    scene_message = Message(
        id=str(uuid.uuid4())[:8],
        speaker_id="narrator",
        speaker_name="📖",
        content=scene_text,
        room_id=room_id,
        message_type="narration",
        spatial_map=spatial_map,
    )
    data_store.add_message(room_id, scene_message)

    response_data = {
        'id': scene_message.id,
        'speaker_id': scene_message.speaker_id,
        'speaker_name': scene_message.speaker_name,
        'content': scene_text,
        'message_type': 'narration',
    }
    if spatial_map:
        response_data['spatial_map'] = spatial_map

    return jsonify(response_data)


@app.route('/rooms/<room_id>/narrate-transition', methods=['POST'])
def narrate_transition(room_id):
    """
    Narrate a scene transition - skip the mundane travel and land at the next scene.

    Context-driven, not random. The DM evaluates:
    - Current situation and destination
    - Threat level and world state
    - Whether the path is safe/familiar or dangerous

    Either narrates a smooth transition or introduces a complication if warranted.
    """
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    data = request.json or {}
    destination_hint = data.get('destination', '').strip()  # Optional explicit destination

    partners = data_store.get_partners()
    player_name = room.player_character_name or settings.user_name or "the player"

    # Get present companions
    present_ids = room.present_character_ids or []
    present_partners = [p for p in partners if p.id in present_ids]
    companion_names = [p.name for p in present_partners]
    companion_text = f"with {', '.join(companion_names)}" if companion_names else "alone"

    # Get recent conversation to understand context
    recent_messages = room.messages[-10:] if room.messages else []
    recent_narrative = "\n".join([
        f"{m.speaker_name}: {m.content[:300]}" for m in recent_messages
    ]) if recent_messages else ""

    # Get world state for threat assessment
    world_state = None
    threat_level = 0
    time_of_day = "day"
    weather = "clear"

    global story_daemon
    if story_daemon:
        world_state = story_daemon.get_world_state(room_id)
        if world_state:
            threat_level = world_state.threat_level
            time_of_day = world_state.time_of_day
            weather = world_state.weather

    # Current location
    current_location = getattr(room, 'player_location', '') or "current location"

    # Build the context-analysis prompt
    # This is the key: we ask the model to EVALUATE whether the transition is safe,
    # not just randomly roll
    analysis_prompt = f"""Analyze this scene transition request. READ THE PHYSICAL DETAILS CAREFULLY.

RECENT NARRATIVE:
{recent_narrative}

WORLD STATE:
- Time: {time_of_day}
- Weather: {weather}
- Threat level: {threat_level}/10
- Genre: {room.genre or 'drama'}

CURRENT LOCATION: {current_location}
DESTINATION HINT: {destination_hint if destination_hint else "(infer from conversation - where are they heading?)"}
TRAVELING: {player_name} {companion_text}

DESTINATION INFERENCE RULES (if no hint provided):
1. DIALOGUE TRUMPS DESCRIPTION: If a character SAYS "let's go to X" or "we should head to X", that's the destination
2. EXPLICIT > IMPLICIT: "The Rustbucket" mentioned in dialogue beats "sawmill" mentioned as scenery
3. NAMED LOCATIONS: Proper nouns (The Rustbucket, Mom's Diner, The Culvert) are likely destinations
4. BACKGROUND ≠ DESTINATION: Scenery details (sawmill in the distance, mountains visible) are NOT destinations

PARSE THE PHYSICAL SITUATION CAREFULLY:
- Where EXACTLY are they right now? (inside a building? on a deck? in a field?)
- HOW are they traveling? (walking, driving, sailing, running?)
- What's BETWEEN them and the destination? (field, road, water, buildings?)
- What physical details were mentioned? (frost, plants, doors, terrain?)

Based on the narrative context, evaluate:
1. Where are they trying to go? (PRIORITIZE explicit mentions in dialogue over background scenery)
2. How are they getting there? (mode of travel)
3. What terrain/environment do they pass through?
4. Is this path familiar/safe or dangerous/unknown?
5. Should something happen during travel, or smooth transition?

IMPORTANT: Don't force drama. If they're walking familiar ground in low-threat conditions
after a meaningful character moment, a smooth transition is often BETTER for pacing.

However, if the situation makes a transition inappropriate, say so:
- If you can't determine a clear destination from context, use TRANSITION_TYPE: unclear
- If threat level is high (7+) or there's active danger, use TRANSITION_TYPE: blocked

Respond in this format:
DESTINATION: [specific place name - no explanations, or "unknown" if unclear]
TRAVEL_MODE: [walking/running/driving/etc]
TERRAIN: [what they pass through - field, road, dock, etc]
TRANSITION_TYPE: [smooth OR interrupted OR unclear OR blocked]
COMPLICATION: [if interrupted, what happens - otherwise "none"]
REASON: [if unclear/blocked, brief explanation why transition can't happen]"""

    try:
        import httpx
        model_to_use = getattr(room, 'room_model', '') or settings.storybuilder_model
        available_models = provider_manager.get_models_for_provider('ollama')
        if model_to_use not in available_models and available_models:
            model_to_use = available_models[0]

        # First: analyze the transition
        with httpx.Client(timeout=45.0) as client:
            analysis_response = client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": model_to_use,
                    "messages": [
                        {"role": "system", "content": "You are a skilled DM evaluating scene transitions. Be thoughtful about pacing - not every journey needs drama."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    "stream": False,
                }
            )

            if analysis_response.status_code != 200:
                return jsonify({'error': 'Failed to analyze transition'}), 500

            analysis_data = analysis_response.json()
            analysis_text = analysis_data.get("message", {}).get("content", "")

            print(f"[Transition] Analysis:\n{analysis_text}")

        # Parse the analysis
        destination = "their destination"
        transition_type = "smooth"
        complication = "none"
        reason = ""

        travel_mode = "walking"
        terrain = "the area"

        for line in analysis_text.split('\n'):
            if line.startswith('DESTINATION:'):
                destination = line.split(':', 1)[1].strip()
                # Clean up any parenthetical explanations the AI added
                if '(' in destination:
                    destination = destination.split('(')[0].strip()
                # Also clean up quotes if present
                destination = destination.strip('"\'')
            elif line.startswith('TRAVEL_MODE:'):
                travel_mode = line.split(':', 1)[1].strip().lower()
            elif line.startswith('TERRAIN:'):
                terrain = line.split(':', 1)[1].strip()
            elif line.startswith('TRANSITION_TYPE:'):
                t = line.split(':', 1)[1].strip().lower()
                if 'interrupt' in t:
                    transition_type = "interrupted"
                elif 'unclear' in t:
                    transition_type = "unclear"
                elif 'block' in t:
                    transition_type = "blocked"
            elif line.startswith('COMPLICATION:'):
                complication = line.split(':', 1)[1].strip()
            elif line.startswith('REASON:'):
                reason = line.split(':', 1)[1].strip()

        # Handle failure cases - return error instead of generating narration
        if transition_type == "unclear":
            print(f"[Transition] UNCLEAR - can't determine destination. Reason: {reason}")
            return jsonify({
                'error': 'unclear_destination',
                'message': reason or "I'm not sure where you're trying to go. Could you be more specific about your destination?",
                'transition_type': 'unclear',
            }), 400

        if transition_type == "blocked":
            print(f"[Transition] BLOCKED - unsafe to transition. Reason: {reason}")
            return jsonify({
                'error': 'blocked_transition',
                'message': reason or "The situation is too dangerous for a smooth transition. You'll need to handle this moment first.",
                'transition_type': 'blocked',
            }), 400

        # Now generate the actual transition narration
        # Genre-aware details
        genre_text = room.genre or 'drama'
        is_apocalypse = any(x in genre_text.lower() for x in ['zombie', 'apocalypse', 'outbreak', 'survival', 'horror'])

        # Genre-specific rules for the narration
        genre_rules = ""
        if is_apocalypse:
            genre_rules = """
GENRE RULES (ZOMBIE/APOCALYPSE):
- The world is DEAD. No "workers", no "activity", no "people going about their day".
- Buildings are abandoned, overgrown, decayed. Show the emptiness.
- Any signs of life should feel rare and potentially dangerous.
- Weather should feel hostile - rain is cold, sun beats down mercilessly.
- The destination should feel like a refuge, not a bustling location."""
        elif 'romance' in genre_text.lower():
            genre_rules = "\nGENRE RULES (ROMANCE): Focus on atmosphere, emotion, and sensory beauty."
        elif 'horror' in genre_text.lower():
            genre_rules = "\nGENRE RULES (HORROR): Build dread. Shadows feel wrong. Silence is oppressive."

        if transition_type == "smooth":
            narration_prompt = f"""Write a cinematic scene transition (4-6 sentences).

GENRE: {genre_text}
{genre_rules}

{player_name} {companion_text} is {travel_mode} from {current_location} to {destination}.
They pass through: {terrain}
Time: {time_of_day}, Weather: {weather}

Write a PROPER transition scene:
1. THE JOURNEY: Show them moving through the terrain. Physical details - what they walk on, what they pass, what they see/hear. Use the weather and time of day.
2. THE APPROACH: As they get closer, what do they see of the destination? Match the GENRE tone.
3. THE LANDING: Where exactly do they end up? Not mid-conversation - they've ARRIVED but haven't engaged yet. Ready for the next beat.

This is smooth - no threats or complications. But make the journey REAL, not a teleport.
Start with "You" (second person). Be vivid and grounded in physical space."""
        else:
            narration_prompt = f"""Write a cinematic scene transition (5-7 sentences).

GENRE: {genre_text}
{genre_rules}

{player_name} {companion_text} is {travel_mode} from {current_location} to {destination}.
They pass through: {terrain}
Time: {time_of_day}, Weather: {weather}

COMPLICATION: {complication}

Write a PROPER transition scene with interruption:
1. THE JOURNEY: Show them moving through the terrain. Physical details.
2. THE INTERRUPTION: Weave in the complication naturally. Something happens.
3. THE MOMENT: End at a point that requires response - don't resolve it.

Start with "You" (second person). Be vivid and grounded in physical space. Honor the GENRE."""

        # Build genre-aware system prompt
        system_prompt = f"You write cinematic scene transitions for {genre_text} stories. Grounded in physical space - show the journey, not just the destination. Second-person, vivid sensory details. The reader should feel like they walked there."
        if is_apocalypse:
            system_prompt += " This is a SURVIVAL story - the world is empty, dangerous, decayed. No normal civilization."

        # Generate the narration
        with httpx.Client(timeout=45.0) as client:
            narration_response = client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": model_to_use,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": narration_prompt}
                    ],
                    "stream": False,
                }
            )

            if narration_response.status_code != 200:
                return jsonify({'error': 'Failed to generate transition'}), 500

            narration_data = narration_response.json()
            narration_text = narration_data.get("message", {}).get("content", "").strip()

        # Update player location
        if destination and destination != "their destination":
            room.player_location = destination
            data_store.save()

        # Create the narration message
        transition_message = Message(
            id=str(uuid.uuid4())[:8],
            speaker_id="narrator",
            speaker_name="📖",
            content=narration_text,
            room_id=room_id,
            message_type="narration",
        )
        data_store.add_message(room_id, transition_message)

        print(f"[Transition] {transition_type.upper()} to {destination}")

        return jsonify({
            'id': transition_message.id,
            'speaker_id': transition_message.speaker_id,
            'speaker_name': transition_message.speaker_name,
            'content': narration_text,
            'message_type': 'narration',
            'transition_type': transition_type,
            'destination': destination,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Transition failed: {str(e)}'}), 500


@app.route('/rooms/<room_id>/dm/private', methods=['POST'])
def ask_dm_private(room_id):
    """Ask the DM a private question - only you see the answer."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    data = request.json or {}
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    partners = data_store.get_partners()
    room_partners = room.get_partners_in_room(partners)

    dm_context = _build_simple_dm_context(room, room_partners)

    # Get private history
    if room_id not in _dm_private_history:
        _dm_private_history[room_id] = []

    history = _dm_private_history[room_id]

    prompt = f"""You are the Dungeon Master. This is a PRIVATE conversation with the player.
They are asking something they do not want other characters to know about.

{dm_context}

The player privately asks: {question}

Be helpful and provide insider information if relevant. This is between you and them."""

    # Add to history
    history.append({"role": "user", "content": prompt})

    messages = [{"role": "system", "content": "You are a helpful DM in private conversation."}]
    messages.extend(history[-10:])  # Keep last 10 exchanges

    # Auto-pick first available model if configured one doesn't exist
    import httpx
    model_to_use = settings.storybuilder_model
    available_models = provider_manager.get_models_for_provider('ollama')
    if model_to_use not in available_models and available_models:
        model_to_use = available_models[0]
        print(f"[DM Private] Model '{settings.storybuilder_model}' not found, using '{model_to_use}'")

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": model_to_use,
                    "messages": messages,
                    "stream": False,
                }
            )
            if response.status_code == 200:
                response_data = response.json()
                dm_response = response_data.get("message", {}).get("content", "The DM ponders...").strip()
            else:
                error_msg = response.json().get('error', 'Unknown error') if response.text else 'No response'
                return jsonify({'error': f"Ollama error: {error_msg}"}), 500
    except httpx.ConnectError:
        return jsonify({'error': 'Cannot connect to Ollama. Is it running?'}), 500
    except Exception as e:
        print(f"[DM Private] Error: {e}")
        return jsonify({'error': f'DM unavailable: {e}'}), 500

    history.append({"role": "assistant", "content": dm_response})

    return jsonify({
        'answer': dm_response,
        'history_length': len(history)
    })


@app.route('/rooms/<room_id>/dm/private/clear', methods=['POST'])
def clear_dm_private(room_id):
    """Clear private DM conversation history."""
    if room_id in _dm_private_history:
        _dm_private_history[room_id] = []
    return jsonify({'status': 'cleared'})


# ============================================================================
# Ambient Sound - ElevenLabs Sound Effects API
# ============================================================================

_room_ambient = {}  # room_id -> {"description": str, "playing": bool}

@app.route('/sound/ambient', methods=['POST'])
def generate_ambient_sound():
    """Generate ambient sound effects using ElevenLabs Sound Effects API."""
    import requests
    import base64

    data = request.json or {}
    description = data.get('description', '').strip()
    duration = min(max(data.get('duration', 22), 0.5), 30)  # 0.5-30 seconds
    loop = data.get('loop', True)

    if not description:
        return jsonify({'error': 'No sound description provided'}), 400

    api_key = settings.elevenlabs_api_key or os.getenv('ELEVENLABS_API_KEY')
    if not api_key:
        return jsonify({'error': 'ElevenLabs API key not configured. Add ELEVENLABS_API_KEY to your .env file.'}), 400

    try:
        url = "https://api.elevenlabs.io/v1/sound-generation"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        payload = {
            "text": description,
            "duration_seconds": duration,
            "prompt_influence": 0.5,
        }

        print(f"\033[38;5;141m[AMBIENT] Generating: {description}\033[0m")
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        audio_data = response.content
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')

        print(f"\033[38;5;141m[AMBIENT] Generated {len(audio_data)} bytes\033[0m")

        return jsonify({
            'audio': audio_b64,
            'format': 'mp3',
            'description': description,
            'duration': duration,
            'loop': loop
        })

    except requests.exceptions.HTTPError as e:
        error_msg = str(e)
        try:
            error_data = e.response.json()
            error_msg = error_data.get('detail', {}).get('message', str(e))
        except:
            pass
        print(f"\033[38;5;196m[AMBIENT ERROR] {error_msg}\033[0m")
        return jsonify({'error': error_msg}), 500
    except Exception as e:
        print(f"\033[38;5;196m[AMBIENT ERROR] {e}\033[0m")
        return jsonify({'error': str(e)}), 500


@app.route('/sound/suggest', methods=['POST'])
def suggest_ambient_sound():
    """Use LLM to suggest ambient sound based on current scene."""
    data = request.json or {}
    room_id = data.get('room_id')

    if not room_id:
        return jsonify({'error': 'No room specified'}), 400

    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    # Get recent context
    recent_messages = room.messages[-5:] if room.messages else []
    context = "\n".join([f"{m.speaker_name}: {m.content[:100]}" for m in recent_messages])

    scenario = room.scenario or ""
    genre = room.genre or ""

    prompt = f"""Based on this scene, suggest a SHORT ambient sound description (10 words max) for background atmosphere.

Genre: {genre}
Scenario: {scenario}

Recent events:
{context}

Respond with ONLY the sound description, nothing else. Examples:
- "Heavy rain on a tin roof with distant thunder"
- "Quiet forest at night with crickets and owls"
- "Crowded tavern with murmured conversations and clinking glasses"
- "Tense silence with occasional wind through broken windows"
- "Crackling campfire with night insects"

Your suggestion:"""

    system = "You are a sound designer. Give a brief, evocative ambient sound description."

    room_model = getattr(room, 'room_model', '') or settings.storybuilder_model
    suggestion = _call_ollama_sync(room_model, prompt, system)

    # Clean up the response
    suggestion = suggestion.strip().strip('"').strip("'")
    if len(suggestion) > 100:
        suggestion = suggestion[:100]

    return jsonify({'suggestion': suggestion})


@app.route('/rooms/<room_id>/ambient', methods=['GET'])
def get_room_ambient(room_id):
    """Get current ambient sound for a room."""
    ambient = _room_ambient.get(room_id, {'description': '', 'playing': False})
    return jsonify(ambient)


@app.route('/rooms/<room_id>/ambient', methods=['POST'])
def set_room_ambient(room_id):
    """Set current ambient sound for a room (called by frontend when playing)."""
    data = request.json or {}
    description = data.get('description', '')
    playing = data.get('playing', False)

    _room_ambient[room_id] = {
        'description': description,
        'playing': playing
    }

    return jsonify({'success': True})


@app.route('/rooms/<room_id>/dm/ambient', methods=['POST'])
def dm_change_ambient(room_id):
    """DM secretly changes the ambient sound. Returns new sound for frontend to play."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    data = request.json or {}
    new_ambient = data.get('description', '').strip()

    if not new_ambient:
        # Auto-suggest based on scene
        recent_messages = room.messages[-5:] if room.messages else []
        context = "\n".join([f"{m.speaker_name}: {m.content[:100]}" for m in recent_messages])

        prompt = f"""Based on this scene, suggest a SHORT ambient sound (10 words max).

Genre: {room.genre or 'unknown'}
Scenario: {room.scenario or 'unknown'}

Recent:
{context}

Respond with ONLY the sound description:"""

        room_model = getattr(room, 'room_model', '') or settings.storybuilder_model
        new_ambient = _call_ollama_sync(room_model, prompt, "Brief ambient sound description only.")
        new_ambient = new_ambient.strip().strip('"').strip("'")[:100]

    if new_ambient:
        _room_ambient[room_id] = {
            'description': new_ambient,
            'playing': True
        }

    return jsonify({'description': new_ambient})


# ============================================================================
# Voice - TTS (Text to Speech) and STT (Speech to Text)
# ============================================================================

@app.route('/voice/tts', methods=['POST'])
def text_to_speech():
    """Convert text to speech using OpenAI TTS or ElevenLabs."""
    data = request.json or {}
    text = data.get('text', '').strip()
    voice = data.get('voice', 'nova')  # OpenAI voice or elevenlabs:voice_id
    partner_id = data.get('partner_id')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Get partner's voice if partner_id provided
    if partner_id:
        partner = data_store.get_partner(partner_id)
        if partner and partner.voice and partner.voice != 'none':
            voice = partner.voice

    # Check if voice is enabled globally
    if not settings.voice_enabled:
        return jsonify({'error': 'Voice is disabled in settings'}), 400

    try:
        audio_format = 'mp3'  # Default format

        if voice.startswith('piper:'):
            # Piper local TTS (free, runs locally)
            voice_name = voice.split(':', 1)[1]
            audio_data = _piper_tts(text, voice_name)
            audio_format = 'wav'  # Piper outputs WAV
        elif voice.startswith('elevenlabs:'):
            # ElevenLabs TTS
            voice_id = voice.split(':', 1)[1]
            audio_data = _elevenlabs_tts(text, voice_id)
        else:
            # OpenAI TTS
            audio_data = _openai_tts(text, voice)

        if audio_data:
            # Return as base64-encoded audio
            import base64
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            return jsonify({
                'audio': audio_b64,
                'format': audio_format,
                'voice': voice
            })
        else:
            return jsonify({'error': 'TTS generation failed'}), 500

    except Exception as e:
        print(f"[TTS Error] {e}")
        return jsonify({'error': str(e)}), 500


def _openai_tts(text: str, voice: str) -> bytes:
    """Generate speech using OpenAI TTS."""
    import openai

    api_key = settings.openai_api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not configured")

    client = openai.OpenAI(api_key=api_key)

    # Valid OpenAI voices
    valid_voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
    if voice not in valid_voices:
        voice = 'nova'  # Default fallback

    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="mp3"
    )

    return response.content


def _elevenlabs_tts(text: str, voice_id: str) -> bytes:
    """Generate speech using ElevenLabs."""
    import requests

    api_key = settings.elevenlabs_api_key or os.getenv('ELEVENLABS_API_KEY')
    if not api_key:
        raise ValueError("ElevenLabs API key not configured")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    return response.content


# ============================================================================
# Piper TTS - Free local text-to-speech
# To install: pip install piper-tts
# Models download automatically on first use
# Easy to remove: just delete this section and the piper: prefix check above
# ============================================================================

def _piper_tts(text: str, voice: str) -> bytes:
    """Generate speech using Piper (local, free TTS).

    Voice format: language-speaker-quality (e.g., en_US-amy-medium)
    Models are downloaded automatically to ~/.local/share/piper/ on first use.
    """
    import subprocess
    import tempfile
    import sys

    # Default voice if not specified or invalid
    if not voice or voice == 'default':
        voice = 'en_US-amy-medium'

    # Create temp file for output
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Try direct piper command first, then fall back to python -m piper
        piper_commands = [
            ['piper', '--model', voice, '--output_file', tmp_path],
            [sys.executable, '-m', 'piper', '--model', voice, '--output_file', tmp_path],
        ]

        result = None
        for cmd in piper_commands:
            try:
                result = subprocess.run(
                    cmd,
                    input=text.encode('utf-8'),
                    capture_output=True,
                    timeout=30  # 30 second timeout
                )
                if result.returncode == 0:
                    break
            except FileNotFoundError:
                continue

        if result is None or result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='replace') if result else 'Piper not found'
            raise RuntimeError(f"Piper failed: {error_msg}")

        # Read the generated audio
        with open(tmp_path, 'rb') as f:
            audio_data = f.read()

        return audio_data

    finally:
        # Clean up temp file
        import os
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _check_piper_available() -> bool:
    """Check if Piper TTS is installed and available."""
    import subprocess
    import sys

    # Try direct command first, then python -m
    commands = [
        ['piper', '--help'],
        [sys.executable, '-m', 'piper', '--help'],
    ]

    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=5)
            if result.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    return False


@app.route('/voice/stt', methods=['POST'])
def speech_to_text():
    """Transcribe audio to text using local Whisper first, fallback to OpenAI."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    # Save to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
        audio_file.save(f.name)
        temp_path = f.name

    # Try local Whisper first (free, no API key needed)
    model = get_local_whisper()
    if model is not None:
        try:
            print("[STT] Using local Whisper...")
            segments, info = model.transcribe(temp_path, beam_size=5, word_timestamps=True)
            text = _add_pause_punctuation(list(segments))
            os.unlink(temp_path)
            return jsonify({
                'text': text,
                'success': True,
                'provider': 'local'
            })
        except Exception as e:
            print(f"[STT] Local Whisper failed: {e}")
            # Fall through to OpenAI

    # Fallback to OpenAI API if local fails and key is available
    api_key = settings.openai_api_key or os.getenv('OPENAI_API_KEY')
    if api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            print("[STT] Using OpenAI Whisper API...")

            max_attempts = 3
            last_error = None

            for attempt in range(1, max_attempts + 1):
                try:
                    with open(temp_path, 'rb') as f:
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=f,
                            response_format="text"
                        )

                    os.unlink(temp_path)
                    return jsonify({
                        'text': transcript.strip(),
                        'success': True,
                        'provider': 'openai'
                    })

                except Exception as e:
                    last_error = e
                    print(f"[STT] OpenAI attempt {attempt}/{max_attempts} failed: {e}")
                    if attempt < max_attempts:
                        import time
                        time.sleep(0.5)

            os.unlink(temp_path)
            return jsonify({'error': f'OpenAI Whisper failed: {last_error}'}), 500

        except Exception as e:
            print(f"[STT] OpenAI error: {e}")

    # No transcription method worked
    try:
        os.unlink(temp_path)
    except:
        pass
    return jsonify({'error': 'No transcription provider available (local Whisper or OpenAI)'}), 400


@app.route('/voice/voices', methods=['GET'])
def get_available_voices():
    """Get list of available TTS voices."""
    voices = {
        'openai': [
            {'id': 'alloy', 'name': 'Alloy', 'description': 'Neutral, balanced'},
            {'id': 'echo', 'name': 'Echo', 'description': 'Warm, conversational'},
            {'id': 'fable', 'name': 'Fable', 'description': 'Expressive, dramatic'},
            {'id': 'onyx', 'name': 'Onyx', 'description': 'Deep, authoritative'},
            {'id': 'nova', 'name': 'Nova', 'description': 'Friendly, upbeat'},
            {'id': 'shimmer', 'name': 'Shimmer', 'description': 'Soft, gentle'},
        ],
        'elevenlabs': [],  # Would need to fetch from API
        'piper': [
            # Common English voices - models download automatically on first use
            {'id': 'piper:en_US-amy-medium', 'name': 'Amy (US)', 'description': 'Female, American'},
            {'id': 'piper:en_US-lessac-medium', 'name': 'Lessac (US)', 'description': 'Female, American, clear'},
            {'id': 'piper:en_US-libritts-high', 'name': 'LibriTTS (US)', 'description': 'Neutral, multi-speaker'},
            {'id': 'piper:en_GB-alba-medium', 'name': 'Alba (UK)', 'description': 'Female, British'},
            {'id': 'piper:en_GB-cori-medium', 'name': 'Cori (UK)', 'description': 'Female, British'},
        ]
    }

    # Check if ElevenLabs is configured
    if settings.elevenlabs_api_key or os.getenv('ELEVENLABS_API_KEY'):
        voices['elevenlabs_available'] = True
    else:
        voices['elevenlabs_available'] = False

    # Check if Piper is installed
    voices['piper_available'] = _check_piper_available()

    return jsonify(voices)

@app.route('/quit', methods=['POST'])
def quit_server():
    """Shutdown the server cleanly."""
    import os
    print("\n[Roundtable] Shutdown requested via /quit endpoint")
    # Use os._exit to force quit without waiting for threads/websockets
    os._exit(0)


if __name__ == '__main__':
    import socket
    import argparse

    parser = argparse.ArgumentParser(description='Roundtable server')
    parser.add_argument('--local', action='store_true', help='Run on localhost only (not visible on network)')
    parser.add_argument('--port', type=int, default=5055, help='Port to run on (default: 5055)')
    args = parser.parse_args()

    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    host = '127.0.0.1' if args.local else '0.0.0.0'
    local_ip = get_local_ip()
    port = args.port

    print("=" * 50)
    print("Roundtable")
    print("=" * 50)

    # SSL setup for HTTPS (required for microphone access on mobile/Tailscale)
    import os
    ssl_dir = os.path.join(os.path.dirname(__file__), 'ssl')
    cert_file = os.path.join(ssl_dir, 'cert.pem')
    key_file = os.path.join(ssl_dir, 'key.pem')

    use_ssl = os.path.exists(cert_file) and os.path.exists(key_file)
    protocol = "https" if use_ssl else "http"

    if args.local:
        print(f"\n{protocol}://localhost:{port}")
        print("(local only - not visible on network)")
    else:
        print(f"\n{protocol}://{local_ip}:{port}")
        print(f"{protocol}://localhost:{port}")

    if use_ssl:
        print("\n[SSL ENABLED] Microphone access should work on mobile/Tailscale")
    else:
        print("\n[WARNING] No SSL certs found - microphone won't work on mobile")
        print(f"  Expected: {ssl_dir}/cert.pem and key.pem")

    print("\nCtrl+C to stop\n")

    # use_reloader=False prevents crashes from Flask watching Python system files
    if use_ssl:
        app.run(host=host, port=port, debug=True, threaded=True, use_reloader=False,
                ssl_context=(cert_file, key_file))
    else:
        app.run(host=host, port=port, debug=True, threaded=True, use_reloader=False)
