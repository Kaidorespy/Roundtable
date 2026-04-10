"""
Roundtable Web - Multi-AI conversation orchestrator (Flask version)
"""

import os
import uuid
import json
import asyncio
import logging
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from config import DataStore, Message, Partner, Room, Settings, settings
from providers import ProviderManager
from dm_agents import get_dm_state, clear_dm_state

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
            if 'global_system_prompt' in data:
                settings.global_system_prompt = data['global_system_prompt']
            if 'saved_system_prompts' in data:
                settings.saved_system_prompts = data['saved_system_prompts']
            if 'storybuilder_model' in data:
                settings.storybuilder_model = data['storybuilder_model']
            if 'voice_enabled' in data:
                settings.voice_enabled = data['voice_enabled']
            if 'message_bubbles' in data:
                settings.message_bubbles = data['message_bubbles']
            if 'anthropic_api_key' in data:
                settings.anthropic_api_key = data['anthropic_api_key']
            if 'openai_api_key' in data:
                settings.openai_api_key = data['openai_api_key']
            if 'elevenlabs_api_key' in data:
                settings.elevenlabs_api_key = data['elevenlabs_api_key']
            if 'ollama_base_url' in data:
                settings.ollama_base_url = data['ollama_base_url']
            if 'comfy_url' in data:
                settings.comfy_url = data['comfy_url']
            if 'proxy_url' in data:
                settings.proxy_url = data['proxy_url']
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
memory_consolidator = MemoryConsolidator(memory_store, settings.ollama_base_url)

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


# ============================================================================
# Vision Model Helpers (llava fallback for non-vision models)
# ============================================================================

# Models known to have vision capabilities
VISION_MODELS = {
    # Ollama vision models
    "llava", "llava:7b", "llava:13b", "llava:34b", "llava:latest",
    "bakllava", "llava-llama3", "llava-phi3", "moondream", "minicpm-v",
    # OpenAI vision models
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-vision-preview",
}

# Providers where all models have vision
VISION_PROVIDERS = {"anthropic"}  # All Claude 3+ models have vision


def model_has_vision(provider: str, model: str) -> bool:
    """Check if a model has native vision capabilities."""
    if provider in VISION_PROVIDERS:
        return True
    # Check model name (handle versioned names like "llava:7b-v1.6")
    model_base = model.split(":")[0].lower() if ":" in model else model.lower()
    return model_base in VISION_MODELS or model.lower() in VISION_MODELS


async def describe_image_with_vision(image_data: str, image_type: str = "image/png") -> str:
    """Use the vision model (llava) to describe an image for non-vision models."""
    import httpx

    vision_model = settings.vision_model
    ollama_url = settings.ollama_base_url.rstrip("/")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{ollama_url}/api/chat",
                json={
                    "model": vision_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Describe this image in detail. Include: the subject(s), their appearance, expression, pose, clothing, setting, lighting, mood, and any notable details. Be specific and vivid.",
                            "images": [image_data]
                        }
                    ],
                    "stream": False,
                },
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("message", {}).get("content", "Unable to describe image.")
            else:
                return f"[Vision model error: {response.status_code}]"

    except Exception as e:
        return f"[Vision model unavailable: {e}]"


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
Do NOT break character. Do NOT explain. Just describe yourself as if for an artist.
IMPORTANT: Only describe physical traits that were explicitly established. Do NOT invent race, ethnicity, skin color, or demographic details that weren't specified."""

            # Generate self-description (AI call)
            update_job(job_id, 'generating_description')
            self_description = run_async(generate_response_async(partner, messages, system_prompt))

            # Check if description is actually an error message
            if self_description.startswith('[API Error:') or self_description.startswith('[Error:'):
                update_job(job_id, 'failed', error=self_description)
                return

        # Try to generate the image
        try:
            from image_gen import get_generator
            generator = get_generator()

            if not generator or not generator.is_available():
                update_job(job_id, 'completed', result={
                    'type': 'selfie',
                    'partner_id': partner.id,
                    'partner_name': partner.name,
                    'description': self_description,
                    'images': [],
                    'comfy_offline': True
                })
                return

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

            update_job(job_id, 'completed', result={
                'type': 'group_photo',
                'description': scene_description,
                'image': str(image_path) if image_path else None,
                'participants': [p.split(':')[0] for p in participants]
            })

            # Cross-post to common room if generated elsewhere
            if room_id and room_id != 'common' and image_path:
                participant_names = [p.split(':')[0] for p in participants]
                crosspost_msg = Message(
                    id=str(uuid.uuid4())[:8],
                    speaker_id='narrator',
                    speaker_name='Scene',
                    content=f"*{', '.join(participant_names)}*",
                    room_id='common',
                    image_path=str(image_path),
                )
                data_store.add_message('common', crosspost_msg)

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
        'partner_id': r.partner_id,
        'partner_ids': r.partner_ids,
        'scenario': r.scenario,
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
        } for m in room.messages],
        'partners': [{'id': p.id, 'name': p.name, 'avatar': p.avatar} for p in room_partners],
        'auto_generate': room.auto_generate,
        'auto_generate_mode': room.auto_generate_mode,
        'auto_generate_count': room.auto_generate_count,
    })


@app.route('/rooms', methods=['POST'])
def create_room():
    """Create a custom room."""
    data = request.json
    name = data.get('name', '').strip()
    partner_ids = data.get('partner_ids', [])
    scenario = data.get('scenario', '')

    if not name:
        return jsonify({'error': 'Name required'}), 400
    if not partner_ids:
        return jsonify({'error': 'Select at least one partner'}), 400

    room = data_store.create_custom_room(name, partner_ids, scenario)
    return jsonify({
        'id': room.id,
        'name': room.name,
        'partner_ids': room.partner_ids,
        'scenario': room.scenario,
    })


@app.route('/rooms/<room_id>', methods=['DELETE'])
def delete_room(room_id):
    """Delete a custom room."""
    data_store.delete_room(room_id)
    return jsonify({'status': 'deleted'})


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
        room.partner_ids = new_partner_ids

    data_store.save()

    return jsonify({
        'id': room.id,
        'name': room.name,
        'scenario': room.scenario,
        'system_prompt': room.system_prompt,
        'partner_ids': room.partner_ids,
    })


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
        new_messages = []
        for msg in room.messages:
            new_msg = Message(
                id=str(uuid.uuid4())[:8],
                speaker_id=msg.speaker_id,
                speaker_name=msg.speaker_name,
                content=msg.content,
                room_id=new_id,
                message_type=getattr(msg, 'message_type', 'message')
            )
            new_messages.append(new_msg)

        # Create the cloned room
        new_room = Room(
            id=new_id,
            name=room.name,
            is_common_room=False,
            partner_id=room.partner_id,
            partner_ids=list(room.partner_ids) if room.partner_ids else [],
            scenario=room.scenario,
            messages=new_messages,
            background_image=None,  # Clones start fresh - they can set their own background
            genre=room.genre,
            factions=room.factions,
            genre_rules=copy.deepcopy(room.genre_rules),
            dm_mode=room.dm_mode,
            dm_secret=room.dm_secret,
            auto_generate=room.auto_generate,
            auto_generate_mode=room.auto_generate_mode,
            auto_generate_count=room.auto_generate_count,
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

    if not room.is_common_room and not room.partner_ids:
        return jsonify({'error': 'Ambient mode only works in common/group rooms'}), 400

    all_partners = data_store.get_partners()

    # Filter partners by allowed providers
    allowed_providers = room.ambient_providers or ['ollama']
    eligible_partners = [
        p for p in all_partners
        if p.provider.lower() in allowed_providers
    ]

    if not eligible_partners:
        return jsonify({'error': 'No eligible partners (check provider settings)'}), 400

    # Pick a random partner
    import random
    partner = random.choice(eligible_partners)

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

    # Build system prompt
    system = partner.get_full_context(
        all_partners if room.is_common_room else room.get_partners_in_room(all_partners),
        settings.user_name,
        settings.global_system_prompt,
        user_physical_description=settings.user_physical_description
    )

    # Add ambient instruction
    starter = random.choice(starters)
    if len(room.messages) < 3:
        # Room is quiet - encourage starting something
        system += f"\n\nThe common room is quiet. You walk in and {starter}. Be natural and in-character. Start a conversation or make an observation."
    else:
        # Conversation happening - join naturally
        system += f"\n\nYou're in the common room. {starter.capitalize()}. Respond naturally to what's happening or take the conversation in an interesting direction."

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
    room = data_store.get_room(room_id)

    # Clear DM canon for characters in this room
    if room:
        partners = data_store.get_partners()
        room_partners = room.get_partners_in_room(partners)
        for partner in room_partners:
            if hasattr(partner, 'dm_canon') and partner.dm_canon:
                print(f"[Clear Room] Clearing DM canon for {partner.name}")
                partner.dm_canon = []
        data_store.save()

    data_store.clear_room(room_id)
    return jsonify({'status': 'cleared'})


@app.route('/rooms/<room_id>/pin', methods=['POST'])
def toggle_room_pin(room_id):
    """Toggle pinned status for a room."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    data = request.json
    room.pinned = data.get('pinned', not room.pinned)
    data_store.save_rooms()

    return jsonify({'status': 'ok', 'pinned': room.pinned})


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
                    partner = data_store.get_partner(partner_id)
                    for img in partner_dir.glob("*.png"):
                        if not is_excluded(img.name):
                            images.append({
                                'path': str(img),
                                'type': 'selfie',
                                'partner_id': partner_id,
                                'partner_name': partner.name if partner else partner_id,
                                'mtime': img.stat().st_mtime,
                                'prompt': get_prompt(img)
                            })

        # Get all scenes
        if scenes_dir.exists():
            for img in scenes_dir.glob("*.png"):
                images.append({
                    'path': str(img),
                    'type': 'scene',
                    'mtime': img.stat().st_mtime,
                    'prompt': get_prompt(img)
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
                        'prompt': get_prompt(img)
                    })

        # Get scenes for this room only (scenes are room-specific)
        if scenes_dir.exists():
            for img in scenes_dir.glob(f"scene_{room_id}_*.png"):
                images.append({
                    'path': str(img),
                    'type': 'scene',
                    'mtime': img.stat().st_mtime,
                    'prompt': get_prompt(img)
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
                            'prompt': get_prompt(img)
                        })

        # Get scenes for this room
        if scenes_dir.exists():
            for img in scenes_dir.glob(f"scene_{room_id}_*.png"):
                images.append({
                    'path': str(img),
                    'type': 'scene',
                    'mtime': img.stat().st_mtime,
                    'prompt': get_prompt(img)
                })

    # Sort by modification time (newest first) and limit
    images.sort(key=lambda x: x['mtime'], reverse=True)
    images = images[:limit]

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
        model=data.get('model', 'llama3.2'),
        avatar=data.get('avatar', '🤖'),
        custom_system_prompt=data.get('custom_system_prompt'),
        memory_mode=data.get('memory_mode', 'local'),
        voice=data.get('voice', 'none'),
        loras=data.get('loras', []),
    )

    data_store.add_partner(partner)

    return jsonify({
        'id': partner.id,
        'name': partner.name,
        'avatar': partner.avatar,
        'provider': partner.provider,
        'model': partner.model,
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
    partner.avatar = data.get('avatar', partner.avatar)
    partner.custom_system_prompt = data.get('custom_system_prompt', partner.custom_system_prompt)
    partner.memory_mode = data.get('memory_mode', partner.memory_mode)
    partner.voice = data.get('voice', partner.voice)
    if 'loras' in data:
        partner.loras = data['loras']

    data_store.update_partner(partner)

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

        return jsonify({
            'id': new_partner.id,
            'name': new_partner.name,
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

    if not message_content:
        return jsonify({'error': 'No message'}), 400
    if not room_id:
        return jsonify({'error': 'No room specified'}), 400

    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    # Create user message (but don't save yet - only save after successful response)
    user_message = Message(
        id=str(uuid.uuid4())[:8],
        speaker_id="user",
        speaker_name=settings.user_name,
        content=message_content,
        room_id=room_id,
    )

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
        return jsonify({
            'user_message': {'id': user_message.id, 'speaker_name': user_message.speaker_name, 'content': user_message.content},
            'needs_responder': True,
        })

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
            else:
                content = f"{msg.speaker_name}: {msg.content}"
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": role, "content": msg.content})

    # Add the current user message (not yet saved to room)
    if is_multi_party:
        messages.append({"role": "user", "content": f"{{{settings.user_name}}}: {message_content}"})
    else:
        messages.append({"role": "user", "content": message_content})

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
        system += f"\n\nNow respond as {partner.name}. Do not prefix your response with your name."
    elif room.partner_ids:
        room_partners = room.get_partners_in_room(all_partners)
        system = partner.get_full_context(room_partners, settings.user_name, base_system_prompt, user_physical_description=settings.user_physical_description)
        # Only inject scenario for first 10 turns - after that, context should carry it
        if room.scenario and len(room.messages) < 10:
            system += f"\n\n---\nSCENARIO:\n{room.scenario}\n---"
        system += mood_context
        system += f"\n\nNow respond as {partner.name}. Do not prefix your response with your name."
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

        # Success! Now save both the user message and response
        data_store.add_message(room_id, user_message)

        response_message = Message(
            id=str(uuid.uuid4())[:8],
            speaker_id=partner.id,
            speaker_name=partner.name,
            content=response_text,
            room_id=room_id,
        )
        data_store.add_message(room_id, response_message)

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

        return jsonify({
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
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/respond', methods=['POST'])
def respond():
    """Have a specific partner respond (for common room clicks)."""
    data = request.json
    room_id = data.get('room_id')
    partner_id = data.get('partner_id')

    if not room_id or not partner_id:
        return jsonify({'error': 'Missing room_id or partner_id'}), 400

    room = data_store.get_room(room_id)
    partner = data_store.get_partner(partner_id)

    if not room or not partner:
        return jsonify({'error': 'Room or partner not found'}), 404

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
        system += f"\n\nNow respond as {partner.name}. Do not prefix your response with your name."
    elif room.partner_ids:
        room_partners = room.get_partners_in_room(all_partners)
        system = partner.get_full_context(room_partners, settings.user_name, base_system_prompt, user_physical_description=settings.user_physical_description)
        # Only inject scenario for first 10 turns - after that, context should carry it
        if room.scenario and len(room.messages) < 10:
            system += f"\n\n---\nSCENARIO:\n{room.scenario}\n---"
        system += mood_context
        system += f"\n\nNow respond as {partner.name}. Do not prefix your response with your name."
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
                has_vision = model_has_vision(partner.provider, partner.model)

                if has_vision:
                    # Model has vision - send image directly
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
                else:
                    # No vision - use llava to describe the image first
                    print(f"[Respond] Partner {partner.name} uses {partner.model} (no vision) - using {settings.vision_model} fallback")
                    image_description = run_async(describe_image_with_vision(image_data, "image/png"))
                    messages.append({
                        "role": "user",
                        "content": f"({settings.user_name} shared an image with the room)\n\nImage description: {image_description}"
                    })

                # Add instruction about the image
                system += f"\n\n{settings.user_name} has shared an image with the room. You can see it and react naturally if you want, or acknowledge it briefly and continue the conversation."
                # Mark as seen
                _mark_pending_image_seen(room_id, partner_id)
            except Exception as e:
                print(f"[Respond] Failed to load pending image: {e}")

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
            'id': response_message.id,
            'speaker_id': partner.id,
            'speaker_name': partner.name,
            'avatar': partner.avatar,
            'avatar_image': partner.avatar_image,
            'content': response_text,
        })
    except Exception as e:
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
        'global_system_prompt': settings.global_system_prompt,
        'favorite_prompts': settings.favorite_prompts,
        'saved_system_prompts': settings.saved_system_prompts,
        'storybuilder_model': settings.storybuilder_model,
        'voice_enabled': settings.voice_enabled,
        'message_bubbles': settings.message_bubbles,
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

    return jsonify({
        'user_name': settings.user_name,
        'user_physical_description': settings.user_physical_description,
        'user_avatar': settings.user_avatar,
        'global_system_prompt': settings.global_system_prompt,
        'voice_enabled': settings.voice_enabled,
        'message_bubbles': settings.message_bubbles,
        'openai_available': bool(openai_api_key),  # For mic button visibility
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
        'comfy_url': settings.comfy_url,
        'proxy_url': settings.proxy_url,
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
        if 'message_bubbles' in data:
            settings.message_bubbles = data['message_bubbles']
        if 'anthropic_api_key' in data and data['anthropic_api_key']:
            settings.anthropic_api_key = data['anthropic_api_key']
        if 'openai_api_key' in data and data['openai_api_key']:
            settings.openai_api_key = data['openai_api_key']
        if 'elevenlabs_api_key' in data and data['elevenlabs_api_key']:
            settings.elevenlabs_api_key = data['elevenlabs_api_key']
        if 'ollama_base_url' in data and data['ollama_base_url']:
            settings.ollama_base_url = data['ollama_base_url'].rstrip('/')
        if 'comfy_url' in data and data['comfy_url']:
            settings.comfy_url = data['comfy_url'].rstrip('/')
        if 'proxy_url' in data:
            settings.proxy_url = data['proxy_url'].strip()

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
            'user_physical_description': settings.user_physical_description,
            'global_system_prompt': settings.global_system_prompt,
            'favorite_prompts': settings.favorite_prompts,
            'saved_system_prompts': settings.saved_system_prompts,
            'storybuilder_model': settings.storybuilder_model,
            'voice_enabled': settings.voice_enabled,
            'message_bubbles': settings.message_bubbles,
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
            'comfy_url': settings.comfy_url,
            'proxy_url': settings.proxy_url,
            'custom_checkpoint': settings.custom_checkpoint,
            'custom_checkpoint_type': settings.custom_checkpoint_type,
        }
        settings_file.write_text(json.dumps(settings_data, indent=2))

        # Reinit providers to pick up any changes (API keys, proxy, etc.)
        provider_manager.reinit_providers()

        result = {'status': 'ok'}
        if settings_warning:
            result['warning'] = settings_warning
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/test-proxy', methods=['POST'])
def test_proxy():
    """Test a proxy connection and return the external IP."""
    import requests

    try:
        data = request.json
        proxy_url = data.get('proxy_url', '').strip()

        if not proxy_url:
            return jsonify({'success': False, 'error': 'No proxy URL provided'})

        # Configure proxy
        proxies = {
            'http': proxy_url,
            'https': proxy_url,
        }

        # Use a simple IP check service
        response = requests.get(
            'https://api.ipify.org?format=json',
            proxies=proxies,
            timeout=10
        )

        if response.status_code == 200:
            ip_data = response.json()
            return jsonify({'success': True, 'ip': ip_data.get('ip', 'Unknown')})
        else:
            return jsonify({'success': False, 'error': f'HTTP {response.status_code}'})

    except requests.exceptions.ProxyError as e:
        return jsonify({'success': False, 'error': 'Proxy connection failed'})
    except requests.exceptions.Timeout:
        return jsonify({'success': False, 'error': 'Connection timed out'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/providers', methods=['GET'])
def get_providers():
    """Get available providers and their models."""
    providers = provider_manager.get_available_providers()
    result = {}
    for p in providers:
        result[p] = provider_manager.get_models_for_provider(p)
    return jsonify(result)


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

    room_id = request.args.get('room_id', 'common')
    favorites = _load_favorites(room_id)
    images = []

    for path in favorites:
        # Verify file still exists
        if Path(path).exists():
            images.append({'path': path})

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

        # Check if partner's model has native vision
        has_vision = model_has_vision(partner.provider, partner.model)

        if has_vision:
            # Model has vision - send image directly
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
        else:
            # No vision - use llava to describe the image first
            print(f"[share-image] Partner {partner.name} uses {partner.model} (no vision) - using {settings.vision_model} fallback")
            image_description = run_async(describe_image_with_vision(image_data, image_type))

            # Send the description as text instead of the raw image
            messages.append({
                "role": "user",
                "content": f"[{settings.user_name} shares an image with you]\n\nImage description: {image_description}\n\n{user_message}"
            })

        # Get full partner context (including knowledge of other participants)
        all_partners = data_store.get_partners()
        room_partners = room.get_partners_in_room(all_partners)
        system = partner.get_full_context(
            room_partners,
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
            # Clear DM canon for characters in this room
            if room:
                partners = data_store.get_partners()
                room_partners = room.get_partners_in_room(partners)
                for p in room_partners:
                    if hasattr(p, 'dm_canon') and p.dm_canon:
                        print(f"[/clear] Clearing DM canon for {p.name}")
                        p.dm_canon = []
                data_store.save()

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

                # Get full partner context for richer reaction
                all_partners = data_store.get_partners()
                room_partners = room.get_partners_in_room(all_partners) if room else [partner]
                reaction_system = partner.get_full_context(
                    room_partners,
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

        # Find a partner to generate the scene description (prefer room's partner with working model)
        scene_partner = None
        candidates = []

        if room.partner_id:
            p = data_store.get_partner(room.partner_id)
            if p:
                candidates = [p]
        elif room.partner_ids:
            candidates = [data_store.get_partner(pid) for pid in room.partner_ids if data_store.get_partner(pid)]
        else:
            # Common room - all partners are candidates
            candidates = data_store.get_partners()

        # Pick a partner with a working model (prefer Anthropic/OpenAI, then check Ollama availability)
        available_ollama = provider_manager.get_models_for_provider('ollama')
        for p in candidates:
            if p.provider in ('anthropic', 'openai'):
                scene_partner = p
                break
            elif p.provider == 'ollama' and p.model in available_ollama:
                scene_partner = p
                break

        # Fallback: just use first candidate and hope for the best
        if not scene_partner and candidates:
            scene_partner = candidates[0]

        if not scene_partner:
            return jsonify({'type': 'error', 'message': 'No partner available to describe the scene'})

        try:
            from image_gen import get_generator

            # Check for override prompt (for regeneration)
            override_prompt = data.get('prompt')

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

            # Try to generate the image
            try:
                generator = get_generator()

                if not generator or not generator.is_available():
                    return jsonify({
                        'type': 'info',
                        'message': f"ComfyUI not running. Scene prompt:\n\n{scene_prompt}"
                    })

                # Use captured_loras from request if provided (for regeneration), else use room settings
                captured_loras = data.get('captured_loras')
                if captured_loras is None and room and room.loras:
                    captured_loras = [l for l in room.loras if l.get('enabled')]

                # Generate scene (landscape - swaps preset width/height)
                # Prepend "no people" to avoid random figures in scene images
                scene_image_prompt = f"no people, no humans, no figures, {scene_prompt}"
                image_path = generator.generate_scene(
                    prompt=scene_image_prompt,
                    room_id=room_id,
                    captured_loras=captured_loras
                )

                # Cross-post to common room if generated elsewhere
                if room_id and room_id != 'common' and image_path:
                    crosspost_msg = Message(
                        id=str(uuid.uuid4())[:8],
                        speaker_id='narrator',
                        speaker_name='Scene',
                        content=f"*a scene from {room.name}*",
                        room_id='common',
                        image_path=str(image_path),
                    )
                    data_store.add_message('common', crosspost_msg)

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

        info_parts.append("\n".join(parts))
    return "\n\n".join(info_parts)


def _build_simple_dm_context(room, room_partners: list) -> str:
    """Build simple DM context - just characters and scenario."""
    sections = []

    # Character info
    char_info = _build_dm_character_info(room_partners)
    if char_info:
        sections.append(f"=== CHARACTERS ===\n{char_info}")

    # Scenario
    if room.scenario:
        sections.append(f"=== SCENARIO ===\n{room.scenario}")

    # Genre
    if hasattr(room, 'genre') and room.genre:
        sections.append(f"Genre: {room.genre}")

    return "\n\n".join(sections)


def _extract_dm_canon(question: str, dm_response: str, room_partners: list) -> dict:
    """
    Extract canon facts from a DM interaction and attribute them to characters.
    Returns {partner_id: [list of facts]} for characters mentioned.
    """
    import httpx

    if not room_partners:
        return {}

    # Build character list for the extraction prompt
    char_list = "\n".join([f"- {p.name} (id: {p.id})" for p in room_partners])

    extraction_prompt = f"""Analyze this DM (Dungeon Master) interaction and extract any facts that were established about specific characters.

CHARACTERS IN SCENE:
{char_list}

PLAYER QUESTION: {question}

DM RESPONSE: {dm_response}

Extract ONLY concrete facts that were established about specific characters (not general world facts).
For each character mentioned, list the specific facts revealed about them.

Respond in this exact JSON format (no other text):
{{
  "character_id_here": ["fact 1", "fact 2"],
  "another_character_id": ["fact about them"]
}}

If no character-specific facts were established, respond with: {{}}

Be specific and concise. Only include facts directly stated, not implications."""

    model_to_use = settings.storybuilder_model
    available_models = provider_manager.get_models_for_provider('ollama')
    if model_to_use not in available_models and available_models:
        model_to_use = available_models[0]

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": model_to_use,
                    "messages": [
                        {"role": "system", "content": "You extract structured information from text. Always respond with valid JSON only."},
                        {"role": "user", "content": extraction_prompt}
                    ],
                    "stream": False,
                }
            )
            if response.status_code == 200:
                response_data = response.json()
                content = response_data.get("message", {}).get("content", "{}").strip()

                # Try to parse JSON from the response
                import json
                # Handle potential markdown code blocks
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()

                try:
                    extracted = json.loads(content)
                    print(f"[DM Canon] Extracted: {extracted}")
                    return extracted
                except json.JSONDecodeError as e:
                    print(f"[DM Canon] Failed to parse JSON: {e}")
                    print(f"[DM Canon] Raw content: {content}")
                    return {}
            else:
                print(f"[DM Canon] Ollama error: {response.status_code}")
                return {}
    except Exception as e:
        print(f"[DM Canon] Error extracting canon: {e}")
        return {}


def _apply_dm_canon(extracted_facts: dict):
    """Apply extracted DM canon facts to the relevant characters."""
    if not extracted_facts:
        return

    for partner_id, facts in extracted_facts.items():
        if not facts:
            continue

        partner = data_store.get_partner(partner_id)
        if partner:
            # Initialize dm_canon if needed
            if not hasattr(partner, 'dm_canon') or partner.dm_canon is None:
                partner.dm_canon = []

            # Add new facts (avoid duplicates)
            for fact in facts:
                if fact and fact not in partner.dm_canon:
                    partner.dm_canon.append(fact)
                    print(f"[DM Canon] Added to {partner.name}: {fact}")

            data_store.save()


@app.route('/rooms/<room_id>/dm', methods=['POST'])
def ask_dm_public(room_id):
    """Ask the DM a public question - everyone sees the Q&A."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    data = request.json or {}
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    partners = data_store.get_partners()
    room_partners = room.get_partners_in_room(partners)

    # Build simple context
    dm_context = _build_simple_dm_context(room, room_partners)

    # Get recent conversation
    recent_messages = room.messages[-10:] if room.messages else []
    conversation_context = "\n".join([
        f"{m.speaker_name}: {m.content[:200]}" for m in recent_messages
    ])

    prompt = f"""You are the Dungeon Master / Game Master for this roleplay scene.
Your role is to make authoritative decisions about the world, NPCs, and reality.
You are NOT a character - you are the arbiter of what is true in this world.

{dm_context}

Recent conversation:
{conversation_context}

The player asks: {question}

Provide a clear, definitive answer about what is true in this world.
Be decisive - don't hedge or say "it could be either way."
You can introduce surprises, twists, or complications.
Keep your answer concise (1-3 sentences usually).
Write in a narrative/descriptive style, not as dialogue."""

    system = "You are a Dungeon Master making authoritative decisions about the game world. Be decisive, creative, and impartial."

    # Call the model - auto-pick first available if configured one doesn't exist
    import httpx
    model_to_use = settings.storybuilder_model
    available_models = provider_manager.get_models_for_provider('ollama')
    if model_to_use not in available_models and available_models:
        model_to_use = available_models[0]
        print(f"[DM Public] Model '{settings.storybuilder_model}' not found, using '{model_to_use}'")

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
                dm_response = response_data.get("message", {}).get("content", "The DM ponders...").strip()
            else:
                error_msg = response.json().get('error', 'Unknown error') if response.text else 'No response'
                return jsonify({'error': f"Ollama error: {error_msg}"}), 500
    except httpx.ConnectError:
        return jsonify({'error': 'Cannot connect to Ollama. Is it running?'}), 500
    except Exception as e:
        print(f"[DM Public] Error: {e}")
        return jsonify({'error': f'DM unavailable: {e}'}), 500

    # Extract and apply canon facts from this DM interaction
    extracted = _extract_dm_canon(question, dm_response, room_partners)
    _apply_dm_canon(extracted)

    # Create the DM message
    dm_message = Message(
        id=str(uuid.uuid4()),
        speaker_id="dm",
        speaker_name="DM",
        content=dm_response,
        timestamp=datetime.now().isoformat(),
        message_type="dm_response",
        metadata={"question": question}
    )
    room.messages.append(dm_message)
    data_store.save()

    # Also create the question message
    question_message = Message(
        id=str(uuid.uuid4()),
        speaker_id="player",
        speaker_name=settings.user_name,
        content=question,
        timestamp=datetime.now().isoformat(),
        message_type="dm_question",
    )

    return jsonify({
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
    })


@app.route('/rooms/<room_id>/inciting-incident', methods=['POST'])
def generate_inciting_incident(room_id):
    """Generate a dramatic inciting incident for the story."""
    room = data_store.get_room(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404

    partners = data_store.get_partners()
    room_partners = room.get_partners_in_room(partners)

    if not room_partners:
        return jsonify({'error': 'No characters in this room'}), 400

    # Build context about who's in the room
    character_list = "\n".join([
        f"- {p.name}: {p.character_description[:200] if p.character_description else 'No description'}"
        for p in room_partners
    ])

    # Get recent conversation for context
    recent_messages = room.messages[-10:] if room.messages else []
    conversation_context = "\n".join([
        f"{m.speaker_name}: {m.content[:150]}" for m in recent_messages
    ]) if recent_messages else "(No conversation yet)"

    # Build the scenario context
    scenario_text = room.scenario if room.scenario else "A group conversation"

    prompt = f"""You are a dramatic narrator generating an INCITING INCIDENT - something that happens in the world that forces the characters to react.

SETTING: {scenario_text}

CHARACTERS PRESENT:
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

    # Call the model (using storybuilder_model setting)
    # Auto-pick first available model if configured one doesn't exist
    import httpx
    model_to_use = settings.storybuilder_model
    available_models = provider_manager.get_models_for_provider('ollama')
    if model_to_use not in available_models and available_models:
        model_to_use = available_models[0]
        print(f"[Inciting Incident] Model '{settings.storybuilder_model}' not found, using '{model_to_use}'")
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

    # Create the incident message
    incident_message = Message(
        id=str(uuid.uuid4())[:8],
        speaker_id="narrator",
        speaker_name="Narrator",
        content=incident_text,
        room_id=room_id,
        message_type="inciting_incident",
    )
    data_store.add_message(room_id, incident_message)

    return jsonify({
        'id': incident_message.id,
        'speaker_id': incident_message.speaker_id,
        'speaker_name': incident_message.speaker_name,
        'content': incident_text,
        'message_type': 'inciting_incident',
    })


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

    # Extract and apply canon facts from this DM interaction
    extracted = _extract_dm_canon(question, dm_response, room_partners)
    _apply_dm_canon(extracted)

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

        # Piper disabled for v1 - Windows support incomplete
        # if voice.startswith('piper:'):
        #     voice_name = voice.split(':', 1)[1]
        #     audio_data = _piper_tts(text, voice_name)
        #     audio_format = 'wav'  # Piper outputs WAV
        if voice.startswith('elevenlabs:'):
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
# Piper TTS - Disabled for v1 (Windows support incomplete)
# Uncomment in future version when cross-platform issues are resolved
# ============================================================================

# def _piper_tts(text: str, voice: str) -> bytes:
#     """Generate speech using Piper (local, free TTS)."""
#     pass

def _check_piper_available() -> bool:
    """Piper disabled for v1."""
    return False


@app.route('/voice/stt', methods=['POST'])
def speech_to_text():
    """Transcribe audio to text using OpenAI Whisper."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    api_key = settings.openai_api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        return jsonify({'error': 'OpenAI API key not configured'}), 400

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        # Save to temp file (Whisper API needs a file)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
            audio_file.save(f.name)
            temp_path = f.name

        # Retry up to 3 times - Whisper can be flaky
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

                # Success - clean up and return
                os.unlink(temp_path)
                return jsonify({
                    'text': transcript.strip(),
                    'success': True
                })

            except Exception as e:
                last_error = e
                print(f"[STT] Attempt {attempt}/{max_attempts} failed: {e}")
                if attempt < max_attempts:
                    import time
                    time.sleep(0.5)  # Brief pause before retry

        # All attempts failed - clean up and return error
        os.unlink(temp_path)
        return jsonify({'error': f'Whisper failed after {max_attempts} attempts: {last_error}'}), 500

    except Exception as e:
        print(f"[STT Error] {e}")
        return jsonify({'error': str(e)}), 500


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
        # Piper disabled for v1
        'piper': []
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

    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    local_ip = get_local_ip()
    port = 5055

    print("=" * 50)
    print("Roundtable")
    print("=" * 50)
    print(f"\nhttp://{local_ip}:{port}")
    print(f"http://localhost:{port}")
    print("\nCtrl+C to stop\n")

    # use_reloader=False prevents crashes from Flask watching Python system files
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True, use_reloader=False)
