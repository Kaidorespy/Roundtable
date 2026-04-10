# Roundtable V1

Multi-character AI companion chat with memory, image generation, and voice.

## Features

### Characters
- Create AI companions with custom personalities
- Per-character settings: name, avatar (emoji or generated image), description, physical appearance
- Provider choice per character: Anthropic Claude, OpenAI, or local Ollama
- Custom system prompts per character (or use global)
- Hidden traits for DM: secret, wound, want, fear, skill, honesty level

### Rooms
- **Private rooms**: 1-on-1 conversations with a single companion
- **Common room**: All companions present, click avatar to select who responds
- **Custom group rooms**: Select specific companions for multi-party conversations
- Room settings: scenario, mood, genre, factions, genre rules (magic/supernatural/tech toggles)
- Draft persistence: unsent text survives navigation and refresh

### Memory System
Three modes per character:
- **Global**: Memories persist across all rooms
- **Local**: Per-room memory only
- **None**: Fresh conversation every time

Memory consolidation runs automatically:
- Texture mutation (relationship vibe/trajectory)
- Anchor management (load-bearing facts)
- Resonance tracking (recurring themes)
- Sediment compression (old memories archived)

### Image Generation
Requires ComfyUI backend.
- **Selfies**: Character portrait based on conversation context
- **Scene images**: Environment/setting for the room
- **Group photos**: Multiple characters together
- Per-character LoRA support with weights
- Model presets: Illustrious, Flux, Pony
- Background job queue (non-blocking)

### Voice
- **TTS**: OpenAI voices or ElevenLabs
- **STT**: OpenAI Whisper for voice input
- Per-character voice assignment

### DM System (Beta)
Public and private DM queries for world arbitration.
- **Inventory Keeper**: Tracks items mentioned in conversation
- **Dramaturge**: Craft advice on mood, tension, pacing
- **Inciting Incidents**: Generate dramatic events

## Installation

```bash
pip install -r requirements.txt
python main.py
```

Open http://localhost:5000

### Optional: ComfyUI (image generation)
See ComfyUI documentation. Set URL in settings.

## Configuration

Settings stored in `~/.roundtable/`:
- `partners.json` - Character data
- `rooms.json` - Room conversations
- `memory/` - Memory files
- `avatars/` - Generated images

Environment variables or settings panel:
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `ELEVENLABS_API_KEY` (optional)

## API

### Chat
- `POST /chat` - Send message, get response
- `POST /respond` - Trigger specific partner response

### Partners
- `GET /partners` - List all
- `POST /partners` - Create
- `PUT /partners/<id>` - Update
- `DELETE /partners/<id>` - Delete

### Rooms
- `GET /rooms` - List all
- `GET /rooms/<id>` - Get room with messages
- `POST /rooms/<id>/clear` - Clear messages

### DM
- `POST /rooms/<id>/dm` - Public query
- `POST /rooms/<id>/dm/private` - Private query
- `POST /rooms/<id>/inciting-incident` - Generate event

### Voice
- `POST /voice/tts` - Text to speech
- `POST /voice/stt` - Speech to text
- `GET /voice/voices` - Available voices

### Images
- `POST /partner/<id>/avatar` - Generate portrait
- `POST /rooms/<id>/background` - Generate scene
- `GET /partner/<id>/gallery` - Browse images

## Tech Stack

- Flask backend
- Vanilla JS frontend
- SQLite-compatible JSON persistence
- ThreadPoolExecutor for background jobs

## What's Not in V1

V1 is focused. These are planned for future versions:
- StoryBuilder (NPC generation/agency)
- World Walkers (cross-world NPCs)
- Story Daemon (background world progression)
- Multiplayer (P2P sync)
- Autopilot (characters act when you're away)
- Combat system
- Fatigue/inventory tracking
- Consequence engine

## Credits

Built by Casey and Claude.
