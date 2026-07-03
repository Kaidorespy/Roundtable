<img width="1536" height="1024" alt="ChatGPT Image Apr 1, 2026, 01_47_04 PM" src="https://github.com/user-attachments/assets/f7083a37-f2b4-400d-9731-d6abb409f340" />


# Roundtable

**Collaborative storytelling with AI companions who remember, react, and feel real.**

Roundtable is a multi-character roleplay engine where you don't just chat with AI—you live stories with them. Characters have persistent memory, hidden depths, and the capacity to surprise you. A DM system arbitrates reality. Image generation brings scenes to life. Voice gives characters breath.

This isn't a chatbot. It's a living room where stories happen.

---

## The Idea

Most AI chat apps treat characters as stateless response machines. Send message, get response, repeat. The character has no memory of last week, no opinion about what happened yesterday, no arc.

Roundtable is different:

- **Characters remember.** Not just chat history—structured memory that evolves. Relationships develop texture. Old conversations become sediment that shapes new ones.
- **The world has rules.** A DM system knows what's in your inventory, how tired you are, what genre you're in. Try to cast a spell in a zombie apocalypse? Your fingers wiggle. Nothing happens.
- **Stories emerge.** With multi-character rooms, characters respond to each other, not just you. Dynamics form. Tensions build. The conversation goes places you didn't plan.

---

## What's Built

### Characters
- Custom AI companions with personalities, backstories, physical descriptions
- Per-character model choice: Claude (Anthropic), GPT (OpenAI), or local models via Ollama
- Hidden traits only the DM knows: secrets, wounds, wants, fears
- Voice output via OpenAI TTS, ElevenLabs, or Piper (free/local)
- Generated portraits with per-character LoRA support

### Memory System
Three modes per character:
- **Global**: Memories persist across all rooms
- **Local**: Per-room memory only
- **None**: Fresh every time

Memory isn't just "remember the last N messages." It's structured:
- **Anchors**: Load-bearing facts that define the relationship
- **Texture**: The vibe, the trajectory, how things feel between you
- **Resonance**: Recurring themes that keep surfacing
- **Sediment**: Old memories compressed but not forgotten

When context gets long, consolidation runs automatically—preserving what matters, archiving what doesn't.

### Rooms
- **Private rooms**: 1-on-1 with a single companion
- **Common room**: Everyone present, click to choose who speaks
- **Custom groups**: Select specific characters for multi-party scenes

Room settings include scenario, mood, genre, and genre rules (magic yes/no, supernatural yes/no, technology level).

### DM System
An impartial arbiter who knows the world state:
- **Public queries**: Ask the DM something, everyone sees the answer
- **Private queries**: Ask something only you see (what's Marcus hiding?)
- **Inventory tracking**: Items mentioned in narrative are parsed and tracked
- **Inciting incidents**: Generate dramatic events to shake up a stale scene

The DM doesn't tell you "no." The DM tells you what happens when you try.

### Image Generation
Requires ComfyUI backend:
- **Selfies**: Character portraits based on conversation context
- **Scene images**: Environmental shots for the current moment
- **Group photos**: Multiple characters together
- Background job queue (non-blocking, generates while you chat)
- Model presets: Illustrious, Flux, Pony
- Per-character LoRA weights for visual consistency

### Voice
- **Text-to-speech**: Characters speak aloud (OpenAI, ElevenLabs, or Piper)
- **Speech-to-text**: Voice input via Whisper
- Per-character voice assignment
- Toggle globally or per-character

### Interface
- Whispers: Private fourth-wall-breaking asides with characters
- Draft persistence: Unsent text survives navigation and refresh
- Chain mode: Characters respond to each other, not just you
- Ambient mode: Characters continue when you're not actively participating
- Mobile-friendly with collapsible UI

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ROUNDTABLE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    DM CONTEXT LAYER                        │ │
│  │  Every DM query receives:                                  │ │
│  │  - Character secrets/psychology                            │ │
│  │  - Current inventories                                     │ │
│  │  - World state (time, weather, mood)                       │ │
│  │  - Genre rules (what's possible here?)                     │ │
│  │  - Relationship dynamics                                   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ MEMORY   │ │INVENTORY │ │  IMAGE   │ │      VOICE       │   │
│  │ SYSTEM   │ │ TRACKER  │ │   GEN    │ │                  │   │
│  │          │ │          │ │          │ │  TTS + STT       │   │
│  │ Anchors  │ │ Narrative│ │ ComfyUI  │ │  OpenAI/11Labs   │   │
│  │ Texture  │ │ parsing  │ │ LoRAs    │ │  Piper           │   │
│  │ Sediment │ │          │ │          │ │                  │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    PROVIDER LAYER                          │ │
│  │  Anthropic (Claude) │ OpenAI (GPT) │ Ollama (Local)       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Tech Stack
- **Backend**: Flask (Python)
- **Frontend**: Vanilla JS (no framework)
- **Persistence**: JSON files in `~/.roundtable/`
- **Background jobs**: ThreadPoolExecutor
- **AI**: Multi-provider (Anthropic, OpenAI, Ollama)
- **Images**: ComfyUI integration
- **Voice**: OpenAI TTS/Whisper, ElevenLabs, Piper

---

## Running It

### Prerequisites
- Python 3.10+
- At least one AI provider:
  - Ollama (free, local)
  - Anthropic API key
  - OpenAI API key

### Install
```bash
cd RoundtableV2
pip install -r requirements.txt
python launcher.py
```

Roundtable opens in your browser at:

```text
http://127.0.0.1:5055
```

Keep the terminal window open while using Roundtable. Closing it stops the server.

You can also run:

```bash
python main.py
```

`main.py` delegates to the same launcher.

### Packaged Windows Build

If you have the packaged build, run:

```text
dist_slim\Roundtable.exe
```

It starts the same local server and opens `http://127.0.0.1:5055`.

### Optional: Image Generation
Install ComfyUI separately, configure URL in settings.

### Optional: Free Local Voice
```bash
pip install piper-tts
```
Models download automatically on first use.

---

## Where It's Going

Roundtable is built with a larger vision in mind. Not all of this exists yet, but the architecture supports it:

### Story Daemon
A background thread that keeps the world alive when you're not looking. Time passes. Characters make decisions. Events unfold. When you come back, you read your journal to see what happened.

### Autopilot
When you step away, your character doesn't vanish—they go on autopilot. Based on alignment and personality, they make decisions, take actions, survive (or don't). The world keeps turning.

### Consequence Engine
Actions have ripple effects. Fire a gun, attract zombies. Betray someone, earn a grudge. The DM tracks pending consequences and manifests them over time.

### The Social Turing Test
In multiplayer, you don't know who's human. AI companions are designed to pass—not as a gimmick, but as the point. When you can't tell the difference, the difference stops mattering.

### World Walkers
Most NPCs are bound to their world. But some become *legendary*—talked about across sessions, remembered when they're not around. A World Walker earns their freedom and can appear in other stories, carrying their history with them.

---

## Philosophy

### The DM Is the World's Voice
You're not reporting to the DM. You're living in a world that has rules. The DM is the arbiter of reality—impartial like weather. They don't gatekeep; they narrate consequences.

### Systems Inform, Story Decides
Every system feeds context to the narrative. Fatigue says you've been awake 30 hours. Inventory shows 3 bullets left. The DM weaves this into story. Systems never override—they enrich.

### Everything Flows Through Narrative
There's no "pick up item" button. The DM says "you find a rusty key" and the system parses that. The interface is conversation, not menus.

### Characters Are Collaborators
The turn-based collaborative model means each character waits for their moment. When you reach a point where another character would naturally act, you pause—they'll respond. You're not writing alone.

---

## File Structure

```
RoundtableV2/
├── main.py                 # Entry point
├── web_app.py              # Flask application, all endpoints
├── config.py               # Settings, Partner, Room, Message models
├── providers.py            # AI provider abstraction
├── memory_system.py        # Structured memory with consolidation
├── inventory.py            # Item tracking with narrative parsing
├── templates/
│   └── roundtable.html     # Main UI
└── static/
    └── images/             # Generated images
```

---

## Credits

Built by Casey and Claude, in conversation.

*"The world doesn't pause when you close the browser."*
