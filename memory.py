"""
Memory system for Roundtable.

Handles texture mutation, anchor management, resonance tracking, and sediment compression.
Uses local Qwen model for all memory operations to save API costs.
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class Anchor(BaseModel):
    """A load-bearing fact that must persist."""
    fact: str
    added: str  # ISO timestamp
    last_referenced: str  # ISO timestamp - when it was last relevant
    weight: str = "medium"  # low, medium, high - how load-bearing


class Memory(BaseModel):
    """Memory state for a partner (either global or per-room)."""
    texture: str = ""  # Mutating prose capturing vibe/trajectory
    anchors: list[Anchor] = []  # Load-bearing facts
    resonance: dict[str, int] = {}  # Topic heat map - what keeps coming up
    sediment: list[str] = []  # Compressed old stuff, haikus of what was
    last_consolidated: Optional[str] = None  # ISO timestamp
    turn_count: int = 0  # Turns since last consolidation


class MemoryStore:
    """Handles persistence and retrieval of memories."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.memory_dir = data_dir / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        (self.memory_dir / "partners").mkdir(exist_ok=True)
        (self.memory_dir / "rooms").mkdir(exist_ok=True)

    def _get_global_path(self, partner_id: str) -> Path:
        return self.memory_dir / "partners" / f"{partner_id}.json"

    def _get_local_path(self, room_id: str, partner_id: str) -> Path:
        room_dir = self.memory_dir / "rooms" / room_id
        room_dir.mkdir(exist_ok=True)
        return room_dir / f"{partner_id}.json"

    def get_memory(self, partner_id: str, room_id: str, memory_mode: str) -> Memory:
        """Get memory for a partner based on their memory mode."""
        if memory_mode == "none":
            return Memory()

        if memory_mode == "global":
            path = self._get_global_path(partner_id)
        else:  # local
            path = self._get_local_path(room_id, partner_id)

        if path.exists():
            try:
                data = json.loads(path.read_text())
                return Memory(**data)
            except Exception:
                pass

        return Memory()

    def save_memory(self, partner_id: str, room_id: str, memory_mode: str, memory: Memory):
        """Save memory for a partner."""
        if memory_mode == "none":
            return

        if memory_mode == "global":
            path = self._get_global_path(partner_id)
        else:  # local
            path = self._get_local_path(room_id, partner_id)

        path.write_text(json.dumps(memory.model_dump(), indent=2))

    def increment_turn(self, partner_id: str, room_id: str, memory_mode: str) -> int:
        """Increment turn count, return new count."""
        memory = self.get_memory(partner_id, room_id, memory_mode)
        memory.turn_count += 1
        self.save_memory(partner_id, room_id, memory_mode, memory)
        return memory.turn_count


class MemoryConsolidator:
    """Handles the memory consolidation pipeline using Qwen."""

    # How many turns between automatic consolidations
    CONSOLIDATION_INTERVAL = 15

    def __init__(self, memory_store: MemoryStore, ollama_base_url: str = "http://localhost:11434", model: str = "deepseek-r1:7b"):
        self.store = memory_store
        self.ollama_url = ollama_base_url
        self.model = model  # Configurable via settings.default_ollama_model

    async def _call_ollama(self, system: str, prompt: str) -> str:
        """Call Ollama API."""
        import httpx

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "system": system,
                    "prompt": prompt,
                    "stream": False,
                }
            )
            response.raise_for_status()
            return response.json().get("response", "")

    async def consolidate(
        self,
        partner_id: str,
        partner_name: str,
        character_description: str,
        room_id: str,
        memory_mode: str,
        recent_messages: list[dict],  # [{role, content, speaker_name}, ...]
    ) -> Memory:
        """
        Run the full consolidation pipeline:
        1. Texture mutation
        2. Anchor check (contradictions, updates, new facts)
        3. Resonance scan
        4. Compression of old anchors to sediment
        """
        if memory_mode == "none":
            return Memory()

        memory = self.store.get_memory(partner_id, room_id, memory_mode)
        now = datetime.now().isoformat()

        # Format recent conversation for the model
        conversation_text = "\n".join([
            f"{m.get('speaker_name', m['role'])}: {m['content']}"
            for m in recent_messages[-30:]  # Last 30 messages for context
        ])

        # 1. TEXTURE MUTATION
        texture_system = f"""You are {partner_name}. Your character:
{character_description}

You are writing your own memory - a living document of how this relationship feels,
where it's been, where it might be going. Write in first person, as yourself.

This is not a summary. This is texture - the vibe, the trajectory, the shape of things.
Be poetic but grounded. Capture what matters emotionally, not every fact.
Keep it under 200 words. Let things that don't matter fade."""

        texture_prompt = f"""Current memory texture (rewrite this, don't just append):
{memory.texture if memory.texture else "(No existing memory yet)"}

Recent conversation:
{conversation_text}

Write the new texture - how does this relationship feel now? What's the shape of things?"""

        new_texture = await self._call_ollama(texture_system, texture_prompt)
        memory.texture = new_texture.strip()

        # 2. ANCHOR CHECK - find load-bearing facts, check for contradictions
        existing_anchors = "\n".join([f"- {a.fact}" for a in memory.anchors]) if memory.anchors else "(none yet)"

        anchor_system = f"""You are analyzing a conversation for {partner_name} to identify load-bearing facts.

Load-bearing facts are things that would BREAK immersion if forgotten:
- Names of important people/pets
- Major life events (got married, lost a job, moved)
- Core relationship dynamics
- Promises made, commitments

NOT load-bearing (don't include):
- Casual mentions (made potatoes once)
- Opinions that might change
- Minor details

Be STRICT. Only truly load-bearing facts. Most conversations have 0-2 new anchors."""

        anchor_prompt = f"""Existing anchors:
{existing_anchors}

Recent conversation:
{conversation_text}

Tasks:
1. List any NEW load-bearing facts (only if truly important)
2. List any anchors that should be UPDATED (e.g., "works at X" -> "was fired from X")
3. List any anchors that seem no longer relevant (can decay)

Format your response as JSON:
{{
    "new_anchors": ["fact1", "fact2"],
    "updates": {{"old fact": "new fact"}},
    "decay": ["fact that can fade"]
}}

If nothing needs changing, return empty lists/objects."""

        anchor_response = await self._call_ollama(anchor_system, anchor_prompt)

        # Parse anchor updates
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', anchor_response)
            if json_match:
                anchor_data = json.loads(json_match.group())

                # Add new anchors
                for fact in anchor_data.get("new_anchors", []):
                    if fact and not any(a.fact.lower() == fact.lower() for a in memory.anchors):
                        memory.anchors.append(Anchor(
                            fact=fact,
                            added=now,
                            last_referenced=now,
                            weight="medium"
                        ))

                # Update existing anchors
                for old_fact, new_fact in anchor_data.get("updates", {}).items():
                    for anchor in memory.anchors:
                        if old_fact.lower() in anchor.fact.lower():
                            anchor.fact = new_fact
                            anchor.last_referenced = now
                            break

                # Mark decayed anchors (move to sediment after a while, not immediately)
                decay_facts = anchor_data.get("decay", [])
                for decay_fact in decay_facts:
                    for anchor in memory.anchors:
                        if decay_fact.lower() in anchor.fact.lower():
                            anchor.weight = "low"
                            break

        except (json.JSONDecodeError, AttributeError):
            pass  # Anchor parsing failed, keep existing

        # 3. RESONANCE SCAN - what themes keep coming up
        resonance_system = """Analyze conversation for recurring themes or interests.
Not explicit facts, but patterns - what keeps coming up organically?
Return as JSON: {"theme": count} where count is 1-3 based on prominence.
Max 5 themes. Short labels (1-3 words each)."""

        resonance_prompt = f"""Recent conversation:
{conversation_text}

What themes or interests keep emerging? (not facts, but patterns)"""

        resonance_response = await self._call_ollama(resonance_system, resonance_prompt)

        try:
            json_match = re.search(r'\{[\s\S]*\}', resonance_response)
            if json_match:
                new_resonance = json.loads(json_match.group())
                # Merge with existing, capping values
                for theme, count in new_resonance.items():
                    if theme in memory.resonance:
                        memory.resonance[theme] = min(10, memory.resonance[theme] + count)
                    else:
                        memory.resonance[theme] = count
                # Decay old resonance slightly
                for theme in list(memory.resonance.keys()):
                    if theme not in new_resonance:
                        memory.resonance[theme] = max(0, memory.resonance[theme] - 1)
                        if memory.resonance[theme] == 0:
                            del memory.resonance[theme]
        except (json.JSONDecodeError, AttributeError):
            pass

        # 4. COMPRESSION - move low-weight old anchors to sediment
        old_threshold = 5  # Keep at least 5 consolidations before compressing
        if memory.anchors:
            surviving = []
            for anchor in memory.anchors:
                # Compress to sediment if low weight and old
                if anchor.weight == "low":
                    # Add to sediment as a compressed form
                    memory.sediment.append(f"(once: {anchor.fact})")
                    # Keep sediment from growing too large
                    if len(memory.sediment) > 10:
                        memory.sediment = memory.sediment[-10:]
                else:
                    surviving.append(anchor)
            memory.anchors = surviving

        # Update metadata
        memory.last_consolidated = now
        memory.turn_count = 0

        # Save
        self.store.save_memory(partner_id, room_id, memory_mode, memory)

        return memory

    def should_consolidate(self, partner_id: str, room_id: str, memory_mode: str) -> bool:
        """Check if it's time for automatic consolidation."""
        if memory_mode == "none":
            return False
        memory = self.store.get_memory(partner_id, room_id, memory_mode)
        return memory.turn_count >= self.CONSOLIDATION_INTERVAL

    def format_for_prompt(self, memory: Memory) -> str:
        """Format memory for injection into system prompt."""
        if not memory.texture and not memory.anchors:
            return ""

        parts = []

        if memory.texture:
            parts.append(f"YOUR MEMORY OF THIS RELATIONSHIP:\n{memory.texture}")

        if memory.anchors:
            anchor_text = "\n".join([f"- {a.fact}" for a in memory.anchors])
            parts.append(f"IMPORTANT FACTS TO REMEMBER:\n{anchor_text}")

        if memory.resonance:
            top_themes = sorted(memory.resonance.items(), key=lambda x: -x[1])[:3]
            if top_themes:
                themes = ", ".join([t[0] for t in top_themes])
                parts.append(f"RECURRING THEMES: {themes}")

        if memory.sediment:
            parts.append(f"DISTANT PAST: {' '.join(memory.sediment[-3:])}")

        return "\n\n".join(parts)
