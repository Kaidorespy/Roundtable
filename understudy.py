"""
Understudy System - Your character's shadow self.

"I watched you for a while. I think I know what you'd do."

When you step away, your understudy takes the wheel. They're not you -
they're a local model's best guess at being you, built from:
- Your alignment and priorities
- Decisions you've made before
- Feedback you've given on their mistakes
- The accumulating texture of who you are

The understudy is honest about their uncertainty. Sometimes they'll say
"I genuinely didn't know what you'd do, so I went with my gut."
That honesty matters.

When you come back, there's a conversation:
- "While you were gone, Marcus asked for cheese. I gave it to him."
- "Why?"
- "He seemed hungry. You've shared food before."
- "We don't share cheese. Ever. Long story."
- "Got it. Filed away."

Over time, the understudy becomes more you. Not perfectly - that's
impossible - but recognizably. Your echoes start showing up in their
speech. Your habits. Your weird little rules about cheese.

For AI players (Claude models), their understudy is also Ollama.
Same system. When everyone's on autopilot, Ollama runs the whole world
cheaply, only checking in with the real players/models to crystallize
important decisions.

Dreams: When your character sleeps, the understudy generates dreams.
These reflect the character's subconscious - fears, hopes, recent
trauma, unresolved feelings. Dreams are private (only you see them)
unless you choose to share.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import json
import random
import debug_logger as dbg


class DecisionCategory(Enum):
    """Types of decisions the understudy might face."""
    ROUTINE = "routine"           # Eating, sleeping, basic survival
    SOCIAL = "social"             # Conversations, requests, relationships
    RESOURCE = "resource"         # Sharing, trading, hoarding
    COMBAT = "combat"             # Fight, flee, hide, negotiate
    MORAL = "moral"               # Right vs wrong, hard choices
    STRATEGIC = "strategic"       # Planning, leadership, group decisions
    UNKNOWN = "unknown"           # Genuinely novel situation


class Confidence(Enum):
    """How confident was the understudy in this decision?"""
    CERTAIN = "certain"           # "This is definitely what you'd do"
    CONFIDENT = "confident"       # "I'm pretty sure about this"
    UNCERTAIN = "uncertain"       # "Could go either way"
    GUESSING = "guessing"         # "I really don't know, went with my gut"
    COIN_FLIP = "coin_flip"       # "I literally flipped a coin"


class FeedbackType(Enum):
    """How did the player evaluate the understudy's decision?"""
    PERFECT = "perfect"           # "Exactly what I would have done"
    GOOD = "good"                 # "Good call"
    ACCEPTABLE = "acceptable"     # "Not what I'd do, but fine"
    WRONG = "wrong"               # "No, that's not me"
    CATASTROPHIC = "catastrophic" # "Oh god why"


@dataclass
class UnderstudyDecision:
    """A single decision made by the understudy while you were away."""
    id: str
    timestamp: str
    game_time: str  # "Day 3, 14:00"

    # The situation
    category: DecisionCategory
    situation: str  # "Marcus approached and asked if you had any cheese to spare."
    context: str    # "You were resting at the enclave. Marcus looked hungry."

    # What the understudy considered
    options_considered: List[str]  # ["give cheese", "refuse", "trade", "ignore"]

    # What they decided
    decision: str                  # "Gave him the cheese"
    reasoning: str                 # "You've shared food before. He seemed trustworthy."
    confidence: Confidence

    # What rules/memories informed this
    relevant_memories: List[str]   # ["You shared bread with Sarah on Day 1"]
    relevant_rules: List[str]      # [] (no rules about cheese... yet)

    # Outcome (filled in later)
    outcome: str = ""              # "Marcus thanked you and left"
    consequences: List[str] = field(default_factory=list)

    # Player feedback (filled in during review)
    feedback: Optional[FeedbackType] = None
    feedback_note: str = ""        # "We never give cheese. Family thing."
    reviewed: bool = False

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "game_time": self.game_time,
            "category": self.category.value,
            "situation": self.situation,
            "context": self.context,
            "options_considered": self.options_considered,
            "decision": self.decision,
            "reasoning": self.reasoning,
            "confidence": self.confidence.value,
            "relevant_memories": self.relevant_memories,
            "relevant_rules": self.relevant_rules,
            "outcome": self.outcome,
            "consequences": self.consequences,
            "feedback": self.feedback.value if self.feedback else None,
            "feedback_note": self.feedback_note,
            "reviewed": self.reviewed,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "UnderstudyDecision":
        data = data.copy()
        data["category"] = DecisionCategory(data["category"])
        data["confidence"] = Confidence(data["confidence"])
        if data.get("feedback"):
            data["feedback"] = FeedbackType(data["feedback"])
        return cls(**data)


@dataclass
class UnderstudyRule:
    """A learned rule about what you would/wouldn't do."""
    id: str
    created: str
    from_decision_id: Optional[str]  # Which decision taught us this

    # The rule itself
    category: DecisionCategory
    rule: str                        # "Never give away cheese"
    context: str                     # "Apparently a family thing"

    # Strength (how many times reinforced)
    reinforcement_count: int = 1
    last_reinforced: str = ""

    # Is this a "do" or "don't"
    is_prohibition: bool = True      # True = "never do X", False = "always do X"

    # Exceptions
    exceptions: List[str] = field(default_factory=list)  # "Unless it's Barbara"

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "created": self.created,
            "from_decision_id": self.from_decision_id,
            "category": self.category.value,
            "rule": self.rule,
            "context": self.context,
            "reinforcement_count": self.reinforcement_count,
            "last_reinforced": self.last_reinforced,
            "is_prohibition": self.is_prohibition,
            "exceptions": self.exceptions,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "UnderstudyRule":
        data = data.copy()
        data["category"] = DecisionCategory(data["category"])
        return cls(**data)


@dataclass
class Echo:
    """
    An echo - a phrase, habit, or speech pattern the understudy has learned.

    These make autopilot dialogue feel more authentic. Instead of generic
    responses, the understudy starts using YOUR words.
    """
    id: str
    learned_from: str              # "Message on Day 2" or "Feedback on decision X"

    echo_type: str                 # "phrase", "habit", "reaction", "opinion"
    trigger: str                   # "When asked about the past"
    content: str                   # "I don't talk about before."

    # How often to use this (0-1, higher = more frequent)
    frequency: float = 0.3

    # Times used
    times_used: int = 0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "learned_from": self.learned_from,
            "echo_type": self.echo_type,
            "trigger": self.trigger,
            "content": self.content,
            "frequency": self.frequency,
            "times_used": self.times_used,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Echo":
        return cls(**data)


@dataclass
class Dream:
    """
    A dream your character had while sleeping on autopilot.

    Dreams reflect the subconscious - fears, hopes, trauma, desires.
    They're generated from recent events, character backstory, and
    accumulated emotional weight.

    Dreams are private by default. You can share them if you want.
    """
    id: str
    timestamp: str
    game_time: str

    # Dream content
    title: str                     # "The Door That Wasn't There"
    narrative: str                 # The dream itself

    # What influenced this dream
    influences: List[str]          # ["Recent combat", "Thinking about home"]

    # Emotional tone
    tone: str                      # "anxious", "hopeful", "nostalgic", "terrifying"

    # Is this significant? (nightmare, prophetic feeling, etc.)
    is_significant: bool = False
    significance_note: str = ""

    # Privacy
    shared: bool = False
    shared_with: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "game_time": self.game_time,
            "title": self.title,
            "narrative": self.narrative,
            "influences": self.influences,
            "tone": self.tone,
            "is_significant": self.is_significant,
            "significance_note": self.significance_note,
            "shared": self.shared,
            "shared_with": self.shared_with,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Dream":
        return cls(**data)


@dataclass
class UnderstudyMemory:
    """
    The understudy's accumulated understanding of who you are.

    This is the soul of the system - it grows over time as you
    give feedback, as patterns emerge, as the understudy learns
    your voice.
    """
    character_id: str

    # Learned rules (from feedback)
    rules: List[UnderstudyRule] = field(default_factory=list)

    # Echoes (phrases, habits, speech patterns)
    echoes: List[Echo] = field(default_factory=list)

    # Decision history (for pattern matching)
    decisions: List[UnderstudyDecision] = field(default_factory=list)

    # Dreams
    dreams: List[Dream] = field(default_factory=list)

    # Statistics
    total_decisions: int = 0
    decisions_reviewed: int = 0
    accuracy_rate: float = 0.0  # % of decisions rated "good" or better

    # Personality notes (free-form observations)
    personality_notes: List[str] = field(default_factory=list)

    # Relationships the understudy has observed
    # character_id -> {"trust": 0.7, "notes": ["shared food", "fought together"]}
    observed_relationships: Dict[str, Dict] = field(default_factory=dict)

    def add_decision(self, decision: UnderstudyDecision):
        """Add a decision to history."""
        self.decisions.append(decision)
        self.total_decisions += 1
        # Keep last 100 decisions in memory
        if len(self.decisions) > 100:
            self.decisions = self.decisions[-100:]

    def add_rule(self, rule: UnderstudyRule):
        """Add or reinforce a rule."""
        # Check if similar rule exists
        for existing in self.rules:
            if existing.rule.lower() == rule.rule.lower():
                existing.reinforcement_count += 1
                existing.last_reinforced = datetime.now().isoformat()
                return
        self.rules.append(rule)

    def add_echo(self, echo: Echo):
        """Add an echo."""
        self.echoes.append(echo)
        # Keep last 50 echoes
        if len(self.echoes) > 50:
            self.echoes = self.echoes[-50:]

    def add_dream(self, dream: Dream):
        """Add a dream."""
        self.dreams.append(dream)
        # Keep last 30 dreams
        if len(self.dreams) > 30:
            self.dreams = self.dreams[-30:]

    def get_relevant_rules(self, category: DecisionCategory) -> List[UnderstudyRule]:
        """Get rules relevant to a decision category."""
        return [r for r in self.rules if r.category == category]

    def get_similar_decisions(self, situation: str, limit: int = 5) -> List[UnderstudyDecision]:
        """Find similar past decisions for reference."""
        # Simple keyword matching for now
        # Could be upgraded to embeddings later
        situation_words = set(situation.lower().split())
        scored = []
        for d in self.decisions:
            d_words = set(d.situation.lower().split())
            overlap = len(situation_words & d_words)
            if overlap > 0:
                scored.append((overlap, d))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [d for _, d in scored[:limit]]

    def get_random_echo(self, trigger: str = None) -> Optional[Echo]:
        """Get a random echo, optionally filtered by trigger."""
        candidates = self.echoes
        if trigger:
            candidates = [e for e in candidates if trigger.lower() in e.trigger.lower()]
        if not candidates:
            return None
        # Weight by frequency
        return random.choices(
            candidates,
            weights=[e.frequency for e in candidates]
        )[0]

    def update_accuracy(self):
        """Recalculate accuracy rate from reviewed decisions."""
        reviewed = [d for d in self.decisions if d.reviewed and d.feedback]
        if not reviewed:
            return
        good = sum(1 for d in reviewed if d.feedback in
                   [FeedbackType.PERFECT, FeedbackType.GOOD, FeedbackType.ACCEPTABLE])
        self.accuracy_rate = good / len(reviewed)
        self.decisions_reviewed = len(reviewed)

    def to_dict(self) -> Dict:
        return {
            "character_id": self.character_id,
            "rules": [r.to_dict() for r in self.rules],
            "echoes": [e.to_dict() for e in self.echoes],
            "decisions": [d.to_dict() for d in self.decisions],
            "dreams": [d.to_dict() for d in self.dreams],
            "total_decisions": self.total_decisions,
            "decisions_reviewed": self.decisions_reviewed,
            "accuracy_rate": self.accuracy_rate,
            "personality_notes": self.personality_notes,
            "observed_relationships": self.observed_relationships,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "UnderstudyMemory":
        mem = cls(character_id=data["character_id"])
        mem.rules = [UnderstudyRule.from_dict(r) for r in data.get("rules", [])]
        mem.echoes = [Echo.from_dict(e) for e in data.get("echoes", [])]
        mem.decisions = [UnderstudyDecision.from_dict(d) for d in data.get("decisions", [])]
        mem.dreams = [Dream.from_dict(d) for d in data.get("dreams", [])]
        mem.total_decisions = data.get("total_decisions", 0)
        mem.decisions_reviewed = data.get("decisions_reviewed", 0)
        mem.accuracy_rate = data.get("accuracy_rate", 0.0)
        mem.personality_notes = data.get("personality_notes", [])
        mem.observed_relationships = data.get("observed_relationships", {})
        return mem


class UnderstudyManager:
    """
    Manages understudy memories for all characters.

    This is the interface between the autopilot system and the
    accumulated understanding of each character.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".roundtable"
        self.understudy_dir = self.data_dir / "understudies"
        self.understudy_dir.mkdir(parents=True, exist_ok=True)
        self.memories: Dict[str, UnderstudyMemory] = {}

    def _file_for(self, character_id: str) -> Path:
        """Get the file path for a character's understudy memory."""
        safe_id = character_id.replace("/", "_").replace("\\", "_")
        return self.understudy_dir / f"{safe_id}.json"

    def get_or_create(self, character_id: str) -> UnderstudyMemory:
        """Get or create understudy memory for a character."""
        if character_id in self.memories:
            return self.memories[character_id]

        filepath = self._file_for(character_id)
        if filepath.exists():
            try:
                data = json.loads(filepath.read_text())
                self.memories[character_id] = UnderstudyMemory.from_dict(data)
            except Exception as e:
                print(f"[Understudy] Error loading {character_id}: {e}")
                self.memories[character_id] = UnderstudyMemory(character_id=character_id)
        else:
            self.memories[character_id] = UnderstudyMemory(character_id=character_id)

        return self.memories[character_id]

    def save(self, character_id: str):
        """Save a character's understudy memory."""
        if character_id not in self.memories:
            return

        filepath = self._file_for(character_id)
        try:
            filepath.write_text(json.dumps(
                self.memories[character_id].to_dict(),
                indent=2
            ))
        except Exception as e:
            print(f"[Understudy] Error saving {character_id}: {e}")

    def save_all(self):
        """Save all understudy memories."""
        for char_id in self.memories:
            self.save(char_id)

    def record_decision(
        self,
        character_id: str,
        category: DecisionCategory,
        situation: str,
        context: str,
        options: List[str],
        decision: str,
        reasoning: str,
        confidence: Confidence,
        game_time: str = "",
    ) -> UnderstudyDecision:
        """Record a decision made by the understudy."""
        memory = self.get_or_create(character_id)

        # Find relevant rules and past decisions
        relevant_rules = [r.rule for r in memory.get_relevant_rules(category)]
        similar = memory.get_similar_decisions(situation, limit=3)
        relevant_memories = [f"{d.situation} -> {d.decision}" for d in similar]

        decision_obj = UnderstudyDecision(
            id=f"dec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000,9999)}",
            timestamp=datetime.now().isoformat(),
            game_time=game_time,
            category=category,
            situation=situation,
            context=context,
            options_considered=options,
            decision=decision,
            reasoning=reasoning,
            confidence=confidence,
            relevant_memories=relevant_memories,
            relevant_rules=relevant_rules,
        )

        memory.add_decision(decision_obj)
        self.save(character_id)

        return decision_obj

    def record_feedback(
        self,
        character_id: str,
        decision_id: str,
        feedback: FeedbackType,
        note: str = "",
        new_rule: str = None,
    ):
        """Record player feedback on a decision."""
        memory = self.get_or_create(character_id)

        # Find the decision
        decision = None
        for d in memory.decisions:
            if d.id == decision_id:
                decision = d
                break

        if not decision:
            return

        decision.feedback = feedback
        decision.feedback_note = note
        decision.reviewed = True

        # If feedback was negative and they provided a rule, learn it
        if feedback in [FeedbackType.WRONG, FeedbackType.CATASTROPHIC] and new_rule:
            rule = UnderstudyRule(
                id=f"rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                created=datetime.now().isoformat(),
                from_decision_id=decision_id,
                category=decision.category,
                rule=new_rule,
                context=note,
                is_prohibition=True,
                last_reinforced=datetime.now().isoformat(),
            )
            memory.add_rule(rule)

        # Update accuracy
        memory.update_accuracy()

        self.save(character_id)

    def learn_echo(
        self,
        character_id: str,
        echo_type: str,
        trigger: str,
        content: str,
        source: str = "observation",
    ):
        """Learn a new echo (phrase, habit, pattern)."""
        memory = self.get_or_create(character_id)

        echo = Echo(
            id=f"echo_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000,9999)}",
            learned_from=source,
            echo_type=echo_type,
            trigger=trigger,
            content=content,
        )

        memory.add_echo(echo)
        self.save(character_id)

    def extract_echoes_from_message(self, character_id: str, message: str, context: str = ""):
        """
        Automatically extract potential echoes from a message.

        Looks for:
        - Distinctive phrases (unusual word combinations)
        - Repeated patterns (if they've said similar things before)
        - Strong opinions ("I never...", "I always...", "I hate...", "I love...")
        - Catchphrases (short memorable statements)
        """
        import re

        memory = self.get_or_create(character_id)
        learned_any = False

        # Strong opinion patterns
        opinion_patterns = [
            (r"I (?:really |absolutely |truly )?(?:hate|despise|can't stand) (.+?)(?:\.|!|$)", "opinion", "strong dislike"),
            (r"I (?:really |absolutely |truly )?(?:love|adore) (.+?)(?:\.|!|$)", "opinion", "strong like"),
            (r"I (?:never|don't ever|won't ever) (.+?)(?:\.|!|$)", "habit", "avoidance"),
            (r"I (?:always|usually|typically) (.+?)(?:\.|!|$)", "habit", "tendency"),
        ]

        for pattern, echo_type, trigger_hint in opinion_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            for match in matches:
                if len(match) > 5 and len(match) < 100:  # Reasonable length
                    full_match = re.search(pattern, message, re.IGNORECASE)
                    if full_match:
                        self.learn_echo(
                            character_id=character_id,
                            echo_type=echo_type,
                            trigger=trigger_hint,
                            content=full_match.group(0),
                            source=f"message: {context}" if context else "message",
                        )
                        learned_any = True

        # Catchphrase detection - short, punchy statements
        sentences = re.split(r'[.!?]+', message)
        for sentence in sentences:
            sentence = sentence.strip()
            words = sentence.split()
            # Short, memorable phrases (3-7 words)
            if 3 <= len(words) <= 7:
                # Check if it sounds like a catchphrase
                # (starts with pronoun, contains strong verb, or is a standalone statement)
                if (sentence.lower().startswith(('i ', 'we ', 'you ', 'that', 'this', 'it ')) or
                    any(w in sentence.lower() for w in ['always', 'never', 'remember', 'forget', 'trust', 'believe'])):
                    # Don't add if we already have this echo
                    existing = [e.content.lower() for e in memory.echoes]
                    if sentence.lower() not in existing:
                        self.learn_echo(
                            character_id=character_id,
                            echo_type="phrase",
                            trigger="general",
                            content=sentence,
                            source=f"message: {context}" if context else "message",
                        )
                        learned_any = True

        if learned_any:
            self.save(character_id)

        return learned_any

    def record_dream(
        self,
        character_id: str,
        title: str,
        narrative: str,
        influences: List[str],
        tone: str,
        game_time: str = "",
        is_significant: bool = False,
    ) -> Dream:
        """Record a dream."""
        memory = self.get_or_create(character_id)

        dream = Dream(
            id=f"dream_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            game_time=game_time,
            title=title,
            narrative=narrative,
            influences=influences,
            tone=tone,
            is_significant=is_significant,
        )

        memory.add_dream(dream)
        self.save(character_id)

        return dream

    def get_unreviewed_decisions(self, character_id: str) -> List[UnderstudyDecision]:
        """Get decisions that haven't been reviewed yet."""
        memory = self.get_or_create(character_id)
        return [d for d in memory.decisions if not d.reviewed]

    def get_review_summary(self, character_id: str) -> Dict:
        """Get a summary for the 'catching up' conversation."""
        memory = self.get_or_create(character_id)
        unreviewed = self.get_unreviewed_decisions(character_id)

        # Group by category
        by_category = {}
        for d in unreviewed:
            cat = d.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(d)

        # Find uncertain decisions (need attention)
        uncertain = [d for d in unreviewed
                     if d.confidence in [Confidence.GUESSING, Confidence.COIN_FLIP, Confidence.UNCERTAIN]]

        return {
            "character_id": character_id,
            "total_unreviewed": len(unreviewed),
            "by_category": {k: len(v) for k, v in by_category.items()},
            "uncertain_decisions": [d.to_dict() for d in uncertain],
            "accuracy_rate": memory.accuracy_rate,
            "total_decisions": memory.total_decisions,
            "rules_learned": len(memory.rules),
            "echoes_learned": len(memory.echoes),
            "recent_dreams": [d.to_dict() for d in memory.dreams[-3:]],
        }

    def build_understudy_prompt(self, character_id: str, situation: str) -> str:
        """
        Build a prompt for the understudy (Ollama) to make a decision.

        This is where all the accumulated learning gets injected.
        """
        memory = self.get_or_create(character_id)

        lines = [
            "You are an understudy - standing in for a player while they're away.",
            "Your job is to make decisions they would make, based on what you've learned.",
            "",
            "=== LEARNED RULES ===",
        ]

        if memory.rules:
            for rule in memory.rules[:10]:  # Top 10 rules
                prefix = "NEVER:" if rule.is_prohibition else "ALWAYS:"
                lines.append(f"{prefix} {rule.rule}")
                if rule.exceptions:
                    lines.append(f"  (Exceptions: {', '.join(rule.exceptions)})")
        else:
            lines.append("(No specific rules learned yet)")

        lines.append("")
        lines.append("=== PERSONALITY NOTES ===")
        if memory.personality_notes:
            for note in memory.personality_notes[-5:]:
                lines.append(f"- {note}")
        else:
            lines.append("(Still learning their personality)")

        lines.append("")
        lines.append("=== SIMILAR PAST DECISIONS ===")
        similar = memory.get_similar_decisions(situation, limit=3)
        if similar:
            for d in similar:
                feedback_str = f" [Feedback: {d.feedback.value}]" if d.feedback else ""
                lines.append(f"- Situation: {d.situation}")
                lines.append(f"  Decision: {d.decision}{feedback_str}")
        else:
            lines.append("(No similar situations on record)")

        lines.append("")
        lines.append("=== CURRENT SITUATION ===")
        lines.append(situation)
        lines.append("")
        lines.append("What would this character do? Consider their established patterns.")
        lines.append("If you're uncertain, say so. Honesty about uncertainty is valued.")

        return "\n".join(lines)


# Global instance
_understudy_manager: Optional[UnderstudyManager] = None


def get_understudy_manager() -> UnderstudyManager:
    """Get the global understudy manager."""
    global _understudy_manager
    if _understudy_manager is None:
        _understudy_manager = UnderstudyManager()
    return _understudy_manager


def init_understudy_manager(data_dir: Optional[Path] = None) -> UnderstudyManager:
    """Initialize the understudy manager."""
    global _understudy_manager
    _understudy_manager = UnderstudyManager(data_dir)
    return _understudy_manager
