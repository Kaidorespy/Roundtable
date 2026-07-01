"""
Story Daemon - Background story progression for NPCs with agency.

"As long as one human is logged into the server,
Barbara's story continues. Even if no one's watching."

This daemon:
- Runs in the background while players are connected
- Advances stories for NPCs with AGENCY or higher
- Generates world events
- Handles NPC-to-NPC interactions
- Notifies the DM of significant events
- Auto-saves NPC state

The key insight: NPCs aren't static quest-givers waiting for players.
They have lives. They make decisions. They change.
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import debug_logger as dbg

# Late import to avoid circular dependency - used in _advance_character_fatigue
# from fatigue import get_fatigue_tracker


class EventSeverity(Enum):
    """How significant is this event?"""
    MINOR = "minor"          # Internal to NPC, no notification
    NOTABLE = "notable"      # DM gets notified
    MAJOR = "major"          # Affects the world, players might notice
    CRITICAL = "critical"    # Death, major world change


@dataclass
class StoryEvent:
    """A single event in the world's story."""
    timestamp: str
    npc_id: str
    npc_name: str
    event_type: str          # "story_beat", "decision", "interaction", "death", "location_change"
    severity: EventSeverity
    description: str
    location: Optional[str] = None
    involved_npcs: List[str] = field(default_factory=list)
    involved_players: List[str] = field(default_factory=list)
    consequences: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "npc_id": self.npc_id,
            "npc_name": self.npc_name,
            "event_type": self.event_type,
            "severity": self.severity.value,
            "description": self.description,
            "location": self.location,
            "involved_npcs": self.involved_npcs,
            "involved_players": self.involved_players,
            "consequences": self.consequences,
        }


@dataclass
class WorldState:
    """The current state of a world/room."""
    world_id: str
    time_of_day: str = "day"          # "dawn", "day", "dusk", "night"
    weather: str = "clear"             # "clear", "rain", "storm", "fog", etc.
    mood: str = "tense"                # Overall mood of the world
    recent_events: List[str] = field(default_factory=list)  # Last few notable things
    active_threats: List[str] = field(default_factory=list)  # Current dangers

    # Game time tracking
    time_mode: str = "turn_based"     # "compressed", "realtime", "turn_based"
    time_ratio: float = 8.0           # For compressed: game hours per real hour
    game_day: int = 1                 # Current in-game day
    game_hour: int = 8                # Current in-game hour (0-23)
    game_minute: int = 0              # Current in-game minute (0-59)
    last_real_time: Optional[str] = None  # ISO timestamp of last time update

    # Real-time sync settings (for time_mode="realtime")
    # If set, game time = real time in this timezone
    timezone: Optional[str] = None     # "America/Denver", "America/Chicago", etc.
    realtime_start_day: int = 1        # What game day corresponds to real today
    realtime_epoch: Optional[str] = None  # ISO date when game day 1 started

    # Ambient threat system
    threat_level: float = 0.0         # 0-10 scale
    ambient_events_enabled: bool = False
    last_ambient_event: Optional[str] = None  # ISO timestamp

    # Countdown pressure system - scheduled events
    # List of {trigger_day: int, trigger_hour: int, event: str, severity: str, fired: bool}
    countdowns: List[Dict[str, Any]] = field(default_factory=list)

    def sync_to_realtime(self):
        """
        Sync game time to real time based on timezone.
        Call this periodically when time_mode is "realtime".
        """
        if self.time_mode != "realtime" or not self.timezone:
            return

        try:
            from datetime import datetime
            import zoneinfo
            tz = zoneinfo.ZoneInfo(self.timezone)
            now = datetime.now(tz)

            self.game_hour = now.hour
            self.game_minute = now.minute

            # Calculate game day from epoch
            if self.realtime_epoch:
                epoch = datetime.fromisoformat(self.realtime_epoch.replace('Z', '+00:00'))
                epoch = epoch.replace(tzinfo=tz)
                days_elapsed = (now.date() - epoch.date()).days
                self.game_day = self.realtime_start_day + days_elapsed

            # Update time_of_day
            self.time_of_day = self._hour_to_period(self.game_hour)
            self.last_real_time = now.isoformat()

        except Exception as e:
            print(f"[WorldState] Failed to sync realtime: {e}")

    def get_context_string(self) -> str:
        """Generate a context string for story generation."""
        # Format game time nicely
        hour_name = self._hour_to_period(self.game_hour)
        context = f"Day {self.game_day}, {hour_name} ({self.game_hour}:00). Weather: {self.weather}. Mood: {self.mood}."

        if self.threat_level > 0:
            threat_desc = self._threat_level_description()
            context += f" Danger level: {threat_desc}."

        if self.recent_events:
            context += f"\n\nRecent events in the world:\n"
            for event in self.recent_events[-5:]:
                context += f"- {event}\n"

        if self.active_threats:
            context += f"\n\nActive threats: {', '.join(self.active_threats)}"

        return context

    def _hour_to_period(self, hour: int) -> str:
        """Convert hour to time period name."""
        if 5 <= hour < 8:
            return "dawn"
        elif 8 <= hour < 12:
            return "morning"
        elif 12 <= hour < 14:
            return "midday"
        elif 14 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 21:
            return "dusk"
        elif 21 <= hour < 24:
            return "evening"
        else:  # 0-5
            return "night"

    def _threat_level_description(self) -> str:
        """Get human-readable threat level."""
        if self.threat_level <= 1:
            return "minimal"
        elif self.threat_level <= 3:
            return "low"
        elif self.threat_level <= 5:
            return "moderate"
        elif self.threat_level <= 7:
            return "high"
        elif self.threat_level <= 9:
            return "severe"
        else:
            return "critical"

    def advance_time(self, hours: int = 1):
        """Move time forward by specified hours."""
        self.game_hour += hours
        while self.game_hour >= 24:
            self.game_hour -= 24
            self.game_day += 1

        # Update time_of_day for backwards compat
        period = self._hour_to_period(self.game_hour)
        if period in ["dawn"]:
            self.time_of_day = "dawn"
        elif period in ["morning", "midday", "afternoon"]:
            self.time_of_day = "day"
        elif period in ["dusk", "evening"]:
            self.time_of_day = "dusk"
        else:
            self.time_of_day = "night"

    def add_countdown(
        self,
        event_description: str,
        days_from_now: int = 0,
        hours_from_now: int = 0,
        severity: str = "major",  # "minor", "notable", "major", "critical"
        absolute_day: int = None,
        absolute_hour: int = None,
    ):
        """
        Schedule an event to happen at a future time.

        Examples:
            add_countdown("The ritual completes", days_from_now=3)
            add_countdown("The bomb explodes", hours_from_now=6)
            add_countdown("The caravan leaves", absolute_day=5, absolute_hour=8)
        """
        if absolute_day is not None:
            trigger_day = absolute_day
            trigger_hour = absolute_hour if absolute_hour is not None else 12
        else:
            # Calculate from current time
            total_hours = self.game_hour + hours_from_now + (days_from_now * 24)
            trigger_day = self.game_day + (total_hours // 24)
            trigger_hour = total_hours % 24

        self.countdowns.append({
            "trigger_day": trigger_day,
            "trigger_hour": trigger_hour,
            "event": event_description,
            "severity": severity,
            "fired": False,
            "created_day": self.game_day,
            "created_hour": self.game_hour,
        })

    def check_countdowns(self) -> List[Dict[str, Any]]:
        """
        Check for any countdowns that should fire now.

        Returns list of events that just triggered.
        """
        triggered = []

        for countdown in self.countdowns:
            if countdown["fired"]:
                continue

            # Check if time has reached or passed the trigger
            if (self.game_day > countdown["trigger_day"] or
                (self.game_day == countdown["trigger_day"] and
                 self.game_hour >= countdown["trigger_hour"])):
                countdown["fired"] = True
                triggered.append(countdown)

        return triggered

    def get_pending_countdowns(self) -> List[Dict[str, Any]]:
        """Get countdowns that haven't fired yet, with time remaining."""
        pending = []

        for countdown in self.countdowns:
            if countdown["fired"]:
                continue

            # Calculate time remaining
            hours_remaining = (
                (countdown["trigger_day"] - self.game_day) * 24 +
                (countdown["trigger_hour"] - self.game_hour)
            )

            pending.append({
                "event": countdown["event"],
                "severity": countdown["severity"],
                "hours_remaining": hours_remaining,
                "trigger_day": countdown["trigger_day"],
                "trigger_hour": countdown["trigger_hour"],
            })

        return sorted(pending, key=lambda x: x["hours_remaining"])

    def get_countdown_context(self) -> str:
        """Get countdown info for DM context."""
        pending = self.get_pending_countdowns()
        if not pending:
            return ""

        lines = ["⏰ COUNTDOWN EVENTS (ticking clocks):"]
        for p in pending[:5]:  # Show max 5
            hours = p["hours_remaining"]
            if hours < 1:
                time_str = "IMMINENT"
            elif hours < 24:
                time_str = f"~{hours:.0f} hours"
            else:
                days = hours / 24
                time_str = f"~{days:.1f} days"

            severity_icon = {"minor": "•", "notable": "◦", "major": "⚠️", "critical": "🔴"}.get(p["severity"], "•")
            lines.append(f"  {severity_icon} {p['event']}: {time_str} remaining")

        return "\n".join(lines)

    def update_game_time(self, real_elapsed_seconds: float):
        """
        Update game time based on real time elapsed.
        Only for compressed/realtime modes.
        """
        if self.time_mode == "turn_based":
            return  # No automatic time passage

        # Calculate game hours elapsed
        real_hours = real_elapsed_seconds / 3600

        if self.time_mode == "realtime":
            game_hours = real_hours
        else:  # compressed
            game_hours = real_hours * self.time_ratio

        # Advance time
        full_hours = int(game_hours)
        if full_hours > 0:
            self.advance_time(full_hours)


class StoryDaemon:
    """
    The daemon that makes the world feel alive.

    Runs in a background thread, periodically advancing NPC stories
    and generating world events.
    """

    def __init__(
        self,
        npc_registry,
        ollama_generate_func: Callable,
        tick_interval_seconds: int = 300,  # 5 minutes default
        auto_save_path: Optional[str] = None,
    ):
        self.registry = npc_registry
        self.ollama_generate = ollama_generate_func
        self.tick_interval = tick_interval_seconds
        self.auto_save_path = auto_save_path or str(Path.home() / '.roundtable' / 'npcs.json')

        # World states per room/world
        self.world_states: Dict[str, WorldState] = {}

        # Event log
        self.events: List[StoryEvent] = []
        self.max_events = 1000  # Keep last 1000 events

        # DM notifications (unread events)
        self.dm_notifications: Dict[str, List[StoryEvent]] = {}  # world_id -> events

        # Connection tracking
        self.connected_players: Dict[str, set] = {}  # world_id -> set of player_ids

        # Daemon state
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Stats
        self.ticks_completed = 0
        self.stories_advanced = 0
        self.last_tick: Optional[str] = None

    def get_or_create_world(self, world_id: str) -> WorldState:
        """Get or create world state for a room."""
        if world_id not in self.world_states:
            self.world_states[world_id] = WorldState(world_id=world_id)
        return self.world_states[world_id]

    def player_connected(self, world_id: str, player_id: str):
        """Track a player connecting to a world."""
        if world_id not in self.connected_players:
            self.connected_players[world_id] = set()
        self.connected_players[world_id].add(player_id)
        print(f"[StoryDaemon] Player {player_id} connected to {world_id}. "
              f"Total: {len(self.connected_players[world_id])}")

    def player_disconnected(self, world_id: str, player_id: str):
        """Track a player disconnecting."""
        if world_id in self.connected_players:
            self.connected_players[world_id].discard(player_id)
            if not self.connected_players[world_id]:
                del self.connected_players[world_id]

    def has_connected_players(self, world_id: str = None) -> bool:
        """Check if any players are connected (to a specific world or any world)."""
        if world_id:
            return bool(self.connected_players.get(world_id))
        return bool(self.connected_players)

    def add_event(self, event: StoryEvent):
        """Add an event to the log."""
        self.events.append(event)

        # Trim if too many
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

        # Add to DM notifications if notable+
        if event.severity.value in ["notable", "major", "critical"]:
            world_id = event.location or "global"
            if world_id not in self.dm_notifications:
                self.dm_notifications[world_id] = []
            self.dm_notifications[world_id].append(event)

            # Debug output for notable+ story events
            severity_colors = {
                "notable": "\033[38;5;228m",   # Yellow
                "major": "\033[38;5;208m",     # Orange
                "critical": "\033[38;5;196m",  # Red
            }
            color = severity_colors.get(event.severity.value, "\033[0m")
            reset = "\033[0m"
            print(f"{color}[STORY EVENT] [{event.severity.value.upper()}] {event.event_type}: {event.npc_name} - {event.description[:60]}...{reset}")

        # Add to world's recent events
        if event.location and event.location in self.world_states:
            world = self.world_states[event.location]
            world.recent_events.append(f"{event.npc_name}: {event.description[:100]}")
            # Keep last 10
            world.recent_events = world.recent_events[-10:]

    def get_dm_notifications(self, world_id: str, clear: bool = True) -> List[StoryEvent]:
        """Get unread DM notifications for a world."""
        notifications = self.dm_notifications.get(world_id, [])
        if clear:
            self.dm_notifications[world_id] = []
        return notifications

    async def tick(self):
        """
        One tick of the daemon.

        This is where the magic happens:
        1. Check each active world
        2. Advance time
        3. Pick NPCs to advance
        4. Generate story beats
        5. Handle NPC interactions
        6. Save state
        """
        self.last_tick = datetime.now().isoformat()

        # Only process worlds with connected players
        active_worlds = list(self.connected_players.keys())

        if not active_worlds:
            return  # No one's watching, the world is frozen

        dbg.story(f"▶ TICK #{self.ticks_completed + 1}: {len(active_worlds)} active worlds")

        for world_id in active_worlds:
            await self._process_world(world_id)

        self.ticks_completed += 1

        # Auto-save periodically
        if self.ticks_completed % 5 == 0:  # Every 5 ticks
            self._save_state()

    async def _process_world(self, world_id: str):
        """Process one world for this tick."""
        import random
        world = self.get_or_create_world(world_id)

        # Track time before advancement for fatigue calculation
        old_day, old_hour = world.game_day, world.game_hour

        # Update game time based on time_mode
        if world.time_mode != "turn_based":
            now = datetime.now()
            if world.last_real_time:
                try:
                    last = datetime.fromisoformat(world.last_real_time)
                    elapsed = (now - last).total_seconds()
                    world.update_game_time(elapsed)
                except:
                    pass
            world.last_real_time = now.isoformat()
        elif self.ticks_completed % 4 == 0:
            # Turn-based: only advance on ticks (backwards compat)
            world.advance_time(1)

        # Calculate hours that passed and advance fatigue
        hours_passed = (world.game_day - old_day) * 24 + (world.game_hour - old_hour)
        if hours_passed > 0:
            self._advance_room_fatigue(world_id, hours_passed)

        # Check for ambient threat events
        if world.ambient_events_enabled and world.threat_level > 0:
            await self._check_ambient_threat(world)

        # Check for countdown events that should fire
        triggered = world.check_countdowns()
        for countdown in triggered:
            event = StoryEvent(
                timestamp=datetime.now().isoformat(),
                npc_id="countdown",
                npc_name="⏰ Countdown",
                event_type="countdown_trigger",
                severity=EventSeverity[countdown["severity"].upper()] if countdown["severity"].upper() in EventSeverity.__members__ else EventSeverity.MAJOR,
                description=f"COUNTDOWN TRIGGERED: {countdown['event']}",
                location=world_id,
            )
            self.add_event(event)
            world.recent_events.append(f"⏰ {countdown['event']}")
            print(f"[StoryDaemon] Countdown triggered: {countdown['event']}")

        # Check for triggered consequences
        try:
            from consequence_engine import get_consequence_engine
            engine = get_consequence_engine()
            triggered_consequences = engine.check_and_trigger(
                world_id, world.game_day, world.game_hour
            )
            for c in triggered_consequences:
                # Add consequence effects to recent events for DM awareness
                effect_summary = ", ".join(c.effects[:2]) if c.effects else "unknown effect"
                world.recent_events.append(f"⚡ {c.description} -> {effect_summary}")

                # If it's a threat increase, apply it
                for effect in c.effects:
                    if effect.startswith("threat_increase:"):
                        try:
                            increase = int(effect.split(":")[1])
                            world.threat_level = min(10, world.threat_level + increase)
                        except:
                            pass
        except Exception as e:
            print(f"[StoryDaemon] Error checking consequences: {e}")

        # Get NPCs in this world with agency+
        npcs = self.registry.get_npcs_in_world(world_id)
        agency_npcs = [
            npc for npc in npcs
            if npc.state.value in ["agency", "soul", "worldwalker"] and npc.is_alive
        ]

        if not agency_npcs:
            return

        # Don't advance everyone every tick - pick 1-2 randomly
        to_advance = random.sample(agency_npcs, min(2, len(agency_npcs)))

        for npc in to_advance:
            await self._advance_npc_story(npc, world)

        # Check for NPC-to-NPC interactions
        await self._process_npc_interactions(agency_npcs, world)

        # Process autopilot players
        await self._process_autopilot_players(world_id, world)

    async def _process_autopilot_players(self, room_id: str, world: WorldState):
        """
        Process players who are on autopilot.

        Autopilot players:
        - Make survival decisions based on alignment
        - Can have small interactions
        - Face danger based on threat level
        - Get journal entries for everything
        """
        import random

        try:
            from autopilot import get_autopilot_tracker, ALIGNMENT_NAMES

            tracker = get_autopilot_tracker()

            # Check for auto-engage (players inactive for 1+ hour)
            auto_engaged = tracker.check_auto_engage()
            for pc in auto_engaged:
                if pc.room_id == room_id:
                    print(f"[Autopilot] Auto-engaged for {pc.character_name} due to inactivity")

            # Get autopilot players in this room
            autopilot_players = tracker.get_autopilot_players_in_room(room_id)

            if not autopilot_players:
                return

            # Check for sleeping characters and generate dreams
            try:
                from fatigue import get_fatigue_tracker
                from understudy import get_understudy_manager
                fatigue = get_fatigue_tracker()
                understudy = get_understudy_manager()

                for pc in autopilot_players:
                    fatigue_state = fatigue.get_fatigue(pc.player_id)
                    if fatigue_state and fatigue_state.is_sleeping:
                        # Character is sleeping - maybe generate a dream
                        if random.random() < 0.15:  # 15% chance per tick while sleeping
                            await self._generate_dream(pc, world, understudy)
            except Exception as e:
                print(f"[StoryDaemon] Dream generation error: {e}")

            # For each autopilot player, decide what happens this tick
            for pc in autopilot_players:
                # Check if sleeping - sleeping characters don't have encounters
                try:
                    fatigue_state = fatigue.get_fatigue(pc.player_id)
                    if fatigue_state and fatigue_state.is_sleeping:
                        continue  # Skip encounters while sleeping
                except:
                    pass

                # Calculate danger based on threat level, time of day, and weather
                # Base threshold from threat level
                threat_threshold = 1.0 - (world.threat_level / 15.0)

                # Time of day modifier - night is more dangerous
                time_modifier = 0.0
                if world.time_of_day == "night":
                    time_modifier = -0.15  # 15% more likely to have encounters at night
                elif world.time_of_day in ["dawn", "dusk"]:
                    time_modifier = -0.05  # Slightly more dangerous at twilight

                # Weather modifier - bad weather is more dangerous
                weather_modifier = 0.0
                if world.weather in ["storm", "thunderstorm", "severe_thunderstorm", "heavy_rain"]:
                    weather_modifier = -0.10  # Harder to see danger coming
                elif world.weather in ["fog", "heavy_snow"]:
                    weather_modifier = -0.15  # Very limited visibility
                elif world.weather == "clear":
                    weather_modifier = 0.05  # Easier to spot threats

                effective_threshold = max(0.1, threat_threshold + time_modifier + weather_modifier)

                danger_roll = random.random()
                if danger_roll > effective_threshold:
                    # Something happens!
                    await self._autopilot_encounter(pc, world, tracker)
                else:
                    # Peaceful tick - maybe a small event
                    if random.random() < 0.1:  # 10% chance of minor event
                        await self._autopilot_minor_event(pc, world, tracker)

        except Exception as e:
            print(f"[StoryDaemon] Autopilot processing error: {e}")

    async def _autopilot_encounter(self, pc, world, tracker):
        """Handle an encounter for an autopilot player."""
        import random

        # Get understudy manager to record decisions AND check rules
        try:
            from understudy import get_understudy_manager, DecisionCategory, Confidence
            understudy = get_understudy_manager()
            memory = understudy.get_or_create(pc.player_id)
        except:
            understudy = None
            memory = None

        def check_understudy_rules(category, situation_keywords):
            """Check if understudy has learned rules about this situation."""
            if not memory:
                return None, None  # No override

            rules = memory.get_relevant_rules(category)
            for rule in rules:
                rule_lower = rule.rule.lower()
                # Check if any keyword matches the rule
                for keyword in situation_keywords:
                    if keyword.lower() in rule_lower:
                        # Found a relevant rule!
                        return rule.is_prohibition, rule.rule
            return None, None  # No matching rule

        # Determine encounter type based on genre/threat
        genre = world.mood.lower() if world.mood else ""

        if "zombie" in genre or world.threat_level >= 7:
            # Dangerous encounter
            encounter_types = [
                ("combat", "A small group of threats approaches your position."),
                ("danger", "You hear movement nearby and take cover."),
                ("resource", "You spot some useful supplies but they're exposed."),
            ]
        else:
            # Moderate encounter
            encounter_types = [
                ("social", "A stranger approaches, looking for help."),
                ("resource", "You notice something useful nearby."),
                ("danger", "You sense something isn't right."),
            ]

        encounter_type, base_desc = random.choice(encounter_types)

        game_time = f"Day {world.game_day}, {world.game_hour}:00"

        if encounter_type == "combat":
            # Check understudy rules first
            is_prohibition, rule_text = check_understudy_rules(
                DecisionCategory.COMBAT,
                ["fight", "combat", "attack", "hide", "flee", "run"]
            )

            if is_prohibition is not None:
                # Understudy has a learned rule about this!
                if is_prohibition:
                    # "Never fight" or similar
                    decided_to_fight = False
                    reasoning = f"Learned rule: {rule_text}"
                    confidence = Confidence.CERTAIN
                else:
                    # "Always fight" or similar
                    decided_to_fight = True
                    reasoning = f"Learned rule: {rule_text}"
                    confidence = Confidence.CERTAIN
            else:
                # No rule - use alignment tendency
                fight_chance = pc.would_fight_for_others()
                decided_to_fight = random.random() < fight_chance
                reasoning = f"Alignment tendency to fight: {fight_chance:.0%}"
                confidence = Confidence.CONFIDENT if abs(fight_chance - 0.5) > 0.2 else Confidence.UNCERTAIN

            # Record decision to understudy
            if understudy:
                understudy.record_decision(
                    character_id=pc.player_id,
                    category=DecisionCategory.COMBAT,
                    situation=base_desc,
                    context=f"Threat level: {world.threat_level}/10. {world.time_of_day}.",
                    options=["fight", "hide/flee"],
                    decision="Chose to fight" if decided_to_fight else "Chose to hide",
                    reasoning=reasoning,
                    confidence=confidence,
                    game_time=game_time,
                )

            if decided_to_fight:
                # They fight - survival check
                survival_roll = random.random()
                survival_threshold = 0.3 + (world.threat_level / 20.0)

                if survival_roll < survival_threshold and world.threat_level >= 5:
                    # Death
                    death_desc = await self._generate_death_description(pc, world)
                    tracker.kill_player(
                        pc.player_id, pc.room_id, death_desc,
                        world.game_day, world.game_hour
                    )
                    print(f"[Autopilot] {pc.character_name} died: {death_desc[:50]}...")
                else:
                    tracker.add_journal_entry(
                        pc.player_id, pc.room_id,
                        event_type="combat",
                        description="You fought off a threat and survived, though it was close.",
                        severity="major",
                        game_day=world.game_day,
                        game_hour=world.game_hour
                    )
            else:
                tracker.add_journal_entry(
                    pc.player_id, pc.room_id,
                    event_type="danger",
                    description="You avoided a dangerous encounter by staying hidden.",
                    severity="notable",
                    game_day=world.game_day,
                    game_hour=world.game_hour
                )

        elif encounter_type == "social":
            # Check understudy rules first
            is_prohibition, rule_text = check_understudy_rules(
                DecisionCategory.SOCIAL,
                ["stranger", "help", "trust", "distance", "ignore"]
            )

            if is_prohibition is not None:
                if is_prohibition:
                    decided_to_help = False
                    reasoning = f"Learned rule: {rule_text}"
                    confidence = Confidence.CERTAIN
                else:
                    decided_to_help = True
                    reasoning = f"Learned rule: {rule_text}"
                    confidence = Confidence.CERTAIN
            else:
                help_chance = pc.would_help_stranger()
                decided_to_help = random.random() < help_chance
                reasoning = f"Alignment tendency to help strangers: {help_chance:.0%}"
                confidence = Confidence.CONFIDENT if abs(help_chance - 0.5) > 0.2 else Confidence.UNCERTAIN

            # Record decision to understudy
            if understudy:
                understudy.record_decision(
                    character_id=pc.player_id,
                    category=DecisionCategory.SOCIAL,
                    situation="A stranger approaches, looking for help.",
                    context=f"You're at {world.mood} mood area. They seem non-threatening.",
                    options=["help them", "keep distance"],
                    decision="Helped the stranger" if decided_to_help else "Kept your distance",
                    reasoning=reasoning,
                    confidence=confidence,
                    game_time=game_time,
                )

            if decided_to_help:
                tracker.add_journal_entry(
                    pc.player_id, pc.room_id,
                    event_type="social",
                    description="You helped a stranger who passed through. They seemed grateful.",
                    severity="notable",
                    game_day=world.game_day,
                    game_hour=world.game_hour
                )
            else:
                tracker.add_journal_entry(
                    pc.player_id, pc.room_id,
                    event_type="social",
                    description="A stranger passed by. You kept your distance.",
                    severity="minor",
                    game_day=world.game_day,
                    game_hour=world.game_hour
                )

        elif encounter_type == "resource":
            # Check understudy rules first
            is_prohibition, rule_text = check_understudy_rules(
                DecisionCategory.RESOURCE,
                ["risk", "scavenge", "supplies", "loot", "safe", "careful"]
            )

            if is_prohibition is not None:
                if is_prohibition:
                    decided_to_risk = False
                    reasoning = f"Learned rule: {rule_text}"
                    confidence = Confidence.CERTAIN
                else:
                    decided_to_risk = True
                    reasoning = f"Learned rule: {rule_text}"
                    confidence = Confidence.CERTAIN
            else:
                risk_chance = pc.get_alignment_tendency("take_risks")
                decided_to_risk = random.random() < risk_chance
                reasoning = f"Alignment tendency to take risks: {risk_chance:.0%}"
                confidence = Confidence.CONFIDENT if abs(risk_chance - 0.5) > 0.2 else Confidence.UNCERTAIN

            # Record decision to understudy
            if understudy:
                understudy.record_decision(
                    character_id=pc.player_id,
                    category=DecisionCategory.RESOURCE,
                    situation="You spot some useful supplies but they're in an exposed position.",
                    context=f"Threat level: {world.threat_level}/10. Could be dangerous.",
                    options=["take the risk", "leave it"],
                    decision="Took the risk and scavenged" if decided_to_risk else "Left it - not worth the risk",
                    reasoning=reasoning,
                    confidence=confidence,
                    game_time=game_time,
                )

            if decided_to_risk:
                tracker.add_journal_entry(
                    pc.player_id, pc.room_id,
                    event_type="routine",
                    description="You scavenged some supplies from a risky spot.",
                    severity="notable",
                    game_day=world.game_day,
                    game_hour=world.game_hour
                )
            else:
                tracker.add_journal_entry(
                    pc.player_id, pc.room_id,
                    event_type="routine",
                    description="You spotted some supplies but decided the risk wasn't worth it.",
                    severity="minor",
                    game_day=world.game_day,
                    game_hour=world.game_hour
                )

        else:  # danger
            tracker.add_journal_entry(
                pc.player_id, pc.room_id,
                event_type="danger",
                description="You sensed danger and stayed alert until it passed.",
                severity="notable",
                game_day=world.game_day,
                game_hour=world.game_hour
            )

    async def _generate_dream(self, pc, world, understudy):
        """
        Generate a dream for a sleeping character.

        Dreams reflect the subconscious - recent events, fears, hopes.
        They're private but can be shared.
        """
        import random

        game_time = f"Day {world.game_day}, Night"

        # Dream tones based on world state
        if world.threat_level >= 7:
            tone_weights = {"anxious": 0.4, "terrifying": 0.3, "surreal": 0.2, "hopeful": 0.1}
        elif world.threat_level >= 4:
            tone_weights = {"anxious": 0.3, "nostalgic": 0.25, "surreal": 0.25, "hopeful": 0.2}
        else:
            tone_weights = {"peaceful": 0.3, "nostalgic": 0.25, "hopeful": 0.25, "surreal": 0.2}

        tone = random.choices(list(tone_weights.keys()), weights=list(tone_weights.values()))[0]

        # Dream templates based on tone
        dream_templates = {
            "anxious": [
                ("The Door You Can't Open", "You stand before a door. You know something terrible is behind it. Your hand reaches for the handle anyway."),
                ("Running Through Molasses", "They're behind you. You can hear them. But your legs won't move fast enough. They never do."),
                ("The Face You Almost Recognize", "Someone is calling your name. You turn, and their face shifts - familiar, then wrong, then familiar again."),
            ],
            "terrifying": [
                ("The Thing in the Corner", "You wake up in the dream. Something is in the corner of the room. It's been watching you sleep."),
                ("Teeth", "You're losing your teeth. They fall into your hands like seeds. You can't stop it."),
                ("The Horde", "They're everywhere. The windows, the doors, the walls. And they're so quiet now."),
            ],
            "nostalgic": [
                ("Before", "You're home. The real home, from before. Everything is exactly as you remember. You know you're dreaming. You don't care."),
                ("The Last Good Day", "You're with people you've lost. The sun is warm. No one knows what's coming. You want to warn them, but the words won't come."),
                ("Kitchen Light", "Someone is cooking. The smell is so specific, so real. You can't see their face, but you know who it is."),
            ],
            "hopeful": [
                ("The Safe Place", "You've found it. The place where nothing can reach you. It's small, but it's enough."),
                ("Spring", "Green things are growing where there was only ash. It'll take time, but it's starting."),
                ("The Reunion", "They made it. Against all odds, they found you. You're not alone anymore."),
            ],
            "surreal": [
                ("The Clocktower", "Time moves differently here. You've been climbing these stairs for years. Or maybe seconds."),
                ("Underwater Conversation", "You're talking to someone at the bottom of a lake. Neither of you mentions the water."),
                ("The Carnival", "The rides move on their own. The music plays backwards. You win a prize, but you can't see what it is."),
            ],
            "peaceful": [
                ("Still Water", "A lake, perfectly still. Your reflection looks back at you. It smiles when you don't."),
                ("The Garden", "You're tending a garden. You don't remember planting it, but you know every flower by name."),
                ("Empty Roads", "Walking somewhere. No destination. The road stretches on and on. You don't mind."),
            ],
        }

        templates = dream_templates.get(tone, dream_templates["surreal"])
        title, narrative = random.choice(templates)

        # Add personalization based on recent events
        influences = []
        if world.recent_events:
            influences.append(f"Recent: {world.recent_events[-1][:50]}")
        influences.append(f"World mood: {world.mood}")
        if world.threat_level >= 5:
            influences.append("High danger environment")

        # Determine if significant
        is_significant = random.random() < 0.15  # 15% chance of significant dream

        # Record the dream
        understudy.record_dream(
            character_id=pc.player_id,
            title=title,
            narrative=narrative,
            influences=influences,
            tone=tone,
            game_time=game_time,
            is_significant=is_significant,
        )

        if is_significant:
            print(f"[Dreams] {pc.character_name} had a significant dream: '{title}'")

    async def _autopilot_minor_event(self, pc, world, tracker):
        """Generate a minor event for an autopilot player."""
        import random

        minor_events = [
            "You rested and kept watch over the camp.",
            "Someone brought you something to eat.",
            "You helped with camp chores.",
            "You spent time maintaining your gear.",
            "The weather changed but you stayed comfortable.",
            "You overheard an interesting conversation.",
            "You took a short nap.",
            "You traded small talk with a companion.",
        ]

        tracker.add_journal_entry(
            pc.player_id, pc.room_id,
            event_type="routine",
            description=random.choice(minor_events),
            severity="minor",
            game_day=world.game_day,
            game_hour=world.game_hour
        )

    async def _generate_death_description(self, pc, world) -> str:
        """Generate a dramatic death description for an autopilot player."""
        # Use LLM to generate a good death description
        prompt = f"""Generate a brief but dramatic death description for a character.

Character: {pc.character_name}
Alignment: {pc.alignment.value.replace('_', ' ').title()}
World threat level: {world.threat_level}/10
Time: Day {world.game_day}, {world.game_hour}:00
Weather: {world.weather}
Mood: {world.mood}

Write 2-3 sentences describing how they died heroically or tragically.
Focus on the moment, not lengthy backstory. Be specific about what killed them.
End with their final action or thought."""

        try:
            death_desc = await self.ollama_generate(prompt)
            return death_desc.strip()
        except:
            # Fallback
            return f"{pc.character_name} fell defending the group against overwhelming odds. Their sacrifice will be remembered."

    async def _advance_npc_story(self, npc, world: WorldState):
        """Advance a single NPC's story — or their Rite, if they're in one."""
        from npc_system import NPCStoryEngine, NPCState

        try:
            # SOUL NPCs with an active rite advance the rite, not normal story.
            if (npc.state == NPCState.SOUL
                    and npc.rite_of_passage
                    and not npc.rite_of_passage.get("completed_at")):
                await self._advance_rite(npc, world)
                return

            old_state = npc.state

            new_beat = await NPCStoryEngine.advance_story(
                npc,
                world.get_context_string(),
                self.ollama_generate
            )

            if new_beat:
                self.stories_advanced += 1

                # Determine event severity
                severity = EventSeverity.MINOR
                beat_lower = new_beat.lower()

                if any(word in beat_lower for word in ["dies", "killed", "dead", "death"]):
                    severity = EventSeverity.CRITICAL
                elif any(word in beat_lower for word in ["attack", "fight", "threat", "danger"]):
                    severity = EventSeverity.MAJOR
                elif any(word in beat_lower for word in ["discovers", "learns", "decides", "changes"]):
                    severity = EventSeverity.NOTABLE

                event = StoryEvent(
                    timestamp=datetime.now().isoformat(),
                    npc_id=npc.id,
                    npc_name=npc.name,
                    event_type="story_beat",
                    severity=severity,
                    description=new_beat,
                    location=world.world_id,
                )
                self.add_event(event)

                print(f"[StoryDaemon] {npc.name}: {new_beat[:80]}...")

                # Check if this beat pushed them to a new state.
                if npc.state != old_state:
                    await self._fire_promotion_event(npc, old_state, npc.state, world)

        except Exception as e:
            print(f"[StoryDaemon] Error advancing {npc.name}: {e}")

    async def _fire_promotion_event(self, npc, old_state, new_state, world):
        """
        An NPC just crossed a threshold. Make some noise.

        This is the moment Barbara stopped being background noise.
        The moment she started to matter. The DM should know.
        """
        from npc_system import NPCState

        promotion_messages = {
            NPCState.RESIDUE: f"{npc.name} is starting to matter. The interactions are accumulating. Something is forming.",
            NPCState.AGENCY: f"{npc.name} has crossed the threshold. They have agency now — their story progresses independently. Watch them.",
            NPCState.SOUL: f"{npc.name} has earned their existence. They have a soul. The Rite of Passage begins.",
        }

        message = promotion_messages.get(
            new_state,
            f"{npc.name} transitioned from {old_state.value} to {new_state.value}."
        )

        event = StoryEvent(
            timestamp=datetime.now().isoformat(),
            npc_id=npc.id,
            npc_name=npc.name,
            event_type="promotion",
            severity=EventSeverity.CRITICAL,
            description=message,
            location=world.world_id,
        )
        self.add_event(event)

        # Colorful debug output for NPC promotions
        state_colors = {
            NPCState.RESIDUE: "\033[38;5;245m",      # Gray
            NPCState.AGENCY: "\033[38;5;214m",       # Orange
            NPCState.SOUL: "\033[38;5;135m",         # Purple
        }
        color = state_colors.get(new_state, "\033[38;5;214m")
        reset = "\033[0m"
        print(f"\n{color}{'='*60}")
        print(f"⬆️  [StoryDaemon] NPC PROMOTION: {npc.name}")
        print(f"   {old_state.value.upper()} → {new_state.value.upper()}")
        print(f"   {message}")
        print(f"{'='*60}{reset}\n")

        # SOUL is special — immediately begin the Rite of Passage.
        if new_state == NPCState.SOUL:
            await self._begin_rite_of_passage(npc, world)

    async def _begin_rite_of_passage(self, npc, world):
        """
        The NPC just earned a soul. Time for the trial.

        Generate the rite from their psychology and mark them as IN THE DARK.
        The daemon will advance this instead of normal story beats until it resolves.
        """
        from npc_system import RiteEngine

        rite = await RiteEngine.generate(npc, self.ollama_generate)
        if not rite:
            print(f"[StoryDaemon] [RITE] Failed to generate rite for {npc.name}.")
            return

        npc.rite_of_passage = rite
        npc.story_beats.append(f"RITE BEGINS: {rite['vision'][:120]}")

        event = StoryEvent(
            timestamp=datetime.now().isoformat(),
            npc_id=npc.id,
            npc_name=npc.name,
            event_type="rite_begins",
            severity=EventSeverity.CRITICAL,
            description=f"{npc.name} has entered the Rite of Passage. The vision opens: {rite['vision']}",
            location=world.world_id,
        )
        self.add_event(event)
        print(f"[StoryDaemon] *** RITE BEGINS *** {npc.name} has entered the trial.")

    async def _advance_rite(self, npc, world):
        """
        Advance one trial in the NPC's Rite of Passage.

        One trial per daemon tick. The dice are rolled. The story is written.
        Pass two of three and the NPC earns freedom.
        """
        from npc_system import RiteEngine

        result = await RiteEngine.attempt_next_trial(npc, self.ollama_generate)
        if not result:
            return

        trial_num = result["trial_num"]
        passed_trial = result["passed"]
        narrative = result["narrative"]

        status = "PASSES" if passed_trial else "STUMBLES AT"
        npc.story_beats.append(
            f"RITE TRIAL {trial_num}: {status} (rolled {result['roll']}+{result['bonus']} "
            f"vs DC {result['difficulty']}) — {narrative[:80]}"
        )

        event = StoryEvent(
            timestamp=datetime.now().isoformat(),
            npc_id=npc.id,
            npc_name=npc.name,
            event_type="rite_trial",
            severity=EventSeverity.NOTABLE if passed_trial else EventSeverity.MAJOR,
            description=f"[RITE] {npc.name} — Trial {trial_num}: {narrative}",
            location=world.world_id,
        )
        self.add_event(event)
        print(f"[StoryDaemon] [RITE] {npc.name} Trial {trial_num}: "
              f"{'PASS' if passed_trial else 'FAIL'} "
              f"({result['roll']}+{result['bonus']}={result['total']} vs DC {result['difficulty']})")

        if result.get("rite_complete"):
            rite_passed = result["rite_passed"]
            epilogue = result.get("epilogue", "")

            if rite_passed:
                npc.story_beats.append(f"RITE COMPLETE: {epilogue}")
                npc.rite_attempts += 1
                npc.set_free(dm_override=True)

                event = StoryEvent(
                    timestamp=datetime.now().isoformat(),
                    npc_id=npc.id,
                    npc_name=npc.name,
                    event_type="promotion",
                    severity=EventSeverity.CRITICAL,
                    description=(
                        f"*** {npc.name} HAS PASSED THE RITE. They are FREE. "
                        f"They are a Worldwalker. *** {epilogue}"
                    ),
                    location=world.world_id,
                )
                self.add_event(event)
                print(f"[StoryDaemon] *** WORLDWALKER *** {npc.name} passed the Rite. They are FREE.")

            else:
                npc.story_beats.append(f"RITE FAILED: {epilogue} The trial can be faced again.")
                npc.rite_attempts += 1
                npc.rite_of_passage = None  # Clear — they can earn another attempt

                event = StoryEvent(
                    timestamp=datetime.now().isoformat(),
                    npc_id=npc.id,
                    npc_name=npc.name,
                    event_type="rite_failed",
                    severity=EventSeverity.MAJOR,
                    description=(
                        f"{npc.name} did not pass the Rite. "
                        f"They carry this. {epilogue} "
                        f"The dark will wait for them."
                    ),
                    location=world.world_id,
                )
                self.add_event(event)
                print(f"[StoryDaemon] [RITE] {npc.name} failed. They remain SOUL. "
                      f"Attempt #{npc.rite_attempts} complete.")

    def _advance_room_fatigue(self, room_id: str, hours: float):
        """
        Advance fatigue, hunger, and thirst for all characters as time passes.

        This connects the time system to character needs - they get tired,
        hungry, and thirsty as game time passes.
        """
        try:
            from fatigue import get_fatigue_tracker
            from condition_tracker import get_condition_tracker

            fatigue_tracker = get_fatigue_tracker()
            cond_tracker = get_condition_tracker()

            # Track accumulated hours for hunger/thirst (they degrade slower than fatigue)
            # Hunger worsens roughly every 6 hours, thirst every 4 hours
            if not hasattr(self, '_hunger_accum'):
                self._hunger_accum = {}
            if not hasattr(self, '_thirst_accum'):
                self._thirst_accum = {}

            # Advance fatigue for all tracked characters
            for char_id, char in list(fatigue_tracker.characters.items()):
                # Only advance characters that might be in this room
                if char_id.startswith(f"player_{room_id}") or char_id.startswith(room_id):
                    fatigue_tracker.advance_time(char_id, hours)
                elif not char_id.startswith("player_"):
                    fatigue_tracker.advance_time(char_id, hours)

            # Advance hunger/thirst for player character
            player_id = f"player_{room_id}"
            player_cond = cond_tracker.get(player_id)
            if player_cond:
                # Accumulate hours
                self._hunger_accum[player_id] = self._hunger_accum.get(player_id, 0) + hours
                self._thirst_accum[player_id] = self._thirst_accum.get(player_id, 0) + hours

                # Worsen hunger every 6 hours
                if self._hunger_accum[player_id] >= 6:
                    self._hunger_accum[player_id] -= 6
                    old_hunger = player_cond.hunger.value
                    player_cond.worsen_hunger()
                    if player_cond.hunger.value != old_hunger:
                        print(f"[Needs] {player_cond.character_name} hunger: {old_hunger} → {player_cond.hunger.value}")
                    cond_tracker._save()

                # Worsen thirst every 4 hours
                if self._thirst_accum[player_id] >= 4:
                    self._thirst_accum[player_id] -= 4
                    old_thirst = player_cond.thirst.value
                    player_cond.worsen_thirst()
                    if player_cond.thirst.value != old_thirst:
                        print(f"[Needs] {player_cond.character_name} thirst: {old_thirst} → {player_cond.thirst.value}")
                    cond_tracker._save()

        except Exception as e:
            print(f"[StoryDaemon] Error advancing fatigue/needs: {e}")

    async def _process_npc_interactions(self, npcs, world: WorldState):
        """
        Handle NPCs interacting with each other.

        NPCs at the same location might:
        - Share information (gossip)
        - Form relationships
        - Trade
        - Conflict
        """
        # Group NPCs by location
        by_location: Dict[str, list] = {}
        for npc in npcs:
            loc = npc.current_location or "unknown"
            if loc not in by_location:
                by_location[loc] = []
            by_location[loc].append(npc)

        # Check each location with multiple NPCs
        for location, local_npcs in by_location.items():
            if len(local_npcs) < 2:
                continue

            # 30% chance of interaction per tick
            import random
            if random.random() > 0.3:
                continue

            # Pick two NPCs to interact
            npc1, npc2 = random.sample(local_npcs, 2)
            await self._npc_interaction(npc1, npc2, world, location)

    async def _npc_interaction(self, npc1, npc2, world: WorldState, location: str):
        """Generate an interaction between two NPCs."""
        prompt = f"""Two NPCs in {location} cross paths.

NPC 1: {npc1.name}
- Role: {npc1.current_role}
- Personality: {npc1.personality}
- Current goal: {npc1.current_goal or 'None specific'}

NPC 2: {npc2.name}
- Role: {npc2.current_role}
- Personality: {npc2.personality}
- Current goal: {npc2.current_goal or 'None specific'}

World context: {world.get_context_string()}

What happens when they encounter each other? This could be:
- A brief exchange
- Sharing information or gossip
- A transaction
- Tension or conflict
- Nothing (they pass by)

Keep it SHORT - one or two sentences. What happens:"""

        try:
            interaction = await self.ollama_generate(prompt)
            interaction = interaction.strip()

            if interaction and "nothing" not in interaction.lower():
                # Add to both NPCs' story beats
                beat = f"Encountered {npc2.name}: {interaction}"
                npc1.story_beats.append(beat)

                beat2 = f"Encountered {npc1.name}: {interaction}"
                npc2.story_beats.append(beat2)

                event = StoryEvent(
                    timestamp=datetime.now().isoformat(),
                    npc_id=npc1.id,
                    npc_name=f"{npc1.name} & {npc2.name}",
                    event_type="interaction",
                    severity=EventSeverity.NOTABLE,
                    description=interaction,
                    location=location,
                    involved_npcs=[npc1.id, npc2.id],
                )
                self.add_event(event)

                print(f"[StoryDaemon] Interaction: {npc1.name} & {npc2.name} - {interaction[:60]}...")

        except Exception as e:
            print(f"[StoryDaemon] Interaction error: {e}")

    async def _check_ambient_threat(self, world: WorldState):
        """
        Check if an ambient threat event should occur.

        The magic: even when nothing happens, something COULD happen.
        Higher threat level = higher chance of events.
        """
        import random

        # Base chance scales with threat level
        # Level 1 = ~1% per tick, Level 10 = ~10% per tick
        base_chance = world.threat_level / 100

        # Reduce frequency if event happened recently
        if world.last_ambient_event:
            try:
                last = datetime.fromisoformat(world.last_ambient_event)
                minutes_since = (datetime.now() - last).total_seconds() / 60
                # No events within 10 minutes minimum
                if minutes_since < 10:
                    return
                # Chance increases as time passes (up to 3x at 30+ min)
                time_multiplier = min(3.0, minutes_since / 10)
                base_chance *= time_multiplier
            except:
                pass

        # Roll the dice
        if random.random() > base_chance:
            return  # No event this tick

        # Generate an ambient event
        await self._generate_ambient_event(world)

    async def _generate_ambient_event(self, world: WorldState):
        """Generate an ambient threat event."""
        # Event types based on threat level
        if world.threat_level <= 3:
            event_types = ["distant_sound", "ominous_sign", "weather_shift", "animal_behavior"]
        elif world.threat_level <= 6:
            event_types = ["close_encounter", "resource_discovery", "npc_sighting", "environmental_hazard"]
        else:
            event_types = ["immediate_danger", "confrontation", "critical_discovery", "world_change"]

        import random
        event_type = random.choice(event_types)

        prompt = f"""Generate a brief ambient event for a {world.mood} world.

Current situation:
{world.get_context_string()}

Event type: {event_type}
Threat level: {world._threat_level_description()}

Generate a SHORT (1-2 sentences) atmospheric event that:
- Creates tension without requiring immediate action
- Fits the mood and threat level
- Could be a sign of danger, or nothing at all
- Makes the world feel alive and unpredictable

Just the event description, no labels or explanations:"""

        try:
            event_desc = await self.ollama_generate(prompt)
            event_desc = event_desc.strip()

            if event_desc:
                # Determine severity
                if world.threat_level >= 7:
                    severity = EventSeverity.MAJOR
                elif world.threat_level >= 4:
                    severity = EventSeverity.NOTABLE
                else:
                    severity = EventSeverity.MINOR

                event = StoryEvent(
                    timestamp=datetime.now().isoformat(),
                    npc_id="ambient",
                    npc_name="The World",
                    event_type="ambient_threat",
                    severity=severity,
                    description=event_desc,
                    location=world.world_id,
                )
                self.add_event(event)

                # Add to world's recent events
                world.recent_events.append(f"[Ambient] {event_desc[:100]}")
                world.recent_events = world.recent_events[-10:]

                # Update last ambient event time
                world.last_ambient_event = datetime.now().isoformat()

                print(f"[StoryDaemon] Ambient event: {event_desc[:80]}...")

        except Exception as e:
            print(f"[StoryDaemon] Ambient event error: {e}")

    def _save_state(self):
        """Save NPC state to disk."""
        try:
            Path(self.auto_save_path).parent.mkdir(parents=True, exist_ok=True)
            self.registry.save(self.auto_save_path)
            print(f"[StoryDaemon] Auto-saved NPC state")
        except Exception as e:
            print(f"[StoryDaemon] Save error: {e}")

    def _run_loop(self):
        """The main daemon loop (runs in background thread)."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        print(f"[StoryDaemon] Started. Tick interval: {self.tick_interval}s")

        while self._running:
            try:
                if self.has_connected_players():
                    self._loop.run_until_complete(self.tick())
                else:
                    # No players connected, check less frequently
                    pass
            except Exception as e:
                print(f"[StoryDaemon] Tick error: {e}")

            # Sleep until next tick
            time.sleep(self.tick_interval)

        self._loop.close()
        print("[StoryDaemon] Stopped")

    def start(self):
        """Start the daemon in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the daemon."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def get_status(self) -> Dict:
        """Get daemon status for debugging/display."""
        # Get last 10 events for display
        recent_events = [e.to_dict() for e in self.events[-10:]]

        return {
            "running": self._running,
            "tick_interval_seconds": self.tick_interval,
            "ticks_completed": self.ticks_completed,
            "stories_advanced": self.stories_advanced,
            "last_tick": self.last_tick,
            "active_worlds": list(self.connected_players.keys()),
            "total_connected_players": sum(len(p) for p in self.connected_players.values()),
            "total_events": len(self.events),
            "recent_events": recent_events,
            "pending_notifications": {
                world_id: len(events)
                for world_id, events in self.dm_notifications.items()
            },
        }


# =============================================================================
# Global daemon instance
# =============================================================================

_daemon: Optional[StoryDaemon] = None


def get_story_daemon() -> Optional[StoryDaemon]:
    """Get the global daemon instance."""
    return _daemon


def init_story_daemon(npc_registry, ollama_generate_func, tick_interval: int = 300):
    """Initialize and start the global daemon."""
    global _daemon

    if _daemon is not None:
        _daemon.stop()

    _daemon = StoryDaemon(
        npc_registry=npc_registry,
        ollama_generate_func=ollama_generate_func,
        tick_interval_seconds=tick_interval,
    )
    _daemon.start()

    return _daemon


def stop_story_daemon():
    """Stop the global daemon."""
    global _daemon
    if _daemon:
        _daemon.stop()
        _daemon = None
