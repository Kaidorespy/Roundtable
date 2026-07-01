# DM System Redesign

## Philosophy

Based on "So You Want To Be A Game Master" by Justin Alexander and our own observations of what's broken.

### Core Principles

1. **Default to Yes** - Unless there's a specific reason not to, players succeed and discover things
2. **Yes, but... / No, but...** - Success with complication, or can't do that but here's an alternative
3. **Failure must be meaningful** - If failure just means "try again," skip the roll
4. **The DM is also discovering** - Sometimes roll to find out, "let's find out together"
5. **Raw material, not experience** - The system provides raw material; the DM makes it specific and alive
6. **Permissive clue-finding** - When players look for information, default to them finding something useful
7. **Concrete details** - Not "you search successfully" but "you find rabbit droppings and a tuft of gray fur"
8. **Don't summarize player actions** - They already told you what they're doing. Tell them what they FIND.

### What the Old Council System Got Wrong

- Framed everything as success/failure adjudication
- Summarized instead of revealed
- Didn't distinguish "what do I find?" from "do I succeed?"
- No concept of "let me check" / discovering together
- DM woke up with no foundation, made rulings from fog

---

## New Architecture

### Layer 1: Parser (runs on EVERY message)

The observer. Mostly passive - 90% of the time it's accumulating information, not triggering action.

**Annotates each message with:**
- `type`: dialogue, action, discovery_attempt, movement, etc.
- `noise_level`: silent, normal, loud (gunshot, scream, breaking glass)
- `movement`: none, local, travel
- `triggers`: immediate consequences (opening the freezer full of zombies)
- `establishments`: player-created locations/facts that become canon

**Key insight:** Parser catches the freezer moment BEFORE it becomes a DM question. It flags triggers so the DM already knows.

---

### Layer 2: Foundation Systems (background, but responsive)

#### Cartographer Light (immediate area)

Maintains location state for the current area. Not the whole world - that's the existing Cartographer.

**Tracks:**
- Known locations: "The island has: a cabin (you're here), gas station 2 miles east, overgrown supermarket, radio tower (visible), scattered vacation homes"
- Player-established locations: If player says "there's a shed out back," that shed exists now
- Explored vs unexplored vs possible

**Updates when:** Movement detected, new location established by player or DM

---

#### Stagehand (discoverable radius)

Maintains what COULD be discovered in the current area. Not locked in - possibilities, not facts.

**Example for forest around cabin:**
```
{
  "terrain": "dense forest, muddy ground, deadfall",
  "wildlife_signs": "likely",
  "small_game": "rabbit/squirrel activity probable",
  "water": "creek 500 yards north (established)",
  "structures": "old hunter's cache (possible, not confirmed)",
  "foraging": "edible plants possible, nothing confirmed"
}
```

**Key insight:** When DM gets asked about tracks, Stagehand already has "small game activity probable" - DM rolls with favorable odds, adds specificity.

**Updates when:** Location changes significantly, OR periodic refresh every N messages in same area

---

#### Tension Keeper (ambient threat state)

Knows where danger is. Tracks threat drift. Knows about dormant triggers.

**Tracks:**
- Zombie density by zone: cabin area = low, supermarket = high, gas station = medium (contained)
- Threat drift: gunshot at cabin → zombies from nearest zones start drifting (hours away, but moving)
- Dormant triggers: "freezer at gas station = dormant swarm, trigger = door opens"
- Alert states: what's asleep, what's hunting, what's drifting

**Responds to:** Noise events from Parser, time passing, trigger flags

**Key insight:** When player says "I open the freezer," Parser flags it, Tension Keeper already knows this is a trap. DM doesn't deliberate - it KNOWS.

---

#### Thread Keeper (narrative diary)

Gives DM context so it's not waking up fresh. Relevant callbacks. Story beats.

**Tracks:**
- Dangling threads: "they're hunting because starving", "Grace promised gravy"
- Character motivations: what they've said they want
- Unresolved questions: "what's at the radio tower?"
- Recent emotional beats: the kiss by the fire, the conversation about death

**Updates when:** Scene changes, significant story moments

---

### Layer 3: The DM

Now when DM gets called, it has:
- Parser already flagged what kind of moment this is
- Stagehand has the environmental foundation
- Tension Keeper knows if shit's about to go down
- Thread Keeper has the narrative context

**DM's job becomes:**
1. Check what the preppers have established
2. Roll if uncertain (show the roll to players)
3. Add specificity and life
4. Speak

**NOT:** "Let me triage this, call advocates, weigh arguments, deliver verdict"

**Just:** "What's here? What happens? Say it with texture."

---

## Data Flow

```
EVERY MESSAGE
     │
     ▼
┌─────────┐
│ PARSER  │ ──────────────────────────────────────┐
└────┬────┘                                       │
     │                                            │
     │ annotates message:                         │
     │ - type (dialogue, action, discovery, etc)  │
     │ - noise level (silent, normal, loud)       │
     │ - movement (none, local, travel)           │
     │ - triggers (freezer door, etc)             │
     │                                            │
     ▼                                            ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
│ CARTOGRAPHER│    │  STAGEHAND  │    │ TENSION KEEPER  │
│    LIGHT    │    │             │    │                 │
└──────┬──────┘    └──────┬──────┘    └────────┬────────┘
       │                  │                    │
       │ updates:         │ updates:           │ updates:
       │ - known places   │ - local radius     │ - threat zones
       │ - player-made    │ - discoverable     │ - drift state
       │   locations      │   things           │ - dormant triggers
       │                  │                    │
       └────────┬─────────┴──────────┬─────────┘
                │                    │
                ▼                    ▼
         ┌──────────────┐    ┌──────────────┐
         │ THREAD KEEPER│    │   WORLD      │
         │ (narrative)  │    │   STATE      │
         └──────┬───────┘    └──────┬───────┘
                │                   │
                └─────────┬─────────┘
                          │
                          ▼
                    ┌───────────┐
                    │    DM     │
                    │           │
                    │ - reads foundation
                    │ - adds specificity
                    │ - rolls if uncertain
                    │ - speaks with life
                    └───────────┘
```

---

## System Triggers

| System | Trigger | Output |
|--------|---------|--------|
| Parser | Every message | Annotations on that message |
| Cartographer Light | Movement detected, new location established | Updated location state |
| Stagehand | Location change, OR periodic (every N messages in same area) | "Discoverable radius" for current location |
| Tension Keeper | Noise events, time passing, trigger flags | Updated threat state |
| Thread Keeper | Scene changes, significant story beats | Narrative summary/diary |
| DM | Player asks question, requests ruling, or trigger fires | Narration |

---

## Example: The Rabbit Tracks

**Setup:** Kaido and Grace hunting in forest at dawn. They're starving.

1. **Earlier messages**: Parser noted "forest", "hunting", "dawn", "low food"
2. **Stagehand** (forest context): prepped `{small_game: "likely", terrain: "muddy forest"}`
3. **Player**: "we inspect the faint depression in the mud"
4. **Parser**: flags `{type: "discovery_attempt", target: "tracks"}`
5. **DM called**, receives:
   - Stagehand: `{small_game: "likely", terrain: "muddy_forest"}`
   - Parser: `{type: "discovery", target: "tracks"}`
   - Thread: `{motivation: "hunting", stakes: "starving", promise: "Grace said she'd make gravy"}`
6. **DM rolls** (favorable odds given Stagehand foundation): success
7. **DM speaks**: "Rabbit droppings near the log, maybe a day old. A tuft of gray-brown fur caught on the bark above. The tracks lead north, toward where you heard water earlier."

**What's different:** DM didn't summarize "Kaido uses his skills to search." DM told them what they FOUND. Specific. Alive. Connected to what they already know (the water).

---

## Example: The Freezer Trap

**Setup:** Characters find gas station. Tension Keeper has flagged: `{gas_station_freezer: {type: "dormant_swarm", trigger: "door_opens"}}`

1. **Player**: "I open the freezer"
2. **Parser**: flags `{action: "open", target: "freezer", location: "gas_station"}`
3. **Parser checks Tension Keeper**: TRIGGER MATCH
4. **DM called with trigger flag** - no deliberation needed
5. **DM speaks**: "The door swings open and the cold hits you first. Then the smell - not rot, something older, like meat left too long in the dark. Then movement. A mass of gray limbs unfolds from the frost, fingers splayed, mouths already open—"

**What's different:** No council debate about "is this a MAJOR action?" The system KNEW this was a trap. DM just had to bring it to life.

---

## Implementation Order

1. **Parser** - foundation everything else reads from
2. **Stagehand** - directly addresses the "tracks in mud" problem
3. **Tension Keeper** - ambient threat, drift, dormant triggers
4. **Cartographer Light** - location state for immediate area
5. **Thread Keeper** - narrative diary
6. **DM rewrite** - gut Council, replace with "read foundation, add life, speak"

---

## Open Questions

- How does Stagehand decide what's "likely" vs "possible" vs "confirmed"?
- How often does Stagehand refresh in a static location?
- How does Tension Keeper calculate drift speed?
- What model handles each system? (Parser could be fast/cheap, DM needs quality)
- How do we show dice rolls to players in the UI?
- How does Thread Keeper decide what's "significant" enough to track?

---

## Source Material

Philosophy drawn from "So You Want To Be A Game Master" by Justin Alexander, particularly:
- The ruling procedure (default to yes, yes-but, no-but)
- Permissive clue-finding
- The Three Clue Rule (for any conclusion, include at least three clues)
- "Raw material, not experience" - system provides foundation, DM brings it to life
- "Let's find out together" - DM is also discovering

---

*Document created: 2026-05-01*
*Ready to implement: Parser first*
