# Opening Scene Matrix Design

**Goal:** Generate vastly different opening scenes each time by rolling across multiple axes. No more "wet thing in the distance" every time.

---

## THE AXES

### 1. Tension Type
- `IMMEDIATE` - It's happening NOW, at your feet
- `DISCOVERY` - You find something that changes things (body, map, radio, supplies)
- `ARRIVAL` - Someone bursts in with news/need
- `DEADLINE` - Clock is ticking (boat leaving, tide rising, patrol changing)
- `PERSONAL` - Connected to character want/fear/backstory
- `OBSERVATION` - Classic "you notice something" (current default, should be rare now)

### 2. Time of Day
- Dawn / grey hour before sunrise
- Morning
- Midday
- Afternoon
- Dusk / golden hour
- Night
- Deep night (3am vibes)

### 3. Weather
- Clear
- Overcast
- Fog / mist (Oregon special)
- Light rain
- Heavy rain / storm
- Cold snap / frost
- Snow (if winter)
- Wind (coastal gales)

### 4. Threat Type (when applicable)
- Zombie (66% in zombie genre)
- Human - hostile
- Human - desperate/neutral
- Animal (normal)
- Animal (infected/zombie)
- Environmental (tide, collapse, fire)
- None (tension without threat)

### 5. Threat Distance (when threat exists)
- Close (at your feet, behind you)
- Medium (across the room, down the pier)
- Far (but visible - a horde far = one close in intensity)

### 6. Threat Awareness
- Hasn't seen you
- Has seen you
- Hunting you
- Doesn't care about you (focused on something else)

### 7. Threat Size
- One
- Small group (3-5)
- Horde / mob

### 8. Companion State
- Alert, scanning
- Noticed something first
- Frozen / afraid
- Injured / struggling
- Distracted / occupied
- Determined / ready to act

### 9. Housing Situation
- Established camp/village (safety nearby)
- Small group shelter (just your people)
- Homeless / between places (recent loss)
- Temporary hideout (not home)

### 10. Resource State
- Well-supplied
- Getting low
- Critical (last of something)
- Just found something good (lucky roll!)
- Just lost supplies (bad luck)

### 11. Weapon Status
- Player armed
- Companion armed
- Both armed
- Neither armed (should be rarer than current)
- Improvised weapons only

---

## LEGENDARY ROLLS (1/100 or 1/1000)

- See a separated cast member (Vesper stumbling out of the fog)
- Find something connected to player's backstory
- Encounter ties directly to character's WANT
- Perfect lucky find (working vehicle, full armory, untouched pharmacy)
- Worst case scenario (horde + storm + injured companion + no weapons)

---

## INTENSITY EQUIVALENCE

These should feel equally tense:
- One zombie close = horde far away
- Storm + no shelter = zombie + shelter nearby
- Critical supplies + safe = well-supplied + immediate threat

---

## IMPLEMENTATION NOTES

1. Roll each axis independently
2. Some combinations may conflict - build resolution rules
3. Weight zombie threats higher in zombie genre (66%+)
4. Player backstory/role should influence some axes:
   - Medic → higher chance of injured companion/arrival
   - Scout → higher chance of discovery/observation
   - Fighter → higher chance of having weapons
5. Pass rolled values to prompt as structured context
6. Keep prompt short - let model interpret thematic seeds

---

## EXAMPLE ROLLS

**Roll 1:**
- Tension: DISCOVERY
- Time: Dusk
- Weather: Fog
- Housing: Homeless
- Companion: Noticed something first

*"Juniper's hand shoots out, stopping you mid-step. Through the fog, half-buried in the sand where the tide retreated, something metallic catches the dying light..."*

**Roll 2:**
- Tension: DEADLINE
- Time: Morning
- Weather: Clear
- Resource: Critical (no water)
- Threat: None

*"Your tongue sticks to the roof of your mouth. The last canteen went dry yesterday. Juniper's eyes are on the water tower across the compound - the one the other group claimed..."*

**Roll 3:** (Legendary)
- Tension: PERSONAL
- See separated character

*"You're about to turn back when Juniper grabs your arm, hard. Through the rain, limping toward the pier... that walk. You'd know it anywhere. Vesper."*

---

## TODO
- [ ] Implement axis rolling in storybuilder
- [ ] Build prompt injection for rolled values
- [ ] Test variety across 10+ generations
- [ ] Add genre-specific axis weights
- [ ] Connect to character WANT/FEAR fields
