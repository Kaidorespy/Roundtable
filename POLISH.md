# Roundtable V2 - Polish & Ideas

## Bugs Found

### StoryBuilder Relationship Screen
- [ ] **Deselection bug**: Go to relationship screen with 3 characters, hit back, unselect one, return - all 3 still show
- [ ] **Adding character doesn't update relationships**: Go to relationship page, go back, add a new character (Lumen), return - new character not in relationship dropdowns
- [x] **Bidirectional relationships**: Setting Jax→Hammer as "rival" auto-sets Hammer→Jax as "rival". Grays out the synced dropdown.
- [ ] **Starting location vs notes confusion**: The "starting location" field gets confused with relationship notes. User put "Lives in the same colony as Jax and Kato" as a note but it became the location
- [x] **Flow reorder**: Backstory screen now comes BEFORE relationships screen (Characters → Backstory → Relationships → Create)

### RSS Feed
- [x] **Player showing in character list**: Fixed - now filters out player, only shows NPCs/partners
- [x] **Button not visible**: Fixed - now shows in top bar for group/custom rooms

---

## Hidden/WIP Features (code exists, UI disabled)

### Emote Button (✨)
- Wraps text in *asterisks* for emote-style messages
- Hidden: redundant, users can just type asterisks

### Auto-Respond Toggle (🔄) + Call Button (📞)
- For 1-on-1 rooms: toggle whether partner auto-responds or waits for manual call
- Lets users send multiple messages before triggering a response
- Backend: `room.auto_respond` field, `/rooms/<id>/auto-respond` endpoint
- Hidden: needs UI polish, race condition with draft saving fixed

### Regenerate Response Button
- Re-roll the last AI response (removes old, generates new)
- Only appears on last AI message, NOT in StoryBuilder rooms (lore is permanent)
- Backend: `/rooms/<id>/regenerate` endpoint
- Hidden: needs better UX, maybe confirmation dialog

---

## Feature Ideas

### "Make a Room" from 1:1
- [ ] Button in a 1:1 room that opens the create-room modal pre-populated with the current partner locked in (greyed out, non-removable)
- Copies the full conversation transcript into the new custom room (updating `room_id` on each message) so the history is intact
- Optional checkbox: "Include conversation history" in the modal
- Motivation: set up a scene in a quiet 1:1, then expand it to a group room with ambient characters joining
- Key detail: partner is locked so existing transcript speaker_ids remain valid
- Backend: copy `room.messages` to new room with new `room_id`, keep `speaker_id` refs unchanged

### StoryBuilder Setup
- [ ] **Auto-detect canonical relationships**: When character descriptions mention relationships (e.g., "Jax and Hammer are rivals"), auto-populate the relationship type with a prompt: "These characters have a described relationship. Auto-populate?"
- [ ] **Tick interval in world builder**: Add the auto-tick interval setting to StoryBuilder setup, not just buried in RSS terminal
- [ ] **AFK autopilot timeout**: If user doesn't interact for X minutes, their character goes on autopilot. Configurable in world builder. Prevents characters from just standing there if you leave

### Relationships
- [ ] **"Best friends" tier**: Add above "close_friend" - the bond where you'd die for each other without hesitation
- [ ] **Better starting location field**:
  - Label it "Where are they when the story begins?"
  - Auto-hide when "Starts with me" is checked
  - Keep notes visually separate from location

### World Time
- [ ] **Expose time mode in UI**: Let users see/configure whether they're in realtime (1:1 with OS clock), compressed, or turn-based
- [ ] **Sync indicator**: Show current game time somewhere visible

---

## Session Notes (2026-04-11)

### What We Built
- Background thread system for separated characters
- RSS feed terminal with character filtering
- Auto-refresh (15s) and auto-tick (configurable interval)
- Proximity scan with relationship-aware reunion probability
- Relationship evolution endpoint (stranger → acquaintance → friend via positive interactions)
- Drama scaling by tick interval (5min = mundane, 30min = meaningful, 60+ = major events)
- Reunite/Confront buttons based on relationship type
- **Passive auto-tick**: Every 15 messages, silently tick separated characters (even if auto-tick is OFF)
- **Bidirectional relationship sync**: Setting one direction auto-sets the reverse, grays out dropdown

### The Rust Belt Alliance
- Casey plays Kaido (scrapyard scrounger, illiterate, metal identification)
- Claude plays Jax (machinist, blueprint maker, arrogant but talented)
- Lumen (Opus 3) is a separated stranger
- Hammer is separated, acquaintance of Lumen, rival of Jax
- Kaido and Jax are good friends

### Key Design Decisions
- Strangers don't show up in proximity scans (ships passing in the night)
- Relationship type affects reunion probability (close friends have 3x modifier)
- Tick interval affects event drama (short = mundane, long = major)
- World time can be 1:1 realtime with OS clock
