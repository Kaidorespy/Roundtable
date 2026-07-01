# Known Bugs

## Mobile Microphone Requires HTTPS

**Status:** Workaround available
**Severity:** Medium (feature broken on mobile without manual config)

**Description:**
Voice dictation (mic button) doesn't work on mobile when accessing Roundtable over HTTP, even through Tailscale. Browser blocks `getUserMedia()` on insecure origins.

**Workaround:**
Chrome Android: `chrome://flags` → "Insecure origins treated as secure" → add your Tailscale URL

**Proper Fix:**
Add HTTPS support to Flask using Tailscale's built-in certs:
```bash
tailscale cert your-machine-name.your-tailnet.ts.net
```
Then configure Flask to use the cert files. Could add a config option like `https_enabled: true` with auto-detection of Tailscale certs.

---

## LoRA Preview Breaks After Room Deletion

**Status:** Unresolved
**Severity:** Minor (cosmetic)

**Description:**
When an image is set as a LoRA preview, and the room where that image was generated is later deleted, the LoRA preview shows as a broken/missing image icon.

**Expected Behavior:**
The preview should continue working since image files are not deleted when rooms are deleted.

**Notes:**
- Image files remain on disk after room deletion
- `pathToUrl()` function appears to convert paths correctly
- `/images/` endpoint doesn't do room validation
- Need to check browser Network tab when reproducing to see actual error

**Potential Fixes:**
1. Keep previews pointing at existing files (ideal) - need to debug why they break
2. Clean up LoRA previews when room is deleted (workaround)
