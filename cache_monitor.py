"""Cache performance monitoring for Roundtable.

Tracks prompt caching savings across sessions to show the real cost impact.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional


class CacheMonitor:
    """Track and report on cache performance over time."""

    def __init__(self):
        self.cache_dir = Path.home() / ".roundtable"
        self.cache_dir.mkdir(exist_ok=True)
        self.stats_file = self.cache_dir / "cache_stats.json"
        self._load_stats()

    def _load_stats(self):
        """Load existing stats from disk."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file) as f:
                    self.stats = json.load(f)
            except:
                self.stats = self._empty_stats()
        else:
            self.stats = self._empty_stats()

    def _empty_stats(self):
        """Create empty stats structure."""
        return {
            "total_requests": 0,
            "total_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "uncached_tokens": 0,
            "total_cost": 0.0,
            "total_cost_without_cache": 0.0,
            "total_savings": 0.0,
            "sessions": [],
            "last_updated": None
        }

    def _save_stats(self):
        """Save stats to disk."""
        self.stats["last_updated"] = datetime.now().isoformat()
        with open(self.stats_file, "w") as f:
            json.dump(self.stats, f, indent=2)

    def record_request(
        self,
        model: str,
        cache_read_tokens: int,
        cache_write_tokens: int,
        uncached_tokens: int,
        output_tokens: int = 0,
        session_id: Optional[str] = None
    ):
        """Record a single API request's cache performance."""
        total_input = cache_read_tokens + cache_write_tokens + uncached_tokens

        # Model pricing (per 1M tokens)
        # Default to Opus 4.7 pricing if unknown model
        pricing = self._get_pricing(model)

        # Calculate costs
        cost_cached = (cache_read_tokens * pricing["cache_read"]) / 1_000_000
        cost_write = (cache_write_tokens * pricing["cache_write"]) / 1_000_000
        cost_uncached = (uncached_tokens * pricing["input"]) / 1_000_000
        cost_output = (output_tokens * pricing["output"]) / 1_000_000
        total_cost = cost_cached + cost_write + cost_uncached + cost_output

        # What it would have cost without caching
        no_cache_cost = (total_input * pricing["input"]) / 1_000_000 + cost_output
        savings = no_cache_cost - total_cost

        # Update totals
        self.stats["total_requests"] += 1
        self.stats["total_tokens"] += total_input + output_tokens
        self.stats["cache_read_tokens"] += cache_read_tokens
        self.stats["cache_write_tokens"] += cache_write_tokens
        self.stats["uncached_tokens"] += uncached_tokens
        self.stats["total_cost"] += total_cost
        self.stats["total_cost_without_cache"] += no_cache_cost
        self.stats["total_savings"] += savings

        # Record session
        session_record = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "model": model,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "uncached_tokens": uncached_tokens,
            "output_tokens": output_tokens,
            "cost": total_cost,
            "savings": savings
        }

        self.stats["sessions"].append(session_record)

        # Keep only last 100 sessions to prevent file bloat
        if len(self.stats["sessions"]) > 100:
            self.stats["sessions"] = self.stats["sessions"][-100:]

        self._save_stats()

        return {
            "cost": total_cost,
            "savings": savings,
            "cache_hit_rate": (cache_read_tokens / total_input * 100) if total_input > 0 else 0
        }

    def _get_pricing(self, model: str) -> dict:
        """Get pricing for a model (per 1M tokens)."""
        model_lower = model.lower()

        # Opus pricing
        if "opus-4-7" in model_lower or "opus-4-6" in model_lower:
            return {
                "input": 5.00,
                "output": 25.00,
                "cache_read": 0.50,   # 90% off
                "cache_write": 6.25    # 25% premium
            }
        elif "opus-4-5" in model_lower or "opus-4" in model_lower:
            return {
                "input": 5.00,
                "output": 25.00,
                "cache_read": 0.50,
                "cache_write": 6.25
            }

        # Sonnet pricing
        elif "sonnet-4-6" in model_lower or "sonnet-4-5" in model_lower:
            return {
                "input": 3.00,
                "output": 15.00,
                "cache_read": 0.30,
                "cache_write": 3.75
            }
        elif "sonnet-3" in model_lower or "sonnet-4" in model_lower:
            return {
                "input": 3.00,
                "output": 15.00,
                "cache_read": 0.30,
                "cache_write": 3.75
            }

        # Haiku pricing
        elif "haiku" in model_lower:
            return {
                "input": 1.00,
                "output": 5.00,
                "cache_read": 0.10,
                "cache_write": 1.25
            }

        # Default to Opus pricing if unknown
        return {
            "input": 5.00,
            "output": 25.00,
            "cache_read": 0.50,
            "cache_write": 6.25
        }

    def get_summary(self) -> str:
        """Get a human-readable summary of cache performance."""
        if self.stats["total_requests"] == 0:
            return "No requests tracked yet."

        total_input = (
            self.stats["cache_read_tokens"] +
            self.stats["cache_write_tokens"] +
            self.stats["uncached_tokens"]
        )

        cache_hit_rate = (
            self.stats["cache_read_tokens"] / total_input * 100
            if total_input > 0 else 0
        )

        savings_percent = (
            self.stats["total_savings"] / self.stats["total_cost_without_cache"] * 100
            if self.stats["total_cost_without_cache"] > 0 else 0
        )

        return f"""
╔════════════════════════════════════════════════════╗
║          PROMPT CACHING PERFORMANCE REPORT         ║
╚════════════════════════════════════════════════════╝

Total Requests: {self.stats['total_requests']:,}
Total Tokens:   {self.stats['total_tokens']:,}

Cache Breakdown:
  • Cached (read):  {self.stats['cache_read_tokens']:,} tokens ({cache_hit_rate:.1f}% hit rate)
  • Cached (write): {self.stats['cache_write_tokens']:,} tokens
  • Uncached:       {self.stats['uncached_tokens']:,} tokens

Cost Analysis:
  • Total cost:        ${self.stats['total_cost']:.2f}
  • Without caching:   ${self.stats['total_cost_without_cache']:.2f}
  • Total saved:       ${self.stats['total_savings']:.2f} ({savings_percent:.1f}% reduction)

Last updated: {self.stats['last_updated']}
""".strip()

    def reset_stats(self):
        """Reset all statistics."""
        self.stats = self._empty_stats()
        self._save_stats()


# Global instance
_monitor = None


def get_monitor() -> CacheMonitor:
    """Get the global cache monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = CacheMonitor()
    return _monitor
