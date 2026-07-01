# Prompt Caching Optimization Guide

## What Changed

Your Roundtable app now has **highly optimized prompt caching** that should reduce your Claude API costs by 60-80%.

### Key Improvements

1. **System Prompt Caching (1-hour TTL)**
   - Your massive character sheets, world info, and rules are now cached for 1 hour
   - First request pays full price, subsequent requests in the same session are ~90% cheaper

2. **Conversation History Caching**
   - Each turn in a conversation incrementally caches the history
   - The longer your session, the more you save

3. **Cache Performance Tracking**
   - All cache stats are now logged and tracked
   - View your savings at any time with `python view_cache_stats.py`

## Expected Savings

Based on your "$1.25 call" (likely 200K+ tokens):

| Scenario | Before Optimization | After Optimization | Savings |
|----------|--------------------|--------------------|---------|
| First turn | $1.25 | $1.56 (write premium) | -$0.31 |
| Second turn | $1.25 | $0.13 (90% cached) | $1.12 ✨ |
| Third turn | $1.25 | $0.13 | $1.12 ✨ |
| 10-turn session | $12.50 | $2.73 | **$9.77** 💰 |

**For a typical multi-turn session, you'll save 70-80% on input costs.**

## How to View Your Savings

```bash
# View overall statistics
python view_cache_stats.py

# View last 10 sessions
python view_cache_stats.py --recent 10

# Reset statistics (if you want to start fresh)
python view_cache_stats.py --reset
```

## What You'll See in Logs

When running the app, you'll now see cache stats in your debug logs:

```
Cache TTL: 5m (conversation has 2 turns)
💰 Cache: 145,234 read / 2,456 write / 3,890 uncached | Hit rate: 95.8% | Cost: $0.1234 (saved $0.6543)
```

Or for longer sessions:
```
Cache TTL: 1h (conversation has 8 turns)
💰 Cache: 345,678 read / 1,234 write / 890 uncached | Hit rate: 99.2% | Cost: $0.0456 (saved $1.2345)
```

Breaking down this example:
- **145,234 tokens** served from cache (0.1x cost)
- **2,456 tokens** written to cache (1.25x cost)
- **3,890 tokens** not cached (1x cost)
- **95.8% cache hit rate** - excellent!
- **Cost: $0.12** vs **$0.79 without caching** = **saved $0.65**

## Understanding Cache Behavior

### First Turn (Cold Start)
```
Request 1:
  System: 50K tokens → WRITE to cache (pay 1.25x)
  User msg: 1K tokens → normal cost
  Total cost: ~$0.32 (slightly higher due to write premium)
```

### Subsequent Turns (Cache Hits!)
```
Request 2:
  System: 50K tokens → READ from cache (pay 0.1x) 💰
  Previous turns: 5K tokens → READ from cache (pay 0.1x) 💰
  New user msg: 1K tokens → normal cost
  Total cost: ~$0.03 (90% cheaper!)
```

## Tips for Maximum Savings

1. **Longer sessions save more** - The cache lasts 1 hour, so longer gameplay sessions stack up huge savings

2. **System prompt stability** - Don't change character sheets or world state mid-session if possible. Each change invalidates the cache.

3. **Monitor your stats** - Run `python view_cache_stats.py` after a few sessions to see your real savings

## Technical Details

### Cache TTL (Dynamic!)
The system now **automatically adjusts** cache duration based on conversation length:

- **Turns 1-2**: 5-minute TTL
  - Cheaper write cost (1.25×)
  - Breaks even at 2 requests
  - Perfect for quick interactions
  
- **Turn 3+**: 1-hour TTL
  - Higher write cost (2×) but worth it for committed sessions
  - Handles bathroom breaks / snack runs without losing cache
  - Perfect for long RPG sessions

**Why dynamic?**
- Short test sessions don't pay the 2× premium unnecessarily
- Long sessions get the benefit of extended cache without gaps killing it
- Best of both worlds! 🎯

### Cache Strategy
- System prompt (character sheets, world rules): Always cached with dynamic TTL
- Last message in conversation: Cached when 2+ messages exist (5-min TTL)
- Incremental caching: Each new turn builds on the previous cache

### Pricing (per 1M tokens)
**Opus 4.7 / 4.6:**
- Normal input: $5.00
- Cache read: $0.50 (90% off)
- Cache write: $6.25 (25% premium)

**Sonnet 4.6:**
- Normal input: $3.00
- Cache read: $0.30 (90% off)
- Cache write: $3.75 (25% premium)

**Haiku 4.5:**
- Normal input: $1.00
- Cache read: $0.10 (90% off)
- Cache write: $1.25 (25% premium)

## Troubleshooting

### "Why is my first request more expensive?"

That's the cache write premium (1.25x). Every subsequent request in the same session will be much cheaper.

### "Cache hit rate is low"

Check if:
- Your system prompts are changing between requests
- You're using different models mid-session (caches are model-specific)
- Sessions are lasting longer than 1 hour (cache expires)

### "Where are stats stored?"

`~/.roundtable/cache_stats.json` - This file tracks all your cache performance over time.

## Questions?

Your optimization is now live! Just start playing and watch the savings roll in. 🎉

Check your stats after your next gaming session with:
```bash
python view_cache_stats.py --recent 5
```
