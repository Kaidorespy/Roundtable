"""
Colored debug logging for Roundtable systems.
Each system gets its own color for easy visual tracking.
"""

import sys
from datetime import datetime

# ANSI color codes
COLORS = {
    'reset': '\033[0m',
    'bold': '\033[1m',
    'dim': '\033[2m',

    # System colors - each major system gets a unique color
    'web':          '\033[38;5;81m',   # Cyan - web routes
    'provider':     '\033[38;5;213m',  # Pink - AI providers
    'memory':       '\033[38;5;220m',  # Gold - memory consolidation
    'image':        '\033[38;5;46m',   # Bright green - image generation
    'dm':           '\033[38;5;208m',  # Orange - DM agents
    'cartographer': '\033[38;5;51m',   # Aqua - world mapping
    'npc':          '\033[38;5;177m',  # Lavender - NPC system
    'story':        '\033[38;5;196m',  # Red - story daemon
    'combat':       '\033[38;5;160m',  # Dark red - combat
    'fatigue':      '\033[38;5;228m',  # Pale yellow - fatigue
    'inventory':    '\033[38;5;114m',  # Sage green - inventory
    'consequence':  '\033[38;5;203m',  # Salmon - consequences
    'action':       '\033[38;5;141m',  # Purple - action resolver
    'autopilot':    '\033[38;5;87m',   # Turquoise - autopilot
    'loot':         '\033[38;5;178m',  # Amber - loot tables
    'weather':      '\033[38;5;117m',  # Sky blue - weather
    'world':        '\033[38;5;149m',  # Olive - world texture
    'relationship': '\033[38;5;218m',  # Light pink - relationships
    'understudy':   '\033[38;5;183m',  # Mauve - understudy
    'background':   '\033[38;5;245m',  # Gray - background threads
    'error':        '\033[38;5;196m',  # Red - errors
}

# System labels (fixed width for alignment)
LABELS = {
    'web':          'WEB      ',
    'provider':     'PROVIDER ',
    'memory':       'MEMORY   ',
    'image':        'IMAGE    ',
    'dm':           'DM       ',
    'cartographer': 'CARTO    ',
    'npc':          'NPC      ',
    'story':        'STORY    ',
    'combat':       'COMBAT   ',
    'fatigue':      'FATIGUE  ',
    'inventory':    'INVENTORY',
    'consequence':  'CONSEQ   ',
    'action':       'ACTION   ',
    'autopilot':    'AUTOPILOT',
    'loot':         'LOOT     ',
    'weather':      'WEATHER  ',
    'world':        'WORLD    ',
    'relationship': 'RELATION ',
    'understudy':   'UNDERSTUDY',
    'background':   'BKGND    ',
    'error':        'ERROR    ',
}

# Global enable/disable
DEBUG_ENABLED = True

# Per-system enable (set to False to silence a noisy system)
SYSTEM_ENABLED = {
    'web': True,
    'provider': True,
    'memory': True,
    'image': True,
    'dm': True,
    'cartographer': True,
    'npc': True,
    'story': True,
    'combat': True,
    'fatigue': True,
    'inventory': True,
    'consequence': True,
    'action': True,
    'autopilot': True,
    'loot': True,
    'weather': True,
    'world': True,
    'relationship': True,
    'understudy': True,
    'background': True,
    'error': True,
}


def debug(system: str, message: str, data: any = None):
    """
    Print a colored debug message.

    Args:
        system: System name (web, provider, memory, etc.)
        message: Debug message
        data: Optional data to print (will be truncated if long)
    """
    if not DEBUG_ENABLED:
        return
    if not SYSTEM_ENABLED.get(system, True):
        return

    color = COLORS.get(system, COLORS['reset'])
    label = LABELS.get(system, system.upper().ljust(9))
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

    # Build the message
    prefix = f"{COLORS['dim']}{timestamp}{COLORS['reset']} {color}{COLORS['bold']}[{label}]{COLORS['reset']} {color}"

    print(f"{prefix}{message}{COLORS['reset']}", file=sys.stderr)

    if data is not None:
        data_str = str(data)
        if len(data_str) > 200:
            data_str = data_str[:200] + "..."
        print(f"{prefix}  └─ {data_str}{COLORS['reset']}", file=sys.stderr)


def debug_start(system: str, operation: str):
    """Log the start of an operation."""
    debug(system, f"▶ {operation}")


def debug_end(system: str, operation: str, result: str = None):
    """Log the end of an operation."""
    if result:
        debug(system, f"✓ {operation} → {result}")
    else:
        debug(system, f"✓ {operation}")


def debug_error(system: str, operation: str, error: any):
    """Log an error."""
    debug(system, f"✗ {operation} FAILED: {error}")


def debug_event(system: str, event: str, details: str = None):
    """Log an event/state change."""
    if details:
        debug(system, f"● {event}: {details}")
    else:
        debug(system, f"● {event}")


# Convenience functions for each system
def web(msg, data=None): debug('web', msg, data)
def provider(msg, data=None): debug('provider', msg, data)
def memory(msg, data=None): debug('memory', msg, data)
def image(msg, data=None): debug('image', msg, data)
def dm(msg, data=None): debug('dm', msg, data)
def cartographer(msg, data=None): debug('cartographer', msg, data)
def npc(msg, data=None): debug('npc', msg, data)
def story(msg, data=None): debug('story', msg, data)
def combat(msg, data=None): debug('combat', msg, data)
def fatigue(msg, data=None): debug('fatigue', msg, data)
def inventory(msg, data=None): debug('inventory', msg, data)
def consequence(msg, data=None): debug('consequence', msg, data)
def action(msg, data=None): debug('action', msg, data)
def autopilot(msg, data=None): debug('autopilot', msg, data)
def loot(msg, data=None): debug('loot', msg, data)
def weather(msg, data=None): debug('weather', msg, data)
def world(msg, data=None): debug('world', msg, data)
def relationship(msg, data=None): debug('relationship', msg, data)
def understudy(msg, data=None): debug('understudy', msg, data)
def background(msg, data=None): debug('background', msg, data)
def error(system, msg, data=None): debug('error', f"[{system}] {msg}", data)

# Alias for compatibility
def info(system, msg, data=None): debug(system, msg, data)


# Print color legend on import
def print_legend():
    """Print a color legend showing all systems."""
    print(f"\n{COLORS['bold']}=== ROUNDTABLE DEBUG SYSTEMS ==={COLORS['reset']}", file=sys.stderr)
    for system, color in COLORS.items():
        if system in LABELS:
            label = LABELS[system]
            enabled = "ON " if SYSTEM_ENABLED.get(system, True) else "OFF"
            print(f"  {color}[{label}]{COLORS['reset']} {enabled}", file=sys.stderr)
    print(f"{COLORS['bold']}================================{COLORS['reset']}\n", file=sys.stderr)


# Auto-print legend when module loads
print_legend()
