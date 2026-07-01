#!/usr/bin/env python3
"""View cache performance statistics for Roundtable."""

import argparse
from cache_monitor import get_monitor


def main():
    parser = argparse.ArgumentParser(description="View Roundtable cache statistics")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset all statistics"
    )
    parser.add_argument(
        "--recent",
        type=int,
        metavar="N",
        help="Show last N sessions"
    )

    args = parser.parse_args()
    monitor = get_monitor()

    if args.reset:
        confirm = input("Are you sure you want to reset all cache statistics? (yes/no): ")
        if confirm.lower() == "yes":
            monitor.reset_stats()
            print("✓ Statistics reset.")
        else:
            print("Cancelled.")
        return

    # Show summary
    print(monitor.get_summary())

    # Show recent sessions if requested
    if args.recent:
        print(f"\n{'='*54}")
        print(f"LAST {args.recent} SESSIONS")
        print(f"{'='*54}\n")

        sessions = monitor.stats["sessions"][-args.recent:]
        for i, session in enumerate(reversed(sessions), 1):
            total_input = (
                session["cache_read_tokens"] +
                session["cache_write_tokens"] +
                session["uncached_tokens"]
            )
            hit_rate = (
                session["cache_read_tokens"] / total_input * 100
                if total_input > 0 else 0
            )

            print(f"{i}. {session['timestamp'][:19]}")
            print(f"   Model: {session['model']}")
            print(f"   Tokens: {total_input:,} input, {session['output_tokens']:,} output")
            print(f"   Cache: {hit_rate:.1f}% hit rate")
            print(f"   Cost: ${session['cost']:.4f} (saved ${session['savings']:.4f})")
            print()


if __name__ == "__main__":
    main()
