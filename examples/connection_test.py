#!/usr/bin/env python3
"""
Connection Test - Verify D1 arm connectivity

Usage:
    python -m examples.connection_test [--ip 192.168.123.18]

Or if the package is installed:
    python examples/connection_test.py [--ip 192.168.123.18]
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running directly from repo root without installing
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.interface.d1_connection import D1Connection


def main():
    parser = argparse.ArgumentParser(description="Test D1 arm connection")
    parser.add_argument("--ip", default="192.168.123.18", help="D1 arm IP address")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    print(f"th3cl4w - D1 Connection Test")
    print(f"=" * 40)
    print(f"Target IP: {args.ip}")
    print()

    # Attempt connection
    print("Connecting to D1 arm...")
    conn = D1Connection(ip=args.ip)

    if conn.connect():
        print("Connection successful!")

        # Read state
        print("\nReading arm state...")
        state = conn.get_state()

        if state:
            print(f"  Joint positions: {state.joint_positions}")
            print(f"  Gripper: {state.gripper_position:.2f}")
            print(f"  Timestamp: {state.timestamp:.3f}")
        else:
            print("  No state received (arm may be initializing)")

        conn.disconnect()
        print("\nTest complete!")
        return 0
    else:
        print("Connection failed!")
        print("\nTroubleshooting:")
        print("  1. Check that D1 arm is powered on")
        print("  2. Verify Ethernet connection")
        print("  3. Confirm IP address (default: 192.168.123.18)")
        print("  4. Check network interface is on 192.168.123.x subnet")
        return 1


if __name__ == "__main__":
    sys.exit(main())
