#!/usr/bin/env python3
"""
Connection Test - Verify D1 arm connectivity

Usage:
    python examples/connection_test.py [--ip 192.168.123.18]
"""

import sys
import argparse
sys.path.insert(0, 'src')

from interface.d1_connection import D1Connection


def main():
    parser = argparse.ArgumentParser(description="Test D1 arm connection")
    parser.add_argument("--ip", default="192.168.123.18", help="D1 arm IP address")
    args = parser.parse_args()
    
    print(f"🦾 th3cl4w - D1 Connection Test")
    print(f"=" * 40)
    print(f"Target IP: {args.ip}")
    print()
    
    # Attempt connection
    print("Connecting to D1 arm...")
    conn = D1Connection(ip=args.ip)
    
    if conn.connect():
        print("✅ Connection successful!")
        
        # Read state
        print("\nReading arm state...")
        state = conn.get_state()
        
        if state:
            print(f"  Joint positions: {state.joint_positions}")
            print(f"  Gripper: {state.gripper_position:.2f}")
            print(f"  Timestamp: {state.timestamp:.3f}")
        else:
            print("  ⚠️  No state received (arm may be initializing)")
        
        conn.disconnect()
        print("\n✅ Test complete!")
        return 0
    else:
        print("❌ Connection failed!")
        print("\nTroubleshooting:")
        print("  1. Check that D1 arm is powered on")
        print("  2. Verify Ethernet connection")
        print("  3. Confirm IP address (default: 192.168.123.18)")
        print("  4. Check network interface is on 192.168.123.x subnet")
        return 1


if __name__ == "__main__":
    sys.exit(main())
