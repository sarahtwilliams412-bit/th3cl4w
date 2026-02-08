#!/usr/bin/env python3.12
"""Joint calibration script for Unitree D1 robotic arm - v2.
Reconnects between joint groups. Uses set_joint for single-joint moves.
"""

import sys
import time
import json
import httpx

sys.path.insert(0, '/home/sarah/.openclaw/workspace/th3cl4w')
from src.interface.d1_dds_connection import D1DDSConnection

HOME = [0, -90, 90, 0, -90, 0, 65]
RESULTS = []

def get_conn():
    """Create a fresh connection."""
    conn = D1DDSConnection()
    conn.connect(interface_name='eno1')
    time.sleep(1)
    # Check status
    status = conn.get_status()
    print(f"  Status: {status}")
    return conn

def snap(path):
    for cam in [0, 1]:
        try:
            r = httpx.get(f'http://localhost:8081/snap/{cam}', timeout=5)
            with open(f'{path}_cam{cam}.jpg', 'wb') as f:
                f.write(r.content)
            print(f"  Snap: {path}_cam{cam}.jpg ({len(r.content)}b)")
        except Exception as e:
            print(f"  WARN cam{cam}: {e}")

def read_angles(conn):
    angles = conn.get_joint_angles()
    if angles is None:
        print("  Actual angles: None (no feedback)")
        return None
    a = [round(float(x), 2) for x in angles]
    print(f"  Actual: {a}")
    return [float(x) for x in angles]

def go_home(conn):
    print("  -> HOME")
    conn.set_all_joints(HOME)
    time.sleep(3)
    return read_angles(conn)

def wait_for_move(conn, joint_idx, target, timeout=5):
    """Wait until joint reaches near target or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        angles = conn.get_joint_angles()
        if angles is not None:
            current = float(angles[joint_idx])
            if abs(current - target) < 2.0:
                return True
        time.sleep(0.2)
    return False

def main():
    print("="*60)
    print("UNITREE D1 JOINT CALIBRATION v2")
    print("="*60)

    conn = get_conn()

    # Go home
    print("\n=== HOME POSITION ===")
    home_actual = go_home(conn)
    snap('/tmp/calib_HOME')
    RESULTS.append({'test': 'HOME', 'commanded': HOME, 'actual': home_actual})

    # Test each joint individually using set_joint
    test_offsets = [-45, 0, 45]

    for j in range(6):
        print(f"\n{'='*60}")
        print(f"=== JOINT {j} ===")
        print(f"{'='*60}")

        # Reconnect for each joint group to avoid stale connection
        try:
            conn.disconnect()
        except:
            pass
        time.sleep(1)
        conn = get_conn()
        go_home(conn)
        time.sleep(1)

        for angle in test_offsets:
            label = f"J{j}_{angle:+d}"
            print(f"\n--- {label} (set_joint({j}, {angle})) ---")

            # Use set_joint for single joint move
            ok = conn.set_joint(j, angle)
            print(f"  Command sent: {ok}")
            time.sleep(2)

            # Check if it moved
            actual = read_angles(conn)
            status = conn.get_status()
            print(f"  Status: {status}")

            snap(f'/tmp/calib_{label}')

            entry = {
                'test': label,
                'joint': j,
                'commanded_angle': angle,
                'actual': actual,
                'status': status,
            }
            if actual:
                entry['actual_joint'] = actual[j]
                entry['error'] = round(actual[j] - angle, 2)
            RESULTS.append(entry)

        # Return home after this joint
        print("\n--- Return HOME ---")
        go_home(conn)
        time.sleep(1)

    # Gripper test
    print(f"\n{'='*60}")
    print(f"=== GRIPPER ===")
    print(f"{'='*60}")

    try:
        conn.disconnect()
    except:
        pass
    time.sleep(1)
    conn = get_conn()
    go_home(conn)

    for g in [65, 32, 0]:
        label = f"GRIP_{g}mm"
        print(f"\n--- {label} ---")
        conn.set_gripper(float(g))
        time.sleep(2)
        actual = read_angles(conn)
        snap(f'/tmp/calib_{label}')
        RESULTS.append({
            'test': label,
            'gripper_cmd': g,
            'actual': actual,
            'actual_gripper': actual[6] if actual else None,
        })

    # Final home
    go_home(conn)

    # Save results
    with open('/tmp/calib_results.json', 'w') as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\n\nSaved {len(RESULTS)} results to /tmp/calib_results.json")
    print("CALIBRATION COMPLETE!")

    try:
        conn.disconnect()
    except:
        pass

if __name__ == '__main__':
    main()
