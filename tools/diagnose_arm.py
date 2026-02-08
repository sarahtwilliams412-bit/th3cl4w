#!/usr/bin/env python3.12
"""
diagnose_arm.py — D1 Arm Diagnostic Tool

Connects to the arm via DDS and captures telemetry to diagnose:
  1. Gripper not responding (command format / feedback state)
  2. Jerky motion (command frequency, position deltas)
  3. Joint mapping issues (visual vs physical ordering)

Usage:
    python3.12 tools/diagnose_arm.py [--interface eno1] [--duration 30] [--output diag.json]
"""

import argparse
import json
import logging
import os
import signal
import statistics
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure project root is on path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
from cyclonedds.domain import DomainParticipant
from cyclonedds.idl import IdlStruct
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("diagnose_arm")


# ---------------------------------------------------------------------------
# IDL type (must match unitree DDS schema)
# ---------------------------------------------------------------------------

@dataclass
class ArmString_(IdlStruct, typename="unitree_arm.msg.dds_.ArmString_"):
    data_: str = ""


# ---------------------------------------------------------------------------
# Data recorder
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticRecorder:
    """Collects raw feedback and command data with timestamps."""

    feedback_samples: list[dict[str, Any]] = field(default_factory=list)
    command_samples: list[dict[str, Any]] = field(default_factory=list)
    feedback_timestamps: list[float] = field(default_factory=list)
    command_timestamps: list[float] = field(default_factory=list)
    joint_angle_history: list[dict[str, float]] = field(default_factory=list)
    gripper_history: list[dict[str, Any]] = field(default_factory=list)
    status_history: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    start_time: float = 0.0

    def record_feedback(self, msg: dict[str, Any]) -> None:
        now = time.monotonic()
        wall = time.time()
        with self.lock:
            self.feedback_samples.append({"wall_time": wall, "mono": now, "msg": msg})
            self.feedback_timestamps.append(now)

            funcode = msg.get("funcode")
            data = msg.get("data")
            if funcode == 1 and isinstance(data, dict):
                angles = {k: v for k, v in data.items() if k.startswith("angle")}
                self.joint_angle_history.append({"t": now - self.start_time, **angles})
                # Track gripper (angle6)
                if "angle6" in data:
                    self.gripper_history.append({
                        "t": now - self.start_time,
                        "position": data["angle6"],
                        "source": "feedback",
                    })
            elif funcode == 3 and isinstance(data, dict):
                self.status_history.append({"t": now - self.start_time, **data})

    def record_command(self, msg: dict[str, Any]) -> None:
        now = time.monotonic()
        wall = time.time()
        with self.lock:
            self.command_samples.append({"wall_time": wall, "mono": now, "msg": msg})
            self.command_timestamps.append(now)

            # Track gripper commands specifically
            funcode = msg.get("funcode")
            data = msg.get("data", {})
            if funcode == 1 and isinstance(data, dict) and data.get("id") == 6:
                self.gripper_history.append({
                    "t": now - self.start_time,
                    "position": data.get("angle"),
                    "source": "command",
                })


# ---------------------------------------------------------------------------
# DDS sniffer (passive — listens to both topics)
# ---------------------------------------------------------------------------

def run_sniffer(
    interface: str,
    duration: float,
    recorder: DiagnosticRecorder,
) -> None:
    """Subscribe to feedback and command topics, record everything."""

    os.environ["CYCLONEDDS_URI"] = (
        "<CycloneDDS>"
        "  <Domain>"
        "    <General>"
        "      <Interfaces>"
        f'        <NetworkInterface name="{interface}" />'
        "      </Interfaces>"
        "    </General>"
        "  </Domain>"
        "</CycloneDDS>"
    )

    dp = DomainParticipant(domain_id=0)

    topic_fb = Topic(dp, "rt/arm_Feedback", ArmString_)
    topic_cmd = Topic(dp, "rt/arm_Command", ArmString_)
    reader_fb = DataReader(dp, topic_fb)
    reader_cmd = DataReader(dp, topic_cmd)

    stop = threading.Event()
    recorder.start_time = time.monotonic()

    def _read_loop():
        logger.info("Sniffing DDS traffic for %.0fs on %s ...", duration, interface)
        deadline = time.monotonic() + duration
        while not stop.is_set() and time.monotonic() < deadline:
            try:
                for sample in reader_fb.take(N=64):
                    try:
                        msg = json.loads(sample.data_)
                        recorder.record_feedback(msg)
                    except (json.JSONDecodeError, AttributeError) as e:
                        recorder.errors.append(f"bad feedback: {e}")
            except Exception:
                pass

            try:
                for sample in reader_cmd.take(N=64):
                    try:
                        msg = json.loads(sample.data_)
                        recorder.record_command(msg)
                    except (json.JSONDecodeError, AttributeError) as e:
                        recorder.errors.append(f"bad command: {e}")
            except Exception:
                pass

            time.sleep(0.002)  # 2ms poll

    thread = threading.Thread(target=_read_loop, daemon=True)
    thread.start()

    # Allow Ctrl-C to stop early
    def _sigint(sig, frame):
        logger.info("Interrupted — generating report with data collected so far")
        stop.set()
    signal.signal(signal.SIGINT, _sigint)

    thread.join(timeout=duration + 1)
    stop.set()
    thread.join(timeout=2)


# ---------------------------------------------------------------------------
# Analysis & report
# ---------------------------------------------------------------------------

def compute_rate(timestamps: list[float]) -> dict[str, Any]:
    """Compute rate stats from a list of monotonic timestamps."""
    if len(timestamps) < 2:
        return {"count": len(timestamps), "rate_hz": 0.0, "intervals": {}}
    intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps) - 1)]
    span = timestamps[-1] - timestamps[0]
    return {
        "count": len(timestamps),
        "rate_hz": round(len(timestamps) / span, 2) if span > 0 else 0.0,
        "intervals": {
            "mean_ms": round(statistics.mean(intervals) * 1000, 3),
            "median_ms": round(statistics.median(intervals) * 1000, 3),
            "stdev_ms": round(statistics.stdev(intervals) * 1000, 3) if len(intervals) > 1 else 0.0,
            "min_ms": round(min(intervals) * 1000, 3),
            "max_ms": round(max(intervals) * 1000, 3),
        },
    }


def compute_position_deltas(history: list[dict[str, float]]) -> dict[str, Any]:
    """Compute per-joint position deltas between consecutive feedback samples."""
    if len(history) < 2:
        return {"samples": len(history), "deltas": {}}

    joint_keys = sorted(k for k in history[0] if k.startswith("angle"))
    deltas_per_joint: dict[str, list[float]] = defaultdict(list)

    for i in range(1, len(history)):
        for k in joint_keys:
            prev = history[i-1].get(k, 0.0)
            curr = history[i].get(k, 0.0)
            deltas_per_joint[k].append(abs(curr - prev))

    result = {}
    for k in joint_keys:
        d = deltas_per_joint[k]
        result[k] = {
            "mean_deg": round(statistics.mean(d), 4),
            "max_deg": round(max(d), 4),
            "total_travel_deg": round(sum(d), 2),
        }
    return {"samples": len(history), "deltas": result}


def compute_tracking_error(recorder: DiagnosticRecorder) -> dict[str, Any]:
    """Compare commanded positions with nearest-in-time feedback positions."""
    # Extract set-joint and set-all-joints commands
    cmd_positions: list[dict[str, Any]] = []
    for s in recorder.command_samples:
        msg = s["msg"]
        fc = msg.get("funcode")
        data = msg.get("data", {})
        if fc == 1 and isinstance(data, dict) and "id" in data:
            cmd_positions.append({"t": s["mono"], "joint": f"angle{data['id']}", "target": data.get("angle", 0)})
        elif fc == 2 and isinstance(data, dict):
            for i in range(7):
                k = f"angle{i}"
                if k in data:
                    cmd_positions.append({"t": s["mono"], "joint": k, "target": data[k]})

    if not cmd_positions or not recorder.joint_angle_history:
        return {"error": "insufficient data"}

    # For each command, find the feedback sample ~200ms later and compute error
    errors_by_joint: dict[str, list[float]] = defaultdict(list)
    fb_times = [h["t"] + recorder.start_time for h in recorder.joint_angle_history]

    for cp in cmd_positions:
        target_t = cp["t"] + 0.2  # look 200ms after command
        # binary search for nearest feedback
        import bisect
        idx = bisect.bisect_left(fb_times, target_t)
        if idx >= len(fb_times):
            idx = len(fb_times) - 1
        if idx < 0:
            continue
        fb = recorder.joint_angle_history[idx]
        actual = fb.get(cp["joint"], None)
        if actual is not None:
            errors_by_joint[cp["joint"]].append(abs(cp["target"] - actual))

    result = {}
    for joint, errs in sorted(errors_by_joint.items()):
        result[joint] = {
            "mean_error_deg": round(statistics.mean(errs), 4),
            "max_error_deg": round(max(errs), 4),
            "samples": len(errs),
        }
    return result


def analyse_gripper(recorder: DiagnosticRecorder) -> dict[str, Any]:
    """Analyse gripper command/feedback patterns."""
    cmds = [g for g in recorder.gripper_history if g["source"] == "command"]
    fbs = [g for g in recorder.gripper_history if g["source"] == "feedback"]

    result: dict[str, Any] = {
        "commands_sent": len(cmds),
        "feedback_received": len(fbs),
    }

    if fbs:
        positions = [f["position"] for f in fbs]
        result["feedback_position_range"] = [round(min(positions), 2), round(max(positions), 2)]
        result["feedback_position_last"] = round(positions[-1], 2)
        # Check if gripper ever moved
        unique = set(round(p, 2) for p in positions)
        result["gripper_moved"] = len(unique) > 1
        result["unique_positions"] = len(unique)
    else:
        result["gripper_moved"] = False
        result["WARNING"] = "No gripper feedback received — joint 6 may not be in feedback data"

    if cmds and fbs:
        # Check if feedback responded to commands
        result["first_command_t"] = round(cmds[0]["t"], 3)
        result["first_feedback_t"] = round(fbs[0]["t"], 3)

    # Check command format from raw data
    gripper_cmds_raw = [
        s["msg"] for s in recorder.command_samples
        if s["msg"].get("funcode") == 1
        and isinstance(s["msg"].get("data"), dict)
        and s["msg"]["data"].get("id") == 6
    ]
    if gripper_cmds_raw:
        result["command_format_sample"] = gripper_cmds_raw[0]
    else:
        # Check if gripper commands use a different format
        gripper_like = [
            s["msg"] for s in recorder.command_samples
            if "gripper" in json.dumps(s["msg"]).lower()
        ]
        if gripper_like:
            result["WARNING_format"] = "Gripper commands found but NOT using joint 6 format"
            result["found_format"] = gripper_like[0]
        else:
            result["note"] = "No gripper commands captured during recording"

    return result


def analyse_joint_mapping(recorder: DiagnosticRecorder) -> dict[str, Any]:
    """Check for potential joint mapping issues."""
    if len(recorder.joint_angle_history) < 10:
        return {"error": "insufficient feedback samples"}

    # Look at which joints are active (changing)
    joint_keys = sorted(k for k in recorder.joint_angle_history[0] if k.startswith("angle"))
    activity: dict[str, float] = {}
    for k in joint_keys:
        values = [h.get(k, 0.0) for h in recorder.joint_angle_history]
        activity[k] = round(max(values) - min(values), 4)

    return {
        "joint_activity_range_deg": activity,
        "joint_order": joint_keys,
        "note": "Compare joint_activity with physical observation to detect swaps",
    }


def generate_report(recorder: DiagnosticRecorder) -> dict[str, Any]:
    """Generate the full diagnostic report."""
    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": {},
        "feedback_rate": compute_rate(recorder.feedback_timestamps),
        "command_rate": compute_rate(recorder.command_timestamps),
        "position_deltas": compute_position_deltas(recorder.joint_angle_history),
        "tracking_error": compute_tracking_error(recorder),
        "gripper_analysis": analyse_gripper(recorder),
        "joint_mapping": analyse_joint_mapping(recorder),
        "status_snapshots": recorder.status_history[:5] + recorder.status_history[-5:] if recorder.status_history else [],
        "errors": recorder.errors,
    }

    # Summary
    fb = report["feedback_rate"]
    cmd = report["command_rate"]
    gripper = report["gripper_analysis"]

    issues = []
    if fb["count"] == 0:
        issues.append("❌ NO FEEDBACK RECEIVED — arm may be off or DDS misconfigured")
    elif fb["rate_hz"] < 50:
        issues.append(f"⚠ Low feedback rate: {fb['rate_hz']} Hz (expected ~200 Hz)")

    if cmd["count"] > 0 and cmd.get("intervals", {}).get("stdev_ms", 0) > 50:
        issues.append(f"⚠ JERKY MOTION — command interval stdev={cmd['intervals']['stdev_ms']}ms (high variance)")

    if cmd["count"] > 0 and cmd.get("intervals", {}).get("max_ms", 0) > 200:
        issues.append(f"⚠ Command gaps up to {cmd['intervals']['max_ms']}ms — may cause jerky motion")

    if gripper["commands_sent"] > 0 and not gripper.get("gripper_moved", False):
        issues.append("❌ GRIPPER NOT RESPONDING — commands sent but no movement detected")

    if not issues:
        issues.append("✅ No obvious issues detected in this capture")

    report["summary"] = {
        "feedback_samples": fb["count"],
        "command_samples": cmd["count"],
        "feedback_rate_hz": fb["rate_hz"],
        "issues": issues,
    }

    return report


def print_report(report: dict[str, Any]) -> None:
    """Pretty-print the diagnostic report."""
    print("\n" + "=" * 70)
    print("  D1 ARM DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"  Time: {report['timestamp']}")
    print()

    summary = report["summary"]
    print(f"  Feedback samples: {summary['feedback_samples']}")
    print(f"  Command samples:  {summary['command_samples']}")
    print(f"  Feedback rate:    {summary['feedback_rate_hz']} Hz")
    print()

    print("  ISSUES:")
    for issue in summary["issues"]:
        print(f"    {issue}")
    print()

    # Feedback rate details
    fb = report["feedback_rate"]
    if fb["intervals"]:
        iv = fb["intervals"]
        print(f"  Feedback intervals: mean={iv['mean_ms']}ms  median={iv['median_ms']}ms  stdev={iv['stdev_ms']}ms")

    cmd = report["command_rate"]
    if cmd["intervals"]:
        iv = cmd["intervals"]
        print(f"  Command intervals:  mean={iv['mean_ms']}ms  median={iv['median_ms']}ms  stdev={iv['stdev_ms']}ms")
    print()

    # Gripper
    ga = report["gripper_analysis"]
    print("  GRIPPER ANALYSIS:")
    for k, v in ga.items():
        print(f"    {k}: {v}")
    print()

    # Position deltas
    pd = report["position_deltas"]
    if pd.get("deltas"):
        print("  POSITION DELTAS (per feedback sample):")
        for joint, stats in pd["deltas"].items():
            print(f"    {joint}: mean={stats['mean_deg']}°  max={stats['max_deg']}°  total={stats['total_travel_deg']}°")
        print()

    # Tracking error
    te = report["tracking_error"]
    if isinstance(te, dict) and "error" not in te:
        print("  TRACKING ERROR (cmd vs feedback @ +200ms):")
        for joint, stats in te.items():
            print(f"    {joint}: mean={stats['mean_error_deg']}°  max={stats['max_error_deg']}°  n={stats['samples']}")
        print()

    # Joint mapping
    jm = report["joint_mapping"]
    if "joint_activity_range_deg" in jm:
        print("  JOINT ACTIVITY (range of motion during capture):")
        for joint, rng in jm["joint_activity_range_deg"].items():
            bar = "█" * int(min(rng, 50))
            print(f"    {joint}: {rng:7.2f}°  {bar}")
        print()

    if report["errors"]:
        print(f"  ERRORS ({len(report['errors'])}):")
        for e in report["errors"][:10]:
            print(f"    • {e}")
        print()

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="D1 Arm Diagnostic Tool")
    parser.add_argument("--interface", default="eno1", help="Network interface (default: eno1)")
    parser.add_argument("--duration", type=float, default=30.0, help="Capture duration in seconds (default: 30)")
    parser.add_argument("--output", default=None, help="Output JSON file (default: diag_<timestamp>.json)")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"diag_{time.strftime('%Y%m%d_%H%M%S')}.json"

    recorder = DiagnosticRecorder()

    logger.info("Starting D1 arm diagnostics")
    logger.info("  Interface: %s", args.interface)
    logger.info("  Duration:  %.0fs", args.duration)
    logger.info("  Output:    %s", args.output)
    logger.info("Press Ctrl-C to stop early and generate report")
    print()

    run_sniffer(args.interface, args.duration, recorder)

    logger.info("Capture complete. Analysing...")

    report = generate_report(recorder)
    print_report(report)

    # Save raw data
    output_data = {
        "report": report,
        "raw": {
            "feedback_count": len(recorder.feedback_samples),
            "command_count": len(recorder.command_samples),
            "feedback_samples": recorder.feedback_samples[:500],  # cap for file size
            "command_samples": recorder.command_samples[:500],
            "joint_angle_history": recorder.joint_angle_history[:1000],
            "gripper_history": recorder.gripper_history,
            "status_history": recorder.status_history,
        },
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(output_data, indent=2, default=str))
    logger.info("Raw data saved to %s (%.1f KB)", out_path, out_path.stat().st_size / 1024)


if __name__ == "__main__":
    main()
