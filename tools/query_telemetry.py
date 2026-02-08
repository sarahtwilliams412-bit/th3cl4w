#!/usr/bin/env python3
"""CLI tool for querying th3cl4w telemetry database."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.telemetry.query import TelemetryQuery


def _fmt_ts(ts: float | None) -> str:
    if ts is None:
        return "—"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _print_table(headers: list[str], rows: list[list[str]], max_col: int = 30) -> None:
    if not rows:
        print("(no data)")
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], min(len(str(cell)), max_col))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["─" * w for w in widths]))
    for row in rows:
        cells = [str(c)[:max_col] for c in row]
        while len(cells) < len(headers):
            cells.append("")
        print(fmt.format(*cells))


def cmd_tail(args: argparse.Namespace) -> None:
    q = TelemetryQuery(args.db)
    event_types = args.types.split(",") if args.types else None
    last_ts = 0.0

    while True:
        events = q.tail(limit=args.limit, event_types=event_types)
        events.reverse()  # oldest first
        new = [e for e in events if e["ts"] > last_ts]
        for e in new:
            print(f"[{_fmt_ts(e['ts'])}] {e.get('level','info'):>7} {e['event_type']:<15} {e.get('source',''):<10} {e.get('detail','')}")
            last_ts = e["ts"]
        if not args.follow:
            break
        time.sleep(0.5)


def cmd_joints(args: argparse.Namespace) -> None:
    q = TelemetryQuery(args.db)
    rows = q.get_joint_history(args.joint, start=args.start, end=args.end, limit=args.limit)
    rows.reverse()
    _print_table(
        ["Timestamp", "Angle (deg)"],
        [[_fmt_ts(r["ts"]), f"{r['angle']:.3f}"] for r in rows],
    )


def cmd_tracking_error(args: argparse.Namespace) -> None:
    q = TelemetryQuery(args.db)
    result = q.get_tracking_error(args.joint, start=args.start, end=args.end)
    print(f"Joint {args.joint} tracking error ({result['samples']} samples):")
    print(f"  Mean: {result['mean_error_deg']:.4f}°")
    print(f"  Max:  {result['max_error_deg']:.4f}°")
    if args.verbose and result["errors"]:
        _print_table(
            ["Timestamp", "Target", "Actual", "Error"],
            [[_fmt_ts(e["ts"]), f"{e['target']:.3f}", f"{e['actual']:.3f}", f"{e['error']:.4f}"]
             for e in result["errors"][-20:]],
        )


def cmd_rate(args: argparse.Namespace) -> None:
    q = TelemetryQuery(args.db)
    cr = q.get_command_rate(args.window)
    fr = q.get_feedback_rate(args.window)
    print(f"Rates (last {args.window}s window):")
    print(f"  Commands:  {cr['rate_hz']:.1f} Hz ({cr['total_commands']} total)")
    print(f"  Feedback:  {fr['rate_hz']:.1f} Hz ({fr['total_feedback']} total)")
    if cr["by_funcode"]:
        print("  By funcode:")
        for fc, hz in sorted(cr["by_funcode"].items()):
            print(f"    {fc}: {hz:.1f} Hz")


def cmd_gripper(args: argparse.Namespace) -> None:
    q = TelemetryQuery(args.db)
    rows = q.get_gripper_log(start=args.start, end=args.end, limit=args.limit)
    rows.reverse()
    _print_table(
        ["Timestamp", "Position", "Source"],
        [[_fmt_ts(r["ts"]), f"{r['position']:.2f}" if r["position"] is not None else "—", r["source"]] for r in rows],
    )


def cmd_events(args: argparse.Namespace) -> None:
    q = TelemetryQuery(args.db)
    rows = q.get_system_events(event_type=args.type, start=args.start, end=args.end, limit=args.limit)
    rows.reverse()
    _print_table(
        ["Timestamp", "Level", "Type", "Source", "Detail"],
        [[_fmt_ts(r["ts"]), r.get("level", ""), r["event_type"], r.get("source", ""), r.get("detail", "") or ""]
         for r in rows],
    )


def cmd_cameras(args: argparse.Namespace) -> None:
    q = TelemetryQuery(args.db)
    rows = q.get_camera_health(camera_id=args.cam, start=args.start, end=args.end, limit=args.limit)
    rows.reverse()
    _print_table(
        ["Timestamp", "Camera", "FPS", "Drops", "Connected", "Stalled"],
        [[_fmt_ts(r["ts"]), r["camera_id"], f"{r.get('actual_fps', 0) or 0:.1f}",
          str(r.get("drop_count", 0)), str(r.get("connected", "")), str(r.get("stalled", ""))]
         for r in rows],
    )


def cmd_latency(args: argparse.Namespace) -> None:
    q = TelemetryQuery(args.db)
    rows = q.get_web_request_latency(endpoint=args.endpoint, start=args.start, end=args.end, limit=args.limit)
    rows.reverse()
    _print_table(
        ["Timestamp", "Method", "Endpoint", "Status", "Latency (ms)", "OK"],
        [[_fmt_ts(r["ts"]), r.get("method", ""), r["endpoint"],
          str(r.get("status_code", "")), f"{r.get('response_ms', 0) or 0:.1f}", str(r.get("ok", ""))]
         for r in rows],
    )


def cmd_smoother(args: argparse.Namespace) -> None:
    q = TelemetryQuery(args.db)
    rows = q.get_smoother_state(joint=args.joint, start=args.start, end=args.end, limit=args.limit)
    rows.reverse()
    _print_table(
        ["Timestamp", "Joint", "Target", "Current", "Sent", "Step"],
        [[_fmt_ts(r["ts"]), str(r["joint_id"]),
          f"{r['target']:.3f}", f"{r['current']:.3f}", f"{r['sent']:.3f}",
          f"{r.get('step_size', 0) or 0:.4f}"]
         for r in rows],
    )


def cmd_summary(args: argparse.Namespace) -> None:
    q = TelemetryQuery(args.db)
    s = q.summary(start=args.start, end=args.end)
    print("Session Summary")
    print("═" * 40)
    print(f"  First event:    {_fmt_ts(s['first_event_ts'])}")
    print(f"  Last event:     {_fmt_ts(s['last_event_ts'])}")
    print(f"  Duration:       {s['duration_s']:.1f}s")
    print(f"  Commands:       {s['total_commands']}")
    print(f"  Feedback:       {s['total_feedback']}")
    print(f"  Events:         {s['total_events']}")
    print(f"  Command rate:   {s['command_rate_hz']:.2f} Hz")
    print(f"  Feedback rate:  {s['feedback_rate_hz']:.2f} Hz")
    print(f"  Errors:         {s['error_count']}")


def cmd_export(args: argparse.Namespace) -> None:
    q = TelemetryQuery(args.db)
    data: dict = {}
    data["summary"] = q.summary(start=args.start, end=args.end)
    data["db_stats"] = q.get_db_stats()
    # Export all queryable data
    if not args.table or args.table == "events":
        data["events"] = q.get_system_events(start=args.start, end=args.end, limit=10000)
    if not args.table or args.table == "commands":
        data["command_count"] = q.get_command_count(start=args.start, end=args.end)
    if not args.table or args.table == "cameras":
        data["cameras"] = q.get_camera_health(start=args.start, end=args.end, limit=10000)
    if not args.table or args.table == "latency":
        data["web_requests"] = q.get_web_request_latency(start=args.start, end=args.end, limit=10000)
    print(json.dumps(data, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Query th3cl4w telemetry database")
    parser.add_argument("--db", default=str(_project_root / "data" / "telemetry.db"),
                        help="Path to telemetry database")

    sub = parser.add_subparsers(dest="command", required=True)

    # tail
    p = sub.add_parser("tail", help="Tail system events")
    p.add_argument("--follow", "-f", action="store_true")
    p.add_argument("--limit", "-n", type=int, default=20)
    p.add_argument("--types", help="Comma-separated event types")

    # joints
    p = sub.add_parser("joints", help="Joint angle history")
    p.add_argument("--joint", "-j", type=int, default=0)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--limit", "-n", type=int, default=50)

    # tracking-error
    p = sub.add_parser("tracking-error", help="Tracking error analysis")
    p.add_argument("--joint", "-j", type=int, default=0)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--verbose", "-v", action="store_true")

    # rate
    p = sub.add_parser("rate", help="Command/feedback rates")
    p.add_argument("--window", "-w", type=float, default=10.0)

    # gripper
    p = sub.add_parser("gripper", help="Gripper position log")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--limit", "-n", type=int, default=50)

    # events
    p = sub.add_parser("events", help="System events")
    p.add_argument("--type", "-t", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--limit", "-n", type=int, default=50)

    # cameras
    p = sub.add_parser("cameras", help="Camera health log")
    p.add_argument("--cam", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--limit", "-n", type=int, default=50)

    # latency
    p = sub.add_parser("latency", help="Web request latency")
    p.add_argument("--endpoint", "-e", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--limit", "-n", type=int, default=50)

    # smoother
    p = sub.add_parser("smoother", help="Smoother state history")
    p.add_argument("--joint", "-j", type=int, default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--limit", "-n", type=int, default=50)

    # summary
    p = sub.add_parser("summary", help="Session summary")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)

    # export
    p = sub.add_parser("export", help="Export data as JSON")
    p.add_argument("--table", default=None, help="Specific table to export")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)

    args = parser.parse_args()
    handlers = {
        "tail": cmd_tail, "joints": cmd_joints, "tracking-error": cmd_tracking_error,
        "rate": cmd_rate, "gripper": cmd_gripper, "events": cmd_events,
        "cameras": cmd_cameras, "latency": cmd_latency, "smoother": cmd_smoother,
        "summary": cmd_summary, "export": cmd_export,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
