"""
Code Improver — the arm's self-modification engine.

Takes feedback (especially parameter_adjustments) and translates them into
actual changes the arm applies to itself:

1. Parameter tuning: adjusts speed factors, tolerances, thresholds,
   PID gains stored in a persistent improvement config.
2. Strategy learning: records which strategies worked for which tasks
   so the arm selects better approaches in the future.
3. Code patches: for structural improvements, generates patch descriptions
   that can be reviewed and applied.

All improvements are:
- Versioned and reversible (every change is logged with a rollback snapshot)
- Validated before application (sanity-checked against safety bounds)
- Gradual (changes are capped to prevent overcorrection)

The arm literally writes its own configuration, and over time, its own code.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .feedback_generator import Feedback

logger = logging.getLogger("th3cl4w.introspection.code_improver")

DEFAULT_IMPROVEMENTS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "improvements.json"
)
DEFAULT_HISTORY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "improvement_history.jsonl"
)


# ---------------------------------------------------------------------------
# Safety bounds on parameter changes
# ---------------------------------------------------------------------------

# These prevent the self-improvement from doing something dangerous.
# All adjustable parameters are clamped to these ranges.
PARAMETER_SAFETY_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "command_smoother": {
        "max_step_deg": (0.1, 5.0),
    },
    "motion_planner": {
        "speed_factor": (0.1, 1.0),
        "dt": (0.005, 0.05),
    },
    "task_planner": {
        "position_tolerance_deg": (1.0, 15.0),
        "gripper_open_mm": (30.0, 65.0),
        "gripper_close_mm": (0.0, 20.0),
        "approach_offset_deg": (5.0, 30.0),
    },
    "episode_analyzer": {
        "POSITION_TOLERANCE_RAD": (0.02, 0.2),
        "TRACKING_ERROR_LIMIT_RAD": (0.05, 0.3),
        "SMOOTHNESS_JERK_LIMIT": (10.0, 200.0),
    },
}

# Maximum change per improvement cycle (multiplicative)
MAX_ADJUSTMENT_FACTOR = 0.5  # at most 50% change in one step


@dataclass
class Improvement:
    """A single parameter or strategy improvement."""

    timestamp: float = 0.0
    episode_id: str = ""
    target: str = ""  # e.g. "motion_planner"
    parameter: str = ""  # e.g. "speed_factor"
    old_value: float | None = None
    new_value: float | None = None
    reason: str = ""
    applied: bool = False
    rolled_back: bool = False


@dataclass
class StrategyRecord:
    """A learned strategy for a specific task type."""

    task_type: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    success_count: int = 0
    failure_count: int = 0
    last_updated: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


class CodeImprover:
    """Applies self-improvements based on introspection feedback.

    Maintains a persistent config of learned parameter values and strategies.
    All changes are bounded, logged, and reversible.
    """

    def __init__(
        self,
        improvements_path: Path = DEFAULT_IMPROVEMENTS_PATH,
        history_path: Path = DEFAULT_HISTORY_PATH,
    ) -> None:
        self.improvements_path = improvements_path
        self.history_path = history_path

        # Current learned parameters (overrides for defaults)
        self.parameters: dict[str, dict[str, float]] = {}

        # Learned strategies per task type
        self.strategies: dict[str, StrategyRecord] = {}

        # Code patch proposals (for review, not auto-applied)
        self.pending_patches: list[dict] = []

        self._load()

    # -- Persistence --

    def _load(self) -> None:
        try:
            if self.improvements_path.exists():
                data = json.loads(self.improvements_path.read_text())
                self.parameters = data.get("parameters", {})
                for task, rec in data.get("strategies", {}).items():
                    self.strategies[task] = StrategyRecord(
                        task_type=rec.get("task_type", task),
                        parameters=rec.get("parameters", {}),
                        success_count=rec.get("success_count", 0),
                        failure_count=rec.get("failure_count", 0),
                        last_updated=rec.get("last_updated", 0.0),
                        notes=rec.get("notes", []),
                    )
                self.pending_patches = data.get("pending_patches", [])
                logger.info(
                    "Loaded improvements: %d parameter sets, %d strategies",
                    len(self.parameters),
                    len(self.strategies),
                )
        except Exception as e:
            logger.error("Failed to load improvements: %s", e)

    def _save(self) -> None:
        try:
            self.improvements_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "parameters": self.parameters,
                "strategies": {
                    k: {
                        "task_type": v.task_type,
                        "parameters": v.parameters,
                        "success_count": v.success_count,
                        "failure_count": v.failure_count,
                        "last_updated": v.last_updated,
                        "notes": v.notes[-20:],
                    }
                    for k, v in self.strategies.items()
                },
                "pending_patches": self.pending_patches[-50:],
                "last_saved": time.time(),
            }
            self.improvements_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error("Failed to save improvements: %s", e)

    def _log_history(self, improvement: Improvement) -> None:
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "timestamp": improvement.timestamp,
                            "episode_id": improvement.episode_id,
                            "target": improvement.target,
                            "parameter": improvement.parameter,
                            "old_value": improvement.old_value,
                            "new_value": improvement.new_value,
                            "reason": improvement.reason,
                            "applied": improvement.applied,
                        }
                    )
                    + "\n"
                )
        except Exception as e:
            logger.error("Failed to log improvement history: %s", e)

    # -- Core improvement logic --

    def process_feedback(self, feedback: Feedback) -> list[Improvement]:
        """Process feedback and apply safe parameter adjustments.

        Returns the list of improvements that were applied.
        """
        improvements: list[Improvement] = []

        # 1. Apply parameter adjustments from feedback
        for adj in feedback.parameter_adjustments:
            imp = self._apply_parameter_adjustment(adj, feedback.episode_id)
            if imp is not None:
                improvements.append(imp)

        # 2. Update strategy record for this task type
        self._update_strategy(feedback)

        # 3. Generate code patches if needed
        self._generate_patches(feedback)

        self._save()
        return improvements

    def _apply_parameter_adjustment(self, adj: dict, episode_id: str) -> Improvement | None:
        """Apply a single parameter adjustment with safety bounds."""
        target = adj.get("target", "")
        param = adj.get("parameter", "")
        direction = adj.get("direction", "")
        factor = adj.get("suggested_factor", 1.0)

        # Validate
        if not target or not param:
            return None

        # Get current value (from learned params or use 1.0 as default scale)
        current = self.parameters.get(target, {}).get(param)
        if current is None:
            # First time adjusting this parameter — use 1.0 as the baseline scale
            current = 1.0

        # Compute new value.
        # suggested_factor encodes the multiplier: 0.8 means "reduce to 80%",
        # 1.3 means "increase to 130%". Direction is informational but the
        # factor already captures the intent.
        if direction == "increase":
            raw_factor = max(factor, 1.0)  # ensure increase
        elif direction == "decrease":
            raw_factor = min(factor, 1.0)  # ensure decrease
        else:
            return None

        # Clamp the adjustment factor to prevent overcorrection
        clamped_factor = np.clip(
            raw_factor,
            1.0 - MAX_ADJUSTMENT_FACTOR,
            1.0 + MAX_ADJUSTMENT_FACTOR,
        )
        new_value = current * clamped_factor

        # Apply safety bounds
        bounds = PARAMETER_SAFETY_BOUNDS.get(target, {}).get(param)
        if bounds is not None:
            new_value = float(np.clip(new_value, bounds[0], bounds[1]))

        # Only apply if the change is meaningful
        if abs(new_value - current) < 1e-6:
            return None

        # Apply
        if target not in self.parameters:
            self.parameters[target] = {}
        self.parameters[target][param] = new_value

        imp = Improvement(
            timestamp=time.time(),
            episode_id=episode_id,
            target=target,
            parameter=param,
            old_value=current,
            new_value=new_value,
            reason=adj.get("reason", ""),
            applied=True,
        )

        self._log_history(imp)

        logger.info(
            "Applied improvement: %s.%s: %.4f -> %.4f (%s)",
            target,
            param,
            current,
            new_value,
            imp.reason,
        )

        return imp

    def _update_strategy(self, feedback: Feedback) -> None:
        """Update the strategy record for this task type."""
        task = feedback.task_name
        if not task:
            return

        if task not in self.strategies:
            self.strategies[task] = StrategyRecord(task_type=task)

        strategy = self.strategies[task]
        strategy.last_updated = time.time()

        if feedback.verdict == "success":
            strategy.success_count += 1
            for note in feedback.strategy_notes:
                if note not in strategy.notes:
                    strategy.notes.append(note)
        elif feedback.verdict == "failure":
            strategy.failure_count += 1

        # Store the parameters that produced this result
        # (successful params overwrite, failed ones don't)
        if feedback.verdict == "success" and feedback.parameter_adjustments:
            for adj in feedback.parameter_adjustments:
                key = f"{adj.get('target', '')}.{adj.get('parameter', '')}"
                val = self.parameters.get(adj.get("target", ""), {}).get(adj.get("parameter", ""))
                if val is not None:
                    strategy.parameters[key] = val

    def _generate_patches(self, feedback: Feedback) -> None:
        """Generate code patch proposals for structural improvements.

        These are not auto-applied — they go into pending_patches for review.
        The arm (or a human) can decide whether to apply them.
        """
        # If recurring failure patterns are detected, propose code changes
        if feedback.verdict == "failure":
            for cause in feedback.root_causes:
                if "collision memory" in cause.lower():
                    self.pending_patches.append(
                        {
                            "timestamp": time.time(),
                            "episode_id": feedback.episode_id,
                            "type": "strategy_change",
                            "description": (
                                "Consider adding pre-execution collision preview "
                                "that checks the planned trajectory against collision "
                                "memory before sending commands."
                            ),
                            "target_file": "src/planning/motion_planner.py",
                            "priority": "medium",
                        }
                    )
                elif "tracking error" in cause.lower() and "pid" in cause.lower():
                    self.pending_patches.append(
                        {
                            "timestamp": time.time(),
                            "episode_id": feedback.episode_id,
                            "type": "parameter_tune",
                            "description": (
                                "Consider implementing adaptive PID gains that "
                                "increase proportional gain when tracking error "
                                "exceeds threshold for more than 200ms."
                            ),
                            "target_file": "src/control/joint_controller.py",
                            "priority": "high",
                        }
                    )

    # -- Query interface --

    def get_parameter(self, target: str, param: str, default: float | None = None) -> float | None:
        """Get the current learned value for a parameter."""
        return self.parameters.get(target, {}).get(param, default)

    def get_strategy(self, task_type: str) -> StrategyRecord | None:
        """Get the learned strategy for a task type."""
        return self.strategies.get(task_type)

    def get_best_speed_factor(self, task_type: str) -> float:
        """Get the best known speed factor for a task type."""
        learned = self.get_parameter("motion_planner", "speed_factor")
        strategy = self.get_strategy(task_type)

        if strategy and strategy.success_rate > 0.7:
            strat_speed = strategy.parameters.get("motion_planner.speed_factor")
            if strat_speed is not None:
                return float(strat_speed)

        if learned is not None:
            return float(learned)

        return 1.0  # default

    def get_pending_patches(self) -> list[dict]:
        """Get code patches awaiting review."""
        return list(self.pending_patches)

    def clear_pending_patches(self) -> None:
        self.pending_patches.clear()
        self._save()

    def rollback_last(self, target: str, param: str) -> bool:
        """Rollback the most recent change to a parameter.

        Reads the history log to find the previous value and restores it.
        """
        if not self.history_path.exists():
            return False

        entries = []
        try:
            with open(self.history_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except Exception:
            return False

        # Find the most recent entry for this target.param
        for entry in reversed(entries):
            if entry.get("target") == target and entry.get("parameter") == param:
                old_val = entry.get("old_value")
                if old_val is not None:
                    if target not in self.parameters:
                        self.parameters[target] = {}
                    self.parameters[target][param] = old_val
                    self._save()
                    logger.info("Rolled back %s.%s to %s", target, param, old_val)
                    return True

        return False

    def summary(self) -> dict:
        """Summary of all learned improvements."""
        total_params = sum(len(v) for v in self.parameters.values())
        total_strategies = len(self.strategies)
        strategy_summary = {
            k: {
                "success_rate": v.success_rate,
                "total_attempts": v.success_count + v.failure_count,
            }
            for k, v in self.strategies.items()
        }

        return {
            "total_learned_parameters": total_params,
            "parameters": self.parameters,
            "total_strategies": total_strategies,
            "strategies": strategy_summary,
            "pending_patches": len(self.pending_patches),
        }
