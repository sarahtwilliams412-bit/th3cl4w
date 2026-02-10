"""Tests for the simulation rehearsal runner."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.planning.sim_rehearsal import (
    AttemptResult,
    RehearsalPhase,
    RehearsalReport,
    RehearsalState,
    SimRehearsalRunner,
)
from src.planning.auto_pick import AutoPickPhase, PickResult


class TestRehearsalReport:
    """Test RehearsalReport data and serialisation."""

    def test_default_report(self):
        report = RehearsalReport()
        assert report.total_attempts == 0
        assert report.success_rate == 0.0
        assert report.recommendation == "no-go"
        assert report.ready_for_physical is False

    def test_report_serialisation(self):
        report = RehearsalReport(
            target="redbull",
            total_attempts=5,
            successful_attempts=4,
            success_rate=0.8,
            required_successes=3,
            best_joints=[1.0, 25.9, 6.7, 0.0, 90.0, 0.0],
            best_target_xy_mm=(100.0, 50.0),
            recommendation="go",
            ready_for_physical=True,
            total_duration_s=12.5,
        )
        d = SimRehearsalRunner._report_to_dict(report)
        assert d["target"] == "redbull"
        assert d["total_attempts"] == 5
        assert d["successful_attempts"] == 4
        assert d["success_rate"] == 0.8
        assert d["recommendation"] == "go"
        assert d["ready_for_physical"] is True
        assert d["best_joints"] == [1.0, 25.9, 6.7, 0.0, 90.0, 0.0]
        assert d["physical_result"] is None

    def test_report_with_physical_result(self):
        report = RehearsalReport(
            target="blue",
            total_attempts=3,
            successful_attempts=3,
            success_rate=1.0,
            required_successes=3,
            recommendation="go",
            ready_for_physical=True,
            physical_result=PickResult(
                success=True,
                phase=AutoPickPhase.DONE,
                joints=[45.0, 30.0, 8.0, 0.0, 90.0, 0.0],
                duration_s=5.2,
            ),
        )
        d = SimRehearsalRunner._report_to_dict(report)
        assert d["physical_result"] is not None
        assert d["physical_result"]["success"] is True
        assert d["physical_result"]["phase"] == "done"


class TestAttemptResult:
    """Test AttemptResult defaults."""

    def test_defaults(self):
        r = AttemptResult(attempt_number=1, success=False, phase_reached="failed")
        assert r.joints == []
        assert r.target_xy_mm == (0.0, 0.0)
        assert r.jitter_applied_mm == (0.0, 0.0)
        assert r.grip_distance_mm == float("inf")
        assert r.error == ""

    def test_success_result(self):
        r = AttemptResult(
            attempt_number=2,
            success=True,
            phase_reached="done",
            joints=[10.0, 25.0, 7.0, 0.0, 90.0, 0.0],
            grip_distance_mm=12.5,
        )
        assert r.success is True
        assert r.grip_distance_mm == 12.5


class TestSimRehearsalRunnerState:
    """Test runner state management (no async execution)."""

    def test_initial_state(self):
        runner = SimRehearsalRunner()
        assert runner.state.phase == RehearsalPhase.IDLE
        assert not runner.running
        assert runner.last_report is None

    def test_get_status(self):
        runner = SimRehearsalRunner()
        status = runner.get_status()
        assert status["phase"] == "idle"
        assert status["running"] is False
        assert status["current_attempt"] == 0
        assert status["report"] is None

    def test_stop_sets_flag(self):
        runner = SimRehearsalRunner()
        runner.stop()
        assert runner._stop_requested is True

    def test_status_includes_report_when_available(self):
        runner = SimRehearsalRunner()
        runner._last_report = RehearsalReport(
            target="redbull",
            total_attempts=3,
            successful_attempts=2,
            success_rate=0.667,
            required_successes=2,
            recommendation="marginal",
            ready_for_physical=True,
        )
        status = runner.get_status()
        assert status["report"] is not None
        assert status["report"]["recommendation"] == "marginal"


class TestAnalysis:
    """Test the _analyse method directly."""

    def _make_runner(self):
        return SimRehearsalRunner()

    def test_all_successes_go(self):
        runner = self._make_runner()
        attempts = [
            AttemptResult(
                attempt_number=i,
                success=True,
                phase_reached="done",
                joints=[1.0, 25.9, 6.7, 0.0, 90.0, 0.0],
                target_xy_mm=(100.0, 50.0),
                grip_distance_mm=10.0 + i,
            )
            for i in range(1, 6)
        ]
        report = runner._analyse("redbull", attempts, required_successes=3)
        assert report.total_attempts == 5
        assert report.successful_attempts == 5
        assert report.success_rate == 1.0
        assert report.recommendation == "go"
        assert report.ready_for_physical is True
        # Best joints should be from attempt with lowest grip distance
        assert report.best_joints == [1.0, 25.9, 6.7, 0.0, 90.0, 0.0]

    def test_all_failures_nogo(self):
        runner = self._make_runner()
        attempts = [
            AttemptResult(
                attempt_number=i,
                success=False,
                phase_reached="failed",
                error="Grip failed",
            )
            for i in range(1, 4)
        ]
        report = runner._analyse("redbull", attempts, required_successes=2)
        assert report.successful_attempts == 0
        assert report.success_rate == 0.0
        assert report.recommendation == "no-go"
        assert report.ready_for_physical is False
        assert report.best_joints == []

    def test_marginal_threshold(self):
        runner = self._make_runner()
        # 3/5 = 60% — exactly at marginal threshold, but meets required_successes
        attempts = []
        for i in range(1, 6):
            success = i <= 3
            attempts.append(
                AttemptResult(
                    attempt_number=i,
                    success=success,
                    phase_reached="done" if success else "gripping",
                    joints=[1.0, 25.9, 6.7, 0.0, 90.0, 0.0] if success else [],
                    target_xy_mm=(100.0, 50.0) if success else (0.0, 0.0),
                    grip_distance_mm=15.0 if success else float("inf"),
                )
            )
        report = runner._analyse("redbull", attempts, required_successes=3)
        assert report.successful_attempts == 3
        assert report.success_rate == 0.6
        assert report.recommendation == "marginal"
        assert report.ready_for_physical is True

    def test_below_marginal_nogo(self):
        runner = self._make_runner()
        # 2/5 = 40% — below marginal
        attempts = []
        for i in range(1, 6):
            success = i <= 2
            attempts.append(
                AttemptResult(
                    attempt_number=i,
                    success=success,
                    phase_reached="done" if success else "detecting",
                    joints=[1.0, 25.9, 6.7, 0.0, 90.0, 0.0] if success else [],
                    grip_distance_mm=15.0 if success else float("inf"),
                )
            )
        report = runner._analyse("redbull", attempts, required_successes=3)
        assert report.successful_attempts == 2
        assert report.recommendation == "no-go"
        assert report.ready_for_physical is False

    def test_phase_failure_counts(self):
        runner = self._make_runner()
        attempts = [
            AttemptResult(attempt_number=1, success=False, phase_reached="detecting", error="No object"),
            AttemptResult(attempt_number=2, success=False, phase_reached="gripping", error="Grip failed"),
            AttemptResult(attempt_number=3, success=False, phase_reached="gripping", error="Grip failed"),
            AttemptResult(attempt_number=4, success=True, phase_reached="done",
                         joints=[1.0, 25.9, 6.7, 0.0, 90.0, 0.0], grip_distance_mm=10.0),
        ]
        report = runner._analyse("redbull", attempts, required_successes=1)
        assert report.phase_failure_counts == {"detecting": 1, "gripping": 2}

    def test_best_joints_picks_lowest_grip_distance(self):
        runner = self._make_runner()
        attempts = [
            AttemptResult(
                attempt_number=1, success=True, phase_reached="done",
                joints=[10.0, 25.0, 7.0, 0.0, 90.0, 0.0],
                target_xy_mm=(100.0, 0.0), grip_distance_mm=20.0,
            ),
            AttemptResult(
                attempt_number=2, success=True, phase_reached="done",
                joints=[10.0, 26.0, 7.5, 0.0, 90.0, 0.0],
                target_xy_mm=(102.0, 1.0), grip_distance_mm=8.0,
            ),
            AttemptResult(
                attempt_number=3, success=True, phase_reached="done",
                joints=[10.0, 24.5, 6.5, 0.0, 90.0, 0.0],
                target_xy_mm=(98.0, -1.0), grip_distance_mm=15.0,
            ),
        ]
        report = runner._analyse("redbull", attempts, required_successes=2)
        # Should pick attempt 2 with grip_distance=8.0
        assert report.best_joints == [10.0, 26.0, 7.5, 0.0, 90.0, 0.0]
        assert report.best_target_xy_mm == (102.0, 1.0)

    def test_avg_grip_distance(self):
        runner = self._make_runner()
        attempts = [
            AttemptResult(attempt_number=1, success=True, phase_reached="done",
                         joints=[1.0, 25.0, 7.0, 0.0, 90.0, 0.0], grip_distance_mm=10.0),
            AttemptResult(attempt_number=2, success=True, phase_reached="done",
                         joints=[1.0, 25.0, 7.0, 0.0, 90.0, 0.0], grip_distance_mm=20.0),
            AttemptResult(attempt_number=3, success=False, phase_reached="failed"),
        ]
        report = runner._analyse("redbull", attempts, required_successes=2)
        assert report.avg_grip_distance_mm == 15.0


class TestValidation:
    """Test input validation."""

    @pytest.mark.asyncio
    async def test_required_exceeds_max_raises(self):
        runner = SimRehearsalRunner()
        with pytest.raises(ValueError, match="required_successes.*cannot exceed"):
            await runner.start(
                target="redbull",
                max_attempts=3,
                required_successes=5,
            )

    @pytest.mark.asyncio
    async def test_double_start_raises(self):
        runner = SimRehearsalRunner()
        runner._running = True
        with pytest.raises(RuntimeError, match="already in progress"):
            await runner.start(target="redbull")

    @pytest.mark.asyncio
    async def test_promote_without_report_raises(self):
        runner = SimRehearsalRunner()
        with pytest.raises(RuntimeError, match="No rehearsal report"):
            await runner.promote_to_physical()

    @pytest.mark.asyncio
    async def test_promote_not_ready_raises(self):
        runner = SimRehearsalRunner()
        runner._last_report = RehearsalReport(
            recommendation="no-go",
            ready_for_physical=False,
        )
        with pytest.raises(RuntimeError, match="not ready for physical"):
            await runner.promote_to_physical()
