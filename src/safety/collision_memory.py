"""
Collision Memory — Persistent record of where the arm has hit things.

Once a collision is detected at a joint angle, that angle (and beyond) is
permanently marked as unsafe. No command should ever go there again until
the memory is explicitly cleared (workspace changed).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("th3cl4w.collision_memory")

DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "collision_memory.json"


class CollisionMemory:
    """
    Tracks safe angle ranges per joint based on collision history.
    
    When a collision is recorded at J2=35°, we mark J2 >= 35° as unsafe
    (or J2 <= -35° for negative direction). The arm should never be
    commanded beyond these learned limits.
    """
    
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = path
        # Per-joint safe ranges: {joint_id: (safe_min, safe_max)}
        # Starts at hardware limits, narrows as collisions are found
        self.safe_ranges: Dict[int, Tuple[float, float]] = {
            0: (-135.0, 135.0),
            1: (-90.0, 90.0),
            2: (-90.0, 90.0),
            3: (-135.0, 135.0),
            4: (-90.0, 90.0),
            5: (-135.0, 135.0),
        }
        # Raw collision events for reference
        self.events: List[Dict] = []
        self._load()
    
    def _load(self):
        """Load collision memory from disk."""
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text())
                if 'safe_ranges' in data:
                    self.safe_ranges = {int(k): tuple(v) for k, v in data['safe_ranges'].items()}
                if 'events' in data:
                    self.events = data['events']
                logger.info("Loaded collision memory: %d events, ranges: %s",
                           len(self.events), self.safe_ranges)
        except Exception as e:
            logger.error("Failed to load collision memory: %s", e)
    
    def _save(self):
        """Persist collision memory to disk."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'safe_ranges': {str(k): list(v) for k, v in self.safe_ranges.items()},
                'events': self.events[-100:],  # keep last 100
            }
            self.path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error("Failed to save collision memory: %s", e)
    
    def record_collision(self, joint_id: int, commanded_deg: float, actual_deg: float):
        """
        Record a collision and narrow the safe range for that joint.
        
        If commanded > 0 and collision occurred, the safe max becomes
        min(current_safe_max, actual_deg - margin).
        If commanded < 0, the safe min becomes max(current_safe_min, actual_deg + margin).
        """
        margin = 5.0  # degrees of safety margin beyond where collision happened
        
        lo, hi = self.safe_ranges.get(joint_id, (-180, 180))
        
        if commanded_deg > 0:
            # Hit something going positive — cap the max
            new_hi = min(hi, actual_deg - margin)
            self.safe_ranges[joint_id] = (lo, max(lo + 1, new_hi))
            logger.warning("J%d: collision at cmd=%.1f° actual=%.1f° → safe max now %.1f°",
                          joint_id, commanded_deg, actual_deg, self.safe_ranges[joint_id][1])
        else:
            # Hit something going negative — raise the min
            new_lo = max(lo, actual_deg + margin)
            self.safe_ranges[joint_id] = (min(hi - 1, new_lo), hi)
            logger.warning("J%d: collision at cmd=%.1f° actual=%.1f° → safe min now %.1f°",
                          joint_id, commanded_deg, actual_deg, self.safe_ranges[joint_id][0])
        
        self.events.append({
            'joint_id': joint_id,
            'commanded_deg': commanded_deg,
            'actual_deg': actual_deg,
            'new_range': list(self.safe_ranges[joint_id]),
        })
        
        self._save()
    
    def is_safe(self, joint_id: int, target_deg: float) -> bool:
        """Check if a target angle is within the known-safe range."""
        lo, hi = self.safe_ranges.get(joint_id, (-180, 180))
        return lo <= target_deg <= hi
    
    def clamp_to_safe(self, joint_id: int, target_deg: float) -> float:
        """Clamp a target angle to the known-safe range."""
        lo, hi = self.safe_ranges.get(joint_id, (-180, 180))
        return max(lo, min(hi, target_deg))
    
    def get_safe_range(self, joint_id: int) -> Tuple[float, float]:
        """Get the current safe range for a joint."""
        return self.safe_ranges.get(joint_id, (-180, 180))
    
    def clear(self):
        """Clear collision memory (e.g., after workspace rearrangement)."""
        self.safe_ranges = {
            0: (-135.0, 135.0),
            1: (-90.0, 90.0),
            2: (-90.0, 90.0),
            3: (-135.0, 135.0),
            4: (-90.0, 90.0),
            5: (-135.0, 135.0),
        }
        self.events = []
        self._save()
        logger.info("Collision memory cleared")


# Global instance
_memory = None


def get_collision_memory(path: Path = DEFAULT_PATH) -> CollisionMemory:
    """Get or create the global collision memory instance."""
    global _memory
    if _memory is None:
        _memory = CollisionMemory(path)
    return _memory
