"""
Control module for the D1 arm.

Provides the real-time control loop and controller implementations.
"""

from src.control.controller import Controller, JointPositionController
from src.control.loop import ControlLoop

__all__ = [
    "Controller",
    "JointPositionController",
    "ControlLoop",
]
