from .d1_connection import D1Connection, D1State, D1Command, connect_d1, NUM_JOINTS

# DDS imports are deferred — cyclonedds is a C library that may not be installed
# in all environments (tests, CI, dev machines without the arm).
# Use: from src.interface.d1_dds_connection import D1DDSConnection
try:
    from .d1_dds_connection import D1DDSConnection, ArmString_, connect_d1_dds
except ImportError:
    D1DDSConnection = None  # type: ignore[assignment,misc]
    ArmString_ = None  # type: ignore[assignment,misc]
    connect_d1_dds = None  # type: ignore[assignment,misc]

__all__ = [
    "D1Connection",
    "D1DDSConnection",
    "D1State",
    "D1Command",
    "ArmString_",
    "connect_d1",
    "connect_d1_dds",
    "NUM_JOINTS",
]
