"""
Central Service Registry for th3cl4w.

All service URLs, ports, and discovery in one place.
Services use environment variables for configuration, with sensible defaults.
"""

import os


class ServiceConfig:
    """Configuration for all th3cl4w services.

    Port assignments:
        8080 — Gateway (unified entry point)
        8081 — Camera
        8082 — World Model
        8083 — Local Model
        8084 — Object ID
        8085 — Kinematics App
        8086 — Mapping
        8087 — Positioning
        8088 — Tasker
        8089 — Simulation
        8090 — Control Plane
        8091 — Telemetry
        8092 — Introspection
        8093 — ASCII
        8094 — Calibration
        6379 — Redis (message bus)
    """

    GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8080"))
    CAMERA_PORT = int(os.getenv("CAMERA_PORT", "8081"))
    WORLD_MODEL_PORT = int(os.getenv("WORLD_MODEL_PORT", "8082"))
    LOCAL_MODEL_PORT = int(os.getenv("LOCAL_MODEL_PORT", "8083"))
    OBJECT_ID_PORT = int(os.getenv("OBJECT_ID_PORT", "8084"))
    KINEMATICS_PORT = int(os.getenv("KINEMATICS_PORT", "8085"))
    MAPPING_PORT = int(os.getenv("MAPPING_PORT", "8086"))
    POSITIONING_PORT = int(os.getenv("POSITIONING_PORT", "8087"))
    TASKER_PORT = int(os.getenv("TASKER_PORT", "8088"))
    SIMULATION_PORT = int(os.getenv("SIMULATION_PORT", "8089"))
    CONTROL_PLANE_PORT = int(os.getenv("CONTROL_PLANE_PORT", "8090"))
    TELEMETRY_PORT = int(os.getenv("TELEMETRY_PORT", "8091"))
    INTROSPECTION_PORT = int(os.getenv("INTROSPECTION_PORT", "8092"))
    ASCII_PORT = int(os.getenv("ASCII_PORT", "8093"))
    CALIBRATION_PORT = int(os.getenv("CALIBRATION_PORT", "8094"))
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

    # Hardware config
    ARM_INTERFACE = os.getenv("ARM_INTERFACE", "eno1")
    SIMULATE = os.getenv("SIMULATE", "false").lower() in ("true", "1", "yes")

    @classmethod
    def url(cls, service: str, path: str = "") -> str:
        """Get the URL for a service.

        Args:
            service: Service name (e.g., 'control_plane', 'camera', 'world_model')
            path: Optional URL path to append (e.g., '/api/state')

        Returns:
            Full URL like 'http://localhost:8090/api/state'
        """
        port_attr = f"{service.upper()}_PORT"
        host_env = f"{service.upper()}_HOST"
        host = os.getenv(host_env, "localhost")
        port = getattr(cls, port_attr, 8080)
        base = f"http://{host}:{port}"
        if path:
            return f"{base}{path}"
        return base

    @classmethod
    def ws_url(cls, service: str, path: str = "") -> str:
        """Get the WebSocket URL for a service."""
        port_attr = f"{service.upper()}_PORT"
        host_env = f"{service.upper()}_HOST"
        host = os.getenv(host_env, "localhost")
        port = getattr(cls, port_attr, 8080)
        base = f"ws://{host}:{port}"
        if path:
            return f"{base}{path}"
        return base

    @classmethod
    def all_services(cls) -> dict[str, int]:
        """Return a dict of service name -> port for all registered services."""
        return {
            "gateway": cls.GATEWAY_PORT,
            "camera": cls.CAMERA_PORT,
            "world_model": cls.WORLD_MODEL_PORT,
            "local_model": cls.LOCAL_MODEL_PORT,
            "object_id": cls.OBJECT_ID_PORT,
            "kinematics": cls.KINEMATICS_PORT,
            "mapping": cls.MAPPING_PORT,
            "positioning": cls.POSITIONING_PORT,
            "tasker": cls.TASKER_PORT,
            "simulation": cls.SIMULATION_PORT,
            "control_plane": cls.CONTROL_PLANE_PORT,
            "telemetry": cls.TELEMETRY_PORT,
            "introspection": cls.INTROSPECTION_PORT,
            "ascii": cls.ASCII_PORT,
            "calibration": cls.CALIBRATION_PORT,
        }
