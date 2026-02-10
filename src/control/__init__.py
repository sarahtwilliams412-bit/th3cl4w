# Lazy imports to avoid circular dependencies.
# joint_service is the foundational module — must be importable without
# triggering imports of joint_controller/gravity_compensation/kinematics.


def __getattr__(name):
    if name == "JointController":
        from src.control.joint_controller import JointController
        return JointController
    if name == "PIDGains":
        from src.control.joint_controller import PIDGains
        return PIDGains
    if name == "cubic_interpolation":
        from src.control.joint_controller import cubic_interpolation
        return cubic_interpolation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["JointController", "PIDGains", "cubic_interpolation"]
