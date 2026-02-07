# 🦾 th3cl4w

**Control system for Unitree D1 robotic arm**

---

## Overview

th3cl4w is a control interface for the [Unitree D1](https://www.unitree.com/d1/) dexterous robotic arm. Designed for integration with Unitree quadrupeds (Go2, B2) and humanoids (G1, H1).

## D1 Arm Specifications

| Spec | Value |
|------|-------|
| **DOF** | 7 (6 arm + 1 gripper) |
| **Reach** | 550mm |
| **Payload** | 1kg (rated), 3kg (max) |
| **Repeatability** | ±0.1mm |
| **Weight** | 2.2kg |
| **Communication** | Ethernet (UDP), CAN |
| **Control Frequency** | 500Hz |

## Features

- [ ] Direct joint control (position, velocity, torque)
- [ ] Cartesian end-effector control
- [ ] Gripper control
- [ ] Trajectory planning
- [ ] ROS2 integration
- [ ] MoveIt2 support
- [ ] Safety limits and collision detection
- [ ] Teleoperation interface

## Project Structure

```
th3cl4w/
├── src/
│   ├── control/       # Core control algorithms
│   ├── kinematics/    # Forward/inverse kinematics
│   ├── planning/      # Motion planning
│   ├── interface/     # Communication with D1
│   └── safety/        # Safety monitors
├── examples/          # Usage examples
└── tests/             # Unit tests
```

## Quick Start

```bash
# Clone the repo
git clone https://github.com/sarahtwilliams412-bit/th3cl4w.git
cd th3cl4w

# Install in development mode
pip install -e ".[dev]"

# Run connection test
python examples/connection_test.py
```

## Requirements

- Python 3.10+
- Unitree SDK (arm)
- NumPy, SciPy
- ROS2 Humble (optional)
- MoveIt2 (optional)

## Hardware Setup

1. Connect D1 arm to host via Ethernet
2. Configure network interface (192.168.123.x subnet)
3. Power on arm and wait for initialization
4. Run connection test: `python examples/connection_test.py`

## Safety

⚠️ **IMPORTANT:** The D1 arm can move with significant force. Always:
- Keep clear of the arm's workspace during operation
- Set appropriate velocity and torque limits
- Test new code in simulation first
- Have emergency stop accessible

## Resources

- [Unitree D1 Product Page](https://www.unitree.com/d1/)
- [Unitree SDK Documentation](https://support.unitree.com/)
- [IShitYouNot Project](https://github.com/adb-123/IShitYouNot) — D1 integration reference

## License

MIT

---

*Part of the Unitree Mastery Mission*
