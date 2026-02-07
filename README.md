# th3cl4w

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
| **Repeatability** | +/-0.1mm |
| **Weight** | 2.2kg |
| **Communication** | Ethernet (UDP), CAN |
| **Control Frequency** | 500Hz |

## Features

- [x] D1 connection interface (UDP, thread-safe, context manager)
- [x] Safety limits (joint position/velocity/torque clamping and validation)
- [x] Watchdog timer (emergency stop on command timeout)
- [x] PD joint position controller
- [x] 500Hz real-time control loop with integrated safety
- [x] Forward kinematics (modified DH convention)
- [x] Joint-space trajectory planning (linear and cubic interpolation)
- [x] Gripper control
- [ ] Cartesian end-effector control
- [ ] Inverse kinematics
- [ ] ROS2 integration
- [ ] MoveIt2 support
- [ ] Collision detection
- [ ] Teleoperation interface

## Project Structure

```
th3cl4w/
├── src/
│   ├── interface/     # D1 connection (UDP), state/command serialization
│   ├── safety/        # Joint limits, command validation, watchdog
│   ├── control/       # Control loop, PD controller
│   ├── kinematics/    # DH parameters, forward kinematics
│   └── planning/      # Trajectory generation and interpolation
├── examples/          # Usage examples
├── tests/             # Unit tests (100 tests)
└── .github/workflows/ # CI pipeline
```

## Quick Start

```bash
# Clone the repo
git clone https://github.com/sarahtwilliams412-bit/th3cl4w.git
cd th3cl4w

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run connection test (requires D1 arm on network)
python examples/connection_test.py
```

## Usage

### Basic connection

```python
from src.interface.d1_connection import D1Connection, D1Command

with D1Connection() as conn:
    state = conn.get_state()
    print(f"Joint positions: {state.joint_positions}")

    conn.send_command(D1Command(mode=0))  # idle
```

### Safe joint control

```python
from src.interface.d1_connection import D1Connection
from src.control.controller import JointPositionController
from src.control.loop import ControlLoop
from src.safety.limits import D1SafetyLimits
import numpy as np

conn = D1Connection()
conn.connect()

ctrl = JointPositionController()
ctrl.set_target(np.array([0.0, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0]))

loop = ControlLoop(conn, ctrl, safety_limits=D1SafetyLimits())
loop.start()

# ... arm moves to target under PD control with safety limits ...

loop.stop()
conn.disconnect()
```

### Trajectory execution

```python
from src.planning.trajectory import create_smooth_trajectory
import numpy as np

start = np.zeros(7)
end = np.array([0.5, 0.3, -0.2, 0.4, 0.1, -0.1, 0.0])
traj = create_smooth_trajectory(start, end, duration=2.0)

# Sample at any time
positions = traj.sample(1.0)  # midpoint
velocity = traj.sample_velocity(1.0)
```

### Forward kinematics

```python
from src.kinematics.forward import forward_kinematics, end_effector_position
import numpy as np

joints = np.array([0.1, -0.2, 0.3, 0.4, -0.1, 0.2, 0.0])
T = forward_kinematics(joints)   # 4x4 homogeneous transform
pos = end_effector_position(joints)  # [x, y, z] in meters
```

## Requirements

- Python 3.10+
- NumPy, SciPy
- Unitree SDK (optional, for direct hardware access)
- ROS2 Humble (optional)
- MoveIt2 (optional)

## Hardware Setup

1. Connect D1 arm to host via Ethernet
2. Configure network interface (192.168.123.x subnet)
3. Power on arm and wait for initialization
4. Run connection test: `python examples/connection_test.py`

## Safety

**IMPORTANT:** The D1 arm can move with significant force. Always:
- Keep clear of the arm's workspace during operation
- Use the built-in safety limits (`D1SafetyLimits`) — they are enforced automatically by `ControlLoop`
- Use the watchdog timer to auto-stop on control program failure
- Set appropriate velocity and torque limits for your application
- Test new code in simulation first
- Have emergency stop accessible

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/ tests/

# Type check
mypy src/ --ignore-missing-imports
```

## Resources

- [Unitree D1 Product Page](https://www.unitree.com/d1/)
- [Unitree SDK Documentation](https://support.unitree.com/)

## License

MIT
