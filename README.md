# rebot_teleop

Teleoperation for the Seeed reBot B601-RS arm (6 DOF + gripper, RobStride motors).

Two modes, built in order:

1. **EE-pose teleop** — a 6-DOF end-effector pose + 1-DOF gripper command is moved with
   keyboard or PS4 controller; IK makes the MuJoCo arm track it; the real arm mirrors the sim.
2. **Leader–follower** (planned) — a second reBot arm acts as a passive leader; the follower
   mirrors its joints directly.

## Setup

```sh
conda create -n rebot_teleop -c conda-forge --override-channels python=3.11 -y
conda activate rebot_teleop
pip install -r requirements.txt
```

On macOS, anything that opens the interactive MuJoCo viewer must run under `mjpython`
(ships with the mujoco wheel), not plain `python`.

## Layout

| Path | What it is |
| --- | --- |
| `sdk/` | Minimal real-arm SDK: `rebotarm.py` (extracted from Seeed's reBotArm_control_py) over the `motorbridge` pip package, plus hardware YAML configs |
| `sim/models/` | MuJoCo model: `rs_arm.xml` (vendor MJCF: actuators, coupled gripper, `tcp` site) + meshes + scenes |
| `sim/` | IK (damped-least-squares on MuJoCo Jacobians) |
| `teleop/` | EE command state, keyboard/PS4 input backends, real-arm mirror loop, leader-arm policy |
| `scripts/` | Entry points (`teleop_sim.py`, …) |
| `robot_assets/` | Official RS description package (URDF + STLs), reference only |
| `dev_helpers/` | Vendor repos used as reference during development — will be removed |

## Hardware notes

- **Transport**: the arm's 7 motors (ids `0x01`–`0x07`) sit on one CAN bus at 1 Mbps.
  On Linux the default is SocketCAN (`can0`); on macOS the Seeed/RobStride USB-CAN serial
  board (CH340, shows up as `/dev/cu.usbserial-*`) is used instead. The SDK auto-selects.
- **Rates**: stream joint commands at **125 Hz** (500 Hz overruns the bus with 7 motors).
  Position feedback is a per-joint register read — poll it at ~20 Hz, outside the send loop.
- **Zeroing**: motor zero = URDF zero = arm fully extended. Zero with the arm at rest,
  motors disabled (`set_zero`), before trusting positions.

## Safety

- Disabling motors (`disable_all`, estop, disconnect) cuts torque — a raised arm **free-falls**.
  Always ramp to a rest pose first.
- Real-arm motion is gated behind an explicit confirmation environment variable; sim-only
  mode never touches the bus.
- First hardware runs: low speed (≤0.4 rad/s), one joint at a time, hand on the e-stop.
