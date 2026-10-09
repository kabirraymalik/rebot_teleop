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

## Run the sim demo

```sh
pip install -e .          # once, from the repo root
mjpython scripts/teleop_sim.py              # keyboard (default)
mjpython scripts/teleop_sim.py --input ps4  # DualShock 4
```

The default scene is empty — floor, arm, and an RGB axis triad (x=red,
y=green, z=blue) marking the commanded EE goal pose; the arm starts at its
resting pose (q = 0, matching the real arm's motor zeros). `--scene
sim/models/teleop_scene.xml` brings back the table + objects.

`--input` accepts `kb` (default) or `ps4`. Key presses go straight to the
MuJoCo viewer window (on macOS; on Linux a small pygame key window opens
instead). Each press moves the EE goal one discrete step (default 1 cm / 5°;
`--lin-step`, `--ang-step`, `--grip-step` to change, F toggles fine
half-steps): arrows = x/y, W/S = z up/down, Q/E roll, Z/X pitch, A/D yaw,
O/C gripper, H home (sim-only), ESC quit. Some letters also flip the viewer's
own display toggles — cosmetic only.
IK tracks the goal and the arm follows through the same low-level stack the
real arm will use (velocity-limited joint reference, 0.3 rad/s default via
`--speed-limit`, + gravity-compensated PD). `--hold` restores hold-to-move
velocity control; `--kinematic` skips physics.

`--mirror` additionally streams to the real arm. It enables torque, latches
the sim to the arm's measured pose (no jump — also your visual check that
zeroing is right), and starts **parked in HOLD**: the real arm holds still
while you preview moves in sim. **H toggles HOLD ↔ TRACK** — on release the
arm catches up to the sim target rate-limited (0.3 rad/s). On exit it ramps
to the rest pose before torque-off. First runs: workspace clear, ≥1 m away,
hand on the power switch.

Headless checks (plain python): `scripts/check_model.py`, `check_ik.py`,
`check_sdk.py`, `check_mirror.py`, `check_e2e.py`.

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

- **Transport**: the arm's 7 motors (ids `0x01`–`0x07`, feedback id `0xFD`) sit on one
  CAN bus at 1 Mbps, reached through the kit's XCAN-USB adapter (enumerates as a PEAK
  PCAN-USB; keep its switch on **120R**, not BOOT). On Linux: SocketCAN `can0`. On macOS:
  the PCBUSB library (`~/.local/lib/libPCBUSB.dylib`, no sudo needed) with channel
  `can0@1000000`. The SDK auto-selects. The CH340-based RobStride AT-mode debug board is
  NOT usable with this stack.
- **Power**: 48 V PSU — check the **115 V/230 V input selector** matches your outlet; on
  the wrong setting the supply browns out and the bus looks completely dead (motors never
  ACK a single CAN frame).
- **Live monitor**: `python scripts/watch_positions.py` (read-only, torque never enabled)
  shows all joint angles at 5 Hz while you move the arm by hand. Only one process can own
  the CAN channel at a time — stop it before running anything else on the bus.
- **Rates**: stream joint commands at **125 Hz** (500 Hz overruns the bus with 7 motors).
  Position feedback is a per-joint register read — poll it at ~20 Hz, outside the send loop.
- **Zeroing**: motor zero = URDF zero = arm fully extended. Zero with the arm at rest,
  motors disabled (`set_zero`), before trusting positions.

## Safety

- Disabling motors (`disable_all`, estop, disconnect) cuts torque — a raised arm **free-falls**.
  Always ramp to a rest pose first.
- `--mirror` starts parked in HOLD — torque on, no motion until you press H. Sim-only
  mode never touches the bus.
- First hardware runs: low speed (≤0.4 rad/s), small steps, hand on the power switch.
