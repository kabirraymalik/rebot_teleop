# ReBot Arm RS Mechanical Description Package

<p align="center">
  <strong>
    <a href="./README_zh.md">简体中文</a> &nbsp;|&nbsp;
    <a href="./README.md">English</a>
  </strong>
</p>

This directory contains the self-contained B601-RS mechanical description resources. See the parent [`README.md`](../README.md) for the overall RS/DM differences and cross-engine reuse guidance.

```text
RS/
├── urdf/
│   └── ReBot_Arm_RS.urdf
└── meshes/
    ├── visual/             # CNC, motor, and PLA parts used only for rendering
    ├── shared/             # Complete meshes shared by rendering and URDF collision
    └── mujoco_collision/   # Detailed MuJoCo finger-collision meshes
```

- `urdf/ReBot_Arm_RS.urdf` is the current main RS model and defines both the arm and gripper.
- `meshes/visual/` contains 22 STL files used only for rendering and colors.
- `meshes/shared/` contains 10 STL files used for both rendering and collision. RS currently has no independent meshes used only for URDF collision.
- `meshes/mujoco_collision/` contains 10 MuJoCo collision meshes for the front, middle, and rear finger segments, carriages, and travel stops.
- The URDF uses `../meshes/...` relative paths and does not depend on ROS package URIs. Copy the complete `RS/` directory to use it.
- The ROS 2 runtime model remains in `rebotarm_ros2_RS/src/rebotarm_bringup/description/`. Compare and synchronize the two locations after model updates.

For example, load it from the repository root with:

```python
from pathlib import Path

urdf_path = Path("Rebot_Arm_description/RS/urdf/ReBot_Arm_RS.urdf")
```
