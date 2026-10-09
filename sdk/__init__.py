"""reBotArm hardware SDK (motorbridge-backed). API cheat-sheet:

    from sdk import RebotArm, make_controller

    arm = RebotArm()                  # cfg_path=None -> sdk/configs/rebotarm.yaml
                                      # (follows its hardware_yaml indirection)
    arm.connect()                     # opens the bus (sdk.transport.make_controller)
                                      # and registers motors; no motion yet
    arm.arm.enable()                  # per-group: arm (joint1..joint6) / gripper
    arm.gripper.enable()              #   (NoOpGroup stub when unconfigured)
    arm.arm.mode_mit()                # or mode_pos_vel() / mode_vel(); set the
    arm.gripper.mode_mit()            #   mode BEFORE streaming commands

    arm.arm.send_mit(q6)              # q6: (6,) rad; optional vel/kp/kd/tau.
                                      #   Config gains used when kp/kd omitted.
    arm.arm.send_pos_vel(q6)          # firmware-side profiled position moves
    q = arm.arm.get_positions()       # (6,) rad; requests + polls feedback
    pos, vel, tau = arm.get_state()   # all 7 motors at once

    arm.set_zero()                    # ARM MUST BE HELD at the zero pose
                                      #   (fully extended): disables, then burns
                                      #   the current position as zero
    arm.estop()                       # disable_all(): motors go torque-free —
                                      #   the arm FREE-FALLS under gravity; catch
                                      #   it or rest it before disabling
    arm.disconnect()                  # stop loop + disable + close bus

    arm.start_control_loop(fn, rate)  # fn(arm, dt) on a daemon thread
    make_controller(channel, transport="auto", baud=921600, vendor="robstride")
                                      # channel/transport resolution (see
                                      #   sdk.transport docstring)

Bus budget: one 1 Mbps CAN bus, 7 motors. Command streaming is validated at
125 Hz (a 500 Hz loop overruns continuously); explicit feedback requests
(get_positions outside the MIT echo) are budgeted at ~20 Hz.
"""
from sdk.rebotarm import (
    JointCfg,
    JointGroup,
    NoOpGroup,
    RebotArm,
    load_cfg,
    load_gravity_compensation_config,
)
from sdk.transport import make_controller

__all__ = [
    "JointCfg",
    "JointGroup",
    "NoOpGroup",
    "RebotArm",
    "load_cfg",
    "load_gravity_compensation_config",
    "make_controller",
]
