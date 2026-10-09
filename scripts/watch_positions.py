#!/usr/bin/env python
"""Live read-only display of all 7 motor positions (5 Hz). Ctrl-C to quit.

Usage (rebot_teleop env):  python scripts/watch_positions.py
Never enables torque; safe to run while moving the arm by hand.
"""
import math
import time

from motorbridge import Controller

MOTORS = {
    1: ("joint1", "rs-06"), 2: ("joint2", "rs-06"), 3: ("joint3", "rs-06"),
    4: ("joint4", "rs-00"), 5: ("joint5", "rs-00"), 6: ("joint6", "rs-00"),
    7: ("gripper", "rs-00"),
}
MECH_POS = 0x7019


def main() -> None:
    ctrl = Controller("can0@1000000")
    motors = {}
    try:
        for mid, (name, model) in MOTORS.items():
            m = ctrl.add_robstride_motor(mid, 0xFD, model)
            m.set_can_timeout_ms(200)
            motors[name] = m
        print("joint positions in degrees (Ctrl-C to quit)")
        print("  ".join(f"{n:>8s}" for n in motors))
        while True:
            vals = []
            for name, m in motors.items():
                try:
                    vals.append(math.degrees(m.robstride_get_param_f32(MECH_POS)))
                except Exception:
                    vals.append(float("nan"))
            print("\r" + "  ".join(f"{v:+8.1f}" for v in vals), end="", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print()
    finally:
        ctrl.close()


if __name__ == "__main__":
    main()
