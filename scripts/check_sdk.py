"""Headless SDK sanity checks — no hardware, no bus is ever opened.

Run with the project env python:
    /Users/krm/miniconda3/envs/rebot_teleop/bin/python scripts/check_sdk.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sdk
import sdk.transport as transport_mod
from sdk import RebotArm, load_cfg


class TestConfigParsing(unittest.TestCase):
    def test_default_config(self) -> None:
        cfg = load_cfg()
        self.assertEqual(len(cfg["joints"]), 7)
        self.assertEqual([j.motor_id for j in cfg["joints"]], list(range(1, 8)))
        self.assertEqual(cfg["rate"], 125.0)
        self.assertEqual(cfg["channel"], "auto")
        self.assertEqual(cfg["transport"], "auto")
        self.assertEqual(cfg["baud"], 921600)
        self.assertTrue(all(j.vendor == "robstride" for j in cfg["joints"]))

    def test_explicit_cfg_path(self) -> None:
        rs = Path(sdk.__file__).parent / "configs" / "rebotarm_rs.yaml"
        cfg = load_cfg(rs)
        self.assertEqual(len(cfg["joints"]), 7)
        self.assertEqual(cfg["rate"], 125.0)


class TestRebotArmConstruction(unittest.TestCase):
    def test_construct_and_connect_with_stubbed_bus(self) -> None:
        fake_ctrl = MagicMock(name="Controller")
        with mock.patch("sdk.transport.make_controller",
                        return_value=fake_ctrl) as mk:
            arm = RebotArm()
            self.assertEqual(arm.num_joints, 7)
            self.assertEqual(
                arm.joint_names,
                ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
                 "gripper"],
            )
            self.assertEqual(set(arm.groups), {"arm", "gripper"})
            self.assertEqual(
                arm.groups["arm"].joint_names,
                ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            )
            self.assertEqual(arm.groups["gripper"].joint_names, ["gripper"])
            self.assertTrue(arm.has_gripper)
            self.assertEqual(arm.rate, 125.0)
            self.assertEqual(arm.arm_control_mode, "mit")

            # Hardware-validated arm MIT gains from the config.
            g = arm.groups["arm"]
            self.assertEqual(list(g._mit_kp), [80.0, 150.0, 150.0, 50.0, 50.0, 50.0])
            self.assertEqual(list(g._mit_kd), [5.0, 10.0, 10.0, 5.0, 4.0, 4.0])

            mk.assert_not_called()  # constructing must not touch the bus
            arm.connect()
            mk.assert_called_once_with("auto", "auto", 921600, "robstride")
            self.assertEqual(fake_ctrl.add_robstride_motor.call_count, 7)
            ids = [c.args[0] for c in fake_ctrl.add_robstride_motor.call_args_list]
            self.assertEqual(ids, list(range(1, 8)))


class TestMakeControllerResolution(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.object(transport_mod, "Controller", autospec=False)
        self.Controller = patcher.start()
        self.addCleanup(patcher.stop)

    def test_explicit_socketcan(self) -> None:
        transport_mod.make_controller("can0", "socketcan")
        self.Controller.assert_called_once_with("can0")

    def test_explicit_pcan_appends_bitrate(self) -> None:
        transport_mod.make_controller("can0", "pcan")
        self.Controller.assert_called_once_with("can0@1000000")

    def test_explicit_pcan_keeps_bitrate(self) -> None:
        transport_mod.make_controller("can0@500000", "pcan")
        self.Controller.assert_called_once_with("can0@500000")

    def test_explicit_mcu_serial(self) -> None:
        transport_mod.make_controller("/dev/cu.wchusbserial110", "mcu-serial", 115200)
        self.Controller.from_mcu_serial.assert_called_once_with(
            "/dev/cu.wchusbserial110", 115200)

    def test_explicit_dm_serial(self) -> None:
        transport_mod.make_controller("/dev/ttyACM0", "dm-serial")
        self.Controller.from_dm_serial.assert_called_once_with(
            "/dev/ttyACM0", 921600)

    def test_unknown_transport_raises(self) -> None:
        with self.assertRaises(ValueError):
            transport_mod.make_controller("can0", "bogus")

    def test_auto_dev_path_robstride_is_mcu_serial(self) -> None:
        transport_mod.make_controller("/dev/cu.usbserial-0001", "auto",
                                      vendor="robstride")
        self.Controller.from_mcu_serial.assert_called_once_with(
            "/dev/cu.usbserial-0001", 921600)

    def test_auto_dev_path_damiao_is_dm_serial(self) -> None:
        transport_mod.make_controller("/dev/ttyACM0", "auto", vendor="damiao")
        self.Controller.from_dm_serial.assert_called_once_with(
            "/dev/ttyACM0", 921600)


def _fake_glob(listing: dict[str, list[str]]):
    def fake(pattern: str) -> list[str]:
        return listing.get(pattern, [])
    return fake


class TestAutoChannelDarwin(unittest.TestCase):
    """channel='auto', transport='auto' on macOS with a fake /dev listing."""

    def setUp(self) -> None:
        for target, kw in [
            (mock.patch.object(transport_mod, "Controller"), "Controller"),
            (mock.patch.object(transport_mod.sys, "platform", "darwin"), None),
        ]:
            started = target.start()
            self.addCleanup(target.stop)
            if kw:
                setattr(self, kw, started)

    def test_pcbusb_wins(self) -> None:
        with mock.patch.object(transport_mod, "can_load_pcbusb",
                               return_value=True):
            transport_mod.make_controller("auto", "auto")
        self.Controller.assert_called_once_with("can0@1000000")

    def test_wch_node_preferred_over_usbserial(self) -> None:
        listing = {
            "/dev/cu.wchusbserial*": ["/dev/cu.wchusbserial110"],
            "/dev/cu.usbserial*": ["/dev/cu.usbserial-0001"],
        }
        with mock.patch.object(transport_mod, "can_load_pcbusb",
                               return_value=False), \
             mock.patch.object(transport_mod.glob, "glob",
                               side_effect=_fake_glob(listing)):
            transport_mod.make_controller("auto", "auto")
        self.Controller.from_mcu_serial.assert_called_once_with(
            "/dev/cu.wchusbserial110", 921600)

    def test_usbserial_fallback(self) -> None:
        listing = {"/dev/cu.usbserial*": ["/dev/cu.usbserial-0001"]}
        with mock.patch.object(transport_mod, "can_load_pcbusb",
                               return_value=False), \
             mock.patch.object(transport_mod.glob, "glob",
                               side_effect=_fake_glob(listing)):
            transport_mod.make_controller("auto", "auto")
        self.Controller.from_mcu_serial.assert_called_once_with(
            "/dev/cu.usbserial-0001", 921600)

    def test_nothing_found_raises_with_hint(self) -> None:
        with mock.patch.object(transport_mod, "can_load_pcbusb",
                               return_value=False), \
             mock.patch.object(transport_mod.glob, "glob",
                               return_value=[]), \
             mock.patch.object(transport_mod, "macos_pcbusb_hint",
                               return_value="HINT-MARKER"):
            with self.assertRaises(RuntimeError) as ctx:
                transport_mod.make_controller("auto", "auto")
        self.assertIn("HINT-MARKER", str(ctx.exception))

    def test_bare_channel_is_pcan_on_darwin(self) -> None:
        transport_mod.make_controller("can0", "auto")
        self.Controller.assert_called_once_with("can0@1000000")


class TestAutoChannelLinux(unittest.TestCase):
    def setUp(self) -> None:
        for p in [
            mock.patch.object(transport_mod, "Controller"),
            mock.patch.object(transport_mod.sys, "platform", "linux"),
        ]:
            started = p.start()
            self.addCleanup(p.stop)
        self.Controller = transport_mod.Controller

    def test_socketcan_can0_preferred(self) -> None:
        with mock.patch.object(transport_mod.os.path, "exists",
                               return_value=True):
            transport_mod.make_controller("auto", "auto")
        self.Controller.assert_called_once_with("can0")

    def test_bare_channel_is_socketcan_on_linux(self) -> None:
        transport_mod.make_controller("can1", "auto")
        self.Controller.assert_called_once_with("can1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
