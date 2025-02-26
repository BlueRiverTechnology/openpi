# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
import collections
import time
from typing import Optional, List
import dm_env
import numpy as np

import constants
from camera import CameraCapture
from motor import MotorProcess
#import robot_utils

from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.motors.configs import ManipulatorRobotConfig
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus

# This is the reset position that is used by the standard Aloha runtime.
DEFAULT_RESET_POSITION = [0, -0.96, 1.16, 0, -0.3, 0]

@RobotConfig.register_subclass("koch")
@dataclass
class KochRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/koch"
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": DynamixelMotorsBusConfig(
                port="/dev/serial/by-id/usb-ROBOTIS_OpenRB-150_EFB274B350304A46462E3120FF0B0B30-if00",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl330-m077"],
                    "shoulder_lift": [2, "xl330-m077"],
                    "elbow_flex": [3, "xl330-m077"],
                    "wrist_flex": [4, "xl330-m077"],
                    "wrist_roll": [5, "xl330-m077"],
                    "gripper": [6, "xl330-m077"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": DynamixelMotorsBusConfig(
                port="/dev/serial/by-id/usb-ROBOTIS_OpenRB-150_2E54F76650304A46462E3120FF083328-if00",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "xl430-w250"],
                    "shoulder_lift": [2, "xl430-w250"],
                    "elbow_flex": [3, "xl330-m288"],
                    "wrist_flex": [4, "xl330-m288"],
                    "wrist_roll": [5, "xl330-m288"],
                    "gripper": [6, "xl330-m288"],
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {})

    # ~ Koch specific settings ~
    # Sets the leader arm in torque mode with the gripper motor set to this angle. This makes it possible
    # to squeeze the gripper and have it spring back to an open position on its own.
    gripper_open_degree: float = 35.156

    mock: bool = False


class RealEnv:
    """
    Environment for real robot bi-manual manipulation
    Action space:      [arm_qpos (6)]             # absolute joint position

    Observation space: {"qpos": Concat[ arm_qpos (6)]          # absolute joint position
                        "qvel": Concat[ left_arm_qvel (6)]         # absolute joint velocity (rad)
                        "images": {"cam_high": (480x640x3),        # h, w, c, dtype='uint8'
                                   "cam_low": (480x640x3),         # h, w, c, dtype='uint8'
                                   "cam_left_wrist": (480x640x3),  # h, w, c, dtype='uint8'
                                   "cam_right_wrist": (480x640x3)} # h, w, c, dtype='uint8'
    """

    def __init__(self):
        self.lerobot_config = KochRobotConfig()
        """self.leader_config = DynamixelMotorsBusConfig(
            port="/dev/serial/by-id/usb-ROBOTIS_OpenRB-150_EFB274B350304A46462E3120FF0B0B30-if00",
            motors={
                # name: (index, model)
                "shoulder_pan": (1, "xl330-m077"),
                "shoulder_lift": (2, "xl330-m077"),
                "elbow_flex": (3, "xl330-m077"),
                "wrist_flex": (4, "xl330-m077"),
                "wrist_roll": (5, "xl330-m077"),
                "gripper": (6, "xl330-m077"),
            },
        )

        self.follower_config = DynamixelMotorsBusConfig(
            port="/dev/serial/by-id/usb-ROBOTIS_OpenRB-150_2E54F76650304A46462E3120FF083328-if00",
            motors={
                # name: (index, model)
                "shoulder_pan": (1, "xl430-w250"),
                "shoulder_lift": (2, "xl430-w250"),
                "elbow_flex": (3, "xl330-m288"),
                "wrist_flex": (4, "xl330-m288"),
                "wrist_roll": (5, "xl330-m288"),
                "gripper": (6, "xl330-m288"),
            },
        )
        """

        #self.leader_arm = DynamixelMotorsBus(self.leader_config)

        self.follower_arm = MotorProcess(ManipulatorRobot(self.lerobot_config))
        self.cameras = {
            "front": CameraCapture(0, frame_shape=(480, 640, 3)),
            "low": CameraCapture(2, frame_shape=(480, 640, 3)),
            "back_near_tractor": CameraCapture(6, frame_shape=(480, 640, 3)),
        }

        for name, cam in self.cameras.items():
            cam.start()

    def get_qpos(self):
        left_qpos_raw = self.follower_arm.latest_reading()
        logging.info('qpos dtype: %s', type(left_pos_raw))
        return left_qpos_raw.astype(float)

    def get_images(self):
        images = collections.OrderedDict()
        for name, cam in self.cameras.items():
            images[name] = cam.latest_frame()

        return images

    def get_observation(self):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos()
        obs["images"] = self.get_images()
        return obs

    def reset(self, *, fake=False):
        return dm_env.TimeStep(step_type=dm_env.StepType.FIRST,
                               reward=0,
                               discount=None,
                               observation=self.get_observation())

    def step(self, action):
        self.follower_arm.command(action)
        time.sleep(constants.DT)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=0, discount=None, observation=self.get_observation()
        )


def make_real_env():
    return RealEnv()
