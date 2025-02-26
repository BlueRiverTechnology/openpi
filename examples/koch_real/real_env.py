# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
import collections
import time
from typing import Optional, List
import dm_env
import numpy as np
import logging

import constants
from camera import CameraCapture
from motor import MotorProcess
#import robot_utils

from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig
#from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
#from lerobot.common.robot_devices.robots.configs import ManipulatorRobotConfig
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus

# This is the reset position that is used by the standard Aloha runtime.
DEFAULT_RESET_POSITION = [0, -0.96, 1.16, 0, -0.3, 0]

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
        self.follower_arm = MotorProcess(None) #self.lerobot_config)
        self.cameras = {
            "front": CameraCapture(0, frame_shape=(480, 640, 3)),
            "low": CameraCapture(2, frame_shape=(480, 640, 3)),
            "back_near_tractor": CameraCapture(6, frame_shape=(480, 640, 3)),
        }

        for name, cam in self.cameras.items():
            cam.start()

    def get_qpos(self):
        while True:
            left_qpos_raw = self.follower_arm.latest_reading()

            if left_qpos_raw is None:
                logging.info('No position yet, trying again')
                time.sleep(0.1)
                continue

            return left_qpos_raw.astype(float)

    def get_images(self):
        #logging.info("Getting images")
        images = collections.OrderedDict()
        for name, cam in self.cameras.items():
            images[name] = cam.latest_frame()

        #logging.info("Images are: %s", images)
        return images

    def get_observation(self):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos()
        obs["images"] = self.get_images()
        #logging.info('GetObs: %s', obs)
        return obs

    def reset(self, *, fake=False):
        observation = self.get_observation()
        return dm_env.TimeStep(step_type=dm_env.StepType.FIRST,
                               reward=0,
                               discount=None,
                               observation=observation)

    def step(self, action):
        self.follower_arm.command(action)
        time.sleep(constants.DT)
        #logging.info('Stepping')
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=0, discount=None, observation=self.get_observation()
        )


def make_real_env():
    return RealEnv()
