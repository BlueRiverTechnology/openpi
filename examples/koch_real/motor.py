import multiprocessing
import logging
import time
import random
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.robots.configs import RobotConfig
from lerobot.common.robot_devices.robots.configs import ManipulatorRobotConfig
from lerobot.common.robot_devices.robots.configs import MotorsBusConfig
from lerobot.common.robot_devices.robots.configs import CameraConfig
from lerobot.common.robot_devices.robots.configs import DynamixelMotorsBusConfig
from dataclasses import dataclass, field
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

class MotorProcess:
    def __init__(self, motor, queue_size=100):
        self.sensor_queue = multiprocessing.Queue(maxsize=queue_size)
        self.command_queue = multiprocessing.Queue()
        self.motor = motor

        self._process = multiprocessing.Process(target=self.run)
        self._last_reading = None

        self.start()

    def start(self):
        logging.info('Starting')
        self._process.start()

    def terminate(self):
        self._process.terminate()

    def run(self):
        logging.info('Starting run')
        self.motor = KochRobotConfig()
        self.motor.leader_arms = {
            "main":
            DynamixelMotorsBusConfig(
                port=
                "/dev/serial/by-id/usb-ROBOTIS_OpenRB-150_EFB274B350304A46462E3120FF0B0B30-if00",
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
        self.motor.follower_arms = {
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
        self.motor.cameras = {}

        robot = ManipulatorRobot(self.motor)
        robot.connect()
        try:
            while True:
                sensor_reading = robot.follower_arms["main"].read("Present_Position")
                #logging.info('Got sensor reading of %s', sensor_reading)

                try:
                    self.sensor_queue.put_nowait(sensor_reading)
                except multiprocessing.queues.Full:
                    try:
                        self.sensor_queue.get_nowait()
                    except multiprocessing.queues.Empty:
                        pass
                    self.sensor_queue.put_nowait(sensor_reading)

                try:
                    target_position = self.command_queue.get_nowait()
                    logging.info('Writing goal of %s', target_position)
                    robot.follower_arms["main"].write("Goal_Position", target_position)
                except multiprocessing.queues.Empty:
                    pass
        finally:
            robot.follower_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
            robot.leader_arms["main"].write("Torque_Enable", TorqueMode.DISABLED.value)
            robot.disconnect()


    def latest_reading(self):
        while True:
            try:
                reading = self.sensor_queue.get_nowait()
            except multiprocessing.queues.Empty:
                #logging.info('Reading of %s', self._last_reading)
                return self._last_reading

            self._last_reading = reading

    def command(self, target_position):
        self.command_queue.put(target_position)
