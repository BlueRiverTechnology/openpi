import multiprocessing
import time
import random


class MotorProcess:
    def __init__(self, motor, queue_size=100):
        self.sensor_queue = multiprocessing.Queue(maxsize=queue_size)
        self.command_queue = multiprocessing.Queue()
        self.motor = motor
        self.latest_sensor_reading = None
        self._process = multiprocessing.Process(target=self.run)

    def start(self):
        self._process.start()

    def terminate(self):
        self._process.terminate()

    def run(self):
        try:
            self.motor.connect()
            while True:
                sensor_reading = self.motor.leader_arms["main"].read("Present_Position")
                self.latest_sensor_reading = sensor_reading

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
                    #self.motor.follower_arms["main"].write("Goal_Position", target_position)
                except multiprocessing.queues.Empty:
                    pass
        finally:
            self.motor.write("Torque_Enable", TorqueMode.DISABLED.value)
            self.motor.disconnect()


    def latest_reading(self):
        return self.latest_sensor_reading

    def command(self, target_position):
        self.command_queue.put(target_position)
