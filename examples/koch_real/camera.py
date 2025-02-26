import cv2, os
import multiprocessing
import time
import ctypes
import numpy as np
from absl import logging
from absl import app


class CameraCapture(object):

    def __init__(self, device, frame_shape=(240, 320, 3)):
        self.device = device
        self.frame_shape = frame_shape
        shared_array_size = self.frame_shape[0] * self.frame_shape[
            1] * self.frame_shape[2]

        # Create shared memory array
        # 'B' for unsigned char (bytes)
        self.__shared_array = multiprocessing.Array('B', shared_array_size)
        self.__shared_frame_id = multiprocessing.Value(ctypes.c_longlong, -1)

        # Create a lock for synchronization
        self.__lock = multiprocessing.Lock()

        self._process = multiprocessing.Process(target=self.__camera_process)
        self._last_frame_id = 0

    def start(self):
        self._process.start()

    def join(self):
        self._process.join()

    def __camera_process(self):
        """Continuously captures images and puts the *latest* one in shared memory."""
        try:
            logging.info('Going to open')
            cap = cv2.VideoCapture(self.device)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_shape[0])
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_shape[1])
            logging.info('Opening')
            if not cap.isOpened():
                raise IOError("Cannot open webcam")

            shared_array = np.ndarray(
                (self.frame_shape[0], self.frame_shape[1],
                 self.frame_shape[2]),
                dtype=np.uint8,
                buffer=self.__shared_array.get_obj())

            t = time.time()
            os.nice(-20)
            while True:
                ret, frame = cap.read()
                new_time = time.time()
                if not ret:
                    break

                logging.info('Inserting frame %s: dt: %f hz: %f', frame.shape,
                             (new_time - t), 1.0 / (new_time - t))
                t = new_time
                with self.__lock:  # Protect access to shared memory
                    # Flatten the frame and copy it into the shared array
                    flat_frame = frame.flatten()
                    self.__shared_frame_id.value += 1
                    np.copyto(shared_array, frame)

            cap.release()
        except Exception as e:
            print(f"Error in camera process: {e}")

    def latest_frame(self):
        """Retrieves and processes the latest image from shared memory."""
        shared_array = np.ndarray(
            (self.frame_shape[0], self.frame_shape[1], self.frame_shape[2]),
            dtype=np.uint8,
            buffer=self.__shared_array.get_obj())
        logging.info('Grabbing frame')
        with self.__lock:  # Protect access to shared memory
            frame = shared_array.copy()  # Copy the array.
            frame_id = self.__shared_frame_id.value

        if self._last_frame_id != frame_id:
            self._last_frame_id = frame_id
            return frame
        else:
            return None


def main(argv):
    logging.set_verbosity(logging.DEBUG)

    camera = CameraCapture(0, (240, 320, 3))
    camera.start()

    while True:
        frame = camera.latest_frame()
        if frame is not None:
            logging.info('Processing frame %s', frame.shape)
        time.sleep(0.5)

    camera.join()

    print("All processes finished.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(main)
