from threading import Thread, Lock
from avp_stream import VisionProStreamer

import numpy as np

class VisionProListener(VisionProStreamer):
    def __init__(self, ip, record=True):
        super().__init__(ip, record)
        self.leap_lock = Lock()

    def get_data(self):
        with self.leap_lock:
            # return self.latest['left_fingers'][:, :3, 3]
            return self.latest
    
    def get_left_keypoints(self):
        with self.leap_lock:
            return self.latest['left_fingers'][:, :3, 3]
        
    def get_left_wrist(self):
        with self.leap_lock:
            return self.latest['left_wrist'][0, :3, 3], self.latest['left_wrist'][0, :3, :3]