from threading import Thread, Lock
from avp_stream import VisionProStreamer
from dphand_utils.math_utils import angle_between
import time

import numpy as np

class VisionProListener():
    def __init__(self, ip, record=False):
        self.streamer = VisionProStreamer(ip, record)
        self.lock = Lock()
        self.hand_angles = np.zeros((5, 4))

        if record == True:
            self.start_time = time.time()

    def get_data(self):
        with self.lock:
            return self.streamer.get_latest()  # 使用get_latest方法确保线程安全
    
    def get_left_keypoints(self):
        with self.lock:
            latest = self.streamer.get_latest()
            if latest is not None:
                return latest['left_fingers'][:, :3, 3]
            return None
        
    def get_left_wrist(self):
        with self.lock:
            latest = self.streamer.get_latest()
            if latest is not None:
                left_wrist = latest['left_wrist']
                return left_wrist[0, :3, 3], left_wrist[0, :3, :3]
            return None, None

    def save_recording_as_npz(self, filename):
        if self.streamer.record:
            total_run_time = time.time() - self.start_time
            np.savez_compressed(filename, data=self.streamer.recording, total_time=total_run_time, allow_pickle=True)
            print(f"Recording saved as {filename}.npz")
        else:
            print("Recording is not enabled.")
        

class DataReplayer():
    def __init__(self, filename, circle_replay=True):
        self.data = np.load(filename, allow_pickle=True)['data'] # dtype = object
        self.index = 0
        self.circle_replay = circle_replay
    
    def get_frame(self):
        if self.data.size == 1:
            return self.data.item()
        if self.index < self.data.size:
            frame = self.data[self.index]
            self.index += 1
            return frame
        elif self.circle_replay:
            self.index = 0
            return self.data[self.index]
        else:
            return self.data[-1]


if __name__ == '__main__': 
    # listener = VisionProListener(ip='192.168.3.27', record=True)
    # import time
    # start_time = time.time()
    # while time.time() - start_time < 10.0:
    #     print(listener.get_left_keypoints())
    
    # listener.save_recording_as_npz('../data/test_recording')

    # test DataReplayer
    replayer = DataReplayer('../data/test_recording.npz')
    while True:
        data = replayer.get_frame()
        print(data['left_fingers'][:, :3, 3])
        time.sleep(0.001)