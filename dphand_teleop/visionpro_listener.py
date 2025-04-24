from threading import Thread, Lock
from avp_stream import VisionProStreamer
from dphand_utils.math_utils import angle_between

import numpy as np

class VisionProListener():
    def __init__(self, ip, record=False):
        self.streamer = VisionProStreamer(ip, record)
        self.lock = Lock()
        self.hand_angles = np.zeros((5, 4))

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
    
    def get_hand_angle(self):
        keypoints = self.get_left_keypoints()
        pre_vec = np.zeros((5, 4, 3))
        next_vec = np.zeros((5, 4, 3))
        # 拇指
        pre_vec[0, 0, :] = keypoints[1, :] - keypoints[5, :]
        next_vec[0, 0, :] = keypoints[2, :] - keypoints[1, :]

        pre_vec[0, 1:, :] = keypoints[1:4, :] - keypoints[0:3, :]
        next_vec[0, 1:, :] = keypoints[2:5, :] - keypoints[1:4, :]

        # 食指-小拇指
        for i in range(1, 5):
            j = i * 5
            # 横向关节
            pre_vec[i, 0, :] = keypoints[21, :] - keypoints[6, :] # 小拇指指根-食指指根
            next_vec[i, 0, :] = keypoints[j+2, :] - keypoints[j+1, :] # 指根-第一个关节
            # 纵向关节
            pre_vec[i, 1:, :] = keypoints[j+1:j+4, :] - keypoints[j:j+3, :]
            next_vec[i, 1:, :] = keypoints[j+2:j+5, :] - keypoints[j+1:j+4, :]


            pre_vec = keypoints[j+2:j+5, :] - keypoints[j+1:j+4, :]
            next_vec = keypoints[j+1:j+4, :] - keypoints[j:j+3, :]
        angle = angle_between(pre_vec, next_vec)
        
        return angle


        
    
if __name__ == '__main__': 
    listener = VisionProListener(ip='192.168.3.45')
    while True:
        data = listener.get_data()
        print(data)