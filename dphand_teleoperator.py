from dphand_retargeting import *
from visionpro_listener import VisionProListener
import numpy as np
from utils.math_utils import mtx2rpy

class DPhandTeleoperator:
    def __init__(self, model, data, ip=None, test=False):
        self.retargeting = DPHandRetargeting(model=model, data=data)
        self.listener = VisionProListener(ip=ip)
        self.test = test
        if self.test:
            self.data = np.load("data/hand_keypoints.npz", allow_pickle=True)['keypoints'].item()
        else:
            self.listener.start_streaming()
    
    def get_target_action(self):
        target_action = np.zeros_like(self.retargeting.data.ctrl)
        if self.test:
            # get the data from the file
            keypoints = self.data['left_fingers'][:, :3, 3].copy()
            left_wrist_rot, left_wrist_pos = self.data['left_wrist'][0, :3, :3].copy(), self.data['left_wrist'][0, :3, 3].copy()
        else:
            # get the data from the listener
            keypoints = self.listener.get_left_keypoints()
            left_wrist_pos, left_wrist_rot = self.listener.get_left_wrist()
        # retargeting
        keypoints = self.retargeting.set_target(keypoints)
        qpos, ctrl = self.retargeting.retarget()
        # control
        target_action[0:3] = left_wrist_pos
        target_action[3:6] = mtx2rpy(left_wrist_rot @ DPHAND_TO_OPERATOR)
        target_action[6:] = ctrl
        return target_action