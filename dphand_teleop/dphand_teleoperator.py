from dphand_teleop.dphand_retargeting import *
from dphand_teleop.visionpro_listener import VisionProListener
import numpy as np
from dphand_utils.math_utils import mtx2rpy

class DPhandTeleoperator:
    def __init__(self, model, data, ip=None, test=True):
        self.retargeting = DPHandRetargeting(model=model, data=data)
        self.test = test
        if self.test:
            self.data = np.load("data/hand_keypoints.npz", allow_pickle=True)['keypoints'].item()
        else:
            self.listener = VisionProListener(ip=ip)

        self.last_action = None
    
    def get_target_action_p2p(self):
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

        self.last_action = target_action
        return target_action
    
    def get_target_action_j2j(self):
        target_action = np.zeros_like(self.retargeting.data.ctrl)
        if self.test:
            # get the data from the file
            keypoints = self.data['left_fingers'][:, :3, 3].copy()
            roll = self.data['left_wrist_roll']
            left_wrist_rot, left_wrist_pos = self.data['left_wrist'][0, :3, :3].copy(), self.data['left_wrist'][0, :3, 3].copy()
        else:
            # get the data from the listener
            keypoints = self.listener.get_left_keypoints()
            roll = self.listener.get_data()['left_wrist_roll']
            left_wrist_pos, left_wrist_rot = self.listener.get_left_wrist()
        # retargeting
        keypoints = self.retargeting.set_target(keypoints)
        angle = self.retargeting.calculate_hand_angle()
        # control
        target_action[0:3] = left_wrist_pos
        target_action[3:6] = mtx2rpy(left_wrist_rot @ DPHAND_TO_OPERATOR)
        target_action[6] = roll - np.pi/2
        target_action[8:] = angle

        self.last_action = target_action
        return target_action