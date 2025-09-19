# from dphand_teleop.dphand_retargeting import *
from dphand_teleop.dphand_retargeting_j2j import *
from dphand_teleop.visionpro_listener import VisionProListener, DataReplayer
import numpy as np
from dphand_utils.math_utils import mtx2rpy
from pathlib import Path
from dphand_utils.filter_utils import MeanFilter

PROJ_DIR = Path(__file__).resolve().parent.parent

class DPhandTeleoperator:
    def __init__(self, ip=None, test=True, n_step=5, use_relative_pose=False):
        self.retargeting = DPHandRetargeting()
        self.counter = 0
        self.filter = MeanFilter(window_size=5, dimensions=len(self.retargeting._data.ctrl))
        self.last_action = None
        self.n_step = n_step

        self.test = test
        if self.test:
            self.data_replayer = DataReplayer(str(PROJ_DIR / 'data/test_recording_new.npz'), circle_replay=True)
            self.pasue = self.data_replayer.pause
        else:
            self.listener = VisionProListener(ip=ip)

        self.use_relative_pose = use_relative_pose
        if self.use_relative_pose:
            self.init_wrist_pos = self.retargeting._data.ctrl[:3].copy()
            if self.test:
                data = self.data_replayer.get_frame()
                self.init_human_wrist_pos = data['left_wrist'][0, :3, 3]
            else:
                self.init_human_wrist_pos = self.listener.get_left_wrist()[0]

    def get_target_action_p2p(self):
        target_action = np.zeros_like(self.retargeting._data.ctrl)
        if self.test:
            # get the data from the buffer
            data = self.data_replayer.get_frame()
            keypoints = data['left_fingers'][:, :3, 3].copy()
            left_wrist_rot, left_wrist_pos = data['left_wrist'][0, :3, :3].copy(), data['left_wrist'][0, :3, 3].copy()
        else:
            # get the data from the listener
            keypoints = self.listener.get_left_keypoints()
            left_wrist_pos, left_wrist_rot = self.listener.get_left_wrist()
        # retargeting
        if self.counter % self.n_step == 0:
            keypoints = self.retargeting.set_target(keypoints)
            qpos, ctrl = self.retargeting.retarget()
            self.last_action = ctrl
        else:
            ctrl = self.last_action
        # control
        if self.use_relative_pose:
            # 计算相对位置
            relative_pos = left_wrist_pos - self.init_wrist_pos
            target_action[0:3] = relative_pos
        else:
            target_action[0:3] = left_wrist_pos
        
        # 将Vision Pro的旋转转换到DPHand的坐标系
        target_action[3:6] = mtx2rpy(left_wrist_rot @ DPHAND_TO_OPERATOR)
        target_action[6] = target_action[3] + 1.22
        target_action[6:] = ctrl
        self.counter += 1
        return target_action

    def get_target_action_j2j(self):
        target_action = np.zeros_like(self.retargeting._data.ctrl)
        if self.test:
            # get the data from the file
            data = self.data_replayer.get_frame()
            keypoints = data['left_fingers'][:, :3, 3].copy()
            roll = data['left_wrist_roll']
            left_wrist_rot, left_wrist_pos = data['left_wrist'][0, :3, :3].copy(), data['left_wrist'][0, :3, 3].copy()
        else:
            # get the data from the listener
            keypoints = self.listener.get_left_keypoints()
            roll = self.listener.get_data()['left_wrist_roll']
            left_wrist_pos, left_wrist_rot = self.listener.get_left_wrist()
        # retargeting
        if self.counter % self.n_step == 0:
            keypoints = self.retargeting.set_target(keypoints)
            qpos, ctrl = self.retargeting.retarget()
            self.last_action = ctrl
        else:
            ctrl = self.last_action
        self.counter += 1
        # control
        if self.use_relative_pose:
            # 计算相对位置
            target_action[0:3] = left_wrist_pos - self.init_human_wrist_pos
        else:
            target_action[0:3] = left_wrist_pos
            
        target_action[3:6] = mtx2rpy(left_wrist_rot @ rpy2mtx(np.pi/2, 0, -np.pi/2))
        # target_action[3:6] = mtx2rpy(left_wrist_rot)
        
        # 用手臂的俯仰角来代替手腕的转角，在手臂固定的情况下使用
        # target_action[6] = target_action[3] + 1.22 # wrist yaw
        target_action[8:] = ctrl
        # return self.filter.update(target_action)
        return target_action