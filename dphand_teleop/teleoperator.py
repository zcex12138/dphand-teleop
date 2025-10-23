from dphand_teleop.dphand_retargeting import DPHandRetargeting
from dphand_teleop.inspire_retargeting import InspireRetargeting
from dphand_teleop.visionpro_listener import VisionProListener, DataReplayer
import numpy as np
from dphand_utils.math_utils import rpy2mtx, mtx2quat
from pathlib import Path

PROJ_DIR = Path(__file__).resolve().parent.parent

class VisionProTeleoperator:
    def __init__(self, 
                ip=None, 
                test=True, 
                n_step=5, 
                use_relative_pose=False,
                type="dphand"
            ):
        if type == "dphand":
            self.retargeting = DPHandRetargeting()
        elif type == "inspire":
            self.retargeting = InspireRetargeting()
        else:
            raise ValueError(f"Invalid type: {type}, please choose from 'dphand' or 'inspire'")
        self.counter = 0
        self.last_action = None
        self.n_step = n_step

        self.test = test
        if self.test:
            self.data_replayer = DataReplayer(str(PROJ_DIR / 'data/test_data_v2.npz'), circle_replay=True)
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

    def get_action(self):
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
            ctrl = self.retargeting.retarget(keypoints)
            self.last_action = ctrl
        else:
            ctrl = self.last_action
        self.counter += 1
        # control
        if self.use_relative_pose:
            # 计算相对位置
            arm_pos = left_wrist_pos - self.init_human_wrist_pos
        else:
            arm_pos = left_wrist_pos
            
        arm_rot = mtx2quat(left_wrist_rot @ rpy2mtx(np.pi/2, 0, -np.pi/2))
        
        # 用手臂的俯仰角来代替手腕的转角，在手臂固定的情况下使用
        # target_action[6] = target_action[3] + 1.22 # wrist yaw
        return arm_pos, arm_rot, ctrl