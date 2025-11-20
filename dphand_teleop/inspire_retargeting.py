import numpy as np
import scipy.optimize as opt

import mujoco

from dphand_utils.math_utils import rpy2mtx, angle_between_signed
from dphand_utils.model_utils import get_actuator_joint_ids, get_all_joint_ids, is_strictly_ascending
from dphand_teleop.retarget import Retarget

from pathlib import Path
# 加载模型
PROJ_DIR = Path(__file__).resolve().parent.parent
XML_PATH = PROJ_DIR / "assets/inspire_hand/left_hand.xml"

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

INSPIRE_TRANS_MATRIX = rpy2mtx(np.pi/2, np.pi/2, 0)
INSPIRE_HAND_JOINT_IDS = get_all_joint_ids(model, model.body("left_hand").id) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# ACTUATOR对应的关节ID必须是升序的
ACTUATOR_IDS = get_actuator_joint_ids(model) # [0, 1, 4, 6, 8, 10]
assert is_strictly_ascending(ACTUATOR_IDS), "OPT_ACTUATOR_IDS is not strictly ascending"

# 利用p2p算法优化的关节ID
P2P_OPT_JOINT_IDS = [model.joint(name).id for name in ["joint_l_thumb_proximal_1", "joint_l_thumb_proximal_2"]]
P2P_OPT_ACTUATOR_IDS = [model.actuator(name).id for name in ["thumb_l_proximal_1_drive", "thumb_l_proximal_2_drive"]]
# 目标关节对应的bodyID
P2P_TARGET_BODY_IDS = [model.body(name).id for name in ["l_thumb_proximal_2", "l_thumb_middle", "l_thumb_distal"]]
P2P_BODY_OFFSET = np.array([[0.02,-0.02,-0.0065],
                            [0.01,-0.005,-0.0065],
                            [0.01,-0.01,-0.0065]])
# visionpro keypoints ID
P2P_TARGET_KEYPOINTS_IDS = [2,3,4]
P2P_TARGET_WEIGHT = np.array([2.0, 1.0, 1.0])

# print(P2P_OPT_JOINT_IDS)

class InspireRetargeting(Retarget):
    def __init__(self):
        self._model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self._data = mujoco.MjData(self._model)
        self.actuator_dim = len(ACTUATOR_IDS)

    def error_function(self, joint_values):
        # calculate forward kinematics
        self._data.qpos[P2P_OPT_JOINT_IDS] = joint_values
        ###################### inspire hand joint constraint ######################
        self._data.qpos[2:4] = self._data.qpos[1]
        ############################################################################
        mujoco.mj_forward(self._model, self._data)
        # get positions for all joints.
        target_pos = self.target_pos[P2P_TARGET_KEYPOINTS_IDS]
        body_pos = self.calculate_joint_pos()
        # error function.
        error = (np.linalg.norm(body_pos - target_pos, axis=1) * P2P_TARGET_WEIGHT).sum() # + 10 * np.linalg.norm(joint_values - self._data.qpos[6:])
        return error

    def calculate_joint_pos(self):
        body_pos = self._data.xpos[P2P_TARGET_BODY_IDS] # (No, 3)
        body_rot = self._data.xmat[P2P_TARGET_BODY_IDS].reshape(len(P2P_TARGET_BODY_IDS), 3, 3) # (No, 3, 3)

        body_keypoints_pos = body_pos + (body_rot @ P2P_BODY_OFFSET[..., np.newaxis]).squeeze(-1) # (No, 3)
        return body_keypoints_pos

    def retarget(self, target_pos):
        self.set_target(target_pos)
        ctrl = np.zeros(self.actuator_dim)
        # index, middle, ring, little
        ctrl[2:] = self.calculate_hand_angle()
        # thumb
        initial_joint_values = self._data.qpos[P2P_OPT_JOINT_IDS].copy()
        result = opt.minimize(
            self.error_function, # error function
            initial_joint_values,   # starting point for iterations
            method='SLSQP',  # algorithms: BFGS, L-BFGS-B, SLSQP
            bounds=model.jnt_range[P2P_OPT_JOINT_IDS],
            options={'maxiter': 20, 'disp': False},  # max iterations
            tol=1e-7  # tolerance for convergence
        )

        if result.success:
            ctrl[P2P_OPT_ACTUATOR_IDS] = result.x
        else:
            ctrl[P2P_OPT_ACTUATOR_IDS] = initial_joint_values
        return ctrl

    def set_target(self, target_pos):
        self.target_pos = self.pre_process_keypoints(target_pos)

    def pre_process_keypoints(self, keypoints):
        """将VisionPro捕捉到的keypoints转换到loacl frame"""
        keypoints = keypoints - keypoints[0]
        keypoints = self.modify_target(keypoints)
        keypoints = (INSPIRE_TRANS_MATRIX @ keypoints.T).T
        keypoints = keypoints - keypoints[0] + self._data.xpos[3]
        return keypoints
    
    def modify_target(self, keypoints):
        """ 手动微调keypoints的位置,缩小人手和DPHand之间的GAP """
        keypoints_raw = keypoints.copy()
        keypoints[[1,11,16]] *= 1.03
        # thumb
        keypoints[2] = (keypoints_raw[2] - keypoints_raw[1]) * 0.9 + keypoints[1]
        keypoints[3] = (keypoints_raw[3] - keypoints_raw[2]) * 0.85 + keypoints[2]
        keypoints[4] = (keypoints_raw[4] - keypoints_raw[3]) * 0.7 + keypoints[3]
        # index
        keypoints[7] = (keypoints_raw[7] - keypoints_raw[6]) * 1.05 + keypoints[6]
        keypoints[8] = (keypoints_raw[8] - keypoints_raw[7]) * 1.1 + keypoints[7]
        keypoints[9] = (keypoints_raw[9] - keypoints_raw[8]) * 0.65 + keypoints[8]
        # middle
        keypoints[12] = (keypoints_raw[12] - keypoints_raw[16]) * 1.05 + keypoints[16]
        keypoints[13] = (keypoints_raw[13] - keypoints_raw[12]) * 1.1 + keypoints[12]
        keypoints[14] = (keypoints_raw[14] - keypoints_raw[13]) * 0.65 + keypoints[13]
        # ring
        keypoints[17] = (keypoints_raw[17] - keypoints_raw[16]) * 1.1 + keypoints[16]
        keypoints[18] = (keypoints_raw[18] - keypoints_raw[17]) * 1.1 + keypoints[17]
        keypoints[19] = (keypoints_raw[19] - keypoints_raw[18]) * 0.65 + keypoints[18]
        # little
        keypoints[22] = (keypoints_raw[22] - keypoints_raw[21]) * 1.1 + keypoints[21]
        keypoints[23] = (keypoints_raw[23] - keypoints_raw[22]) * 1.1 + keypoints[22]
        keypoints[24] = (keypoints_raw[24] - keypoints_raw[23]) * 0.9 + keypoints[23]
        return keypoints
    
    def calculate_hand_angle(self):
        keypoints = self.target_pos
        pre_vec = np.zeros((4, 3))
        next_vec = np.zeros((4, 3))
        angle = np.zeros(4)

        for i in range(4):
            pre_vec[i, :] = keypoints[i*5 + 9, :] - keypoints[i*5 + 6, :]
            next_vec[i, :] = keypoints[i*5 + 6, :] - keypoints[0, :]
        angle = angle_between_signed(pre_vec, next_vec)
        return angle
    