import numpy as np
import scipy.optimize as opt

import mujoco

from dphand_utils.math_utils import rpy2mtx, angle_between_signed
from dphand_utils.model_utils import get_actuator_joint_ids, get_all_joint_ids, is_strictly_ascending
from dphand_teleop.retarget import Retarget

from pathlib import Path
# 加载模型
PROJ_DIR = Path(__file__).resolve().parent.parent
XML_PATH = PROJ_DIR / "assets/dphand/DPHand_free.xml"

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

DPHAND_TRANS_MATRIX = rpy2mtx(-np.pi/2, -np.pi/2, 0)
DPHAND_JOINT_IDS = get_all_joint_ids(model, model.body("Forearm").id)

# ACTUATOR对应的关节ID， 检测OPT_ACTUATOR_IDS是否是升序的
ACTUATOR_IDS = get_actuator_joint_ids(model)
assert is_strictly_ascending(ACTUATOR_IDS), "OPT_ACTUATOR_IDS is not strictly ascending"

# 优化关节ID
P2P_OPT_JOINT_IDS = get_all_joint_ids(model, model.body("Thumb-MCP").id)
P2P_OPT_ACTUATOR_IDS = [model.actuator(name).id - 6 for name in ["Thumb-MCP", "Thumb-PP", "Thumb-PM", "Thumb-PD"]]
# 目标 关节 和 body ID
P2P_TARGET_JOINT_IDS = [model.joint(name).id for name in ["Thumb-PP_Thumb-PIP-Flexion", "Thumb-PM_Thumb-DIP-Flexion", "Thumb-PM_Thumb-DIP-Flexion"]]
P2P_TARGET_BODY_IDS = [model.joint(joint_id).bodyid.item() for joint_id in P2P_TARGET_JOINT_IDS]

# keypoints ID
P2P_TARGET_KEYPOINTS_IDS = [2,3,4]
P2P_TARGET_WEIGHT = np.array([5.0, 1.0, 1.0])

class DPHandRetargeting(Retarget):
    def __init__(self):
        self._model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self._data = mujoco.MjData(self._model)
        self.actuator_dim = len(ACTUATOR_IDS) - 6
        self.finger_tip_index = [2] # 拇指
        self.rot_transform = rpy2mtx(np.pi/2, 0, -np.pi/2)
                                        
    def error_function(self, joint_values):
        # calculate forward kinematics
        self._data.qpos[P2P_OPT_JOINT_IDS] = joint_values
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

        offset = self._model.jnt_pos[P2P_TARGET_JOINT_IDS] - np.array([0.008,0,0]) # (No, 3)
        offset[self.finger_tip_index, :] += np.array([0,0,-0.008])

        body_keypoints_pos = body_pos + (body_rot @ offset[..., np.newaxis]).squeeze(-1) # (No, 3)
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
    
    def pre_process_keypoints(self, keypoints):
        """将VisionPro捕捉到的keypoints转换到loacl frame"""
        keypoints = keypoints - keypoints[0]
        keypoints = self.modify_target(keypoints)
        keypoints = (DPHAND_TRANS_MATRIX @ keypoints.T).T
        keypoints = keypoints - keypoints[0] + self._data.xpos[3]
        return keypoints

    def calculate_hand_angle(self):
        keypoints = self.target_pos
        pre_vec = np.zeros((5, 4, 3))
        next_vec = np.zeros((5, 4, 3))
        # 拇指
        pre_vec[0, 1, :] = keypoints[1, :] - keypoints[5, :]
        next_vec[0, 1, :] = keypoints[2, :] - keypoints[1, :]

        pre_vec[0, [0,2,3], :] = keypoints[1:4, :] - keypoints[0:3, :]
        next_vec[0, [0,2,3], :] = keypoints[2:5, :] - keypoints[1:4, :]

        # 食指-小拇指
        for i in range(1, 5):
            # 横向关节
            if i == 1:
                pre_vec[i, 1, :] = keypoints[11, :] - keypoints[6, :]
            elif i == 2:
                pre_vec[i, 1, :] = keypoints[16, :] - keypoints[6, :]
            elif i == 3:
                pre_vec[i, 1, :] = keypoints[21, :] - keypoints[11, :]
            else:
                pre_vec[i, 1, :] = keypoints[21, :] - keypoints[16, :]
            j = i * 5
            next_vec[i, 1, :] = keypoints[j+2, :] - keypoints[j+1, :]
            # 纵向关节
            pre_vec[i, [0,2,3], :] = keypoints[j+1:j+4, :] - keypoints[j:j+3, :]
            next_vec[i, [0,2,3], :] = keypoints[j+2:j+5, :] - keypoints[j+1:j+4, :]
        pre_vec = pre_vec.reshape(-1, 3)
        next_vec = next_vec.reshape(-1, 3)
        angle = angle_between_signed(pre_vec, next_vec).reshape(5, 4)
        angle[:, 1] = np.pi / 2 - np.abs(angle[:, 1])
        # fine tune
        angle[1, 1] += 0.22
        angle[4, 1] -= 0.11
        angle[:, 1] *= 1.2 # 侧向
        return angle.ravel()