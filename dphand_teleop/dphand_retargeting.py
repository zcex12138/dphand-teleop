import numpy as np
import scipy.optimize as opt

import mujoco

from utils.math_utils import rpy2mtx, mtx2rpy, mtx2quat, quat2rpy, rpy2quat
from utils.model_utils import get_actuator_joint_ids, get_all_joint_ids

# 加载模型
model = mujoco.MjModel.from_xml_path('./assets/DPhand/DPHand_free.xml')
data = mujoco.MjData(model)


DPHAND_TO_OPERATOR = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]) # rpy2mtx(0, np.pi/2, np.pi)

OPERATOR_TO_DPHAND_KEYPOINTS = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]) # rpy2mtx(np.pi, -np.pi/2, 0)

TARGET_JOINT_NAMES = ["Thumb-PP_Thumb-PIP-Flexion", "Thumb-PM_Thumb-DIP-Flexion", "Thumb-PM_Thumb-DIP-Flexion",
              "Index-PP_Index-PIP-Flexion", "Index-PM_Index-DIP-Flexion", "Index-PM_Index-DIP-Flexion",
              "Middle-PP_Middle-PIP-Flexion", "Middle-PM_Middle-DIP-Flexion", "Middle-PM_Middle-DIP-Flexion",
              "Ring-PP_Ring-PIP-Flexion", "Ring-PM_Ring-DIP-Flexion", "Ring-PM_Ring-DIP-Flexion",
              "Little-PP_Little-PIP-Flexion", "Little-PM_Little-DIP-Flexion", "Little-PM_Little-DIP-Flexion"]

TARGET_JOINT_IDS = [model.joint(name).id for name in TARGET_JOINT_NAMES]
# [10, 11, 11, 14, 15, 15, 18, 19, 19, 22, 23, 23, 26, 27, 27]
TARGET_BODY_IDS = [model.joint(joint_id).bodyid.item() for joint_id in TARGET_JOINT_IDS]

FINGER_BASE_NAMES = ["Palm_Thumb-MCP-Flexion", "Palm_Index-MCP-Flexion",
                            "Palm_Middle-MCP-Flexion", "Palm_Ring-MCP-Flexion",
                            "Palm_Little-MCP-Flexion"]

ACTUATOR_NAMES = ["Thumb-PD", "Thumb-PM",
                  "Index-PD", "Index-PM",
                  "Middle-PD", "Middle-PM",
                  "Ring-PD", "Ring-PM",
                  "Little-PD", "Little-PM"]

FINGER_TYPES = ["Thumb", "Index", "Middle", "Ring", "Little"]
BODY_TYPES = ["MCP", "PP", "PM", "PD"]


class DPHandRetargeting:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        self.joint_ids = get_all_joint_ids(model, model.body("base").id)
        self.finger_base_names = FINGER_BASE_NAMES
        self.actuator_names = ACTUATOR_NAMES

        self.target_joint_names = TARGET_JOINT_NAMES
        self.target_joint_ids = TARGET_JOINT_IDS
        self.target_keypoints_indices = [2,3,4,7,8,9,12,13,14,17,18,19,22,23,24]
        self.actuator_joint_ids = get_actuator_joint_ids(model)

        self.weight = np.ones(len(self.target_joint_ids))
        self.weight[[2, 5, 8, 11, 14]] = 1.7
                                        
    def error_function(self, joint_values):
        # calculate forward kinematics
        self.data.qpos[self.joint_ids[6:]] = joint_values
        mujoco.mj_forward(self.model, self.data)
        # get positions for all joints.
        joint_target_positions = self.target_positions[self.target_keypoints_indices] # 15*3
    
        joint_positions = self.calculate_joint_pos_v2()
        # error function.
        error = (np.linalg.norm(joint_positions - joint_target_positions, axis=1) * self.weight).sum() # + 10 * np.linalg.norm(joint_values - self.data.qpos[6:])
        return error

    def retarget(self):
        # initial joint values
        initial_joint_values = self.data.qpos[self.joint_ids[6:]].copy()

        result = opt.minimize(
            self.error_function, # error function
            initial_joint_values,   # starting point for iterations
            method='SLSQP',  # algorithms: BFGS, L-BFGS-B, SLSQP
            bounds=model.jnt_range[self.joint_ids[6:]],
            options={'maxiter': 20, 'disp': False},  # max iterations
            tol=1e-4  # tolerance for convergence
        )

        if result.success:
            qpos = result.x
            ctrl = qpos[self.actuator_joint_ids[6:]]
        else:
            qpos = initial_joint_values
            ctrl = initial_joint_values[self.actuator_joint_ids[6:]]
        return qpos, ctrl

    def set_target(self, target_positions):
        self.target_positions = self.pre_process_keypoints(target_positions)
        return self.target_positions

    def calculate_joint_pos(self, joint_name):
        is_tip = False
        if "TIP" in joint_name:
            is_tip = True
            joint_name = joint_name[:-4]
        joint_id = self.model.joint(joint_name).id
        body_id = self.model.joint(joint_name).bodyid
        # Get the body's global position and orientation
        body_pos = self.data.xpos[body_id]  # Global position of the body
        body_rot = self.data.xmat[body_id].reshape(3, 3)  # Body's rotation matrix (3x3)
        # Get the joint's local position (offset) relative to the body
        joint_offset = self.model.jnt_pos[joint_id] - np.array([0.008,0,0]) # np.array([0.0075,0,0]) is an offset
        if is_tip:
            joint_offset += np.array([0,0,-0.015])
        # Compute the global position of the joint
        joint_offset_pos = body_pos + body_rot @ joint_offset
        return joint_offset_pos, is_tip

    def calculate_joint_pos_v2(self): # 全部使用向量化的计算
        body_pos = self.data.xpos[TARGET_BODY_IDS] # (15, 3)
        body_rot = self.data.xmat[TARGET_BODY_IDS].reshape(len(TARGET_BODY_IDS), 3, 3) # (15, 3, 3)

        joint_offset = self.model.jnt_pos[self.target_joint_ids] - np.array([0.008,0,0]) # (15, 3)
        finger_tip_index = [2, 5, 8, 11, 14] 
        joint_offset[finger_tip_index, :] += np.array([0,0,-0.015])

        joint_offset_pos = body_pos + (body_rot @ joint_offset[..., np.newaxis]).squeeze(-1) # (15, 3)
        return joint_offset_pos

    def modify_target(self, keypoints):
        """ 手动微调keypoints的位置,缩小人手和DPHand之间的GAP """
        keypoints_raw = keypoints.copy()
        keypoints[[1,11,16]] *= 1.03
        # thumb
        keypoints[2] = (keypoints_raw[2] - keypoints_raw[1]) * 0.9 + keypoints[1]
        keypoints[3] = (keypoints_raw[3] - keypoints_raw[2]) * 0.8 + keypoints[2]
        keypoints[4] = (keypoints_raw[4] - keypoints_raw[3]) * 0.55 + keypoints[3]
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
        """将VisionPro捕捉到的keypoints从手腕坐标系转换到DPHand环境的世界坐标系"""
        keypoints = keypoints - keypoints[0]
        keypoints = self.modify_target(keypoints)
        hand_rot = rpy2mtx(*self.data.qpos[3:6])
        keypoints = (hand_rot @ DPHAND_TO_OPERATOR @ keypoints.T).T
        keypoints = keypoints - keypoints[0] + self.data.xpos[3]
        return keypoints


