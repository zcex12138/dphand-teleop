#!/usr/bin/env python3
"""
example of inspire hand teleoperator
"""

import numpy as np
import mujoco
import mujoco.viewer
from dphand_teleop.teleoperator import VisionProTeleoperator
import time
from scipy.spatial.transform import Rotation as R

def render_targets(scn, targets, color=(1, 0, 0), size=0.005):
    targets = np.array(targets)
    if targets.ndim == 1:
        targets = targets.reshape(1, -1)  # 单个点转成二维形式
    for target in targets:
        # 设置小球的属性（颜色、大小等）
        rgba = np.array([color[0], color[1], color[2], 1.0])  # 小球颜色（RGBA）
        # 在 body 的 xpos 位置绘制小球
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([size, size, size]),
            pos=target,
            mat=np.eye(3).flatten(),
            rgba=rgba
        )
        scn.ngeom += 1  # 增加渲染对象的计数
        
# 加载模型
model = mujoco.MjModel.from_xml_path("assets/inspire_hand/scene_left.xml")
data = mujoco.MjData(model)
# reset
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# test=True则从录制的文件中读取数据
dphand_teleoperator = VisionProTeleoperator(ip="192.168.3.11", test=True, use_relative_pose=True, type="inspire")

# 启动 viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer._render_every_frame = False  # 禁用默认渲染
    # 使用XML中定义的固定相机（推荐方法）
    front_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = front_cam_id
    # 使用自由相机
    # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # 显示接触点
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
    # 显示关节
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1
    
    cnt = 0
    start_time = time.time()
    
    while viewer.is_running():
        viewer.user_scn.ngeom = 0
        mujoco.mj_step(model, data)

        delta_pos, arm_rot, ctrl = dphand_teleoperator.get_action() # 28 joints
        data.ctrl = ctrl.copy()
        # data.qpos = dphand_teleoperator.retargeting._data.ctrl.copy()
        # print(ctrl)

        # visualize
        keypoints = dphand_teleoperator.retargeting.target_pos
        keypoints = (data.xmat[0].reshape(3,3) @ (keypoints - keypoints[0]).T).T + data.xpos[0]
        render_targets(viewer.user_scn, keypoints[:5], size=0.005)
        # 同步 viewer
        body_pos = dphand_teleoperator.retargeting.calculate_joint_pos()
        render_targets(viewer.user_scn, body_pos, size=0.005, color=(0, 1, 0))
        viewer.sync()
        
        # 控制帧率
        # time.sleep(0.01)
