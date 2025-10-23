#!/usr/bin/env python3
"""
example of dphand teleoperator with franka panda
"""

import numpy as np
import mujoco
import mujoco.viewer
from dphand_teleop.teleoperator import VisionProTeleoperator
import time
from scipy.spatial.transform import Rotation as R
from franka_controller.opspace import opspace

def render_targets(scn, targets, color=(1, 0, 0), size=0.005):
    """渲染目标关键点"""
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

def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion from scipy format (x,y,z,w) to mujoco format (w,x,y,z)"""
    return np.roll(quat_xyzw, 1)

def main():
    # 加载模型
    model = mujoco.MjModel.from_xml_path("assets/panda_pick_and_place.xml")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    
    # 初始化遥操作器 - test=True则从录制的文件中读取数据
    dphand_teleoperator = VisionProTeleoperator(ip="192.168.3.11", test=True, use_relative_pose=True)
    
    # 获取关节和site ID
    panda_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") 
                      for i in range(1, 8)]
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")

    hand_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Forearm")
    
    # 获取mocap body
    franka_mocap_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pinch")]
    target_mocap_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "teleop")]
    
    # 获取forearm
    # 控制参数
    pos_gains = 800.0 * np.array([1.0, 1.0, 1.0])
    ori_gains = 800.0 * np.array([1.0, 1.0, 1.0])

    # 启动viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer._render_every_frame = False  # 禁用默认渲染
        # 自由相机
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.opt.sitegroup[3] = 1 # render site
        cnt = 0
        start_time = time.time()
        
        init_pos = data.site_xpos[site_id].copy()
        # 主循环
        while viewer.is_running():
            viewer.user_scn.ngeom = 0
            
            # 获取遥操作控制指令
            delta_pos, arm_rot, ctrl = dphand_teleoperator.get_action()  # 28 joints

            target_pos = init_pos + delta_pos * 2.0
            target_rot = (1,0,0,0)
            target_rot = arm_rot
            # target_rot = rpy2quat(np.pi/2, 0, np.pi/2)
            # target_rot = quat_mul(arm_rot, rpy2quat(np.pi/2, 0, np.pi/2))
            # 使用opspace控制器控制franka机械臂
            tau = opspace(
                model=model,
                data=data,
                site_id=site_id,
                dof_ids=np.array(panda_joint_ids),
                pos=target_pos,
                ori=target_rot,  # (w, x, y ,z)
                pos_gains=pos_gains,
                ori_gains=ori_gains,
                damping_ratio=1.0
            )
            
            # 设置控制力矩
            data.ctrl[panda_joint_ids] = tau
            data.ctrl[7:] = ctrl
            
            # 执行仿真步骤
            mujoco.mj_step(model, data)
            
            # 获取当前site位置和姿态
            site_pos = data.site_xpos[site_id].copy()
            site_xmat = data.site_xmat[site_id].reshape((3, 3))
            # 从旋转矩阵获取四元数，然后转换为mujoco格式
            site_quat_scipy = R.from_matrix(site_xmat).as_quat()  # scipy格式 (x,y,z,w)
            site_quat = quat_xyzw_to_wxyz(site_quat_scipy)  # 转换为mujoco格式 (w,x,y,z)
            
            # 更新mocap body跟随attachment_site
            data.mocap_pos[franka_mocap_id] = site_pos
            data.mocap_quat[franka_mocap_id] = site_quat
            # 更新目标mocap body
            data.mocap_pos[target_mocap_id] = target_pos
            data.mocap_quat[target_mocap_id] = target_rot

            # 可视化关键点
            keypoints = dphand_teleoperator.retargeting.target_pos
            # 将关键点转换到世界坐标系
            hand_pos = data.xpos[hand_base_id+1]
            hand_rot = data.xmat[hand_base_id]
            keypoints_world = (hand_rot.reshape(3,3) @ (keypoints - keypoints[0]).T).T + hand_pos
            render_targets(viewer.user_scn, keypoints_world, color=(1, 0, 0), size=0.005)
            
            # 同步viewer
            viewer.sync()
            
            cnt += 1
            if cnt % 100 == 0:  # 每100帧打印一次信息
                fps = cnt / (time.time() - start_time)
                print(f"FPS: {fps:.1f}, pos error: {target_pos - site_pos}")

if __name__ == "__main__":
    main()