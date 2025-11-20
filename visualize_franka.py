#!/usr/bin/env python3
"""
example of franka panda moving to default position using opspace controller
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
from dphand_utils.math_utils import rpy2quat, quat_mul, rpy2mtx, mtx2quat, quat2mtx
from franka_controller.opspace import opspace
from scipy.spatial.transform import Rotation as R
from dphand_teleop.teleoperator import VisionProTeleoperator

def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion from scipy format (x,y,z,w) to mujoco format (w,x,y,z)"""
    return np.roll(quat_xyzw, 1)

def main():
    # 加载模型
    model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    
    # 获取panda的7个关节ID
    panda_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") 
                      for i in range(1, 8)]
    
    # site id
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
    # 获取mocap body
    franka_mocap_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pinch")]
    target_mocap_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "teleop")]

    # 设置目标位置（可以调整）
    target_pos = np.array([0.3, 0.0, 0.5])
    target_rot = (1,0,0,0)
    
    # 默认关节位置（用于零空间控制）
    default_joint_pos = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
    # opspace 控制器参数
    pos_gains = 800.0 * np.array([1.0, 1.0, 1.0])
    ori_gains = 800.0 * np.array([1.0, 1.0, 1.0])
    
    teleoperator = VisionProTeleoperator(ip="192.168.3.100", test=True, use_relative_pose=True, type="franka", n_step=1)
    rot = rpy2mtx(0, 0, -np.pi/2)
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
        # v11 = R1 @ R @ v00 = R1 @ R @ inv(R1) @ v10
        while viewer.is_running():
            delta_pos, arm_rot, ctrl = teleoperator.get_action()
            target_pos = init_pos + rot @ delta_pos * 3.5
            target_rot = mtx2quat(rot @ quat2mtx(arm_rot) @ np.linalg.inv(rot))
            tau = opspace(
                model=model,
                data=data,
                site_id=site_id,
                dof_ids=np.array(panda_joint_ids),
                pos=target_pos,
                ori=target_rot,  # (w, x, y, z)
                pos_gains=pos_gains,
                ori_gains=ori_gains,
                damping_ratio=1.0
            )
            # 设置控制力矩
            data.ctrl[:7] = tau
            
            # PD 控制器参数
            kp = 2000  # 位置增益
            kd = 100   # 阻尼增益（可选）
            current_vel = (data.qvel[7] + data.qvel[8]) / 2
            current_joint_pos = (data.qpos[7] + data.qpos[8]) / 2
            pos_error = ctrl/2 - current_joint_pos
            
            # PD 控制：force = kp * error - kd * velocity
            force_cmd = kp * pos_error - kd * current_vel
            
            # 将力命令转换为控制信号（gainprm[0]=1，所以直接使用力值）
            data.ctrl[7] = force_cmd
            
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
            # 同步viewer
            viewer.sync()
            time.sleep(0.01)
            # cnt += 1
            # if cnt % 100 == 0:  # 每100帧打印一次信息
            #     fps = cnt / (time.time() - start_time)
            #     pos_error = np.linalg.norm(target_pos - current_pos)
            #     print(f"FPS: {fps:.1f}, position error: {pos_error:.4f}")

if __name__ == "__main__":
    main()