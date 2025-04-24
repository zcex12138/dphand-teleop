import numpy as np
import mujoco
import mujoco.viewer
from dphand_teleop.dphand_teleoperator import DPhandTeleoperator
import time

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
model = mujoco.MjModel.from_xml_path('./assets/DPhand/DPHand_free.xml')
data = mujoco.MjData(model)
# test=True则从data中读取一帧数据
dphand_teleoperator = DPhandTeleoperator(model, data, ip="192.168.3.27", test=False)

# reset
mujoco.mj_forward(model, data)


index_1 = [5,6,7,8,9]
# 启动 viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 设置自定义渲染回调
    viewer._render_every_frame = False  # 禁用默认渲染
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE  # 显示站点坐标轴
    # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = model.body("base").id  # 跟踪的物体ID
    viewer.cam.distance = 2.0    # 与目标的距离
    viewer.cam.azimuth = 90.0    # 水平偏移角度
    viewer.cam.elevation = -15.0 # 俯视角度
    cnt = 0
    start_time = time.time()
    while viewer.is_running():
        viewer.user_scn.ngeom = 0
        mujoco.mj_step(model, data)
        # retarget
        angle = dphand_teleoperator.get_target_action_j2j() # 28 joints
        # control
        data.ctrl[:6] = 0 # set wrist and palm to 0
        data.ctrl[6:] = angle[6:]
        # visualize
        # render_targets(viewer.user_scn, keypoints[index_1], size=0.005)
        render_targets(viewer.user_scn, dphand_teleoperator.retargeting.target_positions, size=0.005)
        # render_targets(viewer.user_scn, data.xpos[3], size=0.005)
        # for joint_name in dphang_retarget.joint_names:
        #     joint_pos = dphang_retarget.calculate_joint_pos(joint_name)[0]
        #     render_targets(viewer.user_scn, joint_pos, color=(0,1,1), size=0.005)
        # for joint_name in dphang_retarget.finger_base_names:
        #     joint_pos = dphang_retarget.calculate_joint_pos(joint_name)[0]
        #     render_targets(viewer.user_scn, joint_pos, color=(1,0,1), size=0.005)
        # render_targets(viewer.user_scn, [data.xpos[3]], color=(0,1,0), size=0.02)
        # 同步 viewer
        viewer.sync()