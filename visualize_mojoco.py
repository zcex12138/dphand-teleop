import numpy as np
import mujoco
import mujoco.viewer
from dphand_teleop.dphand_teleoperator import DPhandTeleoperator
import time
from dphand_utils.math_utils import rpy2mtx

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
model = mujoco.MjModel.from_xml_path('./assets/DPhand/dphand_grasp.xml')
data = mujoco.MjData(model)
# reset
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# test=True则从录制的文件中读取数据
dphand_teleoperator = DPhandTeleoperator(ip="192.168.3.11", test=True, relative=True, n_step=15)


index_1 = [2,3,4]
# 启动 viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer._render_every_frame = False  # 禁用默认渲染
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE  # 显示站点坐标轴
    # 自由相机
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # 显示接触点
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
    # 跟踪相机
    # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    # viewer.cam.trackbodyid = model.body("base").id  # 跟踪的物体ID
    # viewer.cam.distance = 2.0    # 与目标的距离
    # viewer.cam.azimuth = 90.0    # 水平偏移角度
    # viewer.cam.elevation = -15.0 # 俯视角度
    cnt = 0
    start_time = time.time()
    init_arm_pos = data.ctrl[:3].copy()
    while viewer.is_running():
        viewer.user_scn.ngeom = 0
        mujoco.mj_step(model, data)
        # control
        ctrl = dphand_teleoperator.get_target_action_p2p() # 28 joints
        data.ctrl[6:] = ctrl[6:]
        data.ctrl[:3] = init_arm_pos + ctrl[:3] * 2.0
        data.ctrl[:3] = 0

        # visualize
        keypoints = dphand_teleoperator.retargeting.target_positions
        keypoints = (rpy2mtx(*data.qpos[3:6]) @ (keypoints - keypoints[0]).T).T + data.xpos[3]
        render_targets(viewer.user_scn, keypoints, size=0.005)
        # render_targets(viewer.user_scn, data.xpos[3], size=0.005)
        
        # 同步 viewer
        viewer.sync()
        # cnt += 1
        # fps = cnt / (time.time() - start_time)
        # print(fps)