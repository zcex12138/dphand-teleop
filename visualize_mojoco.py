import numpy as np
import mujoco
import mujoco.viewer
from dphand_teleop.dphand_teleoperator import DPhandTeleoperator
import time

def render_targets(scn, targets, color=(1, 0, 0), size=0.005):
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
dphand_teleoperator = DPhandTeleoperator(model, data, ip="192.168.3.8", test=True)
# reset
mujoco.mj_forward(model, data)

# index_1 = [2,3,4,7,8,9,12,13,14,17,18,19,22,23,24] # 需要匹配的关键点
# index_2 = [1,6,11,16,21] # 手指根部关节

# import cProfile
# profiler = cProfile.Profile()
# profiler.enable()
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
        # if cnt % 10 == 0:
        #     ctrl = dphand_teleoperator.get_target_action_p2p()
        # else:
        #     ctrl = dphand_teleoperator.last_action
        # control
        ctrl = dphand_teleoperator.get_target_action_p2p()
        # data.ctrl[8:12] = ctrl[8:12]
        data.ctrl[8:] = ctrl[8:]
        # marker
        # data.mocap_pos[0] = ctrl[:3] # r-x, g-y, b-z.
        # data.mocap_quat[0] = rpy2quat(*ctrl[3:6])
        # visualize
        render_targets(viewer.user_scn, dphand_teleoperator.retargeting.target_positions, size=0.005)
        # render_targets(viewer.user_scn, target_keypoints[index_2], size=0.005)
        # for joint_name in dphang_retarget.joint_names:
        #     joint_pos = dphang_retarget.calculate_joint_pos(joint_name)[0]
        #     render_targets(viewer.user_scn, joint_pos, color=(0,1,1), size=0.005)
        # for joint_name in dphang_retarget.finger_base_names:
        #     joint_pos = dphang_retarget.calculate_joint_pos(joint_name)[0]
        #     render_targets(viewer.user_scn, joint_pos, color=(1,0,1), size=0.005)
        # render_targets(viewer.user_scn, [data.xpos[3]], color=(0,1,0), size=0.02)
        # 同步 viewer
        viewer.sync()
        cnt += 1
        # print("time_per_step: ", (time.time() - start_time)/cnt)
        # print(ctrl)

# profiler.disable()
# import pstats
# stats = pstats.Stats(profiler)
# stats.sort_stats(pstats.SortKey.CUMULATIVE)
# stats.print_stats(20)
