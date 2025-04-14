from envs.dphand_env import DPHandManipulationEnv
import mujoco.viewer
import time

from dphand_retargeting import DPHandRetargeting
from visionpro_listener import VisionProListener

env = DPHandManipulationEnv(render_mode="rgb_array")

# initialize the retargeting and listener
retargeting = DPHandRetargeting(model=env.model, data=env.data)
listener = VisionProListener(ip="192.168.3.91")

obs, info = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    step_count = 0
    action = env.action_space.sample()
    flag = 0
    while viewer.is_running():
        step_start = time.time()
        step_count += 1

        if step_count % 100 == 0:
            action = env.action_space.low * flag + env.action_space.high * (1 - flag)
            flag = 1 - flag
        
        # action = env.action_space.sample() # random actions
        action[:6] = 0
        obs, reward, done, truncated, info = env.step(action)
        # if done or truncated:
        #     obs, info = env.reset()

        # 控制仿真时间和现实时间一致
        viewer.sync()
        time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)




# frames = []
# image = env.render()
# if _ % 5 == 0:
#     frames.append(image)
# cv2.imshow("image", image)
# cv2.waitKey(1)
# uncomment to save result as gif
# with imageio.get_writer("media/test.gif", mode="I") as writer:
#     for idx, frame in enumerate(frames):
#         writer.append_data(frame)
