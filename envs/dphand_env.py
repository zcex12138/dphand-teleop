""" minimal example of a DPHand manipulation environment"""

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DPHandManipulationEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    # set default episode_len for truncate episodes
    def __init__(self, episode_len=500, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        # 24 joints + 6 hand pose
        observation_space = Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float64)
        # load your MJCF model with env and choose frames count between actions
        MujocoEnv.__init__(
            self,
            os.path.join(PROJECT_PATH, "./assets/DPhand/DPHand_free.xml"),
            5,
            observation_space=observation_space,
            **kwargs
        )
        self.step_count = 0
        self.episode_len = episode_len

    def step(self, action):
        reward = self._get_reward()
        # Step the simulation n number of frames and applying a control action.
        self.do_simulation(action, self.frame_skip)
        self.step_count += 1

        obs = self._get_obs()
        done = bool(not np.isfinite(obs).all())
        truncated = self.step_count > self.episode_len
        return obs, reward, done, truncated, {}

    def _get_obs(self):
        # Return the observation.
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
    
    def _get_reward(self):
        # Return the reward.
        return 1.0
    
    # define what should happen when the model is reset (at the beginning of each episode)
    def reset_model(self):
        self.step_count = 0

        # for example, noise is added to positions and velocities
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

if __name__ == "__main__":
    import time

    import mujoco.viewer
    env = DPHandManipulationEnv(render_mode="rgb_array")
    obs, info = env.reset()
    frames = []
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