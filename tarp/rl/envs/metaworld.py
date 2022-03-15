import os
import glob
import h5py
import random
import numpy as np
from torchvision.transforms import Resize
from PIL import Image
import torch
import cv2

from tarp.rl.components.environment import GymEnv, GoalConditionedEnv
from tarp.data.metaworld.metaworld_utils import MetaWorldEnvHandler
from tarp.utils.general_utils import ParamDict



class MetaWorldEnv(GymEnv):
    """Tiny wrapper around GymEnv for Metaworld tasks."""
    def _make_env(self, name):
        return MetaWorldEnvHandler.env_from_name(name)

    def _default_hparams(self):
        default_dict = ParamDict({
            "from_pixels": True,
        })

        return super()._default_hparams().overwrite(default_dict)

    def render(self, mode='rgb_array', camera_name='behindGripper'):
        # TODO make env render in the correct size instead of downsizing after for performance
        img = self._env.render(offscreen=True, camera_name=camera_name)
        h, w, c = img.shape
        if c == 1:
            img = img.repeat(3, axis=2)
        img = np.array(Resize((self._hp.screen_height, self._hp.screen_width))(Image.fromarray(img)))
        if camera_name == 'behindGripper':
            img = img[::-1]
        return img / 255.

    def step(self, action):
        if isinstance(action, torch.Tensor): action = ten2ar(action)
        try:
            obs, reward, done, info = self._env.step(action)
            reward = reward / self._hp.reward_norm

            if self._hp.from_pixels:
                obs = self._render()
                obs = cv2.resize(obs, (self._hp.resolution, self._hp.resolution)).astype(np.float32)
                obs /= 255.

        except self._mj_except:
            # this can happen when agent drives simulation to unstable region (e.g. very fast speeds)
            print("Catch env exception!")
            obs = self.reset()
            reward = self._hp.punish_reward     # this avoids that the agent is going to these states again
            done = np.array(True)        # terminate episode (observation will get overwritten by env reset)
            info = {}

        return self._wrap_observation(obs), np.array(reward, dtype=np.float64), np.array(done), info

    def reset(self):
        obs = super().reset()
        if self._hp.from_pixels:
            obs = self._render()
            obs = cv2.resize(obs, (self._hp.resolution, self._hp.resolution)).astype(np.float32)
            obs /= 255.
        return obs

    def _render(self, mode='rgb_array', camera_name='behindGripper'):
        return self._env.render(offscreen=True, camera_name=camera_name)

class GoalConditionedMetaWorldEnv(MetaWorldEnv, GoalConditionedEnv):
    def __init__(self, config):
        # note that we use multiple inheritance
        # so we need to call constructors manually
        MetaWorldEnv.__init__(self, config)
        GoalConditionedEnv.__init__(self)

    def sample_goal(self):
        env_name = self._hp.name
        return np.zeros(12)     # HACK because we don't actually need these goals anymore
        data_path = os.path.join(self._hp.data_dir, '{}/*.h5'.format(env_name))
        h5_path = random.choice(glob.glob(data_path))
        data = h5py.File(h5_path, 'r')
        goal = data['traj0']['states'][-1]
        return goal

    def reset(self):
        GoalConditionedEnv.reset(self)
        return MetaWorldEnv.reset(self)

