from contextlib import contextmanager
from functools import partial
import torch
import numpy as np
from torchvision.transforms import Resize
from PIL import Image
import cv2

from tarp.utils.general_utils import ParamDict, AttrDict, map_recursive
from tarp.utils.pytorch_utils import ar2ten, ten2ar


class BaseEnvironment:
    """Implements basic environment interface."""
    # TODO add frame skip interface

    @contextmanager
    def val_mode(self):
        """Sets validation parameters if desired. To be used like: with env.val_mode(): ...<do something>..."""
        pass; yield; pass

    def _default_hparams(self):
        default_dict = ParamDict({
            'device': None,         # device that all tensors should get transferred to
            'screen_width': 128,     # width of rendered images
            'screen_height': 128,    # height of rendered images
        })
        return default_dict

    def reset(self):
        """Resets all internal variables of the environment."""
        raise NotImplementedError

    def step(self, action):
        """Performs one environment step. Returns dict <next observation, reward, done, info>."""
        raise NotImplementedError

    def render(self, mode='rgb_array'):
        """Renders current environment state. Mode {'rgb_array', 'none'}."""
        raise NotImplementedError

    def _wrap_observation(self, obs):
        """Process raw observation from the environment before return."""
        return obs

    @property
    def agent_params(self):
        """Parameters for agent that can be handed over after env is constructed."""
        return AttrDict()


class GymEnv(BaseEnvironment):
    """Wrapper around openai/gym environments."""
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._env = self._make_env(self._hp.name)
        self.set_config(self._hp)

        from mujoco_py.builder import MujocoException
        self._mj_except = MujocoException

    def _default_hparams(self):
        default_dict = ParamDict({
            'name': None,   # name of openai/gym environment
            'reward_norm': 1.,  # reward normalization factor
            'punish_reward': -100,   # reward used when action leads to simulation crash
            'unwrap_time': True,    # removes time limit wrapper from envs so that done is not set on timeout
        })

        return super()._default_hparams().overwrite(default_dict)

    def reset(self):
        obs = self._env.reset()
        return self._wrap_observation(obs)

    def step(self, action):
        if isinstance(action, torch.Tensor): action = ten2ar(action)
        try:
            obs, reward, done, info = self._env.step(action)
            reward = reward / self._hp.reward_norm
        except self._mj_except:
            # this can happen when agent drives simulation to unstable region (e.g. very fast speeds)
            print("Catch env exception!")
            obs = self.reset()
            reward = self._hp.punish_reward     # this avoids that the agent is going to these states again
            done = np.array(True)        # terminate episode (observation will get overwritten by env reset)
            info = {}

        return self._wrap_observation(obs), np.array(reward), np.array(done), info

    def render(self, mode='rgb_array'):
        # TODO make env render in the correct size instead of downsizing after for performance
        img = self._env.render(mode=mode)
        h, w, c = img.shape
        if c == 1:
            img = img.repeat(3, axis=2)
        img = Resize((self._hp.screen_height, self._hp.screen_width))(Image.fromarray(img))
        return np.array(img) / 255.

    def render_mask(self):
        img = self._env.render_mask()
        h, w, c = img.shape
        if c == 1:
            img = img.repeat(3, axis=2)
        img = Resize((self._hp.screen_height, self._hp.screen_width))(Image.fromarray(img))
        return np.array(img) / 255.

    def _render(self, mode='rgb_array'):
        return self._env.render(mode=mode)

    def set_config(self, spec):
        try:
            self._env.set_config(spec)
        except AttributeError:
            pass
        self._spec = spec

    def get_dataset(self):
        dataset = None
        try:
            dataset = self._env.get_dataset()
            dataset.pop('timeouts')
            dataset['action'] = dataset.pop('actions')
            dataset['done'] = dataset.pop('terminals').astype(np.float32)
            dataset['observation'] = dataset.pop('observations')
            dataset['observation_next'] = np.concatenate([dataset['observation'][1:], np.zeros_like(dataset['observation'][0])[np.newaxis, :]], axis=0)
            dataset['reward'] = dataset.pop('rewards')
        except AttributeError:
            pass
        return dataset

    def _make_env(self, id):
        """Instantiates the environment given the ID."""
        import gym
        # import d4rl
        from gym import wrappers
        env = gym.make(id)
        if isinstance(env, wrappers.TimeLimit) and self._hp.unwrap_time:
            # unwraps env to avoid this bug: https://github.com/openai/gym/issues/1230
            env = env.env
        return env

    def get_episode_info(self):
        """Allows to return logging info about latest episode (sindce last reset)."""
        if hasattr(self._env, "get_episode_info"):
            return self._env.get_episode_info()
        return AttrDict()

class DMControlEnv(BaseEnvironment):
    """Wrapper around openai/gym environments."""
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._env = self._make_env(self._hp.name)
        self.set_config(self._hp)

        from mujoco_py.builder import MujocoException
        self._mj_except = MujocoException

    def _default_hparams(self):
        default_dict = ParamDict({
            'name': None,   # name of openai/gym environment
            'reward_norm': 1.,  # reward normalization factor
            'punish_reward': -100,   # reward used when action leads to simulation crash
            'unwrap_time': True,    # removes time limit wrapper from envs so that done is not set on timeout
            'task_name': None,      # task name for dm control
            'from_pixels': False,  # turn on pixel observation
            'frame_skip': 4,   # frame skip
            'resolution': 64,
        })

        return super()._default_hparams().overwrite(default_dict)

    def reset(self):
        obs = self._env.reset()
        if self._hp.from_pixels:
            obs = cv2.resize(obs.transpose((1, 2, 0)), (self._hp.resolution, self._hp.resolution))
            obs = obs / 255.
        return self._wrap_observation(obs)

    def step(self, action):
        if isinstance(action, torch.Tensor): action = ten2ar(action)
        try:
            obs, reward, done, info = self._env.step(action)
            if self._hp.from_pixels:
                obs = cv2.resize(obs.transpose((1, 2, 0)), (self._hp.resolution, self._hp.resolution))
                obs = obs / 255.

            reward = reward / self._hp.reward_norm
        except self._mj_except:
            # this can happen when agent drives simulation to unstable region (e.g. very fast speeds)
            print("Catch env exception!")
            obs = self.reset()
            reward = self._hp.punish_reward     # this avoids that the agent is going to these states again
            done = np.array(True)        # terminate episode (observation will get overwritten by env reset)
            info = {}

        return self._wrap_observation(obs), np.array(reward, dtype=np.float64), np.array(done), info

    def current_state(self):
        return self._env.current_state

    def current_obs(self):
        return cv2.resize(self._env.render(), (self._hp.resolution, self._hp.resolution)) / 255.

    def render(self, mode='rgb_array'):
        # TODO make env render in the correct size instead of downsizing after for performance
        img = self._env.render(mode=mode)
        h, w, c = img.shape
        if c == 1:
            img = img.repeat(3, axis=2)
        img = Resize((self._hp.screen_height, self._hp.screen_width))(Image.fromarray(img))
        return np.array(img) / 255.

    def _render(self, mode='rgb_array'):
        return self._env.render(mode=mode)

    def set_config(self, spec):
        try:
            self._env.set_config(spec)
        except AttributeError:
            pass
        self._spec = spec

    def get_dataset(self):
        dataset = None
        try:
            dataset = self._env.get_dataset()
            dataset.pop('timeouts')
            dataset['action'] = dataset.pop('actions')
            dataset['done'] = dataset.pop('terminals').astype(np.float32)
            dataset['observation'] = dataset.pop('observations')
            dataset['observation_next'] = np.concatenate([dataset['observation'][1:], np.zeros_like(dataset['observation'][0])[np.newaxis, :]], axis=0)
            dataset['reward'] = dataset.pop('rewards')
        except AttributeError:
            pass
        return dataset

    def _make_env(self, id):
        """Instantiates the environment given the ID."""
        import dmc2gym
        from gym import wrappers
        env = dmc2gym.make(domain_name=id, task_name=self._hp.task_name, seed=self._hp.seed,
                           from_pixels=self._hp.from_pixels, frame_skip=self._hp.frame_skip, visualize_reward=False)
        if isinstance(env, wrappers.TimeLimit) and self._hp.unwrap_time:
            # unwraps env to avoid this bug: https://github.com/openai/gym/issues/1230
            env = env.env
        return env

    def get_episode_info(self):
        """Allows to return logging info about latest episode (sindce last reset)."""
        if hasattr(self._env, "get_episode_info"):
            return self._env.get_episode_info()
        return AttrDict()

class AtariEnv(BaseEnvironment):
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._env = self._make_env(self._hp.name)
        obs_dims = self._env.observation_space
        self.game_over = False
        self.lives = 0
        self.screen_buffer = [
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
        ]

        self.set_config(self._hp)
        from mujoco_py.builder import MujocoException
        self._mj_except = MujocoException

    def _default_hparams(self):
        default_dict = ParamDict({
            'name': None,   # name of openai/gym environment
            'reward_norm': 1.,  # reward normalization factor
            'punish_reward': -100,   # reward used when action leads to simulation crash
            'unwrap_time': True,    # removes time limit wrapper from envs so that done is not set on timeout
            'resolution': 64,
            'frameskip': 4
        })

        return super()._default_hparams().overwrite(default_dict)

    def reset(self):
        obs = self._env.reset()
        self.lives = self._env.ale.lives()
        self._fetch_grayscale_observation(self.screen_buffer[0])
        self.screen_buffer[1].fill(0)
        obs = self._pool_and_resize()
        obs = obs.astype(np.float32)
        return self._wrap_observation(obs / 255.)

    def step(self, action):
        if isinstance(action, torch.Tensor): action = ten2ar(action)
        try:
            reward = 0.
            for time_step in range(self._hp.frameskip):
                _, rew, done, info = self._env.step(action)
                reward += rew
                if done:
                    break
                elif time_step >= self._hp.frameskip - 2:
                    t = time_step - (self._hp.frameskip - 2)
                    self._fetch_grayscale_observation(self.screen_buffer[t])
            obs = self._pool_and_resize()
            self.game_over = done

            # obs, reward, done, info = self._env.step(action)
            # obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            # obs = cv2.resize(obs, (self._hp.resolution, self._hp.resolution))[:, :, None]
            obs = obs.astype(np.float32)
            obs /= 255.
            reward = reward / self._hp.reward_norm
        except self._mj_except:
            # this can happen when agent drives simulation to unstable region (e.g. very fast speeds)
            print("Catch env exception!")
            obs = self.reset()
            reward = self._hp.punish_reward     # this avoids that the agent is going to these states again
            done = np.array(True)        # terminate episode (observation will get overwritten by env reset)
            info = {}

        return self._wrap_observation(obs), reward, np.array(done), info

    def _fetch_grayscale_observation(self, output):
        self._env.ale.getScreenGrayscale(output)
        return output

    def _pool_and_resize(self):
        if self._hp.frameskip > 1:
            np.maximum(self.screen_buffer[0], self.screen_buffer[1], out=self.screen_buffer[0])

        transformed_image = cv2.resize(self.screen_buffer[0],
                                       (self._hp.resolution, self._hp.resolution),
                                       interpolation=cv2.INTER_AREA)
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        return np.expand_dims(int_image, axis=2)

    def render(self, mode='rgb_array'):
        # TODO make env render in the correct size instead of downsizing after for performance
        img = self._env.render(mode=mode)
        h, w, c = img.shape
        if c == 1:
            img = img.repeat(3, axis=2)
        img = Resize((self._hp.screen_height, self._hp.screen_width))(Image.fromarray(img))
        return np.array(img) / 255.

    def _render(self, mode='rgb_array'):
        return self._env.render(mode=mode)

    def set_config(self, spec):
        self._spec = spec

    def _make_env(self, id):
        """Instantiates the environment given the ID."""
        import gym
        from gym import wrappers
        # env = gym.make(id, frameskip=self._hp.frameskip)
        env = gym.make(id)
        if isinstance(env, wrappers.TimeLimit) and self._hp.unwrap_time:
            # unwraps env to avoid this bug: https://github.com/openai/gym/issues/1230
            env = env.env
        return env

    def get_episode_info(self):
        """Allows to return logging info about latest episode (sindce last reset)."""
        if hasattr(self._env, "get_episode_info"):
            return self._env.get_episode_info()
        return AttrDict()


class GoalConditionedEnv(BaseEnvironment):
    def __init__(self):
        self.goal = None

    def sample_goal(self):
        raise NotImplementedError("Please implement this method in a subclass.")

    def reset(self):
        self.goal = self.sample_goal()

    def _wrap_observation(self, obs):
        return np.concatenate([obs, self.goal])
