import sys, os

import gym
from gym.spaces import Box, Discrete
from tarp.utils.general_utils import AttrDict
import itertools as it
import copy
import vizdoom
import random
import time
import numpy as np
import re
import cv2

class DoomEnv(gym.Env):
    def __init__(self, **kwargs):
        self.game = vizdoom.DoomGame()
        self.resolution = 64
        self.screen_width = self.screen_height = 256
        self.frame_skips = 4
        self.vizdoom_config_path = None
        self.use_shaped_reward = False
        self.state = None
        self._episode_step = 0

    def _post_process_action(self, action):
        act = np.zeros(self.action_space.n)
        act[action] = 1
        act = np.uint8(act)
        act = act.tolist()
        return act

    def get_episode_info(self):
        ret = {}
        for i, val in enumerate(self.meas):
            ret[self.game_variable_names[i]] = val
        return ret

    def set_env(self):
        assert self.vizdoom_config_path is None, "vizdoom_config_path is None"
        self.game.load_config(self.doom_config_path)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_screen_resolution(getattr(vizdoom.ScreenResolution, 'RES_160X120'))
        self.game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
        self.game.set_doom_map('MAP01')
        self.available_controls, self.continuous_controls, self.discrete_controls = self.analyze_controls(self.doom_config_path)
        self.num_buttons = self.game.get_available_buttons_size()
        assert(self.num_buttons == len(self.discrete_controls) + len(self.continuous_controls))
        assert(len(self.continuous_controls) == 0) # only discrete for now
        self.num_meas = self.game.get_available_game_variables_size()
        self.game.init()
        self.action_space = Discrete(self.num_buttons)

        self.game_variable_names = []
        for game_variable in self.game.get_available_game_variables():
            self.game_variable_names.append(str(game_variable).replace("GameVariable.", '').lower())
        self.meas = np.zeros(len(self.game_variable_names))

    def seed(self, seed=None):
        np.random.seed(seed)

    def analyze_controls(self, config_file):
        with open(config_file, 'r') as myfile:
            config = myfile.read()
        m = re.search('available_buttons[\s]*\=[\s]*\{([^\}]*)\}', config)
        avail_controls = m.group(1).split()
        cont_controls = np.array([bool(re.match('.*_DELTA', c)) for c in avail_controls])
        discr_controls = np.invert(cont_controls)
        return avail_controls, np.squeeze(np.nonzero(cont_controls)), np.squeeze(np.nonzero(discr_controls))

    def set_config(self, spec):
        self._spec = spec
        self.resolution = self._spec.resolution if 'resolution' in self._spec else self.resolution
        self.observation_space = Box(low=0.0, high=1.0,
                                     shape=(self.resolution, self.resolution), dtype=np.float32)
        if 'use_seg_mask' in self._spec:
            self.use_seg_mask = self._spec.use_seg_mask
        else:
            self.use_seg_mask = False

    def _render(self):
        self.game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
        if self.state is None:
            im = np.zeros((self.screen_width, self.screen_height, 1))
        else:
            im = self.state.screen_buffer
            im = cv2.resize(im, (self.screen_width, self.screen_height))[:, :, None]
        return im

    def render(self, mode='rgb_array'):
        return self._render()

    def get_obs(self):
        if self.game.is_episode_finished() or self.game.is_player_dead():
            obs = np.zeros((self.resolution, self.resolution, 1))
        else:
            obs = self._render()
            obs = cv2.resize(obs, (self.resolution, self.resolution))[:, :, None]
        return obs

    def get_seg_mask(self):
        if self.game.is_episode_finished() or self.game.is_player_dead():
            mask = np.zeros((self.resolution, self.resolution, 1))
        else:
            mask = self.state.labels_buffer
            mask = cv2.resize(mask, (self.resolution, self.resolution))[:, :, None]
        return mask

    def reset(self):
        self.game.new_episode()
        self.state = self.game.get_state()
        self.meas = self.state.game_variables
        obs = self._render()
        obs = cv2.resize(obs, (self.resolution, self.resolution))[:, :, None]
        if self.use_seg_mask:
            mask = self.get_seg_mask()
            obs = np.concatenate((obs, mask), axis=2)
        return obs  / 255.

    def step(self, action):
        info = {}
        self._episode_step += 1
        act = self._post_process_action(action)
        # act = np.zeros(self.action_space.n)
        # act[action] = 1
        # act = np.uint8(act)
        # act = act.tolist()
        reward = self.game.make_action(act, self.frame_skips)
        reward = self._custom_reward(reward)
        state = self.game.get_state()
        if state is not None:
            if self.use_shaped_reward:
                reward = self._reward(state)
            meas = self.state.game_variables
            self.meas = np.array(meas)
        self.state = state


        done = self.game.is_episode_finished() or self.game.is_player_dead()
        if done:
            meas = np.zeros(self.num_meas, dtype=np.uint32)
            if self.use_shaped_reward:
                reward = 0.
            info['meas'] = meas
        obs = self.get_obs()

        if self.use_seg_mask:
            mask = self.get_seg_mask()
            obs = np.concatenate((obs, mask), axis=2)

        return obs / 255., np.array(reward).astype(np.float64), done, info

    def _reward(self, state):
        raise NotImplementedError

    def _custom_reward(self, reward):
        return reward

    def get_random_action(self):
        return [(random.random() >= .5) for i in range(self.num_buttons)]

class D3BattleDoomEnv(DoomEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.doom_config_path = os.path.join(os.path.dirname(__file__), './maps/D3_battle.cfg')
        self.set_env()
        self.objective_coef = np.ones(self.num_meas)
        self.discrete_actions = []
        for perm in it.product([False, True], repeat=len(self.discrete_controls)):
            self.discrete_actions.append(list(perm))

    def set_env(self):
        super().set_env()
        self.use_shaped_reward = True

    def set_config(self, spec):
        super().set_config(spec)
        self.objective_coef = self._spec.objective_coef

    def _post_process_action(self, action):
        acts = np.zeros(len(self.discrete_controls), dtype=np.int)
        acts[:] = self.discrete_actions[action]
        return acts.tolist()

    def _reward(self, state):
        meas = copy.deepcopy(state.game_variables)
        meas[0] /= 7.5
        meas[1] /= 30.0
        meas[2] /= 1.0
        meas[0] -= self.meas[0]/7.5
        meas[1] -= self.meas[1]/30.0
        meas[2] -= self.meas[2]/1.0
        return np.sum(meas * self.objective_coef)

class D3BattleDoomStateEnv(D3BattleDoomEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.objects = ["Clip", "CustomMedikit", "DoomImp"]

    def reset(self):
        self.game.new_episode()
        self.state = self.game.get_state()
        self.meas = self.state.game_variables
        return self.get_obs()

    def set_config(self, spec):
        super().set_config(spec)
        if hasattr(self._spec, 'meas_input'):
            self.meas_input = self._spec.meas_input
        else:
            self.meas_input = False

    def get_obs(self):
        state = self.game.get_state()
        obs = np.zeros(len(self.objects))
        meas = np.zeros(len(self.objective_coef))
        if state is not None:
            labels = state.labels
            for l in labels:
                if l.object_name == self.objects[0]:
                    obs[0] = 1.0
                elif l.object_name == self.objects[1]:
                    obs[1] = 1.0
                elif l.object_name == self.objects[2]:
                    obs[2] = 1.0
            if self.meas_input:
                meas = copy.deepcopy(state.game_variables)
                meas[0] /= 7.5
                meas[1] /= 30.0
                meas[2] /= 1.0
        if self.meas_input:
            obs = np.concatenate((obs, meas))
        return obs

