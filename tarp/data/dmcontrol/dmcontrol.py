import os
import numpy as np
import cv2
from itertools import accumulate
from collections import deque
import pickle
from random import randint
import gc

from tarp.utils.general_utils import AttrDict
from tarp.utils.image_utils import center_crop_images
from tarp.components.data_loader import OfflineVideoDataset
from torchvision import transforms
import tarp.utils.data_aug as aug
import h5py
import gzip
import pickle
import random

class DMControlDataset(OfflineVideoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dict = AttrDict()
        self.collect_samples()

        self.aug_funcs = {}

    def collect_samples(self):
        self.data_dict.observation = []
        self.data_dict.action = []
        self.data_dict.reward = []
        self.data_dict.done = []
        self.data_dict.task_ids = []
        self.data_dict.discounted_returns = []

        for i, task_name in enumerate(self.spec.task_names):
            dir_name = os.path.join(self.spec.dataset_prefix, task_name)
            path_list = os.listdir(dir_name)
            if self.phase is not None:
                if self.phase == 'train':
                    path_list = path_list[:int(len(path_list)*self.spec.split_frac)]
                else:
                    path_list = path_list[int(len(path_list)*self.spec.split_frac):]

            for path in path_list:
                key = 'traj0/'
                with h5py.File(os.path.join(dir_name, path), 'r') as f:
                    reward = f[key + 'reward'][:]
                    self.data_dict.action.extend(f[key + 'action'][:])
                    self.data_dict.reward.extend(reward)
                    self.data_dict.observation.extend(f[key + 'observation'][:].astype(np.uint8))
                    self.data_dict.done.extend(f[key + 'done'][:])
                reversed_rewards = np.array(reward)[::-1]
                values = np.array(list(accumulate(reversed_rewards, lambda x,y: x*self.spec.discount_factor + y))[::-1])
                self.data_dict.discounted_returns.extend(values)
                self.data_dict.task_ids.extend(i*np.ones(len(reward)))

        print("Num Images: ", len(self.data_dict.observation))
        self.data_dict.observation = np.array(self.data_dict.observation).astype(np.uint8)
        # self.data_dict.states = np.array(self.data_dict.observation).astype(np.uint8)
        # self.data_dict.observation_next = np.array(next_images).astype(np.uint8)
        self.data_dict.reward = np.array(self.data_dict.reward).astype(np.float32)
        self.data_dict.done = np.array(self.data_dict.done).astype(np.float32)
        self.data_dict.action = np.array(self.data_dict.action).astype(np.float32)
        self.data_dict.task_ids = np.array(self.data_dict.task_ids).astype(int)
        self.data_dict.discounted_returns = np.array(self.data_dict.discounted_returns).astype(np.float32)

    def __len__(self):
        return len(self.data_dict.observation)

    def get_sample(self):
        idx = np.random.randint(0, len(self.data_dict.observation), size=1)
        images = self.data_dict.observation[idx]

        data_dict = AttrDict()
        if self.spec.n_frames > 1:
            past_frames = deque(maxlen=self.spec.n_frames)
            for offset in reversed(range(self.spec.n_frames)):
                if not past_frames:
                    [past_frames.append(self.data_dict.observation[idx-offset]) for _ in range(self.spec.n_frames)]
                if bool(self.data_dict.done[idx-offset]) and offset > 0:
                    past_frames = deque(maxlen=self.spec.n_frames)
                else:
                    past_frames.append(self.data_dict.observation[idx-offset])
            images = np.concatenate(list(past_frames), axis=1)

        data_dict.observation = images.astype(np.float32) / (255./2.) - 1.0
        data_dict.actions = self.data_dict.action[idx]
        data_dict.rewards = self.data_dict.reward[idx]
        data_dict.terms = self.data_dict.done[idx]
        data_dict.task_id = self.data_dict.task_ids[idx]
        data_dict.discounted_returns = self.data_dict.discounted_returns[idx]
        return data_dict


class DMControlSeqDataset(DMControlDataset):
    def get_sample(self):
        idx = np.random.randint(0, len(self.data_dict.observation)-1-self.spec.delta_t, size=1)[0]
        data_dict = AttrDict()
        images = []
        if self.spec.n_frames > 1:
            past_frames = deque(maxlen=self.spec.n_frames)
            for offset in reversed(range(self.spec.n_frames)):
                if not past_frames:
                    [past_frames.append(self.data_dict.observation[idx-offset]) for _ in range(self.spec.n_frames)]
                if bool(self.data_dict.done[idx-offset]) and offset > 0:
                    past_frames = deque(maxlen=self.spec.n_frames)
                else:
                    past_frames.append(self.data_dict.observation[idx-offset])
            images.append(np.concatenate(list(past_frames), axis=0))

            for i in range(self.spec.delta_t):
                if self.data_dict.done[idx+i]:
                    past_frames.append(np.zeros_like(self.data_dict.observation[idx+i+1]))
                else:
                    past_frames.append(self.data_dict.observation[idx+i+1])
                images.append(np.concatenate(list(past_frames), axis=0))
        else:
            images = self.data_dict.observation[idx:idx+self.spec.delta_t+1]

        data_dict.images = np.array(images).astype(np.float32)
        data_dict.actions = self.data_dict.action[idx]
        data_dict.rewards = self.data_dict.reward[idx]
        data_dict.terms = self.data_dict.done[idx]
        data_dict.task_id = self.data_dict.task_ids[idx]
        return data_dict
