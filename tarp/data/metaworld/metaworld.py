
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

class MetaWorldDataset(OfflineVideoDataset):
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

        for i, task_name in enumerate(self.spec.task_names):
            print('Loading..{}'.format(task_name))
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
                self.data_dict.task_ids.extend(i*np.ones(len(reward)))

        print("Num Images: ", len(self.data_dict.observation))
        self.data_dict.observation = np.array(self.data_dict.observation).astype(np.uint8)
        self.data_dict.reward = np.array(self.data_dict.reward).astype(np.float32)
        self.data_dict.done = np.array(self.data_dict.done).astype(np.float32)
        self.data_dict.action = np.array(self.data_dict.action).astype(np.float32)
        self.data_dict.task_ids = np.array(self.data_dict.task_ids).astype(int)

    def __len__(self):
        return len(self.data_dict.observation)

