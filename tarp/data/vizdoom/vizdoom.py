import os
import numpy as np
import cv2
from itertools import accumulate
from collections import deque
import pickle
from random import randint
import h5py

from tarp.utils.general_utils import AttrDict
from tarp.utils.image_utils import center_crop_images
from tarp.components.data_loader import OfflineVideoDataset
from torchvision import transforms
import tarp.utils.data_aug as aug

class VizdoomDataset(OfflineVideoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dict = AttrDict()
        self.collect_samples()
        self.aug_fcns = self.spec.augs if 'augs' in self.spec else AttrDict(no_aug=aug.no_aug)
        if hasattr(self.spec, 'color_map'):
            self.color_map = self.spec.color_map

    def collect_samples(self):
        images = []
        # next_images = []
        rewards = []
        actions = []
        discounted_returns = []
        terms = []
        seg_targets = []
        obj_labels = []
        data_class = []
        for i, task_name in enumerate(self.spec.task_names):
            dir_name = os.path.join(self.spec.dataset_prefix, task_name)
            path_list = os.listdir(dir_name)
            if self.phase is not None:
                if self.phase == 'train':
                    path_list = path_list[:int(len(path_list)*self.spec.split_frac)]
                else:
                    path_list = path_list[int(len(path_list)*self.spec.split_frac):]
            num_images = 0
            for path in path_list:
                with h5py.File(os.path.join(dir_name, path), 'r') as f:
                    data = f['traj0']
                    obs = np.array(data['images'][()])
                    images.extend(obs)
                    actions.extend(data['actions'][()])
                    terms_ep = np.zeros(len(obs))
                    terms_ep[-1] = 1.0
                    terms.extend(terms_ep)
                    reversed_rewards = data['rewards'][()][::-1]
                    values = np.array(list(accumulate(reversed_rewards, lambda x,y: x*self.spec.discount_factor + y))[::-1])
                    discounted_returns.extend(values)
                    seg_targets.extend(np.argmax(np.array(data['seg_targets'][()]), axis=1)[:, None])
                    obj_labels.extend(data['obj_labels'][()])
                    data_class.extend(i*np.ones(len(obs)))

                    num_images += len(obs)
                    rewards.extend(data['rewards'][()])
            print(task_name + ' # images ', num_images)

        self.data_dict.observation = np.array(images).astype(np.uint8)
        self.data_dict.reward = np.array(rewards).reshape(-1, 1).astype(np.float32)
        self.data_dict.done = np.array(terms).reshape(-1, 1).astype(np.float32)
        self.data_dict.action = np.array(actions).reshape(-1, 1).astype(int)
        self.data_dict.seg_targets = np.array(seg_targets).astype(np.uint8)
        self.data_dict.discounted_returns = np.array(discounted_returns).astype(np.float32)
        self.data_dict.obj_labels = np.array(obj_labels).astype(np.float32)
        # self.data_dict.data_class = np.array(data_class).astype(int)
        self.data_dict.task_ids = np.array(data_class).astype(int)

    def __len__(self):
        return len(self.data_dict.observation)

    def get_sample(self):
        idx = np.random.randint(0, len(self.data_dict.observation)-1, size=1)
        data_dict = AttrDict()
        images = self.data_dict.observation[idx]
        seg_targets = self.data_dict.seg_targets[idx]

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

        image_size = images.shape[1:]
        combined_images = np.concatenate((images, seg_targets), axis=1)
        if self.phase == 'train':
            for aug, func in self.aug_funcs.items():
                if 'crop' in aug or 'cutout' in aug:
                    combined_images = func(combined_images)
                else:
                    combined_images = func(combined_images)

        images = combined_images[:, :self.spec.n_frames]
        seg_targets = combined_images[:, self.spec.n_frames:]
        data_dict.images = images.astype(np.float32)
        data_dict.actions = self.data_dict.action[idx][0]
        data_dict.seg_targets = seg_targets.astype(int)
        data_dict.rewards = self.data_dict.reward[idx]
        data_dict.task_id = self.data_dict.task_ids[idx]
        data_dict.discounted_returns = self.data_dict.discounted_returns[idx]
        data_dict.terms = self.data_dict.done[idx]
        data_dict.obj_labels = self.data_dict.obj_labels[idx]
        data_dict.color_map = np.expand_dims(np.array(self.color_map), axis=0)
        return data_dict

class VizdoomSeqDataset(VizdoomDataset):
    def get_sample(self):
        idx = np.random.randint(0, len(self.data_dict.observation)-1-self.spec.delta_t, size=1)[0]
        data_dict = AttrDict()
        seg_targets = self.data_dict.seg_targets[idx]

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
        data_dict.actions = self.data_dict.action[idx][0]
        data_dict.seg_targets = seg_targets.astype(int)
        data_dict.rewards = self.data_dict.reward[idx]
        data_dict.task_ids = self.data_dict.task_ids[idx]
        data_dict.discounted_returns = self.data_dict.discounted_returns[idx]
        data_dict.terms = self.data_dict.done[idx]
        data_dict.obj_labels = self.data_dict.obj_labels[idx]
        data_dict.color_map = np.expand_dims(np.array(self.color_map), axis=0)
        return data_dict
