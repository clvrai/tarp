import torch
import os
import numpy as np

from tarp.rl.components.agent import BaseAgent
from tarp.utils.general_utils import ParamDict, map_dict, AttrDict
from tarp.utils.pytorch_utils import ten2ar, avg_grad_norm, TensorModule, check_shape, map2torch, map2np
from tarp.rl.utils.mpi import sync_networks
from tarp.utils.image_utils import center_crop_images, center_crop_image
import tarp.utils.data_aug as aug

class ACAgent(BaseAgent):
    """Implements a flat (non-hierarchical) actor-critic agent. (does not implement update function)"""
    def __init__(self, config):
        super().__init__(config)
        self._hp = self._default_hparams().overwrite(config)
        self.policy = self._hp.policy(self._hp.policy_params)
        self.aug_fcns = self._hp.augmentation if self._hp.augmentation is not None else AttrDict(no_aug=aug.no_aug)

        if self.policy.has_trainable_params:
            self.policy_opt = self._get_optimizer(self._hp.optimizer, self.policy, self._hp.policy_lr)

    def _default_hparams(self):
        default_dict = ParamDict({
            'policy': None,     # policy class
            'policy_params': None,  # parameters for the policy class
            'policy_lr': 3e-4,  # learning rate for policy update
            'augmentation': None,                   # augmentation, e.g. random shift
        })
        return super()._default_hparams().overwrite(default_dict)

    def _act(self, obs):
        # TODO implement non-sampling validation mode
        if 'crop' in self.aug_fcns and len(obs.shape)==3:
            obs = center_crop_image(obs, self._hp.policy_params.input_res)
        obs = map2torch(self._obs_normalizer(obs), self._hp.device)
        if len(obs.shape) == 1 or len(obs.shape) == 3:     # we need batched inputs for policy
            policy_output = self._remove_batch(self.policy(obs[None]))
            if 'dist' in policy_output:
                del policy_output['dist']
            return map2np(policy_output) #map2np(self._remove_batch(self.policy(obs[None])))
        return map2np(self.policy(obs))

    def _act_rand(self, obs):
        policy_output = self.policy.sample_rand(map2torch(obs, self.policy.device))
        if 'dist' in policy_output:
            del policy_output['dist']
        return map2np(policy_output)
        return map2np(self.policy.sample_rand(map2torch(obs, self.policy.device)))

    def state_dict(self, *args, **kwargs):
        d = super().state_dict()
        if self.policy.has_trainable_params:
            d['policy_opt'] = self.policy_opt.state_dict()
        return d

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.policy_opt.load_state_dict(state_dict.pop('policy_opt'))
        super().load_state_dict(state_dict, *args, **kwargs)

    def visualize(self, logger, rollout_storage, step):
        super().visualize(logger, rollout_storage, step)
        self.policy.visualize(logger, rollout_storage, step)

    def _aux_info(self, policy_output):
        """Optionally add auxiliary info about policy outputs etc."""
        return AttrDict()

    def sync_networks(self):
        if self.policy.has_trainable_params:
            sync_networks(self.policy)

    def _preprocess_experience(self, experience_batch):
        """Optionally pre-process experience before it is used for policy training."""
        return experience_batch

    def _process_pre_augmentation(self, experience_batch):
        if len(experience_batch.observation.shape) == 4:
            experience_batch.observation = np.array(experience_batch.observation)
            experience_batch.observation_next = np.array(experience_batch.observation_next)
            pre_image_size = experience_batch.observation.shape[2]
            for aug, func in self.aug_fcns.items():
                if 'crop' in aug or 'cutout' in aug:
                    experience_batch.observation = func(experience_batch.observation)
                    experience_batch.observation_next = func(experience_batch.observation_next)
                elif 'translate' in aug:
                    og_obs = center_crop_images(experience_batch.observation, pre_image_size)
                    og_obs_next = center_crop_images(experience_batch.observation_next, pre_image_size)
                    obs, rndm_idxs = func(og_obs, self._hp.policy_params.input_res, return_random_idxs=True)
                    obs_next = func(og_obs_next, self._hp.policy_params.input_res, **rndm_idxs)
                    experience_batch.observation = obs
                    experience_batch.observation_next = obs_next
        return experience_batch

    def _process_post_augmentation(self, experience_batch):
        """tensor object as input"""
        for aug, func in self.aug_fcns.items():
            # skip crop and cutout augs
            if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
                continue
            experience_batch.observation = func(experience_batch.observation)
            experience_batch.observation_next = func(experience_batch.observation_next)
        return experience_batch

