import torch
import os
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from tarp.rl.agents.sac_agent import SACAgent
from tarp.modules.subnetworks import Decoder
from tarp.modules.layers import LayerBuilderParams
from tarp.utils.general_utils import ParamDict, map_dict, AttrDict, listdict2dictlist, dictlist2listdict
from tarp.utils.pytorch_utils import ten2ar, avg_grad_norm, TensorModule, check_shape, map2torch, map2np
from tarp.rl.utils.mpi import sync_networks
from tarp.utils.data_aug import random_shift, subpixel_shift


class SQILAgent(SACAgent):
    """Implements SQIL agent."""
    def __init__(self, config):
        super().__init__(config)
        self.expert_replay_buffer = self._hp.replay(self._hp.expert_replay_params)
        self._hp.expert_data_conf.device = self.device.type
        self.load_expert_data()

    def _default_hparams(self):
        default_dict = ParamDict({
        })
        return super()._default_hparams().overwrite(default_dict)

    def update(self, experience_batch):
        """Updates actor and critics."""
        # push experience batch into replay buffer
        if experience_batch:
            self.replay_buffer.append(experience_batch)
            self._obs_normalizer.update(experience_batch.observation)

        for _ in range(self._hp.update_iterations):
            # sample batch and normalize
            experience_batch = self.replay_buffer.sample(n_samples=int(self._hp.batch_size//2))
            experience_batch.observation = self._obs_normalizer(experience_batch.observation)
            experience_batch.observation_next = self._obs_normalizer(experience_batch.observation_next)
            experience_batch.reward = np.zeros(experience_batch.reward.shape)

            expert_experience_batch = self.expert_replay_buffer.sample(n_samples=int(self._hp.batch_size//2))
            expert_experience_batch.observation = self._obs_normalizer(expert_experience_batch.observation)
            expert_experience_batch.observation_next = self._obs_normalizer(expert_experience_batch.observation_next)

            for key in experience_batch.keys():
                experience_batch[key] = np.concatenate([experience_batch[key], expert_experience_batch[key]])


            experience_batch = map2torch(experience_batch, self._hp.device)
            policy_output = self._run_policy(experience_batch.observation)

            q_target = self._compute_target_q(experience_batch)
            critic_losses, qs = self._compute_critic_loss(experience_batch, q_target)

            # update policy network on policy loss
            if self._update_steps % self._hp.actor_update_freq == 0:
                # update alpha
                alpha_loss = self._compute_alpha_loss(policy_output, experience_batch)
                self._perform_update(alpha_loss, self.alpha_opt, self._log_alpha)

                # compute policy loss
                policy_loss = self._compute_policy_loss(experience_batch, policy_output)
                self._perform_update(policy_loss, self.policy_opt, self.policy)

                info = AttrDict(    # losses
                    policy_loss=policy_loss,
                    alpha_loss=alpha_loss,
                )

            self._update_critics(critic_losses)

            # update target networks
            if self._update_steps % self._hp.critic_target_update_freq == 0:
                [self._soft_update_target_network(critic_target, critic)
                        for critic_target, critic in zip(self.critic_targets, self.critics)]

            # logging
            info = AttrDict(    # losses
                critic_loss_1=critic_losses[0],
                critic_loss_2=critic_losses[1],
            )

            info = self._update_info(info, policy_output, q_target, qs, experience_batch)

            info.update(self._aux_info(policy_output))
            info = map_dict(ten2ar, info)
            self._update_steps += 1

            return info

    def load_expert_data(self):
        rollout = self._hp.expert_data_conf.dataset_spec.dataset_class(
            self._hp.expert_data_path, self._hp.expert_data_conf, resolution=self._hp.expert_data_conf.dataset_spec.res,
            phase="train", shuffle=True).data_dict
        rollout.reward = np.ones(rollout.reward.shape)
        self.expert_replay_buffer.append(rollout)


