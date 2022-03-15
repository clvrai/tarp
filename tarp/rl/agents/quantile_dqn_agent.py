import torch
import numpy as np

from tarp.rl.components.agent import BaseAgent
from tarp.utils.general_utils import ParamDict, map_dict, AttrDict
from tarp.utils.pytorch_utils import ten2ar, avg_grad_norm, TensorModule, check_shape, map2torch, map2np, ar2ten
import tarp.utils.data_aug as aug

class QuantileDQNAgent(BaseAgent):
    """Implements a flat (non-hierarchical) agent."""
    def __init__(self, config):
        super().__init__(config)
        self._hp = self._default_hparams().overwrite(config)
        self._eps = self._hp.init_epsilon

        # set up critic and optimizer
        self._hp.critic_params.output_dim = self._hp.critic_params.action_dim * self._hp.num_quant
        self.critic = self._hp.critic(self._hp.critic_params)
        self.critic_opt = self._get_optimizer(self._hp.optimizer, self.critic, self._hp.critic_lr)
        self.replay_buffer = self._hp.replay(self._hp.replay_params)
        self._update_steps = 0
        self.aug_fcns = self._hp.augmentation if self._hp.augmentation is not None else AttrDict(no_aug=aug.no_aug)

        # set up target network
        if self._hp.use_target_network:
            self.critic_target = self._hp.critic(self._hp.critic_params)
            self._copy_to_target_network(self.critic_target, self.critic)

    def _default_hparams(self):
        default_dict = ParamDict({
            'critic': None,     # Q-network class
            'critic_params': None,  # parameters for the Q-network class
            'critic_lr': 3e-4,  # learning rate for Q-network update
            'init_epsilon': 0.001,     # for epsilon-greedy exploration
            'epsilon_decay': 0.0,   # per-step reduction of epsilon
            'min_epsilon': 0.001,     # minimal epsilon value
            'use_target_network': True,     # if True, uses target network for computing target value
            'target_update_period': 2000,
            'num_quant': 200,
            'kappa': 1.0,
            'update_period': 4,
            'augmentation': None,                   # augmentation, e.g. random shift
        })
        return super()._default_hparams().overwrite(default_dict)

    def _act(self, obs):
        """Predicts Q-value for all actions in the current state and returns action as the argmax of these."""
        assert len(obs.shape) == 1 or len(obs.shape) == 3
        obs = map2torch(self._obs_normalizer(obs), self._hp.device)
        critic_outputs = self.critic(obs[None])

        # epsilon-greedy exploration
        if self._is_train and np.random.uniform() < self._eps:
            critic_outputs.action = self._sample_rand_action(critic_outputs.q)
        else:
            critic_outputs = critic_outputs
            critic_outputs.action = torch.argmax(critic_outputs.q.view(-1, self._hp.critic_params.action_dim, self._hp.num_quant).mean(2), dim=-1)

        return map2np(map_dict(lambda x: x[0] if isinstance(x, torch.Tensor) else x, critic_outputs))

    def _act_rand(self, obs):
        obs = map2torch(self._obs_normalizer(obs), self._hp.device)
        critic_outputs = self.critic(obs[None])
        critic_outputs.action = self._sample_rand_action(critic_outputs.q)
        critic_outputs =  map_dict(lambda x: x[0] if isinstance(x, torch.Tensor) else x, critic_outputs)
        return map2np(critic_outputs)

    def add_experience(self, experience_batch):
        """Adds experience to replay buffer (used during warmup)."""
        self.replay_buffer.append(experience_batch)
        self._obs_normalizer.update(experience_batch.observation)

    def update(self, experience_batch):
        """Updates Q-network."""
        if experience_batch:
            self.replay_buffer.append(experience_batch)
            self._obs_normalizer.update(experience_batch.observation)

        for _ in range(self._hp.update_iterations):
            if self._update_steps % self._hp.update_period == 0:
                experience_batch = self.replay_buffer.sample(n_samples=self._hp.batch_size)
                experience_batch = self._process_pre_augmentation(experience_batch)
                experience_batch = map2torch(experience_batch, self._hp.device)
                experience_batch = self._process_post_augmentation(experience_batch)

                q_est, critic_output = self._compute_q_estimate(experience_batch)
                q_target = self._compute_q_target(experience_batch)

                critic_loss, loss_info = self._compute_critic_loss(experience_batch, q_est, q_target, critic_output)
                aux_loss = self._compute_aux_loss(experience_batch)


                # update critic
                self._perform_update(critic_loss, self.critic_opt, self.critic)

                # update target network
                if self._hp.use_target_network and self._update_steps % self._hp.target_update_period == 0:
                    self._copy_to_target_network(self.critic_target, self.critic)


                # logging
                info = AttrDict(  # losses
                    # quantile_loss=quantile_huber_loss,
                    # min_q_loss=min_q_loss,
                    critic_loss=critic_loss
                )
                info.update(loss_info)
                info.update(aux_loss)
                # info.update(AttrDict(  # misc
                #     q_target=q_target,
                #     q_est=q_est,
                # ))
                info = map_dict(ten2ar, info)
            else:
                info = AttrDict()
            # update epsilon
            self._update_steps += 1
            self._update_eps()
            return info

    def _compute_q_estimate(self, experience_batch):
        critic_output = self.critic(experience_batch.observation)
        critic_output.q = critic_output.q.view(-1, self._hp.critic_params.action_dim, self._hp.num_quant)

        # get current q estimate for executed action
        one_hot_action = torch.eye(self._hp.critic_params.action_dim, device=self._hp.device)[experience_batch.action.type(torch.long)].squeeze(1)
        q_est = (critic_output.q * one_hot_action.unsqueeze(-1).repeat(1, 1, self._hp.num_quant)).sum(dim=1)
        return q_est, critic_output

    def _compute_q_target(self, experience_batch):
        # compute target q value
        with torch.no_grad():
            critic_output_next = self.critic_target(experience_batch.observation_next) if self._hp.use_target_network \
                                    else self.critic_target(experience_batch.observation_next)
            critic_output_next.q = critic_output_next.q.view(-1, self._hp.critic_params.action_dim, self._hp.num_quant)
            q_next = critic_output_next.q[np.arange(critic_output_next.q.shape[0]), critic_output_next.q.mean(2).max(1)[1]]
            # q_next = critic_output_next.q.max(dim=1)[0]
            q_target = experience_batch.reward.reshape(-1, 1) + (1 - experience_batch.done.reshape(-1, 1)) * self._hp.discount_factor * q_next
            q_target = q_target.detach()
        return q_target

    def _compute_critic_loss(self, experience_batch, q_est, q_target, critic_output):
        # compute critic loss
        diff = q_target[:, None, :] - q_est[:, :, None]
        tau = ar2ten((np.arange(self._hp.num_quant)+0.5)/self._hp.num_quant, self._hp.device)
        huber_loss = torch.where(torch.abs(diff) <= self._hp.kappa, 0.5*diff.pow(2), self._hp.kappa*(torch.abs(diff)-0.5*self._hp.kappa))
        quantile_huber_loss = huber_loss * torch.abs(tau[None, :, None]-(diff.detach() < 0).float())
        quantile_huber_loss = quantile_huber_loss.mean(2).sum(1).mean()
        info = AttrDict(
            quantile_huber_loss=quantile_huber_loss
        )
        return quantile_huber_loss, info

    def _compute_aux_loss(self, experience_batch):
        losses = AttrDict()
        return losses


    def _update_eps(self):
        """Reduce epsilon for epsilon-greedy exploration."""
        self._eps = max(self._hp.min_epsilon, self._eps - self._hp.epsilon_decay)

    def _sample_rand_action(self, q):
        batch, n_actions = q.shape
        n_actions = int(n_actions//self._hp.num_quant)
        assert batch == 1       # TODO implement eps greedy exploration for batched act()
        rand_action = np.random.choice(n_actions, batch)
        return torch.tensor(rand_action, device=self._hp.device, dtype=torch.int64)

    def dummy_output(self):
        dummy_output = self.critic.dummy_output()
        dummy_output.update(AttrDict(action=None))
        return dummy_output

    def _process_pre_augmentation(self, experience_batch):
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



class MultiHeadQuantileDQN(QuantileDQNAgent):
    def __init__(self, config):
        super().__init__(config)
        self.task_name_to_id = AttrDict()
        for i, key in enumerate(self._hp.critic_params.head_keys):
            self.task_name_to_id[key] = i

    def _act(self, obs):
        """Predicts Q-value for all actions in the current state and returns action as the argmax of these."""
        assert len(obs.shape) == 1 or len(obs.shape) == 3
        obs = map2torch(self._obs_normalizer(obs), self._hp.device)
        critic_outputs = self.critic(obs[None])

        # epsilon-greedy exploration
        if self._is_train and np.random.uniform() < self._eps:
            critic_outputs.action = self._sample_rand_action(critic_outputs.q)
            return map_dict(lambda x: x[0] if isinstance(x, torch.Tensor) else x, critic_outputs)
        else:
            critic_outputs = critic_outputs
            critic_outputs.action = {name: torch.argmax(critic_outputs.q[name].view(-1, self._hp.critic_params.action_dim,
                                                                                    self._hp.num_quant).mean(2), dim=-1) for name in critic_outputs.q.keys()}
            critic_outputs.action = map_dict(lambda x: x[0] if isinstance(x, torch.Tensor) else x, critic_outputs.action)
            critic_outputs.q = map_dict(lambda x: x[0] if isinstance(x, torch.Tensor) else x, critic_outputs.q)
            return critic_outputs

    def _compute_q_target(self, experience_batch):
        # compute target q value
        with torch.no_grad():
            critic_output_next = self.critic_target(experience_batch.observation_next) if self._hp.use_target_network \
                                    else self.critic_target(experience_batch.observation_next)
            critic_output_next.q = {name: critic_output_next.q[name].view(-1, self._hp.critic_params.action_dim, self._hp.num_quant) for name in critic_output_next.q.keys()}
            q_next = {name: critic_output_next.q[name][np.arange(self._hp.batch_size), critic_output_next.q[name].mean(2).max(1)[1]] for name in critic_output_next.q.keys()}
            # q_next = critic_output_next.q.max(dim=1)[0]
            q_target = {name: experience_batch.reward + (1 - experience_batch.done) * self._hp.discount_factor * q_next[name] for name in critic_output_next.q.keys()}
            q_target = {name: q_target[name].detach() for name in critic_output_next.q.keys()}
        return q_target

    def _compute_q_estimate(self, experience_batch):
        critic_output = self.critic(experience_batch.observation)
        critic_output.q = {name: critic_output.q[name].view(-1, self._hp.critic_params.action_dim, self._hp.num_quant) for name in critic_output.q.keys()}
        # critic_output.q = critic_output.q.view(-1, self._hp.critic_params.action_dim, self._hp.num_quant)

        # get current q estimate for executed action
        one_hot_action = torch.eye(self._hp.critic_params.action_dim, device=self._hp.device)[experience_batch.action.type(torch.long)].squeeze(1)
        q_est = {name: (critic_output.q[name] * one_hot_action.unsqueeze(-1).repeat(1, 1, self._hp.num_quant)).sum(dim=1) for name in critic_output.q.keys()}
        return q_est, critic_output

    def _compute_critic_loss(self, experience_batch, q_est, q_target, critic_output):
        # compute critic loss
        info = AttrDict()
        diff = {name: q_target[name][:, None, :] - q_est[name][:, :, None] for name in q_est.keys()}
        tau = ar2ten((np.arange(self._hp.num_quant)+0.5)/self._hp.num_quant, self._hp.device)
        huber_loss = {name: torch.where(torch.abs(diff[name]) <= self._hp.kappa, 0.5*diff[name].pow(2), self._hp.kappa*(torch.abs(diff[name])-0.5*self._hp.kappa)) for name in q_est.keys()}
        quantile_huber_loss = {name: huber_loss[name] * torch.abs(tau[None, :, None]-(diff[name].detach() < 0).float()) for name in q_est.keys()}
        quantile_huber_loss = {name: quantile_huber_loss[name][experience_batch.task_ids.type(torch.long)==self.task_name_to_id[name]].mean(2).sum(1).mean() for i, name in enumerate(q_est.keys())}
        quantile_huber_loss_info = AttrDict()
        for name in q_est.keys():
            quantile_huber_loss_info[name+'_quantile_huber_loss'] = quantile_huber_loss[name]
        info.update(quantile_huber_loss_info)
        return quantile_huber_loss, info
