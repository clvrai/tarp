import torch
import os
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from tarp.rl.agents.ac_agent import ACAgent
from tarp.modules.subnetworks import Decoder
from tarp.modules.layers import LayerBuilderParams
from tarp.utils.general_utils import ParamDict, map_dict, AttrDict, listdict2dictlist, dictlist2listdict
from tarp.utils.pytorch_utils import ten2ar, avg_grad_norm, TensorModule, check_shape, map2torch, map2np
from tarp.rl.utils.mpi import sync_networks
from tarp.utils.data_aug import random_shift, subpixel_shift
import kornia


class SACAgent(ACAgent):
    """Implements SAC update for a non-hierarchical agent."""
    def __init__(self, config):
        super().__init__(config)
        self._hp = self._default_hparams().overwrite(config)

        # build critics and target networks, copy weights of critics to target networks
        self.critics = torch.nn.ModuleList([self._hp.critic(self._hp.critic_params) for _ in range(2)])
        self.critic_targets = torch.nn.ModuleList([self._hp.critic(self._hp.critic_params) for _ in range(2)])
        [self._copy_to_target_network(target, source) for target, source in zip(self.critics, self.critic_targets)]

        # build optimizers for critics
        self.critic_opts = [self._get_optimizer(self._hp.optimizer, critic, self._hp.critic_lr) for critic in self.critics]

        # define entropy multiplier alpha
        self._log_alpha = TensorModule(torch.zeros(1, requires_grad=True, device=self._hp.device))
        self.alpha_opt = self._get_optimizer(self._hp.optimizer, self._log_alpha, self._hp.alpha_lr)
        self._target_entropy = self._hp.target_entropy if self._hp.target_entropy is not None \
                                        else -1 * self._hp.policy_params.action_dim

        # build replay buffer
        self.replay_buffer = self._hp.replay(self._hp.replay_params)

        self._update_steps = 0                # counts the number of alpha updates for optional variable schedules

    def _default_hparams(self):
        default_dict = ParamDict({
            'critic': None,           # critic class
            'critic_params': None,    # parameters for the critic class
            'replay': None,           # replay buffer class
            'replay_params': None,    # parameters for replay buffer
            'critic_lr': 3e-4,        # learning rate for critic update
            'alpha_lr': 3e-4,
            'reward_scale': 1.0,      # SAC reward scale
            'clip_q_target': False,   # if True, clips Q target
            'target_entropy': None,   # target value for automatic entropy tuning, if None uses -action_dim
            'actor_update_freq': 1,
            'critic_target_update_freq': 1,
            'retain_backprop_graph': False,
            "reset_alpha": False,
            'deterministic_backup': False,
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
            experience_batch = self.replay_buffer.sample(n_samples=self._hp.batch_size)
            experience_batch.observation = self._obs_normalizer(experience_batch.observation)
            experience_batch.observation_next = self._obs_normalizer(experience_batch.observation_next)
            experience_batch = self._process_pre_augmentation(experience_batch)
            experience_batch = map2torch(experience_batch, self._hp.device)
            experience_batch = self._process_post_augmentation(experience_batch)
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

    def add_experience(self, experience_batch):
        """Adds experience to replay buffer (used during warmup)."""
        self.replay_buffer.append(experience_batch)
        self._obs_normalizer.update(experience_batch.observation)

    def _update_info(self, info, policy_output, q_target, qs, experience_batch):
        info.update(AttrDict(       # misc
            alpha=self.alpha,
            pi_log_prob=policy_output.log_prob.mean(),
            policy_entropy=policy_output.dist.entropy().mean(),
            q_target=q_target.mean(),
            q_1=qs[0].mean(),
            q_2=qs[1].mean(),
        ))
        return info

    def _run_policy(self, obs):
        """Allows child classes to post-process policy outputs."""
        return self.policy(obs)

    def _compute_alpha_loss(self, policy_output, experience_batch):
        return -1 * (self.alpha * (self._target_entropy + policy_output.log_prob).detach()).mean()

    def _compute_policy_loss(self, experience_batch, policy_output):
        q_est = torch.min(*[critic(experience_batch.observation, self._prep_action(policy_output.action)).q
                                      for critic in self.critics])
        policy_loss = -1 * q_est + self.alpha * policy_output.log_prob[:, None]
        check_shape(policy_loss, [self._hp.batch_size, 1])
        return policy_loss.mean()

    def _compute_next_value(self, experience_batch, policy_output):
        q_next = torch.min(*[critic_target(experience_batch.observation_next, self._prep_action(policy_output.action)).q
                             for critic_target in self.critic_targets])
        if not self._hp.deterministic_backup:
            next_val = (q_next - self.alpha * policy_output.log_prob[:, None])
            check_shape(next_val, [self._hp.batch_size, 1])
        else:
            next_val = q_next
        return next_val.squeeze(-1)

    def _compute_q_estimates(self, experience_batch):
        return [critic(experience_batch.observation, self._prep_action(experience_batch.action.detach())).q.squeeze(-1)
                    for critic in self.critics]     # no gradient propagation into policy here!

    def _compute_target_q(self, experience_batch):
        # compute target Q value
        with torch.no_grad():
            policy_output_next = self._run_policy(experience_batch.observation_next)
            value_next = self._compute_next_value(experience_batch, policy_output_next)
            q_target = experience_batch.reward * self._hp.reward_scale + \
                            (1 - experience_batch.done) * self._hp.discount_factor * value_next
            if self._hp.clip_q_target:
                q_target = self._clip_q_target(q_target)
            q_target = q_target.detach()
            check_shape(q_target, [self._hp.batch_size])
        return q_target

    def _compute_critic_loss(self, experience_batch, q_target):
        # compute critic loss
        qs = self._compute_q_estimates(experience_batch)
        check_shape(qs[0], [self._hp.batch_size])
        critic_losses = [0.5 * (q - q_target).pow(2).mean() for q in qs]
        return critic_losses, qs

    def _update_critics(self, critic_losses):
        # update critic networks
        [self._perform_update(critic_loss, critic_opt, critic, retain_graph=self._hp.retain_backprop_graph)
                for critic_loss, critic_opt, critic in zip(critic_losses, self.critic_opts, self.critics)]

    def _prep_action(self, action):
        """Preprocessing of action in case of discrete action space."""
        if len(action.shape) == 1: action = action[:, None]  # unsqueeze for single-dim action spaces
        return action.float()

    def _clip_q_target(self, q_target):
        clip = 1 / (1 - self._hp.discount_factor)
        return torch.clamp(q_target, -clip, clip)

    def _aux_info(self, policy_output):
        """Optionally add auxiliary info about policy outputs etc."""
        return AttrDict()


    def sync_networks(self):
        super().sync_networks()
        [sync_networks(critic) for critic in self.critics]
        sync_networks(self._log_alpha)

    def state_dict(self, *args, **kwargs):
        d = super().state_dict()
        d['critic_opts'] = [o.state_dict() for o in self.critic_opts]
        d['alpha_opt'] = self.alpha_opt.state_dict()
        return d

    def reset_alpha(self):
        # define entropy multiplier alpha
        self._log_alpha = TensorModule(torch.zeros(1, requires_grad=True, device=self._hp.device))
        self.alpha_opt = self._get_optimizer(self._hp.optimizer, self._log_alpha, self._hp.alpha_lr)

    def load_state_dict(self, state_dict, *args, **kwargs):
        [o.load_state_dict(d) for o, d in zip(self.critic_opts, state_dict.pop('critic_opts'))]
        self.alpha_opt.load_state_dict(state_dict.pop('alpha_opt'))
        super().load_state_dict(state_dict, *args, **kwargs)
        if self._hp.reset_alpha:
            self.reset_alpha()

    def save_state(self, save_dir):
        """Saves compressed replay buffer to disk."""
        self.replay_buffer.save(os.path.join(save_dir, 'replay'))

    def load_state(self, save_dir):
        """Loads replay buffer from disk."""
        self.replay_buffer.load(os.path.join(save_dir, 'replay'))

    @property
    def alpha(self):
        return self._log_alpha().exp()

    @property
    def schedule_steps(self):
        return self._update_steps

    # @property
    # def critic(self):
    #     return self.critics[0]

class DiscreteSACAgent(SACAgent):
    """SAC agent for discrete action spaces."""
    # TODO: did not update this!
    def _compute_policy_loss(self, experience_batch, policy_output):
        q_est = torch.min(*[critic(experience_batch.observation).q for critic in self.critics])
        entropy = policy_output.output_dist.entropy()
        policy_loss = -1 * (policy_output.output_dist.probs * q_est).sum(-1) - self.alpha * entropy
        return policy_loss.mean()

    def _compute_next_value(self, experience_batch, policy_output):
        q_next = torch.min(*[critic_target(experience_batch.observation_next).q for critic_target in self.critic_targets])
        return (policy_output.output_dist.probs * q_next).sum(-1) \
                    - self.alpha * policy_output.output_dist.entropy()

    def _compute_q_estimates(self, experience_batch):
        return [critic(experience_batch.observation).q.gather(-1, experience_batch.action[:, None]).squeeze(-1)
                    for critic in self.critics]


class NoUpdateACAgent(ACAgent):
    def update(self, experience_batch):
        return {}


class AuxLossSACAgent(SACAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_aux_loss = np.nan

    def _default_hparams(self):
        default_dict = ParamDict({
            'aux_update_prob': 1.0,     # update probability for auxiliary loss
        })
        return super()._default_hparams().overwrite(default_dict)

    def update(self, experience_batch):
        info = super().update(experience_batch)
        if np.random.rand() < self._hp.aux_update_prob:
            # compute auxiliary loss if agent has one defined
            auxiliary_loss_dict = self._get_auxiliary_loss_dict()
            auxiliary_loss = auxiliary_loss_dict.total.value if len(auxiliary_loss_dict) > 0 else 0

            # update policy network on auxiliary loss
            if len(auxiliary_loss_dict) > 0:
                self._perform_update(auxiliary_loss, self.policy_opt, self.policy)
                self._last_aux_loss = auxiliary_loss

        info.update(AttrDict(auxiliary_loss=self._last_aux_loss,))
        return info

    def _get_auxiliary_loss_dict(self):
        return AttrDict()

class SACAEAgent(SACAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # build critics and target networks, copy weights of critics to target networks
        self.critics = torch.nn.ModuleList([self._hp.critic(self._hp.critic_params) for _ in range(1)])
        self.critic_targets = torch.nn.ModuleList([self._hp.critic(self._hp.critic_params) for _ in range(1)])
        [self._copy_to_target_network(target, source) for target, source in zip(self.critics, self.critic_targets)]

        # build optimizers for critics
        self.critic_opts = [self._get_optimizer(self._hp.optimizer, critic, self._hp.critic_lr) for critic in self.critics]
        self._hp.decoder_params.builder = LayerBuilderParams(self._hp.use_convs, self._hp.normalization)
        self._hp.decoder_params.overwrite(self._hp.policy_params)
        self._hp.decoder_params.img_sz = self._hp.decoder_params.input_res
        self.decoder = Decoder(self._hp.decoder_params)
        self.decoder_opt = self._get_optimizer(self._hp.optimizer, self.decoder, self._hp.decoder_lr)
        self.encoder_opt = self._get_optimizer(self._hp.optimizer, self.critics[0].encoder, self._hp.encoder_lr)

        self.policy.encoder.tie_conv_from(self.critics[0].encoder)

    def _default_hparams(self):
        default_dict = ParamDict({
            'decoder_params': ParamDict({
                'pixel_shift_decoder': False,
                'add_weighted_pixel_copy': False,
                'use_skips': False,
                'ngf': 4,
            }),
            'decoder_lr': 3e-4,
            'decoder_latent_lambda': 0.0,
            'encoder_lr': 3e-4,
            'use_convs': True,
            'normalization': 'batch',
            'decoder_update_freq': 1
            # 'skips_stride': 2
        })
        return super()._default_hparams().overwrite(default_dict)

    def _compute_q_estimates(self, experience_batch):
        qs = self.critics[0](experience_batch.observation, self._prep_action(experience_batch.action.detach()))
        q1, q2 = qs.q1.squeeze(-1), qs.q2.squeeze(-1)
        return [q1, q2]

    def _compute_decoder_loss(self, experience_batch):
        h  = self.critics[0].encode(experience_batch.observation)
        rec_obs = self.decoder(h).images
        rec_loss = F.mse_loss(experience_batch.observation,
                              rec_obs)
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        loss = rec_loss + self._hp.decoder_latent_lambda * latent_loss
        return loss, rec_obs[0], experience_batch.observation[0]

    def _run_policy(self, obs):
        return self.policy(obs)

    def _compute_policy_loss(self, experience_batch, policy_output):
        qs = self.critics[0](experience_batch.observation, self._prep_action(policy_output.action), detach_encoder=True)
        q1, q2 = qs.q1, qs.q2
        q_est = torch.min(q1, q2)
        policy_loss = -1 * q_est + self.alpha * policy_output.log_prob[:, None]
        check_shape(policy_loss, [self._hp.batch_size, 1])
        return policy_loss.mean()

    def _compute_critic_loss(self, experience_batch, q_target):
        # compute critic loss
        qs = self._compute_q_estimates(experience_batch)
        check_shape(qs[0], [self._hp.batch_size])
        critic_losses = [0.5 * (q - q_target).pow(2).mean() for q in qs]
        return [critic_losses[0], critic_losses[1]], qs

    def _compute_next_value(self, experience_batch, policy_output):
        qs = self.critic_targets[0](experience_batch.observation, self._prep_action(policy_output.action))
        q_next = torch.min(*[qs.q1, qs.q2])
        next_val = (q_next - self.alpha * policy_output.log_prob[:, None])
        check_shape(next_val, [self._hp.batch_size, 1])
        return next_val.squeeze(-1)

    def _update_critics(self, critic_losses):
        # update critic networks
        self._perform_update(critic_losses[0]+critic_losses[1], self.critic_opts[0], self.critics[0], retain_graph=True)

    def update(self, experience_batch):
        """Updates actor and critics."""
        # push experience batch into replay buffer
        self.replay_buffer.append(experience_batch)
        self._obs_normalizer.update(experience_batch.observation)

        for _ in range(self._hp.update_iterations):
            # sample batch and normalize
            experience_batch = self.replay_buffer.sample(n_samples=self._hp.batch_size)
            experience_batch.observation = self._obs_normalizer(experience_batch.observation)
            experience_batch.observation_next = self._obs_normalizer(experience_batch.observation_next)
            experience_batch = map2torch(experience_batch, self._hp.device)

            policy_output = self._run_policy(experience_batch.observation)



            q_target = self._compute_target_q(experience_batch)
            critic_losses, qs = self._compute_critic_loss(experience_batch, q_target)

            # update policy network on policy loss
            if self._update_steps % self._hp.actor_update_freq == 0:
                # update alpha
                alpha_loss = self._compute_alpha_loss(policy_output, experience_batch)
                self._perform_update(alpha_loss, self.alpha_opt, self._log_alpha, retain_graph=True)

                # compute policy loss
                policy_loss = self._compute_policy_loss(experience_batch, policy_output)
                self._perform_update(policy_loss, self.policy_opt, self.policy, retain_graph=True)

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

            info.update(AttrDict(       # misc
                alpha=self.alpha,
                pi_log_prob=policy_output.log_prob.mean(),
                policy_entropy=policy_output.dist.entropy().mean(),
                q_target=q_target.mean(),
                q_1=qs[0].mean(),
                q_2=qs[1].mean(),
            ))

            info.update(self._aux_info(policy_output))

            if self._update_steps % self._hp.decoder_update_freq == 0:
                decoder_loss, rec_ob, ob = self._compute_decoder_loss(experience_batch)
                self.encoder_opt.zero_grad()
                self.decoder_opt.zero_grad()
                decoder_loss.backward()
                self.encoder_opt.step()
                self.decoder_opt.step()

                info.update(AttrDict(
                    decoder_loss=decoder_loss,
                    rec_ob=rec_ob,
                    ob=ob
                ))

            info = map_dict(ten2ar, info)
            self._update_steps += 1

        return info

    def log_outputs(self, log_outputs, rollout_storage, logger, log_images, step):
        ob, rec_ob = None, None
        if 'ob' in log_outputs.keys():
            ob = log_outputs.pop('ob')
            rec_ob = log_outputs.pop('rec_ob')
        super().log_outputs(log_outputs, rollout_storage, logger, log_images, step)

        if ob is not None:
            ob = ob.transpose((1, 2, 0))
            rec_ob = rec_ob.transpose((1, 2, 0))

            ch = ob.shape[2] // self._hp.n_frames
            if ch == 1:
                ob = np.expand_dims(ob[:, :, 0], axis=2)
                rec_ob = np.expand_dims(rec_ob[:, :, 0], axis=2)
            else:
                ob = ob[:, :, :ch]
                rec_ob = rec_ob[:, :, ch]

            logger.log_image(ob, 'gt_ob', step=step)
            logger.log_image(rec_ob, 'rec_ob', step=step)


class MultiHeadSACAgent(SACAgent):
    def __init__(self, config):
        super().__init__(config)
        self._log_alpha = nn.ModuleDict({key: TensorModule(torch.zeros(1, requires_grad=True, device=self._hp.device)) for key in self._hp.policy_params.head_keys})
        # self.alpha_opt = {self._get_optimizer(self._hp.optimizer, self._log_alpha[key], self._hp.alpha_lr) for key in self._hp.policy_params.head_keys}
        self.alpha_opt = self._get_optimizer(self._hp.optimizer, self._log_alpha, self._hp.alpha_lr)
        self.task_name_to_id = AttrDict()
        torch.cuda.empty_cache()
        for i, key in enumerate(self._hp.policy_params.head_keys):
            self.task_name_to_id[key] = i

    def update(self, experience_batch):
        """Updates actor and critics."""
        # push experience batch into replay buffer
        if experience_batch:
            self.replay_buffer.append(experience_batch)
            self._obs_normalizer.update(experience_batch.observation)

        for _ in range(self._hp.update_iterations):
            # sample batch and normalize
            experience_batch = self.replay_buffer.sample(n_samples=self._hp.batch_size)
            experience_batch.observation = self._obs_normalizer(experience_batch.observation)
            experience_batch.observation_next = self._obs_normalizer(experience_batch.observation_next)
            experience_batch = self._process_pre_augmentation(experience_batch)
            experience_batch = map2torch(experience_batch, self._hp.device)
            experience_batch = self._process_post_augmentation(experience_batch)
            policy_output = self._run_policy(experience_batch.observation)

            q_target = self._compute_target_q(experience_batch)
            critic_losses, qs, min_qf_losses_per_task = self._compute_critic_loss(experience_batch, q_target)

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

            for i in range(len(min_qf_losses_per_task)):
                for key in min_qf_losses_per_task[i].keys():
                    info['min_qf_loss{}_'.format(i)+key] = min_qf_losses_per_task[i][key]

            info = self._update_info(info, policy_output, q_target, qs, experience_batch)

            info.update(self._aux_info(policy_output))
            info = map_dict(ten2ar, info)
            self._update_steps += 1

            return info

    def _compute_policy_loss(self, experience_batch, policy_output):
        q_values = listdict2dictlist([critic(experience_batch.observation, self._prep_action(policy_output.action)).q for critic in self.critics])
        q_est = {key: torch.min(*val) for key, val in q_values.items()}
        policy_loss = {key: -1 * q_est[key] + self.alpha[key] * policy_output.log_prob[key][:, None] for key in policy_output.log_prob.keys()}
        map_dict(lambda x: check_shape(x, [self._hp.batch_size, 1]), policy_loss)
        policy_loss = {key: policy_loss[key][experience_batch.task_ids==self.task_name_to_id[key]].mean() for i, key in enumerate(policy_loss.keys())}
        return torch.stack(list(policy_loss.values())).mean()

    def _compute_next_value(self, experience_batch, policy_output):
        q_values = listdict2dictlist([critic_target(experience_batch.observation_next, self._prep_action(policy_output.action)).q for critic_target in self.critic_targets])
        q_next = {key: torch.min(*val) for key, val in q_values.items()}

        if not self._hp.deterministic_backup:
            next_val = {key: q_next[key] - self.alpha[key] * policy_output.log_prob[key][:, None] for name in q_next.keys()}
            map_dict(lambda x: check_shape(x, [self._hp.batch_size, 1]), next_val)
        else:
            next_val = q_next
        return map_dict(lambda x:x.squeeze(-1), next_val)

    def _compute_q_estimates(self, experience_batch):
        q_values = [critic(experience_batch.observation, self._prep_action(experience_batch.action.detach())).q for critic in self.critics]
        q_values = [map_dict(lambda x:x.squeeze(-1), q) for q in q_values]
        return listdict2dictlist(q_values)

    def _compute_target_q(self, experience_batch):
        # compute target Q value
        with torch.no_grad():
            policy_output_next = self._run_policy(experience_batch.observation_next)
            value_next = self._compute_next_value(experience_batch, policy_output_next)
            q_target = {key: experience_batch.reward * self._hp.reward_scale + \
                            (1 - experience_batch.done) * self._hp.discount_factor * value_next[key] for key in value_next.keys()}
            if self._hp.clip_q_target:
                q_target = self._clip_q_target(q_target)
            q_target = map_dict(lambda x:x.detach(), q_target)
            map_dict(lambda x:check_shape(x, [self._hp.batch_size]), q_target)
        return q_target

    def _compute_critic_loss(self, experience_batch, q_target):
        # compute critic loss
        qs = self._compute_q_estimates(experience_batch)
        map_dict(lambda x: check_shape(x[0], [self._hp.batch_size]), qs)
        critic_losses = {key: [0.5 * (q[experience_batch.task_ids==self.task_name_to_id[key]] - q_target[key][experience_batch.task_ids==self.task_name_to_id[key]]).pow(2).mean() for q in qs[key]] for i, key in enumerate(qs.keys())}
        critic_losses = dictlist2listdict(critic_losses)
        critic_losses = [torch.stack(list(critic_loss.values())).mean() for critic_loss in critic_losses]
        return critic_losses, dictlist2listdict(qs)

    def _clip_q_target(self, q_target):
        clip = 1 / (1 - self._hp.discount_factor)
        return {key: torch.clamp(q_target[key], -clip, clip) for key in q_target.keys()}

    def _prep_action(self, action):
        """Preprocessing of action in case of discrete action space."""
        if isinstance(action, dict):
            action = {key: action[key][:, None].float() if len(action[key].shape) == 1 else action[key].float() for key in action.keys()}
        else:
            if len(action.shape) == 1: action = action[:, None]  # unsqueeze for single-dim action spaces
            action = action.float()
        return action

    def _compute_alpha_loss(self, policy_output, experience_batch):
        return torch.mean(torch.stack([-1 * (self.alpha[key] * (self._target_entropy + policy_output.log_prob[key][experience_batch.task_ids==self.task_name_to_id[key]]).detach()).mean() for i, key in enumerate(policy_output.log_prob.keys())]))

    def _act(self, obs):
        agent_output = super()._act(obs)
        agent_output.action = map_dict(lambda x:x[0], agent_output.action)
        agent_output.log_prob = map_dict(lambda x:x[0], agent_output.log_prob)
        return agent_output

    def _update_info(self, info, policy_output, q_target, qs, experience_batch):
        for i, key in enumerate(q_target.keys()):
            info['alpha_'+key] = self.alpha[key]
            info['pi_log_prob_'+key] = policy_output.log_prob[key][experience_batch.task_ids==self.task_name_to_id[key]].mean()
            info['policy_entropy_'+key] = policy_output.dist[key].entropy()[experience_batch.task_ids==self.task_name_to_id[key]].mean()
            info['q_target_'+key] = q_target[key][experience_batch.task_ids==self.task_name_to_id[key]].mean()
            info['q_1_'+key] = qs[0][key][experience_batch.task_ids==self.task_name_to_id[key]].mean()
            info['q_2_'+key] = qs[1][key][experience_batch.task_ids==self.task_name_to_id[key]].mean()
        return info

    @property
    def alpha(self):
        return {key: self._log_alpha[key]().exp() for key in self._log_alpha.keys()}

