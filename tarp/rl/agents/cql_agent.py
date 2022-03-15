import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy

from tarp.rl.components.agent import BaseAgent
from tarp.rl.agents.quantile_dqn_agent import QuantileDQNAgent, MultiHeadQuantileDQN
from tarp.rl.agents.sac_agent import SACAgent, MultiHeadSACAgent
from tarp.utils.general_utils import ParamDict, map_dict, AttrDict, dictlist2listdict
from tarp.utils.pytorch_utils import ten2ar, avg_grad_norm, TensorModule, check_shape, map2torch, map2np, ar2ten
from tarp.modules.subnetworks import Predictor, Decoder
from tarp.modules.layers import LayerBuilderParams
from tarp.modules.losses import L2Loss, BCELoss, CELoss, BCEWithLogitsLoss


class DiscreteCQLAgent(QuantileDQNAgent):
    """Implements a flat (non-hierarchical) agent."""
    def __init__(self, config):
        super().__init__(config)
        if self._hp.use_aux_loss:
            self._hp.builder = LayerBuilderParams(self._hp.use_conv, self._hp.normalization)
            self._hp.nz_mid = self._hp.critic_params.nz_mid
            self._hp.nz_enc = self._hp.critic_params.nz_enc
            self.decoder_obj = nn.ModuleDict({name: Predictor(self._hp, input_size=self._hp.nz_enc,
                                                              output_size=1, spatial=False, final_activation=nn.Sigmoid()) for name in self._hp.obj_labels})
            seg_decoder_hp = deepcopy(self._hp)
            seg_decoder_hp.input_nc = self._hp.n_class
            seg_decoder_hp.dec_last_activation = None
            seg_decoder_hp.img_sz = self._hp.critic_params.input_res
            self.seg_decoder = Decoder(seg_decoder_hp)
            self.obj_opt = self._get_optimizer(self._hp.optimizer, self.decoder_obj, self._hp.critic_lr)
            self.seg_opt = self._get_optimizer(self._hp.optimizer, self.seg_decoder, self._hp.critic_lr)

    def _default_hparams(self):
        default_dict = ParamDict({
            'min_q_weight': 1.0,
            'use_aux_loss': False,
            'normalization': 'batch',
            'use_conv': True,
            'add_weighted_pixel_copy': False, # if True, adds pixel copying stream for decoder
            'pixel_shift_decoder': False,
            'use_skips': False,
            'ngf': 8,
            'update_period': 1,
        })
        return super()._default_hparams().overwrite(default_dict)

    def _compute_critic_loss(self, experience_batch, q_est, q_target, critic_output):
        loss, info = super()._compute_critic_loss(experience_batch, q_est, q_target, critic_output)
        # cql loss
        # critic_output = self.critic(experience_batch.observation)
        # critic_output.q = critic_output.q.view(-1, self._hp.critic_params.action_dim, self._hp.num_quant)
        one_hot_action = torch.eye(self._hp.critic_params.action_dim, device=self._hp.device)[experience_batch.action.type(torch.long)].squeeze(1)
        dataset_expec = (critic_output.q.mean(2) * one_hot_action).sum(1).mean()
        negative_sampling = torch.logsumexp(critic_output.q.mean(2), 1).mean()
        min_q_loss = self._hp.min_q_weight * (negative_sampling - dataset_expec)
        info.update(
            AttrDict(
                min_q_loss=min_q_loss
            )
        )
        critic_loss = loss + min_q_loss
        return critic_loss, info


    def _compute_aux_loss(self, experience_batch):
        # image reconstruction loss
        info = AttrDict()

        feat = self.critic.encode(experience_batch.observation)
        feat = feat.detach()
        output_seg = self.seg_decoder(feat)
        seg_entropy = CELoss()(output_seg.images,
                                          experience_batch.seg_targets.squeeze(1).type(torch.long)).value

        info.update(AttrDict(
            seg_entropy=seg_entropy
        ))
        obj_labels = AttrDict({name: self.decoder_obj[name](feat) for name in self._hp.obj_labels})

        idx = np.random.randint(len(experience_batch.observation))
        info.update(AttrDict(
            pred_seg=output_seg.images[idx].detach(),
            target_seg=experience_batch.seg_targets[idx].squeeze(1),
            ob=experience_batch.observation[idx]
        ))

        obj_losses = AttrDict({name: BCELoss()(obj_labels[name], experience_batch.obj_labels[:, i].unsqueeze(1)).value
                                for i, name in enumerate(self._hp.obj_labels)})
        total_obj_loss = torch.stack(list(obj_losses.values())).sum()
        info.update(obj_losses)
        self._perform_update(seg_entropy, self.seg_opt, self.seg_decoder)
        self._perform_update(total_obj_loss, self.obj_opt, self.decoder_obj)

        return info

    def log_outputs(self, log_outputs, rollout_storage, logger, log_images, step):
        ob, target_seg, pred_seg = None, None, None
        if self._hp.use_aux_loss:
            if 'ob' in log_outputs.keys():
                ob = log_outputs.pop('ob')
                target_seg = log_outputs.pop('target_seg')
                pred_seg = log_outputs.pop('pred_seg')
            super().log_outputs(log_outputs, rollout_storage, logger, log_images, step)

            if ob is not None:
                nc, h, w = pred_seg.shape
                pred_seg_img = np.zeros((h, w, 3))
                target_seg_img = np.zeros((h, w, 3))
                for c in range(nc):
                    pred_seg_img[np.argmax(pred_seg, axis=0)==c] = self._hp.color_map[c]
                    target_seg_img[target_seg.squeeze(0)==c] = self._hp.color_map[c]
                ob = ob.transpose((1, 2, 0))
                ch = ob.shape[2] // self._hp.n_frames
                if ch == 1:
                    ob = np.expand_dims(ob[:, :, 0], axis=2)
                else:
                    ob = ob[:, :, :ch]
                logger.log_image(ob, 'ob', step=step)
                logger.log_image(target_seg_img, 'gt_seg', step=step)
                logger.log_image(pred_seg_img, 'pred_seg', step=step)
        else:
            pass

class ContinuousCQLAgent(SACAgent):
    def __init__(self, config):
        super().__init__(config)
        if self._hp.with_lagrange:
            self._target_action_gap = self._hp.lagrange_thresh
            self._log_alpha_prime = TensorModule(torch.zeros(1, requires_grad=True, device=self._hp.device))
            self.alpha_prime_opt = self._get_optimizer(self._hp.optimizer, self._log_alpha_prime, self._hp.policy_lr)

    def _default_hparams(self):
        default_dict = ParamDict({
            'num_random': 10,        # number of action samples for estimating Q expectation
            'min_q_version': 3,      # if == 3: uses importance sampling for Q expectation
            'temp': 1.0,             # temperature of logsumexp operation
            'min_q_weight': 5.0,     # multiplier for Q adjustment term
            'with_lagrange': False,   # if True, autotunes the budget factor with dual optimization
            'lagrange_thresh': 5.0,  # budget factor that indicates target action gap
            'retain_backprop_graph': True,  # need to retain backprop graph during q update
            'policy_warmup_steps': 0,       # number of initial steps that policy is trained with BC
            'deterministic_backup': True,
        })
        return super()._default_hparams().overwrite(default_dict)

    def _compute_policy_loss(self, experience_batch, policy_output):
        """Optionally runs BC warmup in the beginning."""
        if self._update_steps < self._hp.policy_warmup_steps:
            return (-1 * policy_output.dist.log_prob(experience_batch.action)
                          + self.alpha * policy_output.log_prob).mean()
        else:
            return super()._compute_policy_loss(experience_batch, policy_output)

    def _compute_critic_loss(self, experience_batch, q_target):
        critic_losses, qs = super()._compute_critic_loss(experience_batch, q_target)

        # prepare tiled obs vectors and sample rand actions for importance sampling
        obs_dim = experience_batch.observation.shape[1:]
        obs = experience_batch.observation.unsqueeze(1).repeat((1, self._hp.num_random) + (1,)*len(obs_dim)).view((self._hp.batch_size*self._hp.num_random,) +
                                                                                                                  experience_batch.observation.shape[1:])
        obs_next = experience_batch.observation_next.unsqueeze(1).repeat((1, self._hp.num_random) + (1,)*len(obs_dim)).view((self._hp.batch_size*self._hp.num_random,) +
                                                                                                                            experience_batch.observation_next.shape[1:])
        rand_actions = ar2ten(np.random.uniform(-1, 1, (q_target.shape[0] * self._hp.num_random, experience_batch.action.shape[-1])), device=self._hp.device)
        # run policy on tiled obs vectors
        curr_policy_output = self._run_policy(obs)
        curr_actions = curr_policy_output.action
        curr_log_prob = curr_policy_output.log_prob.view(self._hp.batch_size, self._hp.num_random)
        next_policy_output = self._run_policy(obs_next)
        next_actions = next_policy_output.action
        next_log_prob = next_policy_output.log_prob.view(self._hp.batch_size, self._hp.num_random)

        # get Q estimates
        rand_qs = [critic(obs, self._prep_action(rand_actions.detach())).q
                       .view(self._hp.batch_size, self._hp.num_random) for critic in self.critics]
        curr_qs = [critic(obs, self._prep_action(curr_actions.detach())).q
                       .view(self._hp.batch_size, self._hp.num_random) for critic in self.critics]
        next_qs = [critic(obs, self._prep_action(next_actions.detach())).q
                       .view(self._hp.batch_size, self._hp.num_random) for critic in self.critics]

        cat_qs = [torch.cat([rand_q, pred_q.unsqueeze(1), curr_q, next_q], 1)
                  for rand_q, pred_q, curr_q, next_q in zip(rand_qs, qs, curr_qs, next_qs)]

        if self._hp.min_q_version == 3:
            # importance sampled version
            random_density = np.log(0.5 ** curr_actions.shape[-1])
            cat_qs = [torch.cat([rand_q-random_density, next_q-next_log_prob.detach(), curr_q-curr_log_prob.detach()], 1)
                      for rand_q, curr_q, next_q in zip(rand_qs, curr_qs, next_qs)]

        # compute q corrections and subtract Q estimate under behavior policy (ie from data)
        min_qf_losses = [torch.logsumexp(cat_q / self._hp.temp, dim=1).mean() * self._hp.min_q_weight * self._hp.temp
                         for cat_q in cat_qs]
        min_qf_losses = [min_qf_loss - q.mean() * self._hp.min_q_weight for min_qf_loss, q in zip(min_qf_losses, qs)]

        # multiply correction terms with alpha and update alpha parameter
        if self._hp.with_lagrange:
            alpha_prime = torch.clamp(self._log_alpha_prime().exp(), min=0.0, max=1000000.0)
            min_qf_losses = [alpha_prime * (min_qf_loss - self._target_action_gap) for min_qf_loss in min_qf_losses]
            alpha_prime_loss = (-min_qf_losses[0] - min_qf_losses[1])*0.5
            self._perform_update(alpha_prime_loss, self.alpha_prime_opt, self._log_alpha_prime, retain_graph=True)

        # adjust critic losses
        critic_losses = [critic_loss + min_qf_loss for critic_loss, min_qf_loss in zip(critic_losses, min_qf_losses)]
        return critic_losses, qs

class MultiHeadDiscreteCQLAgent(MultiHeadQuantileDQN):
    def _default_hparams(self):
        default_dict = ParamDict({
            'min_q_weight': 1.0,
            'use_aux_loss': False,
            'normalization': 'batch',
            'use_conv': True,
            'add_weighted_pixel_copy': False, # if True, adds pixel copying stream for decoder
            'pixel_shift_decoder': False,
            'use_skips': False,
            'ngf': 8,
            'update_period': 1
        })
        return super()._default_hparams().overwrite(default_dict)

    def _compute_critic_loss(self, experience_batch, q_est, q_target, critic_output):
        loss, info = super()._compute_critic_loss(experience_batch, q_est, q_target, critic_output)
        # cql loss
        # critic_output = self.critic(experience_batch.observation)
        # critic_output.q = critic_output.q.view(-1, self._hp.critic_params.action_dim, self._hp.num_quant)
        one_hot_action = torch.eye(self._hp.critic_params.action_dim, device=self._hp.device)[experience_batch.action.type(torch.long)].squeeze(1)
        dataset_expec = {name: (critic_output.q[name].mean(2) * one_hot_action).sum(1)[experience_batch.task_ids.type(torch.long)==self.task_name_to_id[name]].mean() for i, name in enumerate(critic_output.q.keys())}
        negative_sampling = {name: torch.logsumexp(critic_output.q[name].mean(2), 1)[experience_batch.task_ids.type(torch.long)==self.task_name_to_id[name]].mean() for i, name in enumerate(critic_output.q.keys())}
        min_q_loss = {name: self._hp.min_q_weight * (negative_sampling[name] - dataset_expec[name]) for name in critic_output.q.keys()}
        min_q_loss_info = AttrDict()
        for name in critic_output.q.keys():
            min_q_loss_info[name+'_min_q_loss'] = min_q_loss[name]
        info.update(min_q_loss_info)
        critic_loss = {name: loss[name] + min_q_loss[name] for name in critic_output.q.keys()}
        critic_loss = torch.stack(list(critic_loss.values())).sum()
        return critic_loss, info

class MultiHeadContinuousCQLAgent(MultiHeadSACAgent):
    def _default_hparams(self):
        default_dict = ParamDict({
            'num_random': 10,        # number of action samples for estimating Q expectation
            'min_q_version': 3,      # if == 3: uses importance sampling for Q expectation
            'temp': 1.0,             # temperature of logsumexp operation
            'min_q_weight': 5.0,     # multiplier for Q adjustment term
            'with_lagrange': False,   # if True, autotunes the budget factor with dual optimization
            'lagrange_thresh': 5.0,  # budget factor that indicates target action gap
            'retain_backprop_graph': True,  # need to retain backprop graph during q update
            'policy_warmup_steps': 0,       # number of initial steps that policy is trained with BC
            'deterministic_backup': True,
        })
        return super()._default_hparams().overwrite(default_dict)

    def _compute_policy_loss(self, experience_batch, policy_output):
        """Optionally runs BC warmup in the beginning."""
        if self._update_steps < self._hp.policy_warmup_steps:
            policy_loss = {key: -1 * policy_output.dist[key].log_prob(experience_batch.action) + self.alpha[key] * policy_output.log_prob[key] for key in policy_output.log_prob.keys()}
            policy_loss = {key: policy_loss[key][experience_batch.task_ids==self.task_name_to_id[key]].mean() for i, key in enumerate(policy_loss.keys())}
            return torch.stack(list(policy_loss.values())).mean()
        else:
            return super()._compute_policy_loss(experience_batch, policy_output)

    def _compute_critic_loss(self, experience_batch, q_target):
        critic_losses, qs = super()._compute_critic_loss(experience_batch, q_target)

        # prepare tiled obs vectors and sample rand actions for importance sampling
        obs_dim = experience_batch.observation.shape[1:]
        obs = experience_batch.observation.unsqueeze(1).repeat((1, self._hp.num_random) + (1,)*len(obs_dim)).view((self._hp.batch_size*self._hp.num_random,) +
                                                                                                                  experience_batch.observation.shape[1:])
        obs_next = experience_batch.observation_next.unsqueeze(1).repeat((1, self._hp.num_random) + (1,)*len(obs_dim)).view((self._hp.batch_size*self._hp.num_random,) +
                                                                                                                            experience_batch.observation_next.shape[1:])
        rand_actions = ar2ten(np.random.uniform(-1, 1, (self._hp.batch_size * self._hp.num_random, experience_batch.action.shape[-1])), device=self._hp.device)
        # run policy on tiled obs vectors
        curr_policy_output = self._run_policy(obs)
        curr_actions = curr_policy_output.action
        curr_log_prob = map_dict(lambda x: x.view(self._hp.batch_size, self._hp.num_random), curr_policy_output.log_prob)
        next_policy_output = self._run_policy(obs_next)
        next_actions = next_policy_output.action
        next_log_prob = map_dict(lambda x: x.view(self._hp.batch_size, self._hp.num_random), next_policy_output.log_prob)

        # get Q estimates
        rand_qs = [map_dict(lambda x: x.view(self._hp.batch_size, self._hp.num_random),
                            critic(obs, self._prep_action(rand_actions.detach())).q) for critic in self.critics]
        curr_qs = [{key: critic(obs, self._prep_action(curr_actions[key].detach())).q[key].view(self._hp.batch_size, self._hp.num_random)
                    for key in curr_actions.keys()} for critic in self.critics]
        next_qs = [{key: critic(obs, self._prep_action(next_actions[key].detach())).q[key].view(self._hp.batch_size, self._hp.num_random)
                    for key in next_actions.keys()} for critic in self.critics]

        cat_qs = [{key: torch.cat([rand_q[key], pred_q[key].unsqueeze(1), curr_q[key], next_q[key]], 1) for key in rand_q.keys()}
                  for rand_q, pred_q, curr_q, next_q in zip(rand_qs, qs, curr_qs, next_qs)]

        if self._hp.min_q_version == 3:
            # importance sampled version
            random_density = np.log(0.5 ** rand_actions.shape[-1])
            cat_qs = [{key: torch.cat([rand_q[key]-random_density, next_q[key]-next_log_prob[key].detach(), curr_q[key]-curr_log_prob[key].detach()], 1) for key in rand_q.keys()}
                      for rand_q, curr_q, next_q in zip(rand_qs, curr_qs, next_qs)]

        # compute q corrections and subtract Q estimate under behavior policy (ie from data)
        min_qf_losses = [{key: torch.logsumexp(cat_q[key][experience_batch.task_ids==self.task_name_to_id[key]] / self._hp.temp, dim=1).mean() * self._hp.min_q_weight * self._hp.temp
                          for i, key in enumerate(cat_q.keys())} for cat_q in cat_qs]
        # print("LOG SUM EXP: ", min_qf_losses)
        min_qf_losses = [{key: min_qf_loss[key] - q[key][experience_batch.task_ids==self.task_name_to_id[key]].mean() * self._hp.min_q_weight
                          for i, key in enumerate(q.keys())} for min_qf_loss, q in zip(min_qf_losses, qs)]

        # multiply correction terms with alpha and update alpha parameter
        if self._hp.with_lagrange:
            raise NotImplementedError
            # alpha_prime = torch.clamp(self._log_alpha_prime().exp(), min=0.0, max=1000000.0)
            # min_qf_losses = [alpha_prime * (min_qf_loss - self._target_action_gap) for min_qf_loss in min_qf_losses]
            # alpha_prime_loss = (-min_qf_losses[0] - min_qf_losses[1])*0.5
            # self._perform_update(alpha_prime_loss, self.alpha_prime_opt, self._log_alpha_prime, retain_graph=True)

        # adjust critic losses
        min_qf_losses_per_task = min_qf_losses
        min_qf_losses = [torch.stack(list(min_qf_loss.values())).mean() for min_qf_loss in min_qf_losses]
        # print("Critic Loss: ", critic_losses)
        critic_losses = [critic_loss + min_qf_loss for critic_loss, min_qf_loss in zip(critic_losses, min_qf_losses)]
        # print("Min Qf Loss: ", min_qf_losses)
        return critic_losses, qs, min_qf_losses_per_task

