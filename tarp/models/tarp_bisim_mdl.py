from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from tarp.components.base_model import BaseModel
from tarp.modules.losses import L2Loss, NLL, PenaltyLoss
from tarp.modules.subnetworks import Encoder, Predictor
from tarp.utils.general_utils import AttrDict, ParamDict, batch_apply, remove_spatial
from tarp.utils.pytorch_utils import make_one_hot
from tarp.modules.layers import LayerBuilderParams
from tarp.modules.variational_inference import NoClampMultivariateGaussian


class TARPBisimModel(BaseModel):
    """Bisimulation-based representation learning model, based on Zhang et al., ICLR 2021."""
    def __init__(self, params, logger):
        super().__init__(logger)
        self._hp = self._default_hparams()
        self._hp.overwrite(params)
        self._hp.builder = LayerBuilderParams(self._hp.use_convs, self._hp.normalization)

        self._task_names = [task_name().name if not isinstance(task_name, str) else task_name
                            for task_name in self._hp.task_names]

        self.build_network()
        self._copy_to_target_network(self.target_encoder, self.encoder)     # initialize target encoder
        self._step = 0

    def _default_hparams(self):
        default_dict = ParamDict({
            'use_skips': False,
            'use_convs': True,
            'normalization': 'batch',
            'discount': 0.9,                       # discount used to compute bisimilarity
            'target_network_update_factor': 1.,    # lag in the target network params for transition target
            'alternating_opt': False,              # if True, alternates optimization of transition and bisim losses
            'action_space_type': 'continuous',     # action space type (discrete or continuous)
        })

        # Network size
        default_dict.update({
            'action_dim': -1,           # dimensionality of the action input
            'img_sz': 64,               # resolution of the input images
            'input_nc': 3,              # number of channels of input images
            'ngf': 8,                   # number of channels in input layer of encoder --> gets doubled every layer
            'nz_enc': 128,              # representation latent size
            'nz_mid': 128,              # dimensionality of intermediate fully connected layers
            'n_processing_layers': 2,   # number of hidden layers in non-conv nets
        })

        # Loss weights
        default_dict.update({
            'bisim_weight': 1.,         # weight of bisimulation loss component
            'pred_weight': 1.,          # weight of transition and reward prediction loss components
        })
        parent_params = super()._default_hparams()
        parent_params.overwrite(default_dict)
        return parent_params

    def build_network(self):
        self.encoder = Encoder(self._hp)
        self.pred_mdl = Predictor(self._updated_mlp_params(),
                                  input_size=self._hp.nz_enc + self._hp.action_dim,
                                  output_size=2*self._hp.nz_enc)
        self.reward_mdls = nn.ModuleDict({name: Predictor(self._updated_mlp_params(), input_size=self._hp.nz_enc,
                                                          output_size=1) for name in self._task_names})

        self.target_encoder = Encoder(self._hp)

    def forward(self, inputs):
        output = AttrDict()
        assert inputs.images.shape[1] == 2      # we need two consecutive states for training of predictive model
        r = inputs.rewards[:, 0]

        # encode inputs
        z = remove_spatial(self.encoder(inputs.images[:, 0]))
        with torch.no_grad():
            z_prime = remove_spatial(self.target_encoder(inputs.images[:, 1]))

        # randomly build pairs of states
        perm = np.random.permutation(z.shape[0])

        # predict next state
        if self._hp.action_space_type == 'discrete':
            actions = make_one_hot(inputs.actions[:, 0, 0].long(), self._hp.action_dim).float()
        elif self._hp.action_space_type == 'continuous':
            actions = inputs.actions[:, 0]
        else:
            raise NotImplementedError
        z_prime_hat = output.z_prime_hat = NoClampMultivariateGaussian(
            self.pred_mdl(torch.cat((z, actions), dim=-1)))

        # compute latent dist and bisimilarity (ie target for latent dist)
        output.z_dist = F.smooth_l1_loss(z, z[perm], reduction='none').mean(dim=-1)
        output.r_dist = F.smooth_l1_loss(r, r[perm], reduction='none')
        output.trans_dist = z_prime_hat.detach().wasserstein2_distance(
            NoClampMultivariateGaussian(z_prime_hat.detach().tensor()[perm])).mean(dim=-1)
        output.bisimilarity = output.r_dist + self._hp.discount * output.trans_dist

        # compute reward prediction
        z_prime_hat_sample = z_prime_hat.rsample()
        output.reward_pred = AttrDict({name: self.reward_mdls[name](z_prime_hat_sample) for name in self._task_names})

        # store transition model targets
        output.trans_targets = z_prime

        # update target encoder
        self._soft_update_target_network(self.target_encoder, self.encoder)

        return output

    def loss(self, model_output, inputs):
        losses = AttrDict()

        self._step += 1
        if not self._hp.alternating_opt or self._step % 2 == 0:
            # latent bisimulation loss
            losses.bisim_loss = L2Loss(self._hp.bisim_weight)(model_output.z_dist, model_output.bisimilarity)
        if not self._hp.alternating_opt or self._step % 2 == 1:
            # predictive losses for transition and reward models
            losses.transition_loss = PenaltyLoss(self._hp.pred_weight)(self._compute_trans_loss(model_output.z_prime_hat,
                                                                                                model_output.trans_targets))
            losses.update(AttrDict({'reward_loss_' + name: L2Loss(self._hp.pred_weight)
                                    (model_output.reward_pred[name][:, 0][inputs.task_id == i],
                                     inputs.rewards[:, 0][inputs.task_id == i]) for i, name in enumerate(self._task_names)}))

        losses.total = self._compute_total_loss(losses)

        with torch.no_grad():
            if self._hp.alternating_opt and self._step % 2 == 0:
                losses.transition_loss = PenaltyLoss(self._hp.pred_weight)(self._compute_trans_loss(model_output.z_prime_hat,
                                                                                                model_output.trans_targets))
                losses.update(AttrDict({'reward_loss_' + name: L2Loss(self._hp.pred_weight)
                                        (model_output.reward_pred[name][:, 0][inputs.task_id == i],
                                         inputs.rewards[:, 0][inputs.task_id == i]) for i, name in enumerate(self._task_names)}))
            if self._hp.alternating_opt and self._step % 2 == 1:
                losses.bisim_loss = L2Loss(self._hp.bisim_weight)(model_output.z_dist, model_output.bisimilarity)

        return losses

    def log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        super()._log_losses(losses, step, log_images, phase)
        self._logger.log_scalar(model_output.r_dist.mean(), 'reward_distance', step, phase)
        self._logger.log_scalar(model_output.z_dist.mean(), 'latent_distance', step, phase)
        self._logger.log_scalar(model_output.trans_dist.mean(), 'transition_distance', step, phase)
        self._logger.log_scalar(model_output.bisimilarity.mean(), 'bisimilarity', step, phase)

    def forward_encoder(self, inputs):
        enc = self.encoder(inputs)
        return enc

    def _compute_trans_loss(self, z_prime_hat, trans_targets):
        diff = (z_prime_hat.mu - trans_targets.detach()) / z_prime_hat.sigma
        return torch.mean(0.5 * diff.pow(2) + torch.log(z_prime_hat.sigma))

    def _updated_mlp_params(self):
        params = copy.deepcopy(self._hp)
        return params.overwrite(AttrDict(
            use_convs=False,
            builder=LayerBuilderParams(use_convs=False, normalization=self._hp.normalization)
        ))

    def _soft_update_target_network(self, target, source):
        """Copies weights from source to target with weight [0,1]."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self._hp.target_network_update_factor * param.data +
                                    (1 - self._hp.target_network_update_factor) * target_param.data)

    @staticmethod
    def _copy_to_target_network(target, source):
        """Completely copies weights from source to target."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    @property
    def resolution(self):
        return self._hp.img_sz

    @contextmanager
    def val_mode(self):
        pass
        yield
        pass

