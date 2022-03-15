import os, sys
import torch.nn as nn
import torch
import copy

from tarp.modules.layers import LayerBuilderParams
from tarp.modules.mdn import MDN, GMM
from tarp.modules.subnetworks import Predictor, HybridConvMLPEncoder, Encoder
from tarp.modules.variational_inference import MultivariateGaussian, Categorical
from tarp.rl.components.policy import Policy, MultiHeadPolicy
from tarp.utils.general_utils import ParamDict, AttrDict
from tarp.utils.pytorch_utils import RemoveSpatial
from tarp.components.checkpointer import CheckpointHandler

ACTIVATIONS = {
    'relu': nn.ReLU(inplace=True),
    'leaky_relu': nn.LeakyReLU(0.2, inplace=True),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh()
}

class MLPPolicy(Policy):
    """MLP-based Gaussian policy."""
    def __init__(self, config):
        # TODO automate the setup by getting params from the environment
        self._hp = self._default_hparams().overwrite(config)
        self._hp.builder = LayerBuilderParams(use_convs=False, normalization=self._hp.normalization,
                                              activation=self._hp.activation)
        super().__init__()

        if self._hp.policy_checkpoint is not None:
            self._load_policy_weights()

    def _default_hparams(self):
        default_dict = ParamDict({
            'input_dim': 32,                  # dimensionality of the observation input
            'n_layers': 3,                    # number of policy network layers
            'nz_mid': 64,                     # size of the intermediate network layers
            'normalization': 'none',          # normalization used in policy network ['none', 'batch']
            'activation': 'leaky_relu',
            'final_activation': None,
            'action_space_type': 'continuous',
            'policy_checkpoint': None,        # optionally provide checkpoint for policy weight init
        })
        return super()._default_hparams().overwrite(default_dict)

    def _build_network(self):
        return Predictor(self._hp,
                         input_size=self._hp.input_dim,
                         output_size=self.mlp_output_size,
                         mid_size=self._hp.nz_mid,
                         num_layers=self._hp.n_layers,
                         final_activation=None if self._hp.final_activation is None else ACTIVATIONS[self._hp.final_activation],
                         spatial=False)

    def _compute_action_dist(self, obs, **kwargs):
        if self._hp.action_space_type == 'continuous':
            return MultivariateGaussian(self.net(obs))
        elif self._hp.action_space_type == 'discrete':
            return Categorical(self.net(obs))
        else:
            raise NotImplementedError

    @property
    def mlp_output_size(self):
        if self._hp.action_space_type == 'continuous':
            return 2 * self._hp.action_dim      # scale and variance of Gaussian output
        elif self._hp.action_space_type == 'discrete':
            return self._hp.action_dim
        else:
            raise NotImplementedError

    def _load_policy_weights(self):
        """Loads weights for a given model from the given checkpoint directory."""
        checkpoint = self._hp.policy_checkpoint
        epoch = 'latest'
        # self.device = self._hp.device
        checkpoint_dir = checkpoint if os.path.basename(checkpoint) == 'weights' \
                            else os.path.join(checkpoint, 'weights')     # checkpts in 'weights' dir
        checkpoint_path = CheckpointHandler.get_resume_ckpt_file(epoch, checkpoint_dir)
        CheckpointHandler.load_weights(checkpoint_path, model=self.net, model_key='policy')



class MultiHeadMLPPolicy(MultiHeadPolicy):
    """MLP-based Gaussian policy."""
    def __init__(self, config):
        # TODO automate the setup by getting params from the environment
        self._hp = self._default_hparams().overwrite(config)
        self._hp.builder = LayerBuilderParams(use_convs=False, normalization=self._hp.normalization,
                                              activation=self._hp.activation)
        super().__init__()

    def _default_hparams(self):
        default_dict = ParamDict({
            'input_dim': 32,                  # dimensionality of the observation input
            'n_layers': 3,                    # number of policy network layers
            'nz_mid': 64,                     # size of the intermediate network layers
            'normalization': 'none',          # normalization used in policy network ['none', 'batch']
            'activation': 'leaky_relu',
            'final_activation': None,
            'action_space_type': 'continuous',
            'head_keys': ['main']
        })
        return super()._default_hparams().overwrite(default_dict)

    def _build_network(self):
        return nn.Sequential(
            Encoder(self._updated_encoder_params()),
            nn.ModuleDict({
                name: Predictor(self._hp,
                         input_size=self._hp.input_dim,
                         output_size=self.mlp_output_size,
                         mid_size=self._hp.nz_mid,
                         num_layers=self._hp.n_layers,
                         final_activation=None if self._hp.final_activation is None else ACTIVATIONS[self._hp.final_activation],
                         spatial=False)
                for name in self._hp.head_keys
            })
        )

    def _compute_action_dist(self, obs, **kwargs):
        if self._hp.action_space_type == 'continuous':
            return AttrDict({name: MultivariateGaussian(self.net(obs)) for name in self._hp.head_keys})
        elif self._hp.action_space_type == 'discrete':
            return AttrDict({name: Categorical(self.net(obs)) for name in self._hp.head_keys})
        else:
            raise NotImplementedError

    @property
    def mlp_output_size(self):
        if self._hp.action_space_type == 'continuous':
            return 2 * self._hp.action_dim      # scale and variance of Gaussian output
        elif self._hp.action_space_type == 'discrete':
            return self._hp.action_dim
        else:
            raise NotImplementedError

    def _updated_encoder_params(self):
        params = copy.deepcopy(self._hp)
        return params.overwrite(AttrDict(
            use_convs=False,
            use_skips=False,                  # no skip connections needed bc we are not reconstructing
            builder=LayerBuilderParams(use_convs=False, normalization=self._hp.normalization, activation=self._hp.activation),
        ))


class MDNPolicy(MLPPolicy):
    """GMM Policy (based on mixture-density network)."""
    def _default_hparams(self):
        default_dict = ParamDict({
            'num_gaussians': None,          # number of mixture components
        })
        return super()._default_hparams().overwrite(default_dict)

    def _build_network(self):
        assert self._hp.num_gaussians is not None   # need to specify number of mixture components for policy
        return nn.Sequential(
            super()._build_network(),
            MDN(self.mlp_output_size, self._hp.action_dim, self._hp.num_gaussians)
        )

    def _compute_action_dist(self, obs, **kwargs):
        return GMM(self.net(obs))

    @property
    def mlp_output_size(self):
        """Mean, variance and weight for each Gaussian."""
        return self._hp.nz_mid


class SplitObsMLPPolicy(MLPPolicy):
    """Splits off unused part of observations."""
    def _default_hparams(self):
        default_dict = ParamDict({
            'unused_obs_size': None,    # dimensionality of split off observation part
            'discard_part': 'back',     # which part of observation to discard ['front', 'back']
        })
        return super()._default_hparams().overwrite(default_dict)

    def _compute_action_dist(self, raw_obs, **kwargs):
        if self._hp.discard_part == 'front':
            return super()._compute_action_dist(raw_obs[:, self._hp.unused_obs_size:])
        elif self._hp.discard_part == 'back':
            return super()._compute_action_dist(raw_obs[:, :-self._hp.unused_obs_size])
        else:
            raise ValueError("Cannot parse discard_part parameter {}!".format(self._hp.discard_part))

class ConvPolicy(MLPPolicy):
    """Enodes input image with conv encoder, then uses MLP head to produce output action distribution."""
    def __init__(self, config):
        super().__init__(config)

    def _default_hparams(self):
        default_dict = ParamDict({
            'input_res': 32,                  # resolution of the image input
            'input_nc': 3,                    # number of input channels
            'ngf': 8,                         # number of channels in shallowest layer of image encoder
            'nz_enc': 64,                     # number of dimensions in encoder-latent space
            'final_activation': None,
            'activation': 'leaky_relu',
            'input_width': None,
            'input_height': None,
        })
        return super()._default_hparams().overwrite(default_dict)

    def _build_network(self):
        if self._hp.input_width is None and self._hp.input_height is None:
            self._hp.input_width = self._hp.input_res
            self._hp.input_height = self._hp.input_res
        ratio = max(self._hp.input_width//self._hp.input_height, self._hp.input_height//self._hp.input_width)
        enc_size = self._hp.nz_enc * (ratio**2)
        return nn.Sequential(
            Encoder(self._updated_encoder_params()),
            # RemoveSpatial(),
            Predictor(self._hp,
                      input_size=enc_size,
                      output_size=self.mlp_output_size,
                      mid_size=self._hp.nz_mid,
                      num_layers=self._hp.n_layers,
                      final_activation=None if self._hp.final_activation is None else ACTIVATIONS[self._hp.final_activation],
                      spatial=False),
        )

    def encode(self, obs):
        return self.net[0](obs[:, self._hp.input_dim:].reshape(-1, self._hp.input_nc, self._hp.input_height, self._hp.input_width))

    def _compute_action_dist(self, obs):
        return super()._compute_action_dist(obs[:, self._hp.input_dim:].reshape(-1, self._hp.input_nc, self._hp.input_height, self._hp.input_width))

    def _updated_encoder_params(self):
        # params = copy.deepcopy(self._hp)
        params = self._hp
        return params.overwrite(AttrDict(
            use_convs=True,
            use_skips=False,                  # no skip connections needed bc we are not reconstructing
            img_sz=self._hp.input_res,  # image resolution
            builder=LayerBuilderParams(use_convs=True, normalization=self._hp.normalization, activation=self._hp.activation),
        ))

    @property
    def encoder(self):
        return self.net[0]

    def _load_policy_weights(self):
        """Loads weights for a given model from the given checkpoint directory."""
        checkpoint = self._hp.policy_checkpoint
        epoch = 'latest'
        # self.device = self._hp.device
        checkpoint_dir = checkpoint if os.path.basename(checkpoint) == 'weights' \
                            else os.path.join(checkpoint, 'weights')     # checkpts in 'weights' dir
        checkpoint_path = CheckpointHandler.get_resume_ckpt_file(epoch, checkpoint_dir)
        CheckpointHandler.load_weights(checkpoint_path, model=self.net[1], model_key='policy', device=self.device)

class MultiHeadConvPolicy(MultiHeadMLPPolicy):
    """Enodes input image with conv encoder, then uses MLP head to produce output action distribution."""
    def __init__(self, config):
        super().__init__(config)
        if self._hp.encoder_checkpoint is not None:
            self.encoder._load_checkpoint()

    def _default_hparams(self):
        default_dict = ParamDict({
            'input_res': 32,                  # resolution of the image input
            'input_nc': 3,                    # number of input channels
            'ngf': 8,                         # number of channels in shallowest layer of image encoder
            'nz_enc': 64,                     # number of dimensions in encoder-latent space
            'final_activation': None,
            'encoder_checkpoint': None,
            'encoder_epoch': 'latest',
            'finetune': False,
            'head_keys': ['main']
        })
        return super()._default_hparams().overwrite(default_dict)

    @property
    def encoder(self):
        return self.net[0]

    def encode(self, obs):
        if not self._hp.finetune:
            self.encoder.eval()
        return self.encoder(obs.reshape(-1, self._hp.input_nc,
                                                 self._hp.input_res,
                                                 self._hp.input_res))
    def _build_network(self):
        activations = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(0.2, inplace=True),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }
        return nn.Sequential(
            Encoder(self._updated_encoder_params()),
            RemoveSpatial(),
            nn.ModuleDict({
                name: Predictor(self._hp,
                             input_size=self._hp.nz_enc,
                             output_size=self.mlp_output_size,
                             mid_size=self._hp.nz_mid,
                             num_layers=self._hp.n_layers,
                             final_activation=None if self._hp.final_activation is None else activations[self._hp.final_activation],
                             spatial=False)
                for name in self._hp.head_keys
            })
        )

    @property
    def heads(self):
        return self.net[2]

    def _compute_action_dist(self, obs):
        feat = self.net[:2](obs[:, self._hp.input_dim:].reshape(-1, self._hp.input_nc, self._hp.input_res, self._hp.input_res))
        feat = AttrDict({name: self.heads[name](feat) for name in self._hp.head_keys})
        if self._hp.action_space_type == 'continuous':
            return AttrDict({name: MultivariateGaussian(feat[name]) for name in self._hp.head_keys})
        else:
            return AttrDict({name: Categorical(feat[name]) for name in self._hp.head_keys})

    def _updated_encoder_params(self):
        params = copy.deepcopy(self._hp)
        return params.overwrite(AttrDict(
            use_convs=True,
            use_skips=False,                  # no skip connections needed bc we are not reconstructing
            img_sz=self._hp.input_res,  # image resolution
            builder=LayerBuilderParams(use_convs=True, normalization=self._hp.normalization, activation=self._hp.activation),
        ))

class HybridConvMLPPolicy(MLPPolicy):
    """Policy that can incorporate image and scalar inputs by fusing conv and MLP encoder."""
    def _build_network(self):
        return HybridConvMLPEncoder(self._hp.overwrite(AttrDict(output_dim=self.mlp_output_size)))

    def _default_hparams(self):
        default_dict = ParamDict({
            'input_res': 32,                  # resolution of the image input
            'input_nc': 3,                    # number of input channels
            'ngf': 8,                         # number of channels in shallowest layer of image encoder
            'nz_enc': 64,                     # number of dimensions in encoder-latent space
            'final_activation': None,
            'activation': 'leaky_relu',
            'use_custom_convs': False
        })
        return super()._default_hparams().overwrite(default_dict)

    def _updated_encoder_params(self):
        params = copy.deepcopy(self._hp)
        return params.overwrite(AttrDict(
            use_convs=True,
            use_skips=False,                  # no skip connections needed bc we are not reconstructing
            use_custom_convs=self._hp.use_custom_convs,
            img_sz=self._hp.input_res,  # image resolution
            builder=LayerBuilderParams(use_convs=True, normalization=self._hp.normalization, activation=self._hp.activation),
        ))

    def _compute_action_dist(self, obs, **kwargs):
        """Splits concatenated input obs into image and vector observation and passes through network."""
        split_obs = AttrDict(
            vector=obs[:, :self._hp.input_dim],
            image=obs[:, self._hp.input_dim:].reshape(-1, self._hp.input_nc, self._hp.input_res, self._hp.input_res)
        )
        return super()._compute_action_dist(split_obs)

class ConvAuxStatePolicy(ConvPolicy):
    def _build_network(self):
        activations = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(0.2, inplace=True),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }
        return nn.Sequential(
            Encoder(self._updated_encoder_params()),
            RemoveSpatial(),
            Predictor(self._hp,
                      input_size=self._hp.nz_enc+self._hp.input_dim,
                      output_size=self.mlp_output_size,
                      mid_size=self._hp.nz_mid,
                      num_layers=self._hp.n_layers,
                      final_activation=None if self._hp.final_activation is None else activations[self._hp.final_activation],
                      spatial=False),
        )


    def _compute_action_dist(self, obs, **kwargs):
        vector = obs[:, :self._hp.input_dim]
        image = obs[:, self._hp.input_dim:].reshape(-1, self._hp.input_nc, self._hp.input_res, self._hp.input_res)
        feat = self.net[:2](image)
        feat = torch.cat((feat, vector), dim=1)
        out = self.net[2](feat)
        if self._hp.action_space_type == 'continuous':
            return MultivariateGaussian(out)
        elif self._hp.action_space_type == 'discrete':
            return Categorical(out)
        else:
            raise NotImplementedError


