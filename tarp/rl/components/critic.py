import torch.nn as nn
import torch
import copy
import os

from tarp.utils.general_utils import ParamDict, AttrDict
from tarp.modules.layers import LayerBuilderParams
from tarp.modules.subnetworks import Encoder, Predictor, HybridConvMLPEncoder, HybridConvTwinMLPEncoder, MultiHeadHybridConvMLPEncoder
from tarp.utils.pytorch_utils import RemoveSpatial
from tarp.components.checkpointer import CheckpointHandler


class Critic(nn.Module):
    """Base critic class."""
    def __init__(self):
        super().__init__()
        self._net = self._build_network()

    def _default_hparams(self):
        default_dict = ParamDict({
            'action_dim': 1,    # dimensionality of the action space
            'normalization': 'none',        # normalization used in policy network ['none', 'batch']
            'action_input': True,       # forward takes actions as second argument if set to True
        })
        return default_dict

    def forward(self, obs, actions=None):
        raise NotImplementedError("Needs to be implemented by child class.")

    @staticmethod
    def dummy_output():
        return AttrDict(q=None)

    def _build_network(self):
        """Constructs the policy network."""
        raise NotImplementedError("Needs to be implemented by child class.")

    @property
    def device(self):
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class MLPCritic(Critic):
    """MLP-based critic."""
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._hp.builder = LayerBuilderParams(use_convs=False, normalization=self._hp.normalization, activation=self._hp.activation)
        super().__init__()

    def _default_hparams(self):
        default_dict = ParamDict({
            'input_dim': 32,    # dimensionality of the observation input
            'n_layers': 3,      # number of policy network layers
            'nz_mid': 64,       # size of the intermediate network layers
            'output_dim': 1,    # number of outputs, can be >1 for discrete action spaces
            'activation': 'leaky_relu'
        })
        return super()._default_hparams().overwrite(default_dict)

    def forward(self, obs, actions=None):
        input = torch.cat((obs, actions), dim=-1) if self._hp.action_input else obs
        return AttrDict(q=self._net(input))

    def _build_network(self):
        input_size = self._hp.input_dim + self._hp.action_dim if self._hp.action_input else self._hp.input_dim
        return Predictor(self._hp,
                         input_size=input_size,
                         output_size=self._hp.output_dim,
                         mid_size=self._hp.nz_mid,
                         num_layers=self._hp.n_layers,
                         spatial=False)


class ConvCritic(MLPCritic):
    """Critic that can incorporate image and action inputs by fusing conv and MLP encoder."""
    def __init__(self, config):
        super().__init__(config)


    def _default_hparams(self):
        default_dict = ParamDict({
            'input_res': 32,                  # resolution of the image input
            'input_nc': 3,                    # number of input channels
            'ngf': 8,                         # number of channels in shallowest layer of image encoder
            'nz_enc': 64,                     # number of dimensions in encoder-latent space
            'activation': 'leaky_relu',
            'input_width': None,
            'input_height': None,
        })
        return super()._default_hparams().overwrite(default_dict)

    def _build_network(self):
        if self._hp.input_width is None and self._hp.input_height is None:
            self._hp.input_width = self._hp.input_res
            self._hp.input_height = self._hp.input_res
        if self._hp.action_input:
            return HybridConvMLPEncoder(self._hp.overwrite(AttrDict(input_dim=self._hp.action_dim)))
        else:
            ratio = max(self._hp.input_width//self._hp.input_height, self._hp.input_height//self._hp.input_width)
            enc_size = self._hp.nz_enc * (ratio**2)
            return nn.Sequential(
                Encoder(self._updated_encoder_params()),
                RemoveSpatial(),
                Predictor(self._hp,
                          input_size=enc_size,
                          output_size=self._hp.output_dim,
                          mid_size=self._hp.nz_mid,
                          num_layers=self._hp.n_layers,
                          final_activation=None,
                          spatial=False)
            )

    @property
    def encoder(self):
        if self._hp.action_input:
            return self._net.encoder
        else:
            return self._net[0]

    def forward(self, obs, actions=None, **kwargs):
        if self._hp.action_input:
            split_obs = AttrDict(
                vector=actions,
                image=obs.reshape(-1, self._hp.input_nc, self._hp.input_height, self._hp.input_width)
            )
            return AttrDict(q=self._net(split_obs, **kwargs))
        else:
            return AttrDict(q=self._net(obs.reshape(-1, self._hp.input_nc, self._hp.input_res, self._hp.input_res)))

    def _updated_encoder_params(self):
        params = copy.deepcopy(self._hp)
        return params.overwrite(AttrDict(
            use_convs=True,
            use_skips=False,                  # no skip connections needed bc we are not reconstructing
            img_sz=self._hp.input_res,  # image resolution
            builder=LayerBuilderParams(use_convs=True, normalization=self._hp.normalization, activation=self._hp.activation)
        ))


class MultiHeadConvCritic(ConvCritic):
    def _default_hparams(self):
        default_dict = ParamDict({
            'head_keys': ['main']
        })
        return super()._default_hparams().overwrite(default_dict)

    def _build_network(self):
        if self._hp.action_input:
            return MultiHeadHybridConvMLPEncoder(self._hp.overwrite(AttrDict(input_dim=self._hp.action_dim)))
        else:
            return nn.Sequential(
                Encoder(self._updated_encoder_params()),
                RemoveSpatial(),
                nn.ModuleDict({
                    name: Predictor(self._hp,
                              input_size=self._hp.nz_enc,
                              output_size=self._hp.output_dim,
                              mid_size=self._hp.nz_mid,
                              num_layers=2,
                              final_activation=None,
                              spatial=False)
                    for name in self._hp.head_keys
                })
            )

    def forward(self, obs, actions=None, **kwargs):
        if self._hp.action_input:
            if not isinstance(actions, dict):
                actions = {key: actions for key in self._hp.head_keys}
            split_obs = AttrDict(
                vector=actions,
                image=obs.reshape(-1, self._hp.input_nc, self._hp.input_res, self._hp.input_res)
            )
            return AttrDict(q=self._net(split_obs, **kwargs))
        else:
            feat = self._net[:2](obs)
            out = AttrDict({name: self._net[2][name](feat) for name in self._hp.head_keys})
            return AttrDict(q=out)

class HybridConvMLPCritic(ConvCritic):
    def _build_network(self):
        input_dim = copy.deepcopy(self._hp.input_dim)
        if self._hp.action_input:
            input_dim += self._hp.action_dim
        return HybridConvMLPEncoder(self._hp.overwrite(AttrDict(input_dim=input_dim)))

    @property
    def encoder(self):
        if self._hp.action_input:
            return self._net._image_enc
        else:
            return self._net._image_enc

    def encode(self, obs):
        return self.encoder(obs[:, self._hp.input_dim:].reshape(-1, self._hp.input_nc, self._hp.input_res, self._hp.input_res))

    def forward(self, obs, actions=None, **kwargs):
        if self._hp.action_input:
            obs = torch.cat([actions, obs], dim=1)
            split_obs = AttrDict(
                vector=obs[:, :self._hp.input_dim],
                image=obs[:, self._hp.input_dim:].reshape(-1, self._hp.input_nc, self._hp.input_res, self._hp.input_res)
            )
        else:
            split_obs = AttrDict(
                vector=obs[:, :self._hp.input_dim],
                image=obs[:, self._hp.input_dim:].reshape(-1, self._hp.input_nc, self._hp.input_res, self._hp.input_res)
            )
        return AttrDict(q=self._net(split_obs, **kwargs))

    def _updated_encoder_params(self):
        params = copy.deepcopy(self._hp)
        return params.overwrite(AttrDict(
            use_convs=True,
            use_skips=False,                  # no skip connections needed bc we are not reconstructing
            img_sz=self._hp.input_res,  # image resolution
            builder=LayerBuilderParams(use_convs=True, normalization=self._hp.normalization, activation=self._hp.activation)
        ))

class ConvAuxStateCritic(ConvCritic):
    def _build_network(self):
        if self._hp.action_input:
            raise NotImplementedError
        else:
            return nn.Sequential(
                Encoder(self._updated_encoder_params()),
                RemoveSpatial(),
                Predictor(self._hp,
                          input_size=self._hp.nz_enc+self._hp.input_dim,
                          output_size=1,
                          mid_size=self._hp.nz_mid,
                          num_layers=2,
                          final_activation=None,
                          spatial=False)
            )

    def forward(self, obs, actions=None, **kwargs):
        if self._hp.action_input:
            raise NotImplementedError
        else:
            vector = obs[:, :self._hp.input_dim]
            image = obs[:, self._hp.input_dim:].reshape(-1, self._hp.input_nc, self._hp.input_res, self._hp.input_res)
            feat = self._net[:2](image)
            feat = torch.cat((feat, vector), dim=1)
            q = self._net[2](feat)
            return AttrDict(q=q)

class SplitObsMLPCritic(MLPCritic):
    """Splits off unused part of observations."""
    def _default_hparams(self):
        default_dict = ParamDict({
            'unused_obs_size': None,    # dimensionality of split off observation part
            'discard_part': 'back',     # which part of observation to discard ['front', 'back']
        })
        return super()._default_hparams().overwrite(default_dict)

    def forward(self, raw_obs, *args, **kwargs):
        if self._hp.discard_part == 'front':
            return super().forward(raw_obs[:, self._hp.unused_obs_size:], *args, **kwargs)
        elif self._hp.discard_part == 'back':
            return super().forward(raw_obs[:, :-self._hp.unused_obs_size], *args, **kwargs)
        else:
            raise ValueError("Cannot parse discard_part parameter {}!".format(self._hp.discard_part))


class ConvTwinCritic(MLPCritic):
    """Critic that can incorporate image and action inputs by fusing conv and MLP encoder."""
    def _default_hparams(self):
        default_dict = ParamDict({
            'input_res': 32,                  # resolution of the image input
            'input_nc': 3,                    # number of input channels
            'ngf': 8,                         # number of channels in shallowest layer of image encoder
            'nz_enc': 64,                     # number of dimensions in encoder-latent space
        })
        return super()._default_hparams().overwrite(default_dict)

    def _build_network(self):
        if self._hp.action_input:
            return HybridConvTwinMLPEncoder(self._hp.overwrite(AttrDict(input_dim=self._hp.action_dim)))
        else:
            raise NotImplementedError

    @property
    def encoder(self):
        if self._hp.action_input:
            return self._net._image_enc
        else:
            raise NotImplementedError

    def encode(self, obs):
        return self.encoder(obs.reshape(-1, self._hp.input_nc, self._hp.input_res, self._hp.input_res))

    def forward(self, obs, actions=None, **kwargs):
        if self._hp.action_input:
            split_obs = AttrDict(
                vector=actions,
                image=obs.reshape(-1, self._hp.input_nc, self._hp.input_res, self._hp.input_res)
            )
            ret = self._net(split_obs, **kwargs)
            return AttrDict(q1=ret[0], q2=ret[1])
        else:
            raise NotImplementedError

    def _updated_encoder_params(self):
        params = copy.deepcopy(self._hp)
        return params.overwrite(AttrDict(
            use_convs=True,
            use_skips=False,                  # no skip connections needed bc we are not reconstructing
            img_sz=self._hp.input_res,  # image resolution
            builder=LayerBuilderParams(use_convs=True, normalization=self._hp.normalization)
        ))

