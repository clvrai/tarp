import torch.nn as nn
import numpy as np
import torch

from tarp.utils.general_utils import ParamDict, AttrDict, nan_hook, map_recursive


class Policy(nn.Module):
    """Base policy class."""
    def __init__(self):
        super().__init__()
        self.net = self._build_network()
        self._is_train = True         # whether policy is in train or val mode
        self._rollout_mode = False    # whether policy is in rollout mode (can change which outputs are produced)

    def _default_hparams(self):
        default_dict = ParamDict({
            'action_dim': 1,                   # dimensionality of the action space
            'max_action_range': 1.,            # for cont. actions this defines a symmetric action range [-x, x]
            'squash_log_prob': True,           # do not tanh adjust log prob if set to False
        })
        return default_dict

    def forward(self, obs, **kwargs):
        output_dist = self._compute_action_dist(obs, **kwargs)
        #action = output_dist.mu
        if self._hp.action_space_type == 'continuous':
            action = output_dist.rsample()
        elif self._hp.action_space_type == 'discrete':
            action = output_dist.sample()
        else:
            raise NotImplementedError
        log_prob = output_dist.log_prob(action)
        if self._hp.squash_log_prob:
            action, log_prob = self._tanh_squash_output(action, log_prob)
        nan_hook(action); nan_hook(log_prob)
        return AttrDict(action=action, log_prob=log_prob, dist=output_dist)

    def evaluate_action(self, obs, action):
        output_dist = self._compute_action_dist(obs)
        log_prob = output_dist.log_prob(action)
        if self._hp.squash_log_prob:
            action, log_prob = self._tanh_squash_output(action, log_prob)
        nan_hook(action); nan_hook(log_prob)
        return AttrDict(action=action, log_prob=log_prob, dist=output_dist)

    def _build_network(self):
        """Constructs the policy network."""
        raise NotImplementedError("Needs to be implemented by child class.")

    def _compute_action_dist(self, obs, **kwargs):
        raise NotImplementedError("Needs to be implemented by child class.")

    def _tanh_squash_output(self, action, log_prob):
        """Passes continuous output through a tanh function to constrain action range, adjusts log_prob."""
        action_new = self._hp.max_action_range * torch.tanh(action)
        log_prob_update = np.log(self._hp.max_action_range) + 2 * (np.log(2.) - action -
              torch.nn.functional.softplus(-2. * action)).sum(dim=-1)  # maybe more stable version from Youngwoon Lee
        return action_new, log_prob - log_prob_update

    @property
    def has_trainable_params(self):
        """Indicates whether policy has trainable params."""
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad) > 0

    @property
    def action_dim(self):
        return self._hp.action_dim

    @staticmethod
    def dummy_output():
        return AttrDict(action=None, log_prob=None)

    def sample_rand(self, unused_obs):
        """Samples random action."""
        with torch.no_grad():
            # TODO: implement proper ActionSpace class with sample() method
            return AttrDict(
                action=self._hp.max_action_range * (2 * torch.rand((self._hp.action_dim,)) - 1.),
                log_prob=-torch.log(torch.tensor(2 * self._hp.max_action_range)) * self._hp.action_dim,
                # assumes symmetric action range
            )

    def switch_to_val(self):
        self._is_train = False

    def switch_to_train(self):
        self._is_train = True

    def switch_to_rollout(self):
        self._rollout_mode = True

    def switch_to_non_rollout(self):
        self._rollout_mode = False

    def visualize(self, logger, rollout_storage, step):
        """Optionally allows to further visualize the internal state of policy."""
        pass

    @property
    def device(self):
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MultiHeadPolicy(Policy):
    def _default_hparams(self):
        default_dict = ParamDict({
            'action_dim': 1,                   # dimensionality of the action space
            'max_action_range': 1.,            # for cont. actions this defines a symmetric action range [-x, x]
            'squash_log_prob': True,           # do not tanh adjust log prob if set to False
            'head_keys': ['main'],
        })
        return default_dict

    def forward(self, obs, **kwargs):
        output_dist = self._compute_action_dist(obs, **kwargs)
        #action = output_dist.mu
        if self._hp.action_space_type == 'continuous':
            action = {name: dist.rsample() for name, dist in output_dist.items()}
        elif self._hp.action_space_type == 'discrete':
            action = {name: dist.sample() for name, dist in output_dist.items()}
        else:
            raise NotImplementedError

        log_prob = {name: dist.log_prob(action[name]) for name, dist in output_dist.items()}
        # log_prob = output_dist.log_prob(action)
        if self._hp.squash_log_prob:
            squashed_action, squashed_log_prob = {}, {}
            for name in action.keys():
                squashed_action[name], squashed_log_prob[name] = self._tanh_squash_output(action[name], log_prob[name])
            action, log_prob = squashed_action, squashed_log_prob
        map_recursive(nan_hook, action); map_recursive(nan_hook, log_prob)
        return AttrDict(action=action, log_prob=log_prob, dist=output_dist)

    def evaluate_action(self, obs, action):
        output_dist = self._compute_action_dist(obs)
        log_prob = output_dist.log_prob(action)
        if self._hp.squash_log_prob:
            action, log_prob = self._tanh_squash_output(action, log_prob)
        nan_hook(action); nan_hook(log_prob)
        return AttrDict(action=action, log_prob=log_prob, dist=output_dist)

    @staticmethod
    def dummy_output():
        return AttrDict(action=None, log_prob=None)

    def sample_rand(self, unused_obs):
        """Samples random action."""
        with torch.no_grad():
            # TODO: implement proper ActionSpace class with sample() method
            return AttrDict(
                action=self._hp.max_action_range * (2 * torch.rand((self._hp.action_dim,)) - 1.),
                log_prob=-torch.log(torch.tensor(2 * self._hp.max_action_range)) * self._hp.action_dim,
                # assumes symmetric action range
            )

