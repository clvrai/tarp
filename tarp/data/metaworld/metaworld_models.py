import torch
import torch.nn as nn

from tarp.models.rpl_mdl import RPLModel
from tarp.models.skill_space_mdl import SkillSpaceMdl
from tarp.modules.subnetworks import Predictor


class MetaWorldRPLModel(RPLModel):
    """Assumes that last 3 dims of observation contain the goal -> filters those for the LL."""
    def _build_ll_net(self):
        class FilteredLLPredictor(nn.Module):
            def __init__(self, hp):
                super().__init__()
                self._hp = hp
                self._net = Predictor(self._hp, input_size=2 * (self._hp.state_dim - 3), output_size=self._hp.action_dim)

            def forward(self, obs):
                """Filter out dims that contain goal info."""
                assert len(obs.shape) == 2
                obs = torch.cat((obs[:, :(self._hp.state_dim - 3)],
                                 obs[:, self._hp.state_dim : 2*self._hp.state_dim - 3]), dim=-1)
                return self._net(obs)

        return FilteredLLPredictor(self._hp)


class MetaWorldSkillSpaceMdl(SkillSpaceMdl):
    """Filters goal from learned prior input."""
    def _compute_learned_prior(self, prior_mdl, inputs):
        return super()._compute_learned_prior(prior_mdl, inputs[:, :(self._hp.state_dim - 3)])

    @property
    def prior_input_size(self):
        return self._hp.state_dim - 3    # filter goal coord

