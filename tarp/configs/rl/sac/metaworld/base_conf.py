import os

from tarp.utils.general_utils import AttrDict
from tarp.rl.agents.sac_agent import SACAgent
from tarp.rl.policies.mlp_policies import MLPPolicy
from tarp.rl.components.critic import MLPCritic
from tarp.rl.components.replay_buffer import UniformReplayBuffer
from tarp.rl.envs.metaworld import MetaWorldEnv
from tarp.rl.envs.distracting_metaworld import DistractingMetaWorldEnv
from tarp.rl.components.normalization import Normalizer


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'used to test the RL implementation'

configuration = {
    'seed': 41,
    'agent': SACAgent,
    'environment': DistractingMetaWorldEnv,
    'data_dir': '.',
    'num_epochs': 100,
    'max_rollout_len': 150,
    'n_steps_per_epoch': 50000,
    'n_warmup_steps': 1e4,
}
configuration = AttrDict(configuration)

# Policy
policy_params = AttrDict(
    input_dim=39,
    action_dim=4,
    n_layers=5,      #  number of policy network laye
    nz_mid=256,
    max_action_range=1.,
)

# Critic
critic_params = AttrDict(
    action_dim=policy_params.action_dim,
    input_dim=policy_params.input_dim,
    output_dim=1,
    n_layers=2,      #  number of policy network laye
    nz_mid=256,
    action_input=True,
)

# Replay Buffer
replay_params = AttrDict(
)

# Observation Normalization
obs_norm_params = AttrDict(
)

# Agent
agent_config = AttrDict(
    policy=MLPPolicy,
    policy_params=policy_params,
    critic=MLPCritic,
    critic_params=critic_params,
    replay=UniformReplayBuffer,
    replay_params=replay_params,
    clip_q_target=False,
    batch_size=256,
)

# # Dataset - Random data
data_config = AttrDict()
# data_config.dataset_spec = data_spec

# Environment
env_config = AttrDict(
    reward_norm=1.,
    resolution=128,
    screen_height=128,
    screen_width=128,
    from_pixels=False
)
