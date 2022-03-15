import os

from tarp.utils.general_utils import AttrDict
from tarp.rl.agents.sac_agent import SACAgent
from tarp.rl.policies.mlp_policies import MLPPolicy, ConvPolicy
from tarp.rl.components.critic import MLPCritic, ConvCritic
from tarp.rl.components.replay_buffer import UniformReplayBuffer, LowMemoryUniformReplayBuffer
from tarp.rl.envs.metaworld import MetaWorldEnv
from tarp.rl.envs.distracting_metaworld import DistractingMetaWorldEnv
from tarp.rl.components.normalization import Normalizer
from tarp.rl.components.sampler import ImageSampler


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'used to test the RL implementation'

configuration = {
    'seed': 41,
    'agent': SACAgent,
    'environment': DistractingMetaWorldEnv,
    'data_dir': '.',
    'num_epochs': 17,
    'max_rollout_len': 150,
    'n_steps_per_epoch': 30000,
    'n_warmup_steps': 5e3,
    'sampler': ImageSampler
}
configuration = AttrDict(configuration)

# Policy
policy_params = AttrDict(
    input_dim=0,
    action_dim=4,
    n_layers=1,      #  number of policy network laye
    action_space_type='continuous',
    output_distribution='gauss',
    nz_mid=1024,
    nz_enc=256,
    input_res=128,
    input_nc=3,
    max_action_range=1.,
)

# Critic
critic_params = AttrDict(
    action_dim=policy_params.action_dim,
    output_dim=1,
    n_layers=1,      # number of policy network layers
    action_input=True,
    input_nc=policy_params.input_nc,
    input_res=policy_params.input_res,
    nz_enc=policy_params.nz_enc,
    nz_mid=policy_params.nz_mid,
)

# Replay Buffer
replay_params = AttrDict(
    capacity=1e5,
    dump_replay=True,
)

# Observation Normalization
obs_norm_params = AttrDict(
)

# Agent
agent_config = AttrDict(
    policy=ConvPolicy,
    policy_params=policy_params,
    critic=ConvCritic,
    critic_params=critic_params,
    replay=LowMemoryUniformReplayBuffer,
    replay_params=replay_params,
    clip_q_target=False,
    batch_size=128,
    reward_scale=10
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
    from_pixels=True,
    # background_freq=1
)
