import os

from tarp.utils.general_utils import AttrDict
from tarp.rl.agents.sac_agent import SACAgent
from tarp.rl.policies.mlp_policies import MLPPolicy, ConvPolicy
from tarp.rl.components.critic import MLPCritic, ConvCritic
from tarp.rl.components.replay_buffer import UniformReplayBuffer, LowMemoryUniformReplayBuffer
from tarp.rl.envs.carla.carla import CarlaEnv
from tarp.rl.components.sampler import ImageSampler, MultiImageAugmentedSampler
import tarp.utils.data_aug as aug


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'used to test the RL implementation'

configuration = {
    'seed': 42,
    'agent': SACAgent,
    'environment': CarlaEnv,
    'data_dir': '.',
    'num_epochs': 100,
    'max_rollout_len': 600,
    'n_steps_per_epoch': 3e3,
    'n_warmup_steps': 3e3,
    'sampler': MultiImageAugmentedSampler
}
configuration = AttrDict(configuration)

# Policy
policy_params = AttrDict(
    action_dim=3,
    input_dim=0,
    n_layers=1,      # number of policy network layers
    action_space_type='continuous',
    output_distribution='gauss',
    max_action_range=1.,
    input_res=64,
    input_nc=9,
    nz_enc=256,
    nz_mid=1024,
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
    dump_replay=False,
    split_frames=True,
    n_frames=3
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
    batch_size=128,
    clip_q_target=False,
)

# Dataset - Random data
data_config = AttrDict()
sampler_config = AttrDict(
    n_frames=3
)

# Environment
env_config = AttrDict(
)

