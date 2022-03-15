import os

from tarp.utils.general_utils import AttrDict
from tarp.rl.agents.sac_agent import SACAgent
from tarp.rl.policies.mlp_policies import MLPPolicy, ConvPolicy, HybridConvMLPPolicy
from tarp.rl.components.critic import MLPCritic, ConvCritic, HybridConvMLPCritic
from tarp.rl.components.replay_buffer import UniformReplayBuffer, LowMemoryUniformReplayBuffer, AugmentedLowMemoryUniformReplayBuffer
from tarp.rl.components.environment import GymEnv
from tarp.rl.envs.carla.carla import CarlaEnv
from tarp.rl.components.sampler import ImageSampler, MultiImageAugmentedSampler, MultiSegmentationAugmentedSampler
import tarp.utils.data_aug as aug


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'used to test the RL implementation'

configuration = {
    'seed': 42,
    'agent': SACAgent,
    'environment': GymEnv,
    'data_dir': '.',
    'num_epochs': 300,
    'max_rollout_len': 600,
    'n_steps_per_epoch': 3e4,
    'n_warmup_steps': 3e3,
    'sampler': MultiSegmentationAugmentedSampler,
}
configuration = AttrDict(configuration)

# Policy
policy_params = AttrDict(
    action_dim=2,
    input_dim=11,
    n_layers=1,      # number of policy network layera
    action_space_type='continuous',
    output_distribution='gauss',
    max_action_range=1.,
    input_res=128,
    input_nc=9,
    nz_enc=256,
    nz_mid=1024,
)

# Critic
critic_params = AttrDict(
    action_dim=policy_params.action_dim,
    input_dim=policy_params.input_dim,
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
    capacity=3e5,
    dump_replay=True,
    n_frames=3,
    split_frames=True,
    input_dim=policy_params.input_dim,
    resolution=128,
    load_replay=True
)

# Observation Normalization
obs_norm_params = AttrDict(
)

# Agent
agent_config = AttrDict(
    policy=HybridConvMLPPolicy,
    policy_params=policy_params,
    critic=HybridConvMLPCritic,
    critic_params=critic_params,
    replay=AugmentedLowMemoryUniformReplayBuffer,
    replay_params=replay_params,
    batch_size=128,
    clip_q_target=False,
    reward_scale=10.,
)

# Dataset - Random data
data_config = AttrDict()
sampler_config = AttrDict(
    n_frames=3,
    resolution=128
)

# Environment
env_config = AttrDict(
    name='carla-state-v0',
    task_json='./tarp/data/carla/json_data/town05/left.json',
    town='Town05',
    port=2008,
    tm_port=8008,
    num_cameras=1,
    screen_height=128,
    screen_width=128,
    resolution=128,
    num_vehicles=200,
)

