import os

from tarp.utils.general_utils import AttrDict
from tarp.rl.agents.cql_agent import ContinuousCQLAgent, MultiHeadContinuousCQLAgent
from tarp.rl.agents.sac_agent import MultiHeadSACAgent
from tarp.rl.components.critic import ConvCritic, MultiHeadConvCritic
from tarp.rl.policies.mlp_policies import ConvPolicy, MultiHeadConvPolicy
from tarp.rl.envs.distracting_control.distracting_control import DistractingControlEnv, DMControlEnv
from tarp.rl.components.normalization import Normalizer
from tarp.rl.components.sampler import ImageSampler, MultiImageSampler
from tarp.data.dmcontrol.dmcontrol import DMControlDataset
from tarp.configs.default_data_configs.atari import dataset_spec
from tarp.rl.components.replay_buffer import UniformReplayBuffer, LowMemoryUniformReplayBuffer, MultiTaskLowMemoryUniformReplayBuffer


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'used to test the RL implementation'

configuration = {
    'seed': 42,
    'agent': MultiHeadContinuousCQLAgent,
    'environment': DistractingControlEnv,
    'data_dir': '.',
    'num_epochs': 1000,
    'max_rollout_len': 1000,
    'n_steps_per_epoch': 1000,
    'n_warmup_steps': 0,
    'sampler': MultiImageSampler,
    'offline_rl': True,
    'load_offline_data': True
}
configuration = AttrDict(configuration)

# Policy
policy_params = AttrDict(
    action_dim=6,
    input_dim=0,
    n_layers=1,      # number of policy network layera
    action_space_type='continuous',
    output_distribution='gauss',
    max_action_range=1.,
    input_res=64,
    input_nc=9,
    nz_enc=256,
    nz_mid=256,
    # head_keys=['walk', 'stand', 'backward'],
    head_keys=['run', 'stand', 'backward'],
    # head_keys=['walk', 'stand']
)
data_config = AttrDict(
    dataset_spec=dataset_spec,
)
data_config.dataset_spec.data_dir = os.path.join(os.environ['DATA_DIR'], './distracting_control/walker/expert/')
data_config.dataset_spec.dataset_prefix = os.path.join(os.environ['DATA_DIR'], './distracting_control/walker/expert/')
data_config.dataset_spec.dataset_class=DMControlDataset
data_config.dataset_spec.task_names = ['run-expert', 'stand-expert', 'backward-expert']
data_config.dataset_spec.resolution=64
data_config.dataset_spec.discount_factor=0.99

# Critic
critic_params = AttrDict(
    action_dim=policy_params.action_dim,
    output_dim=1,
    n_layers=policy_params.n_layers,      # number of policy network layers
    input_dim=policy_params.input_dim,
    action_input=True,
    input_nc=policy_params.input_nc,
    input_res=policy_params.input_res,
    nz_enc=policy_params.nz_enc,
    nz_mid=policy_params.nz_mid,
    head_keys=policy_params.head_keys,
)

# Replay Buffer
replay_params = AttrDict(
    dump_replay=False,
    capacity=3*3e5,
    # capacity=2*3e5,
    n_frames=3,
    head_keys=policy_params.head_keys
)

# Observation Normalization
obs_norm_params = AttrDict(
)

# Agent
agent_config = AttrDict(
    policy=MultiHeadConvPolicy,
    policy_params=policy_params,
    critic=MultiHeadConvCritic,
    critic_params=critic_params,
    replay=MultiTaskLowMemoryUniformReplayBuffer,
    replay_params=replay_params,
    batch_size=512,
    clip_q_target=False,
    min_q_weight=3.,
    num_random=1,
    deterministic_backup=True,
    policy_lr=4e-4,
    critic_lr=4e-4,
    alpha_lr=4e-4,
    optimizer='rmsprop',
)

sampler_config = AttrDict(
    n_frames=3,
)

# Environment
env_config1 = AttrDict(
    name="walker",
    task_name='run',
    reward_norm=1.,
    from_pixels=True,
    frame_skip=2,
    head_key='run'
)

env_config2 = AttrDict(
    name="walker",
    task_name='stand',
    reward_norm=1.,
    from_pixels=True,
    frame_skip=2,
    head_key='stand'
)
env_config3 = AttrDict(
    name="walker",
    task_name='backward',
    reward_norm=1.,
    from_pixels=True,
    frame_skip=2,
    head_key='backward'
)

env_config = AttrDict(
    conf_list=[env_config1, env_config2, env_config3],
    name='walker',
)
