import os

from tarp.utils.general_utils import AttrDict
from tarp.rl.agents.cql_agent import ContinuousCQLAgent, MultiHeadContinuousCQLAgent
from tarp.rl.agents.sac_agent import MultiHeadSACAgent
from tarp.rl.components.critic import ConvCritic, MultiHeadConvCritic
from tarp.rl.policies.mlp_policies import ConvPolicy, MultiHeadConvPolicy
from tarp.rl.components.environment import GymEnv
from tarp.rl.components.normalization import Normalizer
from tarp.rl.components.sampler import ImageSampler, MultiImageSampler
from tarp.data.carla.carla import CarlaDataset
from tarp.configs.default_data_configs.carla import data_spec
from tarp.rl.components.replay_buffer import UniformReplayBuffer, LowMemoryUniformReplayBuffer, MultiTaskLowMemoryUniformReplayBuffer


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'used to test the RL implementation'

configuration = {
    'seed': 42,
    'agent': MultiHeadContinuousCQLAgent,
    'environment': GymEnv,
    'data_dir': '.',
    'num_epochs': 1000,
    'max_rollout_len': 1000,
    'n_steps_per_epoch': 20000,
    'n_warmup_steps': 0,
    'sampler': MultiImageSampler,
    'offline_rl': True,
    'load_offline_data': True
}
configuration = AttrDict(configuration)

# Policy
policy_params = AttrDict(
    action_dim=2,
    input_dim=0,
    n_layers=1,      # number of policy network layera
    action_space_type='continuous',
    output_distribution='gauss',
    max_action_range=1.,
    input_res=128,
    input_nc=9,
    nz_enc=256,
    nz_mid=1024,
    head_keys=['right', 'left', 'straight'],
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.dataset_class=CarlaDataset
data_config.dataset_spec.data_dir = os.path.join(os.environ['DATA_DIR'], './carla/expert128-town05_h5_L10')
data_config.dataset_spec.task_names = ['right', 'left', 'straight']
data_config.dataset_spec.discount_factor = 0.4
data_config.dataset_spec.subseq_len=8
data_config.dataset_spec.resolution=128

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
    n_frames=3,
    # split_frames=True,
    head_keys=policy_params.head_keys,
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
    batch_size=128,
    clip_q_target=False,
    min_q_weight=3.,
    num_random=1,
    deterministic_backup=True,
    reward_scale=10.,
    optimizer='rmsprop',
)

sampler_config = AttrDict(
    n_frames=3,
)

# Environment
env_config1 = AttrDict(
    name='carla-v0',
    task_name='right',
    task_json='./tarp/data/carla/json_data/town05/right.json',
    port=2012,
    tm_port=8012,
    num_cameras=1,
    resolution=128,
    screen_height=128,
    screen_width=128,
    head_key='right',
)

env_config2 = AttrDict(
    name='carla-v0',
    task_name='left',
    task_json='./tarp/data/carla/json_data/town05/left.json',
    port=2014,
    tm_port=8014,
    num_cameras=1,
    resolution=128,
    screen_height=128,
    screen_width=128,
    head_key='left'
)
env_config3 = AttrDict(
    name='carla-v0',
    task_name='straight',
    task_json='./tarp/data/carla/json_data/town05/straight.json',
    port=2016,
    tm_port=8016,
    num_cameras=1,
    resolution=128,
    screen_height=128,
    screen_width=128,
    head_key='straight'
)

env_config = AttrDict(
    conf_list=[env_config1, env_config2, env_config3],
    name='carla-v0'
)
