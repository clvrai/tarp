import os

from tarp.utils.general_utils import AttrDict
from tarp.rl.agents.cql_agent import DiscreteCQLAgent, MultiHeadDiscreteCQLAgent
from tarp.rl.components.critic import ConvCritic, MultiHeadConvCritic
from tarp.rl.components.replay_buffer import UniformReplayBuffer, LowMemoryUniformReplayBuffer, MultiTaskLowMemoryUniformReplayBuffer
from tarp.rl.components.environment import GymEnv
from tarp.rl.components.normalization import Normalizer
from tarp.rl.components.sampler import ImageSampler, MultiGrayImageAugmentedSampler
from tarp.data.vizdoom.vizdoom import VizdoomDataset
from tarp.configs.default_data_configs.atari import dataset_spec


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'used to test the RL implementation'

configuration = {
    'seed': 42,
    'agent': MultiHeadDiscreteCQLAgent,
    'environment': GymEnv,
    'data_dir': '.',
    'num_epochs': 500,
    'max_rollout_len': 1000,
    'n_steps_per_epoch': 4000,
    'n_warmup_steps': 0,
    'sampler': MultiGrayImageAugmentedSampler,
    'offline_rl': True,
    'load_offline_data': True
}
configuration = AttrDict(configuration)

# Policy
policy_params = AttrDict(
)
data_config = AttrDict(
    dataset_spec=dataset_spec,
)
data_config.dataset_spec.data_dir = os.path.join(os.environ['DATA_DIR'], './vizdoom/')
data_config.dataset_spec.dataset_prefix = data_config.dataset_spec.data_dir
data_config.dataset_spec.task_names = ['0_0-0_0-1_0', '0_0-1_0-0_0', '1_0-1_0--1_0']
data_config.dataset_spec.dataset_class=VizdoomDataset
data_config.dataset_spec.resolution=64
data_config.dataset_spec.discount_factor=0.99

# Critic
critic_params = AttrDict(
    action_dim=256,
    output_dim=256,
    n_layers=1,      # number of policy network layers
    nz_enc=256,
    nz_mid=256,
    action_input=False,
    input_nc=4,
    input_res=64,
    head_keys=['0_0-0_0-1_0', '0_0-1_0-0_0', '1_0-1_0--1_0']
)

# Replay Buffer
replay_params = AttrDict(
    n_frames=4,
    dump_replay=False,
    head_keys=critic_params.head_keys
)

# Observation Normalization
obs_norm_params = AttrDict(
)

# Agent
agent_config = AttrDict(
    critic=MultiHeadConvCritic,
    critic_params=critic_params,
    replay=MultiTaskLowMemoryUniformReplayBuffer,
    replay_params=replay_params,
    batch_size=256,
    n_class=6,
    obj_labels=['Clip', 'CustomMedikit', 'ZoomImp'],
    use_aux_loss=True,
    n_processing_layers=3,
    color_map=[[128, 40, 40], [40, 40, 128], [0, 0, 128], [0, 0, 255], [0, 255, 0], [128, 128, 128]]
)

sampler_config = AttrDict(
    n_frames=4
)

# Environment
env_config1 = AttrDict(
    name="VizdoomD3Battle-v0",
    unwrap_time=False,
    objective_coef=[0.0, 0.0, 1.0],
    head_key='0_0-0_0-1_0',
)

env_config2 = AttrDict(
    name="VizdoomD3Battle-v0",
    unwrap_time=False,
    objective_coef=[0.0, 1.0, 0.0],
    head_key='0_0-1_0-0_0'
)
env_config3 = AttrDict(
    name="VizdoomD3Battle-v0",
    unwrap_time=False,
    objective_coef=[1.0, 1.0, -1.0],
    head_key='1_0-1_0--1_0'
)

env_config = AttrDict(
    conf_list = [env_config1, env_config2, env_config3],
    name="VizdoomD3Battle-v0",
)
