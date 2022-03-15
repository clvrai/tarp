import os
import copy

from tarp.utils.general_utils import AttrDict
from tarp.rl.agents.cql_agent import ContinuousCQLAgent, MultiHeadContinuousCQLAgent
from tarp.rl.agents.sac_agent import MultiHeadSACAgent
from tarp.rl.components.critic import ConvCritic, MultiHeadConvCritic
from tarp.rl.policies.mlp_policies import ConvPolicy, MultiHeadConvPolicy
from tarp.rl.envs.distracting_metaworld import DistractingMetaWorldEnv
from tarp.rl.components.normalization import Normalizer
from tarp.rl.components.sampler import ImageSampler, MultiImageSampler
from tarp.data.metaworld.metaworld import MetaWorldDataset
from tarp.configs.default_data_configs.metaworld import data_spec
from tarp.rl.components.replay_buffer import UniformReplayBuffer, LowMemoryUniformReplayBuffer, MultiTaskLowMemoryUniformReplayBuffer


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'used to test the RL implementation'

configuration = {
    'seed': 42,
    'agent': MultiHeadContinuousCQLAgent,
    'environment': DistractingMetaWorldEnv,
    'data_dir': '.',
    'num_epochs': 100,
    'max_rollout_len': 150,
    'n_steps_per_epoch': 1500,
    'n_warmup_steps': 0,
    'offline_rl': True,
    'load_offline_data': True,
    'sampler': ImageSampler
}
configuration = AttrDict(configuration)

# Policy
policy_params = AttrDict(
    action_dim=4,
    input_dim=0,
    n_layers=1,      # number of policy network layera
    action_space_type='continuous',
    output_distribution='gauss',
    max_action_range=1.,
    input_res=128,
    input_nc=3,
    nz_enc=256,
    nz_mid=128,
    head_keys = ['button_press_topdown_wall', 'button_press', 'button_press_topdown',
                                       'plate_slide', 'plate_slide_back', 'plate_slide_side', 
                                       'handle_pull', 'handle_pull_side', 'handle_press',
                                       'door_open', 'door_lock',
                                       'coffee_button', 'coffee_push',
                                       'pick_out_of_hole', 'pick_place',
                                       'push', 'push_back']
)
data_config = AttrDict(
    dataset_spec=data_spec,
)
data_config.dataset_spec.data_dir = os.path.join(os.environ['DATA_DIR'], './distracting_metaworld_h5')
data_config.dataset_spec.dataset_prefix = os.path.join(os.environ['DATA_DIR'], './distracting_metaworld_h5')
data_config.dataset_spec.dataset_class=MetaWorldDataset
data_config.dataset_spec.task_names = policy_params.head_keys
data_config.dataset_spec.resolution=128
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
    capacity=1300000,
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
    # batch_size=256,
    clip_q_target=False,
    min_q_weight=10.,
    num_random=1,
    deterministic_backup=True,
    policy_lr=4e-4,
    critic_lr=4e-4,
    alpha_lr=4e-4,
    optimizer='rmsprop',
)

sampler_config = AttrDict(
)

# Environment
base_env_config = AttrDict(
    reward_norm=1.,
    resolution=128,
    screen_height=128,
    screen_width=128,
    from_pixels=True,
)

env_config_list = []
for key in policy_params.head_keys:
    env_conf = copy.deepcopy(base_env_config)
    env_conf.update(AttrDict(
        name=key,
        head_key=key
    ))
    env_config_list.append(env_conf)

env_config = AttrDict(
    conf_list=env_config_list,
    name='metaworld',
)

