import os

from tarp.utils.general_utils import AttrDict
from tarp.rl.agents.sqil_agent import SQILAgent
from tarp.configs.default_data_configs.dmcontrol import data_spec
from tarp.data.dmcontrol.dmcontrol import DMControlDataset
from tarp.rl.policies.mlp_policies import MLPPolicy, ConvPolicy
from tarp.rl.components.critic import MLPCritic, ConvCritic
from tarp.rl.components.replay_buffer import UniformReplayBuffer, LowMemoryUniformReplayBuffer
from tarp.rl.components.environment import DMControlEnv
from tarp.rl.envs.distracting_control.distracting_control import DistractingControlEnv
from tarp.rl.components.sampler import ImageSampler, MultiImageSampler


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'used to test the RL implementation'

configuration = {
    'seed': 123,
    'agent': SQILAgent,
    'environment': DistractingControlEnv,
    'data_dir': '.',
    'num_epochs': 50,
    'max_rollout_len': 1000,
    'n_steps_per_epoch': 20000,
    'n_warmup_steps': 5e3,
    'sampler': MultiImageSampler
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
    nz_mid=1024,
    encoder_checkpoint=None
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
    encoder_checkpoint=policy_params.encoder_checkpoint
)

# Replay Buffer
replay_params = AttrDict(
    capacity=1e5,
    dump_replay=False,
    split_frames=True,
    n_frames=3
)

expert_replay_params = AttrDict(
    capacity=1e5,
    dump_replay=False,
    split_frames=False,
    n_frames=3
)
# Observation Normalization
obs_norm_params = AttrDict(
)

# Demo Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.dataset_class = DMControlDataset
data_config.dataset_spec.crop_rand_subseq = True
data_config.dataset_spec.subseq_len = 2
data_config.dataset_spec.task_names = ['run-expert']
data_config.dataset_spec.split = {'train':0.99,'val':0.01}
data_config.dataset_spec.split_frac = 0.99
data_config.dataset_spec.res = 64
data_config.dataset_spec.n_frames = 3
data_config.dataset_spec.data_dir=os.path.join(os.environ['DATA_DIR'], "./datasets/distracting_control/walker/expert_L10/")
data_config.dataset_spec.dataset_prefix=os.path.join(os.environ['DATA_DIR'], "./datasets/distracting_control/walker/expert_L10/")

# Agent
agent_config = AttrDict(
    policy=ConvPolicy,
    policy_params=policy_params,
    critic=ConvCritic,
    critic_params=critic_params,
    replay=LowMemoryUniformReplayBuffer,
    replay_params=replay_params,
    expert_replay_params=expert_replay_params,
    batch_size=128,
    clip_q_target=False,
    expert_data_conf=data_config,
    expert_data_path=os.path.join(os.environ['DATA_DIR'], 'datasets/distracting_control/expert_L10'),
)

# Dataset - Random data
data_config = AttrDict()
sampler_config = AttrDict(
    n_frames=3
)

# Environment
env_config = AttrDict(
    name=None,
    task_name=None,
    reward_norm=1.,
    from_pixels=True,
    frame_skip=2,
    resolution=64,
)

