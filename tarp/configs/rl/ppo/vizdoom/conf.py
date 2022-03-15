import os

from tarp.utils.general_utils import AttrDict, LinearSchedule
#from tarp.configs.default_data_configs.dummy_subskill_seq import data_spec
from tarp.rl.agents.ppo_agent import PPOAgent
from tarp.rl.policies.mlp_policies import ConvPolicy
from tarp.rl.components.critic import ConvCritic
from tarp.rl.components.replay_buffer import UniformReplayBuffer
from tarp.rl.components.environment import GymEnv
from tarp.rl.components.normalization import Normalizer, RewardNormalizer
from tarp.rl.components.sampler import ImageSampler


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'used to test the RL implementation'

configuration = {
    'seed': 42,
    'agent': PPOAgent,
    'environment': GymEnv,
    'data_dir': '.',
    'num_epochs': 100,
    'max_rollout_len': 2048,
    'n_steps_per_update': 2048,
    'n_steps_per_epoch': 100*2048,
    'sampler': ImageSampler
}
configuration = AttrDict(configuration)

# Policy
policy_params = AttrDict(
    action_dim=8,
    n_layers=1,      # number of policy network layera
    input_dim=0,
    input_res=64,
    nz_mid=256,
    nz_enc=256,
    max_action_range=1.,
    squash_log_prob=False,
    input_nc=3,
    action_space_type="discrete"
)

# Critic
critic_params = AttrDict(
    action_dim=policy_params.action_dim,
    input_dim=policy_params.input_dim,
    output_dim=1,
    n_layers=policy_params.n_layers,      # number of policy network layers
    nz_mid=policy_params.nz_mid,
    nz_enc=policy_params.nz_enc,
    action_input=False,
    input_res=policy_params.input_res
)

# Replay Buffer
replay_params = AttrDict(
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
    buffer=UniformReplayBuffer,
    buffer_params=replay_params,
    batch_size=128,
    clip_q_target=False,
    update_iterations=10*16,       # number of policy updates per environment batch collection
    gradient_clip=0.5,
    entropy_coefficient=0.,
    # obs_normalizer=Normalizer,
    clip_value_loss=False,
    # lr_schedule=LinearSchedule,
    # lr_schedule_params=AttrDict(initial_p=7e-4,
    #                             final_p=0.,
    #                             schedule_timesteps=1e6, )
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = AttrDict() #data_spec

sampler_config = AttrDict(
    rew_normalizer=RewardNormalizer
)
# Environment
env_config = AttrDict(
    name="VizdoomBattle-v0",
    reward_norm=1.,
    unwrap_time=False,
    objective_coef=[0.5, 0.5, 1.0]
)
