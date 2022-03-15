import os

from tarp.utils.general_utils import AttrDict
from tarp.rl.agents.sac_agent import SACAgent
from tarp.rl.policies.mlp_policies import MLPPolicy, ConvPolicy
from tarp.rl.components.critic import MLPCritic, ConvCritic
from tarp.rl.components.replay_buffer import UniformReplayBuffer
from tarp.rl.components.environment import DMControlEnv
from tarp.rl.envs.distracting_control.distracting_control import DistractingControlEnv


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'used to test the RL implementation'

configuration = {
    'seed': 42,
    'agent': SACAgent,
    # 'environment': DMControlEnv,
    'environment': DistractingControlEnv,
    'data_dir': '.',
    'num_epochs': 50,
    'max_rollout_len': 1000,
    'n_steps_per_epoch': 2e4,
    'n_warmup_steps': 1e4,
}
configuration = AttrDict(configuration)

# Policy
policy_params = AttrDict(
    action_dim=6,
    input_dim=24,
    n_layers=1,      # number of policy network layera
    action_space_type='continuous',
    output_distribution='gauss',
    max_action_range=1.,
    nz_enc=128,
    nz_mid=1024,
)

# Critic
critic_params = AttrDict(
    action_dim=policy_params.action_dim,
    output_dim=1,
    input_dim=policy_params.input_dim,
    n_layers=1,      # number of policy network layers
    action_input=True,
    nz_enc=policy_params.nz_enc,
    nz_mid=policy_params.nz_mid,
)

# Replay Buffer
replay_params = AttrDict(
    capacity=1e6,
    # dump_replay=True,
    # decompose_init_frames=True
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
    batch_size=256,
    clip_q_target=False,
)

# Dataset - Random data
data_config = AttrDict()
sampler_config = AttrDict(
)

# Environment
env_config = AttrDict(
    name="walker",
    task_name='walk',
    reward_norm=1.,
    from_pixels=False,
    frame_skip=2
)

