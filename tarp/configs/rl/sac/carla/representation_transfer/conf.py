import os

from core.utils.general_utils import AttrDict
from core.rl.agents.sac_agent import SACAgent
from core.rl.policies.mlp_policies import MLPPolicy, ConvPolicy
from core.rl.components.critic import MLPCritic, ConvCritic
from core.rl.components.replay_buffer import UniformReplayBuffer, LowMemoryUniformReplayBuffer
from core.rl.components.environment import GymEnv
from core.rl.components.sampler import ImageSampler, MultiImageSampler
import core.utils.data_aug as aug


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'used to test the RL implementation'

configuration = {
    'seed': 42,
    'agent': SACAgent,
    'environment': GymEnv,
    'data_dir': '.',
    'num_epochs': 300,
    'max_rollout_len': 1000,
    'n_steps_per_epoch': 3e4,
    'n_warmup_steps': 3e3,
    'sampler': MultiImageSampler
}
configuration = AttrDict(configuration)

# Policy
policy_params = AttrDict(
    action_dim=2,
    input_dim=0,
    n_layers=3,      # number of policy network layera
    action_space_type='continuous',
    output_distribution='gauss',
    max_action_range=1.,
    input_res=128,
    input_nc=9,
    nz_enc=256,
    nz_mid=1024,
    policy_checkpoint=os.path.join(os.environ['EXP_DIR'], './model/')
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
    encoder_checkpoint=policy_params.encoder_checkpoint,
    finetune=True
)

# Replay Buffer
replay_params = AttrDict(
    capacity=3e5,
    dump_replay=True,
    split_frames=True,
    n_frames=3,
    # load_replay=True,
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
    reward_scale=10.,
    actor_update_freq=2,
    critic_target_update_freq=2
)

# Dataset - Random data
data_config = AttrDict()
sampler_config = AttrDict(
    n_frames=3
)

# Environment
env_config = AttrDict(
    name='carla-v0',
    task_json='./core/data/carla/json_data/town05/round.json',
    port=2000,
    tm_port=8000,
    resolution=128,
    screen_height=128,
    screen_width=128,
    num_cameras=1,
)
