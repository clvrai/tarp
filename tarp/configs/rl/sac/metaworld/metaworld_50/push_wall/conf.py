from tarp.configs.rl.sac.metaworld.base_conf import *

agent_config.reward_scale = 10.

# Environment
env_config.update(AttrDict(
    name="push_wall",
))
