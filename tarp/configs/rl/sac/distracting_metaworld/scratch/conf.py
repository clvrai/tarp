from tarp.configs.rl.sac.distracting_metaworld.base_conf import *
from tarp.rl.envs.distracting_metaworld import DistractingOverlayMetaWorldEnv
# Environment
env_config.update(AttrDict(
    # name='button_press_wall',
    # name='plate_slide_back_side',
    name='handle_press_side',
    # name='door_unlock'
    # name='coffee_pull'
    # name='pick_place_wall'
    # name='push_wall'
))

