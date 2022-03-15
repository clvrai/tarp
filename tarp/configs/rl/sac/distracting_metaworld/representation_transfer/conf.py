import os
from tarp.configs.rl.sac.distracting_metaworld.base_conf import *

policy_params.update(AttrDict(
    encoder_checkpoint=os.path.join(os.environ['EXP_DIR'], './model/')
))

critic_params.update(AttrDict(
    encoder_checkpoint=policy_params.encoder_checkpoint
))

env_config.update(AttrDict(
    name='button_press_wall',
    # name='plate_slide_back_side',
    # name='handle_press_side',
    # name='door_unlock'
    # name='coffee_pull'
    # name='pick_place_wall'
    # name='push_wall'
))
