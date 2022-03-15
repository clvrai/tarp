from tarp.configs.il.sqil.distracting_control.base_conf import *

policy_params.encoder_checkpoint='./weights/model/atc_mdl/distracting_control/walker/03.12/'
critic_params.encoder_checkpoint='./weights/model/atc_mdl/distracting_control/walker/03.12/'

env_config.name='walker'
env_config.task_name='run'
