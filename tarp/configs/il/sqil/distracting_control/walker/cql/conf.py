from tarp.configs.il.sqil.distracting_control.base_conf import *

policy_params.encoder_checkpoint="./weights/rl/cql/distracting_control/walker/03.13"
critic_params.encoder_checkpoint="./weights/rl/cql/distracting_control/walker/03.13"

env_config.name='walker'
env_config.task_name='run'
