from tarp.configs.il.sqil.distracting_control.base_conf import *

policy_params.encoder_checkpoint=os.path.join(os.environ['EXP_DIR'], './model/bisim_mdl/distracting_control/walker/ddmc-bisim')
critic_params.encoder_checkpoint=os.path.join(os.environ['EXP_DIR'], './model/bisim_mdl/distracting_control/walker/ddmc-bisim')

env_config.name='walker'
env_config.task_name='run'
