from tarp.configs.il.sqil.distracting_control.base_conf import *

# policy_params.encoder_checkpoint='./weights/model/prediction_mdl/distracting_control/walker/03.12/'
# critic_params.encoder_checkpoint='./weights/model/prediction_mdl/distracting_control/walker/03.12/'
policy_params.encoder_checkpoint=os.path.join(os.environ['EXP_DIR'], './model/prediction_mdl/distracting_control/walker/distracting-PredSR-run2')
critic_params.encoder_checkpoint=os.path.join(os.environ['EXP_DIR'], './model/prediction_mdl/distracting_control/walker/distracting-PredSR-run2')

env_config.name='walker'
env_config.task_name='run'