import os

from tarp.models.tarp_bc_mdl import TARPBCModel
from tarp.components.logger import Logger
from tarp.utils.general_utils import AttrDict
from tarp.configs.default_data_configs.dmcontrol import data_spec
from tarp.components.evaluator import ImageEvaluator, MultiImageEvaluator


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': TARPBCModel,
    'model_test': TARPBCModel,
    'logger': Logger,
    'logger_test': Logger,
    'evaluator': MultiImageEvaluator,
    'data_dir': os.path.join(os.environ['DATA_DIR'], './distracting_control/walker/expert_L10'),
    'num_epochs': 100,
    'epoch_cycles_train': 7,
    'batch_size': 128,
    'discount_factor': 0.99,
    'n_frames': 3,
}
configuration = AttrDict(configuration)

model_config = {
    'img_sz': 64,
    'nz_enc': 256,
    'nz_mid': 256,
    'input_nc': 9,
    'normalization': 'none',
    'action_space_type': 'continuous',
    'n_action': 6
}

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.task_names = ['run-expert', 'stand-expert', 'backward-expert']
data_config.dataset_spec.discount_factor = 0.4
