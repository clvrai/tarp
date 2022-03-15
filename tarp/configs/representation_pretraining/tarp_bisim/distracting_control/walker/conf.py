import os

from tarp.models.tarp_bisim_mdl import TARPBisimModel
from tarp.components.logger import Logger
from tarp.utils.general_utils import AttrDict
from tarp.configs.default_data_configs.dmcontrol import data_spec
from tarp.components.evaluator import DummyEvaluator, ImageEvaluator, MultiImageEvaluator


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': TARPBisimModel,
    'model_test': TARPBisimModel,
    'logger': Logger,
    'logger_test': Logger,
    'evaluator': DummyEvaluator,
    'data_dir': os.path.join(os.environ['DATA_DIR'], './distracting_control/walker/expert_L10'),
    'num_epochs': 100,
    'epoch_cycles_train': 3,
    'batch_size': 128,
    'discount_factor': 0.99,
    'n_frames': 3,
    #'gradient_clip': 10,
    'lr': 1e-4,
}
configuration = AttrDict(configuration)

model_config = {
    'action_dim': 6,
    'img_sz': 64,
    'nz_enc': 256,
    'nz_mid': 256,
    'input_nc': 9,
    'normalization': 'none',
    'discount': configuration.discount_factor,
    'pred_weight': 1.,
    'bisim_weight': 0.1,
    #'target_network_update_factor': 1.,
}

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
# data_config.dataset_spec.task_names = ['walk-expert', 'stand-expert', 'backward-expert']
data_config.dataset_spec.task_names = ['run-expert', 'stand-expert', 'backward-expert']
