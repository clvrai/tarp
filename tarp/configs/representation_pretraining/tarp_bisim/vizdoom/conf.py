import os

from tarp.models.tarp_bisim_mdl import TARPBisimModel
from tarp.components.logger import Logger
from tarp.utils.general_utils import AttrDict
from tarp.configs.default_data_configs.vizdoom import data_spec
from tarp.components.evaluator import DummyEvaluator, ImageEvaluator, MultiImageEvaluator
from tarp.data.vizdoom.vizdoom_data_loader import VizdoomDataset


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': TARPBisimModel,
    'model_test': TARPBisimModel,
    'logger': Logger,
    'logger_test': Logger,
    'evaluator': DummyEvaluator,
    'data_dir': os.path.join(os.environ['DATA_DIR'], './vizdoom_L10'),
    'num_epochs': 100,
    'epoch_cycles_train': 7,
    'batch_size': 128,
    'discount_factor': 0.99,
    'n_frames': 4,
    'lr': 1e-4,
}
configuration = AttrDict(configuration)

model_config = {
    'action_dim': 256,
    'img_sz': 64,
    'nz_enc': 256,
    'nz_mid': 256,
    'input_nc': 4,
    'normalization': 'none',
    'discount': configuration.discount_factor,
    'pred_weight': 1.,
    'action_space_type': 'discrete',
    'bisim_weight': 10
    #'target_network_update_factor': 1.,
}

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.dataset_class=VizdoomDataset
data_config.dataset_spec.delta_t = 1
data_config.dataset_spec.task_names = ['0_0-0_0-1_0', '0_0-1_0-0_0', '1_0-1_0--1_0']
