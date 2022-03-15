import os

from tarp.models.tarp_bisim_mdl import TARPBisimModel
from tarp.components.logger import Logger
from tarp.utils.general_utils import AttrDict
from tarp.configs.default_data_configs.carla import data_spec
from tarp.components.evaluator import DummyEvaluator, ImageEvaluator, MultiImageEvaluator
from tarp.data.src.data_loaders import DMControlRescaleDataset


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': TARPBisimModel,
    'model_test': TARPBisimModel,
    'logger': Logger,
    'logger_test': Logger,
    'evaluator': DummyEvaluator,
    'data_dir': os.path.join(os.environ['DATA_DIR'], './carla/expert128-town05_h5_L10'),
    'num_epochs': 100,
    'epoch_cycles_train': 4,
    'batch_size': 128,
    'discount_factor': 0.99,
    'n_frames': 3,
    #'gradient_clip': 10,
    'lr': 1e-4,
}
configuration = AttrDict(configuration)

model_config = {
    'action_dim': 2,
    'img_sz': 128,
    'nz_enc': 256,
    'nz_mid': 256,
    'input_nc': 9,
    'normalization': 'none',
    'discount': configuration.discount_factor,
    'pred_weight': 1.,
    'bisim_weight': 0.01,
    #'target_network_update_factor': 1.,
}
# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.task_names = ['right', 'left', 'straight']
data_config.dataset_spec.discount_factor = 0.4

