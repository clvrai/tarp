import os

from tarp.models.tarp_heterogeneous_mdl import TARPRecurrentHeterogeneousModel
from tarp.components.logger import Logger
from tarp.utils.general_utils import AttrDict
from tarp.configs.default_data_configs.carla import data_spec
from tarp.components.evaluator import ImageEvaluator, MultiImageEvaluator


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': TARPRecurrentHeterogeneousModel,
    'model_test': TARPRecurrentHeterogeneousModel,
    'logger': Logger,
    'logger_test': Logger,
    'evaluator': MultiImageEvaluator,
    'data_dir': os.path.join(os.environ['DATA_DIR'], './carla/expert128-town05_h5_L10'),
    'num_epochs': 100,
    'epoch_cycles_train': 3,
    'batch_size': 16,
    'discount_factor': 0.99,
    'n_frames': 3,
    'lr': 1e-4
}
configuration = AttrDict(configuration)

model_config = {
    'img_sz': 128,
    'nz_enc': 256,
    'nz_mid': 256,
    'input_nc': 9,
    'normalization': 'none',
    'action_space_type': 'continuous',
    'n_action': 2,
    'data_source_maps': AttrDict({
        'right': 'value',
        'left': 'value',
        'straight': 'action'
    })
}

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.data_dir = configuration.data_dir
data_config.dataset_spec.task_names = ['right', 'left', 'straight']
data_config.dataset_spec.discount_factor = 0.4
data_config.dataset_spec.split_frac=0.9
data_config.dataset_spec.subseq_len=8
data_config.dataset_spec.resolution=128
data_config.dataset_spec.img_width = 128
data_config.dataset_spec.img_height =128
