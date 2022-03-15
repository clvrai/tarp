import os

from tarp.models.vae_mdl import VAE
from tarp.components.logger import Logger
from tarp.utils.general_utils import AttrDict
from tarp.configs.default_data_configs.carla import data_spec
from tarp.components.evaluator import ImageEvaluator, MultiImageEvaluator

current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': VAE,
    'model_test': VAE,
    'logger': Logger,
    'logger_test': Logger,
    'evaluator': MultiImageEvaluator,
    'data_dir': os.path.join(os.environ['DATA_DIR'], './carla/expert128-town05_h5_L10'),
    'num_epochs': 100,
    'epoch_cycles_train': 4,
    'batch_size': 128,
    'discount_factor': 0.99,
    'n_frames': 3,
}
configuration = AttrDict(configuration)

model_config = {
    'img_sz': 128,
    'nz_enc': 256,
    'nz_mid': 256,
    'nz_vae': 128,
    'ngf': 8,
    'input_nc': 9,
    'normalization': 'none',
    'target_kl': 10.0,
    # 'use_wide_img': True,
    # 'img_width': 192,
    # 'img_height': 64
}

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.task_names = ['right', 'left', 'straight']
data_config.dataset_spec.discount_factor = 0.4
data_config.dataset_spec.resolution= 128
data_config.dataset_spec.img_width = 128
data_config.dataset_spec.img_height = 128
