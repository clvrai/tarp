import os

from tarp.models.contrastive_mdl import ContrastiveModel
from tarp.components.logger import Logger
from tarp.utils.general_utils import AttrDict
from tarp.configs.default_data_configs.carla import data_spec
from tarp.components.evaluator import ImageEvaluator, MultiImageEvaluator


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': ContrastiveModel,
    'model_test': ContrastiveModel,
    'logger': Logger,
    'logger_test': Logger,
    'evaluator': MultiImageEvaluator,
    'data_dir': os.path.join(os.environ['DATA_DIR'], './carla/expert128-town05_h5_L10'),
    'num_epochs': 100,
    'epoch_cycles_train': 3,
    'batch_size': 128,
    'discount_factor': 0.99,
    'n_frames': 3,
    # 'gradient_clip': 10,
    'lr': 1e-3,
}
configuration = AttrDict(configuration)

model_config = {
    'img_sz': 128,
    'nz_enc': 256,
    'nz_mid': 256,
    'input_nc': 9,
    'normalization': 'none'
}

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len=4
data_config.dataset_spec.task_names = ['right', 'left', 'straight']
