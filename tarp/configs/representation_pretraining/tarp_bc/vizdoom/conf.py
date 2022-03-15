import os

from tarp.models.tarp_bc_mdl import TARPBCModel
from tarp.components.logger import Logger
from tarp.utils.general_utils import AttrDict
from tarp.configs.default_data_configs.vizdoom import data_spec
from tarp.components.evaluator import ImageEvaluator, MultiImageEvaluator
from tarp.data.vizdoom.vizdoom_data_loader import VizdoomDataset


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': TARPBCModel,
    'model_test': TARPBCModel,
    'logger': Logger,
    'logger_test': Logger,
    'evaluator': MultiImageEvaluator,
    'data_dir': os.path.join(os.environ['DATA_DIR'], './vizdoom_L10'),
    'num_epochs': 100,
    'epoch_cycles_train': 3,
    'batch_size': 128,
    'discount_factor': 0.99,
    'n_frames': 4,
    'lr': 1e-4
}
configuration = AttrDict(configuration)

model_config = {
    'img_sz': 64,
    'nz_enc': 256,
    'nz_mid': 256,
    'input_nc': 4,
    'use_seg_mask': True,
    'n_class': 6,
    'n_action': 256,
    'normalization': 'none'
}

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.dataset_class = VizdoomDataset
data_config.dataset_spec.discount_factor = 0.4
data_config.dataset_spec.split = AttrDict(train=0.9, val=0.1, test=0.0)
data_config.dataset_spec.max_seq_len = 500
data_config.dataset_spec.crop_rand_subseq = True
data_config.dataset_spec.subseq_len = 2
data_config.dataset_spec.task_names = ['0_0-0_0-1_0', '0_0-1_0-0_0', '1_0-1_0--1_0']
data_config.dataset_spec.obj_labels = ['Clip', 'CustomMediKit', 'ZoomImp']
