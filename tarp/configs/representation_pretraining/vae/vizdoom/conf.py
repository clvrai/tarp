import os

from tarp.models.vae_mdl import VAE
from tarp.components.logger import Logger
from tarp.utils.general_utils import AttrDict
from tarp.configs.default_data_configs.vizdoom import data_spec
from tarp.components.evaluator import ImageEvaluator, MultiImageEvaluator
from tarp.data.vizdoom.vizdoom_data_loader import VizdoomDataset


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': VAE,
    'model_test': VAE,
    'logger': Logger,
    'logger_test': Logger,
    'evaluator': MultiImageEvaluator,
    'data_dir': os.path.join(os.environ['DATA_DIR'], './vizdoom_L10'),
    'num_epochs': 100,
    'epoch_cycles_train': 10,
    'batch_size': 32,
    'discount_factor': 0.99,
    'n_frames': 4,
    'lr': 3e-4,
}
configuration = AttrDict(configuration)

model_config = {
    'img_sz': 64,
    'nz_enc': 256,
    'nz_mid': 256,
    'nz_vae': 128,
    'ngf': 8,
    'input_nc': 4,
    'target_kl': 100.0,
    'normalization':'none'
    # 'fixed_beta': 1e-4,
}

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.dataset_class = VizdoomDataset
data_config.dataset_spec.discount_factor = 0.4
data_config.dataset_spec.task_names = ['0_0-0_0-1_0', '0_0-1_0-0_0', '1_0-1_0--1_0']
data_config.dataset_spec.obj_labels = ['Clip', 'CustomMediKit', 'ZoomImp']
data_config.dataset_spec.split = AttrDict(train=0.9, val=0.1, test=0.0)
data_config.dataset_spec.max_seq_len = 500
data_config.dataset_spec.crop_rand_subseq = True
data_config.dataset_spec.subseq_len = 2
