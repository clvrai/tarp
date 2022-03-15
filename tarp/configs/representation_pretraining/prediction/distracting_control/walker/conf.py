import os

from tarp.models.prediction_mdl import PredictionMdl
from tarp.components.logger import Logger
from tarp.utils.general_utils import AttrDict
from tarp.configs.default_data_configs.dmcontrol import data_spec
from tarp.components.evaluator import TopOfNSequenceEvaluator, DummyEvaluator


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': PredictionMdl,
    'model_test': PredictionMdl,
    'logger': Logger,
    'logger_test': Logger,
    'evaluator': DummyEvaluator,
    'data_dir': os.path.join(os.environ['DATA_DIR'], './distracting_control/walker/expert_L10'),
    'num_epochs': 100,
    'epoch_cycles_train': 3,
    'batch_size': 128,
    'discount_factor': 0.99,
    'n_frames': 3,
    'lr': 1e-4
    # 'top_of_n_eval': 100,
    # 'top_comp_metric': 'mse',
}
configuration = AttrDict(configuration)

model_config = {
    'img_sz': 64,
    'nz_enc': 256,
    'nz_mid': 512,
    'nz_mid_lstm': 256,
    'nz_vae': 128,
    'ngf': 8,
    'input_nc': 9,
    'target_kl': 50.,
    'normalization': 'none',
    'predict_rewards': True
}

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.task_names = ['run-expert', 'stand-expert', 'backward-expert']
data_config.dataset_spec.discount_factor = 0.4
data_config.dataset_spec.subseq_len = 6  # == prediction_length + 1
