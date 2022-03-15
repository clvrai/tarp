import os

from tarp.models.prediction_mdl import PredictionMdl
from tarp.components.logger import Logger
from tarp.utils.general_utils import AttrDict
from tarp.configs.default_data_configs.metaworld import data_spec
from tarp.components.evaluator import TopOfNSequenceEvaluator, DummyEvaluator


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': PredictionMdl,
    'model_test': PredictionMdl,
    'logger': Logger,
    'logger_test': Logger,
    # 'evaluator': TopOfNSequenceEvaluator,
    'evaluator': DummyEvaluator,
    'data_dir': os.path.join(os.environ['DATA_DIR'], './distracting_metaworld_h5_L10'),
    'num_epochs': 100,
    'epoch_cycles_train': 3,
    'batch_size': 128,
    'discount_factor': 0.99,
    'n_frames': 1,
    'lr': 1e-4
    # 'top_of_n_eval': 100,
    # 'top_comp_metric': 'mse',
}
configuration = AttrDict(configuration)

model_config = {
    'img_sz': 128,
    'nz_enc': 256,
    'nz_mid': 256,
    'nz_mid_lstm': 256,
    'nz_vae': 128,
    'ngf': 8,
    'input_nc': 3,
    'target_kl': 50.,
    'normalization': 'none',
    'predict_rewards': True
}

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.task_names = ['button_press_topdown_wall', 'button_press', 'button_press_topdown',
                                       'plate_slide', 'plate_slide_back', 'plate_slide_side', 
                                       'handle_pull', 'handle_pull_side', 'handle_press',
                                       'door_open', 'door_lock',
                                       'coffee_button', 'coffee_push',
                                       'pick_out_of_hole', 'pick_place',
                                       'push', 'push_back']
data_config.dataset_spec.discount_factor = 0.4
data_config.dataset_spec.subseq_len = 6  # == prediction_length + 1
