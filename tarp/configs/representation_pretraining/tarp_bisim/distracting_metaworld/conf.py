import os

from tarp.models.tarp_bisim_mdl import TARPBisimModel
from tarp.components.logger import Logger
from tarp.utils.general_utils import AttrDict
from tarp.configs.default_data_configs.metaworld import data_spec
from tarp.components.evaluator import DummyEvaluator, ImageEvaluator, MultiImageEvaluator


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': TARPBisimModel,
    'model_test': TARPBisimModel,
    'logger': Logger,
    'logger_test': Logger,
    'evaluator': DummyEvaluator,
    'data_dir': os.path.join(os.environ['DATA_DIR'], './distracting_metaworld_h5_L10'),
    'num_epochs': 100,
    'epoch_cycles_train': 3,
    'batch_size': 128,
    'discount_factor': 0.99,
    'n_frames': 1,
    'lr': 1e-4,
}
configuration = AttrDict(configuration)

model_config = {
    'action_dim': 4,
    'img_sz': 128,
    'nz_enc': 256,
    'nz_mid': 256,
    'input_nc': 3,
    'normalization': 'none',
    'discount': configuration.discount_factor,
    'pred_weight': 1.,
    # 'bisim_weight': 0.1,
    # 'bisim_weight': 1.0,
    'bisim_weight': 0.01,
    #'target_network_update_factor': 1.,
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
