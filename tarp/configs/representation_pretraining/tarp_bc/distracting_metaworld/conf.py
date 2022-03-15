import os

from tarp.models.tarp_bc_mdl import TARPBCModel
from tarp.components.logger import Logger
from tarp.utils.general_utils import AttrDict
from tarp.configs.default_data_configs.metaworld import data_spec
from tarp.components.evaluator import ImageEvaluator, MultiImageEvaluator


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': MultiActionModel,
    'model_test': MultiActionModel,
    'logger': Logger,
    'logger_test': Logger,
    'evaluator': MultiImageEvaluator,
    'data_dir': os.path.join(os.environ['DATA_DIR'], './distracting_metaworld_h5_L10'),
    'num_epochs': 100,
    'epoch_cycles_train': 4,
    'batch_size': 128,
    'discount_factor': 0.99,
    'n_frames': 1,
}
configuration = AttrDict(configuration)

model_config = {
    'img_sz': 128,
    'nz_enc': 256,
    'nz_mid': 256,
    'input_nc': 3,
    'normalization': 'none',
    'action_space_type': 'continuous',
    'n_action': 4
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
