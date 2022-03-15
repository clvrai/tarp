from tarp.utils.general_utils import AttrDict
from tarp.data.carla.carla_data_loader import CarlaDataset


data_spec = AttrDict(
    resolution=64,
    task_names=[],
    dataset_class=CarlaDataset,
    split = AttrDict(train=0.9, val=0.1, test=0.0),
    max_seq_len = 500,
    discount_factor=0.99,
    crop_rand_subseq = True,
    dataset_size_per_task=None,
    subseq_len=2,
)

