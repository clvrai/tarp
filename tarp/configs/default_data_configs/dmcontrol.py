from tarp.utils.general_utils import AttrDict
from tarp.data.dmcontrol.dmcontrol_data_loader import DMControlDataset


data_spec = AttrDict(
    resolution=64,
    task_names=[],
    dataset_class=DMControlDataset,
    split = AttrDict(train=0.9, val=0.1, test=0.0),
    max_seq_len = 500,
    discount_factor=0.99,
    dataset_size_per_task=None,
    crop_rand_subseq = True,
    # delta_t=1,
    subseq_len=2,
)

