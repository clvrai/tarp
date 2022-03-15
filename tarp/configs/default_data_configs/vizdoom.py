from tarp.utils.general_utils import AttrDict
from tarp.data.vizdoom.vizdoom import VizdoomDataset


data_spec = AttrDict(
    resolution=64,
    task_names=['0.5-0.5-1.0'],
    dataset_class=VizdoomDataset,
    split_frac=0.99,
    discount_factor=0.99,
    pre_transform_image_size=80,
    patch_len=5,
    color_map=[[128, 40, 40], [40, 40, 128], [0, 0, 128], [0, 0, 255], [0, 255, 0], [128, 128, 128]],
    delta_t=3,
    max_seq_len=500,
    split=AttrDict(train=0.9, val=0.1, test=0.0),
    crop_rand_subseq=True,
    subseq_len=2
)

