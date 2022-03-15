from tarp.utils.general_utils import AttrDict
from tarp.components.data_loader import RandomVideoDataset


data_spec = AttrDict(
    dataset_class=RandomVideoDataset,
    n_actions=2,
    state_dim=2,
    max_seq_len=20,
)
