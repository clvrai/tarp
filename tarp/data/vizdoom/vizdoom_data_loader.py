import numpy as np
from itertools import accumulate

from tarp.utils.general_utils import AttrDict
from tarp.components.data_loader import GlobalSplitVideoDataset
import tarp.utils.data_aug as aug


class VizdoomDataset(GlobalSplitVideoDataset):
    """Implements Vizdoom-specific data loader functions."""
    DATA_KEYS = ['states', 'actions', 'pad_mask', 'rewards', 'obj_labels', 'seg_targets', 'discounted_returns', 'timestep']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug_fcns = self.spec.augs if 'augs' in self.spec else AttrDict(no_aug=aug.no_aug)

    def _load_raw_data(self, data, F):
        super()._load_raw_data(data, F)

        # stack frames
        images = data.images
        if self.spec.n_frames > 1:
            stacked_len = images.shape[0] - (self.spec.n_frames - 1)
            images = np.concatenate([images[i:i+stacked_len] for i in range(self.spec.n_frames)], axis=1)
            for key in data:    # remove all unnecessary steps
                data[key] = data[key][-stacked_len:]
            data.images = images

        # add data augmentations
        if self.phase == 'train':
            combined_images = np.concatenate((images, data.seg_targets[:, None]), axis=1)
            for aug, func in self.aug_fcns.items():
                combined_images = func(combined_images)
            data.images, data.seg_targets = combined_images[:, :images.shape[1]], combined_images[:, -1]

        # bring images in required uint8 format, segmentations to long
        data.images = np.asarray(data.images.transpose(0, 2, 3, 1), dtype=np.uint8)
        # data.images = np.concatenate([data.images]*3, axis=-1)
        data.seg_targets = np.asarray(data.seg_targets, dtype=np.int64)

    def _get_aux_info(self, data, path):
        aux_info = AttrDict(super()._get_aux_info(data, path))

        # add task ID and color map
        aux_info.task_id = self._get_task_id(path)
        aux_info.color_map = np.asarray(self.spec.color_map)[None]
        return aux_info

    def _get_task_id(self, path):
        for i, task_name in enumerate(self.spec.task_names):
            if task_name in path:
                return i
        raise ValueError("No task found for file {}!".format(path))
