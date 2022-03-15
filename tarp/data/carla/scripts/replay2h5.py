from multiprocessing import Pool
import h5py
import os, sys
import numpy as np
import tqdm
from tarp.utils.general_utils import AttrDict, RecursiveAverageMeter, ParamDict, listdict2dictlist

task_name = 'roundabout'
replay_dir = os.path.join(os.environ['EXP_DIR'], './rl/sac/carla/round_scratch/carla-state-round-straight-COL03-brake02')
d_new = os.path.join(os.environ['DATA_DIR'], './carla/tasks/',  task_name)

with h5py.File(os.path.join(replay_dir, 'replay', 'replay_buffer.hdf5'), 'r') as F:
    data = AttrDict()
    for key in F.keys():
        data[key] = F[key][:]

idxs = np.where(data['done'] == 1)[0]
start_idx = 0
episode_data_list = []
for end_idx in idxs:
    episode_data = AttrDict()
    for key in data.keys():
        episode_data[key] = data[key][start_idx:end_idx]
    episode_data_list.append(episode_data)
    start_idx = end_idx

for i, episode_data in enumerate(tqdm.tqdm(episode_data_list)):
    new_filename = os.path.join(d_new, 'episode_{}.h5'.format(i))
    new_dir = os.path.dirname(new_filename)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    with h5py.File(new_filename, 'w') as F:
        F['traj_per_file'] = 1
        F['traj0/images'] = np.asarray(episode_data['observation'], dtype=np.uint8)
        F['traj0/pad_mask'] = np.ones((len(episode_data['observation'])))
        for key in episode_data.keys():
            F['traj0/{}'.format(key)] = np.asarray(episode_data[key])
