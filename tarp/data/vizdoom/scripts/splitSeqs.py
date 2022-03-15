"""Runs over dataset and splits every sequence into subsequences of fixed length to allow for faster loading."""

from multiprocessing import Pool
import pickle as pkl
import h5py
import os
import numpy as np
import tqdm
from itertools import accumulate

from tarp.utils.general_utils import AttrDict


N_THREADS = 10
SEQ_LEN = 10
discount_factor = 0.4

DATA_KEYS = ['images', 'pad_mask', 'states', 'actions', 'rewards',
             'obj_labels', 'seg_targets']

d = os.getcwd()
d_new = d + "_L{}".format(SEQ_LEN)

# collect all h5 files in directory
filenames = []
for root, dirs, files in os.walk(d):
    for file in files:
        if file.endswith(".h5"): filenames.append(os.path.join(root, file))
n_files = len(filenames)
print("\nDone! Found {} files!".format(n_files))


def modify_batch(filenames):
    for file in tqdm.tqdm(filenames):
        data = AttrDict()
        with h5py.File(file, 'r') as f:
            for key in DATA_KEYS:
                data[key] = f['traj0/{}'.format(key)][()]

        # store data in N new files of shorter length
        n = data.images.shape[0] // SEQ_LEN
        inverse_rewards = data['rewards'][::-1]
        discounted_returns = np.array(list(accumulate(data['rewards'][::-1], lambda x, y: x*discount_factor + y))[::-1], dtype=np.float32)
        data['discounted_returns'] = discounted_returns
        for i in range(n):
            new_filename = os.path.join(d_new, file[len(d)+1:-3]) + '_{}.h5'.format(i)
            new_dir = os.path.dirname(new_filename)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            with h5py.File(new_filename, 'w') as F:
                F['traj_per_file'] = 1
                for key in data:
                    F['traj0/{}'.format(key)] = data[key][i*SEQ_LEN:(i+1)*SEQ_LEN]
                F['traj0/timestep'] = np.arange(i*SEQ_LEN, (i+1)*SEQ_LEN)


chunk_size = int(np.floor(n_files / N_THREADS))
filename_chunks = [filenames[i:i + chunk_size] for i in range(0, n_files, chunk_size)]

p = Pool(N_THREADS)
p.map(modify_batch, filename_chunks)
