"""Runs over dataset and stores every pkl file into a h5 file."""

from multiprocessing import Pool
import pickle as pkl
import h5py
import os
import numpy as np
import tqdm


N_THREADS = 10


d = os.getcwd()
d_new = d + "_h5"

# collect all h5 files in directory
filenames = []
for root, dirs, files in os.walk(d):
    for file in files:
        if file.endswith(".pkl"): filenames.append(os.path.join(root, file))
n_files = len(filenames)
print("\nDone! Found {} files!".format(n_files))


def modify_batch(filenames):
    for file in tqdm.tqdm(filenames):
        with open(file, 'rb') as f:
            data = pkl.load(f)

        seq_len = len(data['obs'])
        new_filename = os.path.join(d_new, file[len(d)+1:-3]) + 'h5'
        new_dir = os.path.dirname(new_filename)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        with h5py.File(new_filename, 'w') as F:
            F['traj_per_file'] = 1
            F['traj0/images'] = np.asarray(data['obs'], dtype=np.uint8)
            F['traj0/pad_mask'] = np.ones((seq_len,))
            F['traj0/states'] = np.zeros_like(np.asarray(data['acts']))
            F['traj0/actions'] = np.asarray(data['acts'])
            F['traj0/rewards'] = np.asarray(data['rwrds'])
            F['traj0/obj_labels'] = np.asarray(data['obj_labels'])
            F['traj0/seg_targets'] = np.asarray(np.asarray(data['seg_targets']).argmax(axis=1), dtype=np.uint8)


chunk_size = int(np.floor(n_files / N_THREADS))
filename_chunks = [filenames[i:i + chunk_size] for i in range(0, n_files, chunk_size)]

p = Pool(N_THREADS)
p.map(modify_batch, filename_chunks)
