import os, sys
import h5py
import numpy as np
from multiprocessing import Pool

import os, sys
import tqdm
import h5py
import numpy as np

N_THREADS = 15
d = os.path.join(os.environ['DATA_DIR'], "./10tasks_distracting_metaworld")
d_new = d + "_h5"

# collect all h5 files in directory
filenames = []
for root, dirs, files in os.walk(d):
    for file in files:
        if file.endswith(".h5"): filenames.append(os.path.join(root, file))
n_files = len(filenames)
print("\nDone! Found {} files!".format(n_files))


def modify_batch(filenames):
    for file in tqdm.tqdm(filenames):
        with h5py.File(file, 'r') as f:
            traj = f['traj0']
            action = traj['actions'][()]
            reward = traj['rewards'][()]
            done = traj['dones'][()]
            states = traj['states'][()]
            images = traj['images'][()].astype(np.uint8)
            pad_mask = np.ones((len(images)))

        new_filename = os.path.join(d_new, file[len(d)+1:-3].replace('rollout', 'episode')+'.h5')
        new_dir = os.path.dirname(new_filename)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        with h5py.File(new_filename, 'w') as F:
            F['traj_per_file'] = 1
            F['traj0/action'] = action
            F['traj0/reward'] = reward
            F['traj0/done'] = done
            F['traj0/observation'] = images.transpose((0, 3, 1, 2))
            F['traj0/states'] = states
            F['traj0/pad_mask'] = pad_mask


chunk_size = int(np.floor(n_files / N_THREADS))
filename_chunks = [filenames[i:i + chunk_size] for i in range(0, n_files, chunk_size)]

p = Pool(N_THREADS)
p.map(modify_batch, filename_chunks)
# modify_batch(filenames)
