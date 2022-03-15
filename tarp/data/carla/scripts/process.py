import os, sys
import h5py
import numpy as np
from multiprocessing import Pool

import os, sys
import tqdm
import h5py
import numpy as np

N_THREADS = 5
d = os.path.joint(os.environ['DATA_DIR'], "./carla/expert128-town05-episodicWeather-300")
d_new = d + "_h5"

# collect all h5 files in directory
filenames = []
for root, dirs, files in os.walk(d):
    for file in files:
        if file.endswith(".h5"): filenames.append(os.path.join(root, file))
n_files = len(filenames)
print("\nDone! Found {} files!".format(n_files))

# def mask_to_label(image):
#     classes = {
#         (0, 0, 0): 0,         # None
#         # (70, 70, 70): 1,      # Buildings
#         # (100, 40, 40): 2,   # Fences
#         (55, 90, 80): 1,       # Other
#         (220, 20, 60): 2,     # Pedestrians
#         # (153, 153, 153): 5,   # Poles
#         (157, 234, 50): 3,    # RoadLines
#         (128, 64, 128): 4,    # Roads
#         # (244, 35, 232): 8,    # Sidewalks
#         # (107, 142, 35):9,    # Vegetation
#         (0, 0, 142):5,      # Vehicles
#         # (102, 102, 156):11,  # Walls
#         # (220, 220, 0):12,     # TrafficSigns
#         # (70, 128, 176): 13,
#         # (70, 122, 166): 13,
#         # (70, 124, 170): 13,     # Sky
#         # (70, 127, 176): 13,     # Sky
#         # (70, 128, 177): 13,     # Sky
#         # (70, 129, 179): 13,     # Sky
#         # (70, 130, 180): 13,     # Sky
#         # (81, 0, 81):14,     # Ground
#         # (150, 100, 100):15,     # Bridge
#         # (230, 150, 140):16,     # RailTrack
#         # (180, 165, 180):17,     # GuardRail
#         # (250, 170, 30):18,     # TrafficLight
#         # (110, 190, 160):19,     # Static
#         # (170, 120, 50):20,     # Dynamic
#         # (45, 60, 150):21,     # Water
#         # (145, 170, 100):22,     # Terrain
#     }
#     w, h, c = image.shape
#     target_seg = np.zeros((w, h))
#     for i in range(w):
#         for j in range(h):
#             key = (image[i, j][0], image[i, j][1], image[i, j][2])
#             if None == classes.get(key):
#                 target_seg[i, j] = 0.
#             else:
#                 target_seg[i, j] = classes.get(key)
#     return target_seg.astype(np.uint8)

MASK_IDS = [3, 4, 6, 7, 10]

def modify_batch(filenames):
    for file in tqdm.tqdm(filenames):
        with h5py.File(file, 'r') as f:
            traj = f['traj0']
            action = traj['actions'][()]
            reward = traj['rewards'][()]
            done = traj['dones'][()]
            images = traj['images'][()].astype(np.uint8)
            # mask_ids = traj['mask_ids'][()].astype(np.uint8)
            # masks = traj['masks'][()].astype(np.uint8)
            pad_mask = np.ones((len(images)))


        # target_segs = np.zeros_like(mask_ids)
        # for i, class_id in enumerate(MASK_IDS):
        #     target_segs[np.where(mask_ids==class_id)] = i+1
        # target_segs = []
        # for mask in masks:
        #     target_segs.append(mask_to_label(mask))
        # target_segs = np.array(target_segs)
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
            # F['traj0/states'] = states
            F['traj0/pad_mask'] = pad_mask
            # F['traj0/mask'] = masks
            # F['traj0/target_seg'] = target_segs


chunk_size = int(np.floor(n_files / N_THREADS))
filename_chunks = [filenames[i:i + chunk_size] for i in range(0, n_files, chunk_size)]

p = Pool(N_THREADS)
p.map(modify_batch, filename_chunks)
# modify_batch(filenames)
