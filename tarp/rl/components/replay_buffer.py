import numpy as np
import gzip
import pickle
import h5py
import os
import copy
import gc
from collections import deque

from tarp.utils.general_utils import AttrDict, RecursiveAverageMeter, ParamDict, listdict2dictlist


class ReplayBuffer:
    """Stores arbitrary rollout outputs that are provided by AttrDicts."""
    def __init__(self, hp):
        # TODO upgrade to more efficient (vectorized) implementation of rollout storage
        self._hp = self._default_hparams().overwrite(hp)
        self._max_capacity = self._hp.capacity
        self._replay_buffer = None
        self._idx = None
        self._size = None       # indicates whether all slots in replay buffer were filled at least once

    def _default_hparams(self):
        default_dict = ParamDict({
            'capacity': 1e6,        # max number of experience samples
            'dump_replay': True,    # whether replay buffer gets dump upon checkpointing
        })
        return default_dict

    def append(self, experience_batch):
        """Appends the vals in the AttrDict experience_batch to the existing replay buffer."""
        if len(experience_batch['observation'][0].shape) == 3 and experience_batch['observation'][0].dtype == np.uint8:
            experience_batch['observation'] = (experience_batch['observation'] / (255./2.) - 1.0)
            experience_batch['observation_next'] = (experience_batch['observation_next'] / (255./2.) - 1.0)

        if self._replay_buffer is None:
            self._init(experience_batch)

        # compute indexing range
        n_samples = self._get_n_samples(experience_batch)
        idxs = np.asarray(np.arange(self._idx, self._idx + n_samples) % self._max_capacity, dtype=int)

        # add batch
        for key in self._replay_buffer:
            self._replay_buffer[key][idxs] = np.stack(experience_batch[key])

        # advance pointer
        self._idx = int((self._idx + n_samples) % self._max_capacity)
        self._size = int(min(self._size + n_samples, self._max_capacity))

    def sample(self, n_samples, filter=None):
        """Samples n_samples from the rollout_storage. Potentially can filter which fields to return."""
        raise NotImplementedError("Needs to be implemented by child class!")

    def get(self):
        """Returns complete replay buffer."""
        return self._replay_buffer

    def reset(self):
        """Deletes all entries from replay buffer and reinitializes."""
        del self._replay_buffer
        self._replay_buffer, self._idx, self._size = None, None, None

    def _init(self, example_batch):
        """Initializes the replay buffer fields given an example experience batch."""
        self._replay_buffer = AttrDict()
        for key in example_batch:
            example_element = example_batch[key][0]
            self._replay_buffer[key] = np.empty([int(self._max_capacity)] + list(example_element.shape),
                                                   dtype=example_element.dtype)
        self._idx = 0
        self._size = 0

    def save(self, save_dir):
        """Stores compressed replay buffer to file."""
        if not self._hp.dump_replay: return
        os.makedirs(save_dir, exist_ok=True)
        dataset = h5py.File(os.path.join(save_dir, "replay_buffer.hdf5"), 'w')
        for k in self._replay_buffer.keys():
            dataset.create_dataset(k, data=self._replay_buffer[k], compression='gzip')
        # with gzip.open(os.path.join(save_dir, "replay_buffer.zip"), 'wb') as f:
        #     pickle.dump(self._replay_buffer, f, protocol=4)
        np.save(os.path.join(save_dir, "idx_size.npy"), np.array([self._idx, self.size]))

    def load(self, save_dir):
        """Loads replay buffer from compressed disk file."""
        assert self._replay_buffer is None      # cannot overwrite existing replay buffer when loading
        with h5py.File(os.path.join(save_dir, 'replay_buffer.hdf5'), 'r') as f:
            data = AttrDict()
            for key in f.keys():
                data[key] = f[key][:]
        self.append(data)
        # self._replay_buffer.append(data)
        # with gzip.open(os.path.join(save_dir, "replay_buffer.zip"), 'rb') as f:
        #     self._replay_buffer = pickle.load(f)
        idx_size = np.load(os.path.join(save_dir, "idx_size.npy"))
        self._idx, self._size = int(idx_size[0]), int(idx_size[1])

    def _sample_idxs(self, n_samples, offset=0):
        assert n_samples <= self.size      # need enough samples in replay buffer
        assert isinstance(self.size, int)   # need integer-valued size
        idxs = np.random.choice(np.arange(self.size-offset), size=n_samples)
        return idxs

    @staticmethod
    def _get_n_samples(batch):
        """Retrieves the number of samples in batch."""
        for key in batch:
            return len(batch[key])

    @property
    def size(self):
        return self._size

    @property
    def capacity(self):
        return self._max_capacity

    def __contains__(self, key):
        return key in self._replay_buffer


class UniformReplayBuffer(ReplayBuffer):
    """Samples batch uniformly from all experience samples in the buffer."""
    def sample(self, n_samples, filter=None):
        idxs = self._sample_idxs(n_samples)

        sampled_transitions = AttrDict()
        for key in self._replay_buffer:
            if filter is None or key in filter:
                sampled_transitions[key] = self._replay_buffer[key][idxs]
        return sampled_transitions

class LowMemoryUniformReplayBuffer(ReplayBuffer):
    """Samples batch uniformly from all experience samples in the buffer."""
    def _default_hparams(self):
        default_dict = ParamDict({
            'n_frames': 1,
            'split_frames': False, # split frames
            'load_replay': False
        })
        return super()._default_hparams().overwrite(default_dict)

    def sample(self, n_samples, filter=None):
        idxs = self._sample_idxs(n_samples, offset=1)

        sampled_transitions = AttrDict()
        for key in self._replay_buffer:
            if filter is None or key in filter:
                if key == 'observation':
                    observation, observation_next = [], []
                    for idx in idxs:
                        past_frames = deque(maxlen=self._hp.n_frames)
                        for offset in reversed(range(self._hp.n_frames)):
                            if not past_frames:
                                [past_frames.append(self._replay_buffer[key][idx-offset]) for _ in range(self._hp.n_frames)]
                            if bool(self._replay_buffer['done'][idx-offset]) and offset > 0:
                                past_frames = deque(maxlen=self._hp.n_frames)
                            else:
                                past_frames.append(self._replay_buffer[key][idx-offset])
                        observation.append(np.concatenate(list(past_frames), axis=0))
                        past_frames.append(self._replay_buffer[key][idx+1])
                        observation_next.append(np.concatenate(list(past_frames), axis=0))
                    sampled_transitions[key] = np.array(observation)
                    sampled_transitions['observation_next'] = np.array(observation_next)
                else:
                    sampled_transitions[key] = self._replay_buffer[key][idxs]

        if len(self._replay_buffer['observation'][0].shape) == 3:
            sampled_transitions['observation'] = sampled_transitions['observation'] / (255./2.) - 1.0
            sampled_transitions['observation_next'] = sampled_transitions['observation_next'] / (255./2.) - 1.0

        return sampled_transitions

    def append(self, experience_batch):
        """Appends the vals in the AttrDict experience_batch to the existing replay buffer."""
        if len(experience_batch['observation'][0].shape) == 3 and experience_batch['observation'][0].dtype != np.uint8:
            experience_batch['observation'] = ((np.stack(experience_batch['observation']) + 1.0) * (255./2.)).astype(np.uint8)

        if self._replay_buffer is None:
            self._init(experience_batch)

        # compute indexing range
        n_samples = self._get_n_samples(experience_batch)
        idxs = np.asarray(np.arange(self._idx, self._idx + n_samples) % self._max_capacity, dtype=int)

        # add batch
        for key in self._replay_buffer:
            if key != 'observation_next':
                data = np.stack(experience_batch[key])
                if key == 'observation' and self._hp.split_frames and not self._hp.load_replay:
                    N, ch, _, _ = data.shape
                    ch = int(ch//self._hp.n_frames)
                    data = data[:, -ch:]
                self._replay_buffer[key][idxs] = data

        self._hp.load_replay = False
        # advance pointer
        self._idx = int((self._idx + n_samples) % self._max_capacity)
        self._size = int(min(self._size + n_samples, self._max_capacity))

    def _init(self, example_batch):
        """Initializes the replay buffer fields given an example experience batch."""
        self._replay_buffer = AttrDict()
        for key in example_batch:
            example_element = example_batch[key][0]
            if key != 'observation_next':
                sample_shape = example_element.shape
                if key == 'observation' and self._hp.split_frames and not self._hp.load_replay:
                    ch = int(sample_shape[0]//self._hp.n_frames)
                    sample_shape = (ch,) + sample_shape[1:]
                self._replay_buffer[key] = np.empty([int(self._max_capacity)] + list(sample_shape),
                                                       dtype=example_element.dtype)
        self._idx = 0
        self._size = 0

class AugmentedLowMemoryUniformReplayBuffer(LowMemoryUniformReplayBuffer):
    """Samples batch uniformly from all experience samples in the buffer."""
    def _default_hparams(self):
        default_dict = ParamDict({
            'n_frames': 1,
            'load_replay': False,
            'input_dim': 0,
            'channels': 3,
            'resolution': 64
        })
        return super()._default_hparams().overwrite(default_dict)

    def sample(self, n_samples, filter=None):
        idxs = self._sample_idxs(n_samples, offset=1)

        sampled_transitions = AttrDict()
        for key in self._replay_buffer:
            if filter is None or key in filter:
                if key == 'observation':
                    observation, observation_next = [], []
                    for idx in idxs:
                        past_frames = deque(maxlen=self._hp.n_frames)
                        for offset in reversed(range(self._hp.n_frames)):
                            if not past_frames:
                                [past_frames.append(self._replay_buffer[key][idx-offset]) for _ in range(self._hp.n_frames)]
                            if bool(self._replay_buffer['done'][idx-offset]) and offset > 0:
                                past_frames = deque(maxlen=self._hp.n_frames)
                            else:
                                past_frames.append(self._replay_buffer[key][idx-offset])
                        observation.append(np.concatenate(list(past_frames), axis=0))
                        past_frames.append(self._replay_buffer[key][idx+1])
                        observation_next.append(np.concatenate(list(past_frames), axis=0))

                    observation = np.stack(observation) / (255./2.) - 1.0
                    observation_next = np.stack(observation_next) / (255./2.) - 1.0
                    observation = np.concatenate([self._replay_buffer['states'][idxs], observation.reshape(n_samples, -1)], axis=1)
                    observation_next = np.concatenate([self._replay_buffer['states'][idxs+1], observation_next.reshape(n_samples, -1)], axis=1)
                    sampled_transitions[key] = np.array(observation)
                    sampled_transitions['observation_next'] = np.array(observation_next)
                elif key != 'states':
                    sampled_transitions[key] = self._replay_buffer[key][idxs]

        return sampled_transitions

    def append(self, experience_batch):
        """Appends the vals in the AttrDict experience_batch to the existing replay buffer."""
        if len(experience_batch['observation'][0].shape) == 1 and experience_batch['observation'][0].dtype != np.uint8:
            batch_size = len(experience_batch['observation'])
            image_observation = np.stack(experience_batch['observation'])[:, self._hp.input_dim:].reshape(batch_size,
                                                                                                self._hp.channels*self._hp.n_frames,
                                                                                                self._hp.resolution, self._hp.resolution)
            vector_observation = np.stack(experience_batch['observation'])[:, :self._hp.input_dim]
            image_observation = ((np.stack(image_observation) + 1.0) * (255./2.)).astype(np.uint8)
            experience_batch['observation'] = image_observation
            experience_batch['states'] = vector_observation

        if self._replay_buffer is None:
            self._init(experience_batch)

        # compute indexing range
        n_samples = self._get_n_samples(experience_batch)
        idxs = np.asarray(np.arange(self._idx, self._idx + n_samples) % self._max_capacity, dtype=int)

        # add batch
        for key in self._replay_buffer:
            if key != 'observation_next':
                data = np.stack(experience_batch[key])
                if key == 'observation' and not self._hp.load_replay:
                    data = data[:, -self._hp.channels:]
                self._replay_buffer[key][idxs] = data

        self._hp.load_replay = False
        # advance pointer
        self._idx = int((self._idx + n_samples) % self._max_capacity)
        self._size = int(min(self._size + n_samples, self._max_capacity))

    def _init(self, example_batch):
        """Initializes the replay buffer fields given an example experience batch."""
        self._replay_buffer = AttrDict()
        for key in example_batch:
            example_element = example_batch[key][0]
            if key != 'observation_next':
                sample_shape = example_element.shape
                if key == 'observation':
                    sample_shape = (self._hp.channels, self._hp.resolution, self._hp.resolution)
                    self._replay_buffer[key] = np.empty([int(self._max_capacity)] + list(sample_shape),
                                                        dtype=np.uint8)
                else:
                    self._replay_buffer[key] = np.empty([int(self._max_capacity)] + list(sample_shape),
                                                           dtype=example_element.dtype)
        self._idx = 0
        self._size = 0

class VideoUniformReplayBuffer(UniformReplayBuffer):
    def append(self, experience_batch):
        """Appends the vals in the AttrDict experience_batch to the existing replay buffer."""
        if len(experience_batch['observation'][0].shape) == 3 and experience_batch['observation'][0].dtype != np.uint8:
            experience_batch['observation'] = ((np.stack(experience_batch['observation']) + 1.0) * (255./2.)).astype(np.uint8)

        if self._replay_buffer is None:
            self._init(experience_batch)

        # compute indexing range
        n_samples = self._get_n_samples(experience_batch)
        idxs = np.asarray(np.arange(self._idx, self._idx + n_samples) % self._max_capacity, dtype=int)

        # add batch
        for key in self._replay_buffer:
            if key != 'observation_next':
                self._replay_buffer[key][idxs] = np.stack(experience_batch[key])

        # advance pointer
        self._idx = int((self._idx + n_samples) % self._max_capacity)
        self._size = int(min(self._size + n_samples, self._max_capacity))

    def _sample_idxs(self, n_samples, offset=0):
        assert n_samples <= self.size      # need enough samples in replay buffer
        assert isinstance(self.size, int)   # need integer-valued size
        choices = np.where(self._replay_buffer['done']==1.0)[0][:-1] + 1
        choices = choices.tolist()
        choices.insert(0, 0)
        self.choices = choices
        idxs = np.random.choice(choices, size=n_samples)
        return idxs

    def sample(self, n_samples, filter=None):
        idxs = self._sample_idxs(n_samples, offset=1)

        sampled_transitions = AttrDict()
        max_horizon = 0
        assert 'observation' in self._replay_buffer.keys(), "observation must be in keys"
        key = 'observation'
        observation, observation_next = [], []
        for idx in idxs:
            end_idx = np.where(self._replay_buffer['done']==1.0)[0][self.choices.index(idx)]
            vids = self._replay_buffer[key][idx:end_idx+1]
            if max_horizon < len(vids):
                max_horizon = len(vids)
            observation.append(vids)
        for i, obs in enumerate(observation):
            if len(obs) < max_horizon:
                observation[i] = np.concatenate((obs, np.zeros((max_horizon-len(obs), *obs.shape[1:]))))
        sampled_transitions[key] = np.array(observation)

        for key in self._replay_buffer:
            if (filter is None or key in filter) and key != 'observation':
                items = []
                for idx in idxs:
                    end_idx = np.where(self._replay_buffer['done']==1.0)[0][self.choices.index(idx)]
                    items.append(np.concatenate((self._replay_buffer[key][idx:end_idx+1], np.zeros((max_horizon-len(self._replay_buffer[key][idx:end_idx+1]), *self._replay_buffer[key][idx].shape)))))
                sampled_transitions[key] = np.array(items)


        if len(self._replay_buffer['observation'][0].shape) == 3:
            sampled_transitions['observation'] = sampled_transitions['observation'] / (255./2.) - 1.0
            # sampled_transitions['observation_next'] = sampled_transitions['observation_next'] / (255./2.) - 1.0

        return sampled_transitions


class MultiTaskLowMemoryUniformReplayBuffer(LowMemoryUniformReplayBuffer):
    """Samples batch uniformly from all experience samples in the buffer."""
    def _default_hparams(self):
        default_dict = ParamDict({
            'head_keys': ['main']
        })
        return super()._default_hparams().overwrite(default_dict)

    def _sample_idxs(self, n_samples, offset=0):
        assert n_samples <= self.size      # need enough samples in replay buffer
        assert isinstance(self.size, int)   # need integer-valued size
        splits = self._split_batch(n_samples)
        idx_list = []
        for i, portion in enumerate(splits):
            idx_list.append(np.random.choice(np.where(self._replay_buffer.task_ids[:self._size] == i)[0][:-1], size=portion))
        idxs = np.concatenate(idx_list)
        return idxs

    def _split_batch(self, n_samples):
        num_tasks = len(self._hp.head_keys)
        if n_samples % num_tasks == 0:
            return np.ones((num_tasks), dtype=int) * (n_samples//num_tasks)
        else:
            surplus = num_tasks - (n_samples % num_tasks)
            splits = []
            for i in range(num_tasks):
                if i >= surplus:
                    splits.append((n_samples//num_tasks) + 1)
                else:
                    splits.append((n_samples//num_tasks))
            return splits


class FilteredReplayBuffer(ReplayBuffer):
    """Has option to *not* store certain attributes in replay (eg to save memory by not storing images."""
    def _default_hparams(self):
        default_dict = ParamDict({
            'filter_keys': [],        # list of keys who's values should not get stored in replay
        })
        return default_dict

    def append(self, experience_batch):
        return super().append(AttrDict({k: v for (k,v) in experience_batch.items() if k not in self._hp.filter_keys}))


class FilteredUniormReplayBuffer(FilteredReplayBuffer, UniformReplayBuffer):
    def sample(self, n_samples, filter=None):
        return UniformReplayBuffer.sample(self, n_samples, filter)


class SplitObsReplayBuffer(ReplayBuffer):
    """Splits off unused part of observation before storing (eg to save memory by not storing images)."""
    def _default_hparams(self):
        default_dict = ParamDict({
            'unused_obs_size': None,    # dimensionality of split off observation part
            'discard_part': 'back',     # which part of observation to discard ['front', 'back']
        })
        return super()._default_hparams().overwrite(default_dict)

    def append(self, experience_batch):
        filtered_experience_batch = copy.deepcopy(experience_batch)
        if self._hp.discard_part == 'front':
            filtered_experience_batch.observation = [o[self._hp.unused_obs_size:] for o in filtered_experience_batch.observation]
            filtered_experience_batch.observation_next = [o[self._hp.unused_obs_size:] for o in filtered_experience_batch.observation_next]
        elif self._hp.discard_part == 'back':
            filtered_experience_batch.observation = [o[:-self._hp.unused_obs_size] for o in filtered_experience_batch.observation]
            filtered_experience_batch.observation_next = [o[:-self._hp.unused_obs_size] for o in filtered_experience_batch.observation_next]
        else:
            raise ValueError("Cannot parse discard_part parameter {}!".format(self._hp.discard_part))
        return super().append(filtered_experience_batch)


class SplitObsUniormReplayBuffer(SplitObsReplayBuffer, UniformReplayBuffer):
    def sample(self, n_samples, filter=None):
        return UniformReplayBuffer.sample(self, n_samples, filter)


class RolloutStorage:
    """Can hold multiple rollouts, can compute statistics over these rollouts."""
    def __init__(self):
        self.rollouts = []

    def append(self, rollout):
        """Adds rollout to storage."""
        self.rollouts.append(rollout)

    def rollout_stats(self):
        """Returns AttrDict of average statistics over the rollouts."""
        assert self.rollouts    # rollout storage should not be empty
        stats = RecursiveAverageMeter()
        for rollout in self.rollouts:
            stat = AttrDict(
                avg_reward=np.stack(rollout.reward).sum()
            )
            info = [list(filter(None, ele)) for ele in rollout.info]
            info = [ele for ele in info if ele]
            if info:
                info = listdict2dictlist([item for sublist in info for item in sublist])
                for key in info:
                    name = 'avg_' + key
                    stat[name] = np.array(info[key]).sum()
            stats.update(stat)
        return stats.avg

    def reset(self):
        del self.rollouts
        self.rollouts = []

    def get(self):
        return self.rollouts

    def __contains__(self, key):
        return self.rollouts and key in self.rollouts[0]






