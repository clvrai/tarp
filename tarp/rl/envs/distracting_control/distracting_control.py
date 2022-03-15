import sys, os

from tarp.utils.general_utils import AttrDict
import copy
import cv2
import collections
from contextlib import contextmanager
from functools import partial
import torch
import numpy as np
from tarp.rl.components.environment import DMControlEnv
from tarp.utils.general_utils import ParamDict, AttrDict, map_recursive
from tarp.utils.pytorch_utils import ar2ten, ten2ar
from dm_control.mujoco.wrapper import mjbindings

DAVIS17_TRAINING_VIDEOS = [
    'bear', 'bmx-bumps', 'boat', 'boxing-fisheye', 'breakdance-flare', 'bus',
    'car-turn', 'cat-girl', 'classic-car', 'color-run', 'crossing',
    'dance-jump', 'dancing', 'disc-jockey', 'dog-agility', 'dog-gooses',
    'dogs-scale', 'drift-turn', 'drone', 'elephant', 'flamingo', 'hike',
    'hockey', 'horsejump-low', 'kid-football', 'kite-walk', 'koala',
    'lady-running', 'lindy-hop', 'longboard', 'lucia', 'mallard-fly',
    'mallard-water', 'miami-surf', 'motocross-bumps', 'motorbike', 'night-race',
    'paragliding', 'planes-water', 'rallye', 'rhino', 'rollerblade',
    'schoolgirls', 'scooter-board', 'scooter-gray', 'sheep', 'skate-park',
    'snowboard', 'soccerball', 'stroller', 'stunt', 'surf', 'swing', 'tennis',
    'tractor-sand', 'train', 'tuk-tuk', 'upside-down', 'varanus-cage', 'walking'
]
DAVIS17_VALIDATION_VIDEOS = [
    'bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel',
    'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog', 'dogs-jump',
    'drift-chicane', 'drift-straight', 'goat', 'gold-fish', 'horsejump-high',
    'india', 'judo', 'kite-surf', 'lab-coat', 'libby', 'loading', 'mbike-trick',
    'motocross-jump', 'paragliding-launch', 'parkour', 'pigs', 'scooter-black',
    'shooting', 'soapbox'
]
SKY_TEXTURE_INDEX = 0
Texture = collections.namedtuple('Texture', ('size', 'address', 'textures'))
DIFFICULTY_SCALE = dict(easy=0.1, medium=0.2, hard=0.3)
DIFFICULTY_NUM_VIDEOS = dict(easy=4, medium=8, hard=None)

class DistractingControlEnv(DMControlEnv):

    def __init__(self, config):
        super().__init__(config)
        self._random_state = np.random.RandomState(self._hp.seed)

    def _default_hparams(self):
        default_dict = ParamDict({
            "background_path": os.path.join(os.environ['DATA_DIR'], "./dmcontrol/background/DAVIS/JPEGImages/480p/"),
            "background_dataset_videos": "train",
            "difficulty": None,
            "num_videos": None,
            "ground_plane_alpha": 0.5,  # reacher: 0.0, walker: 1.0, cheetah: 1.0, others: 0.3
            "shuffle_buffer_size": None,
            "dynamic": True,
        })

        return super()._default_hparams().overwrite(default_dict)

    def _load_background(self):
        num_videos = self._hp.num_videos
        if self._hp.difficulty:
            num_videos = DIFFICULTY_NUM_VIDEOS[self._hp.difficulty]

        if not self._hp.background_path or num_videos == 0:
            self._video_paths = []
        else:
            if not self._hp.background_dataset_videos:
                dataset_videos = sorted(os.listdir(self._hp.background_path))
            elif self._hp.background_dataset_videos in ['train', 'training']:
                dataset_videos = DAVIS17_TRAINING_VIDEOS
            elif self._hp.background_dataset_videos in ['val', 'validation']:
                dataset_videos = DAVIS17_VALIDATION_VIDEOS

            video_paths = [
                os.path.join(self._hp.background_path, subdir) for subdir in dataset_videos
            ]

            if num_videos is not None:
                if num_videos > len(video_paths) or num_videos < 0:
                    raise ValueError(f'`num_bakground_paths` is {num_videos} but '
                               'should not be larger than the number of available '
                               f'background paths ({len(video_paths)}) and at '
                               'least 0.')
                video_paths = video_paths[:num_videos]

            self._video_paths = video_paths

    def reset(self):
        obs = super().reset()
        self._reset_background()
        return obs

    def _blend_to_background(self, image, background):
        if self._hp.ground_plane_alpha == 1.0:
            return image
        elif self._hp.ground_plane_alpha == 0.0:
            return background
        else:
            return (self._hp.ground_plane_alpha * image.astype(np.float32)
                    + (1. - self._hp.ground_plane_alpha) * background.astype(np.float32)).astype(np.uint8)


    def _reset_background(self):
        if self._hp.ground_plane_alpha is not None:
            self._env.physics.named.model.mat_rgba['grid',
                                                   'a'] = self._hp.ground_plane_alpha
        self._env.physics.model.tex_height[SKY_TEXTURE_INDEX] = 800

        sky_height = self._env.physics.model.tex_height[SKY_TEXTURE_INDEX]
        sky_width = self._env.physics.model.tex_width[SKY_TEXTURE_INDEX]
        sky_size = sky_height * sky_width * 3
        sky_address = self._env.physics.model.tex_adr[SKY_TEXTURE_INDEX]

        sky_texture = self._env.physics.model.tex_rgb[sky_address:sky_address + sky_size].astype(np.float32)
        if self._video_paths:
            if self._hp.shuffle_buffer_size:
                file_names = [
                    os.path.join(path, fn.decode('utf-8'))
                    for path in self._video_paths
                    for fn in os.listdir(path)
                ]
                self._random_state.shuffle(file_names)
                file_names = file_names[:self._hp.shuffle_buffer_size]
                images = [cv2.imread(fn) for fn in file_names]
            else:
                video_path = self._random_state.choice(self._video_paths)
                file_names = os.listdir(video_path)
                file_names = [fn.decode('utf-8') if not isinstance(fn, str) else fn for fn in file_names]
                if not self._hp.dynamic:
                    file_names = [self._random_state.choice(file_names)]
                images = [cv2.imread(os.path.join(video_path, fn)) for fn in file_names]

            self._current_img_index = self._random_state.choice(len(images))
            self._step_direction = self._random_state.choice([-1, 1])

            texturized_images = []
            for image in images:
                image_flattened = cv2.resize(image, (sky_height,
                                                     sky_width)).reshape(-1)
                new_texture = self._blend_to_background(image_flattened, sky_texture)
                texturized_images.append(new_texture)
        else:
            self._current_img_index = 0
            texturized_images = [sky_texture]

        self._background = Texture(sky_size, sky_address, texturized_images)
        self._apply()

    def _apply(self):
        if self._background:
            start = self._background.address
            end = self._background.address + self._background.size
            texture = self._background.textures[self._current_img_index]

            self._env.physics.model.tex_rgb[start:end] = texture

            with self._env.physics.contexts.gl.make_current() as ctx:
                ctx.call(
                    mjbindings.mjlib.mjr_uploadTexture,
                    self._env.physics.model.ptr,
                    self._env.physics.contexts.mujoco.ptr,
                    SKY_TEXTURE_INDEX
                )


    def step(self, action):
        if isinstance(action, torch.Tensor): action = ten2ar(action)
        try:
            obs, reward, done, info = self._env.step(action)
            if self._hp.from_pixels:
                obs = cv2.resize(obs.transpose((1, 2, 0)), (self._hp.resolution, self._hp.resolution))
                obs = obs / 255.

            reward = reward / self._hp.reward_norm

            if self._hp.dynamic and self._video_paths:
                self._current_img_index += self._step_direction

                # start moving forward if we are past the start of the images.
                if self._current_img_index <= 0:
                    self._current_img_index = 0
                    self._step_direction = abs(self._step_direction)
                # start moving backwards if we are past the end of the images.
                if self._current_img_index >= len(self._background.textures):
                    self._current_img_index = len(self._background.textures) - 1
                    self._step_direction = -abs(self._step_direction)
                self._apply()

        except self._mj_except:
            # this can happen when agent drives simulation to unstable region (e.g. very fast speeds)
            print("Catch env exception!")
            obs = self.reset()
            reward = self._hp.punish_reward     # this avoids that the agent is going to these states again
            done = np.array(True)        # terminate episode (observation will get overwritten by env reset)
            info = {}


        return self._wrap_observation(obs), np.array(reward, dtype=np.float64), np.array(done), info

    def _make_env(self, id):
        # check later
        env = super()._make_env(id)
        self._load_background()
        return env

