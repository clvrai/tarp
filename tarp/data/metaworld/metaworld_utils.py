# code for handling metaworld environment for visualization purpose

# from metaworld.benchmarks import ML1
import numpy as np
import os
import glob
from collections import OrderedDict
import h5py
from tarp.utils.general_utils import AttrDict

# from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
# from metaworld.envs.mujoco.env_list import MT50_V2_ARGS_KWARGS, ALL_V2_ENVIRONMENTS, ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

DEFAULT_PIXEL_RENDER_KWARGS = {
    'mode': 'rgb_array',
    'width': 200,
    'height': 200,
}

INTERPOLATION_PIXEL_RENDER_KWARGS = {
    'mode': 'rgb_array',
    'width': 200,
    'height': 200,
}

cls_dict = {**ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE}
env_list = [k for k in cls_dict.keys()]

# task_args = {**MT50_V2_ARGS_KWARGS}
# for key in task_args:
#     task_args[key]['obs_type'] = 'with_goal'

# MULTITASK_ENV = MultiClassMultiTaskEnv(
#             task_env_cls_dict=cls_dict,
#             task_args_kwargs=task_args,
#             sample_goals=True,
#             obs_type='with_goal',
#             sample_all=True)
#
class MetaWorldEnvHandler:
    """Handles environments + IDs of MetaWorld environments."""

    ENVs = OrderedDict({
        'assembly': 'assembly-v2',
        'bin_picking': 'bin-picking-v2',
        'box_close': 'box-close-v2',
        'button_press': 'button-press-v2',
        'button_press_topdown': 'button-press-topdown-v2',
        'dial_turn': 'dial-turn-v2',
        'door_open': 'door-open-v2',
        'door_close': 'door-close-v2',
        'drawer_close': 'drawer-close-v2',
        'drawer_open': 'drawer-open-v2',
        'hammer': 'hammer-v2',
        'hand_insert': 'hand-insert-v2',
        'lever_pull': 'lever-pull-v2',
        'peg_insert_side': 'peg-insert-side-v2',
        'pick_place': 'pick-place-v2',
        'shelf_place': 'shelf-place-v2',
        'stick_pull': 'stick-pull-v2',
        'stick_push': 'stick-push-v2',
        'sweep': 'sweep-v2',
        'sweep_into': 'sweep-into-v2',
        'window_close': 'window-close-v2',
        'window_open': 'window-open-v2',
        'coffee_button': 'coffee-button-v2',
        'coffee_push': 'coffee-push-v2',
        'coffee_pull': 'coffee-pull-v2',
        'faucet_open': 'faucet-open-v2',
        'faucet_close': 'faucet-close-v2',
        'peg_unplug_side': 'peg-unplug-side-v2',
        'soccer': 'soccer-v2',
        'basketball': 'basketball-v2',
        'pick_place_wall': 'pick-place-wall-v2',
        'push' : 'push-v2',
        'reach': 'reach-v2',
        'push_wall': 'push-wall-v2',
        'reach_wall' : 'reach-wall-v2',
        'push_back': 'push-back-v2',
        'pick_out_of_hole': 'pick-out-of-hole-v2',
        'shelf_remove': 'shelf-place-v2', # hack, env not in metaworld
        'disassemble': 'disassemble-v2',
        'door_lock': 'door-lock-v2',
        'door_unlock': 'door-unlock-v2',
        'sweep_tool': 'sweep-v2', # hack, env not in metaworld
        'button_press_wall': 'button-press-wall-v2',
        'button_press_topdown_wall': 'button-press-topdown-wall-v2',
        'handle_press_side': 'handle-press-side-v2',
        'handle_press' : 'handle-press-v2',
        'handle_pull_side': 'handle-pull-side-v2',
        'handle_pull' : 'handle-pull-v2',
        'plate_slide': 'plate-slide-v2',
        'plate_slide_back': 'plate-slide-back-v2',
        'plate_slide_side': 'plate-slide-side-v2',
        'plate_slide_back_side': 'plate-slide-back-side-v2',
    })

    @staticmethod
    def get_metaenv(name):
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[MetaWorldEnvHandler.ENVs[name]+'-goal-observable']()
        return env
        # task_id = env_list.index(MetaWorldEnvHandler.ENVs[name])
        # sampled_goal = MULTITASK_ENV._task_envs[task_id].sample_goals_(1)[0]
        # MULTITASK_ENV.set_task(dict(task=task_id, goal=sampled_goal))
        # return MULTITASK_ENV

    @staticmethod
    def get_metatask_id(id):
        return env_list.index(MetaWorldEnvHandler.ENVs[MetaWorldEnvHandler.id2name(id)])

    @staticmethod
    def name2id(name):
        """Returns env ID given name."""
        return list(MetaWorldEnvHandler.ENVs).index(name)

    @staticmethod
    def id2name(id):
        """Returns name for a given environment ID."""
        return list(MetaWorldEnvHandler.ENVs)[id]

    @staticmethod
    def env_from_id(id):
        return MetaWorldEnvHandler.get_metaenv(MetaWorldEnvHandler.id2name(id))

    @staticmethod
    def env_from_name(name):
        return MetaWorldEnvHandler.get_metaenv(name)

    @staticmethod
    def __call__(identifier):
        """Returns environment given either name or id as identifier."""
        if isinstance(identifier, int):
            return MetaWorldEnvHandler.env_from_id(identifier)
        else:
            return MetaWorldEnvHandler.env_from_name(identifier)

    @staticmethod
    def num_envs():
        """Returns number of environments."""
        return len(list(MetaWorldEnvHandler.ENVs))


def _make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_video(video_frames, filename, fps=60, video_format='mp4'):
    assert fps == int(fps), fps
    import skvideo.io
    _make_dir(filename)

    skvideo.io.vwrite(
        filename,
        video_frames,
        inputdict={
            '-r': str(int(fps)),
        },
        outputdict={
            '-f': video_format,
            '-pix_fmt': 'yuv420p', # '-pix_fmt=yuv420p' needed for osx https://github.com/scikit-video/scikit-video/issues/74
        }
    )


# def _set_env():
#     env = ML1.get_train_tasks('reach-v2')
#     tasks = env.sample_tasks(1)
#     env.set_task(tasks[0])
#     env.reset()
#     return env


def save_videos_from_actions(model_output, model_input, log_dir, step, phase, kl_weight, logger, save2mp4):

    [bs, seq_len, action_dim] = model_output.shape
    temp_bs = 5

    permute = np.random.permutation(bs)
    output_video = []
    input_video = []
    for b in range(temp_bs):
        if phase == 'train':
            # input video
            input_sequence = model_input[permute[b]]
            env = _set_env()
            input_images = []
            for i in range(seq_len):
                action = input_sequence[i]
                obs, reward, done, info = env.step(action)
                image = env.render(**DEFAULT_PIXEL_RENDER_KWARGS)
                input_images.append(image)
            input_video.append(input_images)

        output_sequence = model_output[permute[b]]
        env = _set_env()
        output_images = []
        for i in range(seq_len):
            action = output_sequence[i]
            obs, reward, done, info = env.step(action)
            image = env.render(**DEFAULT_PIXEL_RENDER_KWARGS)
            output_images.append(image)
        output_video.append(output_images)

    if phase == 'train':
        input_video_ct = np.concatenate(input_video, axis=2) # input_video_ct: seq_len x height x (3*width) x 3
        output_video_ct = np.concatenate(output_video, axis=2) # output_video_ct: seq_len x height x (3*width) x 3
        combined_ct_video = np.concatenate((np.expand_dims(input_video_ct, axis=0),
                                            np.expand_dims(output_video_ct, axis=0) ), axis=0)
        mean_img = np.asarray(np.sum(combined_ct_video, axis=0) / 2,
                              dtype=combined_ct_video.dtype)
        # combined_video: seq_len x (3*height) x (temp_bs*width) x 3
        combined_video = np.concatenate(np.concatenate((combined_ct_video,
                                                        np.expand_dims(mean_img,0)), axis=0), axis=1)

    else:
        mean_img = np.asarray(np.sum(np.array(output_video), axis=0) / len(output_video), dtype=output_video[0][0].dtype)
        # combined_video: seq_len x height x (3*height) x 3
        combined_video = np.concatenate(output_video + [mean_img], axis=2)

    # saving through logger
    if phase == 'train':
        logging_name = 'input trajectories and output decodings'
    else:
        logging_name = 'output decodings'

    combined_video_log = np.transpose(combined_video, (0,3,1,2))
    logger.log_gif(combined_video_log/255, logging_name, step, phase)
    print('save2mp4: {}'.format(save2mp4))
    if save2mp4:
        filename = 'combined_' + str(kl_weight) + '_' + str(phase) + '_' + str(step) + '.mp4'
        save_video(combined_video, os.path.join(log_dir, filename), fps=30)
        print('video saved at {}'.format(os.path.join(log_dir, filename)))


def save_interpolation_video(output):
    output = output.cpu().detach().numpy()
    video = []
    [n, sl, ad] = output.shape
    for j in range(n):
        env = _set_env()
        input_images = []
        sequence = output[j]
        for i in range(sl):
            action = sequence[i]
            obs, reward, done, info = env.step(action)
            image = env.render(**INTERPOLATION_PIXEL_RENDER_KWARGS)
            input_images.append(image)
        video.append(input_images)

    x=np.array(video)
    y = []
    rows = 6
    cols = 6
    for i in range(rows):
        y.append(np.expand_dims(np.concatenate(x[i * cols:(i + 1) * cols], axis=2), axis=0))
    y = np.vstack(y)
    final_video = np.concatenate(y, axis=1)
    print('saving video')
    save_video(final_video, '/home/smit/interpolation.mp4', fps=6)


# def visualize_Z(self):
#     self.model_test.load_state_dict(self.model.state_dict())
#     self.model_test.eval()
#
#     # train
#     mu_z = []
#     with autograd.no_grad():
#         for batch_idx, sample_batched in enumerate(self.train_loader):
#             # print(batch_idx, len(self.train_loader))
#             inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
#             with self.model_test.val_mode():
#                 output = self.model_test(inputs)
#                 z_value = output.q.mu
#                 mu_z.append(z_value)
#
#     mu_z_np = []
#     for z in mu_z:
#         mu_z_np.append(z.cpu().numpy())
#
#     mu_z_np = np.array(mu_z_np)
#     mu_z_np = np.concatenate(mu_z_np, axis=0)
#     print(mu_z_np.shape)
#     plt.plot(mu_z_np[:,0].tolist(), mu_z_np[:,1].tolist(), 'ro')
#     plt.savefig('/home/smit/z_space_train.png')
#
#     # val
#     mu_z = []
#     with autograd.no_grad():
#         for batch_idx, sample_batched in enumerate(self.val_loader):
#             # print(batch_idx, len(self.val_loader))
#             inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
#             with self.model_test.val_mode():
#                 output = self.model_test(inputs)
#                 z_value = output.q.mu
#                 mu_z.append(z_value)
#
#     mu_z_np = []
#     for z in mu_z:
#         mu_z_np.append(z.cpu().numpy())
#
#     mu_z_np = np.array(mu_z_np)
#     mu_z_np = np.concatenate(mu_z_np, axis=0)
#     print(mu_z_np.shape)
#     plt.plot(mu_z_np[:, 0].tolist(), mu_z_np[:, 1].tolist(), 'ro')
#     plt.savefig('/home/smit/z_space_val.png')
#
#
# def interpolate_z(self, x_min, x_max, y_min, y_max, delta):
#     self.model_test.load_state_dict(self.model.state_dict())
#     self.model_test.eval()
#     output = self.model_test.interpolate(x_min, x_max, y_min, y_max, delta)
#     save_interpolation_video(output)
