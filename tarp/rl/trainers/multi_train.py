import torch
import os
import imp
import json
import torch.nn as nn
import gc
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from collections import defaultdict

from tarp.rl.components.params import get_args
from tarp.train import set_seeds, make_path, datetime_str, save_config, get_exp_dir, save_checkpoint
from tarp.components.checkpointer import CheckpointHandler, save_cmd, save_git, get_config_path
from tarp.utils.general_utils import AttrDict, ParamDict, AverageTimer, timing, pretty_print, listdict2dictlist, \
                                        check_memory_kill_switch
from tarp.rl.utils.mpi import update_with_mpi_config, set_shutdown_hooks, mpi_sum, setup_mpi
from tarp.rl.utils.wandb import WandBLogger
from tarp.rl.utils.rollout_utils import RolloutSaver
from tarp.rl.components.sampler import Sampler
from tarp.rl.components.replay_buffer import RolloutStorage
import tarp.rl.envs

from tarp.rl.train import RLTrainer

class MultiRLTrainer(RLTrainer):
    """Sets up RL training loop, instantiates all components, runs training."""
    def __init__(self, args):
        self.args = args
        self.setup_device()

        # set up params
        self.conf = self.get_config()
        update_with_mpi_config(self.conf)   # self.conf.mpi = AttrDict(is_chef=True)
        setup_mpi()
        self._hp = self._default_hparams()
        self._hp.overwrite(self.conf.general)  # override defaults with config file
        self._hp.exp_path = make_path(self.conf.exp_dir, args.path, args.prefix, args.new_dir)
        self.log_dir = log_dir = os.path.join(self._hp.exp_path, 'log')
        print('using log dir: ', log_dir)

        # set seeds, display, worker shutdown
        if args.seed != -1: self._hp.seed = args.seed   # override from command line if set
        set_seeds(self._hp.seed)
        os.environ["DISPLAY"] = ":1"
        set_shutdown_hooks()

        # set up logging
        if self.is_chef:
            print("Running base worker.")
            self.logger = self.setup_logging(self.conf, self.log_dir)
        else:
            print("Running worker {}, disabled logging.".format(self.conf.mpi.rank))
            self.logger = None

        # build env
        self.conf.env.seed = self._hp.seed
        self.envs = []
        for env_conf in self.conf.env.conf_list:
            env_conf.seed = self._hp.seed
            env = self._hp.environment(env_conf)
            # env.set_config(env_conf)
            self.envs.append(env)
        if self.is_chef:
            pretty_print(self.conf)

        # build agent (that holds actor, critic, exposes update method)
        self.conf.agent.num_workers = self.conf.mpi.num_workers
        self.conf.agent.n_steps_per_update = self._hp.n_steps_per_update
        self.agent = self._hp.agent(self.conf.agent)
        self.agent.to(self.device)

        # build sampler
        sampler_conf_list = []
        for env_conf in self.conf.env.conf_list:
            sampler_conf = deepcopy(self.conf.sampler)
            sampler_conf.head_key = env_conf.head_key
            sampler_conf_list.append(sampler_conf)
        self.samplers = [self._hp.sampler(sampler_conf_list[i], env, self.agent, self.logger, self._hp.max_rollout_len) for i, env in enumerate(self.envs)]

        # load from checkpoint
        self.global_step, self.n_update_steps, start_epoch = 0, 0, 0
        if args.resume or self.conf.ckpt_path is not None:
            start_epoch = self.resume(args.resume, self.conf.ckpt_path)
            self._hp.n_warmup_steps = 0     # no warmup if we reload from checkpoint!


        if self._hp.load_offline_data:
            rollouts = self.load_rollouts()
            self.agent.replay_buffer.append(rollouts)
            del rollouts
            gc.collect()

        # start training/evaluation
        if args.mode == 'train':
            self.train(start_epoch)
        elif args.mode == 'val':
            self.val()
        else:
            self.generate_rollouts()

    def train_epoch(self, epoch):
        """Run inner training loop."""
        # sync network parameters across workers
        if self.conf.mpi.num_workers > 1:
            self.agent.sync_networks()

        # initialize timing
        timers = defaultdict(lambda: AverageTimer())

        [sampler.init(is_train=True) for sampler in self.samplers]
        ep_start_step = self.global_step
        while self.global_step - ep_start_step < self._hp.n_steps_per_epoch:
            # check_memory_kill_switch()

            with timers['batch'].time():
                # collect experience
                if self._hp.offline_rl:
                    experience_batch = {}
                    self.global_step += self.conf.mpi.num_workers
                else:
                    with timers['rollout'].time():
                        experience_batch, env_steps = self.sampler.sample_batch(batch_size=self._hp.n_steps_per_update, global_step=self.global_step)
                        self.global_step += mpi_sum(env_steps)

                # update policy
                with timers['update'].time():
                    agent_outputs = self.agent.update(experience_batch)
                    self.n_update_steps += 1

                # log results
                with timers['log'].time():
                    if self.is_chef and self.log_outputs_now:
                        self.agent.log_outputs(agent_outputs, None, self.logger,
                                               log_images=False, step=self.global_step)
                        self.print_train_update(epoch, agent_outputs, timers)
    def val(self):
        """Evaluate agent."""
        for i, sampler in enumerate(self.samplers):
            val_rollout_storage = RolloutStorage()
            episode_info_list = []
            with self.agent.val_mode():
                with torch.no_grad():
                    with timing("Eval rollout time: "):
                        # for _ in range(WandBLogger.N_LOGGED_SAMPLES):   # for efficiency instead of self.args.n_val_samples
                        for _ in range(1):   # for efficiency instead of self.args.n_val_samples
                            # check_memory_kill_switch()
                            val_rollout_storage.append(sampler.sample_episode(is_train=False, render=True))
                            episode_info_list.append(sampler.get_episode_info())

            rollout_stats = val_rollout_storage.rollout_stats()
            episode_info_dict = listdict2dictlist(episode_info_list)
            env_conf = self.conf.env.conf_list[i]
            prefix = env_conf.head_key
            for key in episode_info_dict:
                episode_info_dict[key] = np.mean(episode_info_dict[key])
            rollout_stats.update(episode_info_dict)
            if self.is_chef:
                with timing("Eval log time: "):
                    self.agent.log_outputs(rollout_stats, val_rollout_storage,
                                           self.logger, log_images=True, step=self.global_step, prefix='-'+prefix)
                print("Evaluation Avg_Reward: {}".format(rollout_stats.avg_reward))
            del val_rollout_storage

    def load_rollouts(self):
        self.conf.data.device = self.device.type
        # check_memory_kill_switch()
        rollouts = self.conf.data.dataset_spec.dataset_class(self.conf.data.dataset_spec.data_dir, self.conf.data,
                                          resolution=self.conf.data.dataset_spec.resolution, phase=None).data_dict
        return rollouts

if __name__ == '__main__':
    MultiRLTrainer(args=get_args())
