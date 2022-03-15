import numpy as np
from tqdm import tqdm

from tarp.utils.general_utils import AttrDict
from tarp.utils.pytorch_utils import ten2ar
from tarp.components.evaluator import Evaluator, TopOfNSequenceEvaluator
from tarp.data.metaworld.src.metaworld_utils import MetaWorldEnvHandler, DEFAULT_PIXEL_RENDER_KWARGS


class MetaWorldEvaluator(TopOfNSequenceEvaluator):
    """Implements model evaluation by rolling out the model in an environment for N steps."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._per_env_rollouts = None

    def reset(self):
        super().reset()
        self._per_env_rollouts = AttrDict()

    def eval(self, inputs, model):
        if not self._per_env_rollouts:      # only run once per evaluation
            print("Running rollout evaluation...")
            for i in tqdm(range(MetaWorldEnvHandler.num_envs())):
                env_name = MetaWorldEnvHandler.id2name(i)
                avg_reward = 0
                for k in range(10):
                    rollout = self._sim_rollout(model, MetaWorldEnvHandler.env_from_name(env_name))
                    avg_reward += rollout.sum_reward
                rollout.sum_reward = avg_reward / 10
                if rollout is None: continue
                self._per_env_rollouts[env_name] = rollout

        return model(inputs)    # compute outputs on the inputs because API requires to return outputs

    @staticmethod
    def _sim_rollout(model, env, env_reset_state=None, env_id=None, max_rollout_len=None):
        """Runs the model+env rollout loop."""
        # reset the environment + agent (aka model)
        if env_reset_state is not None:
            env._task_envs[MetaWorldEnvHandler.get_metatask_id(env_id)].set_env_state(env_reset_state)
            obs = env._augment_observation(env._task_envs[MetaWorldEnvHandler.get_metatask_id(env_id)]._get_obs())    # HACK!
        else:
            obs = env.reset()
        img = env.render(**DEFAULT_PIXEL_RENDER_KWARGS)
        model.reset()

        # loop until terminated
        rollout_imgs, rollout_states = [img], [obs]
        assert (not hasattr(model._hp, 'feed_goal') or not model._hp.feed_goal), 'Feed goal needs to be false.'
        goal = np.zeros_like(obs)[None]
        done = False
        sum_reward, final_reward = 0, 0
        while not done:
            model_output = model.run(obs[None], goal)
            action = ten2ar(model_output.action[0])
            obs, reward, done, info = env.step(action)
            img = env.render(**DEFAULT_PIXEL_RENDER_KWARGS)
            rollout_imgs.append(img); rollout_states.append(obs)
            sum_reward += reward
            if max_rollout_len is not None and len(rollout_imgs) == max_rollout_len: break
        final_reward = reward

        return AttrDict(images=np.stack(rollout_imgs)/255.,
                        states=np.stack(rollout_states),
                        sum_reward=sum_reward,
                        final_reward=final_reward)

    def dump_results(self, it):
        """Writes out gifs and rewards of all generated rollouts."""
        assert self._per_env_rollouts       # we should have generated a bunch of rollouts before

        # write gifs
        for env_name in self._per_env_rollouts:
            with self._logger.log_to(f"{env_name}_full_rollout", it, 'gif'):
                self._logger.log(self._per_env_rollouts[env_name].images)

        # write rewards
        get_item = lambda i: self._per_env_rollouts[MetaWorldEnvHandler.id2name(i)]
        sum_rewards, final_rewards = [], []
        for i in range(MetaWorldEnvHandler.num_envs()):
            env_name = MetaWorldEnvHandler.id2name(i)
            if env_name in self._per_env_rollouts.keys():
                print("{} - {}".format(env_name, get_item(i).sum_reward))
                sum_rewards.append(get_item(i).sum_reward)
                final_rewards.append(get_item(i).final_reward)
            else:
                sum_rewards.append(-1); final_rewards.append(-1)
        with self._logger.log_to("sum_rewards", it, "graph"):
            self._logger.log(np.array(sum_rewards))
        with self._logger.log_to("final_rewards", it, "graph"):
            self._logger.log(np.array(final_rewards))


class MetaWorldTopOfNEvaluator(MetaWorldEvaluator, TopOfNSequenceEvaluator):
    def __init__(self, *args, **kwargs):
        TopOfNSequenceEvaluator.__init__(self, *args, **kwargs)
        MetaWorldEvaluator.__init__(self, *args, **kwargs)

    def eval(self, inputs, model):
        TopOfNSequenceEvaluator.eval(self, inputs, model)
        MetaWorldEvaluator.eval(self, inputs, model)

    def reset(self):
        TopOfNSequenceEvaluator.reset(self)
        MetaWorldEvaluator.reset(self)

    def dump_results(self, it):
        MetaWorldEvaluator.dump_results(self, it)
        TopOfNSequenceEvaluator.dump_results(self, it)      # this one calls reset()


class MetaWorldSSMTopOfNEvaluator(MetaWorldTopOfNEvaluator):
    """Additionally samples random rollouts from prior per env."""
    SAMPLES_PER_ENV = 3     # number of sampled rollouts per environment

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._per_env_rollout_samples = None

    def reset(self):
        super().reset()
        self._per_env_rollout_samples = AttrDict()

    def eval(self, inputs, model):
        if not self._per_env_rollout_samples:      # only run once per evaluation
            print("Running rollout samples...")
            for i in tqdm(range(MetaWorldEnvHandler.num_envs())):
                env_name = MetaWorldEnvHandler.id2name(i)
                base_env = MetaWorldEnvHandler.env_from_name(env_name)
                base_env_state = base_env._task_envs[MetaWorldEnvHandler.get_metatask_id(i)].get_env_state()

                rollouts = [self._sim_rollout(model,
                                              MetaWorldEnvHandler.env_from_name(env_name),
                                              base_env_state,
                                              i,
                                              model.n_rollout_steps) for _ in range(self.SAMPLES_PER_ENV)]
                self._per_env_rollout_samples[env_name] = rollouts

        super().eval(inputs, model)

    def dump_results(self, it):
        # blend rollouts, then log gifs
        for env_name in self._per_env_rollout_samples:
            with self._logger.log_to(f"{env_name}_rollout_samples", it, 'gif'):
                blended_imgs = np.sum(np.array([rollout.images / self.SAMPLES_PER_ENV
                                                for rollout in self._per_env_rollout_samples[env_name]]), axis=0)
                self._logger.log(blended_imgs)

        super().dump_results(it)

