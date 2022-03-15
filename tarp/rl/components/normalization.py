import numpy as np

from tarp.utils.general_utils import ParamDict


class Normalizer:
    """Normalizes quantities (zero-mean, unit-variance)."""
    MIN_STD = 1e-2      # minimum standard deviation

    def __init__(self, hp):
        self._hp = self._default_hparams().overwrite(hp)
        self._sum, self._square_sum = None, None
        self._count = 0
        self._mean, self._std = 0, 0.1

    def _default_hparams(self):
        default_dict = ParamDict({
            'clip_raw_obs': np.array(float("Inf")),        # symmetric value maximum for raw observation
            'clip_norm_obs': np.array(float("Inf")),       # symmetric value maximum for normalized observation
            'update_horizon': 1e7,     # number of values for which statistics get updated
        })
        return default_dict

    def __call__(self, vals):
        """Performs normalization."""
        vals = self._clip(vals, range=self._hp.clip_raw_obs)
        return self._clip((vals - self._mean) / self._std, range=self._hp.clip_norm_obs)

    def update(self, vals):
        """Add new values to internal value, update statistics."""
        if self._count >= self._hp.update_horizon: return

        if isinstance(vals, list):
            vals = np.stack(vals)

        # clip + update summed vals
        vals = self._clip(vals, range=self._hp.clip_raw_obs)
        sum_val, square_sum_val = vals.sum(axis=0), (vals**2).sum(axis=0)
        if self._sum is None:
            self._sum, self._square_sum = sum_val, square_sum_val
        else:
            self._sum += sum_val; self._square_sum += square_sum_val
        self._count += vals.shape[0]

        # update statistics
        self._mean = self._sum / self._count
        self._std = np.sqrt(np.maximum(self.MIN_STD**2 * np.ones_like(self._sum),
                                       self._square_sum / self._count - (self._sum / self._count)**2))

    def reset(self):
        self._sum, self._square_sum = None, None
        self._count = 0
        self._mean, self._std = 0, 0.1

    @staticmethod
    def _clip(val, range):
        return np.clip(val, -range, range)


class DummyNormalizer(Normalizer):
    def __call__(self, vals):
        return vals

    def update(self, vals):
        pass

class RewardNormalizer:
    def __init__(self, hp):
        self._hp = self._default_hparams().overwrite(hp)
        self.ret = 0
        self._mean, self._var = 0, 1.
        self.count = 1e-4

    def _default_hparams(self):
        default_dict = ParamDict({
            'cliprew': 10,
            'gamma': 0.99,
            'epsilon': 1e-8
        })
        return default_dict

    def __call__(self, rew):
        self.ret = self.ret * self._hp.gamma + rew
        self.update(self.ret)
        rew = np.clip(rew / np.sqrt(self._var + self._hp.epsilon), -self._hp.cliprew, self._hp.cliprew)
        return rew


    def update(self, val):
        delta = val - self._mean
        tot_count = self.count + 1

        self._mean = self._mean + delta / tot_count
        m_a = self._var * self.count
        M2 = m_a + np.square(delta) * self.count / tot_count
        self._var = M2 / tot_count
        self.count = tot_count

    def reset(self):
        self.ret = 0.
