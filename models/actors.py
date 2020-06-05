import chainer
import chainer.links as L
import chainer.functions as F

import numpy as np


class _Actor(chainer.Chain):
    def save(self, path):
        if path.exists():
            raise ValueError('File already exist')
        chainer.serializers.save_npz(path.resolve(), self)

    def load(self, path):
        if not path.exists():
            raise ValueError('File {} not found'.format(path))
        chainer.serializers.load_npz(path.resolve(), self)


class MujocoActor(_Actor):
    def __init__(self, state_dim, action_dim, initialW=chainer.initializers.GlorotUniform()):
        super(MujocoActor, self).__init__()
        with self.init_scope():
            self._linear1 = L.Linear(in_size=(state_dim), out_size=256, initialW=initialW)
            self._linear2 = L.Linear(in_size=256, out_size=256, initialW=initialW)
            self._linear_mu = L.Linear(in_size=256, out_size=action_dim, initialW=initialW)
            self._linear_ln_var = L.Linear(in_size=256, out_size=action_dim, initialW=initialW)

    def __call__(self, s):
        mu, ln_var = self._mu_and_ln_var(s)
        return mu
    
    def _sample(self, s):
        mu, ln_var = self._mu_and_ln_var(s)
        x = F.gaussian(mu, ln_var)
        y = F.tanh(x)
        return y

    def action_with_log_pi(self, s):
        mu, ln_var = self._mu_and_ln_var(s)
        # Original code clips the log_sigma between [-20, 2]
        # But ln_var is 2 * log_sigma. Therefore clipping it to [-40, 4]  
        ln_var = F.clip(ln_var, -40, 4)
        x = F.gaussian(mu, ln_var)
        # log_pi
        # = log(N(x|mu, var)*(arctanh(tanh(x))')
        # = logN(x|mu, var) + log(arctanh(tanh(x))')
        log_pi = \
            self._log_normal(x, mu, F.exp(ln_var), ln_var) - \
            self._forward_log_det_jacobian(x)
        y = F.tanh(x)
        return y, F.sum(log_pi, axis=1)

    def _mu_and_ln_var(self, s):
        h = self._linear1(s)
        h = F.relu(h)
        h = self._linear2(h)
        h = F.relu(h)
        mu = self._linear_mu(h)
        ln_var = self._linear_ln_var(h)

        return mu, ln_var

    def _log_normal(self, x, mean, var, ln_var):
        # log N(x|mu, var)
        # = -0.5*log2*pi - 0.5 * ln_var - 0.5 * (x-mu)**2 / var
        return -0.5 * np.log(2 * np.pi) - 0.5 * ln_var - 0.5 * (x-mean) ** 2 / var

    def _forward_log_det_jacobian(self, x):
        # arctanh(y)' = 1/(1 - y^2) (y=tanh(x))
        # Below computes log(1 - tanh(x)^2)
        # For derivation see:
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py
        return 2.0 * (np.log(2.0) - x - F.softplus(-2.0 * x))
