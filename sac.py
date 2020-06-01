import chainer
import chainer.functions as F
from chainer import optimizers
from chainer.dataset import concat_examples


class SAC(object):
    def __init__(self, v_func_builder, q_func_builder, pi_builder, state_dim, action_dim, *,
                 gamma=0.99, tau=0.005, lr=3.0e-4):
        self._v = v_func_builder(state_dim, action_dim)
        self._q1 = q_func_builder(state_dim, action_dim)
        self._q2 = q_func_builder(state_dim, action_dim)
        self._pi = pi_builder(state_dim, action_dim)

        self._v_optimizer = optimizers.Adam(alpha=lr)
        self._q1_optimizer = optimizers.Adam(alpha=lr)
        self._q2_optimizer = optimizers.Adam(alpha=lr)
        self._pi_optimizer = optimizers.Adam(alpha=lr)

        self._v_optimizer.setup(self._v)
        self._q1_optimizer.setup(self._q1)
        self._q2_optimizer.setup(self._q2)
        self._pi_optimizer.setup(self._pi)

        self._v_target = v_func_builder(state_dim, action_dim)

        self._tau = tau
        self._gamma = gamma
