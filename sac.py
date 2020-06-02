import numpy as np

import chainer
import chainer.functions as F
from chainer import iterators
from chainer.datasets import tuple_dataset
from chainer import optimizers
from chainer.dataset import concat_examples


class SAC(object):
    def __init__(self, v_func_builder, q_func_builder, pi_builder, state_dim, action_dim, *,
                 gamma=0.99, tau=0.005, lr=3.0e-4, environment_step=1, gradient_step=1, device=-1):
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
        self._device = device
        self._environment_step = environment_step
        self._gradient_step = gradient_step

    def act_with_policy(self, env, s):
        s = np.float32(s)
        state = chainer.Variable(np.reshape(s, newshape=(1, ) + s.shape))
        if not self._device < 0:
            state.to_gpu()

        a = self._pi(state)
        if not self._device < 0:
            a.to_cpu()

        a = np.squeeze(a.data, axis=0)
        s_next, r, done, _ = env.step(a)

        s_next = np.float32(s_next)
        a = np.float32(a)
        r = np.float32(r)
        return s, a, r, s_next, done

    def train(self, env):
        replay_buffer = []
        for _ in self._environment_step:
            pass
        iterator = self._prepare_iterator(replay_buffer)
        for _ in self._gradient_step:
            batch = iterator.next()
            s_current, action, r, s_next, non_terminal = \
                concat_examples(batch, device=self._device)

            q1 = self._q1(s_current, action)
            q2 = self._q2(s_current, action)

            min_q = F.minimum(q1, q2)

            _, log_pi = self._pi.action_with_log_pi(s_current)
            target_v = min_q - log_pi
            target_v.unchain()
            
            v = self._v(s_current)
            v_loss = F.mean_squared_error(v, target_v)

            self._v_optimizer.target.clear_grads()
            v_loss.backward()
            v_loss.unchain_backward()
            self._v_optimizer.update()

            r = F.reshape(r, shape=(*r.shape, 1))
            non_terminal = F.reshape(
                non_terminal, shape=(*non_terminal.shape, 1))

            v = self._



    def _prepare_iterator(self, buffer):
        return iterators.SerialIterator(buffer, self._batch_size)
