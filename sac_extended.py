import numpy as np

import pathlib

from collections import deque

import chainer
import chainer.functions as F
from chainer import iterators
from chainer.datasets import tuple_dataset
from chainer import optimizers
from chainer.dataset import concat_examples


class AdjustableTemperature(chainer.Chain):
    def __init__(self, initial_value=1.0):
        super(AdjustableTemperature, self).__init__()

        with self.init_scope():
            initializer = np.log(initial_value)
            self._ln_temperature = chainer.Parameter(
                initializer=initializer, shape=(1)
            )

    def __call__(self):
        return self._ln_temperature

    def exp(self):
        return F.exp(self._ln_temperature)

    def save(self, path):
        if path.exists():
            raise ValueError('File already exist')
        chainer.serializers.save_npz(path.resolve(), self)

    def load(self, path):
        if not path.exists():
            raise ValueError('File {} not found'.format(path))
        chainer.serializers.load_npz(path.resolve(), self)


class SAC(object):
    def __init__(self, q_func_builder, pi_builder, state_dim, action_dim, *,
                 gamma=0.99, tau=0.005, lr=3.0e-4, batch_size=256, environment_steps=1, gradient_steps=1,
                 initial_temperature=1.0, temperature_target=None, device=-1):
        self._q1 = q_func_builder(state_dim, action_dim)
        self._q2 = q_func_builder(state_dim, action_dim)
        self._target_q1 = q_func_builder(state_dim, action_dim)
        self._target_q2 = q_func_builder(state_dim, action_dim)
        self._pi = pi_builder(state_dim, action_dim)
        self._alpha = AdjustableTemperature(initial_value=initial_temperature)

        if not device < 0:
            self._q1.to_device(device=device)
            self._q2.to_device(device=device)
            self._target_q1.to_device(device=device)
            self._target_q2.to_device(device=device)
            self._pi.to_device(device=device)
            self._alpha.to_device(device=device)

        self._q1_optimizer = optimizers.Adam(alpha=lr)
        self._q2_optimizer = optimizers.Adam(alpha=lr)
        self._pi_optimizer = optimizers.Adam(alpha=lr)
        self._alpha_optimizer = optimizers.Adam(alpha=lr)

        self._q1_optimizer.setup(self._q1)
        self._q2_optimizer.setup(self._q2)
        self._pi_optimizer.setup(self._pi)
        self._alpha_optimizer.setup(self._alpha)

        self._tau = tau
        self._gamma = gamma
        self._device = device
        self._environment_steps = environment_steps
        self._gradient_steps = gradient_steps
        self._batch_size = batch_size

        self._state = None
        self._replay_buffer = deque(maxlen=1000000)
        self._initialized = False

        self._temperature_target = \
            -np.float(action_dim) if temperature_target is None else np.float(temperature_target)

    def train(self, env):
        if not self._initialized:
            self._initialize_target_networks()
        for _ in range(self._environment_steps):
            experience = self._perform_environment_step(env)
            self._replay_buffer.append(experience)
        if len(self._replay_buffer) < 1000:
            return
        iterator = self._prepare_iterator(self._replay_buffer)
        for _ in range(self._gradient_steps):
            self._perform_gradient_step(iterator)

    def evaluate(self, env, *, n_runs=10):
        returns = []
        for _ in range(n_runs):
            s = env.reset()
            episode_return = 0
            while True:
                a = self._compute_action(s)
                s, r, done, _ = env.step(a)
                episode_return += r
                if done:
                    returns.append(episode_return)
                    break
        return returns, np.mean(returns), np.median(returns), np.std(returns)

    def save_models(self, outdir, prefix):
        q1_filepath = pathlib.Path(outdir, 'q1_iter-{}'.format(prefix))
        q2_filepath = pathlib.Path(outdir, 'q2_iter-{}'.format(prefix))
        pi_filepath = pathlib.Path(outdir, 'pi_iter-{}'.format(prefix))
        alpha_filepath = pathlib.Path(outdir, 'alpha_iter-{}'.format(prefix))

        self._q1.to_cpu()
        self._q2.to_cpu()
        self._pi.to_cpu()
        self._alpha.to_cpu()

        self._q1.save(q1_filepath)
        self._q2.save(q2_filepath)
        self._pi.save(pi_filepath)
        self._alpha.save(alpha_filepath)

        if not self._device < 0:
            self._q1.to_device(device=self._device)
            self._q2.to_device(device=self._device)
            self._pi.to_device(device=self._device)
            self._alpha.to_device(device=self._device)

    def load_models(self, q1_filepath, q2_filepath, pi_filepath, alpha_filepath):
        self._q1.to_cpu()
        self._q2.to_cpu()
        self._pi.to_cpu()
        self._alpha.to_cpu()

        if q1_filepath:
            self._q1.load(q1_filepath)
        if q2_filepath:
            self._q2.load(q2_filepath)
        if pi_filepath:
            self._pi.load(pi_filepath)
        if alpha_filepath:
            self._alpha.load(alpha_filepath)

        if not self._device < 0:
            self._q1.to_device(device=self._device)
            self._q2.to_device(device=self._device)
            self._pi.to_device(device=self._device)
            self._alpha.to_device(device=self._device)

    def _perform_environment_step(self, env):
        if self._state is None:
            self._state = env.reset()
        (s, a, r, s_next, done) = self._act_with_policy(env, self._state)
        non_terminal = np.float32(0 if done else 1)
        if done:
            self._state = env.reset()
        else:
            self._state = s_next
        return (s, a, r, s_next, non_terminal)

    def _perform_gradient_step(self, iterator):
        batch = iterator.next()
        s_current, action, r, s_next, non_terminal = \
            concat_examples(batch, device=self._device)
        r = F.reshape(r, shape=(*r.shape, 1))
        non_terminal = F.reshape(
            non_terminal, shape=(*non_terminal.shape, 1))

        with chainer.using_config('enable_backprop', False), chainer.using_config('train', False):
            a_next, log_pi = self._pi.action_with_log_pi(s_next)
            log_pi = F.reshape(log_pi, shape=(*log_pi.shape, 1))

            # update q
            target_q1 = self._target_q1(s_next, a_next)
            target_q2 = self._target_q2(s_next, a_next)

            min_q = F.minimum(target_q1, target_q2)

            q_target = r + self._gamma * non_terminal * min_q - self._alpha.exp() * log_pi

        q1 = self._q1(s_current, action)
        q2 = self._q2(s_current, action)
        q1_loss = 0.5 * F.mean_squared_error(q_target, q1)
        q2_loss = 0.5 * F.mean_squared_error(q_target, q2)
        q_loss = q1_loss + q2_loss

        self._q1_optimizer.target.cleargrads()
        self._q2_optimizer.target.cleargrads()
        q_loss.backward()
        q_loss.unchain_backward()
        self._q1_optimizer.update()
        self._q2_optimizer.update()

        # update pi
        pi_action, log_pi = self._pi.action_with_log_pi(s_current)
        log_pi = F.reshape(log_pi, shape=(*log_pi.shape, 1))

        q1 = self._q1(s_current, pi_action)
        q2 = self._q2(s_current, pi_action)
        min_q = F.minimum(q1, q2)

        pi_loss = F.mean(self._alpha.exp() * log_pi - min_q)
        self._pi_optimizer.target.cleargrads()
        pi_loss.backward()
        pi_loss.unchain_backward()
        self._pi_optimizer.update()

        # update temperature
        alpha_loss = -self._alpha.exp() * F.mean(log_pi + self._temperature_target)
        self._alpha_optimizer.target.cleargrads()
        alpha_loss.backward()
        alpha_loss.unchain_backward()
        self._alpha_optimizer.update()

        self._update_target_network(self._target_q1, self._q1, self._tau)
        self._update_target_network(self._target_q2, self._q2, self._tau)

    def _act_with_policy(self, env, s):
        with chainer.using_config('train', False), \
                chainer.using_config('enable_backprop', False):
            a = self._compute_action(s)
            s_next, r, done, _ = env.step(a)
            return s, a, r, s_next, done

    def _compute_action(self, s):
        state = chainer.Variable(np.reshape(s, newshape=(1, ) + s.shape))
        if not self._device < 0:
            state.to_device(self._device)

        a = self._pi(state)
        if not self._device < 0:
            a.to_cpu()

        return np.squeeze(a.data, axis=0)

    def _initialize_target_networks(self):
        self._update_target_network(self._target_q1, self._q1, 1.0)
        self._update_target_network(self._target_q2, self._q2, 1.0)
        self._initialized = True

    def _update_target_network(self, target, origin, tau):
        for target_param, origin_param in zip(target.params(), origin.params()):
            target_param.data = tau * origin_param.data + \
                (1.0 - tau) * target_param.data

    def _prepare_iterator(self, buffer):
        return iterators.SerialIterator(buffer, self._batch_size)
