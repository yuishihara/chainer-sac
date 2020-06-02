import chainer
import chainer.links as L
import chainer.functions as F


class _Critic(chainer.Chain):
    def save(self, path):
        if path.exists():
            raise ValueError('File already exist')
        chainer.serializers.save_npz(path.resolve(), self)

    def load(self, path):
        if not path.exists():
            raise ValueError('File {} not found'.format(path))
        chainer.serializers.load_npz(path.resolve(), self)


class VFunction(_Critic):
    def __init__(self, state_dim):
        super(VFunction, self).__init__()
        with self.init_scope():
            self._linear1 = L.Linear(in_size=(state_dim), out_size=256)
            self._linear2 = L.Linear(in_size=256, out_size=256)
            self._linear3 = L.Linear(in_size=256, out_size=1)

    def __call__(self, s):
        h = self._linear1(s)
        h = F.relu(h)
        h = self._linear2(h)
        h = F.relu(h)
        return self._linear3(h)


class QFunction(_Critic):
    def __init__(self, state_dim, action_dim):
        super(QFunction, self).__init__()
        with self.init_scope():
            self._linear1 = L.Linear(
                in_size=(state_dim+action_dim), out_size=256)
            self._linear2 = L.Linear(in_size=256, out_size=256)
            self._linear3 = L.Linear(in_size=256, out_size=1)

    def __call__(self, s, a):
        x = F.concat((s, a))
        h = self._linear1(x)
        h = F.relu(h)
        h = self._linear2(h)
        h = F.relu(h)
        return self._linear3(h)
