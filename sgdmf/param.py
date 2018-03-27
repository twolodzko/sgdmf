
from __future__ import print_function

import numpy as np
from collections import Iterable, defaultdict

def lst(x):
    if isinstance(x, Iterable):
        return x
    else:
        return (x,)

def rng_norm(mean, sd, size):
    def rng():
        return np.random.normal(mean, sd, size = size)
    return rng


class ParamContainer(object):

    def __init__(self, shape = (0, 0, 0), mean = 0.0,
                 sd = 0.1, dynamic = False):

        self.mean = mean
        self.sd = sd
        self.d = shape[2]
        self.dynamic = dynamic
        self.mu = 0.0
        n, m, d = shape

        if self.dynamic:
            self.bi = defaultdict(float)
            self.bj = defaultdict(float)
            self.Pi = defaultdict(rng_norm(self.mean, self.sd, d))
            self.Qj = defaultdict(rng_norm(self.mean, self.sd, d))
        else:
            self.bi = np.zeros(n)
            self.bj = np.zeros(m)
            self.Pi = np.random.normal(self.mean, self.sd, size = (n, d))
            self.Qj = np.random.normal(self.mean, self.sd, size = (m, d))


    def get(self, index):

        if self.dynamic:
            index = [str(ix) for ix in lst(index)]

        mu = self.mu
        bi = self.bi[index[0]]
        bj = self.bj[index[1]]

        if self.dynamic:
            Pi = self.Pi[index[0]]
            Qj = self.Qj[index[1]]
        else:
            Pi = self.Pi[index[0], :]
            Qj = self.Qj[index[1], :]

        return mu, bi, bj, Pi, Qj


    def get_param(self, param, index = None):

        if index is None:
            return getattr(self, param)
        else:
            if self.dynamic:
                index = [str(ix) for ix in lst(index)]
            out = []
            for ix in index:
                out.append(getattr(self, param)[ix])
            return np.array(out)


    def set(self, index, value):

        if self.dynamic:
            index = [str(ix) for ix in lst(index)]

        self.mu = value[0]
        self.bi[index[0]] = value[1]
        self.bj[index[1]] = value[2]

        if self.dynamic:
            self.Pi[index[0]] = value[3]
            self.Qj[index[1]] = value[4]
        else:
            self.Pi[index[0], :] = value[3]
            self.Qj[index[1], :] = value[4]


    def set_param(self, param, value):
        setattr(self, param, value)
        

    def shape(self):
        if self.dynamic:
            assert len(self.bi) == len(self.Pi)
            assert len(self.bj) == len(self.Qj)

            return (len(self.Pi), len(self.Qj), self.d)
        else:
            assert self.bi.shape[0] == self.Pi.shape[0]
            assert self.bj.shape[0] == self.Qj.shape[0]

            return (self.Pi.shape[0], self.Qj.shape[0], self.d)


    def drop(self, index, axis):

        if not self.dynamic:
            raise NotImplementedError()

        if axis not in (0, 1):
            raise ValueError('Incorrect axis parameter')

        index = [str(ix) for ix in lst(index)]

        if axis == 0:
            for ix in index:
                del self.bi[ix]
                del self.Pi[ix]
        else:
            for ix in index:
                del self.bj[ix]
                del self.Qj[ix]


    def to_dict(self):

        if not self.dynamic:
            raise NotImplementedError()

        assert np.all(self.bi.keys() == self.bj.keys())
        assert np.all(self.bj.keys() == self.Pi.keys())
        assert np.all(self.Pi.keys() == self.Qj.keys())

        out = dict()

        for key in ['mean', 'sd', 'd', 'dynamic', 'mu']:
            out[key] = getattr(self, key)

        out['bi'] = dict(self.bi)
        out['bj'] = dict(self.bj)
        out['Pi'] = dict(self.Pi)
        out['Qj'] = dict(self.Qj)

        for key in out['Pi'].keys():
            out['Pi'][key] = list(self.Pi)

        for key in out['Qj'].keys():
            out['Qj'][key] = list(self.Qj)

        return out


    def from_dict(self, input):

        if not self.dynamic or not input['dynamic']:
            raise NotImplementedError()

        self.__init__((0, 0, input['d']), mean = input['mean'],
                      sd = input['sd'], dynamic = input['dynamic'])

        assert np.all(input['bi'].keys() == input['bj'].keys())
        assert np.all(input['bj'].keys() == input['Pi'].keys())
        assert np.all(input['Pi'].keys() == input['Qj'].keys())

        self.mu = input['mu']

        for key in input['bi'].keys():
            self.bi[key] = input['bi'][key]
            self.Pi[key] = np.array(input['Pi'][key])

        for key in input['bj'].keys():
            self.bj[key] = input['bj'][key]
            self.Qj[key] = np.array(input['Qj'][key])

