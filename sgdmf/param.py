
from __future__ import print_function

import numpy as np
from collections import Iterable, defaultdict

def lst(x):
    if isinstance(x, Iterable):
        return x
    else:
        return (x,)


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
            self.bi = defaultdict(lambda : 0.0)
            self.bj = defaultdict(lambda : 0.0)
            self.Pi = defaultdict(lambda : np.random.normal(self.mean, self.sd, size = d))
            self.Qj = defaultdict(lambda : np.random.normal(self.mean, self.sd, size = d))
        else:
            self.bi = np.zeros(n)
            self.bj = np.zeros(m)
            self.Pi = np.random.normal(self.mean, self.sd, size = (n, d))
            self.Qj = np.random.normal(self.mean, self.sd, size = (m, d))


    def get(self, index):

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
            out = []
            for ix in lst(index):
                out += [getattr(self, param)[ix]]
            return np.array(out)


    def set(self, index, value):

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
        

    def size(self):
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

        if axis == 0:
            for i in lst(index):
                del self.bi[i]
                del self.Pi[i]
        else:
            for i in lst(index):
                del self.bj[i]
                del self.Qj[i]

