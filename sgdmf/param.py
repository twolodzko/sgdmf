
from __future__ import print_function

import numpy as np
from collections import Iterable

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
            self.bi = dict()
            self.bj = dict()
            self.Pi = dict()
            self.Qj = dict()
        else:
            self.bi = np.zeros(n)
            self.bj = np.zeros(m)
            self.Pi = np.random.normal(self.mean, self.sd, size = (n, d))
            self.Qj = np.random.normal(self.mean, self.sd, size = (m, d))


    def get(self, index, initialize = False):

        mu = self.mu

        if self.dynamic:

            if initialize:

                if index[0] not in self.bi:
                    self.bi[index[0]] = 0.0
                if index[1] not in self.bj:
                    self.bj[index[1]] = 0.0
                if index[0] not in self.Pi:
                    self.Pi[index[0]] = np.random.normal(self.mean, self.sd, self.d)
                if index[1] not in self.Qj:
                    self.Qj[index[1]] = np.random.normal(self.mean, self.sd, self.d)

            bi = self.bi[index[0]]
            bj = self.bj[index[1]]
            Pi = self.Pi[index[0]]
            Qj = self.Qj[index[1]]

        else:

            if initialize:
                
                if index[0] >= self.bi.shape[0]:
                    self.bi = np.append(self.bi, 0.0)
                if index[1] >= self.bj.shape[0]:
                    self.bj = np.append(self.bj, 0.0)
                if index[0] >= self.Pi.shape[0]:
                    rnd = np.random.normal(self.mean, self.sd, self.d)
                    self.Pi = np.append(self.Pi, rnd, axis = 0)
                if index[1] >= self.Qj.shape[0]:
                    rnd = np.random.normal(self.mean, self.sd, self.d)
                    self.Qj = np.append(self.Qj, rnd, axis = 0)

            bi = self.bi[index[0]]
            bj = self.bj[index[1]]
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

        if self.dynamic:
            self.bi[index[0]] = value[1]
            self.bj[index[1]] = value[2]
            self.Pi[index[0]] = value[3]
            self.Qj[index[1]] = value[4]
        else:
            self.bi[index[0]] = value[1]
            self.bj[index[1]] = value[2]
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

