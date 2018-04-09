
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

    """Container for parameters of MatrixFactorizer model

    Methods:
    --------

    __init__(shape = (0, 0, 0), mean = 0.0, sd = 0.1, dynamic = False)
        Initialize the empty container. When dynamic=True, parameters are 
        stored in numpy arrays, otherwise in defaultdict objects.

    get(index)
        Get parameters for specific (i,j) index.

    get_param(param, index = None)
        Get specific parameter, can be filtered by index.

    set(index, value)
        Set parameters (bi,bj,Pi,Qj) for specific (i,j) index.
    
    set_param(param, value)
        Set specific parameter to given value.

    get_shape()
        Returns the (n,m,d) shape of the parameters.

    drop(index, axis)
        Drop particular index. Works only when dynamic=True. 
    
    to_dict()
        Return object as a JSON-friendly dict.

    from_dict(input)
        Import parameters from dict.

    """

    def __init__(self, shape = (0, 0, 0), mean = 0.0,
                 sd = 0.1, dynamic = False):

        self.mean = mean
        self.sd = sd
        self.shape = shape
        self.dynamic = dynamic
        self.mu = 0.0
        n, m, d = self.shape

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

        self.shape = self.get_shape()


    def set_param(self, param, value):
        setattr(self, param, value)
        self.shape = self.get_shape()
        

    def get_shape(self):

        if self.dynamic:
            assert len(self.bi) == len(self.Pi)
            assert len(self.bj) == len(self.Qj)

            return (len(self.Pi), len(self.Qj), self.shape[2])
        else:
            assert self.bi.shape[0] == self.Pi.shape[0]
            assert self.bj.shape[0] == self.Qj.shape[0]

            return (self.Pi.shape[0], self.Qj.shape[0], self.shape[2])


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

        self.shape = self.get_shape()


    def to_dict(self):

        out = (self.__dict__).copy()
        out['shape'] = self.get_shape()

        if self.dynamic:
            for par in ['bi', 'bj', 'Pi', 'Qj']:
                out[par] = dict(out[par])
                if par in ['Pi', 'Qj']:
                    for k in out[par]:
                        out[par][k] = out[par][k].tolist()
        else:
            for par in ['bi', 'bj', 'Pi', 'Qj']:
                out[par] = out[par].tolist()

        return out


    def from_dict(self, input):
    
        self.__init__(input['shape'], input['mean'],
                      input['sd'], input['dynamic'])

        setattr(self, 'mu', input['mu'])

        if not self.dynamic:
            for par in ['bi', 'bj', 'Pi', 'Qj']:
                setattr(self, par, np.array(input[par]))
        else:    
            for par in ['bi', 'bj']:
                setattr(self, par, defaultdict(float, input[par]))
            for k in input['Pi']:
                self.Pi[k] = np.array(input['Pi'][k])
            for k in input['Qj']:
                self.Qj[k] = np.array(input['Qj'][k])

        self.shape = self.get_shape()

