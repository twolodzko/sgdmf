
from __future__ import print_function

import numpy as np
from .indexer import OnlineIndexer, lst


class ParamMatrix(object):

    def __init__(self, shape, init_mean = 0.0, init_sd = 0.1):

        self.init_mean = init_mean
        self.init_sd = init_sd
        self.shape = np.array(shape, dtype = int)
        self.mu = 0.0
        n, m, d = shape
        self.bi = np.zeros(n)
        self.bj = np.zeros(m)
        self.Pi = np.random.normal(self.init_mean, self.init_sd, size = (n, d))
        self.Qj = np.random.normal(self.init_mean, self.init_sd, size = (m, d))


    def get(self, index):
        i, j = index
        return self.mu, self.bi[i], self.bj[j], self.Pi[i, :], self.Qj[j, :]

    
    def get_param(self, param):
        return getattr(self, param)


    def set(self, value, index):
        i, j = index
        self.mu = value[0]
        self.bi[i] = value[1]
        self.bj[j] = value[2]
        self.Pi[i, :] = value[3]
        self.Qj[j, :] = value[4]


    def set_param(self, param, value):
        setattr(self, param, value)
    

    def expand(self, by, axis):

        if axis not in (0, 1):
            raise ValueError('Incorrect axis parameter')

        d = self.shape[2]

        if axis == 0:
            self.bi = np.append(self.bi, np.zeros(by))
            self.Pi = np.append(self.Pi, np.random.normal(self.init_mean, self.init_sd,
                                                          size = (by, d)), axis = 0)
        else:
            self.bj = np.append(self.bj, np.zeros(by))
            self.Qj = np.append(self.Qj, np.random.normal(self.init_mean, self.init_sd,
                                                          size = (by, d)), axis = 0)


    def drop(self, index, axis):

        if axis not in (0, 1):
            raise ValueError('Incorrect axis parameter')

        if axis == 1:
            np.delete(self.bi, index, axis = None)
            np.delete(self.Pi, index, axis = 0)
        else:
            np.delete(self.bj, index, axis = None)
            np.delete(self.Qj, index, axis = 0)
    
    
class DynamMatrix(ParamMatrix):

    def __init__(self, indexes, d, init_mean = 0.0, init_sd = 0.1):

        self.encoders = [OnlineIndexer(), OnlineIndexer()]

        shape = []
        for c in (0, 1):
            self.encoders[c].fit(indexes[c])
            shape += [self.encoders[c].size()]

        super(DynamMatrix, self).__init__((*shape, d), init_mean, init_sd)


    def get(self, index):

        if index is not None:
            for c in (0, 1):
                index[c] = self.encoders[c].transform(index[c])
                
        return super(DynamMatrix, self).get(index)


    def set(self, value, index):

        if index is not None:
            for c in (0, 1):
                index[c] = self.encoders[c].fit_transform(index[c])
        
        super(DynamMatrix, self).set(value, index)


    def expand(self, index, axis):

        if axis not in (0, 1):
            raise ValueError('Incorrect axis parameter')
        
        prev_size = self.encoders[axis].size()
        self.encoders[axis].fit(index)
        by = self.encoders[axis].size() - prev_size

        super(DynamMatrix, self).expand(by, axis)


    def drop(self, index, axis):

        if axis not in (0, 1):
            raise ValueError('Incorrect axis parameter')
            
        index = self.encoders[axis].transform(index)
        super(DynamMatrix, self).drop(index, axis)

