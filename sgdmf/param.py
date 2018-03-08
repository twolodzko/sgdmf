
from __future__ import print_function

import numpy as np
from .indexer import OnlineIndexer


class ParamMatrix(object):

    def __init__(self, shape, init_mean = 0.0, init_sd = 0.1):

        self.init_mean = init_mean
        self.init_sd = init_sd
        self.shape = np.array(shape, dtype=int)
        self.mu = 0.0
        self.bi = np.zeros(self.shape[0])
        self.bj = np.zeros(self.shape[1])
        self.Pi = np.random.normal(self.init_mean, self.init_sd, size = self.shape[[0, 2]])
        self.Qj = np.random.normal(self.init_mean, self.init_sd, size = self.shape[[1, 2]])


    def get(self, index = None, axis = None):

        if axis is None:
            
            i, j = index
            return self.mu, self.bi[i], self.bj[j], self.Pi[i, :], self.Qj[j, :]

        else:

            if axis not in (-1, 0, 1):
                raise ValueError('Incorrect axis parameter')

            if axis == 0:
                return self.bi[index], self.Pi[index, :]
            elif axis == 1: 
                return self.bj[index], self.Qj[index, :]
            else:
                return self.mu


    def set(self, value, index = None, axis = None):

        if axis is None:
            
            i, j = index
            self.mu = value[0]
            self.bi[i] = value[1]
            self.bj[j] = value[2]
            self.Pi[i, :] = value[3]
            self.Qj[j, :] = value[4]

        else:

            if axis not in (-1, 0, 1):
                raise ValueError('Incorrect axis parameter')

            if axis == 0:
                self.bi[index] = value[0]
                self.Pi[index, :] = value[1]
            elif axis == 1: 
                self.bj[index] = value[0]
                self.Qj[index, :] = value[1]
            else:
                self.mu = value
    

    def expand(self, by, axis):

        if axis not in (0, 1):
            raise ValueError('Incorrect axis parameter')

        if axis == 1:
            self.bi = np.append(self.bi, np.zeros(by))
            self.Pi = np.append(self.Pi, np.random.normal(self.init_mean, self.init_sd,
                                                          size = (by, self.shape[2])), axis = 0)
        else:
            self.bj = np.append(self.bj, np.zeros(by))
            self.Qj = np.append(self.Qj, np.random.normal(self.init_mean, self.init_sd,
                                                          size = (by, self.shape[2])), axis = 0)


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


    def get(self, index = None, axis = None):

        if index is not None:
            index = list(index)
            for c in (0, 1):
                index[c] = self.encoders[c].transform(index[c])
        
        return super(DynamMatrix, self).get(index, axis)


    def set(self, value, index = None, axis = None):

        if axis is None:
            index = list(index)
            for c in (0, 1):
                index[c] = self.encoders[c].transform(index[c])
        else:
            if index is not None:
                if axis in (0, 1):
                    index = self.encoders[axis].fit_transform(index)
        
        super(DynamMatrix, self).set(value, index, axis)


    def expand(self, index, axis):

        if axis not in (0, 1):
            raise ValueError('Incorrect axis parameter')
        
        prev_size = self.encoders[axis].size()
        index = self.encoders[axis].fit_transform(index)
        new_size = self.encoders[axis].size() - prev_size

        super(DynamMatrix, self).expand(new_size, axis)


    def drop(self, index, axis):

        if axis not in (0, 1):
            raise ValueError('Incorrect axis parameter')
            
        index = self.encoders[axis].transform(index)
        super(DynamMatrix, self).drop(index, axis)

