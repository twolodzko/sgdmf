from __future__ import print_function

import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.utils import shuffle



class OnlineIndexer():
    
    """OnlineIndexer class
    
    OnlineIndexer object indexes the given keys, stores the indexes
    and when provided with previously unseen keys, it assigns new
    indexes for them.
    
    Examples
    --------
    >>> keys = ['d', 'c', 'a', 'd', 'a', 'a', 'e']
    >>> idx = OnlineIndexer()
    >>> idx.transform(keys)
    [2, 1, 0, 2, 0, 0, 3]
    >>> idx.transform(['a', 'z'])
    [0, 4]
    >>> idx.transform(['a', 'b', 'c'])
    [0, 5, 1]
    >>> idx.reindex()
    <__main__.OnlineIndexer at 0x7fb0a7d925f8>
    >>> idx.transform(['a', 'b', 'c', 'd'])
    [0, 1, 2, 3]
    
    """
    
    
    def __init__(self):
        
        self.classes_ = defaultdict(lambda : None)
        self.last_index_ = None
    
    
    def reset(self):
        
        """Reset the OnlineIndexer object to vanilla state
        
        Returns
        -------
        self : returns an instance of self.
        """
        
        self.__init__()
        return self
    
    
    def fit(self, y):
        
        """Create indexes for the OnlineIndexer object
        
        Parameters
        ----------
        y : array
            List of keys.
        
        Returns
        -------
        self : returns an instance of self.
        """
        
        new_keys = [k for k in np.unique(y) if k not in self.classes_]
        if len(new_keys):
            for v, k in enumerate(new_keys, start = self.next_index()):
                self.classes_[k] = v
            self.last_index_ = v
        return self
    
    
    def transform(self, y):
        
        """Return the indexes for given keys, create new indexes for the new keys
        
        Parameters
        ----------
        y : array
            List of keys.
        
        Returns
        -------
        array : indexes for the keys.
        """
        
        self.fit(y)
        return [self.classes_[k] for k in y]
    
    
    def keys(self):
        
        """Return the list of keys
        
        Returns
        -------
        array : list of keys.
        """
        
        return list(self.classes_.keys())
    
    
    def values(self):
        
        """Return the list of indexes
        
        Returns
        -------
        array : list of indexes.
        """
        
        return list(self.classes_.values())
    
    
    def next_index(self):
        
        """Next index to be used for the unseen key"""
        
        if self.last_index_:
            return self.last_index_ + 1
        else:
            return 0
    
    
    def delete(self, keys):
        
        """Delete keys from the OnlineIndexer object
        
        Returns
        -------
        self : returns an instance of self.
        """
        
        for k in keys:
            del self.classes_[k]
        return self
    
    
    def reindex(self):
        
        """Reindex the keys
        
        Keys are sorted and indexed again. 
        
        Returns
        -------
        self : returns an instance of self.
        """
        
        keys = sorted(self.keys())
        self.reset()
        for v, k in enumerate(keys):
            self.classes_[k] = v
        return self



class MatrixFactorizer(BaseEstimator, RegressorMixin):
    
    """Matrix Factorizer
    
    Factorize the (n, m) matrix into mu + bi + bj + P*Q, where bi is (1, n) array,
    bj is (1, m) array, P and Q are matrices of shapes (n, n_components) and (n_components, m)
    using stochastic gradient descent.
        
    Parameters
    ----------
    
    n_components : int, default : 100
        Number of latent components to be estimated. The estimated latent matrices P and Q
        have (n, n_components) and (n_components, m) shapes subsequently.
    
    n_epoch : int, default : 5
        Number of training epochs, number of iterations is n_iter_ = n_samples * n_epoch.
    
    learning_rate : float, default : 0.005
        Learning rate parameter.
    
    regularization : float, default : 0.02
        Regularization parameter.
    
    init_mean : float, default : 0.0
        The mean of the normal distribution for factor vectors initialization. 
    
    init_sd : float, default : 0.1
        The standard deviation of the normal distribution for factor vectors initialization. 
    
    fit_intercepts : bool, default : True
        When set to True, the mu, bi, bj intercepts are fitted, otherwise
        only the P and Q latent matrices are fitted.
        
    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as initialization,
        otherwise, just erase the previous solution.

        
    Attributes
    ----------
    
    intercepts_ : array , shape (1, 3)
        Constants in decision function.
    
    P_ : array, shape (n, n_components)
        Latent matrix.
    
    Q_ : array, shape (n_components, m)
        Latent matrix.
    
    n_iter_ : int
        The actual number of iterations.
        
        
    References
    ----------
        
    Koren, Y., Bell, R., & Volinsky, C. (2009).
    Matrix factorization techniques for recommender systems. Computer, 42(8).
        
    Yu, H. F., Hsieh, C. J., Si, S., & Dhillon, I. (2012, December).
    Scalable coordinate descent approaches to parallel matrix factorization for recommender systems.
    In Data Mining (ICDM), 2012 IEEE 12th International Conference on (pp. 765-774). IEEE.
        
    """
    
    
    def __init__(self, n_components = 100, n_epoch = 5, learning_rate = 0.005,
                 regularization = 0.02, init_mean = 0.0, init_sd = 0.1,
                 fit_intercepts = True, warm_start = False):
        
        self.d = n_components
        self.n_epoch = n_epoch
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.init_mean = init_mean
        self.init_sd = init_sd
        self.fit_intercepts = fit_intercepts
        self.warm_start = warm_start
        
        # initialise parameters with empty values
        self._reset_param()
        
            
    def _reset_param(self):
        
        self.intercepts_ = self.P_ = self.Q_ = None
        self.N_ = self.n_iter_ = None
        self.encoders_ = [OnlineIndexer(), OnlineIndexer()]
        
        
    def _init_param(self, X):
        
        """
        Initialise the parameters in intercepts_, P_, Q_ and N_ where
        * intercepts_ are initialised with zeros and
        * P_, Q_ are initialized randomly using normal distribution
          parametrized by mu_init and sd_init.
        """
        
        n, m = X[:, :2].max(axis = 0) + 1
                                       
        if self.P_ is None:
            self.P_ = np.random.normal(self.init_mean, self.init_sd, size = (n, self.d))
            
        if self.Q_ is None:
            self.Q_ = np.random.normal(self.init_mean, self.init_sd, size = (self.d, m))
            
        if self.intercepts_ is None:
            self.intercepts_ = [0.0, np.zeros(n), np.zeros(m)]
        
        if self.N_ is None:
            self.N_ = 0
        
        if self.n_iter_ is None:
            self.n_iter_ = 0
            
            
    def _expand_param(self, X):

        n = self.P_.shape[0]
        m = self.Q_.shape[1]
        
        max_xi, max_xj = X[:, :2].max(axis = 0) + 1
        
        if max_xi > n:
            nn = max_xi - n
            tmp = np.random.normal(self.init_mean, self.init_sd, size = (nn, self.d))
            self.P_ = np.append(self.P_, tmp, axis = 0)
            self.intercepts_[1] = np.append(self.intercepts_[1], np.zeros(nn))
        
        if max_xj > m:
            mm = max_xj - m
            tmp = np.random.normal(self.init_mean, self.init_sd, size = (self.d, mm))
            self.Q_ = np.append(self.Q_, tmp, axis = 1)
            self.intercepts_[2] = np.append(self.intercepts_[2], np.zeros(mm))
            
            
    def _verify_param(self, X):
        
        n = self.P_.shape[0]
        m = self.Q_.shape[1]
        
        max_xi, max_xj = X[:, :2].max(axis = 0) + 1

        if n < max_xi or m < max_xj or \
           len(self.intercepts_[1]) < max_xi or \
           len(self.intercepts_[2]) < max_xj:
            raise ValueError('X contains new labels')
            
            
    def _encode_X(self, X):
        for c in (0, 1):
            X[:, c] = self.encoders_[c].transform(X[:, c])
        return X
                                          
        
    def _sdg_step(self, x, y):
        
        """
        Single step of Stochastic Grandient Descent that computes
        the mu, bi, bj, p, q parameters using X, y.
        """
        
        mu, bi, bj = self.intercepts_
        i, j = x[:2]
        p = self.P_[i, :]
        q = self.Q_[:, j]
        
        e = y - (mu + bi[i] + bj[j] + np.dot(p, q))
            
        if self.fit_intercepts:
            bi[i] = bi[i] + self.learning_rate * (e - self.regularization*bi[i])
            bj[j] = bj[j] + self.learning_rate * (e - self.regularization*bj[j])
            self.intercepts_ = [mu, bi, bj]
        
        self.P_[i, :] = p + self.learning_rate * (e*q - self.regularization*p)
        self.Q_[:, j] = q + self.learning_rate * (e*p - self.regularization*q)
         
    
    def partial_fit(self, X, y):
        
        """Fit the model according to the given training data.
        
        Parameters
        ----------
        
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of training data
            
        y : numpy array of shape (n_samples,)
            Subset of target values
            
        Returns
        -------
        
        self : returns an instance of self.
        
        """
        
        X, y = check_X_y(X, y, y_numeric = True, ensure_2d = True)        
        X = self._encode_X(X)
                    
        self._init_param(X)
        self._expand_param(X)
        self._verify_param(X)
        
        if self.fit_intercepts:
            self.intercepts_[0] = (self.intercepts_[0] * self.N_ + np.sum(y)) / (self.N_ + X.shape[0])
        
        self.N_ += X.shape[0]
        n_iter = int(X.shape[0] * self.n_epoch)
        self.n_iter_ += n_iter
        
        for _ in range(self.n_epoch):
            for row in range(X.shape[0]):
                self._sdg_step(X[row, :2], y[row])
        
        return self
    
    
    def fit(self, X, y):
        
        """Fit the model according to the given training data.
        
        Parameters
        ----------
        
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of training data
            
        y : numpy array of shape (n_samples,)
            Subset of target values
        
        Returns
        -------
        
        self : returns an instance of self.
            
        """
        
        if not self.warm_start:
            # reset parameters so not to conduct partial_fit()
            self._reset_param()
        
        return self.partial_fit(X, y)
    
    
    def predict(self, X):
        
        """Predict using the Matrix Factorization model
        
        Parameters
        ----------
        
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
        
        Returns
        -------
        
        array, shape (n_samples,)
           Predicted target values per element in X.
           
        """
        
        X = self._encode_X(X)
        self._verify_param(X)
        
        mu, bi, bj = self.intercepts_
        yhat = np.empty(X.shape[0])
        
        for row in range(X.shape[0]):
            i, j = X[row, :2]
            p = self.P_[i, :]
            q = self.Q_[:, j]
            yhat[row] = mu + bi[i] + bj[j] + np.dot(p, q)
        
        return yhat
    
    
    def pred_matrix(self):
        
        """Predict the matrix of all the n * m pairs of users and items
        
        Returns
        -------
        
        array, shape (n, m)
           Predicted target values per all n * m indexes in the training data X.
           
        """
        
        n = self.P_.shape[0]
        m = self.Q_.shape[1]
        
        if self.fit_intercepts:
            mu = self.intercepts_[0]
            bi = np.array([self.intercepts_[1]] * m).transpose()
            bj = np.array([self.intercepts_[2]] * n)
            return mu + bi + bj + np.dot(self.P_, self.Q_)
        else:
            return np.dot(self.P_, self.Q_)
    
    
