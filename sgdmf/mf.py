
from __future__ import print_function

import numpy as np
from tqdm import tqdm

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y, check_array, shuffle

from .indexer import OnlineIndexer


class MatrixFactorizer(BaseEstimator, RegressorMixin):
    
    """Matrix Factorizer
    
    Factorize the (n, m) matrix into mu + bi + bj + P*Q, where bi is (1, n) array,
    bj is (1, m) array, P and Q are matrices of shapes (n, n_components) and (n_components, m)
    using stochastic gradient descent.

    The dataset is processed as-is, notice that by default the rows are *not* shuffled, so it
    may be worth setting shuffle=True, or shuffling the data in advance to fitting the model
    (e.g. using sklearn.utils.shuffle).
    
    Parameters
    ----------
    
    n_components : int, default : 100
        Number of latent components to be estimated. The estimated latent matrices P and Q
        have (n, n_components) and (n_components, m) shapes subsequently.
    
    n_epoch : int, default : 5
        Number of training epochs, the actual number of iterations is n_samples * n_epoch.
    
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
        
    dynamic_indexes : bool, default : True
        As new indexes occur, expand the model for those indexes (change dimensions of bi, bj, P, Q).
        If set to False, if new index will occure during partial_fit(), it would result in an error.

    shuffle : bool, default : False
        Whether or not the training data should be shuffled before each epoch.
        
    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as initialization,
        otherwise, just erase the previous solution.

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling the data. If int,
        random_state is the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number generator is
        the RandomState instance used by np.random.

    progress : int (0, 1, 2), default : 0
        Show the progress bar, 0 disables progress bar, 1 shows progress per epoch, 2 shows progress
        per epoch and per case.

    Attributes
    ----------
    
    intercepts_ : array (float, array, array)
        Constants in decision function (mu, bi, bj).
    
    P_ : array, shape (n, n_components)
        Latent matrix.
    
    Q_ : array, shape (n_components, m)
        Latent matrix.

    Examples
    --------

    >>> from sgdmf import MatrixFactorizer
    >>> X = data[['user_id', 'movie_id']]
    >>> y = data['rating']
    >>> mf = MatrixFactorizer()
    >>> mf.partial_fit(X, y)
    MatrixFactorizer(dynamic_indexes=True, fit_intercepts=True, init_mean=0.0,
         init_sd=0.1, learning_rate=0.005, n_components=100, n_epoch=5,
         progress=0, random_state=None, regularization=0.02, shuffle=False,
         warm_start=False)
    
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
                 fit_intercepts = True, dynamic_indexes = True, shuffle = False,
                 warm_start = False, random_state = None, progress = 0):
        
        self.n_components = n_components
        self.n_epoch = n_epoch
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.init_mean = init_mean
        self.init_sd = init_sd
        self.fit_intercepts = fit_intercepts
        self.dynamic_indexes = dynamic_indexes
        self.shuffle = shuffle
        self.warm_start = warm_start
        self.random_state = random_state
        self.progress = int(progress)

        # initialize empty parameters
        self._reset_param()
        
            
    def _reset_param(self):
        
        np.random.seed(self.random_state)

        self.intercepts_ = self.P_ = self.Q_ = None
        self.encoders_ = [OnlineIndexer(), OnlineIndexer()]
        self.N_ = 0

    
    def init_param(self, n, m):
        
        """Initialize the parameters
        
        The mu, bi, bj parameters are initialized with zeros, while
        P, Q are initialized randomly using values drawn from
        normal distribution parametrized by mu_init and sd_init.
        
        Notice that this initialization would be respected only
        when warm_start=True, or when using partial_fit().
        
        Parameters
        ----------
        n : int
            First index of the factorized matrix.
            
        m : int
            Second index of the factorized matrix.
        
        """
        
        d = self.n_components
        self.P_ = np.random.normal(self.init_mean, self.init_sd, size = (n, d))
        self.Q_ = np.random.normal(self.init_mean, self.init_sd, size = (d, m))
        self.intercepts_ = [0.0, np.zeros(n), np.zeros(m)]
    
        
    def _max_Xij(self, X):
        return X.max(axis = 0) + 1
    
    
    def _get_PQ_dims(self):
        n = self.P_.shape[0]
        m = self.Q_.shape[1]
        d = self.n_components
        return n, m, d


    def _check_param(self, X):

        # Check if parameters were initialized and are consistent with data
        
        if self.intercepts_ is None or self.P_ is None or self.Q_ is None:
            raise ValueError('Parameters were not initialized yet')
        
        n, m, d = self._get_PQ_dims()
        max_i, max_j = self._max_Xij(X)

        if n < max_i or m < max_j:
            raise KeyError('X contains new indexes')
            
        if len(self.intercepts_[1]) < max_i or len(self.intercepts_[2]) < max_j:
            raise KeyError('X contains new indexes')
    

    def _check_indexes(self, X):

        if X.shape[1] != 2:
            raise ValueError('X needs to consist of exactly two columns')
        
        if not np.all(X >= 0):
            raise ValueError('Indexes need to be non-negative')

        if not np.all(np.isfinite(X)):
            raise ValueError('Indexes need to be finite')

        if not X.dtype in ('int', 'int32', 'int64'):
            raise ValueError('Indexes need to be integers')
    
    
    def _expand_param(self, X):

        # Initialize the parameters for the previously unseen indexes

        n, m, d = self._get_PQ_dims()
        max_i, max_j = self._max_Xij(X)
        
        if max_i > n:
            new_n = max_i - n
            new_P = np.random.normal(self.init_mean, self.init_sd, size = (new_n, d))
            self.P_ = np.append(self.P_, new_P, axis = 0)
            self.intercepts_[1] = np.append(self.intercepts_[1], np.zeros(new_n))
        
        if max_j > m:
            new_m = max_j - m
            new_Q = np.random.normal(self.init_mean, self.init_sd, size = (d, new_m))
            self.Q_ = np.append(self.Q_, new_Q, axis = 1)
            self.intercepts_[2] = np.append(self.intercepts_[2], np.zeros(new_m))


    def _encode_ij(self, X, update = True):

        # Encode and update the indexes
        # If update=True, the indexes are updated when
        # encountering previously unseen indexes.

        for c in (0, 1):
            if update:
                self.encoders_[c].fit(X[:, c])
            X[:, c] = self.encoders_[c].transform(X[:, c])
        return X


    def delete_index(self, index, axis):

        """Delete indexes

        Remove indexes from the parameters, e.g. in Users * Movies
        dataset some user deletes the account and we do not need to
        update and store her data any more.

        Parameters
        ----------
        index : int
            Index of the element to be removed.
            
        axis : 0 or 1
            Axis of the element to be removed.
        """

        if axis not in (0, 1):
            raise ValueError('Incorrect axis parameter')

        if self.dynamic_indexes:
            index = self.encoders_[axis].transform(index)

        if axis == 0:
            self.P_ = np.delete(self.P_, index, axis = 0)
            self.intercepts_[1] = np.delete(self.intercepts_[1], index, axis = 0)
        else:
            self.Q_ = np.delete(self.Q_, index, axis = 1)
            self.intercepts_[2] = np.delete(self.intercepts_[2], index, axis = 0)
    

    def order_index(self):

        """Order the parameters by indexes

        Parameters are ordered according to indexes, this function can be used
        when using partial_fit() has introduced new indexes.
        """

        if self.dynamic_indexes:

            for c in (0, 1):
                keys = self.encoders_[c].keys()
                order = np.argsort(keys)
                
                if c == 0:
                    self.P_ = self.P_[order, :]
                else:
                    self.Q_ = self.Q_[:, order]

                self.intercepts_[c+1] = self.intercepts_[c+1][order]

                self.encoders_[c].reindex()

    
    def _sdg_step(self, x, y):
        
        # Single step of stochastic grandient descent using single
        # data "point" (a row): x (two indexes), y (numeric value).
        
        mu, bi, bj = self.intercepts_
        i, j = x
        p = self.P_[i, :]
        q = self.Q_[:, j]
        
        err = y - (mu + bi[i] + bj[j] + np.dot(p, q))
            
        if self.fit_intercepts:
            bi[i] = bi[i] + self.learning_rate * (err - self.regularization*bi[i])
            bj[j] = bj[j] + self.learning_rate * (err - self.regularization*bj[j])
            self.intercepts_ = [mu, bi, bj]
        
        self.P_[i, :] = p + self.learning_rate * (err*q - self.regularization*p)
        self.Q_[:, j] = q + self.learning_rate * (err*p - self.regularization*q)
    
    
    def partial_fit(self, X, y):
        
        """Fit the model according to the given training data.
        
        Fit the model by updating the parameters given new data.
        
        Parameters
        ----------
        
        X : array, shape (n_samples, 2)
            Subset of training data.
            
        y : numpy array of shape (n_samples,)
            Subset of target values.
            
        Returns
        -------
        
        self : returns an instance of self.
        
        """

        X, y = check_X_y(X, y, y_numeric = True, ensure_2d = True,
                         force_all_finite = True, dtype = 'int32')

        # using only the first two columns as indexes
        X = X[:, :2]
        
        if self.dynamic_indexes:
            X = self._encode_ij(X)
        
        self._check_indexes(X)
        
        if self.intercepts_ is None and self.P_ is None and self.Q_ is None:
            n, m = self._max_Xij(X)
            self.init_param(n, m)
        
        if self.dynamic_indexes:
            self._expand_param(X)
        
        self._check_param(X)
        
        # update mu intercept using moving average
        if self.fit_intercepts:
            self.intercepts_[0] = (self.intercepts_[0] * self.N_ + np.sum(y)) / (self.N_ + X.shape[0])
        
        self.N_ += X.shape[0]
        
        for _ in tqdm(range(self.n_epoch), disable = self.progress <= 0):
            if self.shuffle:
                X, y = shuffle(X, y)
            for row in tqdm(range(X.shape[0]), disable = self.progress <= 1):
                self._sdg_step(X[row, :], y[row])
        
        return self
    
    
    def fit(self, X, y):
        
        """Fit the model according to the given training data.
        
        Fit the model from the scratch. The dimensions of the bi, bj, P, Q
        parameters are infered from the data, the parameters are
        initialized using the init_param() function.
        
        Parameters
        ----------
        
        X : array, shape (n_samples, 2)
            Subset of training data.
            
        y : numpy array of shape (n_samples,)
            Subset of target values.
        
        Returns
        -------
        
        self : returns an instance of self.
            
        """

        # reset parameters, so NOT to conduct partial_fit()
        if not self.warm_start:
            self._reset_param()
        
        return self.partial_fit(X, y)
    
    
    def predict(self, X):
        
        """Predict using the Matrix Factorization model
        
        Parameters
        ----------
        
        X : array, shape (n_samples, 2)
        
        Returns
        -------
        
        array, shape (n_samples,)
           Predicted target values per element in X.
           
        """

        X = check_array(X, ensure_2d = True, force_all_finite = True,
                        dtype = 'int32')

        # using only the first two columns as indexes)
        X = X[:, :2]
        
        if self.dynamic_indexes:
            X = self._encode_ij(X, update = False)
            
        self._check_indexes(X)
        self._check_param(X)
        
        mu, bi, bj = self.intercepts_
        yhat = np.empty(X.shape[0])
        
        for row in tqdm(range(X.shape[0]), disable = self.progress <= 1):
            i, j = X[row, :]
            p = self.P_[i, :]
            q = self.Q_[:, j]
            yhat[row] = mu + bi[i] + bj[j] + np.dot(p, q)
        
        return yhat
    
    
    def pred_matrix(self):
        
        """Predict the matrix of all the n * m pairs of users and items
        
        WARNING: this may be slow and computationally demanding!
        
        Returns
        -------
        
        array, shape (n, m)
           Predicted target values per all n * m indexes in the training data X.
           
        """
        
        n, m, d = self._get_PQ_dims()
        
        if self.fit_intercepts:
            mu = self.intercepts_[0]
            bi = np.array([self.intercepts_[1]] * m).transpose()
            bj = np.array([self.intercepts_[2]] * n)
            return mu + bi + bj + np.dot(self.P_, self.Q_)
        else:
            return np.dot(self.P_, self.Q_)
    
