
from __future__ import print_function

import numpy as np
from tqdm import tqdm

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y, check_array, shuffle

from .param import DynamMatrix, ParamMatrix


class MatrixFactorizer(BaseEstimator, RegressorMixin):
    
    """Matrix Factorizer
    
    Factorize the matrix R (n, m) into P (n, n_components) and Q (n_components, m) matrices,
    using stochastic gradient descent. Additional intercepts mu, bi, bj can be included,
    leading to the following model

        R[i,j] = mu + bi[i] + bj[j] + P[i,] * Q[,j]

    MatrixFactorizer assumes matrix R to be stored in a "long" format, in two arrays X (n_samples, 2)
    and y (n_samples). The i, j indexes are stored in the rows of X and the R[i,j] values are
    stored in accompanying indices of y (i.e. y[k] = R[i,j], X[k,0] = i, and X[k,1] = j).

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

    >>> import numpy as np
    >>> import pandas as pd
    >>> from sgdmf import MatrixFactorizer
    >>> data = pd.DataFrame({
                    'user_id' : [0,0,1,1,2,2],
                    'movie_id' : [0,1,2,0,1,2],
                    'rating' : [1,1,2,2,3,3]
                })
    >>> X = data[['user_id', 'movie_id']]
    >>> y = data['rating']
    >>> mf = MatrixFactorizer(n_components = 2, n_epoch = 100, random_state = 42)
    >>> mf.partial_fit(X, y)
    MatrixFactorizer(dynamic_indexes=True, fit_intercepts=True, init_mean=0.0,
             init_sd=0.1, learning_rate=0.005, n_components=2, n_epoch=100,
             progress=0, random_state=42, regularization=0.02, shuffle=False,
             warm_start=False)
    >>> mf.score(X, y)
    0.8724592718646421
    >>> mf.partial_fit(X, y)
    MatrixFactorizer(dynamic_indexes=True, fit_intercepts=True, init_mean=0.0,
             init_sd=0.1, learning_rate=0.005, n_components=2, n_epoch=100,
             progress=0, random_state=42, regularization=0.02, shuffle=False,
             warm_start=False)
    >>> mf.score(X, y)
    0.9641971102264503
    >>> mf.predict(X)
    array([1.04945549, 1.22171768, 2.14456295, 1.85838361, 2.78110453,
           2.94767306])
    
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
        self.params_ = None
        self.N_ = 0

    
    def init_param(self, shape = None, indexes = None):
        
        """Initialize the parameters
        
        The mu, bi, bj parameters are initialized with zeros, while
        P, Q are initialized randomly using values drawn from
        normal distribution parametrized by mu_init and sd_init.
        
        Notice that this initialization would be respected only
        when warm_start=True, or when using partial_fit().
        
        Parameters
        ----------
        
        shape : tuple (int, int)

        indexes : list (array, array)
        
        """

        if indexes is not None:

            if shape is not None:
                raise ValueError("Use shape or indexes, but not both")

            if self.dynamic_indexes:
                self.params_ = DynamMatrix(indexes, self.n_components,
                                           self.init_mean, self.init_sd)
            else:
                n = np.max(indexes[0])
                m = np.max(indexes[1])
                shape = (n, m)

        if shape is not None:
            
            shape = (*shape, self.n_components)
            self.params_ = ParamMatrix(shape, self.init_mean, self.init_sd)


    def delete_index(self, index, axis):

        """Delete indexes

        Remove indexes from the parameters, e.g. in Users * Movies
        dataset some user deletes the account and we do not need to
        update and store her data any more.

        Parameters
        ----------
        index : array (n_indexes,)
            Indexes of the elements to be removed.
            
        axis : 0 or 1
            Axis of the element to be removed.
        """

        raise NotImplementedError()
    

    def profiles(self, index, axis):

        """Get the latent profiles

        Get indices of the P or Q matrix.

        Parameters
        ----------
        index : array (n_samples,) or None
            Index of the element to be returned. If None, then entire
            P or Q matrix is returned.
        
        axis : 0 or 1
            Axis of the index.

        Returns
        -------

        array, shape (n_samples, n_components)
           Latent profile for the given index.

        Examples
        --------

        >>> import numpy as np
        >>> import pandas as pd
        >>> from sgdmf import MatrixFactorizer
        >>> data = pd.DataFrame({
                        'user_id' : [0,0,1,1,2,2],
                        'movie_id' : [0,1,2,0,1,2],
                        'rating' : [1,1,2,2,3,3]
                    })
        >>> X = data[['user_id', 'movie_id']]
        >>> y = data['rating']
        >>> mf = MatrixFactorizer(n_components = 2, n_epoch = 500, random_state = 42)
        >>> mf.partial_fit(X, y)
        MatrixFactorizer(dynamic_indexes=True, fit_intercepts=True, init_mean=0.0,
                 init_sd=0.1, learning_rate=0.005, n_components=2, n_epoch=500,
                 progress=0, random_state=42, regularization=0.02, shuffle=False,
                 warm_start=False)
        >>> mf.profiles(X.iloc[:, 0], axis = 0)
        array([[-0.04421832,  0.00246789],
               [-0.04421832,  0.00246789],
               [ 0.11220089,  0.17532256],
               [ 0.11220089,  0.17532256],
               [ 0.00394179, -0.07307977],
               [ 0.00394179, -0.07307977]])

        """

        return self.params_.get(index, axis)

    
    def _sdg_step(self, ij, y):
        
        # Single step of stochastic grandient descent using single
        # data "point" (a row): x (two indexes), y (numeric value).
        
        ij = list(ij)
        mu, bi, bj, p, q = self.params_.get(ij)
        p, q = p[0], q[0]
        
        err = y - (mu + bi + bj + np.dot(p, q))
            
        if self.fit_intercepts:
            bi = bi + self.learning_rate * (err - self.regularization*bi)
            bj = bj + self.learning_rate * (err - self.regularization*bj)
        
        p = p + self.learning_rate * (err*q - self.regularization*p)
        q = q + self.learning_rate * (err*p - self.regularization*q)

        self.params_.set([mu, bi, bj, p, q], ij)


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

        if self.params_ is None:
            if self.dynamic_indexes:
                indexes = []
                for c in (0, 1):
                    indexes += [np.unique(X[c, :])]
                self.init_param(indexes = indexes)
            else:
                n, m = np.max(X, axis = 0) + 1
                self.init_param(shape = (n, m))
        
        # update mu intercept using moving average
        if self.fit_intercepts:
            mu = self.params_.get(axis = -1)
            mu = (mu * self.N_ + np.sum(y)) / (self.N_ + X.shape[0])
            self.params_.set(mu, axis = -1)
        
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
        
        yhat = np.empty(X.shape[0])
        
        for row in tqdm(range(X.shape[0]), disable = self.progress <= 1):
            mu, bi, bj, p, q = self.params_.get(X[row, :])
            yhat[row] = mu + bi + bj + np.dot(p, q)
        
        return yhat
    
