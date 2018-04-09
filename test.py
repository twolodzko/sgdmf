
import numpy as np
from sgdmf import MatrixFactorizer


if __name__ == "__main__":

    ## Train on valid data

    d = 2

    X = np.array([
        [0,0,1,1,2,2],
        [0,1,2,0,1,2],
    ]).T

    y = np.array([1,1,2,2,3,3])

    mf = MatrixFactorizer(n_components = d, n_epoch = 1000)

    mf.fit(X, y)

    n, m = X.max(axis = 0) + 1
    assert np.all(mf.params_.shape == (n, m, d))
    assert mf.score(X, y) > 0.99

    ## Update on new data

    X_new = np.array([
        [2,2,3,1,4,3],
        [0,3,3,0,1,2],
    ]).T

    y_new = np.array([1,4,3,2,4,3])

    mf.partial_fit(X_new, y_new)

    assert np.all(mf.params_.shape == (5, 4, d))


    ## Train with fixed indexes

    X = np.array([
        [0,0,1,1,2,2],
        [0,1,2,0,1,2],
    ]).T

    y = np.array([1,1,2,2,3,3])

    mf = MatrixFactorizer(n_components = d, n_epoch = 1000,
                          dynamic_indexes = False)

    mf.init_param(5, 6)

    assert np.all(mf.params_.shape == (5, 6, d))

    mf.partial_fit(X, y)

    assert np.all(mf.params_.shape == (5, 6, d))
    assert mf.score(X, y) > 0.99

    # shuffle=True works properly
    mf = MatrixFactorizer(n_components = d, n_epoch = 1000, shuffle = True)
    mf.partial_fit(X, y)
    assert mf.score(X, y) > 0.99

    ## Check if seeds work

    mf = MatrixFactorizer(n_components = d, n_epoch = 5)

    # without seed, results are inconsistent
    assert mf.fit(X, y).score(X, y) != mf.fit(X, y).score(X, y)

    mf = MatrixFactorizer(n_components = d, n_epoch = 5, random_state = 42)

    # seeds make results consistent
    assert mf.fit(X, y).score(X, y) == mf.fit(X, y).score(X, y)

    # partial_fit updates, no matter of seed
    assert mf.fit(X, y).score(X, y) != mf.partial_fit(X, y).score(X, y)

    mf = MatrixFactorizer(n_components = d, n_epoch = 5, random_state = 42)

    # partial_fit's are consistent
    assert mf.fit(X, y).partial_fit(X, y).score(X, y) == mf.fit(X, y).partial_fit(X, y).score(X, y)


    ## Deleting indexes works

    X = np.array([
        [0,0,1,1,5,5],
        [0,1,2,0,1,2],
    ]).T

    y = np.array([1,1,2,2,3,3])

    mf = MatrixFactorizer(n_components = d, n_epoch = 5,
                          dynamic_indexes = True)

    mf.fit(X, y)

    assert np.all(mf.params_.shape == (3, 3, d))

    mf.delete_index(0, axis = 0)
    assert np.all(mf.params_.shape == (2, 3, d))

    mf.delete_index(0, axis = 1)
    assert np.all(mf.params_.shape == (2, 2, d))


    ## Import/export of the params_

    d = 2

    X = np.array([
        [0,0,1,1,5,5],
        [0,1,2,0,1,2],
    ]).T

    y = np.array([1,1,2,2,3,3])

    # dynamic indexes
    mf = MatrixFactorizer(n_components = d, n_epoch = 5,
                          dynamic_indexes = True)

    mf.fit(X, y)

    par_copy = mf.__dict__['params_'].to_dict()

    dct = mf.__dict__['params_'].to_dict()
    mf.__dict__['params_'].from_dict(dct)
    assert mf.__dict__['params_'].to_dict() == dct

    # static indexes
    mf = MatrixFactorizer(n_components = d, n_epoch = 5,
                          dynamic_indexes = False)

    mf.fit(X, y)

    dct = mf.__dict__['params_'].to_dict()
    mf.__dict__['params_'].from_dict(dct)
    assert mf.__dict__['params_'].to_dict() == dct


    ## Import/export of the model

    d = 2

    X = np.array([
        [0,0,1,1,5,5],
        [0,1,2,0,1,2],
    ]).T

    y = np.array([1,1,2,2,3,3])

    # dynamic indexes
    mf = MatrixFactorizer(n_components = d, n_epoch = 5,
                          dynamic_indexes = True)

    mf.fit(X, y)

    model_json = mf.to_json()
    mf.from_json(model_json)
    assert model_json == mf.to_json()

    # static indexes
    mf = MatrixFactorizer(n_components = d, n_epoch = 5,
                          dynamic_indexes = False)

    mf.fit(X, y)

    model_json = mf.to_json()
    mf.from_json(model_json)
    assert model_json == mf.to_json()



    