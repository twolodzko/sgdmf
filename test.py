
import numpy as np
from sgdmf import MatrixFactorizer


if __name__ == "__main__":

    ## Train on valid data

    X = np.array([
        [0,0,1,1,2,2],
        [0,1,2,0,1,2],
    ]).T

    y = np.array([1,1,2,2,3,3])

    mf = MatrixFactorizer(n_components = 2, n_epoch = 1000)

    mf.fit(X, y)

    assert mf.P_.shape[0] == X[:, 0].max(axis = 0) + 1
    assert mf.intercepts_[1].shape[0] == X[:, 0].max(axis = 0) + 1
    assert mf.Q_.shape[1] == X[:, 1].max(axis = 0) + 1
    assert mf.intercepts_[2].shape[0] == X[:, 1].max(axis = 0) + 1

    assert mf.score(X, y) > 0.99

    ## Update on new data

    X_new = np.array([
        [2,2,3,1,4,3],
        [0,3,3,0,1,2],
    ]).T

    y_new = np.array([1,4,3,2,4,3])

    mf.partial_fit(X_new, y_new)

    assert mf.P_.shape[0] == X_new[:, 0].max(axis = 0) + 1
    assert mf.intercepts_[1].shape[0] == X_new[:, 0].max(axis = 0) + 1
    assert mf.Q_.shape[1] == X_new[:, 1].max(axis = 0) + 1
    assert mf.intercepts_[2].shape[0] == X_new[:, 1].max(axis = 0) + 1


    ## Train with fixed indexes

    X = np.array([
        [0,0,1,1,2,2],
        [0,1,2,0,1,2],
    ]).T

    y = np.array([1,1,2,2,3,3])

    mf = MatrixFactorizer(n_components = 2, n_epoch = 1000)
    mf.init_param(5, 6)

    assert mf.P_.shape[0] == 5
    assert mf.intercepts_[1].shape[0] == 5
    assert mf.Q_.shape[1] == 6
    assert mf.intercepts_[2].shape[0] == 6

    mf.partial_fit(X, y)

    assert mf.P_.shape[0] == 5
    assert mf.intercepts_[1].shape[0] == 5
    assert mf.Q_.shape[1] == 6
    assert mf.intercepts_[2].shape[0] == 6

    assert mf.score(X, y) > 0.99

    # shuffle=True works properly
    mf = MatrixFactorizer(n_components = 2, n_epoch = 1000, shuffle = True)
    mf.partial_fit(X, y)
    assert mf.score(X, y) > 0.99

    mf = MatrixFactorizer(n_components = 2, n_epoch = 1000, warm_start = True)
    mf.init_param(5, 6)

    assert mf.P_.shape[0] == 5
    assert mf.intercepts_[1].shape[0] == 5
    assert mf.Q_.shape[1] == 6
    assert mf.intercepts_[2].shape[0] == 6

    mf.fit(X, y)

    assert mf.P_.shape[0] == 5
    assert mf.intercepts_[1].shape[0] == 5
    assert mf.Q_.shape[1] == 6
    assert mf.intercepts_[2].shape[0] == 6


    ## Check if seeds work

    mf = MatrixFactorizer(n_components = 2, n_epoch = 5)

    # without seed, results are inconsistent
    assert mf.fit(X, y).score(X, y) != mf.fit(X, y).score(X, y)

    mf = MatrixFactorizer(n_components = 2, n_epoch = 5, random_state = 42)

    # seeds make results consistent
    assert mf.fit(X, y).score(X, y) == mf.fit(X, y).score(X, y)

    # partial_fit updates, no matter of seed
    assert mf.fit(X, y).score(X, y) != mf.partial_fit(X, y).score(X, y)

    mf = MatrixFactorizer(n_components = 2, n_epoch = 5, random_state = 42)

    # partial_fit's are consistent
    assert mf.fit(X, y).partial_fit(X, y).score(X, y) == mf.fit(X, y).partial_fit(X, y).score(X, y)


    ## Order indexes

    X = np.array([
        [0,0,1,1,5,5],
        [0,1,2,0,1,2],
    ]).T

    y = np.array([1,1,2,2,3,3])

    mf = MatrixFactorizer(n_components = 2, n_epoch = 1000)

    mf.fit(X, y)

    X_new = np.array([
        [2,2,3,1,4,3],
        [0,3,3,0,1,2],
    ]).T

    y_new = np.array([1,4,3,2,4,3])

    mf.partial_fit(X_new, y_new)

    old_values = mf.encoders_[0].values()
    old_keys = mf.encoders_[0].keys()

    seq_values = [i for i in range(len(old_values))]

    # are not sorted
    assert not np.all(old_values == seq_values)

    idx_i = np.array(mf.encoders_[0].keys()).argsort()
    sorted_P = mf.P_[idx_i, :]
    sorted_bi = mf.intercepts_[1][idx_i]

    idx_j = np.array(mf.encoders_[1].keys()).argsort()
    sorted_Q = mf.Q_[:, idx_j]
    sorted_bj = mf.intercepts_[2][idx_j]

    mf.order_index()

    new_values = mf.encoders_[0].values()

    # are sorted
    assert np.all(new_values == seq_values)

    # sorting parameters works
    assert np.all(mf.P_ == sorted_P)
    assert np.all(mf.Q_ == sorted_Q)
    assert np.all(mf.intercepts_[1] == sorted_bi)
    assert np.all(mf.intercepts_[2] == sorted_bj)


    ## Deleting indexes works

    X = np.array([
        [0,0,1,1,5,5],
        [0,1,2,0,1,2],
    ]).T

    y = np.array([1,1,2,2,3,3])

    mf = MatrixFactorizer(n_components = 2, n_epoch = 5)

    mf.fit(X, y)

    assert np.all(mf.P_.shape == (3, 2))
    assert np.all(mf.Q_.shape == (2, 3))
    assert np.all(mf.intercepts_[1].shape[0] == 3)
    assert np.all(mf.intercepts_[2].shape[0] == 3)

    mf.delete_index(0, axis = 0)
    assert np.all(mf.P_.shape == (2, 2))
    assert np.all(mf.intercepts_[1].shape[0] == 2)

    mf.delete_index(0, axis = 1)
    assert np.all(mf.Q_.shape == (2, 2))
    assert np.all(mf.intercepts_[2].shape[0] == 2)
