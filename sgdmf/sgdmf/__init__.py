
from __future__ import print_function

from collections import defaultdict
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y
from sklearn.metrics import r2_score

from .indexer import OnlineIndexer
from .mf import MatrixFactorizer