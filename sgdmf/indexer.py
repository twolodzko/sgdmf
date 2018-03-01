
from __future__ import print_function

import numpy as np
from collections import defaultdict, Iterable


def get_iterable(x):
    if isinstance(x, Iterable):
        return x
    else:
        return (x,)


class OnlineIndexer(object):
    
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
            new_keys = get_iterable(new_keys)
            for v, k in enumerate(new_keys, start = self.next_index()):
                self.classes_[k] = v
            self.last_index_ = v
        return self
    
    
    def transform(self, y):
        
        """Return the indexes for given keys
        
        Parameters
        ----------
        y : array
            List of keys.
        
        Returns
        -------
        array : indexes for the keys.
        """
        
        return [self.classes_[k] for k in get_iterable(y)]


    def fit_transform(self, y):

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
        return self.transform(y)
    
    
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
        
        for k in get_iterable(keys):
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
