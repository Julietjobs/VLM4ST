"""
Data normalization utilities
"""

import numpy as np


class MinMaxNormalization:
    """
    MinMax Normalization to [-1, 1] range
    Formula: x = (x - min) / (max - min) * 2 - 1
    """
    
    def __init__(self):
        self._min = None
        self._max = None
    
    def fit(self, X):
        """
        Compute min and max values from the data
        
        Args:
            X: Input data (numpy array or value)
        """
        self._min = X.min()
        self._max = X.max()
        print(f"MinMaxNormalization fitted: min={self._min:.4f}, max={self._max:.4f}")
    
    def transform(self, X):
        """
        Apply normalization to [-1, 1]
        
        Args:
            X: Input data
            
        Returns:
            Normalized data in [-1, 1]
        """
        if self._min is None or self._max is None:
            raise ValueError("Must call fit() before transform()")
        
        X = (X - self._min) / (self._max - self._min)
        X = X * 2.0 - 1.0
        return X
    
    def fit_transform(self, X):
        """
        Fit and transform in one step
        
        Args:
            X: Input data
            
        Returns:
            Normalized data in [-1, 1]
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """
        Inverse normalization from [-1, 1] back to original scale
        
        Args:
            X: Normalized data in [-1, 1]
            
        Returns:
            Data in original scale
        """
        if self._min is None or self._max is None:
            raise ValueError("Must call fit() before inverse_transform()")
        
        X = (X + 1.0) / 2.0
        X = X * (self._max - self._min) + self._min
        return X

