import torch
import torch.nn as nn
import numpy as np
from scipy.special import binom
from scipy.integrate import odeint

class Library():
    """SINDy feature library object.

    SINDy feature library object.
    Modeled after:
    https://pysindy.readthedocs.io/en/latest/api/pysindy.feature_library.html#module-pysindy.feature_library.polynomial_library

    Parameters
    ----------
        self.x_dim: The spatial dimension (int) of the data.
        self.poly_order: The order (int) of the polynomials in the data.
        self.include_constant: Iff True (bool), a constant term is included in
            the SINDy library.

    Attributes:
        self.x_dim: The spatial dimension (int) of the data.
        self.poly_order: The order (int) of the polynomials in the data.
        self.include_constant: Iff True (bool), a constant term is included in
            the SINDy library.
        self.library_size: The number (int) of terms in the library.
        self.feature_names: A list of the names (str) of the functions in the library.

    """
    def __init__(self, x_dim, poly_order=3, include_constant=True):
        self.x_dim = x_dim
        self.poly_order = poly_order
        self.include_constant = include_constant
        self.library_size = calculate_library_size(x_dim, poly_order, include_constant)
        self.feature_names = create_feature_names(x_dim, poly_order, include_constant)


    def transform(self, X, library=None):
        """Transforms X into a SINDy library.

        Creates a SINDy library out of X using the given settings.

        The code for this function was taken from:
        https://github.com/kpchamp/SindyAutoencoders/blob/master/src/sindy_utils.py

        Args:
            X: The data (torch.Tensor of shape (batch_size, x_dim)) to build a
                SINDy library with.
            library: A torch.tensor for the library (batch_size, library_size).
                The default is None. If None, constructs the library. If not None,
                then returns library without doing anything else.
            t: A torch.tensor (batch_size, ) for the corresponding timepoints.

        Returns:
            The SINDy library of X as a torch.Tensor of shape
            (batch_size x library_size).
        """
        if library is None:
            batch_size, device = X.size(0), X.device
            l = self.get_library_size()
            n = self.x_dim
            library = torch.ones((batch_size,l), device=device)
            index = 0
            
            if self.include_constant:
                index = 1

            for i in range(n):
                library[:,index] = X[:,i]
                index += 1

            if self.poly_order > 1:
                for i in range(n):
                    for j in range(i,n):
                        library[:,index] = X[:,i] * X[:,j]
                        index += 1

            if self.poly_order > 2:
                for i in range(n):
                    for j in range(i,n):
                        for k in range(j,n):
                            library[:,index] = X[:,i] * X[:,j] * X[:,k]
                            index += 1

            if self.poly_order > 3:
                for i in range(n):
                    for j in range(i,n):
                        for k in range(j,n):
                            for q in range(k,n):
                                library[:,index] = X[:,i] * X[:,j] * X[:,k] * X[:,q]
                                index += 1
                            
            if self.poly_order > 4:
                for i in range(n):
                    for j in range(i,n):
                        for k in range(j,n):
                            for q in range(k,n):
                                for r in range(q,n):
                                    library[:,index] = X[:,i] * X[:,j] * X[:,k] * X[:,q] * X[:,r]
                                    index += 1

        return library 

    def get_library_size(self,):
        """Gets the size of the SINDy library.

        Gets the number of terms in the SINDy library.

        Args:
            None

        Returns:
            The size (int) of the SINDy library.
        """
        return self.library_size 

    def get_feature_names(self):
        """Gets the names of the features in the SINDy library.

        Gets the names of each of the functions in the SINDy library.

        Args:
            None

        Returns:
            A list of the names (str) of the functions in the SINDy library.
        """
        return self.feature_names 

def calculate_library_size(n, poly_order, include_constant):
    """Calculates the size of the SINDy library.

    Calculates the number of terms in the SINDy library using the given
    parameters.

    The code for this function was taken from:
    https://github.com/kpchamp/SindyAutoencoders/blob/master/src/sindy_utils.py

    Args:
        n: The spatial dimenion (int) of the library.
        poly_order: The maximum degree of the polynomials to include in the
            the library. Includes integer polynomials from 1 up to and
            and including poly_order. Maximum value of poly_order is 5.
        include_constant: Iff True (boolean), includes a constant term in the
            library. The default is True.

    Returns:
        The number of terms (int) in the library.
    """
    l = 0
    for k in range(poly_order+1):
        l += int(binom(n+k-1,k))
    if not include_constant:
        l -= 1  
    return l

def create_feature_names(n, poly_order, include_constant):
    """Creates an equation SINDy library.

    Creates an equation SINDy library with the given settings. For n = 3, the
    result could be a list of the form:
        ["1", "x", "y", "z", "x^2", "xy", ...]
    The terms in the library should correspond to the terms returned by
    sindy_library, but represented as strings instead of the actual floats.

    Args:
        n: The spatial dimenion (int) of the library.
        poly_order: The maximum degree of the polynomials to include in the
            the library. Includes integer polynomials from 1 up to and
            and including poly_order. Maximum value of poly_order is 5.
        include_constant: Iff True (boolean), includes a constant term in the
            library. The default is True.

    Returns:
        The SINDy library of X as a torch.Tensor of shape
        (batch_size x library_dim).
    """
    l = calculate_library_size(n, poly_order, include_constant)
    str_lib = []
    if include_constant:
        index = 1
        str_lib = ['']
    
    X = ['x', 'y', 'z']
    if n > 3:
        X = ['x' + str(i + 1) for i in range(n)]
    
    for i in range(n):
        str_lib.append(X[i])
    
    if poly_order > 1:
        for i in range(n):
            for j in range(i,n):
                str_lib.append(X[i] + X[j])
    
    if poly_order > 2:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    str_lib.append(X[i] + X[j] + X[k])

    if poly_order > 3:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        str_lib.append(X[i] + X[j] + X[k] + X[q])
    
    if poly_order > 4:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        for r in range(q,n):
                            str_lib.append(X[i] + X[j] + X[k] + X[q] + X[r])
            
    return str_lib