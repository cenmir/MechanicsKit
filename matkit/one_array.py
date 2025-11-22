"""
OneArray - 1-Based Indexing Wrapper
====================================

A wrapper around NumPy arrays that provides 1-based indexing for FEM results.
This eliminates the need for manual index translation (e.g., `N[iel-1]`) when
working with element/node-based result arrays.

Philosophy
----------
FEM results are naturally indexed by element/node numbers (1-based), not
array indices (0-based). This wrapper makes accessing results feel natural:

```python
N = OneArray([100, 200, 300])  # Element forces
print(N[1])  # Force in element 1 = 100 (not N[0]!)
print(N[3])  # Force in element 3 = 300 (not N[2]!)
```

This pairs perfectly with Mesh iteration:
```python
for iel in mesh.elements():
    force = N[iel]  # No N[iel-1] mental gymnastics!
```

Author: Based on the 1-Indexing Manifesto pedagogical framework
License: MIT
"""

import numpy as np
from typing import Union, List


class OneArray:
    """
    Array wrapper with 1-based indexing for FEM results.

    This class wraps a NumPy array and translates 1-based indices (element/node
    numbers) to 0-based array indices automatically.

    Examples
    --------
    >>> # Create from list (element forces for elements 1, 2, 3, 4, 5)
    >>> N = OneArray([6052.76, -5582.25, -7274.51, 6380.16, -9912.07])
    >>>
    >>> # Access element 1 force (naturally!)
    >>> N[1]
    6052.76
    >>>
    >>> # Access element 3 force
    >>> N[3]
    -7274.51
    >>>
    >>> # Works with mesh iteration seamlessly
    >>> for iel in mesh.elements():
    ...     print(f"Element {iel}: Force = {N[iel]:.2f} N")
    Element 1: Force = 6052.76 N
    Element 2: Force = -5582.25 N
    ...
    >>>
    >>> # Set values using 1-based indices
    >>> N[1] = 7000.0
    >>> N[1]
    7000.0
    >>>
    >>> # Access multiple elements
    >>> N[[1, 3, 5]]
    array([7000.0, -7274.51, -9912.07])
    >>>
    >>> # Still have access to underlying NumPy array
    >>> N.data
    array([7000.0, -5582.25, -7274.51, 6380.16, -9912.07])
    """

    def __init__(self, data):
        """
        Initialize OneArray from array-like data.

        Parameters
        ----------
        data : array-like
            The data to wrap. Will be converted to NumPy array.
            Indexed as elements/nodes 1, 2, 3, ... (NOT 0, 1, 2, ...)

        Example
        -------
        >>> # Element forces for 5 elements
        >>> N = OneArray([100, 200, 300, 400, 500])
        >>> N[1]  # Element 1 force
        100
        >>> N[5]  # Element 5 force
        500
        """
        self.data = np.array(data)

    def __getitem__(self, index: Union[int, List[int], np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get item(s) using 1-based indexing.

        Parameters
        ----------
        index : int or array-like
            Element/node number(s) (1-based)

        Returns
        -------
        value : scalar or array
            Value(s) at the specified element/node number(s)

        Raises
        ------
        IndexError
            If index is out of valid range [1, len]

        Examples
        --------
        >>> N = OneArray([10, 20, 30])
        >>> N[1]
        10
        >>> N[3]
        30
        >>> N[[1, 3]]
        array([10, 30])
        """
        if np.isscalar(index):
            self._validate_index(index)
            return self.data[index - 1]  # THE TRANSLATION
        else:
            indices = np.atleast_1d(index)
            for idx in indices:
                self._validate_index(idx)
            return self.data[indices - 1]  # Vectorized translation

    def __setitem__(self, index: Union[int, List[int], np.ndarray], value):
        """
        Set item(s) using 1-based indexing.

        Parameters
        ----------
        index : int or array-like
            Element/node number(s) (1-based)
        value : scalar or array-like
            Value(s) to set

        Examples
        --------
        >>> N = OneArray([10, 20, 30])
        >>> N[1] = 15
        >>> N[1]
        15
        >>> N[[1, 3]] = [100, 300]
        >>> N.data
        array([100, 20, 300])
        """
        if np.isscalar(index):
            self._validate_index(index)
            self.data[index - 1] = value
        else:
            indices = np.atleast_1d(index)
            for idx in indices:
                self._validate_index(idx)
            self.data[indices - 1] = value

    def __len__(self) -> int:
        """Return number of elements/nodes."""
        return len(self.data)

    def __repr__(self) -> str:
        """String representation."""
        return f"OneArray({self.data.tolist()})"

    def __str__(self) -> str:
        """Pretty string representation."""
        if len(self.data) <= 6:
            return f"OneArray({self.data})"
        else:
            return f"OneArray([{self.data[0]}, {self.data[1]}, ..., {self.data[-1]}])"

    def _validate_index(self, index: int):
        """Validate that index is in valid 1-based range."""
        if not isinstance(index, (int, np.integer)):
            raise TypeError(f"Index must be integer, got {type(index)}")
        if not (1 <= index <= len(self.data)):
            raise IndexError(
                f"Index {index} out of range [1, {len(self.data)}]. "
                f"Remember: OneArray uses 1-based indexing!"
            )

    # NumPy compatibility - allow arithmetic operations
    def __add__(self, other):
        """Add OneArray or scalar."""
        if isinstance(other, OneArray):
            return OneArray(self.data + other.data)
        return OneArray(self.data + other)

    def __sub__(self, other):
        """Subtract OneArray or scalar."""
        if isinstance(other, OneArray):
            return OneArray(self.data - other.data)
        return OneArray(self.data - other)

    def __mul__(self, other):
        """Multiply by OneArray or scalar."""
        if isinstance(other, OneArray):
            return OneArray(self.data * other.data)
        return OneArray(self.data * other)

    def __truediv__(self, other):
        """Divide by OneArray or scalar."""
        if isinstance(other, OneArray):
            return OneArray(self.data / other.data)
        return OneArray(self.data / other)

    def __neg__(self):
        """Negate."""
        return OneArray(-self.data)

    def __abs__(self):
        """Absolute value."""
        return OneArray(np.abs(self.data))

    # Right-hand operations
    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return OneArray(other - self.data)

    def __rtruediv__(self, other):
        return OneArray(other / self.data)

    # Comparison operations
    def __eq__(self, other):
        """Element-wise equality."""
        if isinstance(other, OneArray):
            return np.array_equal(self.data, other.data)
        return False

    def max(self):
        """Maximum value."""
        return np.max(self.data)

    def min(self):
        """Minimum value."""
        return np.min(self.data)

    def sum(self):
        """Sum of all values."""
        return np.sum(self.data)

    def mean(self):
        """Mean of all values."""
        return np.mean(self.data)

    def std(self):
        """Standard deviation."""
        return np.std(self.data)

    @property
    def shape(self):
        """Shape of underlying array."""
        return self.data.shape

    @property
    def dtype(self):
        """Data type of underlying array."""
        return self.data.dtype

    def copy(self):
        """Return a copy."""
        return OneArray(self.data.copy())

    def to_numpy(self) -> np.ndarray:
        """
        Convert to regular NumPy array (0-based).

        Returns
        -------
        array : ndarray
            The underlying NumPy array (0-based indexing)

        Note
        ----
        Use this when you need to interface with standard NumPy/SciPy
        functions that expect 0-based indexing.

        Example
        -------
        >>> N = OneArray([100, 200, 300])
        >>> N[1]  # 1-based
        100
        >>> arr = N.to_numpy()
        >>> arr[0]  # 0-based
        100
        """
        return self.data

    def _repr_latex_(self):
        """
        LaTeX representation for marimo and Jupyter notebooks.

        This enables pretty printing with the pipe operator:
        >>> N = OneArray([100, 200, 300])
        >>> N | la  # Renders as LaTeX in marimo and Jupyter

        Returns
        -------
        latex_str : str
            LaTeX representation wrapped in $$ ... $$
        """
        from .latex_array import LatexArray
        return LatexArray(self.data)._repr_latex_()
