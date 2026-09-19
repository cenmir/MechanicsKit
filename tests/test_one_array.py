"""
Tests for OneArray - 1-based indexing in one and two dimensions.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from mechanicskit import OneArray


@pytest.fixture
def K():
    """A 3x3 matrix with distinct entries, so a wrong index shows."""
    return OneArray(np.arange(1, 10).reshape(3, 3))


class TestOneDimensional:
    """Behaviour that must not change from earlier versions."""

    def test_scalar_index(self):
        assert OneArray([10, 20, 30])[2] == 20

    def test_list_index(self):
        assert_array_equal(OneArray([10, 20, 30])[[1, 3]], [10, 30])

    def test_array_index(self):
        assert_array_equal(OneArray([10, 20, 30])[np.array([1, 3])], [10, 30])

    def test_setitem_list(self):
        N = OneArray([10, 20, 30])
        N[[1, 3]] = [100, 300]
        assert_array_equal(N.data, [100, 20, 300])

    def test_zero_rejected(self):
        with pytest.raises(IndexError, match="1-based"):
            OneArray([10, 20, 30])[0]

    def test_past_end_rejected(self):
        with pytest.raises(IndexError):
            OneArray([10, 20, 30])[4]


class TestRows:
    """A single index on a 2-D array still selects rows (nodal fields)."""

    def test_single_row(self, K):
        assert_array_equal(K[2], [4, 5, 6])

    def test_row_list(self, K):
        assert_array_equal(K[[1, 3]], [[1, 2, 3], [7, 8, 9]])


class TestTwoDimensional:

    def test_element(self, K):
        assert K[2, 1] == 4

    def test_submatrix(self, K):
        assert_array_equal(K[[1, 3], [1, 3]], [[1, 3], [7, 9]])

    def test_submatrix_mixed_with_scalar(self, K):
        assert_array_equal(K[2, [1, 3]], [4, 6])

    def test_colon(self, K):
        assert_array_equal(K[:, 2], [2, 5, 8])
        assert_array_equal(K[3, :], [7, 8, 9])

    def test_assembly(self):
        """Assemble two rod elements, as in the textbook."""
        k_rod = np.array([[1.0, -1.0], [-1.0, 1.0]])
        elements = OneArray([[1, 2], [2, 3]])
        k = OneArray([2.0, 3.0])
        S = OneArray(np.zeros((3, 3)))
        for e, (i, j) in enumerate(elements, start=1):
            S[[i, j], [i, j]] += k[e] * k_rod
        assert_array_equal(S.data, [[2, -2, 0], [-2, 5, -3], [0, -3, 3]])

    def test_wrong_number_of_indices(self, K):
        with pytest.raises(IndexError, match="3 indices"):
            K[1, 1, 1]

    def test_out_of_range_names_axis(self, K):
        with pytest.raises(IndexError, match="axis 1"):
            K[1, 4]


class TestSlices:

    def test_slice_rejected(self):
        with pytest.raises(IndexError, match="slices"):
            OneArray([10, 20, 30])[1:2]


class TestIteration:

    def test_list(self):
        assert list(OneArray([10, 20, 30])) == [10, 20, 30]

    def test_unpack_rows(self):
        assert [(i, j) for i, j in OneArray([[1, 2], [2, 3]])] == [(1, 2), (2, 3)]

    def test_asarray(self):
        assert_array_equal(np.asarray(OneArray([1.0, 2.0])), [1.0, 2.0])


class TestMatmul:

    def test_matrix_vector(self):
        f = OneArray(np.eye(2) * 2) @ OneArray([1.0, 2.0])
        assert isinstance(f, OneArray)
        assert_array_equal(f.data, [2.0, 4.0])

    def test_with_numpy_on_right(self):
        f = OneArray(np.eye(2)) @ np.array([1.0, 2.0])
        assert_array_equal(f.data, [1.0, 2.0])

    def test_with_numpy_on_left(self):
        """NumPy converts the OneArray itself, so the result is a plain ndarray."""
        f = np.eye(2) @ OneArray([1.0, 2.0])
        assert_array_equal(f, [1.0, 2.0])

    def test_dot_product_is_a_number(self):
        d = OneArray([1.0, 2.0]) @ OneArray([1.0, 2.0])
        assert not isinstance(d, OneArray)
        assert d == 5.0
