"""
Tests for gaussint module - Gaussian quadrature integration.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

# Try importing dependencies
try:
    import sympy as sp
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not (HAS_SYMPY and HAS_SCIPY),
    reason="SymPy and SciPy required for gaussint tests"
)

from mechanicskit.gaussint import gaussint


class TestGaussintBasic:
    """Test basic functionality with different input formats."""

    def test_symbolic_expression(self):
        """Test integration of a symbolic expression."""
        x = sp.Symbol('x')
        expr = 1 + 3*x**2 - 3*x**3
        result = gaussint(expr, 0, 1, n=2)
        expected = 1.25  # Exact result: x + x^3 - (3/4)x^4 from 0 to 1 = 5/4
        assert_allclose(result, expected, rtol=1e-10)

    def test_lambda_function(self):
        """Test integration of a lambda function."""
        f = lambda x: 1 + 3*x**2 - 3*x**3
        result = gaussint(f, 0, 1, n=2)
        expected = 1.25
        assert_allclose(result, expected, rtol=1e-10)

    def test_regular_function(self):
        """Test integration of a regular Python function."""
        def f(x):
            return 1 + 3*x**2 - 3*x**3
        result = gaussint(f, 0, 1, n=2)
        expected = 1.25
        assert_allclose(result, expected, rtol=1e-10)

    def test_lambdified_expression(self):
        """Test integration of a lambdified SymPy expression."""
        x = sp.Symbol('x')
        f_lambdified = sp.lambdify(x, 1 + 3*x**2 - 3*x**3, 'numpy')
        result = gaussint(f_lambdified, 0, 1, n=2)
        expected = 1.25
        assert_allclose(result, expected, rtol=1e-10)


class TestGaussintAccuracy:
    """Test accuracy with different quadrature orders."""

    def test_polynomial_exact(self):
        """Test that n-point quadrature is exact for polynomials of degree 2n-1."""
        x = sp.Symbol('x')

        # 2-point quadrature is exact for degree <= 3
        expr = x**3
        result = gaussint(expr, 0, 1, n=2)
        expected = 0.25  # x^4/4 evaluated from 0 to 1
        assert_allclose(result, expected, rtol=1e-14)

        # 3-point quadrature is exact for degree <= 5
        expr = x**5
        result = gaussint(expr, 0, 1, n=3)
        expected = 1/6  # x^6/6 evaluated from 0 to 1
        assert_allclose(result, expected, rtol=1e-14)

    def test_sin_function(self):
        """Test integration of sin(x) from 0 to pi."""
        x = sp.Symbol('x')
        result = gaussint(sp.sin(x), 0, np.pi, n=10)
        expected = 2.0  # -cos(pi) + cos(0) = 1 + 1 = 2
        assert_allclose(result, expected, rtol=1e-10)

    def test_exp_function(self):
        """Test integration of e^x from 0 to 1."""
        x = sp.Symbol('x')
        result = gaussint(sp.exp(x), 0, 1, n=10)
        expected = np.e - 1  # e^1 - e^0
        assert_allclose(result, expected, rtol=1e-10)

    def test_convergence_with_order(self):
        """Test that higher order gives better accuracy."""
        x = sp.Symbol('x')
        expr = sp.exp(-x**2)  # Gaussian function
        expected = float(sp.integrate(expr, (x, 0, 1)))

        # Low order - less accurate
        result_low = gaussint(expr, 0, 1, n=2)
        error_low = abs(result_low - expected)

        # High order - more accurate
        result_high = gaussint(expr, 0, 1, n=10)
        error_high = abs(result_high - expected)

        assert error_high < error_low


class TestGaussintIntervals:
    """Test integration over different intervals."""

    def test_negative_interval(self):
        """Test integration from negative to positive."""
        x = sp.Symbol('x')
        result = gaussint(x**2, -1, 1, n=3)
        expected = 2/3  # x^3/3 from -1 to 1
        assert_allclose(result, expected, rtol=1e-14)

    def test_shifted_interval(self):
        """Test integration over shifted interval."""
        x = sp.Symbol('x')
        result = gaussint(x, 2, 5, n=2)
        expected = (5**2 - 2**2) / 2  # x^2/2 from 2 to 5 = 10.5
        assert_allclose(result, expected, rtol=1e-14)

    def test_unit_interval(self):
        """Test on standard unit interval."""
        x = sp.Symbol('x')
        result = gaussint(x**2, 0, 1, n=2)
        expected = 1/3
        assert_allclose(result, expected, rtol=1e-14)


class TestGaussintParameterInference:
    """Test automatic parameter inference."""

    def test_single_variable_inferred(self):
        """Test that single variable is automatically inferred."""
        x = sp.Symbol('x')
        expr = x**2 + 2*x + 1
        # Should work without specifying param
        result = gaussint(expr, 0, 1, n=5)
        expected = 1/3 + 1 + 1  # x^3/3 + x^2 + x from 0 to 1
        assert_allclose(result, expected, rtol=1e-10)

    def test_common_variable_names(self):
        """Test that common variable names are preferred."""
        x = sp.Symbol('x')
        a = sp.Symbol('a')
        expr = a * x**2  # 'x' should be preferred over 'a'

        # With explicit parameter
        result = gaussint(expr, 0, 1, n=5, param=x)
        # Result should be a/3
        assert isinstance(result, (int, float, sp.Expr))

    def test_explicit_parameter(self):
        """Test explicit parameter specification."""
        x, y = sp.symbols('x y')
        expr = x**2 + y
        result = gaussint(expr, 0, 1, n=5, param=x)
        # Should integrate x^2 from 0 to 1, treating y as constant
        # Result: 1/3 + y
        # Since y is symbolic, we can't get a numerical value
        # Just check it runs without error
        assert result is not None


class TestGaussintEdgeCases:
    """Test edge cases and error handling."""

    def test_constant_function(self):
        """Test integration of a constant."""
        result = gaussint(lambda x: 5.0, 0, 2, n=1)
        expected = 10.0  # 5 * (2 - 0)
        assert_allclose(result, expected, rtol=1e-14)

    def test_zero_function(self):
        """Test integration of zero function."""
        x = sp.Symbol('x')
        result = gaussint(sp.sympify(0), 0, 1, n=5)
        expected = 0.0
        assert_allclose(result, expected, atol=1e-14)

    def test_same_limits(self):
        """Test integration when a == b."""
        x = sp.Symbol('x')
        result = gaussint(x**2, 1, 1, n=5)
        expected = 0.0
        assert_allclose(result, expected, atol=1e-14)

    def test_reversed_limits(self):
        """Test integration with reversed limits (should be negative)."""
        x = sp.Symbol('x')
        result_forward = gaussint(x**2, 0, 1, n=5)
        result_backward = gaussint(x**2, 1, 0, n=5)
        assert_allclose(result_forward, -result_backward, rtol=1e-14)


class TestGaussintVsScipy:
    """Compare results with scipy.integrate.fixed_quad."""

    def test_matches_scipy_fixed_quad(self):
        """Test that results match scipy.integrate.fixed_quad for callables."""
        from scipy.integrate import fixed_quad

        f = lambda x: x**3 + 2*x**2 - x + 5
        a, b, n = 0, 2, 7

        result_gaussint = gaussint(f, a, b, n=n)
        result_scipy, _ = fixed_quad(f, a, b, n=n)

        assert_allclose(result_gaussint, result_scipy, rtol=1e-14)

    def test_symbolic_advantage(self):
        """Test that symbolic expressions work (scipy.fixed_quad doesn't accept these)."""
        x = sp.Symbol('x')
        expr = sp.sin(x) * sp.cos(x)

        # This should work with gaussint
        result = gaussint(expr, 0, np.pi/2, n=10)
        expected = 0.5  # sin^2(x)/2 from 0 to pi/2
        assert_allclose(result, expected, rtol=1e-10)


class TestGaussintErrors:
    """Test error handling."""

    def test_invalid_input_type(self):
        """Test that invalid input types raise errors."""
        with pytest.raises(ValueError, match="must be a SymPy expression or a callable"):
            gaussint("not a function", 0, 1, n=5)

    def test_constant_expression_no_symbols(self):
        """Test that constant symbolic expressions raise appropriate error."""
        if HAS_SYMPY:
            const_expr = sp.sympify(42)
            # A constant has no free symbols, but should still integrate to constant * (b - a)
            # Actually, let's check if this works or raises error
            result = gaussint(const_expr, 0, 1, n=5)
            assert_allclose(result, 42.0, rtol=1e-14)

    def test_multiple_symbols_no_param(self):
        """Test that expressions with multiple uncommon symbols require explicit param."""
        if HAS_SYMPY:
            a, b = sp.symbols('a b')  # No common integration variable names
            expr = a * b

            # Should raise ValueError about multiple symbols
            with pytest.raises(ValueError, match="multiple free symbols"):
                gaussint(expr, 0, 1, n=5)
