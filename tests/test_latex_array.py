"""
Tests for MechanicsKit latex_array module
====================================

Tests the display_labeled_latex function and LatexArray class to ensure
correct LaTeX generation for various input types.
"""

import pytest
import numpy as np
from mechanicskit.latex_array import display_labeled_latex, LatexArray, la


class TestLatexArray:
    """Test the LatexArray class and la renderer."""

    def test_scalar(self):
        """Test scalar display."""
        arr = np.array(5)
        latex_obj = LatexArray(arr)
        latex_str = latex_obj._repr_latex_()
        assert "5" in latex_str
        assert "$$" in latex_str

    def test_1d_short_vector(self):
        """Test short vector (≤5 elements) displayed as column vector."""
        arr = np.array([1, 2, 3])
        latex_obj = LatexArray(arr)
        latex_str = latex_obj._repr_latex_()
        assert "begin{bmatrix}" in latex_str
        assert "\\\\" in latex_str  # Row separator

    def test_1d_medium_vector(self):
        """Test medium vector (6-8 elements) displayed as row vector."""
        arr = np.arange(6)
        latex_obj = LatexArray(arr)
        latex_str = latex_obj._repr_latex_()
        assert "^\\mathsf T" in latex_str  # Transpose notation

    def test_1d_long_vector(self):
        """Test long vector (>8 elements) displayed truncated."""
        arr = np.arange(20)
        latex_obj = LatexArray(arr)
        latex_str = latex_obj._repr_latex_()
        assert "\\cdots" in latex_str  # Ellipsis for truncation

    def test_2d_matrix(self):
        """Test 2D matrix display."""
        arr = np.array([[1, 2], [3, 4]])
        latex_obj = LatexArray(arr)
        latex_str = latex_obj._repr_latex_()
        assert "begin{bmatrix}" in latex_str
        assert "&" in latex_str  # Column separator
        assert "\\\\" in latex_str  # Row separator

    def test_pipe_syntax(self):
        """Test pipe operator syntax."""
        arr = np.array([1, 2, 3])
        latex_obj = arr | la
        assert isinstance(latex_obj, LatexArray)
        assert np.array_equal(latex_obj.array, arr)

    def test_function_syntax(self):
        """Test function call syntax."""
        arr = np.array([1, 2, 3])
        latex_obj = la(arr)
        assert isinstance(latex_obj, LatexArray)
        assert np.array_equal(latex_obj.array, arr)


class TestDisplayLabeledLatex:
    """Test the display_labeled_latex function."""

    def test_numpy_scalar(self):
        """Test labeled display of scalar."""
        # This would normally display in Jupyter, we just check it doesn't error
        from IPython.display import Latex
        import io
        import sys

        # Capture output to avoid printing during tests
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            display_labeled_latex("x = ", np.array(5), precision=2)
        except Exception as e:
            pytest.fail(f"display_labeled_latex raised {type(e).__name__}: {e}")
        finally:
            sys.stdout = sys.__stdout__

    def test_numpy_vector(self):
        """Test labeled display of vector."""
        try:
            display_labeled_latex("\\mathbf{v} = ", np.array([1.5, 2.5, 3.5]), precision=1)
        except Exception as e:
            pytest.fail(f"display_labeled_latex raised {type(e).__name__}: {e}")

    def test_numpy_matrix(self):
        """Test labeled display of matrix."""
        A = np.array([[1.11, 2.22], [3.33, 4.44]])
        try:
            display_labeled_latex("\\mathbf{A} = ", A, precision=2)
        except Exception as e:
            pytest.fail(f"display_labeled_latex raised {type(e).__name__}: {e}")

    def test_sympy_matrix(self):
        """Test labeled display of SymPy matrix."""
        pytest.importorskip("sympy")
        from sympy import symbols, cos, sin, Matrix

        theta = symbols('theta')
        L_T = Matrix([
            [cos(theta), -sin(theta)],
            [sin(theta), cos(theta)]
        ])

        try:
            display_labeled_latex("\\mathbf{L}^\\mathsf{T} = ", L_T)
        except Exception as e:
            pytest.fail(f"display_labeled_latex raised {type(e).__name__}: {e}")

    def test_arraystretch_parameter(self):
        """Test arrayStretch parameter."""
        A = np.array([[1, 2], [3, 4]])

        # Test with default value
        try:
            display_labeled_latex("\\mathbf{A} = ", A, arrayStretch=1.5)
        except Exception as e:
            pytest.fail(f"display_labeled_latex with arrayStretch raised {type(e).__name__}: {e}")

        # Test with custom value
        try:
            display_labeled_latex("\\mathbf{A} = ", A, arrayStretch=2.0)
        except Exception as e:
            pytest.fail(f"display_labeled_latex with custom arrayStretch raised {type(e).__name__}: {e}")

    def test_precision_parameter(self):
        """Test precision parameter for NumPy arrays."""
        A = np.array([[1.123456, 2.987654]])

        # We can't easily check the output without running in Jupyter,
        # but we can verify it doesn't crash
        try:
            display_labeled_latex("\\mathbf{A} = ", A, precision=4)
        except Exception as e:
            pytest.fail(f"display_labeled_latex with precision raised {type(e).__name__}: {e}")

    def test_show_shape_vector(self):
        """Test show_shape parameter with 1D vector."""
        v = np.array([1, 2, 3, 4])
        try:
            display_labeled_latex("v = ", v, show_shape=True)
        except Exception as e:
            pytest.fail(f"display_labeled_latex with show_shape (vector) raised {type(e).__name__}: {e}")

    def test_show_shape_matrix(self):
        """Test show_shape parameter with 2D matrix."""
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        try:
            display_labeled_latex("U = ", A, show_shape=True)
        except Exception as e:
            pytest.fail(f"display_labeled_latex with show_shape (matrix) raised {type(e).__name__}: {e}")

    def test_show_shape_sympy(self):
        """Test show_shape parameter with SymPy matrix."""
        pytest.importorskip("sympy")
        from sympy import symbols, Matrix

        a, b = symbols('a b')
        M = Matrix([[a, b], [b, a]])

        try:
            display_labeled_latex("M = ", M, show_shape=True)
        except Exception as e:
            pytest.fail(f"display_labeled_latex with show_shape (SymPy) raised {type(e).__name__}: {e}")


class TestComplexNumbers:
    """Test handling of complex numbers."""

    def test_complex_scalar(self):
        """Test complex scalar display."""
        arr = np.array(3 + 4j)
        latex_obj = LatexArray(arr)
        latex_str = latex_obj._repr_latex_()
        assert "3.00" in latex_str
        assert "4.00" in latex_str or "+4.00" in latex_str
        assert "j" in latex_str

    def test_complex_vector(self):
        """Test complex vector display."""
        arr = np.array([1 + 2j, 3 - 4j])
        latex_obj = LatexArray(arr)
        latex_str = latex_obj._repr_latex_()
        assert "j" in latex_str


class TestSymPyExpressions:
    """Test handling of SymPy symbolic expressions in arrays."""

    def test_sympy_scalar_with_sqrt(self):
        """Test SymPy scalar with sqrt renders properly."""
        pytest.importorskip("sympy")
        import sympy as sp

        expr = sp.sqrt(3) / 6
        arr = np.array(expr)
        latex_obj = LatexArray(arr)
        latex_str = latex_obj._repr_latex_()

        # Should contain LaTeX sqrt, not plain text "sqrt"
        assert "\\sqrt{3}" in latex_str or "sqrt{3}" in latex_str
        assert latex_str.count("sqrt") <= 1  # Should be LaTeX command, not text

    def test_sympy_vector(self):
        """Test vector of SymPy expressions."""
        pytest.importorskip("sympy")
        import sympy as sp

        x = sp.Symbol('x')
        arr = np.array([sp.Rational(1, 2), sp.sqrt(2), x**2])
        latex_obj = LatexArray(arr)
        latex_str = latex_obj._repr_latex_()

        # Check for LaTeX fractions and sqrt
        assert "\\frac{1}{2}" in latex_str or "frac{1}{2}" in latex_str
        assert "\\sqrt{2}" in latex_str or "sqrt{2}" in latex_str
        assert "x^{2}" in latex_str or "x^2" in latex_str

    def test_sympy_matrix_from_solve(self):
        """Test matrix of SymPy tuples from solve (user's use case)."""
        pytest.importorskip("sympy")
        import sympy as sp

        # Simulate user's Gaussian quadrature solve result
        w1, w2, x1, x2, x = sp.symbols('w1 w2 x1 x2 x')
        eqs = [w1*x1**i + w2*x2**i - sp.integrate(x**i, (x, 0, 1)) for i in range(4)]
        sol = sp.solve(eqs, [w1, w2, x1, x2])

        # Convert to array and render
        sol_array = np.array(sol)
        latex_obj = LatexArray(sol_array)
        latex_str = latex_obj._repr_latex_()

        # Should contain proper LaTeX rendering, not "sqrt(3)"
        assert "\\sqrt{3}" in latex_str or "sqrt{3}" in latex_str
        assert "\\frac" in latex_str or "frac" in latex_str
        # Should NOT contain plain text sqrt
        assert "sqrt(3)" not in latex_str

    def test_sympy_mixed_types(self):
        """Test array mixing SymPy expressions and numbers."""
        pytest.importorskip("sympy")
        import sympy as sp

        arr = np.array([[sp.Rational(1, 2), sp.sqrt(3)/6],
                        [0.5, 1.732]])
        latex_obj = LatexArray(arr)
        latex_str = latex_obj._repr_latex_()

        # SymPy expressions should use LaTeX
        assert "\\frac" in latex_str or "frac" in latex_str
        # Regular floats should be formatted
        assert "0.50" in latex_str or "1.73" in latex_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
