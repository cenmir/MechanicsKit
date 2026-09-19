"""
Tests for the fplot function.
"""
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Check if SymPy is available
try:
    import sympy as sp
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

pytestmark = pytest.mark.skipif(not HAS_SYMPY, reason="SymPy not installed")

from mechanicskit import fplot


class TestFplotBasic:
    """Test basic fplot functionality."""

    def test_simple_function_auto_detect(self):
        """Test plotting with automatic parameter detection."""
        x = sp.Symbol('x')
        expr = sp.sin(x)

        fig, ax = plt.subplots()
        line = fplot(expr, ax=ax)

        assert line is not None
        # Check that line was added to axes
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_simple_function_explicit_param(self):
        """Test plotting with explicit parameter."""
        x = sp.Symbol('x')
        expr = sp.sin(x)

        fig, ax = plt.subplots()
        line = fplot(expr, x, ax=ax)

        assert line is not None
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_custom_range(self):
        """Test plotting with custom range."""
        x = sp.Symbol('x')
        expr = x**2

        fig, ax = plt.subplots()
        line = fplot(expr, range=(-10, 10), ax=ax)

        # Check that x-values span the range
        xdata = line.get_xdata()
        assert xdata[0] == pytest.approx(-10, abs=0.1)
        assert xdata[-1] == pytest.approx(10, abs=0.1)
        plt.close(fig)

    def test_custom_npoints(self):
        """Test plotting with custom number of points."""
        x = sp.Symbol('x')
        expr = x**2

        fig, ax = plt.subplots()
        line = fplot(expr, npoints=50, ax=ax)

        # Check that we have 50 points
        xdata = line.get_xdata()
        assert len(xdata) == 50
        plt.close(fig)

    def test_no_axes_provided(self):
        """Test that fplot creates axes if none provided."""
        x = sp.Symbol('x')
        expr = sp.sin(x)

        # Clear any existing figures
        plt.close('all')

        line = fplot(expr)

        assert line is not None
        # Should have created a figure
        assert len(plt.get_fignums()) > 0
        plt.close('all')

    def test_styling_kwargs(self):
        """Test that styling kwargs are passed to plot."""
        x = sp.Symbol('x')
        expr = sp.cos(x)

        fig, ax = plt.subplots()
        line = fplot(expr, ax=ax, color='red', linewidth=3, linestyle='--')

        assert line.get_color() == 'red'
        assert line.get_linewidth() == 3
        assert line.get_linestyle() == '--'
        plt.close(fig)


class TestFplotParametric:
    """Test parametric plotting functionality."""

    def test_parametric_auto_detect(self):
        """Test parametric plot with automatic parameter detection."""
        t = sp.Symbol('t')
        xt = sp.cos(t)
        yt = sp.sin(t)

        fig, ax = plt.subplots()
        line = fplot(xt, yt, ax=ax)

        assert line is not None
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_parametric_explicit_param(self):
        """Test parametric plot with explicit parameter."""
        s = sp.Symbol('s')
        xt = s * sp.cos(s)
        yt = s * sp.sin(s)

        fig, ax = plt.subplots()
        line = fplot(xt, yt, s, ax=ax)

        assert line is not None
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_parametric_custom_range(self):
        """Test parametric plot with custom range."""
        t = sp.Symbol('t')
        xt = sp.cos(t)
        yt = sp.sin(t)

        fig, ax = plt.subplots()
        line = fplot(xt, yt, range=(0, 2*np.pi), ax=ax, npoints=100)

        # For a circle, should get closed curve
        xdata = line.get_xdata()
        ydata = line.get_ydata()

        # First and last points should be close (full circle)
        assert xdata[0] == pytest.approx(xdata[-1], abs=0.1)
        assert ydata[0] == pytest.approx(ydata[-1], abs=0.1)
        plt.close(fig)

    def test_spiral(self):
        """Test plotting a spiral (parametric curve)."""
        t = sp.Symbol('t')
        xt = t * sp.cos(t)
        yt = t * sp.sin(t)

        fig, ax = plt.subplots()
        line = fplot(xt, yt, range=(0, 4*np.pi), ax=ax)

        assert line is not None
        # Check that spiral grows (radius increases)
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        radius = np.sqrt(xdata**2 + ydata**2)
        # Radius should be increasing on average
        assert radius[-1] > radius[0]
        plt.close(fig)


class TestFplotParameterInference:
    """Test automatic parameter inference."""

    def test_single_symbol(self):
        """Test inference with single free symbol."""
        x = sp.Symbol('x')
        expr = x**2 + 2*x + 1

        fig, ax = plt.subplots()
        line = fplot(expr, ax=ax)  # Should auto-detect x

        assert line is not None
        plt.close(fig)

    def test_prefer_x(self):
        """Test that 'x' is preferred when multiple symbols present."""
        x, a = sp.symbols('x a')
        expr = a * x**2  # Has both x and a

        # Should fail because we can't determine which is the parameter
        with pytest.raises(ValueError, match="multiple free symbols"):
            fig, ax = plt.subplots()
            fplot(expr, ax=ax)
            plt.close(fig)

    def test_prefer_t_for_parametric(self):
        """Test that 't' is preferred for parametric plots."""
        from mechanicskit.fplot import _infer_parameter_from_multiple
        t, a = sp.symbols('t a')
        xt = a * sp.cos(t)
        yt = a * sp.sin(t)

        assert _infer_parameter_from_multiple(xt, yt) == t

        # With a value for a, the curve plots against t
        fig, ax = plt.subplots()
        line = fplot(xt.subs(a, 2), yt.subs(a, 2), ax=ax)
        assert line is not None
        plt.close(fig)

    def test_parametric_leftover_symbol(self):
        """A symbol other than the parameter without a value is named in the error."""
        t, a = sp.symbols('t a')
        with pytest.raises(ValueError, match="a has no value"):
            fig, ax = plt.subplots()
            fplot(a * sp.cos(t), a * sp.sin(t), ax=ax)
        plt.close(fig)

    def test_no_free_symbols(self):
        """Test error when expression has no free symbols."""
        expr = sp.Integer(5)  # Constant

        with pytest.raises(ValueError, match="no free symbols"):
            fig, ax = plt.subplots()
            fplot(expr, ax=ax)
            plt.close(fig)


class TestFplotEdgeCases:
    """Test edge cases and error handling."""

    def test_no_arguments(self):
        """Test that error is raised with no arguments."""
        with pytest.raises(ValueError, match="at least one"):
            fplot()

    def test_complex_result_warning(self):
        """Test that complex results produce warning and plot real part."""
        x = sp.Symbol('x')
        # sqrt(x) would not do: numpy returns nan for negative floats, not a complex
        expr = sp.exp(sp.I * x)

        fig, ax = plt.subplots()
        with pytest.warns(UserWarning, match="complex values"):
            line = fplot(expr, range=(-2, 2), ax=ax)

        assert line is not None
        # Should have plotted something
        ydata = line.get_ydata()
        # All values should be real
        assert not np.any(np.isnan(ydata))
        plt.close(fig)

    def test_exponential(self):
        """Test plotting exponential function."""
        x = sp.Symbol('x')
        expr = sp.exp(-x**2)  # Gaussian

        fig, ax = plt.subplots()
        line = fplot(expr, range=(-3, 3), ax=ax)

        # Check that peak is near x=0
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        max_idx = np.argmax(ydata)
        assert xdata[max_idx] == pytest.approx(0, abs=0.5)
        plt.close(fig)

    def test_trigonometric_combo(self):
        """Test combination of trig functions."""
        x = sp.Symbol('x')
        expr = sp.sin(x) + sp.cos(2*x)

        fig, ax = plt.subplots()
        line = fplot(expr, ax=ax)

        assert line is not None
        ydata = line.get_ydata()
        # Check reasonable bounds
        assert np.all(ydata >= -2.1)  # min of sin + cos
        assert np.all(ydata <= 2.1)   # max of sin + cos
        plt.close(fig)


class TestFplotMultiplePlots:
    """Test multiple plots on same axes."""

    def test_multiple_functions(self):
        """Test plotting multiple functions on same axes."""
        x = sp.Symbol('x')

        fig, ax = plt.subplots()
        line1 = fplot(sp.sin(x), ax=ax, label='sin')
        line2 = fplot(sp.cos(x), ax=ax, label='cos')

        assert len(ax.lines) == 2
        assert line1.get_label() == 'sin'
        assert line2.get_label() == 'cos'
        plt.close(fig)

    def test_multiple_parametric(self):
        """Test plotting multiple parametric curves."""
        t = sp.Symbol('t')

        fig, ax = plt.subplots()

        # Circle
        fplot(sp.cos(t), sp.sin(t), range=(0, 2*np.pi), ax=ax, label='circle')

        # Ellipse
        fplot(2*sp.cos(t), sp.sin(t), range=(0, 2*np.pi), ax=ax, label='ellipse')

        assert len(ax.lines) == 2
        plt.close(fig)


class TestFplotRealWorldExamples:
    """Test real-world mathematical examples."""

    def test_polynomial(self):
        """Test polynomial function."""
        x = sp.Symbol('x')
        expr = x**3 - 3*x**2 + 2*x

        fig, ax = plt.subplots()
        line = fplot(expr, range=(-1, 3), ax=ax)

        assert line is not None
        plt.close(fig)

    def test_rational_function(self):
        """Test rational function."""
        x = sp.Symbol('x')
        expr = (x**2 - 1) / (x**2 + 1)

        fig, ax = plt.subplots()
        line = fplot(expr, range=(-5, 5), ax=ax)

        ydata = line.get_ydata()
        # Should approach 1 as x -> ±∞
        assert ydata[0] > 0.9  # x = -5
        assert ydata[-1] > 0.9  # x = 5
        plt.close(fig)

    def test_lissajous(self):
        """Test Lissajous curve (parametric)."""
        t = sp.Symbol('t')
        xt = sp.sin(3*t)
        yt = sp.sin(2*t)

        fig, ax = plt.subplots()
        line = fplot(xt, yt, range=(0, 2*np.pi), ax=ax)

        assert line is not None
        plt.close(fig)

    def test_rose_curve(self):
        """Test rose curve in Cartesian form (parametric)."""
        t = sp.Symbol('t')
        k = 5  # Number of petals
        xt = sp.cos(k*t) * sp.cos(t)
        yt = sp.cos(k*t) * sp.sin(t)

        fig, ax = plt.subplots()
        line = fplot(xt, yt, range=(0, 2*np.pi), ax=ax)

        assert line is not None
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
