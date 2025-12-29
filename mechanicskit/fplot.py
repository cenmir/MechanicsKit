"""
MATLAB-style fplot for SymPy symbolic functions.

This module provides fplot(), a function that plots symbolic expressions
similar to MATLAB's fplot, with automatic parameter detection and support
for both regular and parametric curves.
"""

import numpy as np
import matplotlib.pyplot as plt
try:
    import sympy as sp
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False


def fplot(*args, range=(-5, 5), ax=None, npoints=100,
         title=None, xlabel=None, ylabel=None, **kwargs):
    """
    Plot symbolic function(s) over a range, similar to MATLAB's fplot.

    This function automatically detects the independent variable from the
    symbolic expression and creates a plot. It supports both regular 2D
    plots and parametric curves.

    Parameters
    ----------
    *args : sympy expression(s) or tuple
        Positional arguments specifying what to plot:

        For 2D plots:
            - fplot(f): Plot f(x) with auto-detected parameter, default range
            - fplot(f, param): Plot f with explicit parameter
            - fplot(f, (param, min, max)): SymPy-style with parameter and range

        For parametric plots:
            - fplot(xt, yt): Plot parametric curve with auto-detected parameter
            - fplot(xt, yt, param): Plot parametric curve with explicit parameter
            - fplot(xt, yt, (param, min, max)): SymPy-style with parameter and range

    range : tuple of (float, float), optional
        Range [start, end] for the independent variable.
        Default: (-5, 5)
        Note: Overridden if (param, min, max) tuple is provided in args.

    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses plt.gca() to get current axes.

    npoints : int, optional
        Number of points to evaluate the function at.
        Default: 100

    title : str, optional
        Title for the plot. Sets ax.set_title(title).

    xlabel : str, optional
        Label for x-axis. Sets ax.set_xlabel(xlabel).

    ylabel : str, optional
        Label for y-axis. Sets ax.set_ylabel(ylabel).

    **kwargs
        Additional keyword arguments passed to ax.plot().
        Common options: 'color', 'linewidth', 'linestyle', 'label', etc.

    Returns
    -------
    line : matplotlib.lines.Line2D or list of Line2D
        The Line2D object(s) representing the plotted data.

    Raises
    ------
    ImportError
        If SymPy is not installed.
    ValueError
        If the symbolic expression has no free symbols or if arguments are invalid.

    Examples
    --------
    Basic usage with automatic parameter detection:

    >>> import sympy as sp
    >>> import matplotlib.pyplot as plt
    >>> from mechanicskit import fplot
    >>>
    >>> x = sp.Symbol('x')
    >>> fplot(sp.sin(x))  # Plots sin(x) over [-5, 5]
    >>> plt.show()

    Custom range with keyword:

    >>> fplot(sp.sin(x), range=(-10, 10))
    >>> plt.show()

    SymPy-style syntax with tuple:

    >>> t = sp.Symbol('t')
    >>> T = 20 + 80*sp.exp(-0.05*t)
    >>> fplot(T, (t, 0, 60))  # Plots T(t) from 0 to 60
    >>> plt.show()

    With title and labels:

    >>> fplot(T, (t, 0, 60),
    ...       title="Cooling of a hot object",
    ...       xlabel="Time (minutes)",
    ...       ylabel="Temperature (°C)")
    >>> plt.show()

    Multiple plots:

    >>> fig, ax = plt.subplots()
    >>> fplot(sp.sin(x), ax=ax, label='sin(x)')
    >>> fplot(sp.cos(x), ax=ax, label='cos(x)')
    >>> ax.legend()
    >>> plt.show()

    Parametric curves with SymPy syntax:

    >>> t = sp.Symbol('t')
    >>> fplot(sp.cos(t), sp.sin(t), (t, 0, 2*sp.pi))
    >>> plt.axis('equal')
    >>> plt.show()

    With styling:

    >>> fplot(sp.exp(-x**2), range=(-3, 3),
    ...       color='red', linewidth=2, linestyle='--', label='Gaussian')
    >>> plt.legend()
    >>> plt.show()

    See Also
    --------
    matplotlib.pyplot.plot : Standard matplotlib plotting
    sympy.plotting.plot : SymPy's plotting function

    Notes
    -----
    The function uses SymPy's lambdify to convert symbolic expressions
    to numerical functions for efficient evaluation.

    When plotting parametric curves, the range applies to the parameter
    (typically 't' or 's'), not to x or y coordinates.

    If your expression has multiple free symbols, you must specify which
    parameter to use explicitly.

    The SymPy-style tuple syntax (param, min, max) is supported for
    compatibility with sympy.plot(), making it easy to switch between
    the two plotting functions.

    References
    ----------
    Based on MATLAB's fplot:
    https://www.mathworks.com/help/matlab/ref/fplot.html
    """
    if not HAS_SYMPY:
        raise ImportError(
            "SymPy is required for fplot. Install with: pip install sympy"
        )

    # Get or create axes
    if ax is None:
        ax = plt.gca()

    # Parse arguments to determine plot type
    if len(args) == 0:
        raise ValueError("fplot requires at least one symbolic expression")

    # Parse arguments to extract range specification if present
    # This handles SymPy-style syntax: (param, min, max)
    processed_args, plot_range = _parse_args_and_range(args, range)

    # Determine if this is a parametric plot
    # Logic: If we have 2 or 3 args, check if they could be (xt, yt, [param])
    is_parametric = False
    if len(processed_args) >= 2:
        # Check if second arg is NOT a Symbol (if it's an expression, likely parametric)
        if not isinstance(processed_args[1], sp.Symbol):
            is_parametric = True
        # Or if we have 3 args and third is a Symbol
        elif len(processed_args) == 3 and isinstance(processed_args[2], sp.Symbol):
            is_parametric = True

    # Plot
    if is_parametric:
        # Parametric plot: fplot(xt, yt, [param])
        line = _fplot_parametric(processed_args, plot_range, ax, npoints, **kwargs)
    else:
        # Regular 2D plot: fplot(f, [param])
        line = _fplot_2d(processed_args, plot_range, ax, npoints, **kwargs)

    # Set title and labels if provided
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    return line


def _parse_args_and_range(args, default_range):
    """
    Parse arguments to extract range specification from SymPy-style tuples.

    Handles syntax like (param, min, max) and extracts the range from it.

    Parameters
    ----------
    args : tuple
        Original arguments passed to fplot
    default_range : tuple
        Default (min, max) range if none specified

    Returns
    -------
    processed_args : tuple
        Arguments with range tuple replaced by just the parameter
    plot_range : tuple
        The (min, max) range to use
    """
    processed_args = []
    plot_range = default_range

    for arg in args:
        # Check if this is a range specification tuple
        if isinstance(arg, tuple) and len(arg) == 3:
            # Check if first element is a Symbol and next two are numbers
            if isinstance(arg[0], sp.Symbol):
                try:
                    # Try to convert to float to verify they're numbers
                    min_val = float(arg[1])
                    max_val = float(arg[2])
                    # This is a range specification
                    processed_args.append(arg[0])  # Add just the Symbol
                    plot_range = (min_val, max_val)  # Update range
                    continue
                except (TypeError, ValueError):
                    pass
                # Not a range specification, keep as is
        processed_args.append(arg)

    return tuple(processed_args), plot_range


def _fplot_2d(args, plot_range, ax, npoints, **kwargs):
    """
    Internal function to handle 2D (non-parametric) plots.

    Parameters
    ----------
    args : tuple
        Either (expr,) or (expr, param)
    plot_range : tuple
        (start, end) for the parameter
    ax : matplotlib axes
        Axes to plot on
    npoints : int
        Number of evaluation points
    **kwargs
        Passed to ax.plot()

    Returns
    -------
    Line2D object
    """
    if len(args) == 1:
        # fplot(f) - auto-detect parameter
        expr = args[0]
        param = _infer_parameter(expr)
    elif len(args) == 2:
        # fplot(f, param) - explicit parameter
        expr = args[0]
        param = args[1]
        if not isinstance(param, sp.Symbol):
            raise ValueError(
                f"Second argument must be a SymPy Symbol, got {type(param)}"
            )
    else:
        raise ValueError(
            f"Too many arguments for 2D plot: expected 1-2, got {len(args)}"
        )

    # Convert to numerical function
    num_func = sp.lambdify(param, expr, 'numpy')

    # Generate points
    x_vals = np.linspace(plot_range[0], plot_range[1], npoints)

    # Evaluate function (handle complex results)
    y_vals = num_func(x_vals)

    # If result is complex, take real part and warn if imaginary part is significant
    if np.iscomplexobj(y_vals):
        if np.max(np.abs(np.imag(y_vals))) > 1e-10:
            import warnings
            warnings.warn(
                "Function produces complex values. Plotting real part only.",
                UserWarning
            )
        y_vals = np.real(y_vals)

    # Plot
    line = ax.plot(x_vals, y_vals, **kwargs)

    return line[0]


def _fplot_parametric(args, plot_range, ax, npoints, **kwargs):
    """
    Internal function to handle parametric plots.

    Parameters
    ----------
    args : tuple
        Either (xt, yt) or (xt, yt, param)
    plot_range : tuple
        (start, end) for the parameter
    ax : matplotlib axes
        Axes to plot on
    npoints : int
        Number of evaluation points
    **kwargs
        Passed to ax.plot()

    Returns
    -------
    Line2D object
    """
    if len(args) == 2:
        # fplot(xt, yt) - auto-detect parameter
        xt, yt = args
        # Try to infer parameter from both expressions
        param = _infer_parameter_from_multiple(xt, yt)
    elif len(args) == 3:
        # fplot(xt, yt, param) - explicit parameter
        xt, yt, param = args
        if not isinstance(param, sp.Symbol):
            raise ValueError(
                f"Third argument must be a SymPy Symbol, got {type(param)}"
            )
    else:
        raise ValueError(
            f"Invalid number of arguments for parametric plot: got {len(args)}"
        )

    # Convert to numerical functions
    num_func_x = sp.lambdify(param, xt, 'numpy')
    num_func_y = sp.lambdify(param, yt, 'numpy')

    # Generate parameter values
    t_vals = np.linspace(plot_range[0], plot_range[1], npoints)

    # Evaluate functions
    x_vals = num_func_x(t_vals)
    y_vals = num_func_y(t_vals)

    # Handle complex results
    if np.iscomplexobj(x_vals):
        x_vals = np.real(x_vals)
    if np.iscomplexobj(y_vals):
        y_vals = np.real(y_vals)

    # Plot
    line = ax.plot(x_vals, y_vals, **kwargs)

    return line[0]


def _infer_parameter(expr):
    """
    Infer the parameter (independent variable) from a symbolic expression.

    Parameters
    ----------
    expr : sympy expression
        The symbolic expression

    Returns
    -------
    sympy.Symbol
        The inferred parameter

    Raises
    ------
    ValueError
        If no parameter can be inferred or if there are multiple candidates
    """
    # Get free symbols from the expression
    free_syms = expr.free_symbols

    if len(free_syms) == 0:
        raise ValueError(
            "Cannot infer parameter: expression has no free symbols. "
            "The expression might be a constant."
        )
    elif len(free_syms) == 1:
        return list(free_syms)[0]
    else:
        # Multiple symbols - try common conventions
        sym_names = [str(s) for s in free_syms]

        # Check for common variable names in order of preference
        for common_name in ['x', 't', 's', 'u', 'v', 'theta']:
            if common_name in sym_names:
                idx = sym_names.index(common_name)
                return list(free_syms)[idx]

        # No common name found - raise error
        raise ValueError(
            f"Cannot infer parameter: expression has multiple free symbols {free_syms}. "
            f"Please specify the parameter explicitly."
        )


def _infer_parameter_from_multiple(*exprs):
    """
    Infer a common parameter from multiple expressions.

    Parameters
    ----------
    *exprs : sympy expressions
        The symbolic expressions

    Returns
    -------
    sympy.Symbol
        The common parameter

    Raises
    ------
    ValueError
        If no common parameter can be inferred
    """
    # Get intersection of all free symbols
    if len(exprs) == 0:
        raise ValueError("No expressions provided")

    common_syms = exprs[0].free_symbols
    for expr in exprs[1:]:
        common_syms = common_syms.intersection(expr.free_symbols)

    if len(common_syms) == 0:
        raise ValueError(
            "Cannot infer parameter: expressions have no common free symbols. "
            "Please specify the parameter explicitly."
        )
    elif len(common_syms) == 1:
        return list(common_syms)[0]
    else:
        # Multiple common symbols - try common conventions
        sym_names = [str(s) for s in common_syms]

        # For parametric curves, prefer 't', 's', or 'theta'
        for common_name in ['t', 's', 'theta', 'u', 'v', 'x']:
            if common_name in sym_names:
                idx = sym_names.index(common_name)
                return list(common_syms)[idx]

        # No common name found - raise error
        raise ValueError(
            f"Cannot infer parameter: expressions have multiple common symbols {common_syms}. "
            f"Please specify the parameter explicitly."
        )
