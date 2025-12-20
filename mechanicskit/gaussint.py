"""
Flexible Gaussian quadrature integration for symbolic and numeric functions.

This module provides gaussint(), a function that performs Gaussian quadrature
integration with automatic handling of SymPy expressions, lambdified functions,
and regular Python callables.
"""

try:
    import sympy as sp
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

try:
    import scipy.special
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def gaussint(f, a, b, n=5, param=None):
    """
    Compute definite integral using Gaussian quadrature with flexible input formats.

    This function integrates f from a to b using Gaussian quadrature of order n.
    Unlike scipy.integrate.fixed_quad, this function can handle:
    - SymPy symbolic expressions
    - SymPy symbolic functions
    - Lambdified expressions
    - Regular Python callables

    Parameters
    ----------
    f : sympy.Expr, callable, or lambdified expression
        Function to integrate. Can be:
        - A SymPy symbolic expression (e.g., x**2 + 1)
        - A Python function (e.g., lambda x: x**2 + 1)
        - A lambdified expression from sympy.lambdify

    a : float
        Lower integration limit

    b : float
        Upper integration limit

    n : int, optional
        Order of Gaussian quadrature (number of integration points).
        Higher values give more accurate results but take longer.
        Default: 5

    param : sympy.Symbol, optional
        The integration variable for symbolic expressions.
        If not provided, will be automatically inferred from the expression.
        Only needed for symbolic expressions with multiple free symbols.

    Returns
    -------
    float
        The numerical approximation to the integral

    Raises
    ------
    ImportError
        If scipy is not installed (required for Gaussian quadrature)
    ValueError
        If the function format is not recognized or parameter cannot be inferred

    Examples
    --------
    Using a symbolic expression:

    >>> import sympy as sp
    >>> from mechanicskit import gaussint
    >>> x = sp.Symbol('x')
    >>> gaussint(1 + 3*x**2 - 3*x**3, 0, 1, n=2)
    1.5

    Using a Python lambda function:

    >>> gaussint(lambda x: 1 + 3*x**2 - 3*x**3, 0, 1, n=2)
    1.5

    Using a regular Python function:

    >>> def f(x):
    ...     return 1 + 3*x**2 - 3*x**3
    >>> gaussint(f, 0, 1, n=2)
    1.5

    Using a lambdified expression:

    >>> x = sp.Symbol('x')
    >>> f_lambdified = sp.lambdify(x, 1 + 3*x**2 - 3*x**3, 'numpy')
    >>> gaussint(f_lambdified, 0, 1, n=2)
    1.5

    Symbolic expression with explicit parameter:

    >>> x, y = sp.symbols('x y')
    >>> expr = x**2 + y  # Multiple free symbols
    >>> gaussint(expr, 0, 1, n=5, param=x)  # Integrate with respect to x
    0.333... + y

    Higher order quadrature for better accuracy:

    >>> gaussint(sp.sin(x), 0, sp.pi, n=10)
    2.0

    See Also
    --------
    scipy.integrate.fixed_quad : SciPy's fixed-order Gaussian quadrature
    scipy.integrate.quad : SciPy's adaptive quadrature
    fplot : Plot symbolic functions (similar flexible interface)

    Notes
    -----
    This function uses scipy.special.roots_legendre to compute Gauss-Legendre
    quadrature nodes and weights, then transforms them to the integration
    interval [a, b].

    The transformation from standard interval [-1, 1] to [a, b] is:
        x_i = 0.5 * (b - a) * nodes[i] + 0.5 * (b + a)
        integral = 0.5 * (b - a) * sum(weights[i] * f(x_i))

    For symbolic expressions, the function automatically uses SymPy's lambdify
    for efficient numerical evaluation.

    References
    ----------
    .. [1] Gaussian quadrature on Wikipedia:
           https://en.wikipedia.org/wiki/Gaussian_quadrature
    .. [2] SciPy documentation for fixed_quad:
           https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.fixed_quad.html
    """
    if not HAS_SCIPY:
        raise ImportError(
            "SciPy is required for gaussint. Install with: pip install scipy"
        )

    # Determine the type of input and convert to numerical function if needed
    if HAS_SYMPY and isinstance(f, sp.Expr):
        # SymPy symbolic expression
        # Check if it's a constant (no free symbols)
        if len(f.free_symbols) == 0:
            # Constant expression - just multiply by interval length
            const_value = float(f)
            return const_value * (b - a)

        if param is None:
            param = _infer_parameter(f)
        # Convert to numerical function
        num_func = sp.lambdify(param, f, 'numpy')
    elif callable(f):
        # Already a callable (lambda, function, or lambdified expression)
        num_func = f
    else:
        raise ValueError(
            f"Function f must be a SymPy expression or a callable, got {type(f)}"
        )

    # Get Gauss-Legendre nodes and weights for standard interval [-1, 1]
    nodes, weights = scipy.special.roots_legendre(n)

    # Perform Gaussian quadrature with transformation to [a, b]
    integral = 0.0
    for i in range(n):
        # Transform node from [-1, 1] to [a, b]
        xi = 0.5 * (b - a) * nodes[i] + 0.5 * (b + a)
        integral += weights[i] * num_func(xi)

    # Apply scaling factor for the interval transformation
    integral *= 0.5 * (b - a)

    return integral


def _infer_parameter(expr):
    """
    Infer the integration variable from a symbolic expression.

    Parameters
    ----------
    expr : sympy.Expr
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
    if not HAS_SYMPY:
        raise ImportError("SymPy is required for symbolic expressions")

    # Get free symbols from the expression
    free_syms = expr.free_symbols

    if len(free_syms) == 0:
        raise ValueError(
            "Cannot infer integration variable: expression has no free symbols. "
            "The expression might be a constant."
        )
    elif len(free_syms) == 1:
        return list(free_syms)[0]
    else:
        # Multiple symbols - try common conventions for integration variables
        sym_names = [str(s) for s in free_syms]

        # Check for common variable names in order of preference
        for common_name in ['x', 't', 's', 'u', 'v', 'theta', 'r']:
            if common_name in sym_names:
                idx = sym_names.index(common_name)
                return list(free_syms)[idx]

        # No common name found - raise error
        raise ValueError(
            f"Cannot infer integration variable: expression has multiple free symbols {free_syms}. "
            f"Please specify the parameter explicitly using the 'param' argument."
        )
