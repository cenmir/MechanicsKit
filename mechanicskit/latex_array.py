import numpy as np
from IPython.display import display, Latex


def _resolve_wrap(wrap):
    """Normalize the user-facing ``wrap`` argument.

    ``None``/``False`` -> wrapping disabled.
    ``True``           -> one summand per line.
    Integer ``N``      -> ``N`` summands per line (must be >= 1).
    """
    if wrap is None or wrap is False:
        return None
    if wrap is True:
        return 1
    n = int(wrap)
    if n < 1:
        raise ValueError("wrap must be True or a positive integer")
    return n


def _split_add_for_wrap(expr, n_per_line, has_lhs):
    """Render a SymPy ``Add`` as multi-line aligned summands.

    Returns the inner LaTeX (lines joined with ``\\\\``), to be placed
    inside ``\\begin{aligned} ... \\end{aligned}``. The first line carries
    no continuation marker; subsequent lines start with ``&\\quad`` so they
    align under the first chunk's leading column. Returns ``None`` when
    ``expr`` is not an ``Add`` or has at most ``n_per_line`` summands.
    """
    try:
        import sympy as sp
        from sympy import latex as sympy_latex
    except ImportError:
        return None

    if not isinstance(expr, sp.Add):
        return None
    args = list(expr.args)
    if len(args) <= n_per_line:
        return None

    def render(term, is_first):
        s = sympy_latex(term).replace(r'\frac', r'\dfrac')
        if is_first:
            return s
        if s.lstrip().startswith('-'):
            return s
        return '+ ' + s

    rendered = [render(t, i == 0) for i, t in enumerate(args)]
    chunks = [' '.join(rendered[i:i + n_per_line])
              for i in range(0, len(rendered), n_per_line)]

    lines = []
    if has_lhs:
        lines.append(chunks[0])
        for c in chunks[1:]:
            lines.append('&\\quad ' + c)
    else:
        lines.append('& ' + chunks[0])
        for c in chunks[1:]:
            lines.append('&\\quad ' + c)
    return ' \\\\\n'.join(lines)


def _wrap_sympy_inner(obj, n_per_line, external_lhs=False):
    """Return the inner aligned LaTeX for ``obj`` when wrapping applies.

    Handles bare expressions and ``Eq`` instances whose RHS is an ``Add``.
    Returns ``None`` if no wrapping is needed.

    When ``external_lhs`` is True (used by ``ltx``), the caller has already
    provided a left-hand side and alignment ``&``, so the first chunk is
    emitted without a leading ``&``.
    """
    try:
        import sympy as sp
    except ImportError:
        return None

    if isinstance(obj, sp.Equality):
        from sympy import latex as sympy_latex
        rhs_split = _split_add_for_wrap(obj.rhs, n_per_line, has_lhs=True)
        if rhs_split is None:
            return None
        lhs_latex = sympy_latex(obj.lhs).replace(r'\frac', r'\dfrac')
        return lhs_latex + ' &= ' + rhs_split
    return _split_add_for_wrap(obj, n_per_line, has_lhs=external_lhs)


class LatexArray:
    """
    Wraps a NumPy array to provide a rich LaTeX display in marimo
    by implementing the `_repr_latex_` method.

    Usage:
        from mechanicskit import LatexArray, la

        # Function call syntax
        LatexArray(np.arange(10))

        # Pipe syntax (recommended)
        np.arange(10) | la
    """

    def __init__(self, array, arraystretch=1.0, alignedstretch=2.5, evalf=None, simplify=False, trigsimp=False, show_shape=False, wrap=None):
        self._arraystretch = arraystretch
        self.alignedstretch = alignedstretch
        self.evalf = evalf
        self.simplify = simplify
        self.trigsimp = trigsimp
        self.show_shape = show_shape
        self.wrap = _resolve_wrap(wrap)
        self._is_dict = isinstance(array, dict)
        # Detect SymPy objects and pass them through directly
        self._is_sympy = hasattr(array, '__module__') and array.__module__ and 'sympy' in array.__module__
        if self._is_dict:
            self._dict = array
        elif self._is_sympy:
            self._sympy_obj = array
        else:
            self.array = np.asarray(array) # Store the array

    def _wrap_stretch(self, latex_str):
        """Wrap a LaTeX matrix string with \\arraystretch."""
        return f"$$ {{\\def\\arraystretch{{{self._arraystretch}}}{latex_str}}} $$"

    def _repr_latex_(self):
        """
        This is the method marimo looks for, based on the docs.
        It must return a raw LaTeX string surrounded by $$.
        """
        # Handle dicts (e.g. SymPy solver output)
        if self._is_dict:
            return self._dict_to_latex()

        # Handle SymPy objects directly (Matrix, Eq, etc.)
        if self._is_sympy:
            from sympy import latex as sympy_latex, simplify as sympy_simplify, trigsimp as sympy_trigsimp
            obj = self._sympy_obj
            if self.trigsimp:
                try:
                    obj = sympy_trigsimp(obj)
                except Exception:
                    pass
            if self.simplify:
                try:
                    obj = sympy_simplify(obj)
                except Exception:
                    pass
            if self.evalf is not None and hasattr(obj, 'evalf'):
                obj = obj.evalf(self.evalf)
            if self.wrap is not None:
                wrapped = _wrap_sympy_inner(obj, self.wrap)
                if wrapped is not None:
                    aligned_block = "\\begin{aligned}\n" + wrapped + "\n\\end{aligned}"
                    return self._wrap_stretch(aligned_block)
            s = sympy_latex(obj).replace(r'\frac', r'\dfrac')
            if self.show_shape and hasattr(obj, 'shape'):
                rows, cols = obj.shape
                s += f"_{{{rows} \\times {cols}}}"
            return self._wrap_stretch(s)

        array = self.array

        # --- Helper to format numbers ---
        def get_val(v):
            # Check if it's a SymPy expression first
            try:
                if hasattr(v, '__module__') and v.__module__ and 'sympy' in v.__module__:
                    from sympy import latex as sympy_latex
                    return sympy_latex(v).replace(r'\frac', r'\dfrac')
            except (AttributeError, ImportError):
                pass

            if isinstance(v, (np.complexfloating, complex)):
                # Format as 'a+bj' or 'a-bj' with sign handling for imag part
                return f"{v.real:.2f}{v.imag:+.2f}j"
            if isinstance(v, (float, np.floating)):
                return f"{v:.2f}"
            return str(v)

        # --- Branch 0: Scalars (0-dimensional arrays) ---
        if array.ndim == 0:
            return f"$$ {get_val(array.item())} $$"

        # --- Branch 1: 1D Arrays (Vectors) ---
        if array.ndim == 1:
            n = array.shape[0]
            if n == 0:
                latex_str = "\\begin{bmatrix} \\end{bmatrix}"
            elif 1 <= n <= 5:  # Short Column Vector
                latex_str = "\\begin{bmatrix}\n"
                latex_str += " \\\\\n".join([get_val(v) for v in array])
                latex_str += "\n\\end{bmatrix}"
            elif 6 <= n <= 8:  # Full Row Vector + Transpose
                latex_str = "\\begin{bmatrix}\n"
                row_strs = [get_val(v) for v in array]
                latex_str += " & ".join(row_strs) + "\n"
                latex_str += "\\end{bmatrix}^\\mathsf T"
            else:  # n > 8 (Truncated Row Vector + Transpose)
                show_n = 5  # Show first 5 elements
                latex_str = "\\begin{bmatrix}\n"
                row_strs = [get_val(array[i]) for i in range(show_n)]
                row_strs.append("\\cdots")
                row_strs.append(get_val(array[-1]))  # and last element
                latex_str += " & ".join(row_strs) + "\n"
                latex_str += "\\end{bmatrix}^\\mathsf T"
            if self.show_shape:
                latex_str += f"_{{{n} \\times 1}}"
            return self._wrap_stretch(latex_str)

        # --- Branch 2: 2D Arrays (Matrices) ---
        elif array.ndim == 2:
            n_rows, n_cols = array.shape
            threshold = 8
            show_n = 6
            truncate_rows = n_rows > threshold
            truncate_cols = n_cols > threshold
            latex_str = "\\begin{bmatrix}\n"
            row_indices = list(range(n_rows))
            if truncate_rows:
                row_indices = list(range(show_n)) + [-1] + [n_rows - 1]
            for i in row_indices:
                row_strs = []
                col_indices = list(range(n_cols))
                if truncate_cols:
                    col_indices = list(range(show_n)) + [-1] + [n_cols - 1]
                if i == -1:
                    for j in col_indices:
                        row_strs.append("\\vdots" if j != -1 else "\\ddots")
                else:
                    for j in col_indices:
                        if j == -1:
                            row_strs.append("\\cdots")
                        else:
                            row_strs.append(get_val(array[i, j]))
                latex_str += " & ".join(row_strs) + " \\\\\n"
            latex_str += "\\end{bmatrix}"
            if self.show_shape:
                latex_str += f"_{{{n_rows} \\times {n_cols}}}"
            return self._wrap_stretch(latex_str)

        # --- Branch 3: Other ---
        else:
            # Fallback to the default (ugly) text repr
            return f"$$ \\text{{{repr(array)}}} $$"


    def _repr_mimebundle_(self, **kwargs):
        """Fallback for Jupyter environments that prefer mimebundle over _repr_latex_."""
        return {"text/latex": self._repr_latex_()}

    def _dict_to_latex(self):
        """Render a dict as an aligned equation system."""
        try:
            from sympy import latex as sympy_latex
            has_sympy = True
        except ImportError:
            has_sympy = False

        def to_latex(obj):
            if has_sympy and hasattr(obj, '__module__') and 'sympy' in str(getattr(obj, '__module__', '')):
                if self.trigsimp:
                    try:
                        from sympy import trigsimp as sympy_trigsimp
                        obj = sympy_trigsimp(obj)
                    except Exception:
                        pass
                if self.simplify:
                    try:
                        from sympy import simplify as sympy_simplify
                        obj = sympy_simplify(obj)
                    except Exception:
                        pass
                if self.evalf is not None and hasattr(obj, 'evalf'):
                    obj = obj.evalf(self.evalf)
                return sympy_latex(obj).replace(r'\frac', r'\dfrac')
            return str(obj)

        lines = [f"{to_latex(k)} &= {to_latex(v)}" for k, v in self._dict.items()]
        gap = f"{(self.alignedstretch - 1) * 6:.1f}pt"
        latex_str = "\\begin{aligned}\n" + f" \\\\[{gap}]\n".join(lines) + "\n\\end{aligned}"
        return f"$$ {latex_str} $$"


class LatexRenderer:
    """
    LaTeX renderer for arrays, matrices, and SymPy expressions.

    Pipe syntax (recommended):
        A | la                      # render array/matrix/SymPy object
        sol | la.evalf(3)           # evaluate to 3 significant figures
        M | la.simplify()           # apply sp.simplify before rendering
        M | la.trigsimp()           # apply sp.trigsimp before rendering
        A | la.arraystretch(2.5)    # increase row spacing
        A | la.shape()              # show dimensions as subscript

    Function call syntax:
        la(A)                       # render array
        la(A, evalf=3)              # with evalf
        la(A, show_shape=True)      # with dimensions

    Keyword arguments (via function call or pipe):
        arraystretch : float    Row spacing multiplier (default: 1.0)
        alignedstretch : float  Row spacing for aligned/dict display (default: 2.5)
        evalf : int             Significant figures for SymPy (default: None)
        simplify : bool         Apply sp.simplify (default: False)
        trigsimp : bool         Apply sp.trigsimp (default: False)
        show_shape : bool       Show m x n subscript (default: False)
        wrap : bool or int      Break a long top-level SymPy ``Add`` across
                                aligned lines: ``True`` = one summand per line,
                                ``N`` = pack N summands per line. (default: None)

    Chainable methods:
        .evalf(n)          .simplify()        .trigsimp()
        .arraystretch(n)   .shape()           .wrap(n=True)

    Supports: NumPy arrays, SymPy Matrix/Eq/expressions, dicts (as aligned equations)
    """
    # High priority to override NumPy's element-wise operations
    __array_priority__ = 1000

    def __init__(self, arraystretch=1.0, alignedstretch=2.5, evalf_n=None, simplify_flag=False, trigsimp_flag=False, show_shape=False, wrap_n=None):
        self._arraystretch = arraystretch
        self.alignedstretch = alignedstretch
        self._evalf_n = evalf_n
        self._simplify = simplify_flag
        self._trigsimp = trigsimp_flag
        self._show_shape = show_shape
        self._wrap_n = _resolve_wrap(wrap_n)

    def _new(self, **overrides):
        """Return a new renderer with overridden config."""
        kwargs = dict(
            arraystretch=self._arraystretch,
            alignedstretch=self.alignedstretch,
            evalf_n=self._evalf_n,
            simplify_flag=self._simplify,
            trigsimp_flag=self._trigsimp,
            show_shape=self._show_shape,
            wrap_n=self._wrap_n,
        )
        kwargs.update(overrides)
        return LatexRenderer(**kwargs)

    def _wrap(self, obj):
        return LatexArray(
            obj,
            arraystretch=self._arraystretch,
            alignedstretch=self.alignedstretch,
            evalf=self._evalf_n,
            simplify=self._simplify,
            trigsimp=self._trigsimp,
            show_shape=self._show_shape,
            wrap=self._wrap_n,
        )

    def __call__(self, array=None, *, arraystretch=None, alignedstretch=None, evalf=None, simplify=None, trigsimp=None, show_shape=None, wrap=None):
        """
        Called when used as a function.

        - ``la(array)`` -> wraps array as LatexArray
        - ``la(arraystretch=2.0)`` -> returns a configured renderer for piping:
          ``arr | la(arraystretch=2.0)``
        - ``la(evalf=4)`` -> apply ``.evalf(4)`` to SymPy values before rendering
        - ``la(simplify=True)`` -> apply ``sp.simplify`` to SymPy values
        - ``la(trigsimp=True)`` -> apply ``sp.trigsimp`` to SymPy values
        - ``la(show_shape=True)`` -> display matrix dimensions as subscript
        """
        overrides = {}
        if arraystretch is not None: overrides['arraystretch'] = arraystretch
        if alignedstretch is not None: overrides['alignedstretch'] = alignedstretch
        if evalf is not None: overrides['evalf_n'] = evalf
        if simplify is not None: overrides['simplify_flag'] = simplify
        if trigsimp is not None: overrides['trigsimp_flag'] = trigsimp
        if show_shape is not None: overrides['show_shape'] = show_shape
        if wrap is not None: overrides['wrap_n'] = wrap
        if array is None:
            return self._new(**overrides)
        return self._new(**overrides)._wrap(array)

    def __ror__(self, other):
        """Called when this object appears on the right side of |"""
        return self._wrap(other)

    def evalf(self, n=4):
        """Return a configured renderer that calls ``.evalf(n)`` on SymPy values.

        Usage: ``sol | la.evalf(4)``
        """
        return self._new(evalf_n=n)

    def arraystretch(self, n=2.5):
        """Return a configured renderer with the given matrix row stretch.

        Usage: ``mat | la.arraystretch(2.5)``
        """
        return self._new(arraystretch=n)

    def simplify(self):
        """Return a configured renderer that calls ``sp.simplify`` on SymPy values.

        Usage: ``sol | la.simplify()``
        """
        return self._new(simplify_flag=True)

    def trigsimp(self):
        """Return a configured renderer that calls ``sp.trigsimp`` on SymPy values.

        Usage: ``sol | la.trigsimp()``
        """
        return self._new(trigsimp_flag=True)

    def shape(self):
        """Return a configured renderer that displays matrix dimensions as subscript.

        Usage: ``A | la.shape()``
        """
        return self._new(show_shape=True)

    def wrap(self, n=True):
        """Return a configured renderer that breaks long ``Add`` expressions
        across multiple aligned lines.

        ``n=True`` (default) puts one summand per line. An integer ``n``
        packs that many summands per line.

        Usage: ``f | la.wrap()`` or ``f | la.wrap(2)``
        """
        return self._new(wrap_n=n)


# Create the singleton instance that users will import
la = LatexRenderer()


def display_labeled_latex(label, array, precision=2, arrayStretch=1.5, show_shape=False):
    r"""
    Display a labeled LaTeX equation with a NumPy array or SymPy object formatted as a matrix.

    Parameters:
    -----------
    label : str
        The label/prefix for the equation (e.g., "\\mathbf{R} = ")
    array : array_like or sympy object
        The NumPy array or SymPy expression to display
    precision : int, optional
        Number of decimal places to display (default: 2). Ignored for SymPy objects.
    arrayStretch : float, optional
        Vertical spacing multiplier for matrix rows (default: 1.5). Only applies to
        matrices and arrays, not scalars. Set to 1.0 for default LaTeX spacing.
    show_shape : bool, optional
        If True, display the array shape as a subscript on the matrix (default: False).
        For 2D arrays: displays as _{rows \times cols}
        For 1D arrays: displays as _{n}

    Example:
    --------
    >>> display_labeled_latex("\\mathbf{R} = ", R, 0)
    Displays: $$ \mathbf{R} = \begin{bmatrix}1 & -0 \\ 0 & 1\end{bmatrix} $$

    >>> display_labeled_latex("\\mathbf{u} = ", u, 4)
    Displays: $$ \mathbf{u} = \begin{bmatrix}0.0000 \\ 0.0000 \\ -0.1984 \\ ...\end{bmatrix} $$

    >>> display_labeled_latex("\\mathbf{A} = ", A, arrayStretch=2.0)
    Displays matrix with double row spacing

    >>> display_labeled_latex("U = ", U, show_shape=True)
    Displays: $$ U = \begin{bmatrix}...\end{bmatrix}_{4 \times 2} $$
    """
    # Check if the input is a SymPy object
    try:
        from sympy import latex as sympy_latex
        # Check if it has the _sympy_ attribute or is from sympy module
        if hasattr(array, '__module__') and 'sympy' in array.__module__:
            # Use SymPy's latex function for symbolic expressions
            latex_str = sympy_latex(array)
            # Add shape subscript if requested and object has shape
            if show_shape and hasattr(array, 'shape'):
                shape = array.shape
                if len(shape) == 1:
                    latex_str += f"_{{{shape[0]}}}"
                elif len(shape) == 2:
                    latex_str += f"_{{{shape[0]} \\times {shape[1]}}}"
            # Wrap with arraystretch for matrices/vectors only
            if hasattr(array, 'shape'):
                shape = array.shape
                # Apply arraystretch for vectors and matrices (not scalars)
                if len(shape) > 0 and (shape[0] > 1 or len(shape) > 1):
                    latex_str = f"{{\\def\\arraystretch{{{arrayStretch}}}{latex_str}}}"
            full_latex = f"$$ {label}{latex_str} $$"
            display(Latex(full_latex))
            return
    except (ImportError, AttributeError):
        pass  # SymPy not available, continue with NumPy handling

    # Check if input is a OneArray and extract underlying data
    if hasattr(array, 'data') and hasattr(array, '__class__') and array.__class__.__name__ == 'OneArray':
        array = array.data

    array = np.asarray(array)

    # Helper to format numbers with specified precision
    def format_val(v, prec):
        # Check if it's a SymPy expression first
        try:
            if hasattr(v, '__module__') and v.__module__ and 'sympy' in v.__module__:
                from sympy import latex as sympy_latex
                return sympy_latex(v)
        except (AttributeError, ImportError):
            pass

        if isinstance(v, (np.complexfloating, complex)):
            # Format complex numbers
            return f"{v.real:.{prec}f}{v.imag:+.{prec}f}j"
        elif isinstance(v, (float, np.floating)):
            return f"{v:.{prec}f}"
        else:
            return str(v)

    # Handle scalar (0-dimensional array)
    if array.ndim == 0:
        latex_str = format_val(array.item(), precision)

    # Handle 1D array (vector - displayed as column vector)
    elif array.ndim == 1:
        n = array.shape[0]
        threshold = 8
        show_n = 6
        truncate = n > threshold

        latex_str = "\\begin{bmatrix}"
        if truncate:
            # Show first show_n elements, ..., then last element
            values = [format_val(array[i], precision) for i in range(show_n)]
            values.append("\\vdots")
            values.append(format_val(array[-1], precision))
        else:
            values = [format_val(v, precision) for v in array]
        latex_str += " \\\\ ".join(values)
        latex_str += "\\end{bmatrix}"
        # Add shape subscript if requested
        if show_shape:
            latex_str += f"_{{{array.shape[0]}}}"
        # Apply arraystretch for vectors with more than one element
        if len(array) > 1:
            latex_str = f"{{\\def\\arraystretch{{{arrayStretch}}}{latex_str}}}"

    # Handle 2D array (matrix)
    elif array.ndim == 2:
        n_rows, n_cols = array.shape
        threshold = 8
        show_n = 6
        truncate_rows = n_rows > threshold
        truncate_cols = n_cols > threshold

        latex_str = "\\begin{bmatrix}"
        rows = []

        # Determine which row indices to show
        row_indices = list(range(n_rows))
        if truncate_rows:
            row_indices = list(range(show_n)) + [-1] + [n_rows - 1]

        for i in row_indices:
            row_vals = []
            # Determine which column indices to show
            col_indices = list(range(n_cols))
            if truncate_cols:
                col_indices = list(range(show_n)) + [-1] + [n_cols - 1]

            if i == -1:
                # This is the "..." row
                for j in col_indices:
                    row_vals.append("\\vdots" if j != -1 else "\\ddots")
            else:
                for j in col_indices:
                    if j == -1:
                        row_vals.append("\\cdots")
                    else:
                        row_vals.append(format_val(array[i, j], precision))

            rows.append(" & ".join(row_vals))

        latex_str += " \\\\ ".join(rows)
        latex_str += "\\end{bmatrix}"
        # Add shape subscript if requested
        if show_shape:
            latex_str += f"_{{{n_rows} \\times {n_cols}}}"
        # Apply arraystretch for matrices
        latex_str = f"{{\\def\\arraystretch{{{arrayStretch}}}{latex_str}}}"

    else:
        # For higher dimensions, fall back to text representation
        latex_str = f"\\text{{{repr(array)}}}"

    # Combine label and array, wrap in $$
    full_latex = f"$$ {label}{latex_str} $$"

    # Display using IPython
    display(Latex(full_latex))


# Alias for display_labeled_latex
labeled = display_labeled_latex


class LatexExpression:
    """
    Builds a LaTeX expression from variadic arguments of strings and arrays.

    Enables the pedagogical pattern of defining multiple matrices/vectors in one display:
    "Let A=..., B=..., x=..."

    Usage:
        from mechanicskit import latex_expression, ltx

        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        x = np.array([1, 2])

        # Display multiple arrays in one expression
        ltx("A=", A, ",\\ B=", B, ",\\ \\mathbf{x}=", x)

        # With custom precision
        latex_expression("A=", A, ",\\ B=", B, precision=4)
    """

    def __init__(self, *args, precision=None, arraystretch=None, show_shape=False, aligned=False, wrap=None):
        """
        Parameters
        ----------
        *args : str or array_like
            Alternating strings (LaTeX fragments) and arrays/values to format.
            Strings are passed through as-is.
            Arrays are converted to bmatrix format.
        precision : int, optional
            Number of significant figures. For SymPy expressions, calls evalf(n).
            For NumPy floats, formats to n decimal places. (default: None, full precision)
        arraystretch : float, optional
            Row spacing multiplier (default: None, no override)
        show_shape : bool, optional
            Display matrix dimensions as subscript (default: False)
        aligned : bool, optional
            Wrap the composed expression in ``\\begin{aligned} ... \\end{aligned}``
            for left-aligned multi-row equations. Rows are separated by ``\\\\``
            and aligned on ``&`` as usual. (default: False)
        """
        self.args = args
        self.precision = precision
        self._arraystretch = arraystretch
        self._show_shape = show_shape
        self._wrap = _resolve_wrap(wrap)
        # ``wrap`` implies aligned because the broken summands need an
        # ``aligned`` environment to render correctly.
        self._aligned = aligned or self._wrap is not None

    def _format_val(self, v):
        """Format a single value for LaTeX."""
        # Check if it's a SymPy expression first
        try:
            if hasattr(v, '__module__') and v.__module__ and 'sympy' in v.__module__:
                from sympy import latex as sympy_latex
                if self.precision is not None:
                    v = v.evalf(self.precision)
                if self._wrap is not None:
                    wrapped = _wrap_sympy_inner(v, self._wrap, external_lhs=True)
                    if wrapped is not None:
                        return wrapped
                return sympy_latex(v).replace(r'\frac', r'\dfrac')
        except (AttributeError, ImportError):
            pass

        np_precision = self.precision if self.precision is not None else 2
        if isinstance(v, (np.complexfloating, complex)):
            return f"{v.real:.{np_precision}f}{v.imag:+.{np_precision}f}j"
        elif isinstance(v, (float, np.floating)):
            return f"{v:.{np_precision}f}"
        else:
            return str(v)

    def _array_to_latex(self, array):
        """Convert a NumPy array to LaTeX bmatrix format."""
        # Check if input is a OneArray and extract underlying data
        if hasattr(array, 'data') and hasattr(array, '__class__') and array.__class__.__name__ == 'OneArray':
            array = array.data

        # Handle SymPy matrices directly
        try:
            if hasattr(array, '__module__') and 'sympy' in array.__module__:
                from sympy import latex as sympy_latex
                if self.precision is not None and hasattr(array, 'evalf'):
                    array = array.evalf(self.precision)
                if self._wrap is not None:
                    wrapped = _wrap_sympy_inner(array, self._wrap, external_lhs=True)
                    if wrapped is not None:
                        return wrapped
                s = sympy_latex(array).replace(r'\frac', r'\dfrac')
                if self._show_shape and hasattr(array, 'shape'):
                    rows, cols = array.shape
                    s += f"_{{{rows} \\times {cols}}}"
                return s
        except (AttributeError, ImportError):
            pass

        array = np.asarray(array)

        # Scalar
        if array.ndim == 0:
            return self._format_val(array.item())

        # 1D array (vector) - display as column vector
        if array.ndim == 1:
            n = array.shape[0]
            threshold = 8
            show_n = 6
            truncate = n > threshold

            latex_str = "\\begin{bmatrix}"
            if truncate:
                values = [self._format_val(array[i]) for i in range(show_n)]
                values.append("\\vdots")
                values.append(self._format_val(array[-1]))
            else:
                values = [self._format_val(v) for v in array]
            latex_str += " \\\\ ".join(values)
            latex_str += "\\end{bmatrix}"
            if self._show_shape:
                latex_str += f"_{{{n} \\times 1}}"
            return latex_str

        # 2D array (matrix)
        if array.ndim == 2:
            n_rows, n_cols = array.shape
            threshold = 8
            show_n = 6
            truncate_rows = n_rows > threshold
            truncate_cols = n_cols > threshold

            latex_str = "\\begin{bmatrix}"
            rows = []

            row_indices = list(range(n_rows))
            if truncate_rows:
                row_indices = list(range(show_n)) + [-1] + [n_rows - 1]

            for i in row_indices:
                row_vals = []
                col_indices = list(range(n_cols))
                if truncate_cols:
                    col_indices = list(range(show_n)) + [-1] + [n_cols - 1]

                if i == -1:
                    for j in col_indices:
                        row_vals.append("\\vdots" if j != -1 else "\\ddots")
                else:
                    for j in col_indices:
                        if j == -1:
                            row_vals.append("\\cdots")
                        else:
                            row_vals.append(self._format_val(array[i, j]))

                rows.append(" & ".join(row_vals))

            latex_str += " \\\\ ".join(rows)
            latex_str += "\\end{bmatrix}"
            if self._show_shape:
                latex_str += f"_{{{n_rows} \\times {n_cols}}}"
            return latex_str

        # Higher dimensions - fallback
        return f"\\text{{{repr(array)}}}"

    def _build_inner(self):
        """Join positional args into a single LaTeX string, applying aligned/arraystretch wrappers."""
        parts = []
        for arg in self.args:
            if isinstance(arg, str):
                parts.append(arg)
            else:
                parts.append(self._array_to_latex(arg))
        latex_str = "".join(parts)
        if self._aligned:
            latex_str = r"\begin{aligned}" + latex_str + r"\end{aligned}"
        if self._arraystretch is not None:
            latex_str = rf"{{\def\arraystretch{{{self._arraystretch}}}{latex_str}}}"
        return latex_str

    def _repr_latex_(self):
        """
        Return the LaTeX string for notebook display.
        Called automatically by Jupyter/Marimo.
        """
        return f"$$ {self._build_inner()} $$"

    def __str__(self):
        """Return the raw LaTeX string (without $$ delimiters)."""
        return self._build_inner()


def latex_expression(*args, precision=None, arraystretch=None, show_shape=False, aligned=False, wrap=None):
    """
    Labeled LaTeX display for matrices, vectors, and SymPy expressions.

    Combines LaTeX strings with arrays/expressions in a single display line.
    Also available as: ltx, labeled

    Usage:
        ltx(r"A =", A)                          # label + matrix
        ltx(r"A =", A, r",\\ B =", B)           # multiple items
        ltx(r"v(t) =", v, precision=3)           # 3 significant figures
        ltx(r"\\dot{\\bm r} =", M, arraystretch=2.5)  # row spacing
        ltx(r"K =", K, show_shape=True)          # show m x n subscript
        ltx(r"x(t) &=", x_sol, r"\\\\ y(t) &=", y, aligned=True)  # left-aligned rows

    Parameters
    ----------
    *args : str or array_like
        Alternating LaTeX strings and arrays/values.
        Strings are passed through as-is.
        Arrays are converted to bmatrix format.
    precision : int, optional
        Significant figures. SymPy: calls evalf(n). NumPy: decimal places. (default: None)
    arraystretch : float, optional
        Row spacing multiplier (default: None)
    show_shape : bool, optional
        Display matrix dimensions as subscript (default: False)
    aligned : bool, optional
        Wrap output in ``\\begin{aligned}...\\end{aligned}`` for left-aligned
        multi-row equations. Use ``&`` to mark the alignment column and ``\\\\``
        to separate rows. (default: False)
    wrap : bool or int, optional
        Break long top-level SymPy ``Add`` expressions across multiple aligned
        lines so they fit page width in both HTML and PDF. ``True`` puts one
        summand per line; an integer ``N`` packs ``N`` summands per line.
        Automatically enables ``aligned``. Short expressions are left
        unchanged. Pair with a label like ``r"f &="`` so continuation lines
        align after the ``=``. (default: None)

    Returns
    -------
    LatexExpression
        Object that renders as LaTeX in Jupyter/Marimo notebooks.

    Examples
    --------
    >>> from mechanicskit import ltx
    >>>
    >>> A = np.array([[1, 2], [3, 4]])
    >>> B = np.array([[5, 6], [7, 8]])
    >>> x = np.array([1, 2])
    >>>
    >>> # Display multiple arrays in one expression
    >>> ltx("A=", A, ",\\ B=", B, ",\\ \\mathbf{x}=", x)

    >>> # With custom precision
    >>> ltx("A=", A, ",\\ B=", B, precision=4)

    >>> # With row spacing
    >>> ltx("A=", A, arraystretch=2.5)
    >>>
    >>> # Break a long SymPy sum across aligned lines (one summand per line)
    >>> ltx(r"\\dot f &=", fdot, wrap=True)
    """
    return LatexExpression(*args, precision=precision, arraystretch=arraystretch, show_shape=show_shape, aligned=aligned, wrap=wrap)


# Short aliases
ltx = latex_expression
labeled = latex_expression