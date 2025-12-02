import numpy as np
from IPython.display import display, Latex

class LatexArray:
    """
    Wraps a NumPy array to provide a rich LaTeX display in marimo
    by implementing the `_repr_latex_` method.

    Usage:
        from matkit import LatexArray, la

        # Function call syntax
        LatexArray(np.arange(10))

        # Pipe syntax (recommended)
        np.arange(10) | la
    """

    def __init__(self, array):
        self.array = np.asarray(array) # Store the array

    def _repr_latex_(self):
        """
        This is the method marimo looks for, based on the docs.
        It must return a raw LaTeX string surrounded by $$.
        """
        array = self.array

        # --- Helper to format numbers ---
        def get_val(v):
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
            return f"$$ {latex_str} $$"

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
            return f"$$ {latex_str} $$"

        # --- Branch 3: Other ---
        else:
            # Fallback to the default (ugly) text repr
            return f"$$ \\text{{{repr(array)}}} $$"


class LatexRenderer:
    """
    A singleton helper class that enables both function call and pipe syntax.

    Usage:
        from matkit import la

        # Function call syntax
        la(np.array([1, 2, 3]))

        # Pipe syntax
        np.array([1, 2, 3]) | la
    """
    # High priority to override NumPy's element-wise operations
    __array_priority__ = 1000

    def __call__(self, array):
        """Called when used as a function: la(array)"""
        return LatexArray(array)

    def __ror__(self, other):
        """Called when this object appears on the right side of |"""
        return LatexArray(other)


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
        latex_str = "\\begin{bmatrix}"
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
        latex_str = "\\begin{bmatrix}"
        rows = []
        for i in range(n_rows):
            row_vals = [format_val(array[i, j], precision) for j in range(n_cols)]
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