import numpy as np

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