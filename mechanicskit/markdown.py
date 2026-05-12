"""mechanicskit.markdown — type-aware markdown formatting.

Provides ``mk.md(template)``: a markdown helper that interpolates variables
from the calling cell's namespace, with type-aware rendering of scalars,
``LatexArray`` / ``LatexExpression``, NumPy arrays, pandas DataFrames, and
list-of-list / list-of-dict tables.

The output is **markdown source** (not HTML), exposed via ``_repr_markdown_``,
``_mime_("text/markdown", ...)``, and ``__format__``, so the same value
renders correctly in marimo, Jupyter, and quarto.
"""
from __future__ import annotations

import inspect
import string
from functools import singledispatch

import numpy as np


_MAX_TABLE_DIM = 8


class Markdown:
    """A markdown-source wrapper with dual rendering for marimo, Jupyter, and quarto.

    Stores the formatted markdown text. Implements:

    - ``_repr_markdown_()`` so Jupyter and quarto pick up the markdown source.
    - ``_mime_()`` returning ``("text/markdown", ...)`` for marimo.
    - ``__format__`` returning the markdown source so nested ``mk.md``
      calls compose under f-string interpolation (the trick marimo's own
      ``_md`` class uses).
    """

    def __init__(self, text: str):
        self.text = text

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"Markdown({self.text!r})"

    def _repr_markdown_(self) -> str:
        return self.text

    def _mime_(self):
        return "text/markdown", self.text

    def __format__(self, spec: str) -> str:
        del spec
        return self.text


def _format_scalar(v) -> str:
    """Format a single scalar with sensible defaults for inclusion in a table cell."""
    if isinstance(v, (np.complexfloating, complex)):
        return f"{v.real:.4g}{v.imag:+.4g}j"
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def _ndarray_to_markdown_table(arr: np.ndarray) -> str:
    from tabulate import tabulate

    rows = [[_format_scalar(v) for v in row] for row in arr]
    n_cols = arr.shape[1]
    return tabulate(rows, headers=[""] * n_cols, tablefmt="github")


@singledispatch
def _render(value) -> str:
    """Default renderer. Used for any type without a specific dispatch arm.

    If the value implements ``_repr_latex_`` (covers ``LatexArray``,
    ``LatexExpression``, SymPy Matrix/Expr, pandas Styler, ...), we use that
    output directly — it is already a ``$$...$$`` block that markdown will
    treat as display math. Otherwise we fall through to ``str(value)``,
    matching f-string semantics.
    """
    if hasattr(value, "_repr_latex_"):
        try:
            return value._repr_latex_()
        except Exception:
            pass
    return str(value)


@_render.register(np.ndarray)
def _(value: np.ndarray) -> str:
    if value.ndim == 0:
        return _format_scalar(value.item())
    if value.ndim == 1:
        return ", ".join(_format_scalar(v) for v in value)
    if value.ndim == 2:
        n_rows, n_cols = value.shape
        if n_rows <= _MAX_TABLE_DIM and n_cols <= _MAX_TABLE_DIM:
            return _ndarray_to_markdown_table(value)
        from .latex_array import LatexArray
        return LatexArray(value)._repr_latex_()
    return str(value)


@_render.register(list)
def _(value: list) -> str:
    if not value:
        return "[]"
    from tabulate import tabulate

    if all(isinstance(row, dict) for row in value):
        return tabulate(value, headers="keys", tablefmt="github")
    if all(isinstance(row, (list, tuple)) for row in value):
        n_cols = max(len(row) for row in value)
        return tabulate(value, headers=[""] * n_cols, tablefmt="github")
    return str(value)


def _try_register_pandas() -> None:
    """Register a DataFrame dispatch arm if pandas is importable.

    Pandas is not a hard dependency of MechanicsKit, so we register lazily.
    ``DataFrame.to_markdown`` itself uses tabulate under the hood — same
    engine the other table arms go through.
    """
    try:
        import pandas as pd
    except ImportError:
        return

    @_render.register(pd.DataFrame)
    def _(value: "pd.DataFrame") -> str:
        return value.to_markdown(index=False)


_try_register_pandas()


class _MkFormatter(string.Formatter):
    """A ``string.Formatter`` subclass with eval-based field lookup.

    ``str.format`` only allows ``{name}``, ``{name.attr}``, and ``{name[key]}``.
    We override ``get_field`` to ``eval`` the field text as a Python
    expression in the caller's namespace, giving f-string-equivalent power
    (``{a/b}``, ``{f(x)}``, ``{obj.method().attr}``) without the ``f``
    prefix. ``eval`` is run against the caller's own scope, so it cannot
    do anything the user's cell code couldn't already do.
    """

    def __init__(self, namespace: dict):
        super().__init__()
        self._ns = namespace

    def get_field(self, field_name, args, kwargs):
        try:
            value = eval(field_name, self._ns)
        except NameError as e:
            raise NameError(
                f"mk.md: name not found in caller scope while resolving "
                f"{{{field_name}}}: {e}"
            ) from None
        return value, field_name

    def format_field(self, value, spec):
        if spec:
            return format(value, spec)
        return _render(value)


def md(template: str, **kwargs) -> Markdown:
    """Format a markdown template, interpolating from the caller's namespace.

    Variables in ``{...}`` placeholders are resolved against ``kwargs`` first
    (when supplied), then against the caller's local and global scope. Each
    value is rendered type-aware:

    - ``LatexArray`` / ``LatexExpression`` (any ``_repr_latex_``) → its LaTeX
      source (already ``$$...$$``).
    - ``ndarray`` 2-D ≤8×8 → markdown table.
    - ``ndarray`` 2-D >8×8 → truncated LaTeX bmatrix (matches ``LatexArray``).
    - ``ndarray`` 1-D → comma-separated inline.
    - ``DataFrame`` → markdown table via ``to_markdown``.
    - ``list[list]`` / ``list[dict]`` → markdown table.
    - other → ``str(value)`` — matches f-string fallback.

    Standard Python format specs are honored: ``{x:.2f}``, ``{x:>10}``.
    Arithmetic and method calls work inside ``{...}`` because the field is
    evaluated as a Python expression: ``{a/b}``, ``{obj.attr}``, ``{f(x)}``.

    Literal ``{`` and ``}`` must be doubled (``{{`` and ``}}``), the same
    rule as ``str.format``. This affects LaTeX braces — write
    ``\\dfrac{{1}}{{2}}`` for a literal ``\\dfrac{1}{2}``.

    Returns
    -------
    Markdown
        An object with markdown source as its primary representation.
        Renders correctly in marimo, Jupyter, and quarto via
        ``_repr_markdown_``, ``_mime_``, and ``__format__``.

    Examples
    --------
    Scalar with format spec::

        a = 3.14159
        md("a = {a:.2f}")          # -> "a = 3.14"

    Arithmetic in the placeholder::

        F_st = 12345.0
        md("F = {F_st/1000:.2f} kN")   # -> "F = 12.34 kN"

    NumPy matrix becomes a markdown table::

        A = np.array([[1, 2], [3, 4]])
        md("A = {A}")                  # markdown table, not repr(A)

    Compose with ``ltx`` for LaTeX (passed as a kwarg so it does not need to
    live in the cell scope)::

        md("Stiffness: {expr}", expr=ltx(r"K = ", K, r"\\,N/m"))
    """
    frame = inspect.currentframe()
    caller = frame.f_back if frame is not None else None
    if caller is None:
        ns: dict = {}
    else:
        ns = {**caller.f_globals, **caller.f_locals}
    ns.update(kwargs)
    formatter = _MkFormatter(ns)
    return Markdown(formatter.vformat(template, (), {}))
