"""
Arrow plotting for 2D vectors.

Draws arrows in data coordinates. No auto-scaling nonsense.

Usage:
    import mechanicskit as mk

    # Single arrow with direction vector
    mk.arrow([0, 0], [300, 400], color='red')

    # Single arrow from SymPy vectors
    mk.arrow(rr_OA, FF_1, scale=0.001, color='blue')

    # Point-to-point
    mk.arrow(start=[0, 0], end=[1, 1], color='black')

    # Multiple arrows: lists of starts and directions
    mk.arrow([A, B, C], [FF_1, FF_2, FF_3], scale=0.001, color='red')

    # Multiple arrows: X, Y, U, V (like quiver)
    mk.arrow(X, Y, U, V, scale=0.001, color='red')

    # Multiple arrows: U, V only (origins at 0, 1, 2, ...)
    mk.arrow(U, V, scale=0.001, color='red')
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def _to_2d(vec):
    """Convert a SymPy Matrix, list, tuple, or numpy array to a 2D numpy array."""
    if hasattr(vec, 'tolist'):
        vec = vec.tolist()
    arr = np.asarray(vec, dtype=float).flatten()
    return arr[:2]


def _is_list_of_vectors(obj):
    """Check if obj is a list/array of vectors (not a single vector)."""
    if isinstance(obj, np.ndarray) and obj.ndim == 2:
        return True
    if isinstance(obj, (list, tuple)) and len(obj) > 0:
        first = obj[0]
        # A list of vectors: each element is itself a list/array/Matrix
        if isinstance(first, (list, tuple, np.ndarray)):
            return True
        if hasattr(first, 'tolist'):  # SymPy Matrix
            return True
    return False


def arrow(*args, start=None, direction=None, end=None, scale=1.0, ax=None,
          color='black', label=None, linewidth=2, head_scale=None,
          mutation_scale=15, zorder=5, **kwargs):
    """
    Draw 2D arrows in data coordinates.

    Calling conventions:

    1. ``arrow(start, direction)`` -- single or multiple arrows
    2. ``arrow(start=A, end=B)`` -- point-to-point
    3. ``arrow(start=A, end=B, scale=0.5)`` -- partial arrow
    4. ``arrow(U, V)`` -- arrays of components, origins at (0,1,2,...), (0,1,2,...)
    5. ``arrow(X, Y, U, V)`` -- arrays of origins and components

    All inputs accept SymPy Matrices, numpy arrays, lists, or tuples.
    3D vectors are silently truncated to 2D.

    For multiple arrows, pass lists of vectors::

        mk.arrow([A, B], [FF_1, FF_2], scale=0.001, color=['red', 'blue'])

    Parameters
    ----------
    start : array-like or SymPy Matrix
        Origin point(s) of the arrow(s).
    direction : array-like or SymPy Matrix, optional
        Direction and magnitude. Mutually exclusive with ``end``.
    end : array-like or SymPy Matrix, optional
        End point. Mutually exclusive with ``direction``.
    scale : float, optional
        Multiplier. 2 doubles length, 0.5 halves it. Default 1.0.
    ax : matplotlib Axes, optional
        Axes to draw on. Uses ``plt.gca()`` if None.
    color : str or list of str, optional
        Arrow color(s). Default ``'black'``.
    label : str or list of str, optional
        Legend label(s).
    linewidth : float, optional
        Shaft width. Default 2.
    head_scale : float, optional
        Arrowhead size in points. Bigger value = bigger head. Default 15.
        Preferred name; ``mutation_scale`` is kept as a backwards-compatible
        alias for the underlying matplotlib parameter.
    mutation_scale : float, optional
        Alias for ``head_scale`` (matplotlib's name for the same thing).
        Used only when ``head_scale`` is not given.
    zorder : int, optional
        Drawing order. Default 5.
    **kwargs
        Passed to ``FancyArrowPatch``.

    Returns
    -------
    FancyArrowPatch or list of FancyArrowPatch
    """
    # head_scale takes precedence; falls back to mutation_scale for back-compat.
    if head_scale is not None:
        mutation_scale = head_scale
    # Parse positional arguments
    if len(args) == 2:
        a, b = args
        # Check if this is (U, V) mode: two 1D arrays of same length
        a_arr = np.asarray(a, dtype=float) if not hasattr(a, 'tolist') else None
        b_arr = np.asarray(b, dtype=float) if not hasattr(b, 'tolist') else None
        if (a_arr is not None and b_arr is not None
                and a_arr.ndim == 1 and b_arr.ndim == 1
                and len(a_arr) == len(b_arr) and len(a_arr) > 3
                and not _is_list_of_vectors(a)):
            # U, V mode: origins at (0,1,2,...), (0,1,2,...)
            U, V = a_arr, b_arr
            n = len(U)
            X = np.arange(n, dtype=float)
            Y = np.zeros(n)
            start = list(zip(X, Y))
            direction = list(zip(U, V))
        elif _is_list_of_vectors(a) and _is_list_of_vectors(b):
            # Lists of vectors
            start = a
            direction = b
        else:
            # Single arrow: (start, direction)
            start = a
            direction = b
    elif len(args) == 4:
        # X, Y, U, V mode
        X = np.asarray(args[0], dtype=float).flatten()
        Y = np.asarray(args[1], dtype=float).flatten()
        U = np.asarray(args[2], dtype=float).flatten()
        V = np.asarray(args[3], dtype=float).flatten()
        start = list(zip(X, Y))
        direction = list(zip(U, V))
    elif len(args) == 1:
        start = args[0]
    elif len(args) == 0:
        pass  # keyword-only
    else:
        raise ValueError(f"Expected 0, 1, 2, or 4 positional arguments, got {len(args)}")

    if start is None:
        raise ValueError("start is required")
    if direction is not None and end is not None:
        raise ValueError("Specify direction or end, not both")
    if direction is None and end is None:
        raise ValueError("Specify either direction or end")

    if ax is None:
        ax = plt.gca()

    # Detect multiple arrows
    is_multi = _is_list_of_vectors(start)

    if not is_multi:
        # Single arrow
        return _draw_one(start, direction, end, scale, ax, color, label,
                         linewidth, mutation_scale, zorder, **kwargs)
    else:
        # Multiple arrows
        starts = start
        if direction is not None:
            dirs = direction if _is_list_of_vectors(direction) else [direction] * len(starts)
            ends = [None] * len(starts)
        else:
            ends = end if _is_list_of_vectors(end) else [end] * len(starts)
            dirs = [None] * len(starts)

        n = len(starts)
        colors = color if isinstance(color, (list, tuple)) and not isinstance(color, str) else [color] * n
        labels = label if isinstance(label, (list, tuple)) else [label] + [None] * (n - 1) if label else [None] * n

        patches = []
        for i in range(n):
            p = _draw_one(starts[i], dirs[i], ends[i], scale, ax,
                          colors[i % len(colors)],
                          labels[i] if i < len(labels) else None,
                          linewidth, mutation_scale, zorder, **kwargs)
            patches.append(p)
        return patches


def _draw_one(start, direction, end, scale, ax, color, label,
              linewidth, mutation_scale, zorder, **kwargs):
    """Draw a single arrow."""
    start_2d = _to_2d(start)

    if end is not None:
        end_2d = _to_2d(end)
        tip = start_2d + scale * (end_2d - start_2d)
    else:
        dir_2d = _to_2d(direction)
        tip = start_2d + scale * dir_2d

    patch = FancyArrowPatch(
        posA=tuple(start_2d),
        posB=tuple(tip),
        arrowstyle='-|>',
        mutation_scale=mutation_scale,
        color=color,
        linewidth=linewidth,
        label=label,
        zorder=zorder,
        **kwargs,
    )
    ax.add_patch(patch)
    return patch
