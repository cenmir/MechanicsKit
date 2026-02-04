"""
MechanicsKit - Mathematical Toolkit for Engineering Education

A pedagogical package for linear algebra and FEM that bridges mathematical notation
and Python implementation through 1-based indexing and LaTeX rendering.

This package provides:
- LaTeX rendering for NumPy arrays in marimo notebooks (LatexArray, la)
- 1-based indexing mesh interface for FEM education (Mesh, ELEMENT_TYPES)
- Element-agnostic support: ROD, BEAM, TRIA3, QUAD4, TETRA4, HEXA8
- OneArray: 1-based array wrapper for seamless FEM result handling
- Iterator methods for clean, Pythonic 1-based workflows
- Field management for nodal, DOF, and element data
- MATLAB-style patch function for mesh visualization (patch)
- MATLAB-style fplot for symbolic function plotting (fplot)
- Flexible Gaussian quadrature integration (gaussint)
- Animation utilities for responsive, auto-playing animations (to_responsive_html)

Philosophy:
-----------
Students think in mathematical notation (nodes 1, 2, 3..., DOFs 1, 2, 3...).
Python uses 0-based indexing (indices 0, 1, 2...).
MechanicsKit eliminates this translation burden through smart interface design.

Think in mathematics. Code in Python. MechanicsKit handles the translation.
"""
from .latex_array import (
    LatexArray, la, display_labeled_latex,
    LatexExpression, latex_expression, ltx, labeled
)
from .mesh import Mesh, ELEMENT_TYPES
from .one_array import OneArray
from .patch import patch
from .fplot import fplot
from .gaussint import gaussint
from .colormap_utils import colorbar, cmap
from .help import quick_ref
from .animation_utils import to_responsive_html

# Version information
__version__ = '0.3.0'

def version():
    """
    Display MechanicsKit version information.

    Returns
    -------
    str
        Version string

    Examples
    --------
    >>> import mechanicskit as mk
    >>> mk.version()
    'MechanicsKit v0.1.1'
    >>> mk.__version__
    '0.1.1'
    """
    return f'MechanicsKit v{__version__}'

__all__ = [
    'LatexArray', 'la', 'display_labeled_latex',
    'LatexExpression', 'latex_expression', 'ltx', 'labeled',
    'Mesh', 'ELEMENT_TYPES',
    'OneArray',
    'patch',
    'fplot',
    'gaussint',
    'colorbar', 'cmap',
    'version',
    'quick_ref',
    'to_responsive_html',
]
