"""
MatKit - Mathematical Toolkit for Engineering Education

A pedagogical package for linear algebra and FEM that bridges mathematical notation
and Python implementation through 1-based indexing and LaTeX rendering.

This package provides:
- LaTeX rendering for NumPy arrays in marimo notebooks (LatexArray, la)
- 1-based indexing mesh interface for FEM education (Mesh, ELEMENT_TYPES)
- Element-agnostic support: ROD, BEAM, TRIA3, QUAD4, TETRA4, HEXA8
- OneArray: 1-based array wrapper for seamless FEM result handling
- Iterator methods for clean, Pythonic 1-based workflows
- Field management for nodal, DOF, and element data

Philosophy:
-----------
Students think in mathematical notation (nodes 1, 2, 3..., DOFs 1, 2, 3...).
Python uses 0-based indexing (indices 0, 1, 2...).
MatKit eliminates this translation burden through smart interface design.

Think in mathematics. Code in Python. MatKit handles the translation.
"""
from .latex_array import LatexArray, la
from .mesh import Mesh, ELEMENT_TYPES
from .one_array import OneArray

__all__ = [
    'LatexArray', 'la',
    'Mesh', 'ELEMENT_TYPES',
    'OneArray',
]
