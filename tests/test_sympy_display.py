#!/usr/bin/env python3
"""Test script to debug display_labeled_latex with SymPy"""

import numpy as np
from sympy import symbols, cos, sin, Matrix, latex
from matkit import display_labeled_latex

# Create a SymPy matrix like in the notebook
theta = symbols('theta')
L_T = Matrix([
    [cos(theta), -sin(theta), 0, 0],
    [sin(theta), cos(theta), 0, 0],
    [0, 0, cos(theta), -sin(theta)],
    [0, 0, sin(theta), cos(theta)]
])

print("Type of L_T:", type(L_T))
print("Is SymPy Basic?", isinstance(L_T, Matrix))

from sympy import Basic
print("Is SymPy Basic (using Basic)?", isinstance(L_T, Basic))

print("\n--- Using SymPy's latex() directly ---")
print(latex(L_T))

print("\n--- Using display_labeled_latex ---")
display_labeled_latex("\\mathbf{L}^\\mathsf{T} = ", L_T)
