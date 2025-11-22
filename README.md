# MatKit

Mathematical toolkit for teaching linear algebra and finite element methods (FEM). Provides:
- **LaTeX rendering** for NumPy arrays in marimo notebooks
- **1-based indexing** for FEM education (pedagogical bridge between math and code)
- **Field management** for nodal, DOF, and element data
- **Element-agnostic mesh** supporting multiple FEM element types

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/cenmir/MatKit.git
```

For local development, install in editable mode:

```bash
uv pip install -e .
```

## Usage

```python
import numpy as np
from matkit import la

# Pipe syntax (recommended)
np.array([1, 2, 3]) | la

# Function call syntax
la(np.array([1, 2, 3]))

# Works with expressions
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
(A @ B) | la

# Scalars, vectors, and matrices all work
np.dot([1, 2, 3], [4, 5, 6]) | la
```

## Features

- **Automatic LaTeX rendering** for NumPy arrays in marimo notebooks
- **Scalars, vectors (1D), and matrices (2D)** are all supported
- **Intelligent truncation** for large arrays (>8 elements/dimension)
- **Clean pipe syntax** for readable code: `array | la`
- **Format preservation** for integers, floats (2 decimals), and complex numbers

## Example Output

Vectors with ≤5 elements display as column vectors:
```
[1]
[2]
[3]
```

Large matrices automatically truncate with ellipsis notation:
```
[1  2  3  ...  10]
[11 12 13 ... 20]
⋮   ⋮   ⋮   ⋱   ⋮
[91 92 93 ... 100]
```

## Why the Wrapper?

NumPy's `ndarray` is a C-defined immutable type that cannot have display methods added at runtime. Marimo also doesn't provide a public API for registering formatters for third-party types. The pipe syntax `| la` is the cleanest solution for opt-in LaTeX rendering.

## FEM Tools: Element-Agnostic Mesh with 1-Indexing

MatKit includes an **element-agnostic Mesh class** that supports multiple element types with a **hybrid indexing approach** bridging mathematical notation (1-based) and Python arrays (0-based).

### The Problem

In FEM textbooks:
- Nodes numbered: 1, 2, 3, 4, ...
- Element 1 connects nodes 1 and 2
- Node *i* has DOFs based on element type (e.g., *2i-1*, *2i* for RODs)

In Python (0-based indexing):
- `nodes[0]` is "node 1"
- `nodes[2]` is "node 3"
- Students must constantly translate

### The Solution: Mesh

```python
from matkit import Mesh, ELEMENT_TYPES

# Define using natural 1-based notation
coords = [[0, 0], [1, 0], [0, 1], [1, 1]]
connectivity = [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4]]  # 1-based!

# Auto-detect element type (ROD for 2 nodes)
mesh = Mesh(coords, connectivity)

# Or specify explicitly
mesh = Mesh(coords, connectivity, element_type='ROD')

# Access node 3 directly (not index 2!)
coord = mesh.get_node(3)

# Get DOF indices for assembly
dofs = mesh.dofs_for_node(3)  # Returns [4, 5] for 2D ROD
u[dofs] = [0.1, 0.2]  # Set displacement for node 3

# Works with lists too!
coords = mesh.get_node([1, 3])  # Multiple nodes
dofs = mesh.dofs_for_node([1, 3])  # Multiple DOF sets
```

### Supported Element Types (ANSA-style)

```python
print(ELEMENT_TYPES)
# 'ROD': 2 nodes, truss elements (2 or 3 DOFs per node)
# 'BEAM': 2 nodes, beam elements with rotations (3 or 6 DOFs per node)
# 'TRIA3': 3 nodes, 2D triangular elements
# 'QUAD4': 4 nodes, 2D quadrilateral elements
# 'TETRA4': 4 nodes, 3D tetrahedral elements
# 'HEXA8': 8 nodes, 3D hexahedral elements
```

### Why This Approach?

1. **Element-agnostic**: ROD, BEAM, TRIA3, QUAD4, TETRA4, HEXA8 support
2. **Reduces cognitive load**: Students focus on FEM concepts, not index arithmetic
3. **Matches textbooks**: Notation aligns with published literature
4. **Professional preparation**: ANSA-style element types, industry standard
5. **Transparent**: Simple enough to read and understand the source

See `test_mesh.py` for comprehensive examples.

## Development

This allows editing the source code and have the changes immediately reflected in the environment, which is perfect for development.

```bash
uv pip install -e .
```