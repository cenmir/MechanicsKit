# MatKit

Mathematical toolkit for teaching linear algebra and finite element methods (FEM). Provides:
- **LaTeX rendering** for NumPy arrays in marimo notebooks
- **Labeled LaTeX equations** with SymPy support for Jupyter notebooks
- **1-based indexing** for FEM education (pedagogical bridge between math and code)
- **Field management** for nodal, DOF, and element data
- **Element-agnostic mesh** supporting multiple FEM element types
- **MATLAB-style visualization** with `patch()` function

## Recent Updates

**December 2025:**
- ✨ Added `show_shape` parameter to display array dimensions as subscript (e.g., `_{4 \times 2}`)
- ✨ Added `display_labeled_latex()` function for Jupyter notebooks with labeled equations
- ✨ SymPy integration - automatic detection and proper symbolic formatting (e.g., `cos(θ)` instead of `cos(theta)`)
- ✨ Added `arrayStretch` parameter for customizable row spacing in matrices
- 📁 Reorganized repository structure following Python standards
- ✅ Added comprehensive test suite with pytest
- 📚 Created example notebooks and documentation

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

## Labeled LaTeX Display

For Jupyter notebooks, use `display_labeled_latex` to show equations with labels:

```python
from matkit import display_labeled_latex
import numpy as np

# NumPy arrays with precision control
A = np.array([[1.123, 2.456], [3.789, 4.012]])
display_labeled_latex(r"\mathbf{A} = ", A, precision=2)

# Control row spacing with arrayStretch
display_labeled_latex(r"\mathbf{A} = ", A, precision=2, arrayStretch=2.0)
```

**SymPy Support** - Automatically detects and formats symbolic expressions:

```python
from sympy import symbols, cos, sin, Matrix

theta = symbols('theta')
R = Matrix([
    [cos(theta), -sin(theta)],
    [sin(theta),  cos(theta)]
])

# Renders with proper symbolic notation (cos(θ), sin(θ))
display_labeled_latex(r"\mathbf{R}(\theta) = ", R, arrayStretch=1.8)
```

**Parameters:**
- `label`: LaTeX string for the label (e.g., `r"\mathbf{A} = "`)
- `array`: NumPy array or SymPy expression
- `precision`: Decimal places for NumPy arrays (default: 2)
- `arrayStretch`: Vertical row spacing multiplier (default: 1.5)
- `show_shape`: Display array shape as subscript on matrix (default: False)

**Example with shape display:**
```python
U = np.array([[0, 0], [2, -7.2], [1.6, -7.6], [0, 0]])
display_labeled_latex("U = ", U, show_shape=True)
# Displays: U = [matrix]_{4 × 2}
```

See `examples/notebooks/display_labeled_latex_examples.ipynb` for comprehensive examples.

## Features

- **Automatic LaTeX rendering** for NumPy arrays in marimo notebooks
- **Labeled equations** with `display_labeled_latex` for Jupyter notebooks
- **SymPy integration** - Automatic detection and proper symbolic formatting
- **Scalars, vectors (1D), and matrices (2D)** are all supported
- **Intelligent truncation** for large arrays (>8 elements/dimension)
- **Clean pipe syntax** for readable code: `array | la`
- **Format preservation** for integers, floats (customizable decimals), and complex numbers
- **Adjustable spacing** with `arrayStretch` parameter for better readability

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

## Visualization: MATLAB-Style `patch()` Function

MatKit includes a `patch()` function that mimics MATLAB's patch behavior for mesh visualization. This is particularly useful for visualizing FEM meshes with field data.

### Basic Usage

```python
import numpy as np
from matkit import patch
import matplotlib.pyplot as plt

# Define mesh using Faces/Vertices notation
vertices = np.array([[0, 0], [1, 0], [0.5, 0.866]])
faces = np.array([[1, 2, 3]])  # 1-based indexing

fig, ax = plt.subplots()
patch('Faces', faces, 'Vertices', vertices,
      'FaceColor', 'cyan',
      'EdgeColor', 'black',
      ax=ax)
ax.axis('equal')
plt.show()
```

### Key Features

- **Faces/Vertices notation**: Primary interface matching MATLAB syntax
- **1-based indexing**: Auto-detects and converts (faces starting at 1)
- **Per-vertex color interpolation**: Smooth gradients using `FaceColor='interp'`
- **Per-element flat colors**: Single color per element using `FaceColor='flat'`
- **Line elements**: 2-vertex faces for truss/beam visualization
- **Surface elements**: Triangles, quads in 2D/3D
- **Transparency**: `FaceAlpha` and `EdgeAlpha` support
- **Colormap support**: Any matplotlib colormap via `cmap` parameter

### Color Data Examples

**Per-vertex interpolation (smooth gradients):**
```python
# Temperature field at nodes
node_temps = np.array([20.0, 100.0, 60.0, 80.0])

patch('Faces', faces, 'Vertices', vertices,
      'FaceVertexCData', node_temps,
      'FaceColor', 'interp',
      'EdgeColor', 'black',
      'cmap', 'hot',
      ax=ax)
```

**Per-element flat colors:**
```python
# Stress in each element
element_stress = np.array([150, -80, 200, -120])

patch('Faces', faces, 'Vertices', vertices,
      'FaceVertexCData', element_stress,
      'FaceColor', 'flat',
      'EdgeColor', 'black',
      'cmap', 'RdBu_r',  # Red=tension, Blue=compression
      ax=ax)
```

### Truss Visualization

```python
# Define truss nodes and connectivity
P = np.array([[0, 0], [500, 0], [300, 300], [600, 300]])
edges = np.array([[1, 2], [1, 3], [2, 3], [2, 4], [3, 4]])

# Visualize with element forces
forces = np.array([6052.76, -5582.25, -7274.51, 6380.16, -9912.07])

fig, ax = plt.subplots()
patch('Faces', edges, 'Vertices', P,
      'FaceVertexCData', forces,
      'FaceColor', 'flat',
      'LineWidth', 3,
      'cmap', 'RdBu_r',
      ax=ax)
ax.axis('equal')
plt.colorbar(ax.collections[0], ax=ax, label='Force (N)')
```

### Displacement Field Visualization

```python
# Original mesh
P = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
faces = np.array([[1, 2, 3, 4]])

# Compute displacements (from FEM analysis)
U = np.array([[0, 0], [0.1, 0], [0.15, -0.05], [0.05, -0.02]])
U_mag = np.sqrt(np.sum(U**2, axis=1))

# Plot deformed mesh with color-coded displacement magnitude
P_deformed = P + U * scale_factor
patch('Faces', faces, 'Vertices', P_deformed,
      'FaceVertexCData', U_mag,
      'FaceColor', 'interp',
      'EdgeColor', 'black',
      'cmap', 'jet',
      ax=ax)
plt.colorbar(ax.collections[0], label='Displacement')
```

### 3D Support

```python
# 3D truss
P_3d = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0], [0.5, 0.433, 0.8]])
edges_3d = np.array([[1, 2], [1, 3], [2, 3], [1, 4], [2, 4], [3, 4]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
patch('Faces', edges_3d, 'Vertices', P_3d,
      'LineWidth', 2,
      'cmap', 'viridis',
      ax=ax)
```

### Implementation Details

- Uses `matplotlib.tri.Triangulation` with Gouraud shading for true vertex color interpolation
- Automatically subdivides quads into triangles for smooth color gradients
- Draws edges separately as `LineCollection` to respect `EdgeColor` with interpolated faces
- Supports 2D (`LineCollection`, `PolyCollection`) and 3D (`Line3DCollection`, `Poly3DCollection`)

See `examples/patch_demo.py` and `examples/patch_advanced_demo.py` for comprehensive examples.

## Repository Structure

```
MatKit/
├── matkit/              # Source code
│   ├── __init__.py
│   ├── latex_array.py  # LaTeX rendering (la, display_labeled_latex)
│   ├── mesh.py         # FEM mesh with 1-based indexing
│   ├── patch.py        # MATLAB-style patch visualization
│   └── one_array.py    # 1-based array wrapper
├── tests/               # Automated tests (pytest)
│   ├── test_latex_array.py
│   ├── test_field_management.py
│   └── test_seamless.py
├── examples/            # User-facing examples
│   ├── notebooks/      # Jupyter notebook tutorials
│   ├── patch_demo.py
│   └── patch_advanced_demo.py
├── dev/                 # Development notes
└── docs/                # Documentation (if needed)
```

## Testing

Run automated tests using pytest:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=matkit --cov-report=html

# Run specific tests
pytest tests/test_latex_array.py -v
```

See `tests/README.md` for detailed testing documentation.

## Examples

Learn by example:

```bash
# Jupyter notebooks (recommended)
jupyter notebook examples/notebooks/display_labeled_latex_examples.ipynb

# Python scripts
python examples/patch_demo.py
```

See `examples/README.md` for all available examples.

### Tests vs Examples

- **Tests** (`tests/`): Automated correctness checks with assertions, run in CI/CD
- **Examples** (`examples/`): User-facing demonstrations showing real-world usage

Both are important:
- Tests ensure correctness and catch regressions
- Examples teach usage patterns and provide context for AI-assisted development

## Development

Install in editable mode for local development:

```bash
uv pip install -e .
```

This allows editing the source code and having changes immediately reflected in your environment.