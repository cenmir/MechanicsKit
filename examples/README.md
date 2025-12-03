# MatKit Examples

Interactive Jupyter notebook examples demonstrating MatKit features for finite element analysis.

## Quick Start

```bash
# Install Jupyter if needed
pip install jupyter

# Start Jupyter
cd MatKit/examples/notebooks
jupyter notebook
```

## Available Notebooks

### [examples_patch.ipynb](notebooks/examples_patch.ipynb)
Comprehensive guide to the `patch()` function for visualizing FEM meshes:
- Basic truss visualization
- Element-wise and nodal coloring (flat vs interpolated)
- 2D and 3D elements (lines, triangles, quads)
- Displacement fields and deformed shapes
- Stress and temperature field visualization
- Transparency and styling options

**Key features**: FaceColor modes, colormaps, transparency, multi-scale comparisons

### [examples_mesh_onearray.ipynb](notebooks/examples_mesh_onearray.ipynb)
Working with `Mesh` and `OneArray` for natural 1-based indexing:
- OneArray basics (1-based element/node result arrays)
- Creating and querying FEM meshes
- Element types (ROD, TRIA3, QUAD4)
- DOF management and iteration
- Complete truss analysis workflow
- Integration with patch() for visualization

**Key features**: 1-based indexing, natural iteration, DOF mapping

### [display_labeled_latex_examples.ipynb](notebooks/display_labeled_latex_examples.ipynb)
Displaying mathematical expressions with LaTeX formatting:
- Scalars, vectors, and matrices
- Precision control and row spacing
- SymPy symbolic expressions
- Complex numbers
- Array shape annotations
- FEM-specific examples (stiffness matrices, transformations)

**Key features**: LaTeX rendering, precision control, SymPy integration

## Topics Covered

| Topic | Notebook |
|-------|----------|
| Mesh visualization | `examples_patch.ipynb` |
| Element/nodal fields | `examples_patch.ipynb`, `examples_mesh_onearray.ipynb` |
| 1-based indexing | `examples_mesh_onearray.ipynb` |
| LaTeX output | `display_labeled_latex_examples.ipynb` |
| Truss analysis | `examples_patch.ipynb`, `examples_mesh_onearray.ipynb` |
| Displacement plots | `examples_patch.ipynb` |

## Learning Path

1. **New to MatKit?** Start with `display_labeled_latex_examples.ipynb` for basic output
2. **FEM visualization?** See `examples_patch.ipynb` for plotting meshes and fields
3. **Building FEM code?** Use `examples_mesh_onearray.ipynb` for mesh management

## Need Help?

- Check the [main README](../README.md) for installation and API reference
- See the [tests directory](../tests/) for additional usage patterns
- Report issues at the repository issue tracker
