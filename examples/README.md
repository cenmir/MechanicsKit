# MatKit Examples

This directory contains examples demonstrating how to use MatKit features.

## Directory Structure

```
examples/
├── notebooks/              # Jupyter notebook examples
│   ├── display_labeled_latex_examples.ipynb
│   └── example1.ipynb
├── patch_demo.py          # Basic patch visualization examples
└── patch_advanced_demo.py # Advanced patch visualization examples
```

## Running Examples

### Jupyter Notebooks

The `notebooks/` directory contains interactive examples that are best viewed in Jupyter:

```bash
# Install Jupyter if needed
pip install jupyter

# Start Jupyter and open a notebook
jupyter notebook examples/notebooks/display_labeled_latex_examples.ipynb
```

### Python Scripts

Run the Python examples directly:

```bash
# Basic patch visualization
python examples/patch_demo.py

# Advanced patch features
python examples/patch_advanced_demo.py
```

## Example Descriptions

### `notebooks/display_labeled_latex_examples.ipynb`

Comprehensive examples of the `display_labeled_latex` function showing:
- Scalars, vectors, and matrices
- Precision control
- Row spacing adjustment with `arrayStretch`
- SymPy symbolic expressions
- Complex numbers
- Real-world FEM applications

### `notebooks/example1.ipynb`

Original MatKit example notebook demonstrating core features.

### `patch_demo.py`

Basic truss visualization examples using the `patch` function:
- Uniform color visualization
- Element-wise coloring
- Displacement visualization
- Node labeling

### `patch_advanced_demo.py`

Advanced visualization features:
- 3D surface plots
- Interpolated fields
- Transparency and alpha values
- Multiple scales and transformations

## Creating New Examples

When adding examples:

1. **For tutorials**: Use Jupyter notebooks in `examples/notebooks/`
2. **For code snippets**: Use Python scripts in `examples/`
3. **Include docstrings**: Explain what the example demonstrates
4. **Use realistic data**: Show practical use cases
5. **Keep it simple**: Focus on one feature at a time
6. **Add comments**: Explain non-obvious steps

Example structure:

```python
"""
Title: Brief Description
========================

Longer description of what this example demonstrates.
"""

# Step 1: Setup
import numpy as np
from matkit import display_labeled_latex

# Step 2: Create data
data = np.array([1, 2, 3])

# Step 3: Display
display_labeled_latex(r"\\mathbf{v} = ", data)
```

## Difference from Tests

Examples differ from tests:

- **Purpose**: Teaching and demonstration vs. correctness verification
- **Focus**: Clarity and real-world usage vs. edge cases
- **Output**: Visual results and explanations vs. pass/fail assertions
- **Audience**: End users and learners vs. developers

See the `tests/` directory for automated correctness tests.
