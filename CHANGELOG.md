# Changelog

All notable changes to MechanicsKit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## Version Update Guidelines

**When to bump versions:**
- **PATCH (0.1.X)**: Bug fixes, documentation updates, performance improvements
- **MINOR (0.X.0)**: New features, new functions, new element types, backward-compatible changes
- **MAJOR (X.0.0)**: Breaking changes, API changes that require user code updates (rare for educational packages)

**When to update `__version__` in `__init__.py`:**
- After fixing bugs (increment patch)
- After adding features (increment minor)
- When preparing a release/tag (not on every commit)

**Claude will remind you** when changes warrant a version bump!

---

## [Unreleased]

### Added
- **Intelligent color interpolation for 2D surface patches**
  - `patch()` now automatically selects between `tricontourf` and `tripcolor` for optimal visual quality
  - New `interpolation_method='auto'` parameter (default) intelligently chooses rendering method:
    - Uses `tricontourf` (256 levels) for small meshes (<100 elements) with non-linear colormaps (jet, hsv, hot, etc.)
    - Uses `tripcolor` (Gouraud shading) for large meshes or linear colormaps (viridis, plasma, etc.)
  - Manual override available: `interpolation_method='tricontourf'` or `'tripcolor'`
  - Non-linear colormap detection for 30+ common colormaps
  - Eliminates RGB interpolation artifacts with non-linear colormaps
  - Maintains performance for large FEM meshes while ensuring quality for instructional plots

- **Enhanced `colorbar()` function with automatic parameter retrieval**
  - Zero-argument usage: `mk.colorbar()` automatically retrieves colormap and limits from last `patch()` call
  - Eliminates need for manual `ScalarMappable` creation and parameter repetition
  - **Automatic discrete/continuous colorbar selection**:
    - Flat colors (per-element data): Creates discrete colorbar with distinct color bands using `BoundaryNorm`
    - Interpolated colors (FaceColor='interp'): Creates continuous colorbar with smooth gradients
    - Discrete colorbar boundaries are automatically placed at midpoints between unique data values
    - Perfectly matches the actual element colors shown in the patch
  - Supports both `limits=[vmin, vmax]` and `clims=[vmin, vmax]` (MATLAB compatibility)
  - New parameters: `ticks`, `orientation`, `extend`, `format` for complete control
  - Backward compatible with existing code
  - Example: `mk.patch(..., 'cmap', 'jet')` then `mk.colorbar()` - automatically uses jet!

- `patch()` now supports `return_mappable=True` parameter for easy colorbar creation
  - Returns tuple `(collection, mappable)` when `FaceColor='interp'` is used
  - Eliminates need for manual ScalarMappable/Normalize boilerplate
  - Example: `collection, mappable = mk.patch(..., return_mappable=True)` then `plt.colorbar(mappable)`

### Changed
- **Breaking improvement**: When `FaceColor='interp'`, smooth interpolation is always used
  - The `Shading` parameter is now only relevant for non-interpolated rendering
  - `tripcolor` always uses `shading='gouraud'` for interpolation mode
  - `tricontourf` always uses 256 levels for smooth gradients

- `patch()` internal state storage enhanced
  - Now stores `cmap`, `vmin`, `vmax` for automatic colorbar retrieval
  - Updated `_store_patch_state()` signature to include colormap information

### Fixed
- **CRITICAL**: Fixed missing `_create_quad_with_pcolormesh()` function
  - Removed broken reference that caused crashes
  - Replaced with proper `tricontourf`/`tripcolor` implementation
  - All 2D surface interpolation now works correctly

- **Fixed `EdgeAlpha` parameter implementation**
  - `EdgeAlpha` was defined but not properly applied in all patch modes
  - Now correctly controls edge transparency independently of `FaceAlpha`
  - Enhanced `_apply_alpha()` helper to handle both color arrays and single colors
  - Works with all patch types: 2D surfaces, 3D surfaces, and line elements
  - Example: `mk.patch(..., 'EdgeColor', 'red', 'EdgeAlpha', 0.3)` creates semi-transparent edges

- **Fixed crash when all vertex values are identical** (vmin == vmax edge case)
  - `tricontourf` requires strictly increasing contour levels
  - When all values are the same, `np.linspace(vmin, vmax, 256)` creates constant levels
  - Now adds small epsilon (1% of value or 1.0 if zero) to create valid range
  - Prevents `ValueError: "Contour levels must be increasing"` crash
  - Example: mesh with uniform temperature field no longer crashes

## [0.1.1] - 2025-12-02

### Added
- Version information: `__version__` attribute and `version()` function
- `display_labeled_latex()` function for displaying labeled LaTeX equations in Jupyter notebooks
  - Supports both NumPy arrays and SymPy symbolic expressions
  - Automatic detection of SymPy objects for proper symbolic formatting
  - `precision` parameter for controlling decimal places in NumPy arrays
  - `arrayStretch` parameter for adjusting vertical row spacing (default: 1.5)
- SymPy integration with automatic `latex()` conversion
  - Detects SymPy matrices, symbols, and expressions
  - Renders symbolic expressions correctly (e.g., `\cos(\theta)` instead of `\cos(\text{theta})`)
- Comprehensive test suite using pytest
  - Tests for `LatexArray` class
  - Tests for `display_labeled_latex` function
  - Tests for SymPy integration
  - Tests for complex number handling
- Example notebooks in `examples/notebooks/`
  - `display_labeled_latex_examples.ipynb` with 8 comprehensive examples
- Documentation improvements
  - `tests/README.md` for testing guidelines
  - `examples/README.md` for example usage
  - Updated main README with new features and structure
  - Version update guidelines in CHANGELOG.md

### Changed
- Reorganized repository structure to follow Python standards
  - Moved all tests to `tests/` directory
  - Moved all examples to `examples/` directory
  - Separated Jupyter notebooks into `examples/notebooks/`
  - Renamed example files from `test_*.py` to `*_demo.py` for clarity
- Fixed `display_labeled_latex` to properly handle SymPy objects
  - Changed from `isinstance(array, Basic)` to module name checking for compatibility
  - Uses `\def\arraystretch` instead of `\renewcommand` for KaTeX compatibility

### Fixed
- **CRITICAL**: `patch()` function now properly interpolates colors for line elements
  - Previously only averaged vertex colors instead of true interpolation
  - Now subdivides each line element into 100 segments for smooth color gradients
  - Fixes visualization of FEM results like displacement/stress fields on truss elements
- SymPy matrix display now shows proper LaTeX formatting (e.g., `\cos(\theta)` not `cos(theta)`)
- KaTeX parse errors with arraystretch command resolved

## [0.1.0] - Earlier

### Added
- Initial release with core features:
  - `LatexArray` class for LaTeX rendering in marimo notebooks
  - `la` renderer with pipe syntax support
  - `Mesh` class with 1-based indexing for FEM
  - MATLAB-style `patch()` function for visualization
  - `OneArray` for 1-based array indexing
