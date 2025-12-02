# Changelog

All notable changes to MatKit will be documented in this file.

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
- `patch()` now supports `return_mappable=True` parameter for easy colorbar creation
  - Returns tuple `(collection, mappable)` when `FaceColor='interp'` is used
  - Eliminates need for manual ScalarMappable/Normalize boilerplate
  - Example: `collection, mappable = mk.patch(..., return_mappable=True)` then `plt.colorbar(mappable)`

### Changed

### Fixed

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
