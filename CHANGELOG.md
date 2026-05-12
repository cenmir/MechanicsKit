# Changelog

All notable changes to MechanicsKit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## Version Update Guidelines

While in the 0.x series we deliberately move the minor slot slowly so the
version number stays meaningful.

**When to bump versions (pre-1.0 policy):**
- **PATCH (0.X.Y → 0.X.Y+1)**: Bug fixes, documentation, performance, and
  small additive refinements to existing functions (new kwargs, new optional
  behavior). Most changes land here.
- **MINOR (0.X.0 → 0.X+1.0)**: New public functions, new user-visible verbs,
  or meaningful reshuffles of existing ones.
- **MAJOR (X.0.0)**: Breaking changes that require user code updates.
  Reserved for the 1.0 API commitment.

**Rule of thumb:** if the change only adds an option to something that
already exists, it's a patch. If it adds a thing the user can import or
call that didn't exist before, it's a minor bump.

**When to update `__version__` in `__init__.py`:**
- Bump before `git push` (not on every commit) so each pushed state maps
  to a distinct, installable version.
- Keep `pyproject.toml` and `mechanicskit/__init__.py` in lockstep.

A ``pre-push`` git hook prints this policy as a reminder at push time;
see the top of the changelog entry for the change being pushed and
decide patch vs. minor before pushing.

---

## [0.7.3] - 2026-05-12

### Changed
- **`display_labeled_latex_examples.ipynb` rewritten and fixed for
  portable MathJax rendering.** Previous outputs (shipped in v0.7.2)
  rendered as raw LaTeX on GitHub because:
  - Single-row labels used `&=`, which is only valid inside an
    `aligned` environment. Replaced with plain `=`; `&=` is now used
    only in the `aligned=True` example.
  - The "chainable verbs" example piped twice (`A | la.foo() | la.bar()`),
    which re-wraps a `LatexArray` and falls through to `str()`. Replaced
    with `A | la.arraystretch(2.0).shape()` — chain on the renderer.
  - The `aligned=True` example used `\bm`, which is not loaded by
    MathJax in many viewers (including GitHub). Replaced with
    `\boldsymbol`.
- Added a new section demonstrating the `wrap=True` / `wrap=N` option
  introduced in 0.7.1, mirroring the demo notebook at
  `mechanics/test.ipynb`.

## [0.7.2] - 2026-05-12

### Changed
- Re-executed `examples/notebooks/display_labeled_latex_examples.ipynb`
  so the rendered outputs are present in the file. The previous v0.7.0
  commit shipped the rewritten notebook without running it, so GitHub
  showed empty output cells.

## [0.7.1] - 2026-05-12

### Added
- **`wrap` option on `la` / `ltx`.** Breaks long SymPy `Add` expressions
  across multiple aligned lines so they fit within page width in both
  HTML (MathJax) and PDF (LaTeX).
  - `la(expr, wrap=True)` — one summand per line.
  - `la(expr, wrap=N)` — pack `N` summands per line.
  - Pipe form: `expr | la.wrap()` or `expr | la.wrap(N)`.
  - `ltx(r"f &=", expr, wrap=True)` honors the user-supplied label and
    appends continuation lines as `&\quad + ...`.
  - For `sp.Eq(lhs, rhs)`, the RHS is broken and continuation lines align
    after the `=` sign.
  - Short expressions whose number of summands does not exceed the
    per-line count are left untouched, so opting in is safe.

## [0.7.0] - 2026-04-26

### Added
- **`mk.md(template)` — type-aware markdown formatter.** Interpolates
  variables from the calling cell's namespace (no `f` prefix) and
  dispatches on type before producing the final markdown source:
  - `LatexArray`, `LatexExpression`, and any object with `_repr_latex_`
    pass through their LaTeX (already `$$…$$`-wrapped).
  - 2-D `ndarray` ≤ 8×8 → markdown table.
  - 2-D `ndarray` > 8×8 → truncated LaTeX `bmatrix` (matches `LatexArray`).
  - 1-D `ndarray` → comma-separated inline.
  - `pandas.DataFrame` (lazy registration) → `df.to_markdown()`.
  - `list[dict]` / `list[list]` → markdown table.
  - Other types fall through to `str(value)` (matches f-string behavior).
- Output is **markdown source**, not pre-rendered HTML. The returned
  `Markdown` object exposes `_repr_markdown_` (Jupyter, quarto),
  `_mime_("text/markdown", …)` (marimo), and a `__format__` that returns
  markdown source so nested `mk.md` calls compose under f-string
  interpolation.
- Arithmetic and method calls work inside `{...}` (e.g. `{F_st/1000:.2f}`,
  `{obj.method().attr}`) — `mk.md` evaluates the placeholder text as a
  Python expression in the caller's scope.
- Standard Python format specs honored (`{x:.2f}`, `{x:>10}`).
- Variables can also be passed as keyword arguments; kwargs override the
  caller-frame lookup.

### Changed
- New hard dependency: `tabulate>=0.9` (used by the table-rendering paths
  for ndarray, `list[list]`, `list[dict]`, and DataFrames).

## [0.6.2] - 2026-05-03

### Added
- `to_responsive_html(..., default_mode='reflect'|'once'|'loop')` exposes
  the JS player's mode dropdown so callers can preselect ping-pong
  ('reflect') or single-shot ('once') playback. Default unchanged ('loop').

## [0.6.1] - 2026-04-20

### Added
- `ltx(..., aligned=True)` wraps the composed expression in
  `\begin{aligned}...\end{aligned}` for left-aligned multi-row equations.
  Use `&` to mark the alignment column and `\\` to separate rows.
- `la.shape()` chainable verb — `A | la.shape()` appends an `_{m \times n}`
  subscript to the rendered matrix/vector. Sibling of the existing
  `show_shape=True` kwarg.
- `show_shape` is now honored by SymPy rendering and by the 1-D and 2-D
  NumPy branches (previously plumbed through but not applied in all
  branches).

### Changed
- `LatexExpression` (`ltx`) `precision` default changed from `2` to
  `None` (full precision). When set, `precision=n` now also calls
  `evalf(n)` on SymPy values rather than just formatting NumPy floats to
  `n` decimals. Pass `precision=2` explicitly for the old behavior.

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
