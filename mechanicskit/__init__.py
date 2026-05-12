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
# labeled is now an alias for ltx/latex_expression

# Apply Computer Modern math fonts to matplotlib globally on import
try:
    import matplotlib.pyplot as _plt
    _plt.rcParams['mathtext.fontset'] = 'cm'
    _plt.rcParams['font.family'] = 'serif'
except ImportError:
    pass
from .mesh import Mesh, ELEMENT_TYPES
from .one_array import OneArray
from .patch import patch
from .fplot import fplot
from .arrow import arrow
from .gaussint import gaussint
from .colormap_utils import colorbar, cmap
from .help import quick_ref
from .animation_utils import to_responsive_html
from .markdown import md, Markdown

# Version information
__version__ = '0.7.2'

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
    'MechanicsKit v0.5.0'
    >>> mk.__version__
    '0.5.0'
    """
    return f'MechanicsKit v{__version__}'


def _check_version(current_version=None):
    """
    Check GitHub for the latest release and warn if the installed version
    is older. Called in a background thread on import so it never blocks.

    Parameters
    ----------
    current_version : str, optional
        Version to compare against (default: __version__).
        Accepts a custom value for testing.
    """
    import json
    import urllib.request
    import warnings
    import time
    from pathlib import Path

    if current_version is None:
        current_version = __version__

    # Cache: check at most once per day
    cache_dir = Path.home() / '.cache' / 'mechanicskit'
    cache_file = cache_dir / 'version_check.json'
    now = time.time()
    try:
        if cache_file.exists():
            data = json.loads(cache_file.read_text())
            if now - data.get('timestamp', 0) < 86400:
                latest = data.get('latest')
                if latest:
                    _warn_if_outdated(current_version, latest)
                return
    except Exception:
        pass

    # Fetch latest release from GitHub
    try:
        url = "https://api.github.com/repos/cenmir/MechanicsKit/releases/latest"
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})
        with urllib.request.urlopen(req, timeout=3) as r:
            latest = json.loads(r.read())["tag_name"].lstrip("v")
    except Exception:
        return

    # Cache the result
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"timestamp": now, "latest": latest}))
    except Exception:
        pass

    _warn_if_outdated(current_version, latest)


def _warn_if_outdated(current_version, latest):
    """Warn if current_version < latest."""
    import warnings
    try:
        from packaging.version import Version
        if Version(current_version) < Version(latest):
            warnings.warn(
                f"\n\nmechanicskit {current_version} is installed, "
                f"but {latest} is available.\n"
                f"Run: uv pip install -U git+https://github.com/cenmir/MechanicsKit.git\n",
                stacklevel=4,
            )
    except Exception:
        pass


# Run version check in background thread
import threading as _threading
_threading.Thread(target=_check_version, daemon=True).start()

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
    'arrow',
    'quick_ref',
    'to_responsive_html',
    'md', 'Markdown',
]
