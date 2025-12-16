"""Type stub file for mechanicskit - helps VS Code autocomplete."""

from typing import Any, Optional, Callable, Union, List, Dict
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Version
__version__: str

# Functions
def patch(
    *args,
    ax: Optional[Axes] = None,
    return_mappable: bool = False,
    Faces: Optional[np.ndarray] = None,
    Vertices: Optional[np.ndarray] = None,
    CData: Optional[np.ndarray] = None,
    FaceVertexCData: Optional[np.ndarray] = None,
    FaceColor: str = 'flat',
    FaceAlpha: Union[float, np.ndarray] = 1.0,
    EdgeColor: Union[str, tuple] = 'black',
    LineWidth: float = 1.0,
    LineStyle: str = '-',
    EdgeAlpha: float = 1.0,
    cmap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    Shading: str = 'flat',
    interpolation_method: str = 'auto',
    **kwargs
) -> Any: ...

def fplot(
    *args,
    range: tuple = (-5, 5),
    ax: Optional[Axes] = None,
    npoints: int = 100,
    **kwargs
) -> Any: ...

def colorbar(
    cmap: Optional[str] = None,
    limits: Optional[tuple] = None,
    clims: Optional[tuple] = None,
    ax: Optional[Axes] = None,
    label: Optional[str] = None,
    ticks: Optional[List] = None,
    orientation: str = 'vertical',
    extend: str = 'neither',
    format: Optional[str] = None,
    **kwargs
) -> Any: ...

def cmap(name: str) -> Any: ...

def version() -> str: ...

def quick_ref() -> None: ...

def display_labeled_latex(*args, **kwargs) -> Any: ...

# Classes
class LatexArray:
    def __init__(self, data: np.ndarray, name: str = 'A'): ...

class Mesh:
    def __init__(self, nodes: np.ndarray, elements: np.ndarray, element_type: str): ...

class OneArray:
    def __init__(self, data: np.ndarray): ...

# Objects
la: LatexArray

# Constants
ELEMENT_TYPES: Dict[str, Any]

# Submodules (should not autocomplete, but included for completeness)
colormap_utils: Any
help: Any
latex_array: Any
mesh: Any
one_array: Any
