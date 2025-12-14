"""
MATLAB-style colormap utilities for automatic colorbar creation.

Provides convenient functions for adding colorbars to patch visualizations
without manual ScalarMappable creation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# Module-level storage for last patch state
_last_patch_state = {
    'collection': None,
    'cdata': None,
    'ax': None,
    'faces': None,
    'vertices': None,
    'mode': None,
    'n_segments_per_edge': None,
}


def _store_patch_state(collection, cdata, ax, faces=None, vertices=None, mode=None, n_segments_per_edge=None):
    """
    Store patch state for later colorbar creation.

    Called internally by patch() function.
    """
    global _last_patch_state
    _last_patch_state = {
        'collection': collection,
        'cdata': np.asarray(cdata),
        'ax': ax,
        'faces': faces,
        'vertices': vertices,
        'mode': mode,
        'n_segments_per_edge': n_segments_per_edge,
    }


def _generate_interpolated_line_colors_and_segments(CData, Faces, Vertices, n_segments_per_edge, cmap, norm):
    """Generate segments and colors for smooth line interpolation."""
    segments = []
    segment_colors = []
    for face in Faces:
        v1_pos = Vertices[face[0]]
        v2_pos = Vertices[face[1]]
        v1_color = CData[face[0]]
        v2_color = CData[face[1]]
        for i in range(n_segments_per_edge):
            t1 = i / n_segments_per_edge
            t2 = (i + 1) / n_segments_per_edge
            p1 = v1_pos * (1 - t1) + v2_pos * t1
            p2 = v1_pos * (1 - t2) + v2_pos * t2
            segments.append([p1, p2])
            t_mid = (t1 + t2) / 2
            c_mid = v1_color * (1 - t_mid) + v2_color * t_mid
            segment_colors.append(cmap(norm(c_mid)))
    return np.array(segments), np.array(segment_colors)


def colorbar(cmap_name='viridis', label=None, ax=None, vmin=None, vmax=None, **kwargs):
    """
    Apply colormap and add colorbar to the last created patch.

    MATLAB-style convenience function that applies a colormap to the most
    recent patch() call and adds a colorbar. Eliminates the need for manual
    ScalarMappable creation and colormap specification in patch().

    Parameters
    ----------
    cmap_name : str, default 'viridis'
        Colormap name. Common values: 'jet', 'viridis', 'plasma', 'coolwarm',
        'RdBu_r', 'seismic', 'rainbow'.
    label : str, optional
        Colorbar label text.
    ax : matplotlib.axes.Axes, optional
        Axes for colorbar placement. If None, uses axes from last patch().
    vmin, vmax : float, optional
        Color normalization limits. If None, uses data min/max.
    **kwargs : dict
        Additional keyword arguments passed to plt.colorbar():
        - shrink : float, default 1.0 (fraction of original size)
        - orientation : 'vertical' or 'horizontal'
        - pad : float, spacing from axes
        - aspect : float, ratio of long to short dimensions

    Returns
    -------
    cbar : matplotlib.colorbar.Colorbar
        The created colorbar object.

    Raises
    ------
    RuntimeError
        If patch() has not been called before this function.

    Examples
    --------
    Basic usage with element forces (default viridis):

    >>> import mechanicskit as mk
    >>> mk.patch('Faces', elements, 'Vertices', nodes, 'FaceVertexCData', forces)
    >>> mk.colorbar(label='Force [N]')

    Specify colormap:

    >>> mk.patch('Faces', elements, 'Vertices', nodes, 'FaceVertexCData', stress)
    >>> mk.colorbar('jet', label='Stress [MPa]')

    Custom range and horizontal colorbar:

    >>> mk.patch('Faces', elements, 'Vertices', nodes, 'FaceVertexCData', temps)
    >>> mk.colorbar('jet', vmin=0, vmax=100, orientation='horizontal', shrink=0.8)

    Interpolated nodal temperatures:

    >>> mk.patch('Faces', elements, 'Vertices', nodes,
    ...          'FaceVertexCData', nodal_temps, 'FaceColor', 'interp')
    >>> mk.colorbar('jet', label='Temperature [°C]')

    See Also
    --------
    patch : Create colored patches
    cmap : Alias for colorbar (MATLAB-compatible name)
    matplotlib.pyplot.colorbar : Underlying colorbar function
    """
    global _last_patch_state

    # Validate patch was called
    if _last_patch_state['cdata'] is None:
        raise RuntimeError(
            "No patch data found. Call patch() before colorbar()."
        )

    # Get stored state
    state = _last_patch_state
    collection = state['collection']
    cdata = state['cdata']
    ax_to_use = ax if ax is not None else state['ax']

    # Compute limits if not provided
    if vmin is None:
        vmin = cdata.min()
    if vmax is None:
        vmax = cdata.max()

    # Create normalization and colormap
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = cm.get_cmap(cmap_name)

    # Update colors on the existing collection based on its type
    if state.get('mode') == 'interp':
        if 'LineCollection' in collection.__class__.__name__:
            # Case 1: Interpolated LineCollection
            n_segments = state.get('n_segments_per_edge', 100)
            _, colors = _generate_interpolated_line_colors_and_segments(
                cdata, state['faces'], state['vertices'], n_segments, cmap_obj, norm
            )
            collection.set_colors(colors)
        elif hasattr(collection, 'set_cmap'):
            # Case 2: Interpolated surface (e.g., from tripcolor), which is a mappable
            collection.set_cmap(cmap_obj)
            collection.set_norm(norm)
    else:
        # Case 3: Flat colors
        colors = cmap_obj(norm(cdata))
        if hasattr(collection, 'set_facecolors'):
            # For PolyCollection
            collection.set_facecolors(colors)
        else:
            # For LineCollection with flat colors
            collection.set_colors(colors)

    # Create ScalarMappable for the colorbar itself
    mappable = cm.ScalarMappable(cmap=cmap_name, norm=norm)
    mappable.set_array([])

    # Create colorbar
    cbar = plt.colorbar(mappable, ax=ax_to_use, **kwargs)

    # Set label if provided
    if label is not None:
        cbar.set_label(label)

    return cbar


# MATLAB-compatible alias
cmap = colorbar
