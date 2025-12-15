"""
MATLAB-style colormap utilities for automatic colorbar creation.

Provides convenient functions for adding colorbars to patch visualizations
without manual ScalarMappable creation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, BoundaryNorm

# Module-level storage for last patch state
_last_patch_state = {
    'collection': None,
    'cdata': None,
    'ax': None,
    'faces': None,
    'vertices': None,
    'mode': None,
    'n_segments_per_edge': None,
    'cmap': None,
    'vmin': None,
    'vmax': None,
}


def _store_patch_state(collection, cdata, ax, faces=None, vertices=None, mode=None,
                       n_segments_per_edge=None, cmap=None, vmin=None, vmax=None):
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
        'cmap': cmap,
        'vmin': vmin,
        'vmax': vmax,
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


def colorbar(cmap=None, limits=None, clims=None, ax=None, label=None,
              ticks=None, orientation='vertical', extend='neither',
              format=None, **kwargs):
    """
    Create a colorbar for a patch plot.

    When called without arguments, automatically retrieves colormap and
    data limits from the most recent patch() call on the axes. This provides
    MATLAB-like convenience where you don't need to manually specify the
    colormap and limits again.

    The colorbar automatically adapts to the data type:
    - **Discrete colorbar**: For flat colors (per-element data), shows distinct
      color bands matching the actual element colors
    - **Continuous colorbar**: For interpolated data (FaceColor='interp'),
      shows smooth color gradient

    Parameters
    ----------
    cmap : str or Colormap, optional
        Colormap to use. If None, retrieves from last patch() call.
        Common values: 'jet', 'viridis', 'plasma', 'coolwarm', 'hot', 'rainbow'.
    limits : list/tuple of [vmin, vmax], optional
        Color normalization limits. If None, retrieves from last patch() call.
    clims : list/tuple of [vmin, vmax], optional
        Alias for `limits` (MATLAB compatibility). If both specified, `limits` takes precedence.
    ax : Axes, optional
        Axes for the colorbar. If None, uses axes from last patch() call.
    label : str, optional
        Colorbar label text.
    ticks : array-like, optional
        Colorbar tick positions.
    orientation : {'vertical', 'horizontal'}, default 'vertical'
        Colorbar orientation.
    extend : {'neither', 'both', 'min', 'max'}, default 'neither'
        Add extension arrows to colorbar ends to indicate out-of-range values.
    format : str or Formatter, optional
        Format string for tick labels (e.g., '%.2f', '%d', '%.1e').
    **kwargs
        Additional arguments passed to matplotlib's fig.colorbar()
        (e.g., shrink, pad, aspect, fraction).

    Returns
    -------
    colorbar : matplotlib.colorbar.Colorbar
        The created colorbar object, which can be further customized.

    Raises
    ------
    RuntimeError
        If patch() has not been called before this function.

    Examples
    --------
    Automatic (most common - zero arguments):

    >>> import mechanicskit as mk
    >>> mk.patch('Faces', F, 'Vertices', V, 'FaceVertexCData', T,
    ...          'FaceColor', 'interp', 'cmap', 'jet')
    >>> mk.colorbar()  # That's it! Auto-retrieves 'jet' and data limits

    With label:

    >>> mk.colorbar(label='Temperature (°C)')

    Override limits while keeping stored colormap:

    >>> mk.colorbar(limits=[0, 100])

    Override both colormap and limits:

    >>> mk.colorbar('viridis', [0, 50])

    Using clims (MATLAB-style):

    >>> mk.colorbar(clims=[20, 40], label='Stress [MPa]')

    Custom ticks and formatting:

    >>> mk.colorbar(label='Temperature', ticks=np.arange(0, 101, 20), format='%d°C')

    Horizontal colorbar:

    >>> mk.colorbar(orientation='horizontal', label='Force [N]')

    Discrete colorbar for flat colors (per-element data):

    >>> mk.patch('Faces', elements, 'Vertices', nodes, 'FaceVertexCData', element_stresses)
    >>> mk.colorbar(label='Element Stress [MPa]')  # Automatically shows discrete color bands

    See Also
    --------
    patch : Create colored patches
    matplotlib.pyplot.colorbar : Underlying colorbar function

    Notes
    -----
    This function stores state from the last patch() call, including the colormap,
    normalization limits, and color data. This eliminates the need to manually
    create ScalarMappable objects or repeat colormap specifications.

    The colorbar automatically detects whether to use discrete or continuous
    rendering based on the patch's FaceColor mode:

    - **Flat colors** (FaceColor='flat' or per-element data): Creates a discrete
      colorbar using BoundaryNorm with boundaries at midpoints between unique
      data values. This ensures the colorbar perfectly matches the discrete
      element colors shown in the patch.

    - **Interpolated colors** (FaceColor='interp'): Creates a continuous colorbar
      using standard Normalize for smooth color gradients.
    """
    global _last_patch_state

    # Validate patch was called
    if _last_patch_state['cdata'] is None:
        raise RuntimeError(
            "No patch data found. Call patch() before colorbar().\n"
            "Example: mk.patch('Faces', F, 'Vertices', V, 'FaceVertexCData', data)"
        )

    # Get stored state
    state = _last_patch_state
    collection = state['collection']
    cdata = state['cdata']
    mode = state.get('mode')

    # Resolve axes
    ax_to_use = ax if ax is not None else state['ax']
    if ax_to_use is None:
        ax_to_use = plt.gca()

    # Resolve colormap (prioritize: function arg > stored state > default)
    cmap_to_use = cmap if cmap is not None else state.get('cmap', 'viridis')

    # Resolve limits (prioritize: limits > clims > stored vmin/vmax > data min/max)
    if limits is not None:
        vmin, vmax = limits
    elif clims is not None:
        vmin, vmax = clims
    else:
        vmin = state.get('vmin') if state.get('vmin') is not None else cdata.min()
        vmax = state.get('vmax') if state.get('vmax') is not None else cdata.max()

    # Get colormap object
    cmap_obj = cm.get_cmap(cmap_to_use)

    # Determine if we should use discrete or continuous colorbar
    # Discrete: flat colors (per-element/per-face data)
    # Continuous: interpolated colors (per-vertex data with FaceColor='interp')
    use_discrete = (mode != 'interp')

    if use_discrete:
        # Create discrete colorbar with boundaries at unique data values
        # This makes the colorbar match the actual discrete element colors
        unique_vals = np.unique(cdata)

        # If user specified custom limits, include them in the range
        unique_vals = unique_vals[(unique_vals >= vmin) & (unique_vals <= vmax)]

        if len(unique_vals) > 1:
            # Create boundaries at midpoints between consecutive unique values
            boundaries = []
            # Lower boundary: slightly below the minimum
            boundaries.append(vmin - 0.5 * (unique_vals[1] - unique_vals[0]))

            # Midpoint boundaries
            for i in range(len(unique_vals) - 1):
                midpoint = (unique_vals[i] + unique_vals[i + 1]) / 2.0
                boundaries.append(midpoint)

            # Upper boundary: slightly above the maximum
            boundaries.append(vmax + 0.5 * (unique_vals[-1] - unique_vals[-2]))

            boundaries = np.array(boundaries)
            norm = BoundaryNorm(boundaries, cmap_obj.N)
        else:
            # Only one unique value - fall back to continuous
            norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        # Continuous colorbar for interpolated data
        norm = Normalize(vmin=vmin, vmax=vmax)

    # Create ScalarMappable for the colorbar
    # Note: We don't update the collection's colors - that was already done by patch()
    # The colorbar is just a reference indicator
    mappable = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    mappable.set_array([])

    # Build kwargs for colorbar
    cbar_kwargs = {
        'orientation': orientation,
        'extend': extend,
        **kwargs
    }
    if ticks is not None:
        cbar_kwargs['ticks'] = ticks
    if format is not None:
        cbar_kwargs['format'] = format

    # Create colorbar
    cbar = plt.colorbar(mappable, ax=ax_to_use, **cbar_kwargs)

    # Set label if provided
    if label is not None:
        cbar.set_label(label)

    return cbar


# MATLAB-compatible alias (legacy support)
# Users can call either mk.colorbar() or mk.cmap()
cmap = colorbar
