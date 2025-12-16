"""
MATLAB-style patch function for matplotlib.

Provides MATLAB-compatible patch() function with support for Faces/Vertices
notation, per-vertex color interpolation, per-face colors, line elements,
surface elements, and 3D patches.

Primary use case: FEM mesh visualization with natural 1-based indexing.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from . import colormap_utils


# List of common non-linear colormaps
# These colormaps have non-uniform perceptual gradients and benefit from
# tricontourf's contour-based approach over tripcolor's linear RGB interpolation
_NONLINEAR_COLORMAPS = {
    'jet', 'hsv', 'rainbow', 'gist_rainbow', 'gist_ncar',
    'nipy_spectral', 'gnuplot', 'gnuplot2', 'gist_stern',
    'hot', 'cool', 'spring', 'summer', 'autumn', 'winter',
    'copper', 'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
    'gist_heat', 'CMRmap', 'cubehelix', 'brg', 'bwr', 'seismic',
    'twilight', 'twilight_shifted', 'hsv'
}


def _is_nonlinear_colormap(cmap_name):
    """
    Check if a colormap is non-linear (non-uniform perceptual gradient).

    Non-linear colormaps benefit from tricontourf's approach of applying
    the colormap after interpolation, rather than tripcolor's approach of
    interpolating RGB values directly.

    Parameters
    ----------
    cmap_name : str
        Name of the colormap.

    Returns
    -------
    bool
        True if colormap is non-linear.
    """
    # Handle reversed colormaps (e.g., 'jet_r')
    base_name = cmap_name.replace('_r', '')
    return base_name.lower() in _NONLINEAR_COLORMAPS


def patch(*args, ax=None, return_mappable=False, **kwargs):
    """
    Create colored patches using MATLAB-style or Pythonic syntax.

    This function recreates MATLAB's patch functionality in matplotlib,
    with primary support for the Faces/Vertices notation commonly used
    in FEM mesh visualization.

    Parameters
    ----------
    *args : variable
        MATLAB-style name-value pairs or direct coordinates.

        Faces/Vertices notation (MATLAB-style):
            patch('Faces', F, 'Vertices', V, ...)

        Direct coordinate notation:
            patch(x, y, c)
            patch(x, y, z, c)

    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.

    return_mappable : bool, default False
        If True and color interpolation is used, returns a tuple
        (collection, mappable) where mappable can be passed directly
        to plt.colorbar(). This eliminates the need for manual
        ScalarMappable creation. If False, returns only the collection.

    **kwargs : dict
        Additional properties (can also be passed as name-value pairs in args).
        Supports both MATLAB-style and Pythonic naming conventions.

        Core properties:
        - Faces : array (n_faces, n_vertices_per_face)
            Face connectivity. Can be 1-based (auto-detected) or 0-based.
        - Vertices : array (n_vertices, ndim)
            Vertex coordinates (2D or 3D).
        - FaceVertexCData or CData : array
            Color data. Shape determines mode:
            - (n_vertices,): per-vertex colors (interpolated)
            - (n_faces,): per-face colors (flat)
            - (n_vertices, 3): per-vertex RGB
            - (n_faces, 3): per-face RGB
            Note: CData is a Pythonic alias for FaceVertexCData.
        - FaceColor : str or RGB, default 'flat'
            'flat', 'interp', 'none', RGB tuple, or color name.
        - FaceAlpha : float or array, default 1.0
            Face transparency (0=transparent, 1=opaque).
            Can be scalar or per-face array.
        - EdgeColor : str or RGB, default 'black'
            Edge color specification.
        - EdgeAlpha : float, default 1.0
            Edge transparency (0=transparent, 1=opaque).
            Controls edge opacity independently of FaceAlpha.
        - LineWidth : float, default 1.0
            Edge line width.
        - LineStyle : str, default '-'
            Edge line style: '-', '--', ':', '-.', 'none'.
        - cmap : str or Colormap, default 'viridis'
            Colormap for scalar color data.
        - vmin, vmax : float, optional
            Color normalization limits.
        - Shading : str, default 'flat'
            Shading mode for interpolated surface patches: 'flat' or 'gouraud'.
            'flat' shows discrete color bands (MATLAB-like), 'gouraud' shows smooth gradients.
            Note: When FaceColor='interp', smooth interpolation is always used.
        - interpolation_method : str, default 'auto'
            Method for color interpolation on 2D surface patches with FaceColor='interp':
            - 'auto': Automatically select based on mesh size and colormap.
              Uses 'tricontourf' for small meshes (<100 elements) with non-linear
              colormaps (e.g., 'jet'), otherwise uses 'tripcolor' for performance.
            - 'tricontourf': Use contour-filled triangulation (high quality, slower).
              Better visual accuracy for non-linear colormaps.
            - 'tripcolor': Use Gouraud-shaded triangulation (fast, hardware-accelerated).
              Interpolates RGB values directly, which can produce visual artifacts
              with non-linear colormaps.

    Returns
    -------
    collection : LineCollection, PolyCollection, or Poly3DCollection
        The created patch object.
    mappable : ScalarMappable (only if return_mappable=True and interpolation used)
        Colorbar-compatible mappable object. Use with plt.colorbar(mappable, ...).

    Examples
    --------
    MATLAB-style syntax:

    >>> P = np.array([[0, 0], [500, 0], [300, 300], [600, 300]])
    >>> edges = np.array([[1, 2], [1, 3], [2, 3], [2, 4], [3, 4]])
    >>> forces = np.array([6052, -5582, -7274, 6380, -9912])
    >>> patch('Faces', edges, 'Vertices', P,
    ...       'FaceVertexCData', forces, 'LineWidth', 3)

    Pythonic keyword argument syntax:

    >>> temps = np.array([20, 25, 30, 22])
    >>> patch(Faces=edges, Vertices=P, CData=temps,
    ...       FaceColor='interp', cmap='jet')

    Easy colorbar with return_mappable:

    >>> collection, mappable = patch(Faces=edges, Vertices=P, CData=temps,
    ...                               FaceColor='interp', return_mappable=True)
    >>> plt.colorbar(mappable, label='Temperature (°C)')

    3D surface with transparency:

    >>> vertices = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]])
    >>> faces = np.array([[1, 2, 3, 4]])
    >>> colors = np.array([[1, 0, 0]])
    >>> patch(Faces=faces, Vertices=vertices, CData=colors, FaceAlpha=0.5)

    See Also
    --------
    matplotlib.collections.LineCollection : For line segments
    matplotlib.collections.PolyCollection : For 2D polygons
    mpl_toolkits.mplot3d.art3d.Poly3DCollection : For 3D polygons
    """
    # Get or create axes
    if ax is None:
        ax = plt.gca()

    # Parse arguments
    params = _parse_arguments(args, kwargs)

    # Extract core parameters
    Faces = params['Faces']
    Vertices = params['Vertices']

    if Faces is None or Vertices is None:
        raise ValueError("Must provide 'Faces' and 'Vertices'")

    # Convert to numpy arrays
    Faces = np.asarray(Faces)
    Vertices = np.asarray(Vertices)

    # Normalize faces (1-based → 0-based if needed)
    Faces = _normalize_faces(Faces)

    # Determine dimensionality
    ndim = Vertices.shape[1]

    # Detect element type
    n_vertices_per_face = Faces.shape[1]

    if n_vertices_per_face == 2:
        # Line elements (trusses, wireframes)
        return _create_line_patch(ax, Faces, Vertices, params, ndim, return_mappable)
    else:
        # Surface elements (triangles, quads, etc.)
        if ndim == 2:
            return _create_2d_surface_patch(ax, Faces, Vertices, params, return_mappable)
        else:
            return _create_3d_surface_patch(ax, Faces, Vertices, params, return_mappable)


def _parse_arguments(args, kwargs):
    """
    Parse MATLAB-style name-value pairs, keyword arguments, or direct coordinate input.

    Supports three calling styles:
    1. MATLAB-style: patch('Faces', F, 'Vertices', V, 'CData', C)
    2. Pythonic: patch(Faces=F, Vertices=V, CData=C)
    3. Mixed: patch('Faces', F, Vertices=V, CData=C)

    Returns
    -------
    params : dict
        Dictionary of parameters.
    """
    params = {
        'Faces': None,
        'Vertices': None,
        'FaceVertexCData': None,
        'FaceColor': 'flat',
        'FaceAlpha': 1.0,
        'EdgeColor': 'black',
        'LineWidth': 1.0,
        'LineStyle': '-',
        'EdgeAlpha': 1.0,
        'cmap': 'viridis',
        'vmin': None,
        'vmax': None,
        'Shading': 'flat',
        'interpolation_method': 'auto',
    }

    # Check if first argument is a string (MATLAB-style name-value pairs)
    if args and isinstance(args[0], str):
        # Parse name-value pairs from args
        i = 0
        while i < len(args):
            if isinstance(args[i], str):
                key = args[i]
                # Normalize key (handle case variations)
                key_lower = key.lower()

                # Map to canonical names
                key_map = {
                    'faces': 'Faces',
                    'vertices': 'Vertices',
                    'facevertexcdata': 'FaceVertexCData',
                    'cdata': 'FaceVertexCData',  # Pythonic alias
                    'facecolor': 'FaceColor',
                    'facealpha': 'FaceAlpha',
                    'edgecolor': 'EdgeColor',
                    'linewidth': 'LineWidth',
                    'linestyle': 'LineStyle',
                    'edgealpha': 'EdgeAlpha',
                    'shading': 'Shading',
                    'interpolation_method': 'interpolation_method',
                }

                canonical_key = key_map.get(key_lower, key)

                if i + 1 < len(args):
                    params[canonical_key] = args[i + 1]
                    i += 2
                else:
                    raise ValueError(f"No value provided for '{key}'")
            else:
                i += 1
    elif args:
        # Direct coordinate input: patch(x, y, c) or patch(x, y, z, c)
        # TODO: Implement this for completeness
        raise NotImplementedError("Direct coordinate input not yet implemented. Use Faces/Vertices notation.")

    # Process kwargs with key normalization
    # This allows both MATLAB-style (FaceVertexCData) and Pythonic (CData) naming
    for key, value in kwargs.items():
        key_lower = key.lower()

        # Map to canonical names
        key_map = {
            'faces': 'Faces',
            'vertices': 'Vertices',
            'facevertexcdata': 'FaceVertexCData',
            'cdata': 'FaceVertexCData',  # Pythonic alias
            'facecolor': 'FaceColor',
            'facealpha': 'FaceAlpha',
            'edgecolor': 'EdgeColor',
            'linewidth': 'LineWidth',
            'linestyle': 'LineStyle',
            'edgealpha': 'EdgeAlpha',
            'shading': 'Shading',
            'interpolation_method': 'interpolation_method',
        }

        canonical_key = key_map.get(key_lower, key)
        params[canonical_key] = value

    return params


def _normalize_faces(Faces):
    """
    Convert 1-based face indices to 0-based if needed.

    Parameters
    ----------
    Faces : array
        Face connectivity array.

    Returns
    -------
    Faces : array
        0-based face connectivity.
    """
    if Faces.size == 0:
        return Faces

    # Check if 1-based (minimum index is 1)
    if Faces.min() == 1:
        return Faces - 1
    else:
        return Faces


def _process_colors(CData, n_faces, n_vertices, Faces, mode, cmap_name, vmin, vmax):
    """
    Process color data and return face/edge colors.

    Parameters
    ----------
    CData : array or None
        Color data.
    n_faces : int
        Number of faces.
    n_vertices : int
        Number of vertices.
    Faces : array
        Face connectivity (0-based).
    mode : str
        Color mode: 'flat', 'interp', or color specification.
    cmap_name : str
        Colormap name.
    vmin, vmax : float or None
        Color normalization limits.

    Returns
    -------
    colors : array
        RGBA colors for faces (n_faces, 4).
    """
    if CData is None:
        # Use uniform color from mode
        if mode == 'flat' or mode == 'interp':
            # Default to gray
            color = [0.5, 0.5, 0.5, 1.0]
        else:
            # Parse color specification
            color = _parse_color(mode)
        return np.tile(color, (n_faces, 1))

    CData = np.asarray(CData)

    # Determine color mode from shape
    if CData.shape[0] == n_vertices:
        # Per-vertex colors
        if mode == 'interp':
            colors = _interpolate_vertex_colors(CData, Faces, cmap_name, vmin, vmax)
        else:
            # Average vertex colors for each face
            colors = _average_vertex_colors(CData, Faces, cmap_name, vmin, vmax)
    elif CData.shape[0] == n_faces:
        # Per-face colors
        colors = _apply_colormap(CData, cmap_name, vmin, vmax)
    else:
        raise ValueError(
            f"CData shape {CData.shape} doesn't match "
            f"n_vertices={n_vertices} or n_faces={n_faces}"
        )

    return colors


def _parse_color(color_spec):
    """Parse color specification to RGBA."""
    from matplotlib.colors import to_rgba

    if isinstance(color_spec, str):
        return to_rgba(color_spec)
    else:
        # Assume RGB or RGBA tuple/list
        color = np.asarray(color_spec, dtype=float)
        if color.size == 3:
            return np.append(color, 1.0)
        elif color.size == 4:
            return color
        else:
            raise ValueError(f"Invalid color specification: {color_spec}")


def _apply_colormap(data, cmap_name, vmin, vmax):
    """
    Apply colormap to scalar or RGB data.

    Parameters
    ----------
    data : array
        Color data (n, ) or (n, 3).
    cmap_name : str
        Colormap name.
    vmin, vmax : float or None
        Normalization limits.

    Returns
    -------
    colors : array (n, 4)
        RGBA colors.
    """
    data = np.asarray(data)

    if data.ndim == 1:
        # Scalar data - apply colormap
        norm = Normalize(
            vmin=vmin if vmin is not None else data.min(),
            vmax=vmax if vmax is not None else data.max()
        )
        cmap = get_cmap(cmap_name)
        colors = cmap(norm(data))
    elif data.ndim == 2 and data.shape[1] == 3:
        # RGB data - add alpha channel
        alpha = np.ones((data.shape[0], 1))
        colors = np.hstack([data, alpha])
    elif data.ndim == 2 and data.shape[1] == 4:
        # RGBA data - use as is
        colors = data
    else:
        raise ValueError(f"Invalid color data shape: {data.shape}")

    return colors


def _average_vertex_colors(CData, Faces, cmap_name, vmin, vmax):
    """
    Average vertex colors for each face (simple approximation of interpolation).

    Parameters
    ----------
    CData : array
        Per-vertex color data.
    Faces : array
        Face connectivity (0-based).
    cmap_name : str
        Colormap name.
    vmin, vmax : float or None
        Normalization limits.

    Returns
    -------
    face_colors : array (n_faces, 4)
        RGBA colors for each face.
    """
    # First apply colormap to vertex data
    vertex_colors = _apply_colormap(CData, cmap_name, vmin, vmax)

    # Average colors for each face
    n_faces = Faces.shape[0]
    face_colors = np.zeros((n_faces, 4))

    for i, face in enumerate(Faces):
        face_colors[i] = vertex_colors[face].mean(axis=0)

    return face_colors


def _interpolate_vertex_colors(CData, Faces, cmap_name, vmin, vmax):
    """
    Interpolate vertex colors (for now, same as averaging).

    Future enhancement: subdivide faces for true interpolation.

    Parameters
    ----------
    CData : array
        Per-vertex color data.
    Faces : array
        Face connectivity (0-based).
    cmap_name : str
        Colormap name.
    vmin, vmax : float or None
        Normalization limits.

    Returns
    -------
    face_colors : array (n_faces, 4)
        RGBA colors for each face.
    """
    # For now, use averaging
    # TODO: Implement true interpolation via face subdivision
    return _average_vertex_colors(CData, Faces, cmap_name, vmin, vmax)


def _subdivide_to_triangles(Faces):
    """
    Subdivide quad (or polygon) faces into triangles.

    For quads (4 vertices): splits into 2 triangles
    For triangles: returns as-is
    For polygons with n>4: uses fan triangulation

    Parameters
    ----------
    Faces : array (n_faces, n_vertices_per_face)
        Face connectivity (0-based).

    Returns
    -------
    triangles : array (n_triangles, 3)
        Triangle connectivity.
    """
    triangles = []

    for face in Faces:
        n_verts = len(face)

        if n_verts == 3:
            # Already a triangle
            triangles.append(face)
        elif n_verts == 4:
            # Quad - split into 2 triangles
            # Triangle 1: vertices 0, 1, 2
            # Triangle 2: vertices 0, 2, 3
            triangles.append([face[0], face[1], face[2]])
            triangles.append([face[0], face[2], face[3]])
        else:
            # Polygon with n > 4 - fan triangulation from first vertex
            for i in range(1, n_verts - 1):
                triangles.append([face[0], face[i], face[i+1]])

    return np.array(triangles)


def _apply_alpha(colors, alpha):
    """
    Apply alpha (transparency) to colors.

    Parameters
    ----------
    colors : array (n, 4) or str or tuple
        RGBA colors array, color name, or RGB tuple.
    alpha : float or array
        Alpha values.

    Returns
    -------
    colors : array (n, 4) or tuple
        Colors with updated alpha. Returns tuple (r,g,b,a) for single color input,
        array for multiple colors.
    """
    # Handle string color names or RGB tuples (for single edge colors)
    if isinstance(colors, str) or (isinstance(colors, (tuple, list)) and len(colors) in [3, 4]):
        from matplotlib.colors import to_rgba
        rgba = to_rgba(colors)
        # Apply alpha
        if np.isscalar(alpha):
            return (rgba[0], rgba[1], rgba[2], alpha)
        else:
            # Single color with array of alphas - not typical use case
            return (rgba[0], rgba[1], rgba[2], alpha[0] if len(alpha) > 0 else rgba[3])

    # Handle array of colors
    colors = np.asarray(colors)
    colors = colors.copy()

    if np.isscalar(alpha):
        # Uniform alpha
        colors[:, 3] = alpha
    else:
        # Per-face alpha
        alpha = np.asarray(alpha)
        if alpha.size == colors.shape[0]:
            colors[:, 3] = alpha
        else:
            raise ValueError(f"Alpha size {alpha.size} doesn't match number of faces {colors.shape[0]}")

    return colors


def _create_line_patch(ax, Faces, Vertices, params, ndim, return_mappable=False):
    """
    Create line patch (for trusses, wireframes).

    Uses LineCollection (2D) or Line3DCollection (3D).
    """
    n_faces = Faces.shape[0]
    n_vertices = Vertices.shape[0]

    # Get color parameters
    CData = params['FaceVertexCData']
    mode = params['FaceColor']
    cmap_name = params['cmap']
    vmin = params['vmin']
    vmax = params['vmax']

    # Get edge properties
    linewidth = params['LineWidth']
    linestyle = params['LineStyle']
    alpha = params['FaceAlpha']

    # Check if we need true interpolation for line elements
    need_interpolation = (mode == 'interp' and CData is not None and
                         CData.shape[0] == n_vertices and CData.ndim == 1)

    # Track colormap info for optional mappable return
    mappable_info = None

    if need_interpolation:
        # Subdivide line elements for smooth color interpolation
        n_segments_per_edge = 100  # Number of sub-segments per element

        # Apply colormap to vertex data
        norm = Normalize(
            vmin=vmin if vmin is not None else CData.min(),
            vmax=vmax if vmax is not None else CData.max()
        )
        cmap = get_cmap(cmap_name)

        # Store for optional mappable return
        if return_mappable:
            mappable_info = (cmap, norm)

        # Create subdivided segments with interpolated colors
        segments, colors = colormap_utils._generate_interpolated_line_colors_and_segments(
            CData, Faces, Vertices, n_segments_per_edge, cmap, norm
        )

        # Apply alpha
        colors = _apply_alpha(colors, alpha)

    else:
        # Use flat colors (one color per element)
        colors = _process_colors(CData, n_faces, n_vertices, Faces, mode, cmap_name, vmin, vmax)
        colors = _apply_alpha(colors, alpha)
        segments = Vertices[Faces]  # (n_faces, 2, ndim)

    if ndim == 2:
        # 2D lines
        lc = LineCollection(
            segments,
            colors=colors,
            linewidths=linewidth,
            linestyles=linestyle
        )
        ax.add_collection(lc)
    else:
        # 3D lines
        lc = Line3DCollection(
            segments,
            colors=colors,
            linewidths=linewidth,
            linestyles=linestyle
        )
        ax.add_collection3d(lc)

    # Update axis limits
    ax.autoscale_view()

    # Store state for automatic colorbar
    if CData is not None:
        if need_interpolation:
            colormap_utils._store_patch_state(
                collection=lc,
                cdata=np.asarray(CData),
                ax=ax,
                faces=Faces,
                vertices=Vertices,
                mode='interp',
                n_segments_per_edge=n_segments_per_edge,
                cmap=cmap_name,
                vmin=vmin,
                vmax=vmax
            )
        else:
            colormap_utils._store_patch_state(
                collection=lc,
                cdata=np.asarray(CData),
                ax=ax,
                cmap=cmap_name,
                vmin=vmin,
                vmax=vmax
            )

    # Return collection with optional mappable for colorbar
    if return_mappable and mappable_info is not None:
        import matplotlib.cm as cm
        cmap, norm = mappable_info
        mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])
        return lc, mappable
    else:
        return lc


def _create_2d_surface_patch(ax, Faces, Vertices, params, return_mappable=False):
    """
    Create 2D surface patch using PolyCollection, tripcolor, or tricontourf.

    For FaceColor='interp', automatically selects between tripcolor and tricontourf
    based on mesh size and colormap linearity for optimal visual quality.
    """
    n_faces = Faces.shape[0]
    n_vertices = Vertices.shape[0]

    # Get color parameters
    CData = params['FaceVertexCData']
    face_mode = params['FaceColor']
    cmap_name = params['cmap']
    vmin = params['vmin']
    vmax = params['vmax']
    interpolation_method = params['interpolation_method']

    # Check if we need true interpolation
    need_interpolation = (face_mode == 'interp' and CData is not None and
                         CData.shape[0] == n_vertices and CData.ndim == 1)

    if need_interpolation:
        # Decide whether to use tricontourf or tripcolor
        use_tricontourf = False

        if interpolation_method == 'tricontourf':
            use_tricontourf = True
        elif interpolation_method == 'tripcolor':
            use_tricontourf = False
        elif interpolation_method == 'auto':
            # Auto-select: use tricontourf for small meshes with non-linear colormaps
            is_small_mesh = n_faces < 100
            is_nonlinear_cmap = _is_nonlinear_colormap(cmap_name)
            use_tricontourf = is_small_mesh and is_nonlinear_cmap
        else:
            raise ValueError(
                f"Invalid interpolation_method '{interpolation_method}'. "
                "Must be 'auto', 'tricontourf', or 'tripcolor'."
            )

        # Import triangulation tools
        from matplotlib.tri import Triangulation

        # Subdivide quads and polygons into triangles
        triangles = _subdivide_to_triangles(Faces)

        # Create triangulation
        x = Vertices[:, 0]
        y = Vertices[:, 1]
        triang = Triangulation(x, y, triangles)

        # Normalize colors
        vmin_val = vmin if vmin is not None else CData.min()
        vmax_val = vmax if vmax is not None else CData.max()

        # Handle edge case: all values are the same (vmin == vmax)
        # This would create invalid contour levels or normalization
        if vmin_val == vmax_val:
            # Add small epsilon to create a valid range
            # Use 1% of the value or 1.0 if value is zero
            epsilon = abs(vmin_val) * 0.01 if vmin_val != 0 else 1.0
            vmax_val = vmin_val + epsilon

        norm = Normalize(vmin=vmin_val, vmax=vmax_val)

        # Get parameters
        edge_color = params['EdgeColor']
        linewidth = params['LineWidth']
        linestyle = params['LineStyle']
        edge_alpha = params['EdgeAlpha']
        face_alpha = params['FaceAlpha']
        cmap = get_cmap(cmap_name)

        if use_tricontourf:
            # Use tricontourf for high-quality interpolation
            # Always use 256 levels for smooth gradients
            levels = np.linspace(vmin_val, vmax_val, 256)
            tc = ax.tricontourf(triang, CData, levels=levels, cmap=cmap,
                               norm=norm)

            # tricontourf doesn't support alpha directly on the collection
            # Apply alpha to all polygons if needed
            if face_alpha != 1.0:
                for collection in tc.collections:
                    collection.set_alpha(face_alpha)

        else:
            # Use tripcolor with gouraud shading for performance
            # When FaceColor='interp', always use smooth shading
            tc = ax.tripcolor(triang, CData,
                             cmap=cmap,
                             norm=norm,
                             shading='gouraud',
                             alpha=face_alpha if np.isscalar(face_alpha) else None)

        # Draw original face edges separately if needed
        if edge_color != 'none':
            edge_segments = []
            for face in Faces:
                face_verts = Vertices[face]
                for i in range(len(face)):
                    v1 = face_verts[i]
                    v2 = face_verts[(i + 1) % len(face)]
                    edge_segments.append([v1, v2])

            lc = LineCollection(edge_segments,
                              colors=edge_color,
                              linewidths=linewidth,
                              linestyles=linestyle,
                              alpha=edge_alpha)
            ax.add_collection(lc)

        ax.autoscale_view()

        # Store state for automatic colorbar
        colormap_utils._store_patch_state(
            collection=tc,
            cdata=np.asarray(CData),
            ax=ax,
            mode='interp',
            faces=Faces,
            vertices=Vertices,
            cmap=cmap_name,
            vmin=vmin,
            vmax=vmax
        )

        if return_mappable:
            return tc, tc
        else:
            return tc

    else:
        # Use PolyCollection for flat colors or uniform colors
        # Process face colors
        if face_mode == 'none':
            face_colors = 'none'
        else:
            face_colors = _process_colors(CData, n_faces, n_vertices, Faces, face_mode, cmap_name, vmin, vmax)

            # Apply face alpha
            face_alpha = params['FaceAlpha']
            face_colors = _apply_alpha(face_colors, face_alpha)

        # Get edge parameters
        edge_color = params['EdgeColor']
        linewidth = params['LineWidth']
        linestyle = params['LineStyle']
        edge_alpha = params['EdgeAlpha']

        # Apply edge alpha to edge color
        if edge_color != 'none' and edge_alpha != 1.0:
            edge_color = _apply_alpha(edge_color, edge_alpha)

        # Create polygon vertices
        verts = [Vertices[face] for face in Faces]

        # Create collection
        pc = PolyCollection(
            verts,
            facecolors=face_colors,
            edgecolors=edge_color,
            linewidths=linewidth,
            linestyles=linestyle
        )

        ax.add_collection(pc)
        ax.autoscale_view()

        # Store state for automatic colorbar
        if CData is not None:
            colormap_utils._store_patch_state(
                collection=pc,
                cdata=np.asarray(CData),
                ax=ax,
                cmap=cmap_name,
                vmin=vmin,
                vmax=vmax
            )

        return pc


def _create_3d_surface_patch(ax, Faces, Vertices, params, return_mappable=False):
    """
    Create 3D surface patch using Poly3DCollection.
    """
    n_faces = Faces.shape[0]
    n_vertices = Vertices.shape[0]

    # Get color parameters
    CData = params['FaceVertexCData']
    face_mode = params['FaceColor']
    cmap_name = params['cmap']
    vmin = params['vmin']
    vmax = params['vmax']

    # Process face colors
    if face_mode == 'none':
        face_colors = 'none'
    else:
        face_colors = _process_colors(CData, n_faces, n_vertices, Faces, face_mode, cmap_name, vmin, vmax)

        # Apply face alpha
        face_alpha = params['FaceAlpha']
        face_colors = _apply_alpha(face_colors, face_alpha)

    # Get edge parameters
    edge_color = params['EdgeColor']
    linewidth = params['LineWidth']
    edge_alpha = params['EdgeAlpha']

    # Apply edge alpha to edge color
    if edge_color != 'none' and edge_alpha != 1.0:
        edge_color = _apply_alpha(edge_color, edge_alpha)

    # Create polygon vertices
    verts = [Vertices[face] for face in Faces]

    # Create collection
    pc = Poly3DCollection(
        verts,
        facecolors=face_colors,
        edgecolors=edge_color,
        linewidths=linewidth
    )

    ax.add_collection3d(pc)

    # Set 3D axis limits manually
    ax.set_xlim(Vertices[:, 0].min(), Vertices[:, 0].max())
    ax.set_ylim(Vertices[:, 1].min(), Vertices[:, 1].max())
    ax.set_zlim(Vertices[:, 2].min(), Vertices[:, 2].max())

    # Store state for automatic colorbar
    if CData is not None:
        colormap_utils._store_patch_state(
            collection=pc,
            cdata=np.asarray(CData),
            ax=ax,
            mode=face_mode,
            cmap=cmap_name,
            vmin=vmin,
            vmax=vmax
        )

    return pc
