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


def patch(*args, ax=None, return_mappable=False, **kwargs):
    """
    Create colored patches using MATLAB-style syntax.

    This function recreates MATLAB's patch functionality in matplotlib,
    with primary support for the Faces/Vertices notation commonly used
    in FEM mesh visualization.

    Parameters
    ----------
    *args : variable
        MATLAB-style name-value pairs or direct coordinates.

        Faces/Vertices notation (primary):
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

        Core properties:
        - Faces : array (n_faces, n_vertices_per_face)
            Face connectivity. Can be 1-based (auto-detected) or 0-based.
        - Vertices : array (n_vertices, ndim)
            Vertex coordinates (2D or 3D).
        - FaceVertexCData : array
            Color data. Shape determines mode:
            - (n_vertices,): per-vertex colors (interpolated)
            - (n_faces,): per-face colors (flat)
            - (n_vertices, 3): per-vertex RGB
            - (n_faces, 3): per-face RGB
        - FaceColor : str or RGB, default 'flat'
            'flat', 'interp', 'none', RGB tuple, or color name.
        - FaceAlpha : float or array, default 1.0
            Face transparency (0=transparent, 1=opaque).
            Can be scalar or per-face array.
        - EdgeColor : str or RGB, default 'black'
            Edge color specification.
        - LineWidth : float, default 1.0
            Edge line width.
        - LineStyle : str, default '-'
            Edge line style: '-', '--', ':', '-.', 'none'.
        - cmap : str or Colormap, default 'viridis'
            Colormap for scalar color data.
        - vmin, vmax : float, optional
            Color normalization limits.

    Returns
    -------
    collection : LineCollection, PolyCollection, or Poly3DCollection
        The created patch object.
    mappable : ScalarMappable (only if return_mappable=True and interpolation used)
        Colorbar-compatible mappable object. Use with plt.colorbar(mappable, ...).

    Examples
    --------
    Truss with per-element colors:

    >>> P = np.array([[0, 0], [500, 0], [300, 300], [600, 300]])
    >>> edges = np.array([[1, 2], [1, 3], [2, 3], [2, 4], [3, 4]])
    >>> forces = np.array([6052, -5582, -7274, 6380, -9912])
    >>> patch('Faces', edges, 'Vertices', P,
    ...       'FaceVertexCData', forces, 'LineWidth', 3)

    Truss with interpolated nodal colors:

    >>> temps = np.array([20, 25, 30, 22])  # Temperature at each node
    >>> patch('Faces', edges, 'Vertices', P,
    ...       'FaceVertexCData', temps, 'FaceColor', 'interp')

    Easy colorbar with return_mappable (no ScalarMappable boilerplate):

    >>> collection, mappable = patch('Faces', edges, 'Vertices', P,
    ...                               'FaceVertexCData', temps,
    ...                               'FaceColor', 'interp',
    ...                               return_mappable=True)
    >>> plt.colorbar(mappable, label='Temperature (°C)')

    3D surface with transparency:

    >>> vertices = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]])
    >>> faces = np.array([[1, 2, 3, 4]])
    >>> colors = np.array([[1, 0, 0]])
    >>> patch('Faces', faces, 'Vertices', vertices,
    ...       'FaceVertexCData', colors, 'FaceAlpha', 0.5)

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
    Parse MATLAB-style name-value pairs or direct coordinate input.

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
                    'facecolor': 'FaceColor',
                    'facealpha': 'FaceAlpha',
                    'edgecolor': 'EdgeColor',
                    'linewidth': 'LineWidth',
                    'linestyle': 'LineStyle',
                    'edgealpha': 'EdgeAlpha',
                }

                canonical_key = key_map.get(key_lower, key)

                if i + 1 < len(args):
                    params[canonical_key] = args[i + 1]
                    i += 2
                else:
                    raise ValueError(f"No value provided for '{key}'")
            else:
                i += 1
    else:
        # Direct coordinate input: patch(x, y, c) or patch(x, y, z, c)
        # TODO: Implement this for completeness
        raise NotImplementedError("Direct coordinate input not yet implemented. Use Faces/Vertices notation.")

    # Override with any kwargs
    params.update(kwargs)

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
    colors : array (n, 4)
        RGBA colors.
    alpha : float or array
        Alpha values.

    Returns
    -------
    colors : array (n, 4)
        Colors with updated alpha.
    """
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
        segments = []
        segment_colors = []

        for face in Faces:
            # Get vertex positions and color values
            v1_pos = Vertices[face[0]]
            v2_pos = Vertices[face[1]]
            v1_color = CData[face[0]]
            v2_color = CData[face[1]]

            # Create interpolated segments
            for i in range(n_segments_per_edge):
                t1 = i / n_segments_per_edge
                t2 = (i + 1) / n_segments_per_edge

                # Interpolate positions
                p1 = v1_pos * (1 - t1) + v2_pos * t1
                p2 = v1_pos * (1 - t2) + v2_pos * t2

                segments.append([p1, p2])

                # Interpolate color value at segment midpoint
                t_mid = (t1 + t2) / 2
                c_mid = v1_color * (1 - t_mid) + v2_color * t_mid
                segment_colors.append(cmap(norm(c_mid)))

        segments = np.array(segments)
        colors = np.array(segment_colors)

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
    Create 2D surface patch using PolyCollection or TriMesh for interpolation.
    """
    n_faces = Faces.shape[0]
    n_vertices = Vertices.shape[0]

    # Get color parameters
    CData = params['FaceVertexCData']
    face_mode = params['FaceColor']
    cmap_name = params['cmap']
    vmin = params['vmin']
    vmax = params['vmax']

    # Check if we need true interpolation
    need_interpolation = (face_mode == 'interp' and CData is not None and
                         CData.shape[0] == n_vertices)

    if need_interpolation:
        # Use matplotlib's tripcolor for true vertex color interpolation
        from matplotlib.tri import Triangulation

        # Subdivide faces into triangles
        triangles = _subdivide_to_triangles(Faces)

        # Create triangulation
        x = Vertices[:, 0]
        y = Vertices[:, 1]

        # Process vertex colors
        if CData.ndim == 1:
            # Scalar data - will be mapped through colormap
            vertex_colors = CData
        elif CData.ndim == 2 and CData.shape[1] == 3:
            # RGB data - need to use scalar representation
            # Convert RGB to grayscale for triangulation
            vertex_colors = 0.299 * CData[:, 0] + 0.587 * CData[:, 1] + 0.114 * CData[:, 2]
        else:
            vertex_colors = CData

        # Create triangulation
        triang = Triangulation(x, y, triangles)

        # Normalize colors
        norm = Normalize(
            vmin=vmin if vmin is not None else vertex_colors.min(),
            vmax=vmax if vmax is not None else vertex_colors.max()
        )

        # Get edge parameters
        edge_color = params['EdgeColor']
        linewidth = params['LineWidth']
        linestyle = params['LineStyle']
        edge_alpha = params['EdgeAlpha']
        face_alpha = params['FaceAlpha']

        # Use tripcolor for smooth interpolation (without edges)
        cmap = get_cmap(cmap_name)
        tc = ax.tripcolor(triang, vertex_colors,
                         cmap=cmap,
                         norm=norm,
                         shading='gouraud',  # Smooth interpolation
                         alpha=face_alpha if np.isscalar(face_alpha) else None)

        # Draw original face edges separately if needed
        if edge_color != 'none':
            # Create edge segments from original faces (not subdivided triangles)
            edge_segments = []
            for face in Faces:
                # Create closed polygon by connecting vertices in order
                face_verts = Vertices[face]
                for i in range(len(face)):
                    v1 = face_verts[i]
                    v2 = face_verts[(i + 1) % len(face)]
                    edge_segments.append([v1, v2])

            # Draw edges
            lc = LineCollection(edge_segments,
                              colors=edge_color,
                              linewidths=linewidth,
                              linestyles=linestyle,
                              alpha=edge_alpha)
            ax.add_collection(lc)

        ax.autoscale_view()
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

        # Create polygon vertices
        verts = [Vertices[face] for face in Faces]

        # Create collection
        pc = PolyCollection(
            verts,
            facecolors=face_colors,
            edgecolors=edge_color,
            linewidths=linewidth,
            linestyles=linestyle,
            alpha=edge_alpha if face_mode == 'none' else None
        )

        ax.add_collection(pc)
        ax.autoscale_view()

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

    return pc
