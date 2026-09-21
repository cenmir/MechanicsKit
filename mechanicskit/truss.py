"""Draw a plane truss: bars, numbered nodes and elements, supports and loads.

Everything here is **1-based**, the way a truss is numbered on paper. Node
``i`` owns degrees of freedom ``2i-1`` (its x) and ``2i`` (its y), elements
refer to nodes by their number, and prescribed degrees of freedom are given as
those same numbers.

    from mechanicskit import draw_truss

    draw_truss(nodes, elements, presc=[1, 2, 4], loads=[[3, 0, -10e3]])

The support symbols follow FBD Lab: a rounded-top triangle with a pin hole,
standing on a base bar that fades out. A node with both degrees of freedom
held gets a pin; a node with one held gets a roller, turned so that its wheels
run along the direction that is still free.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrow, Polygon

__all__ = ["draw_truss", "animate_truss", "pin_support", "roller_support"]

# FBD Lab's palette and proportions (fbd_lab/items/pin_support.py)
BODY = "#d8ba94"          # tan triangle
GROUND = "#9c8876"        # the base bar, the body darkened
EDGE = "black"
NODE_FACE = "c"           # cyan disc behind a node number
ELEM_FACE = "y"           # yellow square behind an element number
BAR = "black"
LOAD = "#c00000"

PIN_HOLE_RATIO = 0.12     # pin hole radius, as a fraction of the symbol height
ROUND_RATIO = 0.35        # rounded top radius, as a fraction of the half width
BASE_RATIO = 0.80         # base bar height, as a fraction of the symbol height
BASE_WIDTH_RATIO = 2.0    # base bar width, as a fraction of the symbol height


def _rot(angle_deg):
    t = np.radians(angle_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def _place(pts, origin, angle_deg):
    return np.asarray(pts) @ _rot(angle_deg).T + np.asarray(origin)


def _base_bar(ax, origin, angle, h, zorder):
    """The ground under a support: a bar that fades away downwards."""
    half_w, bh, n = BASE_WIDTH_RATIO*h/2, BASE_RATIO*h, 24
    for k in range(n):
        y0, y1 = -h - bh*k/n, -h - bh*(k + 1)/n
        quad = _place([(-half_w, y0), (half_w, y0), (half_w, y1), (-half_w, y1)],
                      origin, angle)
        ax.add_patch(Polygon(quad, closed=True, facecolor=GROUND, edgecolor="none",
                             alpha=(1 - k/n)*0.9, zorder=zorder))


def _triangle(ax, origin, angle, h, tri_h, zorder):
    """The rounded-top triangle, apex at the node, hanging below it."""
    half = h/2
    r = half*ROUND_RATIO
    arc = [(r*np.cos(a), -r + r*np.sin(a))
           for a in np.linspace(0, np.pi, 40)]                 # over the top
    pts = [(half, -tri_h), *arc, (-half, -tri_h)]
    ax.add_patch(Polygon(_place(pts, origin, angle), closed=True, facecolor=BODY,
                         edgecolor=EDGE, lw=1.1, zorder=zorder))
    hole = _place([(0, -r)], origin, angle)[0]
    ax.add_patch(Circle(hole, PIN_HOLE_RATIO*h, facecolor="white", edgecolor=EDGE,
                        lw=0.8, zorder=zorder + 1))


def pin_support(ax, xy, h, angle=0.0, zorder=3):
    """A pin: the triangle sits straight on the ground."""
    _triangle(ax, xy, angle, h, h, zorder)
    _base_bar(ax, xy, angle, h, zorder)


def roller_support(ax, xy, h, angle=0.0, zorder=3):
    """A roller: the same triangle, up on two wheels that are free to run."""
    r = 0.13*h
    tri_h = h - 2*r
    _triangle(ax, xy, angle, h, tri_h, zorder)
    for sx in (-1, 1):
        c = _place([(sx*tri_h/4, -(tri_h + r))], xy, angle)[0]
        ax.add_patch(Circle(c, r, facecolor=BODY, edgecolor=EDGE, lw=1.0,
                            zorder=zorder + 1))
    _base_bar(ax, xy, angle, h, zorder)


def _normalise_loads(loads, nnod):
    """Accept a force vector of length 2n, or rows of [node, Fx, Fy]."""
    if loads is None:
        return []
    if hasattr(loads, "data"):                   # a OneArray
        loads = loads.data
    arr = np.asarray(loads, dtype=float)
    if arr.ndim == 1:                            # a full force vector
        if arr.size != 2*nnod:
            raise ValueError(f"a force vector must have {2*nnod} entries, got {arr.size}")
        return [(int(k)//2 + 1, arr[k], 0.0) if k % 2 == 0 else (int(k)//2 + 1, 0.0, arr[k])
                for k in range(arr.size) if arr[k] != 0.0]
    return [(int(row[0]), float(row[1]), float(row[2])) for row in arr]


def draw_truss(nodes, elements, presc=(), loads=None, ax=None, *,
               node_numbers=True, element_numbers=True, supports=True,
               displacements=None, scale=1.0, values=None, cmap="viridis",
               value_label=None, title=None, figsize=(9, 4.5), fontsize=8,
               support_size=None, support_angles=None, load_length=None,
               display_undeformed=True, colorbar=True, bar_color=BAR,
               bar_lw=2.0, bar_ls="-"):
    """Draw a plane truss.

    Parameters
    ----------
    nodes : (n, 2) array
        Coordinates. Row ``i-1`` is node ``i``.
    elements : (m, 2) or (m, 3) array
        Node numbers of each bar, 1-based. A third column is ignored, so a
        table that also carries a section can be passed straight in.
    presc : sequence of int
        Prescribed degree-of-freedom numbers, 1-based. Node ``i`` owns
        ``2i-1`` and ``2i``.
    loads : (2n,) array or rows of [node, Fx, Fy]
        Arrows are drawn along the degree of freedom each entry belongs to.
    displacements : (n, 2) or (2n,) array, optional
        Drawn on top of the undeformed shape, multiplied by ``scale``.
    values : (m,) array, optional
        One number per bar; colours the bars and adds a colourbar.
    support_angles : dict, optional
        Degrees to turn the support symbol at a given node, as
        ``{node: angle}``. The default hangs every symbol below its node,
        which is right for a support standing on the ground; a node held
        against a wall on its left wants ``-90``.
    display_undeformed : bool
        Whether to leave the undeformed shape behind as a dashed ghost. Only
        has an effect when ``displacements`` is given.
    colorbar : bool
        Whether to add the colourbar that goes with ``values``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    nodes = np.asarray(getattr(nodes, "data", nodes), dtype=float)
    elements = np.asarray(getattr(elements, "data", elements))[:, :2].astype(int)
    presc = getattr(presc, "data", presc)
    nnod, nele = len(nodes), len(elements)
    presc = [int(k) for k in (presc if presc is not None else [])]
    support_angles = dict(support_angles or {})

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    span = np.ptp(nodes, axis=0)
    ref = float(max(span.max(), 1.0))
    h = support_size if support_size is not None else 0.080*ref
    if load_length is not None:
        arrow = load_length
    else:                        # keep arrows from dwarfing a shallow truss
        arrow = 0.13*ref if span[1] == 0 else min(0.13*ref, 0.75*span[1])


    # --- bars ---------------------------------------------------------------
    # With displacements given, the truss is drawn where it has moved to and
    # the undeformed shape stays behind as a ghost.
    if displacements is not None:
        d = np.asarray(getattr(displacements, "data", displacements),
                       dtype=float).reshape(-1, 2)
        drawn = nodes + scale*d
        if display_undeformed:
            for i, j in elements:
                ax.plot(*zip(nodes[i-1], nodes[j-1]), color="0.72", lw=1.0,
                        ls="--", zorder=1)
    else:
        drawn = nodes

    if values is not None:
        values = np.asarray(getattr(values, "data", values), dtype=float)
        norm = plt.Normalize(values.min(), values.max())
        colours = plt.get_cmap(cmap)(norm(values))
    for e, (i, j) in enumerate(elements):
        col = colours[e] if values is not None else bar_color
        ax.plot(*zip(drawn[i-1], drawn[j-1]), color=col, lw=bar_lw, ls=bar_ls,
                solid_capstyle="round", zorder=2)

    centroid = drawn.mean(axis=0)

    # --- supports -----------------------------------------------------------
    if supports:
        for i in range(1, nnod + 1):
            fx, fy = (2*i - 1) in presc, (2*i) in presc
            turn = float(support_angles.get(i, 0.0))
            if fx and fy:
                pin_support(ax, drawn[i-1], h, angle=turn)
            elif fy:                       # held vertically, free along x
                roller_support(ax, drawn[i-1], h, angle=turn)
            elif fx:                       # held horizontally, free along y
                roller_support(ax, drawn[i-1], h, angle=turn - 90)

    # --- loads --------------------------------------------------------------
    for node, fxv, fyv in _normalise_loads(loads, nnod):
        for comp, (ux, uy) in ((fxv, (1, 0)), (fyv, (0, 1))):
            if comp == 0:
                continue
            s = np.sign(comp)
            d = np.array([ux, uy])*s
            p = drawn[node-1]
            gap = d*0.024*ref                     # keep clear of the node marker
            # Put the arrow on the outward side of the node, so that it never
            # has to cross the truss to reach the joint it loads.
            axis = 1 if uy else 0
            outward = np.sign(p[axis] - centroid[axis]) or -d[axis]
            if outward == np.sign(d[axis]):       # load points away: hang it off
                tail, tip = p + gap, p + d*arrow
            else:                                 # load points in: come from outside
                tip, tail = p - gap, p - gap - d*arrow
            ax.add_patch(FancyArrow(*tail, *(tip - tail), width=0.004*ref,
                                    head_width=0.026*ref, head_length=0.030*ref,
                                    length_includes_head=True, color=LOAD,
                                    zorder=4))
            # vertical arrows label beyond the tail, horizontal ones above it
            far = tail if np.hypot(*(tail - p)) > np.hypot(*(tip - p)) else tip
            anchor = far if uy else (tail + tip)/2
            off = (0, 11*np.sign(far[1] - p[1])) if uy else (0, 9)
            ax.annotate(f"{abs(comp):g}", anchor, textcoords="offset points",
                        xytext=off, ha="center", va="center", fontsize=fontsize,
                        color=LOAD, zorder=7,
                        bbox=dict(boxstyle="square,pad=0.12", facecolor="white",
                                  edgecolor="none", alpha=0.85))

    # --- numbers ------------------------------------------------------------
    if element_numbers:
        for e, (i, j) in enumerate(elements, start=1):
            mid = (drawn[i-1] + drawn[j-1])/2
            ax.text(*mid, str(e), ha="center", va="center", fontsize=fontsize,
                    zorder=5, bbox=dict(boxstyle="square,pad=0.30", facecolor=ELEM_FACE,
                                        edgecolor="0.35", lw=0.6))
    if node_numbers:
        for i, p in enumerate(drawn, start=1):
            # sized in points, like the number it holds, so the disc fits the
            # label whatever units the model happens to be in
            ax.plot(*p, "o", ms=1.9*fontsize + 2.0*(len(str(i)) - 1),
                    mfc=NODE_FACE, mec="0.25", mew=0.8, zorder=5)
            ax.text(*p, str(i), ha="center", va="center", fontsize=fontsize,
                    zorder=6)

    if values is not None and colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = ax.figure.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
        if value_label:
            cb.set_label(value_label)
    if title:
        ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.autoscale_view()
    ax.margins(0.12, 0.18)
    return ax


def animate_truss(nodes, elements, displacements, presc=(), loads=None, *,
                  scale_max=80.0, frames=20, interval=120, figsize=(8, 4.5),
                  title="deformation scale {scale:.0f}x", ax=None, **kwargs):
    """Grow the deformation from nothing up to ``scale_max``.

    The undeformed shape is not drawn, so the truss is seen moving on its own.
    The axes are held still for the whole run, so nothing jumps between frames.

        anim = animate_truss(nodes, elements, U, presc=presc)
        HTML(anim.to_jshtml())

    Parameters
    ----------
    displacements : (n, 2) or (2n,) array
        The solved displacements, drawn at a scale that runs from 0 to
        ``scale_max`` over ``frames`` frames.
    title : str
        Formatted with ``scale`` for each frame. Pass ``None`` for no title.
    **kwargs
        Passed on to :func:`draw_truss`.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Turn it into something a notebook will show with ``.to_jshtml()``.
    """
    kwargs.pop("display_undeformed", None)
    kwargs.pop("scale", None)
    values = kwargs.get("values")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    fig = ax.figure

    scales = np.linspace(0.0, scale_max, frames)

    def frame(scale, colorbar):
        draw_truss(nodes, elements, presc=presc, loads=loads, ax=ax,
                   displacements=displacements, scale=scale,
                   display_undeformed=False, colorbar=colorbar,
                   title=None if title is None else title.format(scale=scale),
                   **kwargs)

    # Fix the view once, over both extremes, so the truss does not swim about.
    frame(0.0, False)
    lim = np.array([ax.get_xlim(), ax.get_ylim()])
    ax.clear()
    frame(scale_max, False)
    lim = np.array([[min(lim[0][0], ax.get_xlim()[0]), max(lim[0][1], ax.get_xlim()[1])],
                    [min(lim[1][0], ax.get_ylim()[0]), max(lim[1][1], ax.get_ylim()[1])]])
    ax.clear()

    # The colourbar belongs to the figure, so it is added once, not per frame.
    if values is not None:
        v = np.asarray(getattr(values, "data", values), dtype=float)
        sm = plt.cm.ScalarMappable(cmap=kwargs.get("cmap", "viridis"),
                                   norm=plt.Normalize(v.min(), v.max()))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
        if kwargs.get("value_label"):
            cb.set_label(kwargs["value_label"])

    def draw(k):
        ax.clear()
        frame(scales[k], False)
        ax.set_xlim(*lim[0])
        ax.set_ylim(*lim[1])
        ax.set_aspect("equal")
        ax.axis("off")
        return []

    anim = FuncAnimation(fig, draw, frames=frames, interval=interval, blit=False)
    plt.close(fig)          # otherwise the notebook shows a still as well
    return anim
