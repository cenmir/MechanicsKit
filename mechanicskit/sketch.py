r"""Sketch the parts a problem figure is made of.

Textbook figures show a machine or a structure with the loads on it, and every
figure in the book is drawn with the same handful of parts: a fixed surface,
a pin, a link, a spring, a gear, an arrow for a force, an arc for an angle and
a dimension line. This module draws each one, in the palette ``draw_truss``
already uses, so that a figure is a short script rather than a drawing.

    import matplotlib.pyplot as plt
    from mechanicskit import sketch

    fig, ax = sketch.canvas()
    sketch.ground(ax, [(0, 0), (1, 0)])
    sketch.link(ax, (0.5, 0), (1.0, 0.8))
    sketch.pin(ax, (0.5, 0))
    sketch.force(ax, (1.0, 0.8), (0, -1), 0.4, r'$P$')
    sketch.angle(ax, (0.5, 0), 0.25, 0, 58, r'$58^\circ$')

Everything is drawn in data coordinates with ``ax`` as the first argument,
the way ``pin_support`` and ``roller_support`` are. Points are ``(x, y)``
pairs; a SymPy column vector or a longer array is cut down to its first two
entries, so the vectors of a calculation can be handed straight in.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

from .truss import BODY, GROUND, EDGE, LOAD

__all__ = [
    "BODY", "GROUND", "EDGE", "LOAD", "BLUE", "GREEN", "GREY", "STEEL",
    "canvas", "ground", "body", "link", "pin", "spring", "gear", "hub",
    "force", "angle", "dimension", "axes", "guide", "rotation", "label",
    "gear_outline", "unit", "normal",
]

BLUE = "#4472c4"          # displacements, degrees of freedom, coordinate axes
GREEN = "#2e7d32"         # internal forces and rotations
GREY = "0.35"             # construction lines, angle arcs, dimensions
STEEL = "#bcd4e3"         # machine parts: gears, cylinders, pistons
STEEL_EDGE = "#236b8e"    # their outline, the gear animation's fill colour

LW = 1.1                  # outline weight of a filled part


def _xy(p):
    """A point as a length-2 float array, whatever it was given as."""
    if hasattr(p, "tolist"):
        p = p.tolist()
    return np.asarray(p, dtype=float).flatten()[:2]


def _unit(v):
    v = _xy(v)
    return v / np.linalg.norm(v)


def _normal(v):
    """The unit vector 90 degrees anticlockwise from ``v``."""
    u = _unit(v)
    return np.array([-u[1], u[0]])


def unit(v):
    """The unit vector along ``v``, in the plane."""
    return _unit(v)


def normal(v):
    """The unit vector 90 degrees anticlockwise from ``v``."""
    return _normal(v)


def canvas(figsize=(5, 4)):
    """A figure and an axes with equal aspect and no frame, ready to sketch on."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# --- surfaces and bodies ------------------------------------------------------

def ground(ax, points, depth=0.08, zorder=1):
    """A fixed surface: a polyline whose solid side fades away.

    Walk along ``points`` in order; the solid is on your right. A floor is
    therefore drawn left to right, a ceiling right to left, a wall whose solid
    lies to its left top to bottom, and one whose solid lies to its right
    bottom to top. ``depth`` is how far the
    shading reaches into the solid, in data units.
    """
    pts = [_xy(p) for p in points]
    n = 24
    for p0, p1 in zip(pts[:-1], pts[1:]):
        inward = -_normal(p1 - p0)                  # to the right of travel
        for k in range(n):
            d0, d1 = depth*k/n, depth*(k + 1)/n
            quad = [p0 + inward*d0, p1 + inward*d0, p1 + inward*d1, p0 + inward*d1]
            ax.add_patch(Polygon(quad, closed=True, facecolor=GROUND, edgecolor="none",
                                 alpha=(1 - k/n)*0.9, zorder=zorder))
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=EDGE, lw=LW, solid_capstyle="round", zorder=zorder + 1)


def body(ax, points, facecolor=BODY, edgecolor=EDGE, zorder=2, **kwargs):
    """A filled polygon: a bracket, a block, a pedestal."""
    pts = [_xy(p) for p in points]
    patch = Polygon(pts, closed=True, facecolor=facecolor, edgecolor=edgecolor,
                    lw=LW, zorder=zorder, **kwargs)
    ax.add_patch(patch)
    return patch


def link(ax, p0, p1, width=0.1, facecolor=BODY, edgecolor=EDGE, zorder=2,
         round_ends=True):
    """A rigid bar between two points, drawn as a stadium of the given width."""
    p0, p1 = _xy(p0), _xy(p1)
    a, nrm, r = _unit(p1 - p0), _normal(p1 - p0), width/2
    if round_ends:
        t = np.linspace(-np.pi/2, np.pi/2, 24)
        end1 = [p1 + r*(np.cos(s)*a + np.sin(s)*nrm) for s in t]
        end0 = [p0 - r*(np.cos(s)*a + np.sin(s)*nrm) for s in t]
        pts = end1 + end0
    else:
        pts = [p0 + r*nrm, p1 + r*nrm, p1 - r*nrm, p0 - r*nrm]
    return body(ax, pts, facecolor=facecolor, edgecolor=edgecolor, zorder=zorder)


def pin(ax, xy, r=0.03, zorder=4):
    """A pin joint: the white-faced disc the truss supports use."""
    patch = Circle(_xy(xy), r, facecolor="white", edgecolor=EDGE, lw=0.9, zorder=zorder)
    ax.add_patch(patch)
    return patch


def spring(ax, p0, p1, coils=6, width=0.08, lead=None, color=EDGE, lw=1.3, zorder=2):
    """A coil spring from ``p0`` to ``p1``: straight leads and a zigzag between."""
    p0, p1 = _xy(p0), _xy(p1)
    L = np.linalg.norm(p1 - p0)
    a, nrm = _unit(p1 - p0), _normal(p1 - p0)
    lead = 0.12*L if lead is None else lead
    inner = np.linspace(lead, L - lead, 2*coils + 1)
    pts = [p0, p0 + lead*a]
    for k, t in enumerate(inner[1:-1], start=1):
        pts.append(p0 + t*a + (width/2)*(1 if k % 2 else -1)*nrm)
    pts += [p0 + (L - lead)*a, p1]
    xs, ys = zip(*pts)
    return ax.plot(xs, ys, color=color, lw=lw, solid_joinstyle="miter", zorder=zorder)


# --- gears ----------------------------------------------------------------------

def gear_outline(N, module=1.0, phi_deg=20.0, n_pts=120):
    """The closed outline of an involute spur gear, centred at the origin.

    A tooth is centred on the positive x axis. Returns an (n, 2) array. Below
    the base circle the flank continues as a radial line, so the same curve
    serves gears with few teeth, whose root circle lies inside the base circle.
    """
    phi = np.radians(phi_deg)
    pr = module*N/2
    br, tr, rr = pr*np.cos(phi), module*(N + 2)/2, pr - 1.25*module
    t_tip = np.sqrt((tr/br)**2 - 1)

    v = np.linspace(-1, 1, n_pts)*t_tip
    v_pos, v_neg = (v + np.abs(v))/2, (v - np.abs(v))/2
    xc = br*(np.sin(v_pos) - v_pos*np.cos(v_pos))
    yc = br*(np.cos(v_pos) + v_pos*np.sin(v_pos)) + v_neg*br
    rc = np.hypot(xc, yc)
    keep = (rc >= rr - 1e-9) & (rc <= tr + 1e-9)
    xc, yc = xc[keep].copy(), yc[keep].copy()
    xc[0], yc[0] = xc[0]*rr/np.hypot(xc[0], yc[0]), yc[0]*rr/np.hypot(xc[0], yc[0])
    xc[-1], yc[-1] = xc[-1]*tr/np.hypot(xc[-1], yc[-1]), yc[-1]*tr/np.hypot(xc[-1], yc[-1])

    alpha_p = np.sqrt((pr/br)**2 - 1)
    inv_at_pitch = np.arctan2(br*(np.sin(alpha_p) - alpha_p*np.cos(alpha_p)),
                              br*(np.cos(alpha_p) + alpha_p*np.sin(alpha_p)))
    rot = np.pi/(2*N) + inv_at_pitch
    pitch = 2*np.pi/N

    def turned(x, y, ang):
        c, s = np.cos(ang), np.sin(ang)
        return c*x - s*y, s*x + c*y

    out = []
    for k in range(N):
        ang = k*pitch
        rx, ry = turned(-xc, yc, ang - rot)             # right flank, root to tip
        lx, ly = turned(xc, yc, ang + rot)              # left flank, root to tip
        out.append(np.column_stack([rx, ry]))
        th0, th1 = np.arctan2(ry[-1], rx[-1]), np.arctan2(ly[-1], lx[-1])
        th = np.linspace(th0, th0 + (th1 - th0) % (2*np.pi), 8)
        out.append(np.column_stack([tr*np.cos(th), tr*np.sin(th)]))
        out.append(np.column_stack([lx[::-1], ly[::-1]]))
        nx, ny = turned(-xc[0], yc[0], ang + pitch - rot)
        th0, th1 = np.arctan2(ly[0], lx[0]), np.arctan2(ny, nx)
        th = np.linspace(th0, th0 + (th1 - th0) % (2*np.pi), 8)
        out.append(np.column_stack([rr*np.cos(th), rr*np.sin(th)]))
    return np.vstack(out)


def gear(ax, centre, N, module=1.0, angle=0.0, phi_deg=20.0, facecolor=STEEL,
         edgecolor=STEEL_EDGE, zorder=2, hub_ratio=0.18):
    """A spur gear with ``N`` teeth, turned by ``angle`` degrees, with a hub.

    The pitch radius is ``module*N/2``; two gears mesh when their centres are
    that far apart in sum and one of them is turned by half a tooth.
    """
    c = _xy(centre)
    t = np.radians(angle)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    pts = gear_outline(N, module, phi_deg) @ R.T + c
    patch = Polygon(pts, closed=True, facecolor=facecolor, edgecolor=edgecolor,
                    lw=LW, zorder=zorder)
    ax.add_patch(patch)
    hub(ax, c, hub_ratio*module*N/2, zorder=zorder + 1)
    return patch


def hub(ax, centre, r, zorder=3):
    """A shaft end: a white disc with centre lines through it."""
    c = _xy(centre)
    ax.add_patch(Circle(c, r, facecolor="white", edgecolor=EDGE, lw=0.9, zorder=zorder))
    for d in (np.array([1, 0]), np.array([0, 1])):
        ax.plot(*zip(c - 1.6*r*d, c + 1.6*r*d), color=GREY, lw=0.6, zorder=zorder + 1)


# --- annotation -------------------------------------------------------------

def force(ax, point, direction, length, text=None, color=LOAD, head=False,
          offset=None, fontsize=12, lw=2.0, zorder=6):
    """A force arrow of the given length along ``direction``.

    The arrow starts at ``point``. With ``head=True`` it ends there instead,
    which is how a push is drawn. The label goes beside the free end, moved by
    ``offset`` if that is given.
    """
    p, u = _xy(point), _unit(direction)
    tail, tip = (p - length*u, p) if head else (p, p + length*u)
    ax.add_patch(FancyArrowPatch(tail, tip, arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0,
                                 zorder=zorder))
    if text is not None:
        free = tail if head else tip
        off = _xy(offset) if offset is not None else 0.35*length*u*(-1 if head else 1)
        label(ax, free + off, text, color=color, fontsize=fontsize, zorder=zorder + 1)


def angle(ax, centre, r, a0, a1, text=None, color=GREY, fontsize=11, text_r=None,
          zorder=3):
    """An arc from ``a0`` to ``a1`` degrees, anticlockwise, with its label."""
    c = _xy(centre)
    th = np.radians(np.linspace(a0, a1, 40))
    ax.plot(c[0] + r*np.cos(th), c[1] + r*np.sin(th), color=color, lw=0.9, zorder=zorder)
    if text is not None:
        tm = np.radians((a0 + a1)/2)
        rt = 1.45*r if text_r is None else text_r
        label(ax, c + rt*np.array([np.cos(tm), np.sin(tm)]), text, color="black",
              fontsize=fontsize, zorder=zorder + 1)


def dimension(ax, p0, p1, offset, text, color=GREY, fontsize=11, gap=0.02,
              text_side=1, zorder=3):
    """A dimension between two points, drawn ``offset`` to the left of p0 to p1.

    Two extension lines run out to the dimension line, which carries a head
    at each end. A negative ``offset`` puts it on the other side.
    """
    p0, p1 = _xy(p0), _xy(p1)
    nrm = _normal(p1 - p0)
    q0, q1 = p0 + offset*nrm, p1 + offset*nrm
    ext = np.sign(offset)*gap
    over = np.sign(offset)*0.6*gap
    for p, q in ((p0, q0), (p1, q1)):
        ax.plot(*zip(p + ext*nrm, q + over*nrm), color=color, lw=0.7, zorder=zorder)
    ax.add_patch(FancyArrowPatch(q0, q1, arrowstyle="<|-|>", mutation_scale=9,
                                 color=color, lw=0.8, shrinkA=0, shrinkB=0,
                                 zorder=zorder))
    mid = (q0 + q1)/2 + text_side*np.sign(offset)*2.2*gap*nrm
    label(ax, mid, text, color="black", fontsize=fontsize, zorder=zorder + 1)


def axes(ax, origin, length, angle_deg=0.0, labels=("$x$", "$y$"), color=EDGE,
         ls="-", fontsize=12, zorder=3):
    """A pair of coordinate axes, turned ``angle_deg`` anticlockwise.

    ``length`` is one number for both axes or a pair ``(lx, ly)``.
    """
    o = _xy(origin)
    t = np.radians(angle_deg)
    ex, ey = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
    lx, ly = (length, length) if np.isscalar(length) else length
    for d, L, lab in ((ex, lx, labels[0]), (ey, ly, labels[1])):
        ax.add_patch(FancyArrowPatch(o, o + L*d, arrowstyle="-|>",
                                     mutation_scale=12, color=color, lw=1.2, ls=ls,
                                     shrinkA=0, shrinkB=0, zorder=zorder))
        label(ax, o + (L + 0.05*max(lx, ly))*d + 0.06*max(lx, ly)*(ex if d is ey else ey),
              lab, color=color, fontsize=fontsize, zorder=zorder)


def guide(ax, p0, p1, color=GREY, ls="--", lw=0.8, zorder=2):
    """A construction line: the horizontal a slope is measured from, a line of action."""
    p0, p1 = _xy(p0), _xy(p1)
    return ax.plot(*zip(p0, p1), color=color, ls=ls, lw=lw, zorder=zorder)


def rotation(ax, centre, r, a0=60, a1=120, color=GREEN, lw=1.6, zorder=5):
    """A curved arrow showing a sense of rotation, from ``a0`` to ``a1`` degrees.

    Give ``a1 < a0`` for a clockwise arrow.
    """
    c = _xy(centre)
    th = np.radians(np.linspace(a0, a1, 30))
    xs, ys = c[0] + r*np.cos(th), c[1] + r*np.sin(th)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, zorder=zorder)
    ax.add_patch(FancyArrowPatch((xs[-2], ys[-2]), (xs[-1], ys[-1]),
                                 arrowstyle="-|>", mutation_scale=12, color=color,
                                 lw=lw, shrinkA=0, shrinkB=0, zorder=zorder))


def label(ax, xy, text, color="black", fontsize=12, zorder=7, **kwargs):
    """Centred text with a white backing so it reads over whatever it sits on."""
    kw = dict(ha="center", va="center", fontsize=fontsize, color=color, zorder=zorder,
              bbox=dict(boxstyle="square,pad=0.08", facecolor="white",
                        edgecolor="none", alpha=0.85))
    kw.update(kwargs)
    return ax.text(*_xy(xy), text, **kw)
