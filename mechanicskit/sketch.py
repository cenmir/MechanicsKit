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
from matplotlib import patheffects
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

from .truss import BODY, GROUND, EDGE, LOAD

__all__ = [
    "BODY", "GROUND", "EDGE", "LOAD", "BLUE", "GREEN", "GREY", "STEEL", "STEEL_EDGE",
    "LABEL_HALO",
    "canvas", "ground", "body", "link", "pin", "spring", "coil", "dashpot", "gas_spring",
    "helix_centres", "helical_spring",
    "gear", "hub",
    "box", "rounded_rect", "trapezoid", "ellipse", "cog",
    "force", "moment_vector", "angle", "dimension", "axes", "triad", "guide", "centreline", "rotation",
    "curl", "radius", "leader", "arrow", "unit_vectors", "direction_line", "break_line",
    "label", "gear_outline", "unit", "normal", "polar", "rot", "arc", "ccw",
    "mirror_outline", "outward",
]

BLUE = "#4472c4"          # displacements, degrees of freedom, coordinate axes
GREEN = "#2e7d32"         # internal forces and rotations
GREY = "0.35"             # construction lines, angle arcs, dimensions
STEEL = "#bcd4e3"         # machine parts: gears, cylinders, pistons
STEEL_EDGE = "#236b8e"    # their outline, the gear animation's fill colour

LW = 1.1                  # outline weight of a filled part

# Every label carries a thin white outline unless asked otherwise. Set this to False
# to turn the outline off for a whole figure or script; halo= on a call overrides it.
LABEL_HALO = True


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


def polar(deg, r=1.0):
    """The vector of length ``r`` at ``deg`` degrees from the x axis, anticlockwise."""
    t = np.radians(deg)
    return r*np.array([np.cos(t), np.sin(t)])


def rot(deg):
    """The 2x2 matrix that turns a vector ``deg`` degrees anticlockwise."""
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def arc(centre, r, a0, a1, n=24):
    """Points on a circular arc from ``a0`` to ``a1`` degrees, as an (n, 2) array."""
    c = _xy(centre)
    t = np.radians(np.linspace(a0, a1, n))
    return np.c_[c[0] + r*np.cos(t), c[1] + r*np.sin(t)]


def ccw(points):
    """The closed outline ``points``, reordered to run anticlockwise if it did not."""
    pts = np.asarray(points, float)
    area = np.sum(pts[:, 0]*np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1)*pts[:, 1])
    return pts if area > 0 else pts[::-1]


def mirror_outline(top):
    """A closed outline symmetric about the x axis, from its upper half ``top``.

    ``top`` runs left to right; the result runs along the mirrored lower half from
    right to left and then back along ``top``, the profile of a turned part.
    """
    top = np.asarray(top, float)
    return np.vstack([top[::-1]*[1, -1], top])


def outward(direction, point, centre):
    """``direction`` or its opposite, whichever points from ``centre`` past ``point``.

    Used to put a label or a load on the outside of a structure.
    """
    d = _xy(direction)
    return d if np.dot(_xy(point) - _xy(centre), d) >= 0 else -d


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


def coil(ax, p0, p1, coils=2, width=0.2, color=EDGE, lw=1.3, zorder=2):
    """A smooth spring from ``p0`` to ``p1``: one S-shaped Bezier wave per coil.

    The rounded counterpart of ``spring``, as FBD Lab draws it.
    """
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    p0, p1 = _xy(p0), _xy(p1)
    L = np.linalg.norm(p1 - p0)
    a, nrm = _unit(p1 - p0), _normal(p1 - p0)
    half = 2*coils
    s = L/half
    verts, codes = [p0], [Path.MOVETO]
    for k in range(half):
        side = (width/2)*(1 if k % 2 == 0 else -1)
        verts += [p0 + (k + 1/3)*s*a + side*nrm, p0 + (k + 2/3)*s*a + side*nrm,
                  p0 + (k + 1)*s*a]
        codes += [Path.CURVE4]*3
    patch = PathPatch(Path(verts, codes), facecolor="none", edgecolor=color, lw=lw,
                      zorder=zorder)
    ax.add_patch(patch)
    return patch


def dashpot(ax, p0, p1, width=0.5, lead=0.15, cup=0.45, color=EDGE, lw=1.3, plate_lw=2.2,
            zorder=3):
    """A damper from ``p0`` (the cylinder end) to ``p1`` (the rod end), at any angle.

    ``lead`` and ``cup`` are the rod before the cylinder and the cylinder's length,
    as fractions of the whole length; the cylinder is open toward the rod.
    """
    p0, p1 = _xy(p0), _xy(p1)
    L = np.linalg.norm(p1 - p0)
    a, n = _unit(p1 - p0), _normal(p1 - p0)
    c0, c1 = p0 + lead*L*a, p0 + (lead + cup)*L*a
    ax.plot(*zip(p0, c0), color=color, lw=lw, zorder=zorder)
    ax.plot(*zip(c1 + width/2*n, c0 + width/2*n, c0 - width/2*n, c1 - width/2*n),
            color=color, lw=lw, zorder=zorder)
    q = c0 + 0.6*cup*L*a                                # the piston plate
    ax.plot(*zip(q + 0.38*width*n, q - 0.38*width*n), color=color, lw=plate_lw,
            zorder=zorder)
    ax.plot(*zip(q, p1), color=color, lw=lw, zorder=zorder)


def gas_spring(ax, p0, p1, width=0.12, tube=0.55, rod=0.4, eye=None, facecolor=STEEL,
               edgecolor=STEEL_EDGE, zorder=3):
    """A gas spring or hydraulic cylinder from ``p0`` (tube end) to ``p1`` (rod end).

    The tube is ``width`` wide and ``tube`` of the whole length long, the rod is ``rod``
    times as wide, and each end carries a mounting eye of radius ``eye`` (default
    ``0.45*width``) centred on the end point, so ``p0`` and ``p1`` are the pin centres.
    """
    p0, p1 = _xy(p0), _xy(p1)
    L = np.linalg.norm(p1 - p0)
    a = _unit(p1 - p0)
    r = 0.45*width if eye is None else eye
    link(ax, p0 + tube*L*a - 0.5*width*a, p1, width=rod*width, facecolor=facecolor,
         edgecolor=edgecolor, zorder=zorder, round_ends=False)
    link(ax, p0 + r*a, p0 + tube*L*a, width=width, facecolor=facecolor,
         edgecolor=edgecolor, zorder=zorder + 0.1, round_ends=True)
    for p in (p0, p1):
        ax.add_patch(Circle(p, 1.6*r, facecolor=facecolor, edgecolor=edgecolor, lw=LW,
                            zorder=zorder + 0.2))
        pin(ax, p, r=0.7*r, zorder=zorder + 0.3)



def helix_centres(d, p, n_active, n_closed=1):
    """Heights of the wire centre of a helical spring at each half turn, bottom coil first.

    Even entries lie on the right of the axis, odd ones on the left. The ``n_closed``
    end coils at each end advance by ``d`` per turn, the active coils by the pitch ``p``.
    The bottom of the wire lies at height 0.
    """
    steps = [d/2]*(2*n_closed) + [p/2]*(2*n_active) + [d/2]*(2*n_closed)
    return d/2 + np.r_[0, np.cumsum(steps)]


def helical_spring(ax, x0, D, d, z, z0=0.0, cut_top=False, facecolor=STEEL,
                   edgecolor=STEEL_EDGE, zorder=3, ground=None):
    """Side view of a helical spring of coil diameter ``D`` and wire ``d``, as a drawing shows it.

    ``z`` are the wire-centre heights of :func:`helix_centres`. Every crossing of the
    outline is drawn as a section of the wire, and the front half of each turn as a band
    from the right section up to the next left one. ``cut_top`` leaves the last section
    open, for a spring cut there. ``ground=(z_bottom, z_top)`` grinds the ends flat: the
    wire is cut off below ``z_bottom`` and above ``z_top`` (heights in the frame of ``z``),
    and each cut is closed by a straight edge, the ground face. Returns the section centres.
    """
    R = D/2
    pts = [(x0 + (R if k % 2 == 0 else -R), z0 + zk) for k, zk in enumerate(z)]
    clip = None
    if ground is not None:
        lo, hi = z0 + ground[0], z0 + ground[1]
        clip = Polygon([(x0 - R - d, lo), (x0 + R + d, lo), (x0 + R + d, hi), (x0 - R - d, hi)],
                       closed=True, facecolor="none", edgecolor="none")
        ax.add_patch(clip)
    patches = []
    for (xa, za), (xb, zb) in zip(pts[0::2], pts[1::2]):
        t = np.array([xb - xa, zb - za])
        nrm = np.array([-t[1], t[0]])/np.hypot(*t)*d/2
        patches.append(body(ax, [(xa, za) + nrm, (xb, zb) + nrm, (xb, zb) - nrm, (xa, za) - nrm],
                            facecolor=facecolor, edgecolor=edgecolor, zorder=zorder))
    last = len(pts) - 1 if cut_top else len(pts)
    for x, zz in pts[:last]:
        patches.append(ax.add_patch(Circle((x, zz), d/2, facecolor=facecolor,
                                           edgecolor=edgecolor, lw=LW, zorder=zorder + 1)))
    if clip is not None:
        for patch in patches:
            patch.set_clip_path(clip)
        # the ground faces: a straight edge wherever the cut passes through the wire,
        # in the round sections and in the bands between them
        for zc in (lo, hi):
            spans = []
            for x, zz in pts[:last]:
                h = abs(zc - zz)
                if h < d/2:
                    w = np.sqrt((d/2)**2 - h**2)
                    spans.append((x - w, x + w))
            for patch in patches:
                if isinstance(patch, Circle):
                    continue
                xy = patch.get_xy()
                xs = [xa + (zc - za)*(xb - xa)/(zb - za)
                      for (xa, za), (xb, zb) in zip(xy[:-1], xy[1:])
                      if (za - zc)*(zb - zc) < 0]
                if len(xs) >= 2:
                    spans.append((min(xs), max(xs)))
            for xa, xb in spans:
                ax.plot([xa, xb], [zc, zc], color=edgecolor, lw=LW, solid_capstyle="butt",
                        zorder=zorder + 2)
    return pts

def box(ax, xy, w, h, angle=0.0, centred=False, **kwargs):
    """A rectangle of width ``w`` and height ``h``: a block, a crate, a slider.

    ``xy`` is the lower left corner, or the centre with ``centred=True``; the
    rectangle is turned ``angle`` degrees about that point. Takes the keyword
    arguments of ``body``.
    """
    x0 = np.array([-w/2, -h/2]) if centred else np.zeros(2)
    pts = [x0, x0 + (w, 0), x0 + (w, h), x0 + (0, h)]
    R = rot(angle)
    return body(ax, [_xy(xy) + R @ p for p in pts], **kwargs)


def rounded_rect(xy, w, h, r, m=12):
    """The outline of a rectangle with corner ``xy``, its corners rounded with radius r."""
    x0, y0 = _xy(xy)
    x1, y1 = x0 + w, y0 + h
    pts = []
    for cx, cy, a0 in ((x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180),
                       (x1 - r, y0 + r, 270)):
        for a in np.radians(np.linspace(a0, a0 + 90, m)):
            pts.append((cx + r*np.cos(a), cy + r*np.sin(a)))
    return np.array(pts)


def trapezoid(ax, centre, w_bottom, w_top, h, angle=0.0, **kwargs):
    """A symmetric trapezoid about ``centre``, turned ``angle`` degrees: a wedge, a pad."""
    pts = [(-w_bottom/2, -h/2), (w_bottom/2, -h/2), (w_top/2, h/2), (-w_top/2, h/2)]
    R = rot(angle)
    return body(ax, [_xy(centre) + R @ np.array(p) for p in pts], **kwargs)


def ellipse(ax, centre, w, h, angle=0.0, facecolor=BODY, edgecolor=EDGE, zorder=2, **kwargs):
    """An ellipse of width ``w`` and height ``h`` turned ``angle`` degrees: a disc, a wheel."""
    from matplotlib.patches import Ellipse
    patch = Ellipse(_xy(centre), w, h, angle=angle, facecolor=facecolor,
                    edgecolor=edgecolor, lw=LW, zorder=zorder, **kwargs)
    ax.add_patch(patch)
    return patch


def cog(ax, p, r=0.09, zorder=6):
    """The centre-of-mass symbol: a circle with its first and third quarters black."""
    p = _xy(p)
    ax.add_patch(Circle(p, r, facecolor="white", edgecolor="black", lw=1.0, zorder=zorder))
    for a0 in (0, 180):
        t = np.radians(np.linspace(a0, a0 + 90, 20))
        ax.add_patch(Polygon(np.r_[[p], np.c_[p[0] + r*np.cos(t), p[1] + r*np.sin(t)]],
                             closed=True, facecolor="black", lw=0, zorder=zorder + 1))


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
          offset=None, fontsize=12, lw=2.0, zorder=6, text_color="black"):
    """A force arrow of the given length along ``direction``.

    The arrow starts at ``point``. With ``head=True`` it ends there instead,
    which is how a push is drawn. The label goes beside the free end, moved by
    ``offset`` if that is given. The book's rule is that a label takes the colour
    of its arrow, so pass ``text_color`` (``LOAD`` for an applied force, ``BLUE``
    for a resultant, ``GREEN`` for an internal force); the default is black.
    """
    p, u = _xy(point), _unit(direction)
    tail, tip = (p - length*u, p) if head else (p, p + length*u)
    ax.add_patch(FancyArrowPatch(tail, tip, arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0,
                                 zorder=zorder))
    if text is not None:
        free = tail if head else tip
        off = _xy(offset) if offset is not None else 0.35*length*u*(-1 if head else 1)
        label(ax, free + off, text, color=text_color, fontsize=fontsize, zorder=zorder + 1)



def moment_vector(ax, point, direction, length, text=None, color=LOAD, text_color=None,
                  head_gap=None, offset=None, fontsize=12, lw=1.6, ms=13, zorder=6):
    """A moment or torque drawn as a vector with a double head, from ``point`` along ``direction``.

    The sense of rotation follows the right-hand rule about the arrow, which reads the same
    from any side, unlike a curved arrow drawn around an axis. The second head sits
    ``head_gap`` (default 0.18 of the length) behind the first. The label goes beside the
    middle of the arrow, moved by ``offset``, in ``text_color`` (default: the arrow's colour).
    """
    p, u = _xy(point), _unit(direction)
    gap = 0.18*length if head_gap is None else head_gap
    for k in (0.0, gap):
        ax.add_patch(FancyArrowPatch(p, p + (length - k)*u, arrowstyle="-|>",
                                     mutation_scale=ms, color=color, lw=lw, shrinkA=0,
                                     shrinkB=0, zorder=zorder))
    if text is not None:
        off = _xy(offset) if offset is not None else 0.25*length*_normal(u)
        label(ax, p + 0.5*length*u + off, text, color=text_color or color, fontsize=fontsize,
              zorder=zorder + 1)

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
              text_side=1, zorder=3, text_offset=None, text_at=0.5, text_shift=(0, 0),
              **label_kwargs):
    """A dimension between two points, drawn ``offset`` to the left of p0 to p1.

    Two extension lines run out to the dimension line, which carries a head
    at each end. A negative ``offset`` puts it on the other side, and
    ``offset=0`` draws the dimension line on p0 to p1 itself, without extension
    lines: a span marked along an edge, or a leader.

    The value sits ``text_at`` of the way along the line (0.5, the middle), a
    distance from it set by ``gap`` and ``text_side``, or exactly
    ``text_offset`` to the left of p0 to p1 when that is given; ``text_shift``
    moves it further, in data units. The value has a white box behind it, which
    interrupts the line the way a drawing does; ``bg=None`` gives the outline
    instead. Other keyword arguments go to ``label``.
    """
    p0, p1 = _xy(p0), _xy(p1)
    nrm = _normal(p1 - p0)
    q0, q1 = p0 + offset*nrm, p1 + offset*nrm
    if offset != 0:
        ext = np.sign(offset)*gap
        over = np.sign(offset)*0.6*gap
        for p, q in ((p0, q0), (p1, q1)):
            ax.plot(*zip(p + ext*nrm, q + over*nrm), color=color, lw=0.7, zorder=zorder)
    ax.add_patch(FancyArrowPatch(q0, q1, arrowstyle="<|-|>", mutation_scale=9,
                                 color=color, lw=0.8, shrinkA=0, shrinkB=0,
                                 zorder=zorder))
    if not text:
        return
    at = q0 + text_at*(q1 - q0)
    if text_offset is None:
        at = at + text_side*np.sign(offset)*2.2*gap*nrm
    else:
        at = at + text_offset*nrm
    label_kwargs.setdefault("bg", "white")
    label_kwargs.setdefault("bgalpha", 1.0)
    label(ax, at + _xy(text_shift), text, color="black", fontsize=fontsize,
          zorder=zorder + 1, **label_kwargs)


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


def label(ax, xy, text, color="black", fontsize=12, zorder=7, bg=None, bgalpha=0.85,
          halo=None, **kwargs):
    """Centred text with a thin white outline around the glyphs.

    The outline (``halo``) lets a label cross a line, a hatch or a tinted fill and
    still read, while hiding far less of the drawing than a box would. A label is
    still best placed clear of every line. ``halo=False`` gives plain text, and
    ``LABEL_HALO = False`` turns the outline off for every label. ``bg`` puts a
    box of that colour behind the text instead, with opacity ``bgalpha``: the
    drafting style for a value that sits on a dimension line. A ``bbox`` passed
    in ``kwargs`` overrides ``bg``.
    """
    kw = dict(ha="center", va="center", fontsize=fontsize, color=color, zorder=zorder)
    if bg is not None:
        kw["bbox"] = dict(boxstyle="square,pad=0.08", facecolor=bg, edgecolor="none",
                          alpha=bgalpha)
    if halo is None:
        halo = LABEL_HALO and bg is None and "bbox" not in kwargs
    if halo:
        kw["path_effects"] = [patheffects.withStroke(linewidth=3, foreground="white")]
    kw.update(kwargs)
    return ax.text(*_xy(xy), text, **kw)


def arrow(ax, p0, p1, color=EDGE, lw=1.3, ms=12, style="-|>", ls="-", rad=0.0, zorder=6):
    """A plain arrow from ``p0`` to ``p1``: a velocity, an axis, a displacement.

    ``ms`` is the head size in points, ``style`` a matplotlib arrow style
    (``"<|-|>"`` for heads at both ends, ``"-"`` for none) and ``rad`` bends it
    into an arc. A load is drawn with ``force`` instead.
    """
    kw = dict(connectionstyle=f"arc3,rad={rad}") if rad else {}
    patch = FancyArrowPatch(_xy(p0), _xy(p1), arrowstyle=style, mutation_scale=ms,
                            color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0, zorder=zorder,
                            **kw)
    ax.add_patch(patch)
    return patch


def unit_vectors(ax, p, pairs, length=1.0, fontsize=11, lw=1.4, zorder=6):
    """Black unit vectors from ``p``, such as e_t and e_n of natural coordinates.

    ``pairs`` holds (direction, label, label offset from the tip); the arrows are
    black and thinner than a load so that they do not read as forces.
    """
    for d, text, off in pairs:
        d = _unit(d)
        force(ax, p, d, length, color="black", lw=lw, zorder=zorder)
        label(ax, _xy(p) + length*d + _xy(off), text, fontsize=fontsize)


def triad(ax, origin, dirs=((1, 0), (0, 1)), labels=("$x$", "$y$"), length=0.5, gap=0.16,
          offsets=None, color="black", lw=1.0, ms=9, ls="-", fontsize=12, zorder=6):
    """Coordinate axes along any ``dirs``, each letter just past its arrow tip.

    Unlike ``axes`` the letters sit on the line of each axis, ``gap`` beyond the
    tip, clear of the arrowheads; ``offsets`` (one per axis) places them anywhere
    relative to the tip instead.
    """
    o = _xy(origin)
    for k, (d, text) in enumerate(zip(dirs, labels)):
        d = _unit(d)
        tip = o + length*d
        arrow(ax, o, tip, color=color, lw=lw, ms=ms, ls=ls, zorder=zorder)
        at = tip + gap*d if offsets is None else tip + _xy(offsets[k])
        if text:
            label(ax, at, text, color=color, fontsize=fontsize)


def curl(ax, centre, rx, ry, a0, a1, color=LOAD, lw=1.6, ms=13, n=60, zorder=6,
         back_zorder=None, back=1, head=1):
    """A curved arrow along an ellipse, head at ``a1``: a torque seen at an angle.

    A torque about a shaft seen from the side is a narrow ellipse beyond the
    shaft end (``rx`` about 0.3 of ``ry``, from -60 to 230 degrees). With
    ``back_zorder`` the arc wraps the shaft instead: the half on the side
    ``back`` (+1 right, -1 left) of the centre is drawn at ``back_zorder``,
    behind the shaft, and the rest at ``zorder``. ``head`` is how many points
    of the curve the arrowhead spans.
    """
    c = _xy(centre)
    th = np.radians(np.linspace(a0, a1, n))
    xs, ys = c[0] + rx*np.cos(th), c[1] + ry*np.sin(th)
    if back_zorder is None:
        ax.plot(xs[:-head], ys[:-head], color=color, lw=lw, zorder=zorder)
    else:
        behind = back*np.cos(th) > 0
        for mask, z in ((behind, back_zorder), (~behind, zorder)):
            ax.plot(xs, np.where(mask, ys, np.nan), color=color, lw=lw, zorder=z,
                    solid_capstyle="round")
    arrow(ax, (xs[-1 - head], ys[-1 - head]), (xs[-1], ys[-1]), color=color, lw=lw, ms=ms,
          zorder=zorder)


def radius(ax, centre, r, deg, text=None, at=None, color="black", lw=0.9, ms=10, fontsize=12,
           zorder=5):
    """A radius: an arrow from ``centre`` out to ``r`` at ``deg`` degrees, labelled at ``at``.

    Without ``at`` the label sits beside the middle of the arrow, on its left.
    """
    c = _xy(centre)
    u = polar(deg)
    arrow(ax, c, c + r*u, color=color, lw=lw, ms=ms, zorder=zorder)
    if text:
        at = c + 0.5*r*u + 0.12*r*_normal(u) if at is None else _xy(at)
        label(ax, at, text, fontsize=fontsize)


def leader(ax, point, start, text=None, gap=0.18, color=GREY, lw=0.8, ms=9, fontsize=12,
           zorder=6):
    """A leader: an arrow from ``start`` in to ``point`` on an edge, labelled beyond its tail.

    The label sits ``gap`` past ``start`` on the line from ``point``, the way a
    small radius or a fillet is called out.
    """
    p, q = _xy(point), _xy(start)
    arrow(ax, q, p, color=color, lw=lw, ms=ms, zorder=zorder)
    if text:
        label(ax, q + gap*_unit(q - p), text, fontsize=fontsize)


def centreline(ax, p0, p1, color=GREY, lw=0.8, ls=(0, (10, 3, 2, 3)), zorder=3):
    """A dash-dot centre line: the axis of a shaft, a line of symmetry."""
    return ax.plot(*zip(_xy(p0), _xy(p1)), color=color, lw=lw, ls=ls, zorder=zorder)


def direction_line(ax, p0, p1, head=True, color=EDGE, lw=1.2, head_length=None,
                   ls=(0, (5, 3)), zorder=4):
    """A dashed line of action from ``p0`` to ``p1``, with an open triangular head.

    The FBD Lab mark for a direction: where a force acts along, not the force.
    """
    p0, p1 = _xy(p0), _xy(p1)
    a, n = _unit(p1 - p0), _normal(p1 - p0)
    hl = 0.08*np.linalg.norm(p1 - p0) if head_length is None else head_length
    end = p1 - hl*a if head else p1
    ax.plot(*zip(p0, end), color=color, lw=lw, ls=ls, zorder=zorder)
    if head:
        ax.add_patch(Polygon([p1, end + 0.4*hl*n, end - 0.4*hl*n], closed=True, fill=False,
                             edgecolor=color, lw=lw, joinstyle="miter", zorder=zorder))


def break_line(ax, p0, p1, waves=1.5, amplitude=0.06, color=EDGE, lw=1.0, zorder=4):
    """A wavy break line from ``p0`` to ``p1``: where a part is cut off in the drawing."""
    p0, p1 = _xy(p0), _xy(p1)
    L = np.linalg.norm(p1 - p0)
    a, n = _unit(p1 - p0), _normal(p1 - p0)
    t = np.linspace(0, 1, max(40, int(30*waves)))
    pts = p0 + np.outer(t*L, a) + np.outer(amplitude*np.sin(2*np.pi*waves*t), n)
    return ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=zorder)
