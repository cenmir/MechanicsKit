"""Drawing parts for problem figures beyond mechanicskit.sketch, and the SVG saver.

    from mechanicskit.sketch_extra import save_svg, ground, shade

    save_svg      save a figure with named parts and a data-to-SVG map, for hand edits
    Proj          a fixed axonometric projection for the 3D problem figures
    shade         a polygon filled with a linear gradient, built from thin strips
    shaded_bar    a round-ended bar shaded across its thickness, like a cylinder
    shaded_rect   an axis-aligned rectangle shaded along x or y
    embed_svg     paste an external SVG drawing into a saved matplotlib SVG
    support_pin, support_roller, support_rollers, support_slider, pedestal, wall,
    hatched_wall, sliding_block, bracket
                  supports and blocks standing on the gradient ground
    tone          a face colour lit by a light direction, for 3D figures

The gradients are stacks of clipped strips, so the output stays pure vector
and renders the same in every viewer.

Hand edits. Figures saved with ``save_svg`` can be touched up in Inkscape and the
change carried back into the script, which stays the source of truth:

1. Open ``<figure>.svg`` in Inkscape. Every part is a group named after what it is:
   ``label-A`` for the label $A$, ``fancyarrow-03`` for the third arrow, and so on
   (Objects panel). Move, recolour, delete or draw, but do not resize the page.
2. Save as ``<figure>.edited.svg`` next to the original.
3. Run ``python -m mechanicskit.svg_roundtrip <figure>.svg``: it lists each part that moved (in
   data units), changed style, was deleted, or was drawn new.
4. Update the script to match, regenerate, and run the diff again until it reports
   nothing; then delete the ``.edited.svg``.
"""
import copy
import xml.etree.ElementTree as ET

import numpy as np
from matplotlib import patheffects
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
from mechanicskit import sketch as sk

# cylinder-like shading across a bar: dark edge, light band, dark edge
STEEL_STOPS = ['#6f93a8', '#d6e6ef', '#f4f9fc', '#c3d8e5', '#7fa3b8']
BRASS_STOPS = ['#9a7420', '#e3c66d', '#f6e5a4', '#d9b955', '#8f6a1a']
TAN_STOPS = ['#a88a63', '#dcc39f', '#efe0c6', '#d3b88f', '#9d7f58']
NUT_STOPS = ['#3f9fc2', '#86c3dc', '#cde7f3']     # a zinc-plated nut, dark on top
DARK_STEEL_STOPS = ['#3f5f73', '#8fb0c4', '#c9dde8', '#86a7bb', '#4b6b80']
# polished steel, as Mirza drew the hydraulic cylinder: (offset, colour) pairs
CHROME_STOPS = [(0, '#92b1d8'), (0.09, '#92b1d8'), (0.18, '#e5eaf4'), (0.25, '#afc3df'),
                (0.44, '#f7f9fc'), (0.58, '#d6cec9'), (1, '#6a6d70')]

# True: a shaded part is one SVG object with a real gradient, filled in by save_svg.
# False: the gradient is built from strips, for raster output such as the animations.
SVG_GRADIENTS = True
_GRADIENTS = []          # (patch, spec) waiting for save_svg to turn into <linearGradient>


def _stop_list(stops):
    """Stops as (offset, colour, opacity): colours spread evenly, or given with offsets."""
    out = []
    for k, st in enumerate(stops):
        if isinstance(st, str):
            out.append((k/(len(stops) - 1), st, 1.0))
        else:
            out.append((st[0], st[1], st[2] if len(st) > 2 else 1.0))
    return out


def _stops_to_colors(stops, n):
    """``n`` colours interpolated through the list of ``stops``."""
    rgb = np.array([to_rgb(c) for c in stops])
    t = np.linspace(0, 1, n)
    s = np.linspace(0, 1, len(stops))
    return [tuple(np.interp(t_, s, rgb[:, k]) for k in range(3)) for t_ in t]


def shade(ax, pts, stops, direction, n=36, edgecolor=sk.EDGE, lw=sk.LW, zorder=2):
    """Fill the polygon ``pts`` with a gradient along ``direction``.

    ``stops`` run from the far side to the near side of the polygon in the given
    direction: a list of colours spread evenly, or (offset, colour) pairs. In SVG
    output the polygon is a single object whose fill save_svg turns into a real
    linear gradient; with SVG_GRADIENTS off it is built from ``n`` strips.
    Returns the patch that carries the outline.
    """
    pts = np.asarray(pts, float)
    d = sk.unit(np.asarray(direction, float))
    proj = pts @ d
    lo, hi = proj.min(), proj.max()
    c = pts.mean(axis=0)
    stop_list = _stop_list(stops)
    if SVG_GRADIENTS:
        patch = Polygon(pts, closed=True, facecolor=stop_list[len(stop_list)//2][1],
                        edgecolor=edgecolor, lw=lw, zorder=zorder)
        ax.add_patch(patch)
        _GRADIENTS.append((patch, {'kind': 'linear', 'p0': c + (lo - c @ d)*d,
                                   'p1': c + (hi - c @ d)*d, 'stops': stop_list}))
        return patch
    nrm = sk.normal(d)
    span = pts @ nrm
    half = (span.max() - span.min())
    clip = Polygon(pts, closed=True, facecolor='none', edgecolor='none')
    ax.add_patch(clip)
    edges = np.linspace(lo, hi, n + 1)
    offs = [o for o, _, _ in stop_list]
    rgb = np.array([to_rgb(col) for _, col, _ in stop_list])
    for k in range(n):
        t = (k + 0.5)/n
        col = tuple(np.interp(t, offs, rgb[:, j]) for j in range(3))
        a, b = edges[k], edges[k + 1] + (hi - lo)/n*0.05        # a hair of overlap
        quad = [c + (a - c @ d)*d + s*half*nrm for s in (-1, 1)]
        quad += [c + (b - c @ d)*d + s*half*nrm for s in (1, -1)]
        strip = Polygon(quad, closed=True, facecolor=col, edgecolor='none',
                        zorder=zorder)
        strip.set_clip_path(clip)
        ax.add_patch(strip)
    outline = Polygon(pts, closed=True, facecolor='none', edgecolor=edgecolor, lw=lw,
                      zorder=zorder + 0.01)
    ax.add_patch(outline)
    return outline


def ground(ax, points, depth=0.08, zorder=1, lw=sk.LW):
    """A fixed surface, as mechanicskit's sk.ground, with the fade as one gradient.

    Walk along ``points``; the solid is on your right. Each segment is one object
    whose colour fades from the edge into the solid.
    """
    if not SVG_GRADIENTS:
        return sk.ground(ax, points, depth=depth, zorder=zorder)
    pts = [np.asarray(p, float) for p in points]
    for p0, p1 in zip(pts[:-1], pts[1:]):
        inward = -sk.normal(p1 - p0)
        quad = [p0, p1, p1 + inward*depth, p0 + inward*depth]
        patch = Polygon(quad, closed=True, facecolor=sk.GROUND, edgecolor='none',
                        zorder=zorder)
        ax.add_patch(patch)
        mid = (p0 + p1)/2
        _GRADIENTS.append((patch, {'kind': 'linear', 'p0': mid, 'p1': mid + inward*depth,
                                   'stops': [(0, sk.GROUND, 0.9), (1, sk.GROUND, 0.0)]}))
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=sk.EDGE, lw=lw, solid_capstyle='round', zorder=zorder + 1)


def stadium(p0, p1, width, round_ends=True, m=24):
    """The outline points of a bar from ``p0`` to ``p1`` with the given width."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    a, nrm, r = sk.unit(p1 - p0), sk.normal(p1 - p0), width/2
    if not round_ends:
        return [p0 + r*nrm, p1 + r*nrm, p1 - r*nrm, p0 - r*nrm]
    t = np.linspace(-np.pi/2, np.pi/2, m)
    end1 = [p1 + r*(np.cos(s)*a + np.sin(s)*nrm) for s in t]
    end0 = [p0 - r*(np.cos(s)*a + np.sin(s)*nrm) for s in t]
    return end1 + end0


def shaded_bar(ax, p0, p1, width, stops=STEEL_STOPS, round_ends=True, edgecolor=sk.EDGE,
               lw=sk.LW, zorder=2, n=36):
    """A bar from ``p0`` to ``p1`` shaded across its thickness."""
    pts = stadium(p0, p1, width, round_ends)
    return shade(ax, pts, stops, sk.normal(np.subtract(p1, p0)), n=n, edgecolor=edgecolor,
                 lw=lw, zorder=zorder)


def shaded_rect(ax, xy, w, h, stops=STEEL_STOPS, axis='y', **kwargs):
    """An axis-aligned rectangle with corner ``xy``, shaded along ``axis``."""
    x, y = xy
    pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    direction = (0, 1) if axis == 'y' else (1, 0)
    return shade(ax, pts, stops, direction, **kwargs)


def hex_nut(ax, face, direction, e, m, stops=NUT_STOPS, edgecolor=sk.EDGE, lw=sk.LW,
            zorder=4):
    """A hex nut or bolt head seen from the side, across the corners.

    ``face`` is the centre of the bearing face, on the bolt axis, and the nut stands
    ``m`` from it along ``direction``. ``e`` is the width across the corners, so the
    outline is ``e`` high and the edges between the three visible faces lie at
    +-e/4. The outer end has the usual 30 degree chamfer down to the width across
    the flats: straight lines at the outline's corners and, where the chamfer cone
    cuts the flat faces, an arc on each face. Shaded with ``stops`` from top to
    bottom. Returns the outline patch.
    """
    R = e/2
    r_c = R*np.sqrt(3)/2                       # the chamfer ends at the width across flats
    c = (R - r_c)*np.sqrt(3)                   # axial length of the chamfer at a corner
    u = sk.unit(np.asarray(direction, float))
    v = sk.normal(u)
    if v[1] < 0 or (v[1] == 0 and v[0] < 0):    # across the axis, upwards: dark side on top
        v = -v
    f = np.asarray(face, float)

    def P(a, b):
        return f + a*u + b*v

    outline = [P(0, -R), P(m - c, -R), P(m, -r_c), P(m, r_c), P(m - c, R), P(0, R)]
    patch = shade(ax, outline, stops, -v, edgecolor=edgecolor, lw=lw, zorder=zorder)
    # the chamfer cone r(a) = r_c + (m - a)/sqrt(3) meets each flat face where the
    # face point's distance from the axis equals r(a)
    corners = [R*np.array([np.cos(t), np.sin(t)]) for t in np.radians([-90, -30, 30, 90])]
    for q0, q1 in zip(corners[:-1], corners[1:]):
        t = np.linspace(0, 1, 40)
        q = np.outer(1 - t, q0) + np.outer(t, q1)
        a = m - np.sqrt(3)*(np.linalg.norm(q, axis=1) - r_c)
        pts = np.array([P(ai, qi[1]) for ai, qi in zip(a, q)])
        ax.plot(pts[:, 0], pts[:, 1], color=edgecolor, lw=lw, zorder=zorder + 0.02,
                solid_capstyle='round')
    for b in (-R/2, R/2):
        pts = np.array([P(0, b), P(m - c, b)])
        ax.plot(pts[:, 0], pts[:, 1], color=edgecolor, lw=lw, zorder=zorder + 0.02)
    return patch


# --- supports -----------------------------------------------------------------------
#
# Each is drawn with the gradient ground above. ``angle`` turns a support about the
# point it holds, anticlockwise in degrees; 0 is the upright pose described.

def _turn(points, about, angle):
    R = sk.rot(angle)
    c = np.asarray(about, float)
    return [c + R @ (np.asarray(p, float) - c) for p in points]


def support_pin(ax, p, h=0.55, w=0.6, ground_w=1.0, depth=0.18, r=0.07, angle=0.0,
                zorder=1):
    """A pin support: a triangle from the pin at ``p`` down to the ground.

    ``h`` and ``w`` are the triangle's height and base, ``ground_w`` the width of
    the ground under it.
    """
    x, y = p
    yb = y - h
    sk.body(ax, _turn([(x, y), (x + w/2, yb), (x - w/2, yb)], p, angle), zorder=zorder)
    ground(ax, _turn([(x - ground_w/2, yb), (x + ground_w/2, yb)], p, angle), depth=depth)
    sk.pin(ax, p, r, zorder=5)


def support_roller(ax, p, r=0.22, ground_w=1.0, depth=0.18, angle=0.0, zorder=1):
    """A roller support: one wheel of radius ``r`` under ``p``, on the ground."""
    x, y = p
    c = _turn([(x, y - r)], p, angle)[0]
    ax.add_patch(Circle(c, r, facecolor=sk.BODY, edgecolor=sk.EDGE, lw=sk.LW, zorder=zorder))
    ground(ax, _turn([(x - ground_w/2, y - 2*r), (x + ground_w/2, y - 2*r)], p, angle),
           depth=depth)


def support_rollers(ax, p0, p1, n, side, r=0.7, depth=1.4, zorder=3):
    """A row of ``n`` rollers under the edge p0-p1, on the side given by the unit vector
    ``side``, with the fixed surface they run on behind them."""
    p0, p1, side = (np.asarray(x, float) for x in (p0, p1, side))
    for s in np.linspace(0.1, 0.9, n):
        c = p0 + s*(p1 - p0) + r*side
        ax.add_patch(Circle(c, r, facecolor='white', edgecolor=sk.EDGE, lw=1.0, zorder=zorder))
    g0, g1 = p0 + 2*r*side, p1 + 2*r*side
    v = g1 - g0
    if v[1]*side[0] - v[0]*side[1] < 0:              # walk it with the solid behind
        g0, g1 = g1, g0
    ground(ax, [g0, g1], depth=depth)


def support_slider(ax, p, base=0.42, half=0.22, wheel=0.065, wheel_at=0.12, wall=0.42,
                   depth=0.14, r=0.06, angle=0.0, zorder=4):
    """A pin on a rolling slider: the triangle's tip is the pin at ``p``, its base
    stands on two wheels that roll along a wall on the left.

    The support of a column head that moves along its axis but not across it.
    """
    x0, y0 = p
    xb = x0 - base
    sk.body(ax, _turn([(x0, y0), (xb, y0 + half), (xb, y0 - half)], p, angle), zorder=zorder)
    for yw in (y0 + wheel_at, y0 - wheel_at):
        c = _turn([(xb - wheel, yw)], p, angle)[0]
        ax.add_patch(Circle(c, wheel, facecolor='white', edgecolor=sk.EDGE, lw=sk.LW,
                            zorder=zorder))
    xw = xb - 2*wheel
    ground(ax, _turn([(xw, y0 + wall), (xw, y0 - wall)], p, angle), depth=depth)
    sk.pin(ax, p, r, zorder=5)


def pedestal(ax, c, w, h, ground_w=None, depth=None, angle=0.0, zorder=1):
    """A pin pedestal standing on the floor under ``c``: a trapezoid ``h`` high whose
    foot is ``2w`` wide, on a strip of ground.

    ``angle`` turns it about ``c``: 180 hangs it from a ceiling as the lug that
    carries the upper pin of a hanger rod, 90 or -90 fixes it to a wall.
    """
    y0 = c[1] - h
    gw = 2.6*w if ground_w is None else ground_w
    ground(ax, _turn([(c[0] - gw/2, y0), (c[0] + gw/2, y0)], c, angle),
           depth=0.35*h if depth is None else depth)
    sk.body(ax, _turn([(c[0] - w, y0), (c[0] + w, y0), (c[0] + 0.45*w, c[1]),
                       (c[0] - 0.45*w, c[1])], c, angle), zorder=zorder)


def wall(ax, p, half_height=0.9, solid='left', depth=0.28):
    """A vertical wall through ``p`` with its solid on the side ``solid``: a clamp."""
    x, y = p
    top, bottom = (x, y + half_height), (x, y - half_height)
    ground(ax, [top, bottom] if solid == 'left' else [bottom, top], depth=depth)


def hatched_wall(ax, x, side, h, width=0.26, lw=1.0, zorder=2):
    """A rigid wall at ``x`` drawn the classic way, hatched, from -h to h; its solid
    lies on the side ``side`` (+1 right, -1 left) of the face."""
    ax.add_patch(Rectangle((x, -h), side*width, 2*h, facecolor='none', edgecolor='k',
                           hatch='////', lw=lw, zorder=zorder))
    ax.plot([x, x], [-h, h], 'k-', lw=2.0, zorder=zorder + 2)


def sliding_block(ax, x0, x1, y, h=0.34, overhang=0.35, depth=0.14, zorder=4):
    """A block from x0 to x1, centred at height y, resting on a strip of ground:
    a friction slider."""
    sk.body(ax, [(x0, y - h/2), (x1, y - h/2), (x1, y + h/2), (x0, y + h/2)], zorder=zorder)
    ground(ax, [(x0 - overhang, y - h/2), (x1 + overhang, y - h/2)], depth=depth)


def bracket(ax, base_x, tip, base_half=0.09, tip_half=0.045, zorder=2):
    """A tapered arm from a wall at ``base_x`` out to ``tip``."""
    x, y = tip
    sk.body(ax, [(base_x, y + base_half), (base_x, y - base_half), (x, y - tip_half),
                 (x, y + tip_half)], zorder=zorder)


# --- beams, rods and line loads ------------------------------------------------------

CLEVIS_STOPS = ['#7d95b8', '#b7c9e3', '#dce6f3', '#a5bbdb', '#7890b2']   # a forged fork


def i_beam(ax, p0, p1, h, tf=None, stops=STEEL_STOPS, edgecolor=sk.EDGE, lw=sk.LW,
           zorder=2):
    """A rolled I-beam (wide-flange section) seen from the side, from ``p0`` to ``p1``.

    ``p0`` and ``p1`` are the ends of its centre line and ``h`` the depth of the
    section. The flanges are the bands ``tf`` thick (default ``0.12*h``) along both
    faces, the web between them is shaded across the depth with ``stops``, so the
    part reads as a steel beam and not as a flat plate. Returns the outline patch.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = sk.normal(p1 - p0)
    tf = 0.12*h if tf is None else tf
    a = h/2
    outline = [p0 - a*n, p1 - a*n, p1 + a*n, p0 + a*n]
    patch = shade(ax, outline, stops, -n, edgecolor=edgecolor, lw=lw, zorder=zorder)
    stop_list = _stop_list(stops)
    light = stop_list[len(stop_list)//2][1]
    for s in (1, -1):                              # the flanges, light bands on the faces
        band = [p0 + s*a*n, p1 + s*a*n, p1 + s*(a - tf)*n, p0 + s*(a - tf)*n]
        sk.body(ax, band, facecolor=light, edgecolor=edgecolor, zorder=zorder + 0.01)
    return patch


def clevis(ax, pin, direction, w, L=None, stops=CLEVIS_STOPS, pin_r=None,
           edgecolor=sk.EDGE, lw=sk.LW, zorder=4):
    """A clevis (rod-end fork) seen from the side, around the pin at ``pin``.

    The fork is ``w`` wide, rounded about the pin, and runs ``L`` (default
    ``2.1*w``) along ``direction`` to where the rod is screwed in; a line across at
    three quarters of that marks the solid head of the fork. The pin is a white
    disc of radius ``pin_r`` (default ``0.3*w``) with its centre dotted. Returns
    the centre of the fork's far end, where the rod starts.
    """
    c = np.asarray(pin, float)
    u = sk.unit(np.asarray(direction, float))
    v = sk.normal(u)
    L = 2.1*w if L is None else L
    r = w/2
    t = np.linspace(np.pi/2, 3*np.pi/2, 24)
    pts = [c + L*u + r*v] + [c + r*(np.cos(s)*u + np.sin(s)*v) for s in t] + [c + L*u - r*v]
    shade(ax, pts, stops, v, edgecolor=edgecolor, lw=lw, zorder=zorder)
    head = c + 0.75*L*u
    ax.plot(*zip(head - r*v, head + r*v), color=edgecolor, lw=0.8*lw, zorder=zorder + 0.01)
    sk.pin(ax, c, 0.3*w if pin_r is None else pin_r, zorder=zorder + 0.02)
    dot(ax, c, 2.5, zorder=zorder + 0.03)
    return c + L*u


def clevis_rod(ax, p0, p1, d, w=None, L=None, ends=(True, True), stops=CHROME_STOPS,
               fork_stops=CLEVIS_STOPS, zorder=4):
    """A tie rod or hanger between the pins ``p0`` and ``p1``, a clevis at each end.

    A two-force member as it is built: a round rod ``d`` thick, shaded across like a
    polished cylinder, screwed into a fork ``w`` wide (default ``2.4*d``) and ``L``
    long around each pin (see :func:`clevis`). ``ends`` says which ends carry a fork;
    an end without one is a flat cut through the rod at that point, the stub of a
    rod cut free in a free body diagram. Returns the two points where the rod
    leaves the forks (or the cut points).
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    u = sk.unit(p1 - p0)
    w = 2.4*d if w is None else w
    q0 = clevis(ax, p0, u, w, L, fork_stops, zorder=zorder + 0.1) if ends[0] else p0
    q1 = clevis(ax, p1, -u, w, L, fork_stops, zorder=zorder + 0.1) if ends[1] else p1
    a = q0 - 0.1*w*u if ends[0] else q0            # tuck the rod into the forks
    b = q1 + 0.1*w*u if ends[1] else q1
    shaded_bar(ax, a, b, d, stops, round_ends=False, zorder=zorder)
    return q0, q1


def line_load(ax, p0, p1, height, n=None, text=None, q=(1.0, 1.0), inset=0.0,
              color=sk.LOAD, lw=1.0, ms=8, fontsize=13, text_offset=None, zorder=6):
    """A distributed load (a line load) on the edge from ``p0`` to ``p1``.

    The arrows come in from the left of the walk p0 to p1 and end on the edge, so a
    load on the top of a beam is walked left to right. ``height`` is the arrow length
    at full intensity; ``q`` gives the intensity at the two ends as fractions of it,
    so ``(0, 1)`` draws a triangular load. ``n`` arrows (default about one per
    ``0.8*height``) are spread between ``inset`` from each end (a pair for the two
    ends: clear of a rod or a wall at one of them), and a line joins their tails
    over the whole length. ``text`` (the intensity, as ``3 kN/m`` or
    ``q_0``) goes ``text_offset`` (default ``0.3*height``) beyond the middle of that
    line, in the load's colour. Returns the points of the tail line.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    e = p1 - p0
    Ln = np.linalg.norm(e)
    nrm = sk.normal(e)
    i0, i1 = np.broadcast_to(np.asarray(inset, float), 2)
    n = n or max(int(round((Ln - i0 - i1)/(0.8*height))) + 1, 2)
    tail = lambda s: p0 + s*e + height*np.interp(s, (0, 1), q)*nrm
    for s in np.linspace(i0/Ln, 1 - i1/Ln, n):
        if np.interp(s, (0, 1), q) > 1e-9:
            ax.add_patch(FancyArrowPatch(tail(s), p0 + s*e, arrowstyle='-|>',
                                         mutation_scale=ms, color=color, lw=lw, shrinkA=0,
                                         shrinkB=0, zorder=zorder))
    ends = np.array([tail(0.0), tail(1.0)])
    ax.plot(ends[:, 0], ends[:, 1], color=color, lw=lw, solid_capstyle='butt', zorder=zorder)
    if text:
        off = 0.3*height if text_offset is None else text_offset
        sk.label(ax, tail(0.5) + off*nrm, text, color=color, fontsize=fontsize,
                 zorder=zorder + 1)
    return ends


def tone(base, normal, light=(-0.3, -0.6, 0.75)):
    """The colour ``base`` (an RGB triple) lit by ``light`` on a face with outward ``normal``."""
    k = 0.72 + 0.38*max(float(np.dot(normal, light)), 0.0)
    return tuple(np.clip(np.asarray(base, float)*k, 0, 1))


class Proj:
    """An axonometric projection: the page images of the three unit vectors."""

    def __init__(self, ex, ey, ez=(0, 1)):
        self.e = np.array([ex, ey, ez], float)

    def __call__(self, p):
        return np.asarray(p, float) @ self.e

    def axes(self, ax, origin, length, labels=('$x$', '$y$', '$z$'), lw=1.2,
             offsets=None, fontsize=12):
        """The coordinate triad at ``origin``; ``offsets`` moves a label, by axis index."""
        o = self(origin)
        offsets = offsets or {}
        for k, lab in enumerate(labels):
            d = np.zeros(3)
            d[k] = 1
            tip = self(np.asarray(origin) + length*d)
            ax.add_patch(FancyArrowPatch(o, tip, arrowstyle='-|>', mutation_scale=12,
                                         color=sk.EDGE, lw=lw, shrinkA=0, shrinkB=0,
                                         zorder=5))
            sk.label(ax, tip + 0.08*length*sk.unit(tip - o) + np.asarray(offsets.get(k, (0, 0))),
                     lab, fontsize=fontsize)

    @classmethod
    def from_view(cls, d, up=(0, 0, 1.0)):
        """A parallel view looking back along ``d`` (toward the viewer), ``up`` upright."""
        d = np.asarray(d, float)
        d = d/np.linalg.norm(d)
        right = np.cross(-d, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, -d)
        return cls((right[0], up[0]), (right[1], up[1]), (right[2], up[2]))

    def arc(self, ax, centre, u, v, r, a0, a1, color, arrow=True, lw=1.3, ms=10, zorder=6):
        """An arc of radius ``r`` about the 3D ``centre`` in the plane of the unit vectors
        ``u`` and ``v``, from ``a0`` to ``a1`` radians, with a head at ``a1``.

        Returns the page point at the middle of the arc, for its label.
        """
        centre, u, v = (np.asarray(x, float) for x in (centre, u, v))
        t = np.linspace(a0, a1, 40)
        pts = np.array([self(centre + r*(np.cos(s)*u + np.sin(s)*v)) for s in t])
        ax.plot(pts[:-1, 0], pts[:-1, 1], color=color, lw=lw, zorder=zorder)
        if arrow:
            ax.add_patch(FancyArrowPatch(pts[-4], pts[-1], arrowstyle='-|>', mutation_scale=ms,
                                         color=color, lw=lw, shrinkA=0, shrinkB=0,
                                         zorder=zorder))
        else:
            ax.plot(pts[-2:, 0], pts[-2:, 1], color=color, lw=lw, zorder=zorder)
        m = (a0 + a1)/2
        return self(centre + r*(np.cos(m)*u + np.sin(m)*v))

    def force(self, ax, p, d, length, text=None, offset=(0, 0), color=sk.LOAD, head=False,
              fontsize=13, text_color='black'):
        """A force of 3D ``length`` along the 3D direction ``d`` at the 3D point ``p``.

        Pass ``text_color=color`` for a label in the colour of its arrow, the book's rule.
        """
        p, d = np.asarray(p, float), np.asarray(d, float)
        d = d/np.linalg.norm(d)
        tail, tip = (p - length*d, p) if head else (p, p + length*d)
        vec(ax, self(tail), self(tip), color, text, offset, fontsize, text_color=text_color)


def vec(ax, p0, p1, color, text=None, offset=(0, 0), fontsize=13, lw=2.0, zorder=6,
        text_color='black'):
    """An arrow from ``p0`` to ``p1`` with its label beside the head, in ``text_color``."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=15,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=zorder))
    if text is not None:
        sk.label(ax, p1 + np.asarray(offset), text, color=text_color, fontsize=fontsize,
                 zorder=zorder + 1)


# the white outline every label in the book carries, for text drawn without sk.label
HALO = [patheffects.withStroke(linewidth=3, foreground='white')]


def dot(ax, p, ms=5, zorder=6):
    ax.plot(*p, 'o', ms=ms, color='black', zorder=zorder)


def circled(ax, xy, text, fontsize=11, zorder=8):
    """A number in a circle, the way the free body diagrams tag their bodies."""
    return ax.text(*xy, text, ha='center', va='center', fontsize=fontsize, zorder=zorder,
                   color=sk.GREEN,
                   bbox=dict(boxstyle='circle,pad=0.25', facecolor='white',
                             edgecolor=sk.GREEN, lw=1.0))


SVG_NS = 'http://www.w3.org/2000/svg'
XLINK_NS = 'http://www.w3.org/1999/xlink'


def data_to_svg(fig, ax, xy):
    """Where a data point lands in the saved SVG, in its user units (points)."""
    old = fig.dpi
    fig.dpi = 72
    try:
        px = ax.transData.transform(xy)
        height = fig.get_size_inches()[1]*72
    finally:
        fig.dpi = old
    return np.array([px[0], height - px[1]])


def embed_svg(out_path, src_path, corner, size, mirror=False):
    """Paste ``src_path`` into the saved SVG ``out_path``.

    ``corner`` is the SVG position (points) of the drawing's top-left corner
    and ``size`` its width and height there. With ``mirror`` the drawing is
    flipped left to right. The figure must have been saved without a tight
    bounding box, so that ``data_to_svg`` gives the right positions.
    """
    ET.register_namespace('', SVG_NS)
    ET.register_namespace('xlink', XLINK_NS)
    out = ET.parse(out_path)
    src = ET.parse(src_path).getroot()
    vb = [float(v) for v in src.get('viewBox').split()]
    sx, sy = size[0]/vb[2], size[1]/vb[3]
    tx = corner[0] - vb[0]*sx
    ty = corner[1] - vb[1]*sy
    if mirror:
        transform = f'translate({tx + size[0]:.3f},{ty:.3f}) scale({-sx:.5f},{sy:.5f})'
    else:
        transform = f'translate({tx:.3f},{ty:.3f}) scale({sx:.5f},{sy:.5f})'
    g = ET.SubElement(out.getroot(), f'{{{SVG_NS}}}g', {'transform': transform})
    for child in list(src):
        if child.tag.endswith('metadata'):
            continue
        g.append(copy.deepcopy(child))
    out.write(out_path, xml_declaration=True, encoding='utf-8')


# --- 3D parts for axonometric figures -------------------------------------------
#
# Each takes a Proj and draws a solid by painter's order: callers draw the far
# parts first. Horizontal circles become ellipses; their front half is the
# lower chain on the page, since every Proj looks down on the xy-plane.

def view_dir(P):
    """The 3D direction toward the viewer: the one that projects to a point."""
    d = np.cross(P.e[:, 0], P.e[:, 1])
    return d/np.linalg.norm(d)


def ring(P, centre, r, n=180, axis=2):
    """Page points of a circle of radius ``r`` about ``centre``, normal to ``axis``."""
    c = np.asarray(centre, float)
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    u, v = [k for k in range(3) if k != axis]
    pts = np.zeros((n, 3)) + c
    pts[:, u] += r*np.cos(t)
    pts[:, v] += r*np.sin(t)
    return np.array([P(p) for p in pts])


def _chains(pts):
    """Split a closed page curve at its leftmost and rightmost points: (lower, upper)."""
    i0, i1 = np.argmin(pts[:, 0]), np.argmax(pts[:, 0])
    a = np.roll(pts, -i0, axis=0)
    k = (i1 - i0) % len(pts)
    c1, c2 = a[:k + 1], np.vstack([a[k:], a[:1]])[::-1]
    return (c1, c2) if c1[:, 1].mean() < c2[:, 1].mean() else (c2, c1)


def vcylinder(ax, P, centre, r, z0, z1, stops=STEEL_STOPS, top='#e4eff6', edgecolor=sk.EDGE,
              lw=sk.LW, zorder=2, texture=None):
    """A vertical cylinder standing on ``centre`` (x, y) from z0 to z1.

    The side is shaded left to right, the top face is flat ``top`` (or None to
    leave it open). ``texture`` is a callable ``f(ax, polygon, zorder)`` laid over
    the side, such as ``speckle``. Returns the side and top polygons.
    """
    cx, cy = centre
    bot, tp = ring(P, (cx, cy, z0), r), ring(P, (cx, cy, z1), r)
    lower, _ = _chains(bot)
    _, upper = _chains(tp)
    side = np.vstack([lower, upper[::-1]])
    shade(ax, side, stops, (1, 0), n=48, edgecolor=edgecolor, lw=lw, zorder=zorder)
    if texture is not None:
        texture(ax, side, zorder + 0.02)
    if top is not None:
        ax.add_patch(Polygon(tp, closed=True, facecolor=top, edgecolor=edgecolor, lw=lw,
                             zorder=zorder + 0.05))
    return side, tp


def rod3d(ax, P, p0, p1, r, stops=STEEL_STOPS, edgecolor=sk.EDGE, lw=sk.LW, zorder=2):
    """A round rod from ``p0`` to ``p1`` in 3D, its silhouette shaded across its width."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    a, b = P(p0), P(p1)
    d = sk.unit(b - a)
    nrm = sk.normal(d)
    # the page half-width of the rod is the extent of its cross-section across the rod
    axis = (p1 - p0)/np.linalg.norm(p1 - p0)
    helper = np.array([0, 0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(axis, helper); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    t = np.linspace(0, 2*np.pi, 90)
    half = max(abs((P(r*(np.cos(s)*u + np.sin(s)*v)) - P(np.zeros(3))) @ nrm) for s in t)
    pts = [a + half*nrm, b + half*nrm, b - half*nrm, a - half*nrm]
    return shade(ax, pts, stops, nrm, n=30, edgecolor=edgecolor, lw=lw, zorder=zorder)


def sphere(ax, centre, r, stops=BRASS_STOPS, light=(-0.4, 0.45), n=40, edgecolor=sk.EDGE,
           lw=sk.LW, zorder=3):
    """A shaded ball, lit from ``light`` (a fraction of the radius from the centre).

    ``stops`` run from the shadowed rim to the highlight. In SVG output this is one
    circle with a radial gradient; otherwise stacked discs shrinking toward the light.
    """
    c = np.asarray(centre, float)
    h = c + r*np.asarray(light, float)
    if SVG_GRADIENTS:
        patch = Circle(c, r, facecolor=_stop_list(stops)[-1][1], edgecolor=edgecolor, lw=lw,
                       zorder=zorder)
        ax.add_patch(patch)
        # offset 0 at the highlight, 1 at the rim
        st = [(1 - o, col, a) for o, col, a in reversed(_stop_list(stops))]
        _GRADIENTS.append((patch, {'kind': 'radial', 'centre': c, 'r': r, 'focus': h,
                                   'stops': st}))
        return patch
    cols = _stops_to_colors(stops, n)
    for k, col in enumerate(cols):
        f = 1 - k/n                                        # 1 at the rim, toward 0 at the light
        ax.add_patch(Circle(h + f*(c - h), f*r, facecolor=col, edgecolor='none',
                            zorder=zorder + 0.001*k))
    ax.add_patch(Circle(c, r, facecolor='none', edgecolor=edgecolor, lw=lw,
                        zorder=zorder + 0.1))


def box3d(ax, P, lo, hi, colors=('#9fbdd0', '#c3d8e5', '#e4eff6'), edgecolor=sk.EDGE,
          lw=sk.LW, zorder=1, texture=None):
    """An axis-aligned box from corner ``lo`` to ``hi``: the +x, +y and +z faces.

    ``colors`` are the fills of those three faces. Returns the face polygons.
    """
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    faces = [[(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],     # +x
             [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],     # +y
             [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]]     # +z
    polys = []
    for f, col in zip(faces, colors):
        pts = np.array([P(p) for p in f])
        ax.add_patch(Polygon(pts, closed=True, facecolor=col, edgecolor=edgecolor, lw=lw,
                             zorder=zorder))
        if texture is not None:
            texture(ax, pts, zorder + 0.02)
        polys.append(pts)
    return polys


def speckle(color='white', density=260, length=0.05, lw=0.6, alpha=0.8, seed=3):
    """A texture of short random squiggles, like the cast surface of a stone base.

    Returns ``f(ax, polygon, zorder)`` for ``vcylinder`` and ``box3d``; ``density``
    is squiggles per unit page area and ``length`` their page size.
    """
    rng = np.random.default_rng(seed)

    def f(ax, poly, zorder):
        poly = np.asarray(poly, float)
        clip = Polygon(poly, closed=True, facecolor='none', edgecolor='none')
        ax.add_patch(clip)
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        area = 0.5*abs(np.dot(poly[:, 0], np.roll(poly[:, 1], 1))
                       - np.dot(poly[:, 1], np.roll(poly[:, 0], 1)))
        for _ in range(int(density*area)):
            p = lo + rng.random(2)*(hi - lo)
            a = rng.random()*2*np.pi
            s = np.linspace(-1, 1, 6)
            wig = 0.3*np.sin(3*s + rng.random()*6)
            pts = p + length*np.c_[s*np.cos(a) - wig*np.sin(a), s*np.sin(a) + wig*np.cos(a)]
            line, = ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, alpha=alpha,
                            solid_capstyle='round', zorder=zorder)
            line.set_clip_path(clip)
    return f


def thread3d(ax, P, centre, r_root, r_crest, z0, z1, pitch, stops=STEEL_STOPS,
             ridge=('#3f5f73', '#9fbdd0', '#eef5f9'), zorder=2):
    """A vertical threaded rod: a shaded core with one rounded crest per pitch.

    Each crest is the front half of one turn of the helix at radius ``r_crest``,
    stroked three times, dark and wide, mid and narrower, light and thin and a
    little higher, so that it reads as a rounded ridge lit from above. The
    strokes are sized from the axes limits, so set those first.
    """
    cx, cy = centre
    vcylinder(ax, P, centre, r_root, z0, z1, stops=stops, top=None, zorder=zorder)
    t = np.linspace(0, 2*np.pi, 120)
    d = view_dir(P)
    unit = np.linalg.norm(P((0, 0, 1)) - P((0, 0, 0)))/_data_per_point(ax)   # points per z unit
    for z in np.arange(z0 + 0.6*pitch, z1 - 0.4*pitch, pitch):
        # one turn of the helix, centred on the front so that the visible arc is contiguous
        phi = t + np.arctan2(d[1], d[0]) - np.pi
        pts3 = np.c_[cx + r_crest*np.cos(phi), cy + r_crest*np.sin(phi),
                     z + pitch*(t/(2*np.pi) - 0.5)]
        front = (np.cos(phi)*d[0] + np.sin(phi)*d[1]) > -0.05
        seg = np.array([P(p) for p in pts3[front]])
        for col, w, lift in zip(ridge, (0.95, 0.72, 0.28), (0.0, 0.04, 0.16)):
            ax.plot(seg[:, 0], seg[:, 1] + lift*pitch*unit*_data_per_point(ax), color=col,
                    lw=w*pitch*unit, solid_capstyle='round', zorder=zorder + 0.1)


def _data_per_point(ax):
    """How many points one data unit spans on the page, along x."""
    fig = ax.figure
    x0, x1 = ax.get_xlim()
    return (x1 - x0)/(ax.get_position().width*fig.get_size_inches()[0]*72)


def _thicken(el, factor):
    """Multiply every stroke width under ``el`` by ``factor``, in attributes and styles."""
    import re
    for node in el.iter():
        if node.get('stroke-width'):
            node.set('stroke-width', f"{float(node.get('stroke-width'))*factor:g}")
        style = node.get('style')
        if style and 'stroke-width' in style:
            node.set('style', re.sub(r'stroke-width:\s*([0-9.]+)',
                                     lambda m: f'stroke-width:{float(m.group(1))*factor:g}',
                                     style))


def embed_svg_at(out_path, src_path, centre, width, angle=0.0, opacity=1.0, stroke=1.0):
    """Paste ``src_path`` into the saved SVG ``out_path``, centred and turned.

    ``centre`` is the SVG position (points) of the drawing's centre, ``width`` its
    width there before turning, and ``angle`` an anticlockwise turn in degrees as
    seen on the page. ``opacity`` below one gives a faded copy, and ``stroke``
    thickens the drawing's lines so that a detailed drawing survives being shrunk.
    The figure must
    have been saved without a tight bounding box, as for ``embed_svg``.
    """
    ET.register_namespace('', SVG_NS)
    ET.register_namespace('xlink', XLINK_NS)
    out = ET.parse(out_path)
    src = ET.parse(src_path).getroot()
    vb = [float(v) for v in src.get('viewBox').split()]
    s = width/vb[2]
    cx, cy = vb[0] + vb[2]/2, vb[1] + vb[3]/2
    # SVG turns clockwise for a positive angle, since its y axis points down
    transform = (f'translate({centre[0]:.3f},{centre[1]:.3f}) rotate({-angle:.3f}) '
                 f'scale({s:.6f}) translate({-cx:.3f},{-cy:.3f})')
    g = ET.SubElement(out.getroot(), f'{{{SVG_NS}}}g',
                      {'transform': transform, 'opacity': f'{opacity:.3f}'})
    for child in list(src):
        if child.tag.endswith('metadata'):
            continue
        g.append(copy.deepcopy(child))
    if stroke != 1.0:
        _thicken(g, stroke)
    out.write(out_path, xml_declaration=True, encoding='utf-8')


# --- saving for the hand-edit round trip ------------------------------------------
#
# A figure saved with save_svg can be opened in Inkscape, adjusted by hand and saved
# as <name>.edited.svg; mechanicskit.svg_roundtrip then lists what changed, in the
# figure's own data units, so that the script can be updated to draw the edit.

import json as _json
import re as _re

RT_META_ID = 'roundtrip'


def _slug(text):
    """A readable, XML-safe id fragment for a label: '$\\boldsymbol{r}_{OA}$' -> 'r_OA'."""
    t = _re.sub(r'\\(boldsymbol|mathbf|mathrm|text|bm)\b', '', text)
    t = t.replace('\\', '').replace('$', '')
    t = _re.sub(r'[^0-9A-Za-z]+', '_', t).strip('_')
    return t or 'text'


def _artists(ax):
    """The drawable artists of ``ax`` in draw order, without the axes' own decoration."""
    arts = list(ax.patches) + list(ax.lines) + list(ax.collections) + list(ax.texts) \
        + list(ax.artists) + list(ax.images)
    return sorted((a for a in arts if a.get_visible()), key=lambda a: a.get_zorder())


def _tag(fig):
    """Give every artist a stable id; return {id: description} for the metadata.

    Texts are named by what they say, everything else by its kind and draw order, so
    the same script gives the same ids on every run. The description holds the kind,
    the text and the centre in data units, which leads back to the call that drew it.
    """
    table, used = {}, set()
    renderer = fig.canvas.get_renderer()
    for k, ax in enumerate(fig.axes):
        pre = f'a{k}-' if len(fig.axes) > 1 else ''
        count = {}
        inv = ax.transData.inverted()
        for art in _artists(ax):
            kind = type(art).__name__.lower().replace('patch', '') or 'patch'
            text = art.get_text() if hasattr(art, 'get_text') else None
            if art.get_gid():
                gid = art.get_gid()
            elif text is not None:
                base = pre + 'label-' + _slug(text)
                gid, n = base, 2
                while gid in used:
                    gid, n = f'{base}-{n}', n + 1
            else:
                count[kind] = count.get(kind, 0) + 1
                gid = f'{pre}{kind}-{count[kind]:02d}'
            used.add(gid)
            art.set_gid(gid)
            try:
                bb = art.get_window_extent(renderer)
                centre = inv.transform(((bb.x0 + bb.x1)/2, (bb.y0 + bb.y1)/2)).tolist()
            except Exception:
                centre = None
            table[gid] = {'kind': kind, 'axes': k, 'text': text,
                          'centre': [round(c, 6) for c in centre] if centre else None}
    return table


def _write_gradients(fig, tree, axes):
    """Give each shaded part of ``fig`` its gradient: one <defs> entry and a url fill."""
    root = tree.getroot()
    defs = root.find(f'{{{SVG_NS}}}defs')
    if defs is None:
        defs = _ET_mod().Element(f'{{{SVG_NS}}}defs')
        root.insert(0, defs)
    groups = {g.get('id'): g for g in root.iter() if g.get('id')}
    mine = [(p, spec) for p, spec in _GRADIENTS if p.figure is fig]
    for patch, spec in mine:
        g = groups.get(patch.get_gid())
        if g is None:
            continue
        m = axes[fig.axes.index(patch.axes)]
        to = lambda q: (m['scale'][0]*q[0] + m['offset'][0], m['scale'][1]*q[1] + m['offset'][1])
        gid = 'grad-' + patch.get_gid()
        if spec['kind'] == 'linear':
            (x1, y1), (x2, y2) = to(spec['p0']), to(spec['p1'])
            el = _ET_mod().SubElement(defs, f'{{{SVG_NS}}}linearGradient', {
                'id': gid, 'gradientUnits': 'userSpaceOnUse',
                'x1': f'{x1:.3f}', 'y1': f'{y1:.3f}', 'x2': f'{x2:.3f}', 'y2': f'{y2:.3f}'})
        else:
            (cx, cy), (fx, fy) = to(spec['centre']), to(spec['focus'])
            el = _ET_mod().SubElement(defs, f'{{{SVG_NS}}}radialGradient', {
                'id': gid, 'gradientUnits': 'userSpaceOnUse', 'cx': f'{cx:.3f}',
                'cy': f'{cy:.3f}', 'r': f'{abs(m["scale"][0])*spec["r"]:.3f}',
                'fx': f'{fx:.3f}', 'fy': f'{fy:.3f}'})
        for off, col, a in spec['stops']:
            _ET_mod().SubElement(el, f'{{{SVG_NS}}}stop', {
                'offset': f'{off:.4g}', 'style': f'stop-color:{col};stop-opacity:{a:.3g}'})
        for shape in g.iter():
            st = shape.get('style')
            if st and 'fill:' in st.replace(' ', ''):
                shape.set('style', _re.sub(r'fill:\s*[^;]+', f'fill: url(#{gid})', st, count=1))
    _GRADIENTS[:] = [(p, s) for p, s in _GRADIENTS if p.figure is not fig]


def _ET_mod():
    import xml.etree.ElementTree as _ET
    return _ET


def save_svg(fig, path, tight=True, pad_inches=0.04):
    """Save ``fig`` as SVG ready for hand editing: stable ids and a data-to-SVG map.

    Each artist becomes an SVG group whose id names it (Inkscape shows these in its
    Objects panel). A <metadata id="roundtrip"> element records, for every axes, the
    affine map from data to SVG user units, svg = scale*data + offset, so that
    mechanicskit.svg_roundtrip can turn moves made in an editor back into data units.
    """
    fig.canvas.draw()
    table = _tag(fig)
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi
    if tight:
        bb = fig.get_tightbbox(renderer).padded(pad_inches)       # inches
    else:
        w, h = fig.get_size_inches()
        from matplotlib.transforms import Bbox
        bb = Bbox.from_bounds(0, 0, w, h)
    axes = []
    for ax in fig.axes:
        def to_svg(p, ax=ax):
            x, y = ax.transData.transform(p)/dpi                   # inches
            return (x - bb.x0)*72, (bb.y1 - y)*72
        (x0, y0), (x1, _), (_, y1) = to_svg((0, 0)), to_svg((1, 0)), to_svg((0, 1))
        axes.append({'scale': [x1 - x0, y1 - y0], 'offset': [x0, y0],
                     'xlim': list(ax.get_xlim()), 'ylim': list(ax.get_ylim())})
    kw = dict(bbox_inches='tight', pad_inches=pad_inches) if tight else {}
    fig.savefig(path, facecolor='white', **kw)

    import xml.etree.ElementTree as _ET
    _ET.register_namespace('', SVG_NS)
    _ET.register_namespace('xlink', XLINK_NS)
    tree = _ET.parse(path)
    _write_gradients(fig, tree, axes)
    meta = _ET.SubElement(tree.getroot(), f'{{{SVG_NS}}}metadata', {'id': RT_META_ID})
    meta.text = _json.dumps({'version': 1, 'axes': axes, 'items': table},
                            separators=(',', ':'))
    tree.write(path, xml_declaration=True, encoding='utf-8')
    print('wrote', path)


def bevel_section(ax, apex, axis, delta_deg, A0, b, m, ha=1.0, hf=1.25, depth=1.5,
                  bore=None, sides=(1, -1), facecolor=sk.STEEL, edgecolor=sk.EDGE,
                  lw=sk.LW, zorder=2):
    """The axial section of a straight bevel gear: the plane through its axis.

    The pitch cone has its apex at ``apex`` (2D) and its axis along ``axis``, the direction
    from the apex into the gear. Its generators leave the apex at ``delta_deg`` to the axis;
    along each the teeth run from the cone distance ``A0 - b`` (the front, or toe) to ``A0``
    (the back, or heel), with the addendum ``ha*m`` and dedendum ``hf*m`` at the back and
    tapering towards the apex, so tip and root lines also run to the apex. The tooth zone is
    bounded by the back cone and the front cone, both normal to the pitch generator. Under
    the root the body goes ``depth*m`` further along the back cone, then square to the axis
    down to the bore radius ``bore/2`` (to the axis if ``bore`` is None) and back along it to
    the front. ``sides`` draws the half above (1) and below (-1) the axis, as seen walking
    from the apex along ``axis``: +1 is on the left.

    Returns, per side, a dict of 2D points: 'pitch_back', 'pitch_front', 'tip_back',
    'tip_front', 'root_back', 'root_front', and the unit vectors 'g' (pitch generator from
    the apex) and 'n' (normal to the pitch cone, away from the axis).
    """
    O = np.asarray(apex, float)
    a = np.asarray(axis, float)
    a = a/np.linalg.norm(a)
    d = np.radians(delta_deg)
    k = (A0 - b)/A0
    r_bore = 0.0 if bore is None else bore/2
    out = {}
    for s in sides:
        e_r = s*np.array([-a[1], a[0]])                          # radial, on this side
        g = np.cos(d)*a + np.sin(d)*e_r
        n = -np.sin(d)*a + np.cos(d)*e_r
        P0, P1 = O + A0*g, O + (A0 - b)*g
        pts = {'pitch_back': P0, 'pitch_front': P1,
               'tip_back': P0 + ha*m*n, 'tip_front': P1 + k*ha*m*n,
               'root_back': P0 - hf*m*n, 'root_front': P1 - k*hf*m*n, 'g': g, 'n': n}
        tooth = [pts['root_front'], pts['root_back'], pts['tip_back'], pts['tip_front']]
        ax.add_patch(Polygon(tooth, closed=True, facecolor=facecolor, edgecolor=edgecolor,
                             lw=lw, zorder=zorder))
        low_b = pts['root_back'] - depth*m*n
        low_f = pts['root_front'] - k*depth*m*n
        def drop(p):                                              # square to the axis, to the bore
            return O + ((p - O) @ a)*a + r_bore*e_r
        body = [pts['root_front'], pts['root_back'], low_b, drop(low_b), drop(low_f), low_f]
        ax.add_patch(Polygon(body, closed=True, facecolor=facecolor, edgecolor=edgecolor,
                             lw=lw, zorder=zorder - 0.05))
        out[s] = pts
    return out


def circle_tangents(centre, r, point):
    """The two points where lines from ``point`` touch the circle (``centre``, ``r``).

    A band or a belt leaving a drum towards an anchor or a lever runs along such a
    tangent. Returns ``(left, right)``: the touching points seen from ``point``
    looking at the centre, the first on the left of that line, the second on its
    right. ``point`` must lie outside the circle.
    """
    c, p = np.asarray(centre, float), np.asarray(point, float)
    d = p - c
    dist = np.linalg.norm(d)
    if dist <= r:
        raise ValueError('point lies inside the circle')
    base = np.arctan2(d[1], d[0])
    off = np.arccos(r/dist)
    q1 = c + r*np.array([np.cos(base + off), np.sin(base + off)])
    q2 = c + r*np.array([np.cos(base - off), np.sin(base - off)])
    # seen from p looking at c, which of the two lies on the left
    look = c - p
    left_first = look[0]*(q1 - p)[1] - look[1]*(q1 - p)[0] > 0
    return (q1, q2) if left_first else (q2, q1)


def out_of_plane(ax, p, r, into=True, color=sk.LOAD, lw=1.6, zorder=6):
    """A force or velocity normal to the page at ``p``: a circle of radius ``r`` (data
    units) with a cross (``into=True``, the tail of an arrow going into the page) or a dot
    (``into=False``, the head of an arrow coming out). Label it with ``sk.label`` in the
    same colour."""
    c = np.asarray(p, float)
    ax.add_patch(Circle(c, r, facecolor='white', edgecolor=color, lw=lw, zorder=zorder))
    if into:
        k = r/np.sqrt(2)
        for s in (1, -1):
            ax.plot([c[0] - k, c[0] + k], [c[1] - s*k, c[1] + s*k], color=color, lw=lw,
                    solid_capstyle='round', zorder=zorder + 0.1)
    else:
        ax.add_patch(Circle(c, 0.28*r, facecolor=color, edgecolor='none', zorder=zorder + 0.1))


# --- bearings in section ------------------------------------------------------------
#
# A rolling bearing is drawn the way catalogues and assembly drawings show it: cut
# along the shaft axis, the rings solid, the rolling elements shaded. Every function
# takes the true radii, so that a bearing, its shaft and its housing drawn with the
# same axis line up, and draws both halves unless told otherwise.

BALL_STOPS = ['#1f4f6e', '#5d93b8', '#d4e7f4']                     # rim to highlight
ROLLER_STOPS = ['#33627f', '#9fc3da', '#e6f1f8', '#86aec8', '#33627f']
BRONZE = '#d9b36c'                                                 # plain bearings
SECTION = '#e4e7ea'                                                # a cut housing


def _line_x(p, d, q, e):
    """Intersection of the line p + s d with the line q + t e."""
    A = np.array([d, -np.asarray(e, float)]).T
    s, _ = np.linalg.solve(A, np.asarray(q, float) - np.asarray(p, float))
    return np.asarray(p, float) + s*np.asarray(d, float)


def _notched(x0, x1, y_back, y_l, y_r, centre, rn, outer, chamfer):
    """A ring of the bearing between x0 and x1 (local frame, y the radius). Its back
    face lies at ``y_back``, its working face at ``y_l`` on the left and ``y_r`` on the
    right, and a raceway groove of radius ``rn`` about ``centre`` is cut into it.
    ``outer`` is the ring above the rolling element."""
    xc, yc = centre
    c = chamfer
    sgn = 1 if outer else -1
    pts = [(x0 + c, y_back), (x1 - c, y_back), (x1, y_back - sgn*c), (x1, y_r)]
    if rn is not None:
        sr, sl = (y_r - yc)/rn, (y_l - yc)/rn
        if outer:
            t = np.linspace(np.arcsin(sr), np.pi - np.arcsin(sl), 40)
        else:
            t = np.linspace(np.arcsin(sr), -np.pi - np.arcsin(sl), 40)
        pts += [(xc + rn*np.cos(a), yc + rn*np.sin(a)) for a in t]
    pts += [(x0, y_l), (x0, y_back - sgn*c)]
    return pts


def _barrel(P, t, m, L, r, crown, n=30):
    """A barrel roller about the centre ``P``: axis ``t``, across it ``m``, length
    ``L``, radius ``r`` at the middle, falling by ``crown`` at the ends."""
    u = np.linspace(-L/2, L/2, n)
    k = crown/(L/2)**2
    top = [P + a*t + (r - k*a*a)*m for a in u]
    bot = [P + a*t - (r - k*a*a)*m for a in u[::-1]]
    return np.array(top + bot)


def bearing_geometry(kind, r_in, r_out, B, alpha=None, ribs='NU', inner=True):
    """The parts of a bearing in section, in a local frame: x along the axis from the
    bearing's mid-plane, y the radius, upper half only. Returns a dict with ``rings``
    (polygons), ``balls`` ((centre, radius) pairs), ``rollers`` ((polygon, axis
    direction) pairs), ``lines`` (extra line segments) and ``contact`` ((point, unit
    direction) pairs: the load lines through the rolling elements, pointing from the
    outer ring towards the inner ring). See ``bearing_section`` for the kinds."""
    h = r_out - r_in
    r_m = r_in + h/2
    ch = 0.06*min(h, B)
    g = {'rings': [], 'balls': [], 'rollers': [], 'lines': [], 'contact': []}

    if kind in ('deep_groove', 'angular', 'four_point'):
        a = np.radians({'deep_groove': 0, 'angular': 40, 'four_point': 35}[kind]
                       if alpha is None else alpha)
        rb = min(0.28*h, 0.38*B)
        rn = 1.04*rb
        u = np.array([np.sin(a), np.cos(a)])                # to the outer contact
        C = np.array([0.0, r_m])
        if kind == 'angular':
            # deep shoulder of the outer ring on +x, of the inner ring on -x
            o_l, o_r = r_m + 0.78*rb, r_m + 0.25*rb
            i_l, i_r = r_m - 0.25*rb, r_m - 0.78*rb
            co, ci = C - (rn - rb)*u, C + (rn - rb)*u
            g['contact'].append((C, -u))
        else:
            o_l = o_r = r_m + 0.38*rb
            i_l = i_r = r_m - 0.38*rb
            up = np.array([0.0, 1.0])
            co, ci = C - (rn - rb)*up, C + (rn - rb)*up
            if kind == 'four_point':
                g['contact'] += [(C, -u), (C, -u*[-1, 1])]
            else:
                g['contact'].append((C, -up))
        g['rings'].append(_notched(-B/2, B/2, r_out, o_l, o_r, co, rn, True, ch))
        g['rings'].append(_notched(-B/2, B/2, r_in, i_l, i_r, ci, rn, False, ch))
        if kind == 'four_point':                            # the split inner ring
            g['lines'].append(((0, r_in), (0, ci[1] - rn)))
        g['balls'].append((C, rb))

    elif kind == 'self_aligning':
        rb = min(0.25*h, 0.22*B)
        xa = max(1.05*rb, 0.24*B)
        Rs = np.hypot(xa, r_m) + rb                          # the spherical outer raceway
        t = np.linspace(np.arccos(B/2/Rs), np.arccos(-B/2/Rs), 60)
        face = [(Rs*np.cos(s), Rs*np.sin(s)) for s in t]   # right to left
        g['rings'].append([(-B/2 + ch, r_out), (B/2 - ch, r_out), (B/2, r_out - ch)]
                          + face + [(-B/2, r_out - ch)])
        rn = 1.04*rb
        yi = r_m - 0.4*rb
        pts = [(-B/2 + ch, r_in), (B/2 - ch, r_in), (B/2, r_in + ch), (B/2, yi)]
        for xb in (xa, -xa):
            s = (yi - (r_m + rn - rb))/rn
            tt = np.linspace(np.arcsin(s), -np.pi - np.arcsin(s), 30)
            pts += [(xb + rn*np.cos(q), r_m + rn - rb + rn*np.sin(q)) for q in tt]
        pts += [(-B/2, yi), (-B/2, r_in + ch)]
        g['rings'].append(pts)
        for xb in (-xa, xa):
            C = np.array([xb, r_m])
            g['balls'].append((C, rb))
            g['contact'].append((C, -C/np.linalg.norm(C)))

    elif kind in ('cylindrical', 'needle'):
        rr = (0.25 if kind == 'cylindrical' else 0.3)*h
        L = (0.6 if kind == 'cylindrical' else 0.8)*B
        gap = 0.03*B
        yo, yi = r_m + rr, r_m - rr
        rib = r_m + 0.2*rr, r_m - 0.2*rr
        o_ribs = ribs in ('NU', 'NJ', 'NUP')
        i_ribs = {'NU': (False, False), 'N': (True, True), 'NJ': (True, False),
                  'NUP': (True, True)}[ribs]
        if o_ribs:
            xr = L/2 + gap
            g['rings'].append([(-B/2 + ch, r_out), (B/2 - ch, r_out), (B/2, r_out - ch),
                               (B/2, rib[0]), (xr, rib[0]), (xr, yo), (-xr, yo),
                               (-xr, rib[0]), (-B/2, rib[0]), (-B/2, r_out - ch)])
        else:
            g['rings'].append(_notched(-B/2, B/2, r_out, yo, yo, (0, 0), None, True, ch))
        if inner:
            xr = L/2 + gap
            pts = [(-B/2 + ch, r_in), (B/2 - ch, r_in), (B/2, r_in + ch)]
            pts += [(B/2, rib[1]), (xr, rib[1]), (xr, yi)] if i_ribs[1] else [(B/2, yi)]
            pts += [(-xr, yi), (-xr, rib[1]), (-B/2, rib[1])] if i_ribs[0] else [(-B/2, yi)]
            pts += [(-B/2, r_in + ch)]
            g['rings'].append(pts)
        C = np.array([0.0, r_m])
        g['rollers'].append((_barrel(C, np.array([1.0, 0]), np.array([0, 1.0]), L, rr,
                                     0.0, n=2), np.array([1.0, 0])))
        g['contact'].append((C, np.array([0, -1.0])))

    elif kind == 'tapered':
        # built with the large end of the rollers at +x; bearing_section mirrors it
        a = np.radians(15 if alpha is None else alpha)
        rr, L = 0.2*h, 0.55*B
        r_m = r_in + 0.42*h                                 # low, so the cup keeps some wall
        C = np.array([0.0, r_m])
        phi = a
        for _ in range(20):                                 # roller axis and half taper
            th = np.arcsin(rr*np.sin(phi)/r_m)
            phi = a - th
        apex = np.array([-r_m/np.tan(phi), 0.0])
        d_cup = np.array([np.cos(a), np.sin(a)])
        d_cone = np.array([np.cos(phi - th), np.sin(phi - th)])
        t = np.array([np.cos(phi), np.sin(phi)])
        n = sk.normal(t)
        e0, e1 = C - L/2*t, C + L/2*t                      # small and large end planes
        roller = [_line_x(apex, d_cone, e0, n), _line_x(apex, d_cone, e1, n),
                  _line_x(apex, d_cup, e1, n), _line_x(apex, d_cup, e0, n)]
        g['rollers'].append((np.array(roller), t))
        cup_y = lambda x: np.tan(a)*(x - apex[0])
        g['rings'].append([(-B/2 + ch, r_out), (B/2 - ch, r_out), (B/2, r_out - ch),
                           (B/2, cup_y(B/2)), (-B/2, cup_y(-B/2)), (-B/2, r_out - ch)])
        gap = 0.03*B
        big, small = e1 + gap*t, e0 - gap*t
        p_big = _line_x(apex, d_cone, big, n)
        p_small = _line_x(apex, d_cone, small, n)
        y_big = p_big[1] + 1.2*rr
        y_small = p_small[1] + 0.45*rr
        q_big = _line_x(big, n, (0, y_big), (1, 0))
        q_small = _line_x(small, n, (0, y_small), (1, 0))
        g['rings'].append([(-B/2 + ch, r_in), (B/2 - ch, r_in), (B/2, r_in + ch),
                           (B/2, y_big), q_big, p_big, p_small, q_small,
                           (-B/2, y_small), (-B/2, r_in + ch)])
        g['contact'].append((C, np.array([np.sin(a), -np.cos(a)])))

    elif kind in ('spherical_roller', 'carb'):
        if kind == 'carb':
            rr, L = 0.25*h, 0.62*B
            C = np.array([0.0, r_m])
            crown, gap = 0.18*rr, 0.03*h
            g['rollers'].append((_barrel(C, np.array([1.0, 0]), np.array([0, 1.0]), L, rr,
                                         crown), np.array([1.0, 0])))
            x = np.linspace(B/2, -B/2, 40)
            k = crown/(L/2)**2
            top = [(xi, r_m + rr + gap - k*xi*xi) for xi in x]
            bot = [(xi, r_m - rr - gap + k*xi*xi) for xi in x]
            g['rings'].append([(-B/2 + ch, r_out), (B/2 - ch, r_out), (B/2, r_out - ch)]
                              + top + [(-B/2, r_out - ch)])
            g['rings'].append([(-B/2 + ch, r_in), (B/2 - ch, r_in), (B/2, r_in + ch)]
                              + bot + [(-B/2, r_in + ch)])
            g['contact'].append((C, np.array([0, -1.0])))
        else:
            rr, L = 0.22*h, 0.36*B
            xa = 0.23*B
            gap = 0.03*h
            centres = [np.array([-xa, r_m]), np.array([xa, r_m])]
            Rs = np.hypot(xa, r_m) + rr
            t = np.linspace(np.arccos(B/2/Rs), np.arccos(-B/2/Rs), 60)
            face = [((Rs + gap)*np.cos(s), (Rs + gap)*np.sin(s)) for s in t]
            g['rings'].append([(-B/2 + ch, r_out), (B/2 - ch, r_out), (B/2, r_out - ch)]
                              + face + [(-B/2, r_out - ch)])
            inner_face = []
            for C in centres[::-1]:                         # right roller first
                m = C/np.linalg.norm(C)
                ax_t = np.array([m[1], -m[0]])
                poly = _barrel(C, ax_t, m, L, rr, 0.12*rr)
                g['rollers'].append((poly, ax_t))
                g['contact'].append((C, -m))
                u = np.linspace(L/2 + 0.04*B, -L/2 - 0.04*B, 20)
                inner_face += [C + a*ax_t - (rr + gap)*m for a in u]
            inner_face = np.array(inner_face)
            # the centre rib between the rows and the lips at the ends
            y_lip = r_m - 0.55*rr
            pts = [(-B/2 + ch, r_in), (B/2 - ch, r_in), (B/2, r_in + ch),
                   (B/2, min(y_lip, inner_face[0, 1]))]
            pts += [tuple(p) for p in inner_face[:20]]
            pts += [(0.02*B, r_m - 0.6*rr), (-0.02*B, r_m - 0.6*rr)]
            pts += [tuple(p) for p in inner_face[20:]]
            pts += [(-B/2, min(y_lip, inner_face[-1, 1])), (-B/2, r_in + ch)]
            g['rings'].append(pts)

    elif kind == 'thrust_ball':
        # the shaft washer on -x, the housing washer on +x
        tw = 0.28*B
        rb = 0.3*B
        rn = 1.04*rb
        C = np.array([0.0, r_m])
        clr = 0.06*h
        for side in (-1, 1):
            xs = side*B/2
            xf = side*(B/2 - tw)
            lo, hi = (r_in, r_out - clr) if side < 0 else (r_in + clr, r_out)
            cx = side*(rn - rb)
            a0 = np.arccos(np.clip((xf - cx)/rn, -1, 1))
            # the groove: the part of the circle about (cx, r_m) inside the washer
            tt = np.linspace(a0, 2*np.pi - a0, 30) if side < 0 else np.linspace(a0, -a0, 30)
            groove = [(cx + rn*np.cos(q), r_m + rn*np.sin(q)) for q in tt]
            k = -side*ch
            pts = [(xs, lo + ch), (xs, hi - ch), (xs + k, hi), (xf, hi)] + groove \
                + [(xf, lo), (xs + k, lo)]
            g['rings'].append(pts)
        g['balls'].append((C, rb))
        g['contact'].append((C, np.array([-1.0, 0])))

    elif kind == 'plain':
        g['rings'].append([(-B/2, r_in), (B/2, r_in), (B/2, r_out), (-B/2, r_out)])

    elif kind == 'linear_ball':
        rb = 0.16*h
        n_b = max(3, int(0.8*B/(2.2*rb)))
        xs = np.linspace(-0.4*B + rb, 0.4*B - rb, n_b)
        yb = r_in + rb
        g['rings'].append([(-B/2 + ch, r_out), (B/2 - ch, r_out), (B/2, r_out - ch),
                           (B/2, r_in + 0.6*rb), (0.42*B, r_in + 0.6*rb),
                           (0.42*B, yb + 1.15*rb), (-0.42*B, yb + 1.15*rb),
                           (-0.42*B, r_in + 0.6*rb), (-B/2, r_in + 0.6*rb),
                           (-B/2, r_out - ch)])
        for xb in xs:
            g['balls'].append((np.array([xb, yb]), rb))
        g['contact'].append((np.array([0.0, yb]), np.array([0, -1.0])))
    else:
        raise ValueError(f'unknown bearing kind {kind!r}')
    return g


def bearing_section(ax, x, r_in, r_out, B, kind='deep_groove', direction=1, alpha=None,
                    ribs='NU', inner=True, axis_y=0.0, halves=(1, -1), contact=0.0,
                    ring_color=sk.STEEL, angle=0.0, tilt=0.0, zorder=4):
    """A rolling or plain bearing cut along the shaft axis.

    The bearing sits with its mid-plane at ``x`` and its axis at ``axis_y``; ``r_in``
    and ``r_out`` are its bore and outside radii, ``B`` its width. ``halves`` picks the
    halves to draw: ``(1, -1)`` both, ``(1,)`` the upper one only.

    ``kind`` is one of ``'deep_groove'``, ``'angular'`` (contact angle ``alpha``,
    default 40 degrees), ``'four_point'``, ``'self_aligning'`` (ball), ``'cylindrical'``
    (``ribs`` 'NU', 'N', 'NJ' or 'NUP' say which ring carries the ribs that guide the
    rollers), ``'needle'`` (``inner=False`` runs the needles on the shaft),
    ``'tapered'`` (cup angle ``alpha``, default 15 degrees), ``'spherical_roller'``,
    ``'carb'`` (toroidal roller), ``'thrust_ball'``, ``'plain'`` (a bushing, in
    bronze) and ``'linear_ball'`` (a ball bushing for a linear guide, seen along the
    rod).

    ``angle`` turns the whole bearing (degrees, anticlockwise) about the point where its
    mid-plane meets the axis, for a vertical shaft. ``tilt`` turns only the inner ring
    and the rolling elements, with the shaft, while the outer ring (the first ring of a
    thrust bearing) stays in the housing: a misaligned shaft in a self-aligning bearing.

    ``direction`` (+1 or -1) is the axial direction of the force on the shaft that a
    one-way bearing (angular, tapered, thrust) carries; its load line then meets the
    axis on the opposite side. ``contact`` > 0 draws the load lines as dash-dot lines
    reaching that far either side of each rolling element. Returns the load lines
    in drawing coordinates as (point, unit direction) pairs, upper half first, so a
    figure can extend them to the axis.
    """
    flip = direction if kind != 'tapered' else -direction
    if kind == 'thrust_ball':
        flip = direction
    g = bearing_geometry(kind, r_in, r_out, B, alpha, ribs, inner)
    out = []
    R_all, R_in = sk.rot(angle), sk.rot(angle + tilt)
    O = np.array([x, axis_y])
    for s in halves:
        def T(p, s=s, R=R_in):
            p = np.asarray(p, float)
            return O + R @ np.array([flip*p[0], s*p[1]])

        def Td(d, s=s):
            return R_in @ np.array([flip*d[0], s*d[1]])

        face = BRONZE if kind == 'plain' else ring_color
        for k, poly in enumerate(g['rings']):
            R = R_all if k == 0 else R_in
            sk.body(ax, [T(p, R=R) for p in poly], facecolor=face, edgecolor=sk.EDGE,
                    zorder=zorder)
        for p, q in g['lines']:
            P, Q = T(p), T(q)
            ax.plot([P[0], Q[0]], [P[1], Q[1]], color=sk.EDGE, lw=sk.LW, zorder=zorder + 0.1)
        for c, r in g['balls']:
            sphere(ax, T(c), r, stops=BALL_STOPS, zorder=zorder + 0.2)
        for poly, t in g['rollers']:
            shade(ax, [T(p) for p in poly], ROLLER_STOPS, sk.normal(Td(t)),
                  zorder=zorder + 0.2)
        for c, d in g['contact']:
            P, D = T(c), sk.unit(Td(d))
            out.append((P, D))
            if contact > 0:
                sk.centreline(ax, P - contact*D, P + contact*D, zorder=zorder + 0.5)
    return out


def retaining_ring(ax, x, r, w, h, axis_y=0.0, internal=False, halves=(1, -1),
                   zorder=5):
    """A retaining ring (circlip) in its groove at ``x``: on a shaft of radius ``r``,
    or with ``internal=True`` in a bore of radius ``r``. ``w`` is the ring's thickness
    along the axis and ``h`` its radial height; two fifths of it sit in the groove."""
    for s in halves:
        y0 = r - 0.4*h if not internal else r - 0.6*h
        y = axis_y + s*y0 if s > 0 else axis_y - (y0 + h)
        sk.box(ax, (x - w/2, y), w, h, facecolor='#4a4f55', edgecolor=sk.EDGE,
               zorder=zorder)


def section(ax, points, facecolor=SECTION, hatch='////', hatch_color='0.55',
            zorder=2, mirror_y=None):
    """A cut face of a housing, a cover or a spacer: ``points`` filled and hatched,
    the hatch in a lighter grey than the black outline. ``mirror_y`` draws the mirror
    image about the line y = mirror_y as well, for the lower half of a turned part."""
    polys = [np.asarray(points, float)]
    if mirror_y is not None:
        m = polys[0].copy()
        m[:, 1] = 2*mirror_y - m[:, 1]
        polys.append(m)
    for pts in polys:
        ax.add_patch(Polygon(pts, closed=True, facecolor=facecolor, edgecolor=hatch_color,
                             hatch=hatch, lw=0, zorder=zorder))
        ax.add_patch(Polygon(pts, closed=True, facecolor='none', edgecolor=sk.EDGE,
                             lw=sk.LW, zorder=zorder + 0.01))


def face_stress(ax, x, r, side, N=None, sigma=None, axis_y=0.0, n=4, length=0.3,
                N_length=0.75, color=sk.GREEN, fontsize=13, zorder=6):
    """The normal stress on a cut face of a bar seen from the side, drawn positive.

    The face stands at ``x`` with half-height ``r``; ``side`` is +1 for a face that
    looks right and -1 for one that looks left, so the arrows leave the face (tension).
    ``n`` thin arrows of ``length`` spread over the face carry the stress, labelled
    ``sigma`` beside the lowest; the resultant is a thick arrow of ``N_length`` on the
    axis, labelled ``N`` past its head. Both are internal forces, green.
    """
    d = (side, 0)
    ys = np.linspace(-0.8*r, 0.8*r, n) if n > 1 else np.array([0.0])
    for yi in ys[np.abs(ys) > 0.15*r]:                   # the axis is kept for N
        sk.force(ax, (x, axis_y + yi), d, length, color=color, lw=1.2, zorder=zorder)
    sk.force(ax, (x, axis_y), d, N_length, color=color, lw=2.4, zorder=zorder + 0.1)
    if N is not None:
        sk.label(ax, (x + side*(N_length + 0.25), axis_y + 0.02), N, color=color,
                 fontsize=fontsize, zorder=zorder + 1)
    if sigma is not None:
        sk.label(ax, (x + side*(length + 0.22), axis_y + ys[0] - 0.2), sigma, color=color,
                 fontsize=fontsize, zorder=zorder + 1)


def stepped_shaft(ax, steps, axis_y=0.0, stops=STEEL_STOPS, zorder=3, ends=('flat', 'flat'),
                  waves=1.0, amplitude=0.06):
    """A turned shaft seen from the side, not cut: ``steps`` is a list of
    ``(x0, x1, r)``, one cylinder per step, each shaded across its diameter and
    outlined, so every change of diameter shows its edge. Shafts, pins and bolts stay
    uncut in an assembly section, as in a drawing. ``ends`` (left, right) takes the
    end kinds of ``sk.bar_outline``: ``'cut'`` draws a piece of a bar in a free body
    diagram, ``'round'`` a round-ended rod."""
    last = len(steps) - 1
    for i, step in enumerate(steps):
        kinds = (ends[0] if i == 0 else 'flat', ends[1] if i == last else 'flat')
        shade(ax, sk.bar_outline([step], kinds, waves, amplitude, axis_y), stops, (0, 1),
              zorder=zorder)


def bearing_end_view(ax, centre, r_in, r_out, n=9, angle=0.0, zorder=4):
    """A ball bearing seen along its axis: inner and outer ring, ``n`` balls on the
    pitch circle, the first at ``angle`` degrees. Returns the pitch radius and the ball
    radius."""
    c = np.asarray(centre, float)
    h = r_out - r_in
    r_m, rb = r_in + h/2, 0.3*h
    sk.annulus(ax, c, r_in, r_m - 0.55*rb, facecolor=sk.STEEL, zorder=zorder)
    sk.annulus(ax, c, r_m + 0.55*rb, r_out, facecolor=sk.STEEL, zorder=zorder)
    for k in range(n):
        a = np.radians(angle + 360*k/n)
        sphere(ax, c + r_m*np.array([np.cos(a), np.sin(a)]), rb, stops=BALL_STOPS,
               zorder=zorder + 0.2)
    return r_m, rb
