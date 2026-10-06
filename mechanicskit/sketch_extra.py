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


def pedestal(ax, c, w, h, ground_w=None, depth=None, zorder=1):
    """A pin pedestal standing on the floor under ``c``: a trapezoid ``h`` high whose
    foot is ``2w`` wide, on a strip of ground."""
    y0 = c[1] - h
    gw = 2.6*w if ground_w is None else ground_w
    ground(ax, [(c[0] - gw/2, y0), (c[0] + gw/2, y0)], depth=0.35*h if depth is None else depth)
    sk.body(ax, [(c[0] - w, y0), (c[0] + w, y0), (c[0] + 0.45*w, c[1]),
                 (c[0] - 0.45*w, c[1])], zorder=zorder)


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
              fontsize=13):
        """A force of 3D ``length`` along the 3D direction ``d`` at the 3D point ``p``."""
        p, d = np.asarray(p, float), np.asarray(d, float)
        d = d/np.linalg.norm(d)
        tail, tip = (p - length*d, p) if head else (p, p + length*d)
        vec(ax, self(tail), self(tip), color, text, offset, fontsize)


def vec(ax, p0, p1, color, text=None, offset=(0, 0), fontsize=13, lw=2.0, zorder=6):
    """An arrow from ``p0`` to ``p1`` with its label beside the head."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=15,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=zorder))
    if text is not None:
        sk.label(ax, p1 + np.asarray(offset), text, fontsize=fontsize, zorder=zorder + 1)


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
