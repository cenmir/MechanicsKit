"""3D problem figures from a scene spec: solids in, figure SVG out.

A scene spec (JSON or a dict) names the problem parameters, the parts built
from named primitives, the annotations as 3D objects, and a few candidate
cameras. The pipeline renders every candidate and keeps the best one:

1. the parts become exact B-rep solids in build123d (OpenCascade kernel);
2. OpenCascade's hidden-line removal (HLR) gives the visible outlines and
   sharp edges as true vector curves;
3. the solids are tessellated, every triangle gets a Lambert shade band, a
   z-buffer decides which (part, band) is nearest the viewer at each pixel,
   and the regions are traced back into simplified vector fills;
4. annotations (forces, dimensions, radii, circles, labels) are given in 3D
   and projected with the same orthographic camera, drawn with
   ``mechanicskit.sketch`` so they follow the book's figure standard;
5. checks: rope or belt tangency and load placement from the 3D model,
   occlusion of every annotation by ray casting, label overlaps and labels
   sitting on drawn edges in screen space. Each candidate camera gets a score
   and the lowest score wins.

Usage::

    python -m mechanicskit.scene3d winch.json            # writes winch/ next to it
    python -m mechanicskit.scene3d winch.json --out figs/

    from mechanicskit import scene3d
    report = scene3d.run("winch.json")

Requires the optional CAD dependency: ``uv pip install "mechanicskit[cad]"``
(build123d, which brings the OCP bindings of OpenCascade).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path as MPath

from . import sketch as sk

try:
    import build123d as b
    from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
    from OCP.HLRAlgo import HLRAlgo_Projector
    from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir, gp_Lin
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_QuasiUniformDeflection
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopoDS import TopoDS
    from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
except ImportError as err:  # pragma: no cover
    raise ImportError("mechanicskit.scene3d needs build123d: "
                      'uv pip install "mechanicskit[cad]"') from err

COLORS = {"steel": sk.STEEL, "body": sk.BODY, "block": "#cfcfc9", "rope": "#c9c9c9",
          "mass": "#e3b9a0", "ground": "#9c8876", "sand": "#d39a7c", "wood": "#e8d2a6",
          "table": "#cfe0ea"}
LEVELS = 14          # shade bands per part; enough to read as a gradient
RASTER_W = 1800      # z-buffer width in pixels; it only decides visibility
SIMPLIFY = 0.7       # Douglas-Peucker tolerance of the traced fills, in raster pixels


# ---------------------------------------------------------------- spec values
MATH = dict(pi=np.pi, sqrt=np.sqrt, sin=np.sin, cos=np.cos, tan=np.tan, arctan2=np.arctan2,
            radians=np.radians, degrees=np.degrees, cross=np.cross, norm=np.linalg.norm,
            arccos=np.arccos, arcsin=np.arcsin, linspace=np.linspace,
            vec=lambda *c: np.array(c, float))


def ev(v, P):
    """A spec value: a number, an expression in the parameters (a scalar or a vector),
    or a list of them. Expressions see the parameters, the derived vectors and MATH."""
    if isinstance(v, str):
        r = eval(v, {"__builtins__": {}, **MATH, **P})
        return np.asarray(r, float) if np.ndim(r) else float(r)
    if isinstance(v, dict):
        return {k: ev(x, P) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        items = [ev(x, P) for x in v]
        if any(isinstance(x, dict) for x in items):
            return items
        return np.array(items, float)
    return v


# ----------------------------------------------------------------- primitives
PRIMITIVES = {}


def primitive(fn):
    """Register a part builder under its function name; it returns a solid or a list."""
    PRIMITIVES[fn.__name__] = fn
    return fn


def _rod(p0, p1, r):
    d = np.asarray(p1, float) - np.asarray(p0, float)
    return b.Solid.make_cylinder(r, float(np.linalg.norm(d)),
                                 b.Plane(origin=tuple(p0), z_dir=tuple(d)))


def _xcyl(x0, x1, r):
    return _rod((x0, 0, 0), (x1, 0, 0), r)


@primitive
def rod(p0, p1, r):
    """Cylinder from p0 to p1: shafts, ropes, hanging masses."""
    return _rod(p0, p1, r)


@primitive
def box(lo, hi):
    """Axis-aligned box between two corners."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return b.Solid.make_box(*(hi - lo), b.Plane(origin=tuple(lo)))


@primitive
def drum(x0, w, r_core, r_flange, t_flange, r_hub, hub_len):
    """Flanged drum on the x axis with a hub boss on each side."""
    s = _xcyl(x0, x0 + w, r_core)
    s = s.fuse(_xcyl(x0 - t_flange, x0, r_flange)).fuse(_xcyl(x0 + w, x0 + w + t_flange, r_flange))
    s = s.fuse(_xcyl(x0 + w + t_flange, x0 + w + t_flange + hub_len, r_hub))
    return s.fuse(_xcyl(x0 - t_flange - hub_len, x0 - t_flange, r_hub)).clean()


@primitive
def rope_wraps(x0, w, r, d):
    """Rope wound on a drum, one torus per turn; r is the radius to the rope centre."""
    n = int(w // d)
    return [b.Solid.make_torus(r, 0.49*d, b.Plane(origin=(x0 + d/2 + i*d, 0, 0), z_dir=(1, 0, 0)))
            for i in range(n)]


@primitive
def pillow_block(x0, x1, r_top, half_w, z_base, base_half_w, t_base):
    """Bearing block on the x axis: rounded housing on a base plate."""
    body = b.Solid.make_box(x1 - x0, 2*half_w, -(z_base + t_base),
                            b.Plane(origin=(x0, -half_w, z_base + t_base)))
    body = body.fuse(_xcyl(x0, x1, r_top))
    base = b.Solid.make_box(x1 - x0 + 40, 2*base_half_w, t_base,
                            b.Plane(origin=(x0 - 20, -base_half_w, z_base)))
    s = body.fuse(base).clean()
    try:
        s = s.fillet(6, [e for e in s.edges() if e.geom_type == b.GeomType.LINE])
    except Exception:
        pass
    return s


@primitive
def crank(x_mid, t, L, r_hub, r_tip, dir):
    """Tapered crank arm in the plane x = x_mid, pointing along dir, with hub and pin."""
    dir = np.asarray(dir, float)
    face = b.make_hull([b.Edge.make_circle(r_hub),
                        b.Edge.make_circle(r_tip, b.Plane.XY.move(b.Location((L, 0, 0))))])
    face = face if isinstance(face, b.Face) else face.faces()[0]
    arm = b.Solid.extrude(face, b.Vector(0, 0, t))
    try:
        arm = arm.fillet(5, arm.edges())
    except Exception:
        pass
    arm = arm.moved(b.Location(b.Plane(origin=(x_mid - t/2, 0, 0), x_dir=tuple(dir), z_dir=(1, 0, 0))))
    boss = _xcyl(x_mid - t/2 - 12, x_mid + t/2 + 12, 0.72*r_hub)
    tip = L*dir
    pin = _rod((x_mid + t/2, tip[1], tip[2]), (x_mid + t/2 + 8, tip[1], tip[2]), 0.45*r_tip)
    return arm.fuse(boss).fuse(pin).clean()


def _xform(s, origin=(0, 0, 0), x_dir=(1, 0, 0), z_dir=(0, 0, 1)):
    return s.moved(b.Location(b.Plane(origin=tuple(origin), x_dir=tuple(x_dir), z_dir=tuple(z_dir))))


@primitive
def motor(x0, x1, r, z_base, foot_half_w, flange_half_w, t_flange, vents=8, r_hub=None):
    """Electric motor on the x axis: housing, end cap with vent slots, foot and bolted flange.

    The output end is x1; the foot stands on the plane z = z_base.
    """
    s = _xcyl(x0, x1, r)
    cap = _xcyl(x1 - 0.12*(x1 - x0), x1, 1.05*r)
    s = s.fuse(cap)
    for i in range(vents):
        t = 2*np.pi*i/vents
        slot = b.Solid.make_box(14, 0.42*r, 0.17*r, b.Plane(origin=(x1 - 8, -0.21*r, -0.085*r)))
        slot = slot.moved(b.Location((0, 0, 0), (1, 0, 0), np.degrees(t)))
        slot = slot.moved(b.Location((0, 0.68*r*np.cos(t), 0.68*r*np.sin(t))))
        s = s.cut(slot)
    hub = _xcyl(x1, x1 + 0.12*r, r_hub or 0.32*r)
    foot = b.Solid.make_box(0.85*(x1 - x0), 2*foot_half_w, -0.4*r - z_base,
                            b.Plane(origin=(x0 + 0.08*(x1 - x0), -foot_half_w, z_base)))
    flange = b.Solid.make_box(0.85*(x1 - x0), 2*flange_half_w, t_flange,
                              b.Plane(origin=(x0 + 0.08*(x1 - x0), -flange_half_w, z_base)))
    s = s.fuse(hub).fuse(foot).fuse(flange)
    for xb in (x0 + 0.2*(x1 - x0), x0 + 0.8*(x1 - x0)):
        for yb in (-0.8*flange_half_w, 0.8*flange_half_w):
            s = s.fuse(_rod((xb, yb, z_base + t_flange), (xb, yb, z_base + t_flange + 9), 0.09*flange_half_w))
    return s.clean()


@primitive
def table(lo, hi, slot_x=None, slot_w=16, slot_d=10):
    """Rectangular plate between two corners, with an optional slot across it (along y) at x = slot_x."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    s = b.Solid.make_box(*(hi - lo), b.Plane(origin=tuple(lo)))
    if slot_x is not None:
        s = s.cut(b.Solid.make_box(slot_w, hi[1] - lo[1] + 2, slot_d,
                                   b.Plane(origin=(slot_x - slot_w/2, lo[1] - 1, hi[2] - slot_d))))
    return s


@primitive
def half_disc(centre, r, t, facing=(1, 0, 0), bar=None):
    """Flat half disc lying on a plane z = const (a miter gauge), the straight edge
    facing the direction ``facing``; ``bar`` = [length, width] adds a fence along that edge."""
    f = np.asarray(facing, float); f /= np.linalg.norm(f)
    disc = b.Solid.make_cylinder(r, t, b.Plane(origin=tuple(centre)))
    keep = b.Solid.make_box(2*r + 2, 2*r + 2, t + 2,
                            b.Plane(origin=(-r - 1, 0, -1)))  # local y = -facing: the round side
    keep = _xform(keep, centre, x_dir=(-f[1], f[0], 0))
    s = disc.intersect(keep)
    s = s if isinstance(s, b.Solid) else s.solids()[0]
    if bar is not None:
        L, w = bar
        fence = b.Solid.make_box(L, w, 1.6*t, b.Plane(origin=(-L/2, 0, 0)))
        s = s.fuse(_xform(fence, centre, x_dir=(-f[1], f[0], 0)))
    return s


@primitive
def plate(lo, hi, r_corner=0.0, round_side=None, holes=(), hole_r=2.5, slots=()):
    """Box between two corners; the thinnest direction is its thickness.

    ``round_side`` (e.g. "+z", "-y") rounds the two thickness-direction edges on that
    side with radius ``r_corner``; ``holes`` are 3D centres of through holes.
    ``slots`` are arc slots, each {"centre", "r", "w", "a0", "a1"}: a slot of width w
    on the circle of radius r about centre, from angle a0 to a1 in degrees, measured
    in the plate's plane from its first in-plane axis towards its second (a plate
    thin in y has the axes x and z, so 90 is straight up).
    """
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    s = b.Solid.make_box(*(hi - lo), b.Plane(origin=tuple(lo)))
    k = int(np.argmin(hi - lo))
    if round_side and r_corner > 0:
        j = "xyz".index(round_side[1]); at = hi[j] if round_side[0] == "+" else lo[j]
        axis = np.eye(3)[k]
        edges = [e for e in s.edges()
                 if abs(abs(np.dot(np.array(tuple(e.tangent_at(0.5))), axis)) - 1) < 1e-6
                 and abs(tuple(e.center())[j] - at) < 1e-6]
        s = s.fillet(r_corner, edges)
    for c in holes:
        c = np.asarray(c, float)
        p0, p1 = c.copy(), c.copy(); p0[k], p1[k] = lo[k] - 1, hi[k] + 1
        s = s.cut(_rod(p0, p1, hole_r))
    u, v = (np.eye(3)[i] for i in range(3) if i != k)
    n = np.eye(3)[k]
    for sl in slots:
        c = np.asarray(sl["centre"], float); c[k] = lo[k] - 1
        r, w = sl["r"], sl["w"]
        a = np.radians(np.linspace(sl["a0"], sl["a1"], 24))
        ray = lambda t, R: c + R*(np.cos(t)*u + np.sin(t)*v)
        depth = hi[k] - lo[k] + 2
        ring = _rod(c, c + depth*n, r + w/2).cut(_rod(c - n, c + (depth + 1)*n, r - w/2))
        wedge = b.Face(b.Wire.make_polygon([tuple(c)] + [tuple(ray(t, 2*r + w)) for t in a], close=True))
        sector = ring.intersect(b.Solid.extrude(wedge, b.Vector(*(depth*n))))
        sector = sector if isinstance(sector, b.Solid) else sector.solids()[0]
        for t in (a[0], a[-1]):
            sector = sector.fuse(_rod(ray(t, r), ray(t, r) + depth*n, w/2))
        s = s.cut(sector)
    return s


@primitive
def wire(points, r):
    """A bent round wire or rod through a polyline of points, with a ball at every
    joint so the bends read as smooth (wire brackets, hooks, springs drawn straight)."""
    pts = [np.asarray(q, float) for q in points]
    s = None
    for p0, p1 in zip(pts[:-1], pts[1:]):
        seg = _rod(p0, p1, r)
        s = seg if s is None else s.fuse(seg)
    for q in pts:
        s = s.fuse(b.Solid.make_sphere(r, b.Plane(origin=tuple(q))))
    return s.clean()


@primitive
def step(path):
    """Every solid in a STEP file, as modelled: a part from CAD drawn in its own coordinates.
    A relative path is read from the folder the renderer runs in (the book root)."""
    return list(b.import_step(str(path)).solids())


@primitive
def knob(p, dir, r, L):
    """A clamp knob or bolt head: a shaft of length L from p along dir with a domed end."""
    p, d = np.asarray(p, float), np.asarray(dir, float)
    d /= np.linalg.norm(d)
    end = p + L*d
    return _rod(p, end, r).fuse(b.Solid.make_sphere(r, b.Plane(origin=tuple(end)))).clean()


@primitive
def bar_link(points, w, h, normal, round_ends=True):
    """A rectangular link through a polyline of points: width w in the plane of the
    link, thickness h along ``normal``, rounded at every joint (and at the ends,
    unless ``round_ends`` is false, which gives a plain block for two points)."""
    pts = [np.asarray(p, float) for p in points]
    n = np.asarray(normal, float); n /= np.linalg.norm(n)
    s = None
    for p0, p1 in zip(pts[:-1], pts[1:]):
        d = p1 - p0; L = np.linalg.norm(d)
        box = b.Solid.make_box(L, w, h, b.Plane(origin=(0, -w/2, -h/2)))
        box = _xform(box, p0, x_dir=d/L, z_dir=n - np.dot(n, d/L)*d/L)
        s = box if s is None else s.fuse(box)
    for i, p in enumerate(pts):
        if round_ends or 0 < i < len(pts) - 1:
            s = s.fuse(_rod(p - h/2*n, p + h/2*n, w/2))
    return s.clean()


def _arg(v, P):
    """A part argument: evaluated if it is an expression, kept as is if it is a word ("+z")."""
    try:
        return ev(v, P)
    except (NameError, SyntaxError, TypeError):
        return v


def spec_path(path):
    """The spec file for a path: the path itself, or, for a file inside a render folder
    (report.json, a candidate SVG), the spec named after that folder next to it."""
    path = Path(path).resolve()
    if not path.exists():
        raise SpecError(f"{path} does not exist")
    if path.suffix == ".json":
        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError as err:
            raise SpecError(f"{path} is not valid JSON: {err}") from err
        if "parts" in d:
            return path
    sibling = path.parent.with_suffix(".json")
    if sibling.exists() and "parts" in json.loads(sibling.read_text()):
        print(f"note: {path} is render output; using the spec {sibling}")
        return sibling
    raise SpecError(f"{path} is not a scene spec (it has no 'parts'); "
                    "pass the spec JSON, e.g. tools/scene3d/tv_mount.json")


def load_spec(spec):
    """A spec dict from a dict, or from a spec path (see spec_path)."""
    return spec if isinstance(spec, dict) else json.loads(spec_path(spec).read_text())


class SpecError(ValueError):
    """The file given is not a usable scene spec."""


def build(spec):
    """Evaluate the parameters and build every part: {id: ([solids], colour)}."""
    P = dict(spec["params"])
    for k, v in spec.get("derived", {}).items():  # in order, each may use the ones before
        P[k] = ev(v, P)
    parts = {}
    for part in spec["parts"]:
        args = {k: _arg(v, P) for k, v in part["args"].items()}
        s = PRIMITIVES[part["kind"]](**args)
        parts[part["id"]] = (s if isinstance(s, list) else [s],
                             COLORS.get(part["color"], part["color"]))
    return P, parts


# --------------------------------------------------------------------- camera
class Camera:
    """Orthographic camera at azimuth az and elevation el, in degrees.

    az = 0 looks along +y (the viewer on the -y side), az > 0 swings the viewer
    towards +x, el lifts it above the xy plane. ``e`` points from the scene to
    the viewer; X and Y are the screen axes, so a point projects to (p.X, p.Y).
    """

    def __init__(self, az, el):
        a, h = np.radians(az), np.radians(el)
        self.az, self.el = az, el
        self.e = np.array([np.cos(h)*np.sin(a), -np.cos(h)*np.cos(a), np.sin(h)])
        self.X = np.cross([0, 0, 1], self.e); self.X /= np.linalg.norm(self.X)
        self.Y = np.cross(self.e, self.X)

    def __call__(self, p):
        p = np.asarray(p, float)
        return np.stack([p @ self.X, p @ self.Y], axis=-1)

    def depth(self, p):
        return np.asarray(p, float) @ self.e


# --------------------------------------------------------------- hidden lines
def _edges(compound, defl=0.25):
    out = []
    if compound.IsNull():
        return out
    ex = TopExp_Explorer(compound, TopAbs_EDGE)
    while ex.More():
        q = GCPnts_QuasiUniformDeflection(BRepAdaptor_Curve(TopoDS.Edge(ex.Current())), defl)
        if q.IsDone() and q.NbPoints() > 1:
            out.append(np.array([[q.Value(i).X(), q.Value(i).Y()] for i in range(1, q.NbPoints() + 1)]))
        ex.Next()
    return out


def hidden_lines(solids, cam):
    """Visible silhouettes ("outline") and sharp edges, in screen coordinates.

    The HLR projector frame has its main direction along e and its x direction
    along cam.X, so HLR output and ``cam(p)`` share one coordinate system.
    Smooth (tangent) edges are left out, as in a hand drawing.
    """
    algo = HLRBRep_Algo()
    for s in solids:
        algo.Add(s.wrapped)
    algo.Projector(HLRAlgo_Projector(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*cam.e), gp_Dir(*cam.X))))
    algo.Update()
    algo.Hide()
    h = HLRBRep_HLRToShape(algo)
    return {"outline": _edges(h.OutLineVCompound()), "sharp": _edges(h.VCompound())}


# -------------------------------------------------------------------- shading
def tone(base, I):
    """Shade of a base colour at intensity I in [0, 1]: 70 % of it to a pastel tint."""
    c = np.array(matplotlib.colors.to_rgb(base))
    dark, light = 0.70*c, c + 0.45*(1 - c)
    return dark + (light - dark)*I


def triangles(parts, cam, light):
    """Front-facing triangles of every part, their vertex depths and shade groups.

    ``light`` is given in the camera frame (X right, Y up, e towards the
    viewer), so the key light stays at the viewer's upper left for any camera.
    """
    L = light[0]*cam.X + light[1]*cam.Y + light[2]*cam.e
    L /= np.linalg.norm(L)
    tris, depth, gid, groups = [], [], [], []
    for name, (solids, _) in parts.items():
        for s in solids:
            V, T = s.tessellate(0.6, 0.15)
            V = np.array([(v.X, v.Y, v.Z) for v in V]); T = np.array(T)
            A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
            n = np.cross(B - A, C - A)
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            front = n @ cam.e > 1e-6
            I = 0.25 + 0.75*np.clip(n @ L, 0, 1)
            lev = np.minimum((I*LEVELS).astype(int), LEVELS - 1)
            for k in np.nonzero(front)[0]:
                key = (name, int(lev[k]))
                if key not in groups:
                    groups.append(key)
                tri = np.array([A[k], B[k], C[k]])
                tris.append(cam(tri)); depth.append(cam.depth(tri))
                gid.append(groups.index(key) + 1)
    return np.array(tris), np.array(depth), np.array(gid), groups


def id_raster(tris, depth, gid, lim):
    """Z-buffer: the shade group nearest the viewer at every pixel (0 is background)."""
    (x0, x1), (y0, y1) = lim
    W = RASTER_W; H = int(round(W*(y1 - y0)/(x1 - x0)))
    px = (tris[..., 0] - x0)/(x1 - x0)*W
    py = (y1 - tris[..., 1])/(y1 - y0)*H
    zbuf = np.full((H, W), -np.inf); ids = np.zeros((H, W), int)
    for (ax_, bx, cx), (ay, by, cy), (za, zb, zc), g in zip(px, py, depth, gid):
        i0, i1 = max(int(min(ay, by, cy)), 0), min(int(max(ay, by, cy)) + 1, H)
        j0, j1 = max(int(min(ax_, bx, cx)), 0), min(int(max(ax_, bx, cx)) + 1, W)
        den = (by - cy)*(ax_ - cx) + (cx - bx)*(ay - cy)
        if i1 <= i0 or j1 <= j0 or abs(den) < 1e-9:
            continue
        J, I = np.meshgrid(np.arange(j0, j1) + 0.5, np.arange(i0, i1) + 0.5)
        l1 = ((by - cy)*(J - cx) + (cx - bx)*(I - cy))/den
        l2 = ((cy - ay)*(J - cx) + (ax_ - cx)*(I - cy))/den
        l3 = 1 - l1 - l2
        z = l1*za + l2*zb + l3*zc
        win = (l1 >= -1e-4) & (l2 >= -1e-4) & (l3 >= -1e-4) & (z > zbuf[i0:i1, j0:j1])
        zbuf[i0:i1, j0:j1][win] = z[win]; ids[i0:i1, j0:j1][win] = g
    xs = x0 + (np.arange(W) + 0.5)*(x1 - x0)/W
    ys = y1 - (np.arange(H) + 0.5)*(y1 - y0)/H
    return ids, xs, ys


def _rdp(P, tol):
    """Douglas-Peucker simplification of an open polyline."""
    keep = np.zeros(len(P), bool); keep[[0, -1]] = True
    stack = [(0, len(P) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, d = P[i], P[j] - P[i]
        seg = P[i + 1:j] - a
        nd = np.hypot(*d)
        dist = np.abs(seg[:, 0]*d[1] - seg[:, 1]*d[0])/nd if nd > 0 else np.hypot(seg[:, 0], seg[:, 1])
        k = int(np.argmax(dist))
        if dist[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return P[keep]


def _traced(xs, ys, mask, tol):
    """The region where mask is true as one simplified compound matplotlib path."""
    gen = plt.figure()
    cs = gen.gca().contourf(xs, ys, mask.astype(float), levels=[0.5, 2])
    verts, codes = [], []
    for path in cs.get_paths():
        for ring in path.to_polygons(closed_only=True):
            if len(ring) < 4:
                continue
            r = _rdp(np.vstack([ring[:-1], ring[:1]]), tol)
            if len(r) < 4:
                continue
            verts += [*r[:-1], r[0]]
            codes += [MPath.MOVETO] + [MPath.LINETO]*(len(r) - 2) + [MPath.CLOSEPOLY]
    plt.close(gen)
    return MPath(verts, codes) if verts else None


def fills(ax, ids, xs, ys, groups, parts):
    """Trace the z-buffer into vector fills: an underlay per part, then its shade bands."""
    tol = SIMPLIFY*(xs[1] - xs[0])
    for name, (_, color) in parts.items():
        mine = [g for g, (n, _) in enumerate(groups, start=1) if n == name]
        if not mine:
            continue
        under = _traced(xs, ys, np.isin(ids, mine), tol)
        if under is not None:
            ax.add_patch(PathPatch(under, facecolor=tone(color, 0.55), edgecolor="none", zorder=0.9))
        for g in mine:
            lev = groups[g - 1][1]
            mask = ids == g
            if mask.sum() < 6:
                continue
            path = _traced(xs, ys, mask, tol)
            if path is not None:
                c = tone(color, (lev + 0.5)/LEVELS)
                ax.add_patch(PathPatch(path, facecolor=c, edgecolor=c, lw=0.3, zorder=1))


# ---------------------------------------------------------------- annotations
NUMERIC_FIELDS = {"dir", "length", "offset", "dist", "axis", "r", "r0", "angle_deg", "a0", "a1",
                  "text_gap"}
def annotate(ax, spec, P, cam):
    """Draw the 3D annotations; return the 3D sample points each one occupies.

    Each label is also recorded with the screen point it names (ax._s3d_anchors),
    so the checks can flag a label that has drifted away from its object.
    """
    samples = {}
    ax._s3d_anchors = []

    def anchored(anchor, artist):
        if artist is not None:
            ax._s3d_anchors.append((artist, np.asarray(anchor, float)))

    for a in spec.get("annotations", []):
        a = {key: (ev(v, P) if key in NUMERIC_FIELDS else v) for key, v in a.items()}
        k = a["kind"]
        shift = np.array(a.get("screen_offset", [0, 0]), float)
        if k == "force":
            p, d = ev(a["at"], P), np.array(a["dir"], float)
            L, head = a["length"], a.get("head", False)
            d2 = cam(p + d) - cam(p)
            color = {"load": sk.LOAD, "resultant": sk.BLUE, "internal": sk.GREEN}[a.get("role", "load")]
            sk.force(ax, cam(p), d2, L*np.linalg.norm(d2), a.get("text"), head=head,
                     color=color, text_color=color, fontsize=13, zorder=8,
                     offset=shift if a.get("screen_offset") else None)
            samples[a["id"]] = np.linspace(p - L*d, p, 12) if head else np.linspace(p, p + L*d, 12)
        elif k == "dim":
            p0, p1 = ev(a["p0"], P), ev(a["p1"], P)
            o = np.array(a["offset"], float); o /= np.linalg.norm(o)
            q0, q1 = p0 + a["dist"]*o, p1 + a["dist"]*o
            for p, q in ((p0, q0), (p1, q1)):
                ax.plot(*zip(cam(p + 12*o), cam(q + 14*o)), color=sk.GREY, lw=0.7, zorder=6)
            ax.add_patch(FancyArrowPatch(cam(q0), cam(q1), arrowstyle="<|-|>", mutation_scale=9,
                                         color=sk.GREY, lw=0.8, shrinkA=0, shrinkB=0, zorder=6))
            u = cam(q1) - cam(q0); u /= np.linalg.norm(u)
            n2 = cam(o) - cam(0*o); n2 /= np.linalg.norm(n2)
            if a.get("text_at") == "outside":
                sk.label(ax, cam(q1) + 26*u + shift, a["text"], fontsize=10.5, halo=True, zorder=9,
                         ha="left" if u[0] > 0 else "right")
            else:
                sk.label(ax, (cam(q0) + cam(q1))/2 + a.get("text_gap", 22)*n2 + shift, a["text"], fontsize=10.5,
                         halo=True, zorder=9)
            samples[a["id"]] = np.vstack([np.linspace(q0, q1, 15), np.linspace(p0 + 12*o, q0, 6),
                                          np.linspace(p1 + 12*o, q1, 6)])
        elif k in ("radius", "circle"):
            c, axis, r = ev(a["centre"], P), np.array(a["axis"], float), ev(a["r"], P)
            u = np.cross(axis, [0, 0, 1]) if abs(axis[2]) < 0.9 else np.array([1., 0, 0])
            u /= np.linalg.norm(u); v = np.cross(axis, u)
            if k == "circle":
                t = np.linspace(0, 2*np.pi, 200)
                ring = c + r*(np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
                ax.plot(*cam(ring).T, color=sk.GREY, lw=0.7, ls=(0, (6, 2, 1, 2)), zorder=5)
                continue
            t = np.radians(a.get("angle_deg", 0)); rad = np.cos(t)*u + np.sin(t)*v
            p0, p1 = c + a.get("r0", 0)*rad, c + r*rad
            ax.add_patch(FancyArrowPatch(cam(p0), cam(p1), arrowstyle="-|>", mutation_scale=10,
                                         color="black", lw=0.9, shrinkA=0, shrinkB=0, zorder=6))
            w = cam(p1) - cam(p0); nrm = np.array([-w[1], w[0]])/np.linalg.norm(w)
            nrm *= np.sign(nrm[1]) or 1
            sk.label(ax, (cam(p0) + cam(p1))/2 + a.get("text_gap", 30)*nrm + shift, a["text"],
                     fontsize=10.5, halo=True, zorder=9)
            samples[a["id"]] = np.linspace(p0, p1, 12)
        elif k == "arc":
            c, axis, r = ev(a["centre"], P), np.array(a["axis"], float), ev(a["r"], P)
            u = np.cross(axis, [0, 0, 1]) if abs(axis[2]) < 0.9 else np.array([1., 0, 0])
            u /= np.linalg.norm(u); v = np.cross(axis, u)
            t = np.radians(np.linspace(a["a0"], a["a1"], 60))
            arc = c + r*(np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
            s2 = cam(arc)
            role = a.get("role", "rotation")
            color = {"rotation": sk.GREEN, "load": sk.LOAD, "angle": sk.GREY}[role]
            if role == "angle":  # a geometric angle: thin, no head, black label
                ax.plot(*s2.T, color=color, lw=0.8, zorder=7)
                color = "black"
            else:
                ax.plot(*s2[:-3].T, color=color, lw=1.6, zorder=8, solid_capstyle="round")
                ax.add_patch(FancyArrowPatch(s2[-4], s2[-1], arrowstyle="-|>", mutation_scale=13,
                                             color=color, lw=1.6, shrinkA=0, shrinkB=0, zorder=8))
            if a.get("text"):
                anchored(s2, sk.label(ax, s2[len(s2)//2] + shift, a["text"], color=color,
                                                  fontsize=13, zorder=9))
            samples[a["id"]] = arc[::5]
        elif k == "line":
            p0, p1 = ev(a["p0"], P), ev(a["p1"], P)
            ls = {"centre": (0, (12, 3, 2, 3)), "dashed": (0, (5, 3)), "solid": "-"}[a.get("style", "centre")]
            ax.plot(*cam(np.array([p0, p1])).T, color=a.get("color", sk.GREY), lw=a.get("lw", 0.7), ls=ls, zorder=5)
            if a.get("style", "centre") != "centre":  # centre lines are drawn through parts
                samples[a["id"]] = np.linspace(p0, p1, 10)
        elif k == "point":
            p = ev(a["at"], P)
            ax.plot(*cam(p), "o", color="black", ms=a.get("ms", 6), zorder=9,
                    markeredgecolor="white", markeredgewidth=0.6)
            if a.get("text"):
                anchored(cam(p), sk.label(ax, cam(p) + (shift if a.get("screen_offset") else np.array([12, 10])),
                                          a["text"], fontsize=12, halo=True, zorder=10))
        elif k == "cog":
            p = ev(a["at"], P)
            sk.cog(ax, cam(p), r=a.get("size", 14), zorder=9)
            if a.get("text"):
                anchored(cam(p), sk.label(ax, cam(p) + shift, a["text"], fontsize=12, zorder=10))
        elif k == "triad":
            o, L = ev(a["at"], P), a.get("length", 50)
            cols = a.get("colors", ["#d62728", "#2ca02c", "#1f5fbf"])
            names = a.get("labels", ["$x$", "$y$", "$z$"])
            for i in range(3):
                tip = o + L*np.eye(3)[i]
                ax.add_patch(FancyArrowPatch(cam(o), cam(tip), arrowstyle="-|>", mutation_scale=13,
                                             color=cols[i], lw=2.2, shrinkA=0, shrinkB=0, zorder=8))
                if names[i]:
                    d = cam(tip) - cam(o); d /= np.linalg.norm(d)
                    sk.label(ax, cam(tip) + 14*d, names[i], color=cols[i], fontsize=12, zorder=9)
            samples[a["id"]] = np.vstack([np.linspace(o, o + L*np.eye(3)[i], 6)[1:] for i in range(3)])
        elif k == "label":
            color = {"load": sk.LOAD, "resultant": sk.BLUE}.get(a.get("role"), "black")
            sk.label(ax, cam(ev(a["at"], P)) + shift, a["text"], fontsize=12, halo=True,
                     zorder=9, color=color)
        else:
            raise ValueError(f"unknown annotation kind {k!r}")
    return samples


# --------------------------------------------------------------------- checks
def _inside(solid, p, tol=0.5):
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.TopAbs import TopAbs_IN
    c = BRepClass3d_SolidClassifier(solid.wrapped, gp_Pnt(*p), tol)
    return c.State() == TopAbs_IN


def occlusion(solids, cam, samples):
    """Fraction of each annotation's sample points hidden behind a solid.

    A ray goes from every sample towards the viewer, starting 2 mm out so that
    a sample lying on a visible face (an arrow drawn on a board) counts as seen.
    Samples strictly inside a solid (an extension line starting in a hub) are
    attachment points and are skipped.
    """
    comp = b.Compound(solids)
    hit = IntCurvesFace_ShapeIntersector(); hit.Load(comp.wrapped, 1e-3)
    out = {}
    for name, pts in samples.items():
        pts = [p for p in pts if not any(_inside(sol, p) for sol in solids)]
        if pts:
            blocked = 0
            for p in pts:
                hit.Perform(gp_Lin(gp_Pnt(*p), gp_Dir(*cam.e)), 2.0, 1e5)
                blocked += hit.NbPnt() > 0
            out[name] = blocked/len(pts)
    return out


def text_checks(fig, ax, edges):
    """Labels that overlap each other, sit on drawn edges, leave the figure, or stand
    more than 14 pt from the point or arc they name."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in ax.texts if t.get_text()]
    boxes = [t.get_window_extent(r).expanded(1.05, 1.1) for t in texts]
    overlaps = [(texts[i].get_text(), texts[j].get_text()) for i in range(len(boxes))
                for j in range(i + 1, len(boxes)) if boxes[i].overlaps(boxes[j])]
    allpts = [e for v in edges.values() for e in v]
    pts = ax.transData.transform(np.vstack(allpts)) if allpts else np.zeros((0, 2))
    on_edges = {}
    for t, bb in zip(texts, boxes):
        n = int(((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) & (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1)).sum())
        if n:
            on_edges[t.get_text()] = n
    far = {}
    for t, anchor in getattr(ax, "_s3d_anchors", []):
        bb = t.get_window_extent(r)
        q = ax.transData.transform(np.atleast_2d(anchor))  # a point, or the points of an arc
        dx = np.maximum.reduce([bb.x0 - q[:, 0], np.zeros(len(q)), q[:, 0] - bb.x1])
        dy = np.maximum.reduce([bb.y0 - q[:, 1], np.zeros(len(q)), q[:, 1] - bb.y1])
        gap = float(np.min(np.hypot(dx, dy)))*72/fig.dpi  # points, to the nearest anchor point
        if gap > 14:
            far[t.get_text()] = round(gap, 1)
    fb = fig.bbox
    outside = [t.get_text() for t, bb in zip(texts, boxes)
               if bb.x0 < fb.x0 or bb.x1 > fb.x1 or bb.y0 < fb.y0 or bb.y1 > fb.y1]
    return {"label_overlaps": overlaps, "labels_on_edges": on_edges, "labels_outside": outside,
            "labels_far_from_anchor_pt": far}


def physics_checks(spec, P, parts):
    """Checks on the 3D model itself, independent of the camera."""
    out = {}
    c = spec.get("checks", {})
    if "rope_tangent" in c:
        rt = c["rope_tangent"]
        args = next(p for p in spec["parts"] if p["id"] == rt["rope"])["args"]
        p0, p1 = ev(args["p0"], P), ev(args["p1"], P)
        a0, a1 = (np.array(v, float) for v in rt["axis"])
        u = (a1 - a0)/np.linalg.norm(a1 - a0)
        radial = (p0 - a0) - ((p0 - a0) @ u)*u
        d = (p1 - p0)/np.linalg.norm(p1 - p0)
        out["rope_tangent"] = {"radius_err_mm": float(abs(np.linalg.norm(radial) - ev(rt["r"], P))),
                               "cos_rope_radius": float(abs(d @ radial)/np.linalg.norm(radial))}
    if "load_below" in c:
        lb = c["load_below"]
        top = max(s.bounding_box().max.Z for s in parts[lb["load"]][0])
        low = min(s.bounding_box().min.Z for s in parts[lb["above"]][0])
        out["load_below"] = {"clearance_mm": float(low - top)}
    for pid, want in c.get("bbox_expect", {}).items():
        bb = [s.bounding_box() for s in parts[pid][0]]
        have = {"xmin": min(x.min.X for x in bb), "xmax": max(x.max.X for x in bb),
                "ymin": min(x.min.Y for x in bb), "ymax": max(x.max.Y for x in bb),
                "zmin": min(x.min.Z for x in bb), "zmax": max(x.max.Z for x in bb)}
        out[f"bbox_{pid}_err_mm"] = {k: float(have[k] - ev(v, P)) for k, v in want.items()}
    return out


def score(rep):
    """Lower is better: occluded annotations dominate, then label collisions."""
    return (10*sum(rep["occluded_fraction"].values()) + 3*len(rep["label_overlaps"])
            + 0.02*sum(rep["labels_on_edges"].values()) + 5*len(rep["labels_outside"])
            + 2*len(rep.get("labels_far_from_anchor_pt", {})))


# ---------------------------------------------------------------------- render
def render(spec, P, parts, cam, out, width=5.2):
    """Draw one camera to out.svg and out.png and return its check report."""
    solids = [s for v, _ in parts.values() for s in v]
    edges = hidden_lines(solids, cam)
    tris, depth, gid, groups = triangles(parts, cam, spec.get("light", [-0.25, 0.45, 0.85]))
    pts = tris.reshape(-1, 2)
    (x0, y0), (x1, y1) = pts.min(0) - 70, pts.max(0) + 70
    fig = plt.figure(figsize=(width, width*(y1 - y0)/(x1 - x0)))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_aspect("equal"); ax.axis("off")
    ids, xs, ys = id_raster(tris, depth, gid, ((x0, x1), (y0, y1)))
    fills(ax, ids, xs, ys, groups, parts)
    ax.add_collection(LineCollection(edges["sharp"], colors="0.15", linewidths=0.6, zorder=2,
                                     capstyle="round"))
    ax.add_collection(LineCollection(edges["outline"], colors="0.1", linewidths=0.9, zorder=3,
                                     capstyle="round"))
    samples = annotate(ax, spec, P, cam)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    report = {"camera": {"az": cam.az, "el": cam.el},
              "occluded_fraction": occlusion(solids, cam, samples),
              **text_checks(fig, ax, edges)}
    report["score"] = score(report)
    out = Path(out)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return report


def run(spec, out=None):
    """Render every candidate camera, score them, copy the best to <name>.svg.

    ``spec`` is a path to a JSON file or a dict. Output goes to ``out``
    (default: a folder named after the spec file, next to it). The returned
    report is also written to report.json there.
    """
    if not isinstance(spec, dict):
        path = spec_path(spec)
        spec = json.loads(path.read_text())
        out = Path(out) if out else path.with_suffix("")
    out = Path(out or spec.get("name", "scene"))
    out.mkdir(parents=True, exist_ok=True)
    P, parts = build(spec)
    report = {"physics": physics_checks(spec, P, parts), "candidates": []}
    for c in spec.get("cameras", [{"az": 30, "el": 25}]):
        cam = Camera(c["az"], c["el"])
        report["candidates"].append(render(spec, P, parts, cam, out/f"az{c['az']}_el{c['el']}",
                                           width=spec.get("width_in", 5.2)))
    best = min(report["candidates"], key=lambda r: r["score"])
    pick = spec.get("pick") or best["camera"]
    stem = f"az{pick['az']}_el{pick['el']}"
    name = spec.get("name", "scene")
    shutil.copy(out/f"{stem}.svg", out/f"{name}.svg")
    shutil.copy(out/f"{stem}.png", out/f"{name}.png")
    report["best"], report["used"] = best["camera"], pick
    (out/"report.json").write_text(json.dumps(report, indent=1))
    return report


def _labelled(parts):
    """One labelled build123d compound per part, for a CAD viewer or a STEP file."""
    out = []
    for name, (solids, color) in parts.items():
        c = b.Compound(solids)
        c.label, c.color = name, b.Color(color)
        out.append(c)
    return out


def _annotation_edges(spec, P):
    """The annotations as 3D polylines: their sample points, joined where they are close."""
    fig = plt.figure()
    samples = annotate(fig.gca(), spec, P, Camera(30, 25))
    plt.close(fig)
    out = {}
    for name, pts in samples.items():
        pts = np.asarray(pts, float)
        if len(pts) < 2:
            continue
        step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        gap = 3*np.median(step) + 1e-6
        edges = [b.Edge.make_line(tuple(p0), tuple(p1))
                 for p0, p1, d in zip(pts[:-1], pts[1:], step) if 1e-6 < d < gap]
        if edges:
            out[name] = b.Compound(edges)
    return out


def export_step(spec, path):
    """Write the parts to a STEP file, one named body per part (for SolidWorks, FreeCAD, Onshape)."""
    spec = load_spec(spec)
    P, parts = build(spec)
    b.export_step(b.Compound(children=_labelled(parts)), str(path))
    return path


def show(spec, keep_camera=False):
    """Send the parts and the annotations to the OCP CAD Viewer (VS Code or browser).

    Needs ``uv pip install ocp_vscode`` and an open viewer: in VS Code the
    "OCP CAD Viewer: Open viewer" command (Ctrl+K V), or ``python -m ocp_viewer``
    (``uv pip install ocp_viewer``) for a browser viewer on port 3939.
    """
    import ocp_vscode as ov
    spec = load_spec(spec)
    P, parts = build(spec)
    objs, names, colors = [], [], []
    for c in _labelled(parts):
        objs.append(c); names.append(c.label); colors.append(parts[c.label][1])
    role = {a["id"]: a for a in spec.get("annotations", [])}
    for name, e in _annotation_edges(spec, P).items():
        kind = role[name]["kind"]
        objs.append(e); names.append(f"ann_{name}")
        colors.append(sk.LOAD if kind == "force" else sk.GREEN if kind == "arc" else "#555555")
    if ov.get_port() is None:
        ov.find_and_set_port()  # a viewer opened since the last call
    if ov.get_port() is None:
        raise RuntimeError("no OCP CAD Viewer is running. In VS Code run the command "
                           "'OCP CAD Viewer: Open viewer' (Ctrl+K V), or start a browser viewer "
                           "with 'python -m ocp_viewer' (uv pip install ocp_viewer) and open http://localhost:3939/viewer")
    ov.show(*objs, names=names, colors=colors, ortho=True, black_edges=True,
            reset_camera=ov.Camera.KEEP if keep_camera else ov.Camera.RESET)


def viewer_camera():
    """The az/el of the OCP CAD Viewer's current view, in this module's convention.

    The view direction is the unit vector from the viewer's target to its camera
    position; roll is ignored, since the figures keep z up.
    """
    import ocp_vscode as ov
    if ov.get_port() is None:
        ov.find_and_set_port()
    if ov.get_port() is None:
        raise RuntimeError("no OCP CAD Viewer is running; open it and show the spec first")
    st = ov.status()
    e = np.asarray(st["position"], float) - np.asarray(st["target"], float)
    e /= np.linalg.norm(e)
    az = float(np.degrees(np.arctan2(e[0], -e[1])))
    el = float(np.degrees(np.arcsin(np.clip(e[2], -1, 1))))
    return {"az": round(az, 1), "el": round(el, 1)}


def render_view(spec, out=None, camera=None):
    """Render one camera (default: the viewer's current view) to <out>/<name>_view.svg/.png."""
    if not isinstance(spec, dict):
        path = spec_path(spec)
        spec = json.loads(path.read_text())
        out = Path(out) if out else path.with_suffix("")
    out = Path(out or spec.get("name", "scene"))
    out.mkdir(parents=True, exist_ok=True)
    camera = camera or viewer_camera()
    P, parts = build(spec)
    rep = render(spec, P, parts, Camera(camera["az"], camera["el"]),
                 out/f"{spec.get('name', 'scene')}_view", width=spec.get("width_in", 5.2))
    rep["physics"] = physics_checks(spec, P, parts)
    return rep


def watch(spec_path, interval=0.5):
    """Redraw in the CAD viewer every time the spec file is saved; Ctrl-C stops."""
    import time
    path = globals()["spec_path"](spec_path)
    seen, first = None, True
    print(f"watching {path} (Ctrl-C to stop)")
    while True:
        m = path.stat().st_mtime
        if m != seen:
            seen = m
            try:
                show(path, keep_camera=not first)
                first = False
                print(time.strftime("%H:%M:%S"), "shown")
            except Exception as err:  # a half-edited spec must not end the watch
                print(time.strftime("%H:%M:%S"), "error:", repr(err))
        time.sleep(interval)


VSCODE_TASKS = [
    ("scene3d: watch in viewer", "--show --watch", True,
     "$(eye) 3D watch", "Show the open spec in the OCP CAD Viewer, redraw on save"),
    ("scene3d: render this view", "--from-viewer", False,
     "$(device-camera) Render view", "Render the textbook SVG from the viewer's current view"),
    ("scene3d: render all cameras", "", False,
     "$(files) Render all", "Render and score every camera in the spec"),
]
VSCODE_EXTENSIONS = ["bernhard-42.ocp-cad-viewer", "spencerwmiles.vscode-task-buttons"]


def _read_jsonc(path):
    """A VS Code JSON file: // comments and trailing commas allowed. {} if absent."""
    import re
    if not path.exists():
        return {}
    text = path.read_text()
    text = re.sub(r'("(?:\\.|[^"\\])*")|//[^\n]*', lambda m: m.group(1) or "", text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return json.loads(text) if text.strip() else {}


def setup_vscode(folder=".", install_extensions=True):
    """Add the scene3d tasks, status-bar buttons and viewer extensions to a VS Code workspace.

    The tasks run on the file open in the editor and call the ``scene3d`` of the
    Python running this setup, so they use that environment. Existing tasks and
    settings are kept; files are rewritten without their comments.
    """
    import os
    import subprocess
    import sys
    folder = Path(folder).resolve()
    vs = folder/".vscode"; vs.mkdir(exist_ok=True)
    exe = Path(sys.executable).parent/("scene3d.exe" if os.name == "nt" else "scene3d")
    run = (f"env -u PYTHONPATH '{exe}'" if os.name != "nt" else f'"{exe}"')
    tasks_path = vs/"tasks.json"
    tasks = _read_jsonc(tasks_path) or {"version": "2.0.0", "tasks": []}
    mine = {label for label, *_ in VSCODE_TASKS}
    tasks["tasks"] = [t for t in tasks.get("tasks", []) if t.get("label") not in mine]
    for label, opts, background, _, _ in VSCODE_TASKS:
        t = {"label": label, "type": "shell",
             "command": f"{run} '${{file}}' {opts}".strip(), "problemMatcher": [],
             "presentation": {"reveal": "always", "panel": "dedicated" if background else "shared",
                              "clear": not background}}
        if background:
            t["isBackground"] = True
        tasks["tasks"].append(t)
    tasks_path.write_text(json.dumps(tasks, indent=2) + "\n")
    settings_path = vs/"settings.json"
    settings = _read_jsonc(settings_path)
    buttons = [b_ for b_ in settings.get("VsCodeTaskButtons.tasks", []) if b_.get("task") not in mine]
    buttons += [{"label": icon, "task": label, "tooltip": tip} for label, _, _, icon, tip in VSCODE_TASKS]
    settings["VsCodeTaskButtons.tasks"] = buttons
    settings_path.write_text(json.dumps(settings, indent=4) + "\n")
    done = [str(tasks_path), str(settings_path)]
    if install_extensions:
        code = shutil.which("code")
        if code is None:
            done.append("extensions not installed: no 'code' command; install "
                        + " and ".join(VSCODE_EXTENSIONS) + " from the Extensions view")
        else:
            for ext in VSCODE_EXTENSIONS:
                r = subprocess.run([code, "--install-extension", ext], capture_output=True, text=True)
                done.append(f"{ext}: {'installed' if r.returncode == 0 else r.stderr.strip()[:200]}")
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(prog="scene3d", description=__doc__.split("\n")[0])
    ap.add_argument("spec", nargs="?", help="scene spec JSON file (default: the last one used)")
    ap.add_argument("--out", help="output folder (default: next to the spec, named after it)")
    ap.add_argument("--show", action="store_true", help="show the solids in the OCP CAD Viewer instead of rendering")
    ap.add_argument("--watch", action="store_true", help="with --show: redraw on every save of the spec")
    ap.add_argument("--step", metavar="FILE", help="also export the solids to a STEP file")
    ap.add_argument("--setup-vscode", nargs="?", const=".", metavar="FOLDER",
                    help="add scene3d tasks, status-bar buttons and viewer extensions to a VS Code workspace")
    ap.add_argument("--no-extensions", action="store_true", help="with --setup-vscode: do not install extensions")
    ap.add_argument("--from-viewer", action="store_true",
                    help="render only the CAD viewer's current view, to <name>_view.svg/.png")
    args = ap.parse_args(argv)
    if args.setup_vscode:
        for line in setup_vscode(args.setup_vscode, not args.no_extensions):
            print(line)
        print("reload the VS Code window (Developer: Reload Window) to see the buttons")
        return
    last = Path.home()/".cache"/"mechanicskit"/"scene3d_last"
    if args.spec is None:
        if not last.exists():
            ap.error("no spec given and no previous one to reuse")
        args.spec = last.read_text().strip()
        print(f"spec: {args.spec}")
    try:
        args.spec = str(spec_path(args.spec))
    except SpecError as err:
        raise SystemExit(f"error: {err}")
    last.parent.mkdir(parents=True, exist_ok=True)
    last.write_text(str(Path(args.spec).resolve()))
    if args.from_viewer:
        try:
            rep = render_view(args.spec, args.out)
        except RuntimeError as err:
            raise SystemExit(f"error: {err}")
        cam = rep["camera"]
        print(json.dumps(rep, indent=1))
        print(f'\ncamera for the spec: {{"az": {cam["az"]}, "el": {cam["el"]}}}')
        return
    if args.step:
        print("wrote", export_step(args.spec, args.step))
    if args.show or args.watch:
        try:
            watch(args.spec) if args.watch else show(args.spec)
        except RuntimeError as err:
            raise SystemExit(f"error: {err}")
        except KeyboardInterrupt:
            pass
        return
    if not args.step:
        print(json.dumps(run(args.spec, args.out), indent=1))


if __name__ == "__main__":
    main()
