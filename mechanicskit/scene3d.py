"""3D problem figures from a scene spec: solids in, textbook SVG out.

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
          "mass": "#e3b9a0", "ground": "#9c8876"}
LEVELS = 14          # shade bands per part; enough to read as a gradient
RASTER_W = 1800      # z-buffer width in pixels; it only decides visibility
SIMPLIFY = 0.7       # Douglas-Peucker tolerance of the traced fills, in raster pixels


# ---------------------------------------------------------------- spec values
def ev(v, P):
    """A spec value is a number, an expression in the parameters, or a list of them."""
    if isinstance(v, str):
        return float(eval(v, {"__builtins__": {}}, dict(P, pi=np.pi, sqrt=np.sqrt)))
    if isinstance(v, (list, tuple)):
        return np.array([ev(x, P) for x in v], float)
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


def build(spec):
    """Evaluate the parameters and build every part: {id: ([solids], colour)}."""
    P = dict(spec["params"])
    parts = {}
    for part in spec["parts"]:
        args = {k: (v if k == "dir" else ev(v, P)) for k, v in part["args"].items()}
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
def annotate(ax, spec, P, cam):
    """Draw the 3D annotations; return the 3D sample points each one occupies."""
    samples = {}
    for a in spec.get("annotations", []):
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
                sk.label(ax, (cam(q0) + cam(q1))/2 + 22*n2 + shift, a["text"], fontsize=10.5,
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
        elif k == "label":
            color = {"load": sk.LOAD, "resultant": sk.BLUE}.get(a.get("role"), "black")
            sk.label(ax, cam(ev(a["at"], P)) + shift, a["text"], fontsize=12, halo=True,
                     zorder=9, color=color)
        else:
            raise ValueError(f"unknown annotation kind {k!r}")
    return samples


# --------------------------------------------------------------------- checks
def occlusion(solids, cam, samples):
    """Fraction of each annotation's sample points hidden behind a solid.

    A ray goes from every sample towards the viewer; points within 3 mm of a
    solid are attachment points (an arrow tip on a crank) and are skipped.
    """
    comp = b.Compound(solids)
    hit = IntCurvesFace_ShapeIntersector(); hit.Load(comp.wrapped, 1e-3)
    out = {}
    for name, pts in samples.items():
        pts = [p for p in pts if comp.distance_to(b.Vector(*p)) > 3.0]
        if pts:
            blocked = 0
            for p in pts:
                hit.Perform(gp_Lin(gp_Pnt(*p), gp_Dir(*cam.e)), 2.0, 1e5)
                blocked += hit.NbPnt() > 0
            out[name] = blocked/len(pts)
    return out


def text_checks(fig, ax, edges):
    """Labels that overlap each other, sit on drawn edges, or leave the figure."""
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
    fb = fig.bbox
    outside = [t.get_text() for t, bb in zip(texts, boxes)
               if bb.x0 < fb.x0 or bb.x1 > fb.x1 or bb.y0 < fb.y0 or bb.y1 > fb.y1]
    return {"label_overlaps": overlaps, "labels_on_edges": on_edges, "labels_outside": outside}


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
    return out


def score(rep):
    """Lower is better: occluded annotations dominate, then label collisions."""
    return (10*sum(rep["occluded_fraction"].values()) + 3*len(rep["label_overlaps"])
            + 0.02*sum(rep["labels_on_edges"].values()) + 5*len(rep["labels_outside"]))


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
        path = Path(spec)
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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mechanicskit.scene3d", description=__doc__.split("\n")[0])
    ap.add_argument("spec", help="scene spec JSON file")
    ap.add_argument("--out", help="output folder (default: next to the spec, named after it)")
    args = ap.parse_args(argv)
    print(json.dumps(run(args.spec, args.out), indent=1))


if __name__ == "__main__":
    main()
