"""The scene3d pipeline one layer at a time, for docs/scene3d.qmd.

    python examples/scene3d/layers.py      # writes docs/scene3d/layer_*.png; montage them into layers.png
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from mechanicskit import scene3d as s3

here = Path(__file__).parent
out = here.parent.parent/"docs"/"scene3d"
out.mkdir(parents=True, exist_ok=True)
spec = json.loads((here/"winch.json").read_text())
P, parts = s3.build(spec)
cam = s3.Camera(20, 22)
solids = [s for v, _ in parts.values() for s in v]
edges = s3.hidden_lines(solids, cam)
tris, depth, gid, groups = s3.triangles(parts, cam, spec["light"])
pts = tris.reshape(-1, 2)
(x0, y0), (x1, y1) = pts.min(0) - 70, pts.max(0) + 70
ids, xs, ys = s3.id_raster(tris, depth, gid, ((x0, x1), (y0, y1)))


def panel(name, draw):
    fig = plt.figure(figsize=(3.2, 3.2*(y1 - y0)/(x1 - x0)))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_aspect("equal"); ax.axis("off")
    draw(ax)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    fig.savefig(out/f"layer_{name}.png", dpi=220)
    plt.close(fig)


def lines(ax):
    ax.add_collection(LineCollection(edges["sharp"], colors="0.15", linewidths=0.6))
    ax.add_collection(LineCollection(edges["outline"], colors="0.1", linewidths=0.9))


panel("1_hlr", lines)
panel("2_zbuffer", lambda ax: ax.imshow(np.ma.masked_equal(ids, 0) % 20, cmap="tab20", extent=(x0, x1, y0, y1),
                                        interpolation="nearest"))
panel("3_fills", lambda ax: s3.fills(ax, ids, xs, ys, groups, parts))
