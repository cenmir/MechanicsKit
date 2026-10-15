---
name: scene3d
description: Make a shaded 3D problem figure (SVG) from a textbook scan, a CAD screenshot, a photograph with arrows drawn on it, or a hand sketch, by writing a mechanicskit.scene3d JSON scene spec and iterating on its render. Use when asked to recreate or draw a figure "in 3D", an axonometric or three-quarter view of a machine or structure (winch, crank, bracket, shelf, mount, pulley, hinged plate), or to edit, re-render or preview a spec in tools/scene3d or examples/scene3d.
---

# 3D problem figures with scene3d

The full method is `docs/scene3d.qmd` in the MechanicsKit repo (PDF beside it): the spec
format, every primitive and annotation, the checks and the camera maths. **Read its
"The scene spec" chapter before the first figure of a session.** Find it with
`find ~ -path '*MechanicsKit/docs/scene3d.qmd' 2>/dev/null | head -1`. Six worked specs
are in `examples/scene3d/` (in the book: `tools/scene3d/`); start from the closest one.
2D figures follow `docs/figures.md` instead (skill `figure-redraw`).

Needs `mechanicskit[cad]` 0.22 or later: `scene3d --help` must work. In the mechanics
book run everything from the book root as `env -u PYTHONPATH .venv/bin/scene3d ...`.

## The loop

1. **Read the source** with Read. Write down the bodies, the supports, every load,
   dimension, angle and label, and what the problem is about (which dimensions enter
   the equations). Say what is ambiguous and ask before building if it changes the model
   (a support type, what a dimension measures, which way a force acts).
2. **Find the numbers.** If the problem lives in the book, grep the chapter for its
   coordinates and kinematics and copy them into `params` and `derived`, so the figure
   is the model the chapter computes with. Otherwise estimate from the image against
   one known size and say so in the spec's `comment`.
3. **Fix the frame and the camera.** Choose axes (z up, the wall or shaft on an axis).
   Read the camera off the source: from a CAD triad, or from two edges along x and y on
   the page, `sin^2(el) = tan(phi_x) tan(phi_y)` and `tan(az) = sin(el)/tan(phi_y)`.
   Convention: az = 0 puts the viewer on -y, az > 0 swings towards +x. List three
   cameras around the estimate.
4. **Write the spec** in `tools/scene3d/<name>.json`: parts from the primitives,
   annotations as 3D objects. Leave out what the problem does not need (cables, clutter).
5. **Render**: `scene3d tools/scene3d/<name>.json`. Read the printed report: physics
   checks, `occluded_fraction`, `label_overlaps`, `labels_on_edges`,
   `labels_far_from_anchor_pt`, score per camera.
6. **Look at every candidate PNG** in `tools/scene3d/<name>/` with Read, against the
   checklist below. The checks miss things a look catches, and the reverse.
7. **Edit the spec, never the engine, for this figure**: `screen_offset` and `text_gap`
   for labels, `dist`/`offset` for dimensions, `angle_deg`/`a0`/`a1` for arcs, a camera,
   a crop of a large part. Re-render. Two or three rounds are normal.
8. **Pick and deliver**: set `"pick"` if the best-scoring camera is not the best figure,
   render once more, and give the path of `<name>/<name>.svg`. For a chapter, render with
   `--out <Chapter>/graphics/<name>` and look at the figure before captioning it.

## Checklist for the visual review

- Same view and same story as the source; nothing the problem needs is hidden.
- Every load, dimension and angle present; values or symbols as in the problem.
- Ropes and belts tangent to their pulleys, loads hanging plumb, contacts touching.
- Force arrows in red with red labels, rotations green, geometric angles thin grey with a
  black label, dimensions grey; each label next to the thing it names.
- No arrow or extension line running along another line; no label on an edge.
- No thin part covering another on the page (no check sees this yet; use `pick` or move
  the part).
- A large body that hides the mechanism is cut away (as the TV in `tv_mount.json`).

## When the spec cannot express it

A missing part shape, annotation kind or check goes into `mechanicskit/scene3d.py`
(a `@primitive` function, a branch in `annotate` that returns sample points, or
`physics_checks`), with a row in the tables of `docs/scene3d.qmd` and a CHANGELOG line,
never into a local script. Keep primitives general and the problem's numbers in the
spec. Bump the version before pushing.

## Pitfalls met so far

- A parameter named like a derived point (`D` the depth and `D` the point) silently
  breaks expressions; give parameters distinct names.
- A word argument (`"+z"`) is passed through; anything else in `args` is evaluated.
- The first camera guess is often on the wrong side; check one known edge direction in
  the render against the source before polishing labels.
- An arc or label radius equal to a link length lands the label on the joint (θ₁ on B).
- A 16-bit PNG (ImageMagick `montage` output) draws blank in xelatex; convert to 8-bit.
- Never `quarto render` while a book preview runs; build the guide's PDF with pandoc.

## Preview tools

`scene3d <spec> --show --watch` shows the solids and annotations in the OCP CAD Viewer
and redraws on save; `scene3d --from-viewer` renders the viewer's current view and prints
its az/el for the spec; `--step f.step` exports the model. `scene3d --setup-vscode` adds
status-bar buttons for these to a VS Code workspace.
