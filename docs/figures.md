# Redrawing problem figures as SVG

This guide is the method behind the problem figures in Mirza Cenanovic's mechanics book: a
lecture slide, a hand drawing or a photo of a sketch goes in, and a
short Python script comes out that draws the same problem in the book's own style and
saves it as SVG. It is written for a person and for Claude Code alike. The matching
Claude Code skill is [`skills/figure-redraw/SKILL.md`](../skills/figure-redraw/SKILL.md).

Every rule below was paid for: each one comes from a figure that was wrong, a correction
Mirza made, or a pitfall in the code. The figure scripts of the book (`tools/*_figures.py`
in the mechanicsBook repo, about thirty of them) are the body of examples, and
[`examples/figures/bar_spring_figures.py`](../examples/figures/bar_spring_figures.py) is
a small, complete one that runs from this repo alone.

## Why a script, and why SVG

A figure that is a script can be regenerated when a number, a convention or the house
style changes. When the label style changed in October 2026, all 128 SVGs of the book
were regenerated in one pass; a hand-drawn figure would have been redrawn by hand. The
script also draws from the problem's engineering parameters, so the picture is the
problem the equations describe, not a trace of someone else's drawing.

SVG keeps line art sharp at any size, keeps text as glyph paths (no font dependency in
the browser) and can be opened in Inkscape for a hand touch-up that then flows back into
the script. PNG is the exception, for raster content and animations only, and then at
200 dpi or more.

## Setup

```bash
pip install git+https://github.com/cenmir/mechanicskit.git     # or uv pip install -e . in a clone
sudo apt install librsvg2-bin imagemagick                      # rsvg-convert and montage, to look at renders
```

Inkscape is optional and only needed for the hand-edit round trip. Importing
`mechanicskit` sets matplotlib's maths font to Computer Modern (`mathtext.fontset='cm'`,
`font.family='serif'`), which is how the figures match the book's typeset maths. A later
`plt.style.use(...)` resets this; set the two rcParams again after it, or do not call it.

## The parts

`mechanicskit.sketch` (import as `sk`) draws the parts every problem figure is made of:

| Call | Draws |
|---|---|
| `sk.canvas(figsize)` | figure and axes: equal aspect, axes off, white |
| `sk.ground(ax, points, depth)` | a fixed surface with a fading band (use `sketch_extra.ground` for SVG) |
| `sk.body(ax, points)` | a filled polygon in `BODY` tan |
| `sk.link(ax, p0, p1, width)` | a round-ended bar |
| `sk.pin(ax, xy, r)` | a pin: white disc, black edge |
| `sk.spring(ax, p0, p1, coils, width)` | a zigzag spring |
| `sk.coil(ax, p0, p1, coils, width)` | a smooth spring of S-shaped waves, as FBD Lab draws it |
| `sk.dashpot(ax, p0, p1, width, lead, cup)` | a damper, cylinder end at p0, at any angle |
| `sk.gas_spring(ax, p0, p1, width, tube, rod)` | a gas spring or hydraulic cylinder in steel, tube at p0, eyes centred on both pins |
| `sk.helix_centres(d, p, n_active, n_closed)` / `sk.helical_spring(ax, x0, D, d, z)` | a helical compression spring in side view, wire sections and front bands, closed end coils |
| `sk.box(ax, xy, w, h, angle, centred)` | a block or crate: corner (or centre) and size, turned |
| `sk.trapezoid(ax, centre, w_bottom, w_top, h, angle)` | a wedge or a pad |
| `sk.ellipse(ax, centre, w, h, angle)` | a disc or a wheel seen at an angle |
| `sk.cog(ax, p, r)` | the centre-of-mass symbol, a quartered circle |
| `sk.gear(ax, centre, N, module, angle)` / `sk.hub` | an involute spur gear, a hub |
| `sk.force(ax, point, direction, length, text, head=False, offset=None)` | a force arrow; starts at `point`, or ends there with `head=True` |
| `sk.moment_vector(ax, point, direction, length, text, head_gap)` | a moment or torque as a double-headed vector, right-hand rule; unambiguous from any side |
| `sk.angle(ax, centre, r, a0, a1, text, text_r)` | an angle arc, anticlockwise from `a0` to `a1` degrees |
| `sk.dimension(ax, p0, p1, offset, text, gap, text_side, text_offset, text_at, text_shift)` | a dimension with extension lines, `offset` to the left of p0→p1; `offset=0` marks the span on p0→p1 itself with no extension lines; `text_offset` puts the value that far to the left of the line, `text_at` that fraction along it |
| `sk.axes(ax, origin, length, angle_deg, labels)` | a pair of coordinate axes |
| `sk.triad(ax, origin, dirs, labels, length, gap, offsets)` | axes along any directions, each letter past its tip, clear of the head |
| `sk.unit_vectors(ax, p, [(dir, label, offset), ...], length)` | black unit vectors, such as e_t and e_n |
| `sk.arrow(ax, p0, p1, color, lw, ms, style, ls, rad)` | a plain arrow (not a load): velocity, axis, displacement; `style='<\|-\|>'` for two heads, `rad` bends it |
| `sk.guide(ax, p0, p1, ls)` | a grey dashed construction line |
| `sk.centreline(ax, p0, p1)` | a dash-dot centre line |
| `sk.direction_line(ax, p0, p1, head=True)` | a dashed line of action with an open triangular head |
| `sk.break_line(ax, p0, p1, waves, amplitude)` | a wavy line where a part is cut off |
| `sk.rotation(ax, centre, r, a0, a1)` | a green curved arrow; `a1 < a0` for clockwise |
| `sk.curl(ax, centre, rx, ry, a0, a1, back_zorder=None)` | a curved arrow on an ellipse: a torque about a shaft seen from the side, or wrapping it with the back half hidden |
| `sk.radius(ax, centre, r, deg, text, at)` | an arrow from the centre out to the radius |
| `sk.leader(ax, point, start, text)` | an arrow in to an edge, labelled beyond its tail: a fillet radius |
| `sk.label(ax, xy, text, color, fontsize, halo=None, bg=None, **text_kwargs)` | a label; see [Labels](#labels) for the outline and the box |
| `sk.unit(v)`, `sk.normal(v)`, `sk.polar(deg, r)`, `sk.rot(deg)` | unit vector, left normal, the vector at an angle, the rotation matrix (degrees) |
| `sk.arc(centre, r, a0, a1, n)`, `sk.ccw(pts)`, `sk.mirror_outline(top)`, `sk.outward(d, point, centre)` | arc points, an outline made anticlockwise, a turned part's outline from its upper half, the direction pointing out of a structure |

`mechanicskit.sketch_extra` adds what the book needed beyond those:

| Call | Does |
|---|---|
| `save_svg(fig, path, tight=True)` | saves the SVG with named parts, real gradients and the data-to-SVG map; **always use it** instead of `fig.savefig` |
| `ground(ax, points, depth)` | the fixed surface as one SVG gradient per segment |
| `shade(ax, pts, stops, direction)` | a polygon with a linear gradient along `direction` |
| `shaded_bar(ax, p0, p1, width, stops)` | a round-ended bar shaded across its thickness, like a cylinder |
| `shaded_rect(ax, xy, w, h, stops, axis)` | a rectangle shaded along x or y |
| `STEEL_STOPS`, `TAN_STOPS`, `BRASS_STOPS`, `DARK_STEEL_STOPS`, `CHROME_STOPS` | gradient stops: dark edge, light band, dark edge |
| `vec(ax, p0, p1, color, text, offset)` | an arrow from p0 to p1 with a label at the head |
| `dot(ax, p, ms)` | a black point |
| `circled(ax, xy, text)` | a green number in a circle, to tag bodies in a free body diagram |
| `HALO` | the white outline, for raw `ax.text`: `path_effects=HALO` |
| `support_pin`, `support_roller`, `support_rollers`, `support_slider`, `pedestal`, `wall`, `hatched_wall`, `sliding_block`, `bracket` | supports and blocks on the gradient ground; `angle` turns the pin, roller and slider supports |
| `Proj(ex, ey, ez)`, `Proj.from_view(d, up)` | a fixed axonometric projection for 3D figures, given by the page images of the unit vectors or by a viewing direction, with `.axes`, `.force` and `.arc` |
| `tone(base, normal, light)` | a face colour lit from `light`, for shaded 3D faces |
| `vcylinder`, `rod3d`, `sphere`, `box3d`, `thread3d`, `ring`, `speckle` | 3D solids for `Proj` figures |
| `embed_svg`, `embed_svg_at`, `data_to_svg` | paste a licensed SVG drawing into a saved figure |
| `SVG_GRADIENTS` | `True`: real SVG gradients; set `False` for raster output (animations), which builds them from strips |

`python -m mechanicskit.svg_roundtrip fig.svg` reports hand edits (see
[The hand-edit round trip](#the-hand-edit-round-trip)).

## The method

### 1. Read the source image before anything else

Open the image and look at it; never work from its file name or from the prose around
it. Two figures in the book were once captioned "Tension" and "Compression" from their
names; they were a tensile-test record and a diagram from another chapter.
For several images at once, read a contact sheet:

```bash
montage -label '%f' *.png -tile 3x -geometry 380x380+6+6 sheet.png
```

Run `identify` first: some legacy PNGs are animations with hundreds of frames
(`'file.png[0]'` takes the first). When a PNG and an SVG of the same figure exist,
trust the PNG; several legacy SVGs are incomplete.

Then write down what the figure must carry: the bodies and how they are supported, every
load, every given dimension and angle and which line each angle is measured from, the
coordinate axes and origin, and the point labels the problem text refers to. If the
text says "node 1" or "the angle from the horizontal", the figure has to show node 1
and an arc from the horizontal.

### 2. Get the geometry from the problem, not from the pixels

Put the problem's parameters at the top of the script in drawing units, with the scale
in a comment (`R = 3.0  # r = d = 0.6 m drawn as 3 units`), and compute every point from
them. A gear train uses the true module and tooth counts and sums pitch radii for the
centre distance; a mechanism solves its loop closure; a ground spring end is ground to
d/4 as in the standard. When one dimension is not to scale on purpose (an interference
drawn a hundred times too large, or it would not show), say so in the script's docstring
and in the caption.

Pick one drawing unit so that parts meet: rollers seat on the channel walls, the wheel is
as wide as its slot, the plates are big against the shaft. A figure where parts nearly
touch reads as wrong.

### 3. Draw it in the house style

Write `<chapter>_figures.py` with one function per figure (template below). Draw with the
palette, the label rules and the force conventions of the next sections. Replace what
the source needed but the problem does not: people holding a board become posts or
support symbols (and the prose is rewritten to match), decorative fills go, redundant
dimensions go. Use the book's notation, not the source's (bold italic vectors, the
book's symbol names).

Keep 3D when the original was 3D for a reason. A 2D redraw of a clevis pin could not
show which kind of joint sits at A and B; the redraw went back to 3D with `Proj`.

### 4. Render, look, fix, and repeat

Never ship a first render. Rasterise and read it:

```bash
rsvg-convert -z 2 -b white fig.svg -o fig.png          # whole figure
convert fig.png -crop 600x400+900+300 crop.png          # zoom into a crowded spot
```

Two to five rounds are normal; the friction screw free body diagrams took five. Go
through the checklist in [Checking a render](#checking-a-render) on every round, and fix
all the problems of a round in one batch before rendering again.

### 5. Compare with the original, side by side

```bash
convert original.png -resize x500 a.png; rsvg-convert -h 500 -b white fig.svg -o b.png
convert a.png b.png +append compare.png
```

The redraw must carry the same information as the original, in our style. Mirza
rejected the first clutch redraw because it did not match the original, and asked for it
"bigger and more clear": the second version had bigger plates, shorter shafts, a
taller aspect and the area element centred on the axis. For the lead screw the
projection was changed until the viewpoint matched the original's. Clearer than the original
is the goal; different from it is not.

### 6. Put it in the document

See [In the mechanics book](#in-the-mechanics-book) for captions, sizes and credits.

## A figure script

```python
"""Draw the figures for the examples in Kinetics/WorkEnergyPower.ipynb.

    WorkEnergy_collar.svg        a collar on a curved guide, held by a spring to the wall
    WorkEnergy_collar_fbd.svg    the collar at an angle theta, in natural coordinates

    python tools/work_energy_figures.py [name ...]
"""
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mechanicskit import sketch as sk
from mechanicskit.sketch_extra import dot, ground, save_svg

OUT = pathlib.Path(__file__).resolve().parent.parent / 'Kinetics' / 'graphics'
FS = 14                                  # one label size for every figure in the set

R = 3.0                                  # r = 0.6 m drawn as 3 units


def _save(fig, name):
    save_svg(fig, OUT / name)
    plt.close(fig)


def collar():
    fig, ax = sk.canvas((6.2, 5.0))
    ...
    ax.set_xlim(-R - 0.9, R + 0.6)       # always explicit limits for 2D figures
    ax.set_ylim(-R - 1.0, 1.6)
    _save(fig, 'WorkEnergy_collar.svg')


FIGURES = {'collar': collar}

if __name__ == '__main__':
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
```

The docstring lists every output with one line, and the run line. Files are named
`<Chapter>_<thing>.svg`, with `_fbd` for the free body diagram. A setup figure and its
free body diagram usually share a `_scene(ax, ...)` helper and the module constants, so
the two stay consistent. A series that must read at one scale shares its `XLIM`, `YLIM`
and `figsize`. Figure sizes run from about 4.4×4 inches for a single figure to about
8.6×5 for a setup with its free body diagram beside it.

A failing label (an unsupported TeX command) raises in `__main__` and stops every later
figure, so run all figures after a change and check that each one was written.

## House style

### Palette

| Constant | Value | For | Its label |
|---|---|---|---|
| `sk.BODY` | `#d8ba94` tan | brackets, links, blocks, pedestals | black point letters |
| `sk.GROUND` | `#9c8876` | fixed surfaces, a band fading from 0.9 to 0 opacity | none |
| `sk.EDGE` | black | outlines, `sk.LW = 1.1` | black |
| `sk.LOAD` | `#c00000` red | applied forces and moments, weights, reactions, contact forces | red |
| `sk.BLUE` | `#4472c4` | resultants, displacements, velocities, degrees of freedom | blue |
| `sk.GREEN` | `#2e7d32` | internal forces and moments, rotations, circled body numbers | green |
| `sk.GREY` | `0.35` | construction lines, angle arcs, dimensions | black |
| `sk.STEEL` / `sk.STEEL_EDGE` | `#bcd4e3` / `#236b8e` | machine parts, gears, balls, pulleys | black |

**Colour carries meaning, so the label of an arrow is drawn in the arrow's colour.** A
reader then pairs every symbol with its arrow at a glance, even where arrows crowd at a
contact point. The screw figures of the Friction chapter are the reference: the weight
$W$, the push $M/r$, the friction $F$ and the normal force $N$ are red with red labels,
their resultant $R$ is blue with a blue label, and the angles $\theta$ and $\alpha$, the
normal $n$ and the points are black. Everything that is geometry (points, angles,
dimensions, axes, unit vectors) stays black.

```python
sk.force(ax, B, e_CB, 70, r'$\mathbf{F}_S$', text_color=sk.LOAD)          # applied: red
sk.force(ax, O, e_R, 1.2, '$R$', color=sk.BLUE, text_color=sk.BLUE)       # resultant: blue
sk.force(ax, cut, (0, 1), 9, '$V$', color=sk.GREEN, text_color=sk.GREEN)  # internal: green
sk.rotation(ax, c, 5, a0=200, a1=-70)                                       # green by default
sk.label(ax, c + (9, 4), '$M$', color=sk.GREEN)                             # its label too
sk.angle(ax, O, 1.0, 0, 30, r'$\theta$')                                     # geometry: black
```

`sk.force` draws its label black unless `text_color` is given, so pass it every time.

Unit vectors are black and thinner than loads (`sk.force(..., color='black', lw=1.4)`)
so they do not read as forces. A ghost position (start, end, apparent position) is the
same part at `alpha=0.35`. A part that has to stand out (a hot hub, a package) takes a
local accent colour such as `#f2b48a` or `#e8a87c`. Backgrounds are white; never a dark
background, and no decorative fills (a green fill under a stress curve was removed as
clutter). In contour plots the colour map stays the same across a chapter, and the
caption names the quantity the colours show.

### Lines and layers

- Centre lines: grey, `ls=(0, (10, 3, 2, 3))`, lw 0.6 to 0.8. Dotted construction:
  `ls=(0, (2, 3))`. Hidden edges (the bore of a hollow shaft) are dashed.
- zorder bands: ground 1, bodies 2, outlines and ropes 3, pins 4 to 5, arrows 6,
  labels 7 and up. Supports sit under the beam (support 1, beam 2). Use fractional
  zorders (2.1, 2.15) for the painter's order inside one part.
- A revolved part seen from the side gets thin edge lines across it at each change of
  diameter (`color=sk.STEEL_EDGE, lw=0.9`), or it reads as a flat plate.
- Shading reads as material: a bar or shaft is shaded across its thickness
  (`shaded_bar`), a sectioned face is hatched (`hatch='////'`).

### Labels

- **Every label has the white outline.** This is Mirza's standing instruction (October
  2026). `sk.label` does it by default since 0.9.4, and so do the labels that `force`,
  `angle`, `dimension` and `axes` draw. A raw `ax.text` passes `path_effects=HALO`.
- **Move a label rather than back it.** The outline is the safety net, not the
  placement. In Mirza's words, "move the labels around to not need a background; in
  case the label needs to sit on top of a line, we need a background". White boxes
  behind labels were removed from the whole book in September 2026; `bg=` still exists
  but is rarely right.
- Put a point label diagonally off its point, on the side away from every member, arrow
  and support that meets it. The same point often needs a different offset in the setup
  and in the free body diagram, because the reactions arrive there.
- **Outline or box.** `halo=False` turns the outline off for one label and
  `sk.LABEL_HALO = False` for the whole script. `bg='white'` puts a box behind the
  text instead of the outline: `sk.dimension` does this for its value by default,
  the drafting style where the value interrupts the line (`bg=None` gives the outline).
- **A label takes the colour of the arrow it names**: red for applied forces, blue for a
  resultant or displacement, green for internal forces and moments. Labels of geometry
  (points, angles, dimensions, axes) stay black. Coloured force labels make a free body
  diagram easier to read (Mirza, 8 October 2026, replacing an earlier all-black rule).
  `sk.force` still draws its label black by default, so pass `text_color=` with the
  arrow's colour.
- Label size: one `FS` per script, 13 or 14 for figures shown at 50 to 70 % width, up to
  16 for a figure shown at full width. Grey notes ("free body diagram", "taut") use
  `FS - 3` in `sk.GREY`.
- Text beside a vertical dimension or a line uses `ha='left'` or `ha='right'`; text
  along a member uses `rotation=angle, rotation_mode='anchor'`.

### Notation in labels

Matplotlib mathtext is not LaTeX:

- `\bm` does not exist and raises `Unknown symbol`. Vectors are `\boldsymbol{F}`, which
  gives the bold italic of the book's `\bm`. Not `\mathbf` (upright) and never
  `\mathbb`.
- Transpose is `^\mathsf{T}`, as in the book, and it renders.
- Write labels as raw strings, `r'$\theta$'`. When labels are written by a script into
  JSON or through a shell, backslashes double and `$0` becomes `/bin/bash`; edit label
  strings with Python or the editor, never with `sed` or a heredoc.
- Two italic letters read as a product: write `R_{Ax}`, not `RAx`.
- Numbers: no `1e+06`, no `-0`, and a true minus sign on negative ticks.
- Use the chapter's symbols: free length `H_0` not `H`, the shear force `T` in the shear
  chapter, `v` and `a` on a velocity and acceleration (not `F`).

## Rules by topic

### The figure must obey the physics in the text

These were real errors, each caught after the figure was drawn:

- **Small deformations.** The truss unit-displacement figures rotated the rods to meet
  the displaced joint. Under the small-deformation assumption each rod keeps its line:
  draw the elongation in red along the rod, then a dotted perpendicular with a
  right-angle mark to the moved joint. A rod with zero elongation is labelled
  `\delta_k = 0`, not left blank.
- **Deformed shapes.** An eccentrically loaded column bows away from the load, so the
  lever arm e + w grows. Check every deformed shape against the load.
- **Sign conventions.** Draw internal forces positive the way the chapter defines them
  and say so in a comment at the call (`# drawn positive (tension)`, `# clockwise, the
  way theta is positive`). Tension arrows leave a cut face; compression arrows point
  into it.
- **Coordinates.** x right, y up, stated in the text. Truss Example 1 once measured y
  downward and only its arrowheads said so.
- **Angles** are measured from the line the text names. A 120° arc was drawn from the
  vertical (90° to 120°) while the text measured from +x; it must run from 0° to 120°.
  Draw a `sk.guide` along the reference line of every angle.
- **Slopes** given as 3-4-5 get a drawn triangle with each leg labelled and a
  right-angle mark, oriented the way the vector actually runs (the first version had its
  corner on the wrong side).
- **The origin** is a dot labelled O with the axes drawn from it, whenever positions are
  measured from it.

### Forces and moments

- `sk.force(ax, p, d, L)` starts at `p`: a pull, or a reaction drawn leaving the body.
  `head=True` ends at `p`: a push, a load arriving on a surface, a contact force.
  `offset=` is measured from the free end, which is the tail when `head=True`.
- Draw the arrow without `text` and place `sk.label` yourself: the default label sits on
  the line of action. Never give a label both in `vec()`/`force()` and in a separate
  call; it then appears twice.
- Loads approach the body from outside and never cross the structure. Decide the
  direction against the body's centroid.
- Forces act where they act: weight at the centre of gravity, a normal force at the
  contact point. Keep kinematic arrows (velocity, acceleration) out of the crowd of
  forces.
- The unknown reactions in a free body diagram point in the positive axis directions,
  and a reaction pair is `R` on one body and `-R` on the other.
- Pressure: fewer, longer, bolder arrows read better (3 pairs at lw 1.5 beat 4 pairs at
  1.3).
- A torque about a shaft seen from the side goes outside the shaft end, as an arc of
  about 290°, split into a back half (zorder below the shaft) and a front half (above),
  so the shaft hides the back. The centre of gravity is a quartered disc.
- Anything sized in points (arrowheads, pin discs, markers, the label outline) does not
  scale with the data. Set the axis limits first, then place such things. MechanicsKit
  0.9.2 had to measure node-disc clearance from the real transform for this reason.

### Dimensions and angles

- `sk.dimension` puts its text on the dimension line, offset by `text_side*2.2*gap`.
  The default `gap=0.02` suits figures a few units across; scale it with the figure
  (`gap=0.1` for a figure 5 units across). When the text still collides, draw the
  dimension with `''` and place `sk.label` beside it.
- `offset=0` gives no extension lines and puts the text on the line. That is right for a
  leader (a radius, `D/2`) and a trap otherwise.
- Put dimensions on the free side of the part: above the beam when a rod hangs below,
  below a shaft rather than across its centre. Spread out crowded dimensions; use an
  L-shaped leader for a small gap.
- `sk.angle` runs anticlockwise from `a0` to `a1`. To measure θ from the downward
  vertical, use `270, 270 + theta`, not `270 - theta, 270`. Make arcs big enough to read:
  past the arrow tails when two forces are close. If a guide line cuts the arc, draw the
  arc without text and put the label in the larger sector yourself.
- `sk.axes` puts its letters on the arrowheads, where they hit other lines. Use
  `labels=('', '')` and place `$x$`, `$y$` with `sk.label`.

### Ground and supports

- `ground` puts the solid on the **right** of the direction you walk the points: a floor
  left to right, a ceiling right to left, a wall with its solid on the left from top to
  bottom, a wall with its solid on the right from bottom to top. A closed channel walked
  anticlockwise has its solid outside. This is the most common single mistake; check
  the fade side on every render.
- A zero-length segment gives a NaN gradient and an artefact in the SVG.
- Use `sketch_extra.ground` (one gradient object) rather than `sk.ground` (strips).
- Use the support symbol the course already uses: the buckling chapter's rolling slider
  is the triangle on two wheels from the lecture notes. Subfigures of one figure use the
  same symbols (if one has no centre pin, none has).
- Keep construction lines clear of support symbols; label lower nodes beside their
  supports, not below them.

### Shading and 3D

- In SVG, `shade`, `shaded_bar`, `shaded_rect`, `ground` and `sphere` are single objects
  with a real `<linearGradient>` or `<radialGradient>`, which `save_svg` writes. With
  `fig.savefig` the gradient is lost and the part comes out in its flat middle colour.
  For PNG frames of an animation set `sketch_extra.SVG_GRADIENTS = False`.
- 3D figures use `Proj(ex, ey, ez)`, the page images of the three unit vectors, never
  matplotlib's `Axes3D`. Projections used in the book: a board
  `Proj(ex=(0.94, -0.34), ey=(0.54, 0.45))`, isometric
  `Proj((0.87, -0.42), (-0.87, -0.42), (0, 1))`. Choose one that matches the source's
  viewpoint.
- Draw far parts first and say in a comment which way the view looks. Set the axis
  limits (or `ax.autoscale_view()`) before `thread3d`, which sizes its strokes from
  them. Face tones by a light vector: `k = 0.72 + 0.38*max(n @ LIGHT, 0)` with
  `LIGHT = (-0.3, -0.6, 0.75)`.
- Label offsets in a `Proj` figure are page offsets; `Proj.axes(..., offsets={k: (dx,
  dy)})` moves one axis letter.

### Drawings by others

Some objects have no mechanical stand-in (a tractor, an aircraft seen from above). Use
only public domain, CC0, CC BY or CC BY-SA drawings, with the licence checked on the
source's own page; a figure containing CC BY-SA material is itself CC BY-SA. Keep the
drawing in `tools/assets/<name>.svg` with `<name>.LICENSE.txt` (Source, Author, Licence,
Retrieved, Modified) and a line in `tools/assets/README.txt`, and remove its caption text
and background rectangle. Paste it after saving:

```python
fig.subplots_adjust(0, 0, 1, 1)
save_svg(fig, out, tight=False)                        # data_to_svg needs the untrimmed page
tl, br = data_to_svg(fig, ax, (x0, y1)), data_to_svg(fig, ax, (x1, y0))
embed_svg(out, 'tools/assets/Tractor_side.svg', corner=tl, size=br - tl, mirror=True)
# or turned, faded and with thicker strokes so it survives being small:
embed_svg_at(out, src, centre, width_pts, angle=30, opacity=0.35, stroke=2.2)
```

Take the height from the source's `viewBox`, not a guessed ratio.

## Checking a render

On every render, look for each of these:

1. Does every label clear every line, arrowhead, hatch and other label? Zoom into the
   crowded spots.
2. Does every ground fade lie on the solid side?
3. Does every arrow point the way the text and the sign convention say, start or end at
   the right point, and come from outside the body?
4. Is every angle measured from the line the text names, with a guide along it, and big
   enough to read?
5. Are the dimensions on the free side, with their text off the lines?
6. Do parts meet that should meet, and are the proportions those of the problem?
7. Is the deformed or moved shape physically right?
8. Does every symbol the text uses appear, in the book's notation, and nothing the text
   does not use?
9. Is anything clipped at the figure edge? Moment arcs that overrun need `clip_on=False`,
   or wider limits.
10. Side by side with the original: is the same information there?

## The hand-edit round trip

Sometimes it is faster to drag a label in Inkscape than to guess an offset. The script
stays the source of truth:

1. The figure was saved with `save_svg`, so every part is a group named after what it
   is: `label-A` for the label $A$, `label-F_s` for $F_s$, `fancyarrow-03` for the third
   arrow, with an `a<k>-` prefix in figures with several axes.
2. Open `fig.svg` in Inkscape, move, recolour, delete or draw, and save as
   `fig.edited.svg` next to it. Do not resize the page: that shows up as a move of every
   part.
3. Run `python -m mechanicskit.svg_roundtrip fig.svg`. It lists each part that moved (in
   data units, with its new centre), was resized, restyled or deleted, and each part
   drawn new.
4. Change the script to match (an offset, a hand-placed coordinate with a comment
   `# as placed by hand`, a reversed ground walk for a flipped surface), regenerate, and
   rerun until it says `no changes found`. Delete the `.edited.svg`.

Read an edit for its intent: a white rectangle drawn under a dimension label means "give
this label a background". A uniform scale of every part of one body means the axis
limits differ, not that the body grew. Whitespace differences in `stroke-dasharray`
report as false restyles.

## In the mechanics book

- **Where things go.** The script is `tools/<chapter>_figures.py`, the SVGs go to
  `<Part>/graphics/`, run from the book root with
  `env -u PYTHONPATH .venv/bin/python tools/<chapter>_figures.py [name ...]`. After a
  redraw, `git rm` the old image and grep the notebooks for its old file name.
- **Every worked example is numbered** ("Example 3: ...") and its problem statement has
  a figure showing the geometry, supports, loads and given values. The order in an
  example is figure, equations, code, then a plot that verifies.
- **A course lab is never a book example** (Vedklyven, Trissorna, Bygeln, 3D-skivan).
- **Embedding a saved SVG**, in a markdown cell:
  `![The collar slides on the guide from $1$ to $2$. The angle $\theta$ is measured from the horizontal through $O$.](graphics/WorkEnergy_collar.svg){#fig-we-collar fig-align="center" width=60%}`.
  Widths of 40 to 70 % are usual. Every figure has a `#fig-` label and is cited in the
  prose as `@fig-...`, and the caption says what is drawn and how the angles are
  measured, in the book's notation.
- **A figure made in a notebook code cell** loses `fig-cap` in Quarto. Wrap it: a
  markdown cell `::: {#fig-x}`, the code cell, then a markdown cell with the caption and
  `:::`.
- **Side by side** only when both stay readable; a two-panel figure whose right panel is
  too small is split onto two rows. Use `::: {#fig-x layout-ncol=2}` with `{#fig-x-a}`
  children for real subfigures.
- **Credits** for drawings by others go in `ImageCredits.qmd`, never in the caption.
- **Dark mode**: the theme gives every SVG a white backing. `save_svg` writes a white
  figure background; transparent line art from elsewhere takes the `.white-bg` class.
- **Code cells** that draw are folded (`#| code-fold: true`), and a plot labels the
  minimum and maximum of every axis and colour bar.

## Adding a part to the library

A figure script only composes library calls. When a figure needs a part, a symbol, a
support or a geometry function that `sketch` and `sketch_extra` cannot draw, add it to
MechanicsKit first and then call it: a new parameter when an existing function almost
fits, a new function otherwise. 2D parts and annotation go in `sketch`; anything standing
on the gradient ground, the supports, 3D and SVG output go in `sketch_extra`. Give it a
docstring that says what it draws in mechanics terms, list it in the tables above, add a
CHANGELOG entry and bump the version before pushing. Local functions in a figure script
are only chapter scenes that call the library with the chapter's constants (a `_scene`
shared by a setup and its free body diagram, a `_floor` at the chapter's height).

The book learned this the hard way: by October 2026 its scripts carried about fifteen
helpers copied between them with drifting defaults (a bare arrow in eight scripts, a
unit vector at an angle in eight, three versions of a side-labelled dimension). They were
moved into the library in 0.10.0.
