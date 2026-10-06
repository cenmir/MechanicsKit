---
name: figure-redraw
description: Redraw a problem figure (a lecture slide, a hand drawing or a photo of a sketch) as a Python script that saves an SVG in the mechanics book's house style, using mechanicskit.sketch and mechanicskit.sketch_extra. Use when asked to recreate, redraw, replace or vectorise a figure, draw a free body diagram or problem figure, make a figure "like the ones in the book", or fix labels, arrows or layout in an existing figure script.
---

# Redrawing a problem figure as SVG

The full method, with every rule and the reason behind it, is `docs/figures.md` in the
MechanicsKit repo. **Read it before the first figure of a session.** Find it in a local
clone (`find ~ -path '*MechanicsKit/docs/figures.md' 2>/dev/null | head -1`), or fetch
<https://raw.githubusercontent.com/cenmir/MechanicsKit/main/docs/figures.md>. A complete,
runnable example is `examples/figures/bar_spring_figures.py` in the same repo.

Needs `mechanicskit` 0.10 or later (`python -c "import mechanicskit as m; print(m.__version__)"`;
install with `pip install git+https://github.com/cenmir/mechanicskit.git`), plus
`rsvg-convert` and ImageMagick to look at the result.

## Steps

1. **Look at the source image** with Read. List what it must carry: bodies and
   supports, every load, every given dimension and angle with the line it is measured
   from, axes and origin, the point labels the problem text uses.
2. **Put the problem's parameters at the top of the script** in drawing units, with the
   scale in a comment, and compute every point from them. Never trace pixel positions.
3. **Write one function per figure** in `<chapter>_figures.py`: `sk.canvas`, the parts,
   explicit `set_xlim`/`set_ylim`, then `save_svg` from `mechanicskit.sketch_extra`
   (never `fig.savefig`). Keep a `FIGURES` dict so that `python script.py name` redraws
   one figure.
4. **Render and look**: `rsvg-convert -z 2 -b white fig.svg -o /tmp/fig.png`, then Read
   the PNG; crop crowded spots with `convert -crop`. Go through the checklist in the
   guide. Expect two to five rounds; never report a figure you have not looked at.
5. **Compare side by side with the original** (`convert a.png b.png +append`): the same
   information, in our style, clearer if anything.
6. Write the caption only after looking at the final render.

## The rules that are broken most often

- `ground(ax, points)` puts the solid on the **right** of the walking direction: floor
  left→right, ceiling right→left, wall with solid on its left top→bottom.
- Labels: move them clear of lines first; the white outline (`halo=True`, the default)
  is the safety net. All labels are black, force labels too; colour belongs to the
  arrows. Draw `sk.force` without `text` and place a black `sk.label` yourself.
- `sk.force` starts at the point; `head=True` ends there (a push, a contact force).
  Reactions in a free body diagram point in the positive axis directions.
- Angles are measured from the line the text names, with a `sk.guide` along it.
  `sk.angle` runs anticlockwise from `a0` to `a1`.
- `sk.dimension` puts its value on a white box near the line; `text_offset` moves it
  beside the line, `offset=0` drops the extension lines. `sk.axes` letters land on the
  arrowheads: pass `labels=('', '')` and label by hand.
- Mathtext has no `\bm`: vectors are `\boldsymbol{F}`, transpose `^\mathsf{T}`. Use raw
  strings, and never edit label strings with `sed` or a shell heredoc.
- The figure must obey the physics the text states: small deformations keep rods on
  their lines, columns bow away from the load, sign conventions match the chapter.
- Palette: `sk.LOAD` red forces, `sk.BLUE` displacements and coordinates, `sk.GREEN`
  internal forces and rotations, `sk.GREY` construction, `sk.BODY` tan bodies,
  `sk.STEEL` machine parts, white background.

## New parts go into MechanicsKit

Never write a drawing helper in the figure script. If `sk`/`sketch_extra` cannot draw
what the figure needs, add it to MechanicsKit (a parameter or a new function, with a
CHANGELOG entry) and call it; see "Adding a part to the library" in the guide.

## Hand edits

If the user edited the SVG in Inkscape and saved `fig.edited.svg`, run
`python -m mechanicskit.svg_roundtrip fig.svg`, carry each reported change into the
script, regenerate, and repeat until it reports `no changes found`.
