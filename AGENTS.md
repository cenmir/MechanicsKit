# Notes for AI agents

MechanicsKit is a Python toolkit for mechanics teaching: LaTeX display of NumPy/SymPy
(`la`, `ltx`), 1-based FEM helpers (`Mesh`, `OneArray`), truss drawing, and problem
figures (`mechanicskit.sketch`, `mechanicskit.sketch_extra`).

## Drawing figures

Read `docs/figures.md` before the first figure of a session; `skills/figure-redraw/SKILL.md`
is the same method in short form, and `examples/figures/bar_spring_figures.py` a runnable
example. Save figures as SVG with `sketch_extra.save_svg`, render them to PNG and look at
them before reporting.

Colours carry meaning, and a label takes the colour of the arrow it names:

| Arrow | Colour | Label |
|---|---|---|
| applied force or moment, weight, reaction, contact force | `sk.LOAD` red | red |
| resultant, displacement, velocity, degree of freedom | `sk.BLUE` | blue |
| internal force or moment, rotation arrow | `sk.GREEN` | green |
| unit vector | black, `lw=1.4` | black |

Points, angles, dimensions and axes are labelled in black. `sk.force` draws its label
black unless `text_color=` is given, so pass it:

```python
sk.force(ax, B, direction, 70, r'$\mathbf{F}_S$', text_color=sk.LOAD)
sk.force(ax, O, e_R, 1.2, '$R$', color=sk.BLUE, text_color=sk.BLUE)
```

A drawing helper that a figure needs goes into the library, not into the figure script.

## Releasing

See `CLAUDE.md` for the version policy: bump `pyproject.toml` and
`mechanicskit/__init__.py` together and add a `CHANGELOG.md` entry before pushing.
