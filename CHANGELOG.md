# Changelog

All notable changes to MechanicsKit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## Version Update Guidelines

While in the 0.x series we deliberately move the minor slot slowly so the
version number stays meaningful.

**When to bump versions (pre-1.0 policy):**
- **PATCH (0.X.Y → 0.X.Y+1)**: Bug fixes, documentation, performance, and
  small additive refinements to existing functions (new kwargs, new optional
  behavior). Most changes land here.
- **MINOR (0.X.0 → 0.X+1.0)**: New public functions, new user-visible verbs,
  or meaningful reshuffles of existing ones.
- **MAJOR (X.0.0)**: Breaking changes that require user code updates.
  Reserved for the 1.0 API commitment.

**Rule of thumb:** if the change only adds an option to something that
already exists, it's a patch. If it adds a thing the user can import or
call that didn't exist before, it's a minor bump.

**When to update `__version__` in `__init__.py`:**
- Bump before `git push` (not on every commit) so each pushed state maps
  to a distinct, installable version.
- Keep `pyproject.toml` and `mechanicskit/__init__.py` in lockstep.

A ``pre-push`` git hook prints this policy as a reminder at push time;
see the top of the changelog entry for the change being pushed and
decide patch vs. minor before pushing.

---

## [0.25.0] - 2026-10-18

### Added
- `sketch.bar_outline(steps, ends, waves, amplitude, axis_y)`: the side-view outline of a
  straight or stepped bar, each end `'flat'`, `'round'` or `'cut'` (a break wave), for a
  part or a piece of one in a free body diagram. The normal stress animations of the book
  draw their pieces with it too.
- `sketch_extra.face_stress(ax, x, r, side, N, sigma)`: the normal stress on a cut face of
  a bar seen from the side, thin arrows over the face and the resultant on the axis, drawn
  positive (out of the face), green.
- `sketch_extra.i_beam`, `clevis` with `CLEVIS_STOPS`, `clevis_rod` and `line_load`: a
  rolled beam from the side, a rod-end fork, a tie rod with a fork at each end, and a
  distributed load with uniform or linear intensity.

### Changed
- `sketch_extra.stepped_shaft(..., ends=)`: the ends of the shaft take the kinds of
  `bar_outline`, so a cut piece of a stepped bar keeps one shaded, outlined cylinder per
  step and with it the edge at every change of diameter.
- `sketch_extra.pedestal(..., angle=)`: turns the pedestal about its pin; `angle=180` hangs
  it from a ceiling.

## [0.24.1] - 2026-10-17

### Fixed
- `scene3d`: the frame of a render grows to hold every annotation, so a force arrow, a
  dimension or a dashed line that reaches past the solids (a free body diagram with the
  supports removed) is no longer clipped.
- `scene3d`: a `wire` with one straight segment kept only its end ball and lost the rod; the
  pieces are now joined with a fuzzy fuse.

## [0.24.0] - 2026-10-17

### Added
- `scene3d`: a `step` part, `{"kind": "step", "args": {"path": "..."}}`, that draws the solids
  of a STEP file in their own coordinates, so a part modelled in CAD can be shown with the
  annotations of a scene. The `line` annotation takes an optional `color`.
- `sketch.ring_angle(N_sun, N_planet, N_ring, sun_angle, psi)`: the angle that meshes
  `ring_gear` with planets placed by `planet_angle`.
- `sketch.tyre_top(ax, centre, D, w, turn, angle)`: a wheel seen from above, a black tread
  with a white stripe across it that runs over the top as the wheel turns, for top-view
  animations of wheels at different speeds (the differential of a car in a corner).
- `sketch.break_points(p0, p1, waves, amplitude)`: the points of a wavy break line, so that a
  part cut off in a drawing (a plate, a shank, a thread section) can be filled up to the same
  curve that `break_line` draws on it. `break_line` now uses it; its output is unchanged.
- `sketch.dimension(..., outside=L)`: a short span drawn the drafting way, the dimension line
  without heads and two arrows of length `L` coming in from outside (a crest flat P/8, a slice
  dx).

- `sketch_extra.bearing_section`, `bearing_geometry`, `bearing_end_view`: rolling and plain
  bearings in section for bearing arrangement drawings (twelve kinds, load lines returned so
  that a figure can carry them to the axis, `angle` and `tilt` for vertical and misaligned
  shafts), with `stepped_shaft`, `section` (a hatched cut face) and `retaining_ring` for the
  shaft and housing around them. Palette constants `BALL_STOPS`, `ROLLER_STOPS`, `BRONZE`,
  `SECTION`.

### Fixed
- `sketch.planet_angle` claimed that its planets also mesh with `ring_gear` at
  `angle=0`. They do not in general (the 15/15/45 set with the sun at 8 degrees had every
  planet tooth on a ring tooth); the ring needs `ring_angle`.

## [0.23.0] - 2026-10-15

### Added
- `mechanicskit.friction`: coefficients of friction with a source per table. Material
  pairs static and sliding, dry and lubricated (Marks 8th ed. via NASA RP-1228 Table IV),
  the VDI 2230 bolt friction classes A to E, brake and clutch linings (Shigley Table 16-3)
  and interference-fit slip coefficients (DIN 7190). Lookups `pair`, `bolt_class`,
  `classify_bolt` and `pressed_fit`, and `friction.table(name)`, which returns a table as
  Markdown for Jupyter, Quarto and marimo.
- `friction.materials()`, `friction.partners(material)` and `friction.SOURCES_SHORT`, for
  pickers such as the FrictionPairs notebook of the mechanics book.
- `friction.PAIRS` keeps only the pairs with both dry coefficients, static and sliding.
- `sketch.spring_scale`: a spring balance (force gauge) at any angle, for measurement
  setups such as the pull test of the Friction chapter.

## [0.22.0] - 2026-10-13

### Added
- `scene3d` primitives `motor`, `table` (plate with a slot) and `half_disc` (miter gauge);
  annotation kinds `arc` (rotation arrow in any plane) and `line` (centre and extension
  lines); the check `bbox_expect` (a part's bounding box against parameter expressions).
- `scene3d.show`, `watch` and `export_step`, and the CLI options `--show`, `--watch`
  (OCP CAD Viewer, live redraw on save) and `--step FILE`.
- `scene3d.viewer_camera`, `render_view` and the CLI option `--from-viewer`: render the
  figure from the CAD viewer's current view and print its az/el for the spec.
- Console command `scene3d` (same as `python -m mechanicskit.scene3d`); the spec argument
  is optional and defaults to the last spec used.
- `scene3d --setup-vscode [FOLDER]` (`scene3d.setup_vscode`): adds three tasks on the open
  spec (watch in viewer, render this view, render all cameras) to a VS Code workspace, puts
  them on the status bar as buttons, and installs the OCP CAD Viewer and Task Buttons
  extensions when the `code` command is available.
- `scene3d` spec `derived` section: named values and vectors computed in order from the
  parameters, so a spec carries the problem's kinematics; expressions gain trig, `vec`,
  `cross`, `norm`, `linspace`, list comprehensions, and evaluation inside dicts.
- `scene3d` primitives `plate` (rounded corners, holes, arc slots), `bar_link`, `wire` and
  `knob`; annotations `point`, `cog`, `triad`, and the `angle` role for `arc`.
- `scene3d` check `labels_far_from_anchor_pt`: a label more than 14 pt from the point or
  arc it names.
- `scene3d.spec_path` / `load_spec`: a path into a render folder (report.json, a candidate
  image) resolves to the spec beside it; anything else gives a one-line error.
- Examples `disk_sander`, `motor_spring` (Fusion 360 model), `tv_mount` and `wall_shelf`
  (photographs), `hinged_plate` (hand sketch), and a gallery in `docs/scene3d.qmd`.

### Fixed
- The occlusion check skipped every sample within 3 mm of a solid, so arrows drawn on a
  face were never checked; it now skips only samples strictly inside a solid. `line`
  annotations are checked too, except dash-dot centre lines.

## [0.21.0] - 2026-10-13

### Added
- `mechanicskit.scene3d`: 3D problem figures from a JSON scene spec. Parts are
  exact build123d (OpenCascade) solids, edges come from OpenCascade hidden-line
  removal, shading is Lambert bands traced from a z-buffer into simplified vector
  fills, annotations are 3D objects projected with the same orthographic camera
  and drawn with `sketch`. Every candidate camera is rendered and scored by
  ray-cast occlusion of the annotations, label overlaps and labels on edges;
  physics checks cover rope tangency and load placement. CLI:
  `python -m mechanicskit.scene3d spec.json`. New optional extra
  `mechanicskit[cad]` (build123d); `scene3d` is not imported by `mechanicskit`
  itself, so the base install stays light.
- `examples/scene3d/winch.json`: a hand winch as a worked spec.
- `docs/scene3d.qmd` (PDF): how the pipeline works and how to use it.

### Changed
- README installation section: `uv pip` commands, a clone-and-install recipe, and the
  optional `[cad]` extra explained.

## [0.20.0] - 2026-10-13

### Added

- `sketch.ring_gear(ax, centre, N, module, angle, rim)`: an internal gear, the complement of
  an external involute outline with addendum and dedendum swapped; a space is centred on +y.
- `sketch.planet_angle(N_sun, N_planet, psi, sun_angle)`: the angle to give `gear` for a
  planet at the polar angle psi so that it meshes with the sun (and the ring, when the
  planets satisfy the assembly condition).
- `sketch.gear` takes `warp`, a radial displacement dr(theta) of the outline (the bent
  flexspline of a strain wave gear), and `hub_ratio=0` leaves out the hub;
  `sketch.gear_outline` takes the addendum and dedendum `ha`, `hf`.

### Fixed

- `gear_outline` and `gear` documented a tooth on +x; it has always been on +y. The
  docstrings now say so (the geometry is unchanged, so existing figures keep their angles).

- `sketch_extra.bevel_section(ax, apex, axis, delta_deg, A0, b, m)`: the axial section of a
  straight bevel gear for the bevel gear chapter. The tooth zone lies between the back and
  front cones, normal to the pitch generator, with addendum and dedendum tapering to the
  apex; the body under it runs to a bore. Returns the pitch, tip and root points at both
  ends and the unit vectors along and normal to the pitch cone, for annotation.
- `sketch_extra.out_of_plane(ax, p, r, into=True, color)`: a force normal to the page, a
  circle with a cross (into the page) or a dot (out of it), in the force's colour.
- `sketch_extra.circle_tangents(centre, r, point)`: the two points where lines from a
  point touch a circle, left and right as seen from the point. Draws the free ends of a
  brake band or a belt running from a drum to a lever or an anchor (brakes chapter).

## [0.19.0] - 2026-10-12

### Added

- `sketch_extra.hex_nut(ax, face, direction, e, m)`: a hex nut or bolt head seen from the
  side, across the corners, drawn the way Mirza draws them: shaded from dark blue on top
  (`NUT_STOPS`), the edges between the three visible faces at +-e/4, and the 30 degree
  chamfer of the outer end as straight lines at the outline's corners and an arc where the
  chamfer cone cuts each flat face.

## [0.18.1] - 2026-10-12

### Changed

- `sketch.leader` takes `stop`: the head ends that far short of the point, so a leader
  that points at a thin line (a reference line, a dashed line) does not cover it.

## [0.18.0] - 2026-10-11

### Added

- `breadboard`: breadboard circuits drawn the way Tinkercad Circuits shows them, for
  readers who copy a circuit hole by hole. `canvas(xlim, ylim, unit_pt)` fixes one unit
  to the 0.1 inch pitch; `Breadboard(ax, 'half' | 'full')` draws the board (rows a to j,
  centre gap, rails `'T+'`, `'T-'`, `'B+'`, `'B-'`) with `hole('a12')`, `rail('T+', 12)`,
  `xy(...)`, `strip(...)` and `highlight_strip(...)`, and places parts with their legs in
  named holes: `led` (5 mm, flat on the cathode side, kinked anode leg, `lit` glow),
  `resistor` (four bands from `resistor_bands(ohms, tol)`), `wire` (straight, `'hv'`,
  `'vh'`, `'arc'`), `esp32` (a 30-pin DOIT ESP32 DevKit V1 straddling the gap). Off the
  board: `lead`, `cable`, `lab_supply` (BASETech BT-305 style bench supply), `battery_9v`,
  `usb_breakout`, `multimeter` (display, rotary dial, jacks) and `probe`, and `callout`.
  Stand-alone figures `esp32_pinout()` (pins by class with a legend) and `led_pinout()`.

## [0.17.0] - 2026-10-11

### Added

- `gears`: the load capacity of spur gears, the two lines of a load diagram
  (belastningsdiagram): `flank_torque` (Hertz pressure at the pitch point, with
  `elasticity_factor` Z_E and `zone_factor` Z_H as in ISO 6336-2), `root_torque` (Lewis
  with a fillet stress concentration), `load_limit` (the smaller, and which governs),
  `lewis_form_factor` from Shigley's Table 14-2, and `iso6336_normalised` for the
  allowable stresses of normalised steel by hardness (ISO 6336-5).

## [0.16.2] - 2026-10-11

### Changed

- `fits.PROCESSES` says which features each process makes, and `fits.processes(grade,
  feature)` takes 'hole' or 'shaft': a reamer makes no shafts.

## [0.16.1] - 2026-10-11

### Fixed

- `fits.tolerance` covers IT4 and IT14, the ends of the ranges in `fits.PROCESSES`.

## [0.16.0] - 2026-10-11

### Added

- `fits.PROCESSES` and `fits.processes(grade)`: the machining processes and the tolerance
  grades they reach in normal production (a workshop rule of thumb, printing included),
  and the processes that can make a given grade, cheapest first. A tolerance is only worth
  asking for if the workshop can make it.

## [0.15.0] - 2026-10-11

### Added

- `mechanicskit.fits`: ISO 286 limits and fits on the hole basis. Tolerance grades IT5 to
  IT13 and the fundamental deviations of the shaft letters c to u for sizes up to 400 mm;
  `limits('H7/s6', 60)`, `interference(fit, d)`, `kind(fit, d)` and the list
  `PREFERRED_FITS` with the character and typical uses of each fit. Used by the shaft-hub
  chapter of the book and its ISO fits notebook.
- `sketch.annulus(ax, centre, r_in, r_out, a0, a1, hatch)`: a ring or ring segment in
  section, for hubs, tubes and bearing rings.
- `sketch.radial_load(ax, centre, r, angles, length, side)`: a pressure on a circular
  surface as arrows ending on it, from outside or from inside, with one length per angle
  for a distribution.
- `sketch.spline_outline(centre, r_major, r_minor, n, width)` and
  `sketch.polygon_outline(centre, r, e, lobes)`: a straight-sided spline shaft and a
  polygon (P3G) shaft in section, as outlines.

## [0.14.0] - 2026-10-10

### Added

- `sketch.thread_profile(x0, x1, P, r_minor, r_major)`: the ISO 68-1 basic thread
  profile in an axial section, as a polyline, for bolts, nuts and thread details.
- `sketch.helix(ax, xc, r, y0, y1, lead, back=False)`: a helix on a cylinder seen
  from the side, front half solid and back half dashed, for screws and threads.

## [0.13.0] - 2026-10-08

### Added

- `sketch.right_angle(ax, corner, d1, d2, size)`: the square that marks a right
  angle between two directions, as at the foot of a perpendicular.

## [0.12.0] - 2026-10-08

### Added

- `sketch.helix_centres(d, p, n_active, n_closed)` and
  `sketch.helical_spring(ax, x0, D, d, z)`: a helical compression spring in side view,
  wire sections and the front half of each turn, with closed end coils. Moved here from
  the book's springs figure script; also used by the SpringCheck and SpringMeasure
  notebooks.
- `sketch.moment_vector(ax, point, direction, length, text)`: a moment or torque as a
  double-headed vector (right-hand rule), replacing two copies in the book's figure scripts.
- `sketch_extra.vec(..., text_color=)` and `Proj.force(..., text_color=)`, so that a
  label can take the colour of its arrow.

### Changed

- The figure guide, the figure-redraw skill, `CLAUDE.md` and a new `AGENTS.md` state the
  colour rule: a label takes the colour of the arrow it names.

## [0.11.0] - 2026-10-08

### Added

- `sketch.gas_spring(ax, p0, p1, width, tube, rod, eye)`: a gas spring or
  hydraulic cylinder, tube end at `p0` and rod end at `p1`, with a mounting eye
  centred on each pin. Drawn for the toy-box lid example of the book's Springs
  chapter.

### Changed

- README: the capability list names the truss and figure tools, and the update
  history covers 0.8.0 to 0.11.0.

## [0.10.0] - 2026-10-06

### Added

- `mechanicskit.sketch_extra`, moved here from the mechanics book's
  `tools/` so that anyone can draw figures the way the book does:
  `save_svg` (named SVG parts, real gradients and a data-to-SVG map for hand
  edits), the gradient fills `shade`, `shaded_bar`, `shaded_rect` and a
  fading `ground`, the shading stops (`STEEL_STOPS`, `TAN_STOPS`, ...), the
  axonometric `Proj` with the 3D solids `vcylinder`, `rod3d`, `sphere`,
  `box3d`, `thread3d` and the `speckle` texture, `embed_svg` and
  `embed_svg_at` for licensed drawings, and the small helpers `vec`, `dot`,
  `circled` and `HALO`.
- `mechanicskit.svg_roundtrip`: `python -m mechanicskit.svg_roundtrip
  fig.svg` lists what was moved, restyled, deleted or drawn in
  `fig.edited.svg`, in the figure's own data units.
- `docs/figures.md`, the method for redrawing a problem figure from a lecture
  slide or a hand drawing, with the rules the book learned the hard way, and
  `skills/figure-redraw/SKILL.md`, the same method as a Claude Code skill.
- `examples/figures/bar_spring_figures.py`, a worked setup figure and free
  body diagram.
- The drawing helpers the book's figure scripts had copied between them, now in
  `sketch`: `polar`, `rot`, `arc`, `ccw`, `mirror_outline`, `outward`, `arrow`,
  `unit_vectors`, `triad`, `curl` (a torque about a shaft), `radius`, `leader`,
  `centreline`, `dashpot`, `box`, `rounded_rect`, `cog`; and from FBD Lab `coil`,
  `trapezoid`, `ellipse`, `direction_line`, `break_line`. In `sketch_extra`: the
  supports `support_pin`, `support_roller`, `support_rollers`, `support_slider`,
  `pedestal`, `wall`, `hatched_wall`, `sliding_block`, `bracket`, plus
  `Proj.from_view`, `Proj.arc` and `tone`.
- `sketch.dimension` takes `text_offset`, `text_at` and `text_shift` to place its
  value beside the line, and `offset=0` marks a span without extension lines.
- `sketch.LABEL_HALO` turns the white label outline off for a whole script.

### Changed

- `sketch.force` labels are black (`text_color=`); colour belongs to the arrow.
  `sketch_extra.vec` labels likewise.
- `sketch.dimension` puts a white box behind its value instead of the outline
  (`bg=None` for the outline), and any label given `bg=` drops the outline.

---

## [0.9.4] - 2026-10-04

### Changed

- `sketch.label` draws its white outline (`halo=True`) by default, and so do
  the labels that `force`, `angle`, `dimension` and `axes` draw. Labels that
  have to cross a line or a tinted fill now read without each call asking for
  it. `halo=False` gives the plain text that 0.9.3 made the default; `bg=` and
  a `bbox` passed through work as before.

## [0.9.3] - 2026-09-25

### Changed

- `sketch.label` draws plain text by default. It used to put a white box
  behind every label, which hid lines that a better placed label would not
  have touched. A label that has to sit on a line now asks for its backing:
  `bg="white"` (any colour) with `bgalpha` for the box, or `halo=True` for a
  thin white outline around the glyphs, which hides less and reads over a
  tinted fill too. The labels that `force`, `angle`, `dimension` and `axes`
  draw follow the same default. A `bbox` passed through still overrides both.

### Fixed

- `ltx(..., precision=n)` now means n significant figures for every kind of
  number. A plain Python float or a NumPy float used to be printed with n
  decimal places while a SymPy number got n significant figures, so
  `ltx(345/1.2, precision=4)` showed `287.5000` next to a SymPy result shown
  as `4.689`. Floats are now rounded and printed exactly as SymPy prints the
  same number, `287.5`. Without a precision, floats still show two decimals.

## [0.9.2] - 2026-09-22

### Fixed

- `draw_truss`: a load arrow that points into a node stops clear of the node
  disc, whatever the size of the figure. The discs are sized in points, and
  the old fixed clearance in data units hid the arrowhead behind the disc on
  small figures. The arrows are now drawn once the limits are known, so the
  clearance is read off the real transform.

## [0.9.1] - 2026-09-22

### Added

- `array | OneArray` as a second way to build a `OneArray`, so that
  `f = np.zeros(ndofs) | OneArray` reads left to right instead of nesting a
  call inside a call. The whole array is wrapped, not each element: NumPy is
  told to decline the operator so that Python hands the array over intact.

## [0.9.0] - 2026-09-21

### Added

- `mechanicskit.sketch` — the parts a problem figure is made of, drawn in the
  palette `draw_truss` uses so that every figure in a book looks like the
  same hand drew it: `ground` (a fixed surface whose solid side fades away),
  `body`, `link`, `pin`, `spring`, `gear` and `hub`, `force`, `angle`,
  `dimension`, `axes`, `guide`, `rotation` and `label`, plus `canvas` for a
  frameless equal-aspect axes, `gear_outline` for the bare involute
  polygon, and `unit` and `normal` for placing labels along and beside an arrow. Every function takes `ax` first and points as `(x, y)` pairs; a
  SymPy column vector is accepted and cut down to its first two entries.

## [0.8.1] - 2026-09-21

### Added

- `draw_truss(..., support_angles={node: degrees})` — turns the support symbol
  at a given node. The default hangs every symbol below its node, which is
  right for a support standing on the ground; a node pinned to a wall on its
  left wants `-90`. The angle passes through `animate_truss` unchanged.

## [0.8.0] - 2026-09-20

### Added

- `draw_truss(nodes, elements, presc, loads, ...)` — draws a plane truss with
  numbered nodes and elements, support symbols and load arrows. Everything is
  1-based: node `i` owns degrees of freedom `2i-1` and `2i`, and `presc` is a
  list of those numbers. A node with both degrees of freedom held is drawn as a
  pin, a node with one held as a roller turned so its wheels run along the
  direction still free.
- `pin_support(ax, xy, h, angle)` and `roller_support(ax, xy, h, angle)` — the
  symbols on their own, for anyone drawing their own figure. The geometry
  follows FBD Lab: a rounded-top triangle with a pin hole on a base bar that
  fades out.
- `animate_truss(nodes, elements, displacements, ...)` — grows the deformation
  from nothing up to `scale_max` over `frames` frames, with the undeformed
  shape left out so the truss is seen moving on its own. The axes are held
  still for the whole run. Returns a `FuncAnimation`, so a notebook shows it
  with `HTML(anim.to_jshtml())`, and the title reports the scale of each frame.

Node discs are sized in points rather than in model units, so the number always
fits inside its disc whatever the truss is measured in.

Load arrows are placed on the outward side of the joint, so a deck load on a
bottom chord hangs below the joint instead of crossing the truss to reach it.
Support symbols are drawn on the deformed shape, so a joint given a prescribed
displacement carries its roller along with it. `display_undeformed=False` drops
the dashed ghost, and `colorbar=False` suppresses the colourbar that goes with
`values`.

## [0.7.6] - 2026-09-19

### Added
- **`OneArray` now indexes in two dimensions**, so a global stiffness matrix
  can be assembled in the notation the mathematics uses:

  ```python
  K = OneArray(np.zeros((3, 3)))
  for e, (i, j) in enumerate(elements, start=1):
      K[[i, j], [i, j]] += k[e] * k_rod     # node numbers, not node numbers minus one
  ```

  A tuple of index lists means the **submatrix** of those rows and columns,
  as `K([i,j],[i,j])` does in a textbook and in MATLAB. This deliberately
  differs from NumPy, where `A[[1,2],[1,2]]` picks elements pairwise; the
  submatrix reading is the one the assembly formula intends. Scalars, index
  lists and a bare `:` may be mixed, so `K[2, 3]`, `K[2, :]` and
  `K[[1,2], [1,2]]` all work.
- **`__matmul__`**, so `f = K @ u` works directly on 1-based arrays instead of
  forcing a detour through `.data`. A dot product `u @ v` returns a plain
  number.
- Slices other than `:` now raise an `IndexError` naming the problem, because
  a 1-based slice is ambiguous about whether its end point is included.
  Previously they raised a confusing `TypeError` from the index validator.

### Fixed
- **Iterating a `OneArray` silently produced nothing.** With no `__iter__`,
  Python fell back to the legacy sequence protocol, which calls
  `__getitem__(0)`; a 1-based array rejects index 0 with `IndexError`, and
  Python reads that as "the sequence ended". So `list(OneArray([1, 2, 3]))`
  returned `[]` and every `for row in one_array:` loop was a silent no-op.
  `__iter__` now yields the values, so
  `for e, (i, j) in enumerate(elements, start=1)` reads element numbers and
  node numbers alike in 1-based form.
- `np.asarray(one_array)` returned an empty array for the same reason; it now
  returns the values.
- **`fplot` with a symbol left unassigned** (`fplot(a*x**2)`) picked `x` as the
  parameter and then failed inside matplotlib with "Cannot convert expression
  to float". It now raises a `ValueError` that names the symbol without a
  value. Two tests in `test_fplot.py` that expected impossible behaviour were
  corrected, and `test_labeled_alias` now checks the `labeled = ltx` alias
  that has been in place since 0.6.1.

### Changed
- **`K[i, j]` on a two-dimensional `OneArray` now returns the element.**
  Before, the pair was read as a list of row numbers, so `K[2, 1]` silently
  returned rows 2 and 1. Code that relied on that should write `K[[2, 1]]`.

### Unchanged
- One-dimensional behaviour is exactly as before, and a single index still
  applies to the first axis, so a nodal field of shape `(n_nodes, n_comp)`
  still answers `U[2]` with node 2's row.

## [0.7.5] - 2026-09-18

### Fixed
- **Import failure on matplotlib 3.9 and later.** `matplotlib.cm.get_cmap`
  was removed in 3.9, and `patch.py` imported it at module level, so
  `import mechanicskit` raised `ImportError` — the whole package was
  unusable, not just `patch()`. `patch.py` now looks colormaps up through
  `matplotlib.colormaps` (available since 3.6) behind a small `get_cmap`
  helper that still passes a `Colormap` object through unchanged, and
  `colormap_utils.build_colorbar` does the same. Seen on matplotlib
  3.11.2 with Python 3.14.

  The test suite could not be collected before this fix, since every test
  module imports the package.

## [0.7.4] - 2026-05-16

### Added
- **`label` option on `la` / `ltx` / `labeled`.** Attaches a Quarto
  cross-reference label to a rendered equation so prose can refer to
  it with `@label`. Implementation: when `label="eq-step"` is set,
  `LatexExpression._repr_markdown_` returns
  `$$ ... $$ {#eq-step}` (instead of `None`), and Quarto picks up the
  `text/markdown` representation over `text/latex`. With no `label`
  argument, `_repr_markdown_` returns `None` and Jupyter / Marimo /
  Quarto fall back to the existing `_repr_latex_` bare `$$ ... $$`
  form, so the change is opt-in and does not affect unlabelled cells.
  Usage:
  ```python
  ltx(r"y(t) =", y_sol, label="eq-step-symbolic")
  ```
  Prose can then write `@eq-step-symbolic` to cross-reference the
  rendered equation by its auto-assigned Quarto number.

## [0.7.3] - 2026-05-12

### Changed
- **`display_labeled_latex_examples.ipynb` rewritten and fixed for
  portable MathJax rendering.** Previous outputs (shipped in v0.7.2)
  rendered as raw LaTeX on GitHub because:
  - Single-row labels used `&=`, which is only valid inside an
    `aligned` environment. Replaced with plain `=`; `&=` is now used
    only in the `aligned=True` example.
  - The "chainable verbs" example piped twice (`A | la.foo() | la.bar()`),
    which re-wraps a `LatexArray` and falls through to `str()`. Replaced
    with `A | la.arraystretch(2.0).shape()` — chain on the renderer.
  - The `aligned=True` example used `\bm`, which is not loaded by
    MathJax in many viewers (including GitHub). Replaced with
    `\boldsymbol`.
- Added a new section demonstrating the `wrap=True` / `wrap=N` option
  introduced in 0.7.1, mirroring the demo notebook at
  `mechanics/test.ipynb`.

## [0.7.2] - 2026-05-12

### Changed
- Re-executed `examples/notebooks/display_labeled_latex_examples.ipynb`
  so the rendered outputs are present in the file. The previous v0.7.0
  commit shipped the rewritten notebook without running it, so GitHub
  showed empty output cells.

## [0.7.1] - 2026-05-12

### Added
- **`wrap` option on `la` / `ltx`.** Breaks long SymPy `Add` expressions
  across multiple aligned lines so they fit within page width in both
  HTML (MathJax) and PDF (LaTeX).
  - `la(expr, wrap=True)` — one summand per line.
  - `la(expr, wrap=N)` — pack `N` summands per line.
  - Pipe form: `expr | la.wrap()` or `expr | la.wrap(N)`.
  - `ltx(r"f &=", expr, wrap=True)` honors the user-supplied label and
    appends continuation lines as `&\quad + ...`.
  - For `sp.Eq(lhs, rhs)`, the RHS is broken and continuation lines align
    after the `=` sign.
  - Short expressions whose number of summands does not exceed the
    per-line count are left untouched, so opting in is safe.

## [0.7.0] - 2026-04-26

### Added
- **`mk.md(template)` — type-aware markdown formatter.** Interpolates
  variables from the calling cell's namespace (no `f` prefix) and
  dispatches on type before producing the final markdown source:
  - `LatexArray`, `LatexExpression`, and any object with `_repr_latex_`
    pass through their LaTeX (already `$$…$$`-wrapped).
  - 2-D `ndarray` ≤ 8×8 → markdown table.
  - 2-D `ndarray` > 8×8 → truncated LaTeX `bmatrix` (matches `LatexArray`).
  - 1-D `ndarray` → comma-separated inline.
  - `pandas.DataFrame` (lazy registration) → `df.to_markdown()`.
  - `list[dict]` / `list[list]` → markdown table.
  - Other types fall through to `str(value)` (matches f-string behavior).
- Output is **markdown source**, not pre-rendered HTML. The returned
  `Markdown` object exposes `_repr_markdown_` (Jupyter, quarto),
  `_mime_("text/markdown", …)` (marimo), and a `__format__` that returns
  markdown source so nested `mk.md` calls compose under f-string
  interpolation.
- Arithmetic and method calls work inside `{...}` (e.g. `{F_st/1000:.2f}`,
  `{obj.method().attr}`) — `mk.md` evaluates the placeholder text as a
  Python expression in the caller's scope.
- Standard Python format specs honored (`{x:.2f}`, `{x:>10}`).
- Variables can also be passed as keyword arguments; kwargs override the
  caller-frame lookup.

### Changed
- New hard dependency: `tabulate>=0.9` (used by the table-rendering paths
  for ndarray, `list[list]`, `list[dict]`, and DataFrames).

## [0.6.2] - 2026-05-03

### Added
- `to_responsive_html(..., default_mode='reflect'|'once'|'loop')` exposes
  the JS player's mode dropdown so callers can preselect ping-pong
  ('reflect') or single-shot ('once') playback. Default unchanged ('loop').

## [0.6.1] - 2026-04-20

### Added
- `ltx(..., aligned=True)` wraps the composed expression in
  `\begin{aligned}...\end{aligned}` for left-aligned multi-row equations.
  Use `&` to mark the alignment column and `\\` to separate rows.
- `la.shape()` chainable verb — `A | la.shape()` appends an `_{m \times n}`
  subscript to the rendered matrix/vector. Sibling of the existing
  `show_shape=True` kwarg.
- `show_shape` is now honored by SymPy rendering and by the 1-D and 2-D
  NumPy branches (previously plumbed through but not applied in all
  branches).

### Changed
- `LatexExpression` (`ltx`) `precision` default changed from `2` to
  `None` (full precision). When set, `precision=n` now also calls
  `evalf(n)` on SymPy values rather than just formatting NumPy floats to
  `n` decimals. Pass `precision=2` explicitly for the old behavior.

## [Unreleased]

### Added
- **Intelligent color interpolation for 2D surface patches**
  - `patch()` now automatically selects between `tricontourf` and `tripcolor` for optimal visual quality
  - New `interpolation_method='auto'` parameter (default) intelligently chooses rendering method:
    - Uses `tricontourf` (256 levels) for small meshes (<100 elements) with non-linear colormaps (jet, hsv, hot, etc.)
    - Uses `tripcolor` (Gouraud shading) for large meshes or linear colormaps (viridis, plasma, etc.)
  - Manual override available: `interpolation_method='tricontourf'` or `'tripcolor'`
  - Non-linear colormap detection for 30+ common colormaps
  - Eliminates RGB interpolation artifacts with non-linear colormaps
  - Maintains performance for large FEM meshes while ensuring quality for instructional plots

- **Enhanced `colorbar()` function with automatic parameter retrieval**
  - Zero-argument usage: `mk.colorbar()` automatically retrieves colormap and limits from last `patch()` call
  - Eliminates need for manual `ScalarMappable` creation and parameter repetition
  - **Automatic discrete/continuous colorbar selection**:
    - Flat colors (per-element data): Creates discrete colorbar with distinct color bands using `BoundaryNorm`
    - Interpolated colors (FaceColor='interp'): Creates continuous colorbar with smooth gradients
    - Discrete colorbar boundaries are automatically placed at midpoints between unique data values
    - Perfectly matches the actual element colors shown in the patch
  - Supports both `limits=[vmin, vmax]` and `clims=[vmin, vmax]` (MATLAB compatibility)
  - New parameters: `ticks`, `orientation`, `extend`, `format` for complete control
  - Backward compatible with existing code
  - Example: `mk.patch(..., 'cmap', 'jet')` then `mk.colorbar()` - automatically uses jet!

- `patch()` now supports `return_mappable=True` parameter for easy colorbar creation
  - Returns tuple `(collection, mappable)` when `FaceColor='interp'` is used
  - Eliminates need for manual ScalarMappable/Normalize boilerplate
  - Example: `collection, mappable = mk.patch(..., return_mappable=True)` then `plt.colorbar(mappable)`

### Changed
- **Breaking improvement**: When `FaceColor='interp'`, smooth interpolation is always used
  - The `Shading` parameter is now only relevant for non-interpolated rendering
  - `tripcolor` always uses `shading='gouraud'` for interpolation mode
  - `tricontourf` always uses 256 levels for smooth gradients

- `patch()` internal state storage enhanced
  - Now stores `cmap`, `vmin`, `vmax` for automatic colorbar retrieval
  - Updated `_store_patch_state()` signature to include colormap information

### Fixed
- **CRITICAL**: Fixed missing `_create_quad_with_pcolormesh()` function
  - Removed broken reference that caused crashes
  - Replaced with proper `tricontourf`/`tripcolor` implementation
  - All 2D surface interpolation now works correctly

- **Fixed `EdgeAlpha` parameter implementation**
  - `EdgeAlpha` was defined but not properly applied in all patch modes
  - Now correctly controls edge transparency independently of `FaceAlpha`
  - Enhanced `_apply_alpha()` helper to handle both color arrays and single colors
  - Works with all patch types: 2D surfaces, 3D surfaces, and line elements
  - Example: `mk.patch(..., 'EdgeColor', 'red', 'EdgeAlpha', 0.3)` creates semi-transparent edges

- **Fixed crash when all vertex values are identical** (vmin == vmax edge case)
  - `tricontourf` requires strictly increasing contour levels
  - When all values are the same, `np.linspace(vmin, vmax, 256)` creates constant levels
  - Now adds small epsilon (1% of value or 1.0 if zero) to create valid range
  - Prevents `ValueError: "Contour levels must be increasing"` crash
  - Example: mesh with uniform temperature field no longer crashes

## [0.1.1] - 2025-12-02

### Added
- Version information: `__version__` attribute and `version()` function
- `display_labeled_latex()` function for displaying labeled LaTeX equations in Jupyter notebooks
  - Supports both NumPy arrays and SymPy symbolic expressions
  - Automatic detection of SymPy objects for proper symbolic formatting
  - `precision` parameter for controlling decimal places in NumPy arrays
  - `arrayStretch` parameter for adjusting vertical row spacing (default: 1.5)
- SymPy integration with automatic `latex()` conversion
  - Detects SymPy matrices, symbols, and expressions
  - Renders symbolic expressions correctly (e.g., `\cos(\theta)` instead of `\cos(\text{theta})`)
- Comprehensive test suite using pytest
  - Tests for `LatexArray` class
  - Tests for `display_labeled_latex` function
  - Tests for SymPy integration
  - Tests for complex number handling
- Example notebooks in `examples/notebooks/`
  - `display_labeled_latex_examples.ipynb` with 8 comprehensive examples
- Documentation improvements
  - `tests/README.md` for testing guidelines
  - `examples/README.md` for example usage
  - Updated main README with new features and structure
  - Version update guidelines in CHANGELOG.md

### Changed
- Reorganized repository structure to follow Python standards
  - Moved all tests to `tests/` directory
  - Moved all examples to `examples/` directory
  - Separated Jupyter notebooks into `examples/notebooks/`
  - Renamed example files from `test_*.py` to `*_demo.py` for clarity
- Fixed `display_labeled_latex` to properly handle SymPy objects
  - Changed from `isinstance(array, Basic)` to module name checking for compatibility
  - Uses `\def\arraystretch` instead of `\renewcommand` for KaTeX compatibility

### Fixed
- **CRITICAL**: `patch()` function now properly interpolates colors for line elements
  - Previously only averaged vertex colors instead of true interpolation
  - Now subdivides each line element into 100 segments for smooth color gradients
  - Fixes visualization of FEM results like displacement/stress fields on truss elements
- SymPy matrix display now shows proper LaTeX formatting (e.g., `\cos(\theta)` not `cos(theta)`)
- KaTeX parse errors with arraystretch command resolved

## [0.1.0] - Earlier

### Added
- Initial release with core features:
  - `LatexArray` class for LaTeX rendering in marimo notebooks
  - `la` renderer with pipe syntax support
  - `Mesh` class with 1-based indexing for FEM
  - MATLAB-style `patch()` function for visualization
  - `OneArray` for 1-based array indexing
