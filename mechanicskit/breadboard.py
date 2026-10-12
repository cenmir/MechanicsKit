r"""Draw breadboard circuits the way Tinkercad Circuits shows them.

A student who has never built a circuit copies it from a picture, hole by hole.
This module draws such pictures: a solderless breadboard with its holes, rails
and centre gap, and the parts that go into it (LEDs, resistors with their colour
code, jumper wires, an ESP32 DevKit), together with the instruments around it
(a bench power supply, a 9 V battery, a USB-C breakout, a handheld multimeter
with its probes). Flat colours, light grey board, dark square holes.

    from mechanicskit import breadboard as bb
    from mechanicskit.sketch_extra import save_svg

    fig, ax = bb.canvas((-3, 34), (-14, 11))
    board = bb.Breadboard(ax, 'half')
    board.resistor('c19', 'c15', 220)
    board.led('e15', 'e14', lit=True, body_dir=(0, -1))
    board.wire('a14', 'T-12', 'black')
    save_svg(fig, 'circuit.svg')

**Units.** One drawing unit is the breadboard pitch, 0.1 inch (2.54 mm). Column
``c`` of the board sits at ``x = c`` and the rows run, from the top, ``a`` to
``e`` (``y = 4`` down to ``0``), the centre gap, then ``f`` to ``j`` (``y = -3``
down to ``-7``). ``canvas`` fixes how many points one unit is, so that wires, legs
and the printing on the instruments can be sized in units; text that is to be read
at book size (callouts) is given in points.

**Holes** are named as on the board: ``'a12'`` is row a, column 12. The four power
rails are ``'T+'``, ``'T-'`` (top pair) and ``'B+'``, ``'B-'`` (bottom pair), and
``'T+12'`` is the hole of the top + rail at column 12 (rail holes come in groups
of five, so not every column has one; ``Breadboard.rail_cols`` lists them). Any
``(x, y)`` pair is accepted wherever a hole is, for parts off the board.

The top pair has − outside and + inside, the bottom pair − inside and + outside,
so a circuit can run from the top + rail down to the bottom − rail without a wire
crossing a rail.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

__all__ = [
    "canvas", "Breadboard", "WIRE_COLORS", "LED_COLORS",
    "led", "resistor", "resistor_bands", "wire", "lead", "cable",
    "lab_supply", "battery_9v", "usb_breakout", "multimeter", "probe",
    "esp32_devkit", "ESP32_LEFT", "ESP32_RIGHT", "ESP32_GPIO", "ESP32_CLASSES",
    "callout", "esp32_pinout", "led_pinout",
]

# --- palette -------------------------------------------------------------------

BOARD = "#e6e6e6"
BOARD_EDGE = "#c2c2c2"
GAP = "#d4d4d4"
HOLE = "#4e4e4e"
RAIL_RED = "#d62828"
RAIL_BLUE = "#1f5fbf"
LEG = "#a3a3a3"
LEG_EDGE = "#6e6e6e"
INK = "#333333"
SANS = "DejaVu Sans"

WIRE_COLORS = {
    "red": "#e53935", "black": "#2e2e2e", "blue": "#1e88e5", "green": "#43a047",
    "yellow": "#fdd835", "orange": "#fb8c00", "white": "#f4f4f4", "purple": "#8e24aa",
    "grey": "#8d8d8d", "brown": "#8d5524",
}
LED_COLORS = {"red": "#e8302a", "green": "#2fb344", "yellow": "#f7d21e",
              "blue": "#2a7fe0", "white": "#f2f2f2", "orange": "#ff8a1e"}

BAND_COLORS = {
    "black": "#1b1b1b", "brown": "#7a3d12", "red": "#d32f2f", "orange": "#f57c00",
    "yellow": "#fbd02d", "green": "#2e9e44", "blue": "#1e6fd0", "violet": "#7b2fa8",
    "grey": "#8a8a8a", "white": "#fafafa", "gold": "#c9a227", "silver": "#bdbdbd",
}
DIGIT_COLOR = ["black", "brown", "red", "orange", "yellow", "green", "blue",
               "violet", "grey", "white"]
TOLERANCE_COLOR = {1: "brown", 2: "red", 0.5: "green", 0.25: "blue", 0.1: "violet",
                   0.05: "grey", 5: "gold", 10: "silver"}

ROWS = {"a": 4, "b": 3, "c": 2, "d": 1, "e": 0, "f": -3, "g": -4, "h": -5, "i": -6, "j": -7}
RAILS = {"T-": 8.0, "T+": 7.0, "B-": -10.0, "B+": -11.0}


# --- small helpers -------------------------------------------------------------

def _xy(p):
    return np.asarray(p, dtype=float).flatten()[:2]


def _unit(v):
    v = _xy(v)
    return v/np.linalg.norm(v)


def _perp(v):
    u = _unit(v)
    return np.array([-u[1], u[0]])


def _mix(color, other, f):
    c, o = np.array(to_rgb(color)), np.array(to_rgb(other))
    return tuple((1 - f)*c + f*o)


def _dark(color, f=0.35):
    return _mix(color, "black", f)


def _light(color, f=0.5):
    return _mix(color, "white", f)


def _pt(ax):
    """Points per drawing unit on this axes (fixed by ``canvas``)."""
    u = getattr(ax, "_bb_unit_pt", None)
    if u is not None:
        return u
    fig = ax.figure
    x0, x1 = ax.get_xlim()
    return ax.get_window_extent().width/fig.dpi*72/(x1 - x0)


def _halo(lw=3):
    return [patheffects.withStroke(linewidth=lw, foreground="white")]


def _text(ax, xy, s, size, color=INK, family=SANS, weight="normal", ha="center",
          va="center", rotation=0, zorder=8, halo=False, **kw):
    """Text whose height is ``size`` drawing units (printing on a part)."""
    kw = dict(kw)
    if halo:
        kw["path_effects"] = _halo()
    return ax.text(*_xy(xy), s, fontsize=size*_pt(ax), color=color, family=family,
                   weight=weight, ha=ha, va=va, rotation=rotation, zorder=zorder,
                   rotation_mode="anchor", **kw)


def _line(ax, pts, width, color, zorder=4, edge=None, cap="round", join="round",
          alpha=1.0):
    """A polyline ``width`` units wide, with an optional darker outline."""
    pts = np.asarray(pts, float)
    k = _pt(ax)
    if edge is not None:
        ax.add_line(Line2D(pts[:, 0], pts[:, 1], lw=(width + 0.1)*k, color=edge,
                           solid_capstyle=cap, solid_joinstyle=join, zorder=zorder,
                           alpha=alpha))
    ax.add_line(Line2D(pts[:, 0], pts[:, 1], lw=width*k, color=color,
                       solid_capstyle=cap, solid_joinstyle=join, zorder=zorder + 0.01,
                       alpha=alpha))


def _poly(ax, pts, fc, ec="none", lw=0.8, zorder=3, alpha=1.0, **kw):
    p = Polygon(np.asarray(pts, float), closed=True, facecolor=fc, edgecolor=ec, lw=lw,
                zorder=zorder, alpha=alpha, joinstyle="round", **kw)
    ax.add_patch(p)
    return p


def _rrect(ax, xy, w, h, r, fc, ec="none", lw=0.8, zorder=3, alpha=1.0):
    """A rounded rectangle, lower-left corner ``xy``, corner radius ``r``."""
    x, y = _xy(xy)
    p = FancyBboxPatch((x + r, y + r), w - 2*r, h - 2*r,
                       boxstyle=f"round,pad={r},rounding_size={r}", facecolor=fc,
                       edgecolor=ec, lw=lw, zorder=zorder, alpha=alpha)
    ax.add_patch(p)
    return p


def _circle(ax, c, r, fc, ec="none", lw=0.8, zorder=3, alpha=1.0):
    p = Circle(_xy(c), r, facecolor=fc, edgecolor=ec, lw=lw, zorder=zorder, alpha=alpha)
    ax.add_patch(p)
    return p


def _bezier(p0, p1, p2, p3, n=60):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t)**3*p0 + 3*(1 - t)**2*t*p1 + 3*(1 - t)*t**2*p2 + t**3*p3)


class _Frame:
    """A local frame: origin and rotation, local (x, y) -> page."""

    def __init__(self, origin, angle=0.0):
        self.o = _xy(origin)
        t = np.radians(angle)
        self.R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        self.angle = angle

    def __call__(self, *p):
        p = _xy(p[0] if len(p) == 1 else p)
        return self.o + self.R @ p

    def pts(self, pts):
        return np.asarray(pts, float) @ self.R.T + self.o


def canvas(xlim, ylim, unit_pt=7.2):
    """A figure in which one drawing unit (0.1 inch of breadboard) is ``unit_pt`` points.

    The axes fill the figure, have equal aspect, no frame and a white background;
    the limits are fixed here, because the widths of wires and legs and the text on
    the parts are computed from the scale. ``unit_pt=7.2`` draws the breadboard at
    true size; a close-up (one LED, the ESP32 pinout) takes 20 or more.
    """
    (x0, x1), (y0, y1) = xlim, ylim
    fig = plt.figure(figsize=((x1 - x0)*unit_pt/72, (y1 - y0)*unit_pt/72))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax._bb_unit_pt = unit_pt
    return fig, ax


# --- the breadboard -------------------------------------------------------------

class Breadboard:
    """A solderless breadboard, ``'half'`` (30 columns) or ``'full'`` (63 columns).

    Each column of rows a to e is one connected strip of five holes, and so is each
    column of rows f to j; the centre gap separates the two halves. Each rail is
    connected along its whole length. The board is drawn on creation; the methods
    place parts with their legs in named holes (see the module docstring).
    """

    def __init__(self, ax, size="half", origin=(0, 0), labels=True, zorder=1):
        self.ax = ax
        self.ncol = {"half": 30, "full": 63}[size] if isinstance(size, str) else int(size)
        self.o = _xy(origin)
        self.z = zorder
        span = 10*6 - 1 if self.ncol > 40 else 5*6 - 1
        start = (self.ncol - span)//2 + 1 if self.ncol > 40 else 2
        ngroups = 10 if self.ncol > 40 else 5
        self.rail_cols = [start + 6*g + i for g in range(ngroups) for i in range(5)]
        self._draw(labels)

    # coordinates
    @property
    def extent(self):
        """(x0, x1, y0, y1) of the board outline."""
        return (self.o[0] - 2.3, self.o[0] + self.ncol + 2.3,
                self.o[1] - 12.4, self.o[1] + 9.4)

    def hole(self, name):
        """The centre of hole ``'a12'``."""
        row, col = name[0].lower(), int(name[1:])
        if row not in ROWS or not 1 <= col <= self.ncol:
            raise ValueError(f"no hole {name!r} on this board")
        return self.o + (col, ROWS[row])

    def rail(self, rail, col):
        """The hole of rail ``'T+'``, ``'T-'``, ``'B+'`` or ``'B-'`` at column ``col``."""
        if col not in self.rail_cols:
            raise ValueError(f"rail {rail} has no hole at column {col}; "
                             f"holes at {self.rail_cols}")
        return self.o + (col, RAILS[rail])

    def xy(self, spec):
        """A hole name (``'a12'``, ``'T+12'``) or an ``(x, y)`` pair, as a point."""
        if isinstance(spec, str):
            if spec[0] in "TB":
                return self.rail(spec[:2], int(spec[2:]))
            return self.hole(spec)
        return _xy(spec)

    def strip(self, name):
        """The five holes connected to hole ``name`` (a column of a to e or of f to j)."""
        row, col = name[0].lower(), int(name[1:])
        rows = "abcde" if row in "abcde" else "fghij"
        return [f"{r}{col}" for r in rows]

    # drawing
    def _draw(self, labels):
        ax, o = self.ax, self.o
        x0, x1, y0, y1 = self.extent
        _rrect(ax, (x0, y0), x1 - x0, y1 - y0, 0.6, BOARD, BOARD_EDGE, 1.0, self.z)
        # centre gap, and the grooves that separate the rails from the field
        _rrect(ax, (x0 + 0.6, o[1] - 1.85), x1 - x0 - 1.2, 0.7, 0.2, GAP, zorder=self.z + 0.01)
        for yc in (5.85, -8.85):
            ax.add_patch(Rectangle((x0 + 0.6, o[1] + yc - 0.06), x1 - x0 - 1.2, 0.12,
                                   facecolor=GAP, edgecolor="none", zorder=self.z + 0.01))
        s = 0.42
        for c in range(1, self.ncol + 1):
            for r, y in ROWS.items():
                ax.add_patch(Rectangle(o + (c - s/2, y - s/2), s, s, facecolor=HOLE,
                                       edgecolor="none", zorder=self.z + 0.02))
        for name, y in RAILS.items():
            for c in self.rail_cols:
                ax.add_patch(Rectangle(o + (c - s/2, y - s/2), s, s, facecolor=HOLE,
                                       edgecolor="none", zorder=self.z + 0.02))
        # rail lines: red beside +, blue beside −, on the outer side of each rail
        xa, xb = o[0] + self.rail_cols[0] - 0.8, o[0] + self.rail_cols[-1] + 0.8
        for name, dy in (("T-", 0.62), ("T+", -0.62), ("B-", 0.62), ("B+", -0.62)):
            y = o[1] + RAILS[name] + dy
            col = RAIL_RED if name[1] == "+" else RAIL_BLUE
            ax.add_line(Line2D([xa, xb], [y, y], lw=0.09*_pt(ax), color=col,
                               zorder=self.z + 0.03, solid_capstyle="butt"))
            sign = "+" if name[1] == "+" else "−"
            for x in (x0 + 1.0, x1 - 1.0):
                _text(ax, (x, o[1] + RAILS[name]), sign, 1.0, col, weight="bold",
                      zorder=self.z + 0.04)
        if labels:
            for c in range(1, self.ncol + 1):
                if c == 1 or c % 5 == 0:
                    for y in (5.2, -8.2):
                        _text(ax, o + (c, y), str(c), 0.62, "#8a8a8a",
                              zorder=self.z + 0.04)
            for r, y in ROWS.items():
                for x in (x0 + 1.0, x1 - 1.0):
                    _text(ax, (x, o[1] + y), r, 0.66, "#8a8a8a", zorder=self.z + 0.04)

    def highlight_strip(self, name, color="#7cc242", alpha=0.45, pad=0.42):
        """Shade the holes connected to ``name``: a five-hole strip, or a whole rail
        (``'T+'``, ``'B-'``, ...)."""
        ax, o = self.ax, self.o
        if name[:2] in RAILS and len(name) == 2:
            y = o[1] + RAILS[name]
            xa, xb = o[0] + self.rail_cols[0], o[0] + self.rail_cols[-1]
            return _rrect(ax, (xa - pad, y - pad), xb - xa + 2*pad, 2*pad, pad*0.8,
                          color, zorder=self.z + 0.015, alpha=alpha)
        pts = np.array([self.hole(h) for h in self.strip(name)])
        xa, ya, yb = pts[0, 0], pts[:, 1].min(), pts[:, 1].max()
        return _rrect(ax, (xa - pad, ya - pad), 2*pad, yb - ya + 2*pad, pad*0.8, color,
                      zorder=self.z + 0.015, alpha=alpha)

    # parts in holes
    def led(self, anode, cathode, color="red", lit=0.0, body_dir=None, **kw):
        """An LED with its anode leg in hole ``anode`` and cathode leg in ``cathode``."""
        return led(self.ax, self.xy(anode), self.xy(cathode), color, lit, body_dir, **kw)

    def resistor(self, hole1, hole2, ohms=220, tol=5, **kw):
        """A resistor between two holes, its bands read from ``hole1``."""
        return resistor(self.ax, self.xy(hole1), self.xy(hole2), ohms, tol, **kw)

    def wire(self, hole1, hole2, color="red", route="straight", **kw):
        """A jumper wire between two holes; see ``wire`` for ``route``."""
        return wire(self.ax, self.xy(hole1), self.xy(hole2), color, route, **kw)

    def esp32(self, col, **kw):
        """An ESP32 DevKit V1 straddling the gap: pins in rows b and j from column
        ``col`` to ``col + 14``, antenna to the left, USB to the right.

        Returns a dict pin name -> hole name (``'D23': 'b49'``); the free holes of the
        strips above the pins are then ``'a49'`` and so on.
        """
        o = self.o + (col, (ROWS["b"] + ROWS["j"])/2)
        esp32_devkit(self.ax, o, angle=90, **kw)
        holes = {}
        for k, n in enumerate(ESP32_RIGHT):
            holes.setdefault(n, f"b{col + k}")
        for k, n in enumerate(ESP32_LEFT):
            holes.setdefault(n if n != "GND" else "GND2", f"j{col + k}")
        return holes


# --- LED ----------------------------------------------------------------------

def led(ax, anode, cathode, color="red", lit=0.0, body_dir=None, lift=1.7, zorder=5,
        kink=True):
    """A 5 mm LED seen from the side, its legs ending at ``anode`` and ``cathode``.

    The body (rim with the flat on the cathode side, cylinder, dome) stands ``lift``
    units from the midpoint of the two leg ends, in direction ``body_dir`` (default
    perpendicular to the legs, upwards or to the right). The anode leg carries a
    kink, the way an LED's longer leg is bent to sit level. ``lit`` from 0 to 1
    lights it, with a glow; 0.25 is a dim glow. Returns a dict with the body
    centre ``'dome'``, the flat edge point ``'flat'`` and the leg ends.
    """
    A, K = _xy(anode), _xy(cathode)
    d = A - K
    if body_dir is None:
        n = _perp(d)
        body_dir = n if (n[1] > 1e-9 or (abs(n[1]) < 1e-9 and n[0] > 0)) else -n
    v = _unit(body_dir)
    w = d - (d @ v)*v
    u = _unit(w) if np.linalg.norm(w) > 1e-9 else _perp(v)
    base = (A + K)/2 + lift*v
    F = _Frame(base, np.degrees(np.arctan2(v[1], v[0])) - 90)
    # local frame: x along the rim (the flipped one if the anode is not at +x)
    sgn = 1.0 if (F.R @ np.array([1.0, 0.0])) @ u > 0 else -1.0

    def P(x, y):
        return F(sgn*x, y)

    col = LED_COLORS.get(color, color)
    edge = _dark(col, 0.45)
    body = _light(col, 0.35*lit) if lit else col
    # legs: from the rim down to the holes
    exits = {"A": P(0.5, 0), "K": P(-0.5, 0)}
    k = _pt(ax)
    for key, end in (("K", K), ("A", A)):
        e = exits[key]
        if key == "A" and kink:
            pts = [e, P(0.5, -0.35), P(0.78, -0.6), P(0.5, -0.85), end]
        else:
            pts = [e, end]
        _line(ax, pts, 0.2, LEG, zorder, edge=LEG_EDGE, cap="butt")
        _rrect(ax, end - 0.17, 0.34, 0.34, 0.06, _dark(LEG, 0.3), zorder=zorder + 0.02)
    # glow
    dome = P(0, 2.45)
    if lit:
        for r, a in ((4.2, 0.10), (3.3, 0.13), (2.5, 0.17), (1.8, 0.22)):
            _circle(ax, dome, r*(0.6 + 0.4*min(lit, 1)), _light(col, 0.25), zorder=zorder - 0.5,
                    alpha=a*min(1.0, 0.35 + lit))
    # rim with the flat on the cathode side, cylinder and dome
    rim = [P(-0.98, 0), P(1.25, 0), P(1.25, 0.45), P(-0.98, 0.45)]
    _poly(ax, rim, body, edge, 0.8, zorder + 0.1, alpha=0.95)
    t = np.linspace(0, np.pi, 40)
    dome_pts = [P(0.98*np.cos(a), 2.45 + 0.98*np.sin(a)) for a in t]
    outline = [P(0.98, 0.45)] + dome_pts + [P(-0.98, 0.45)]
    _poly(ax, outline, body, edge, 0.8, zorder + 0.12, alpha=0.95)
    # highlight: a light stripe and a reflection in the dome
    stripe = [P(-0.62, 0.6), P(-0.38, 0.6), P(-0.38, 2.4), P(-0.62, 2.35)]
    _poly(ax, stripe, "white", zorder=zorder + 0.14, alpha=0.45)
    t = np.linspace(np.radians(110), np.radians(160), 12)
    refl = [P(0.7*np.cos(a), 2.45 + 0.7*np.sin(a)) for a in t]
    _line(ax, refl, 0.16, "white", zorder + 0.15, alpha=0.7)
    if lit:
        _circle(ax, P(0, 2.2), 0.55, _light(col, 0.75), zorder=zorder + 0.13, alpha=0.6*min(lit, 1))
    return {"dome": dome, "flat": P(-0.98, 0.22), "anode": A, "cathode": K,
            "rim_anode": P(1.25, 0.22), "base": base, "up": v}


# --- resistor -----------------------------------------------------------------

def resistor_bands(ohms, tol=5):
    """The four colour bands of a resistor: digit, digit, multiplier, tolerance.

    ``ohms`` is rounded to two significant digits (the E12 and E24 values all have
    two). 220 gives red, red, brown, gold; 1000 gives brown, black, red, gold.
    """
    if ohms <= 0:
        raise ValueError("ohms must be positive")
    e = int(np.floor(np.log10(ohms))) - 1
    value = int(round(ohms/10.0**e))
    if value >= 100:
        value //= 10
        e += 1
    if not -2 <= e <= 9:
        raise ValueError(f"{ohms} ohm is outside the four-band colour code")
    mult = {-2: "silver", -1: "gold"}.get(e, DIGIT_COLOR[e] if e >= 0 else None)
    return [DIGIT_COLOR[value//10], DIGIT_COLOR[value % 10], mult, TOLERANCE_COLOR[tol]]


def _resistor_profile(L=2.7, r_end=0.52, r_mid=0.41, n=80):
    """Half-height of the dog-bone body against the position s along it."""
    s = np.linspace(-L/2, L/2, n)
    a = np.abs(s)
    cap0 = L/2 - r_end                      # where the rounded end begins
    h = np.full_like(s, r_mid)
    bulb = a > 0.52
    h[bulb] = r_end
    tr = (a > 0.36) & (a <= 0.52)
    h[tr] = r_mid + (r_end - r_mid)*0.5*(1 - np.cos(np.pi*(a[tr] - 0.36)/0.16))
    end = a > cap0
    h[end] = r_end*np.sqrt(np.clip(1 - ((a[end] - cap0)/r_end)**2, 0, 1))*0.9 + 0.1*r_end
    return s, h


BAND_POS = (-0.83, -0.47, -0.13, 0.78)          # centre of each band along the body


def resistor(ax, p1, p2, ohms=220, tol=5, body_len=2.7, zorder=5):
    """A resistor lying on the board, legs ending at ``p1`` and ``p2``.

    The beige body sits midway with its four bands (``resistor_bands``) read from
    the ``p1`` end, tolerance band nearest ``p2``. Returns the page position of the
    body centre and of each band (for labelling).
    """
    p1, p2 = _xy(p1), _xy(p2)
    u = _unit(p2 - p1)
    c = (p1 + p2)/2
    F = _Frame(c, np.degrees(np.arctan2(u[1], u[0])))
    L = body_len
    for end, s in ((p1, -L/2), (p2, L/2)):
        _line(ax, [F(s, 0), end], 0.2, LEG, zorder, edge=LEG_EDGE, cap="butt")
        _rrect(ax, end - 0.17, 0.34, 0.34, 0.06, _dark(LEG, 0.3), zorder=zorder + 0.02)
    s, h = _resistor_profile(L)
    outline = F.pts(np.r_[np.c_[s, h], np.c_[s[::-1], -h[::-1]]])
    _poly(ax, outline, "#e3c591", "#9c7f4e", 0.8, zorder + 0.1)
    bands = resistor_bands(ohms, tol)
    where = []
    bw = 0.2
    for name, sc in zip(bands, BAND_POS):
        sel = (s >= sc - bw/2) & (s <= sc + bw/2)
        ss = np.r_[sc - bw/2, s[sel], sc + bw/2]
        hh = np.interp(ss, s, h)
        pts = F.pts(np.r_[np.c_[ss, hh], np.c_[ss[::-1], -hh[::-1]]])
        _poly(ax, pts, BAND_COLORS[name], _dark(BAND_COLORS[name], 0.3), 0.3, zorder + 0.12)
        where.append(F(sc, 0))
    sh = [F(x, y) for x, y in ((-0.9, 0.22), (0.9, 0.22), (0.9, 0.3), (-0.9, 0.3))]
    _poly(ax, sh, "white", zorder=zorder + 0.11, alpha=0.3)
    return {"centre": c, "bands": where, "colors": bands, "axis": u}


# --- wires and cables ---------------------------------------------------------

def wire(ax, p1, p2, color="red", route="straight", height=1.5, via=None, width=0.42,
         zorder=6):
    """A jumper wire from ``p1`` to ``p2`` with dark pin ends in the holes.

    ``route``: ``'straight'``; ``'hv'`` (along x first, then y) or ``'vh'``; ``'arc'``,
    a smooth hump ``height`` units to the left of the travel direction (negative
    for the right). ``via`` adds corner points in between.
    """
    p1, p2 = _xy(p1), _xy(p2)
    col = WIRE_COLORS.get(color, color)
    if route == "arc":
        n = _perp(p2 - p1)
        m = (p1 + p2)/2 + 2*height*n
        t = np.linspace(0, 1, 50)[:, None]
        pts = (1 - t)**2*p1 + 2*(1 - t)*t*m + t**2*p2
    else:
        pts = [p1] + [_xy(q) for q in (via or [])]
        if route == "hv":
            pts.append((p2[0], pts[-1][1]))
        elif route == "vh":
            pts.append((pts[-1][0], p2[1]))
        pts = np.array(pts + [p2])
    _line(ax, pts, width, col, zorder, edge=_dark(col, 0.35))
    for p in (p1, p2):
        _circle(ax, p, 0.2, _dark(col, 0.45), zorder=zorder + 0.02)
    return pts


def cable(ax, p0, d0, p1, d1, color="red", width=0.36, slack=None, zorder=6):
    """A flexible cable from ``p0`` leaving along ``d0`` to ``p1`` arriving along ``d1``.

    A cubic Bezier; ``slack`` (default a third of the distance) is how far the
    cable keeps its end directions.
    """
    p0, p1 = _xy(p0), _xy(p1)
    k = slack if slack is not None else 0.4*np.linalg.norm(p1 - p0)
    pts = _bezier(p0, p0 + k*_unit(d0), p1 - k*_unit(d1), p1)
    col = WIRE_COLORS.get(color, color)
    _line(ax, pts, width, col, zorder, edge=_dark(col, 0.35))
    return pts


def lead(ax, start, hole, color="red", d0=(0, -1), d1=None, slack=None, zorder=6.5):
    """An off-board test lead from ``start`` (a terminal) to ``hole``.

    The lead leaves ``start`` along ``d0`` and ends in a pin housing that sits in the
    hole, arriving along ``d1`` (default the straight direction).
    """
    start, hole = _xy(start), _xy(hole)
    d1 = _unit(hole - start) if d1 is None else _unit(d1)
    col = WIRE_COLORS.get(color, color)
    tail = hole - 1.0*d1
    cable(ax, start, d0, tail, d1, color, slack=slack, zorder=zorder)
    _line(ax, [tail - 0.15*d1, hole], 0.6, _dark(col, 0.15), zorder + 0.02,
          edge=_dark(col, 0.5), cap="butt")
    _circle(ax, hole, 0.2, _dark(col, 0.5), zorder=zorder + 0.04)
    _circle(ax, start, 0.62, col, _dark(col, 0.45), 0.8, zorder=zorder + 0.05)
    _circle(ax, start, 0.25, _dark(col, 0.3), zorder=zorder + 0.06)


# --- seven-segment digits -----------------------------------------------------

_SEG = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd",
        "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcdfg", "-": "g", " ": ""}


def _seg7(ax, xy, text, h, color, ghost=True, zorder=8):
    """Seven-segment digits of height ``h``, lower-left at ``xy``; '.' is a point."""
    x0, y0 = _xy(xy)
    w, t, g = 0.52*h, 0.13*h, 0.03*h
    ends = {"a": ((0, h), (w, h)), "b": ((w, h), (w, h/2)), "c": ((w, h/2), (w, 0)),
            "d": ((0, 0), (w, 0)), "e": ((0, h/2), (0, 0)), "f": ((0, h), (0, h/2)),
            "g": ((0, h/2), (w, h/2))}
    skew = 0.1
    x = x0
    for ch in text:
        if ch == ".":
            _circle(ax, (x - 0.14*h + skew*0.0, y0 + 0.05*h), 0.07*h, color, zorder=zorder)
            continue
        on = _SEG.get(ch, "")
        for sname, (p, q) in ends.items():
            p, q = np.array(p, float), np.array(q, float)
            uu = (q - p)/np.linalg.norm(q - p)
            nn = np.array([-uu[1], uu[0]])
            pts = [p + g*uu, p + (g + t/2)*uu + t/2*nn, q - (g + t/2)*uu + t/2*nn,
                   q - g*uu, q - (g + t/2)*uu - t/2*nn, p + (g + t/2)*uu - t/2*nn]
            pts = np.array([(x + px + skew*py, y0 + py) for px, py in pts])
            if sname in on:
                _poly(ax, pts, color, zorder=zorder)
            elif ghost:
                _poly(ax, pts, color, zorder=zorder - 0.01, alpha=0.08)
        x += 0.78*h


def _dc_mark(ax, xy, w, color, zorder=8):
    """The DC symbol (a line over a dashed line), ``w`` wide, centred at ``xy``."""
    x, y = _xy(xy)
    k = _pt(ax)
    lw = 0.09*w*k
    ax.add_line(Line2D([x - w/2, x + w/2], [y + 0.12*w, y + 0.12*w], lw=lw, color=color,
                       zorder=zorder, solid_capstyle="butt"))
    for i in range(3):
        a = x - w/2 + i*w/2.6
        ax.add_line(Line2D([a, a + w/4.2], [y - 0.12*w]*2, lw=lw, color=color,
                           zorder=zorder, solid_capstyle="butt"))


# --- bench power supply -------------------------------------------------------

def lab_supply(ax, xy, volts=2.0, amps=0.02, mode="CV", output=True, model="BASETech BT-305",
               zorder=3):
    """A bench power supply in the style of the BASETech BT-305, lower-left at ``xy``,
    30 by 19 units.

    Cream case with the model name on top; a dark window with the red current
    display (A) on the left and the green voltage display (V) on the right, each
    with three digits, the C.C. lamp (red) under the current display and the C.V.
    lamp (green) under the voltage display, the active one lit (``mode``), and the
    OUTPUT button at the right of the window (lit when ``output``). Below: the
    CURRENT knob (the current limit) and the VOLTAGE box with FINE and COARSE knobs.
    Bottom strip: POWER switch, rating and the terminals − (black), earth (yellow)
    and + (red). Returns a dict of points: ``'+'``, ``'-'``, ``'earth'``,
    ``'I_knob'``, ``'V_fine'``, ``'V_coarse'``, ``'CV'``, ``'CC'``, ``'output'``,
    ``'power'``, ``'V_display'``, ``'A_display'``.
    """
    x, y = _xy(xy)
    P = lambda a, b: np.array([x + a, y + b])
    W, H = 30.0, 19.0
    ink = "#4a4a4a"
    _rrect(ax, P(0, 0), W, H, 0.9, "#f1ede2", "#b3ab98", 1.0, zorder)
    _text(ax, P(1.3, 17.75), model, 1.05, ink, weight="bold", ha="left", zorder=zorder + 0.05)
    out = {}
    # display window
    _rrect(ax, P(1.0, 10.4), 20.6, 6.3, 0.4, "#1c1f22", "#0d0f10", 0.8, zorder + 0.02)

    def three(v):
        s = f"{v:.2f}" if v < 9.995 else f"{v:.1f}"
        return s if sum(ch.isdigit() for ch in s) == 3 else f"{v:.0f}"

    for x0, val, unit, col, key, lamp, lname, lcol in (
            (1.5, three(amps), "A", "#ff3b30", "A_display", (5.6, 11.25), "C.C.", "#ff3b30"),
            (11.5, three(volts), "V", "#3ddc5c", "V_display", (15.6, 11.25), "C.V.", "#3ddc5c")):
        hh = 2.7
        _seg7(ax, P(x0 + 0.9, 12.6), val, hh, col, zorder=zorder + 0.04)
        _text(ax, P(x0 + 8.3, 13.0), unit, 1.15, col, weight="bold", zorder=zorder + 0.04)
        out[key] = P(x0 + 4.5, 16.7)
        c = P(*lamp)
        active = lname.replace(".", "") == mode
        if active:
            for r, al in ((0.9, 0.2), (0.65, 0.3)):
                _circle(ax, c, r, lcol, zorder=zorder + 0.03, alpha=al)
        _circle(ax, c, 0.38, lcol if active else _mix(lcol, "#333333", 0.75), zorder=zorder + 0.04)
        _text(ax, c + (0.8, 0), lname, 0.72, "#c8c8c8", ha="left", weight="bold",
              zorder=zorder + 0.05)
        out[lname.replace(".", "")] = c
    # OUTPUT button
    b = P(25.6, 13.9)
    _rrect(ax, b - (2.2, 1.1), 4.4, 2.2, 0.35, "#5c6168" if not output else "#3a8f46",
           "#2a2d31", 0.8, zorder + 0.03)
    if output:
        _rrect(ax, b - (1.6, 0.25), 3.2, 0.5, 0.2, "#7dff8c", zorder=zorder + 0.04, alpha=0.8)
    _text(ax, b + (0, -1.85), "OUTPUT", 0.72, ink, weight="bold", zorder=zorder + 0.05)
    out["output"] = b

    def knob(c, r, ang):
        _circle(ax, c, r, "#2c2f33", "#15171a", 0.9, zorder + 0.03)
        _circle(ax, c, 0.78*r, "#3a3e43", zorder=zorder + 0.035)
        a = np.radians(ang)
        e = np.array([np.cos(a), np.sin(a)])
        _line(ax, [c + 0.2*r*e, c + 0.85*r*e], 0.18, "white", zorder + 0.05)

    # CURRENT knob, VOLTAGE box with FINE and COARSE
    out["I_knob"] = P(5.0, 6.9)
    knob(out["I_knob"], 1.9, 115)
    _text(ax, P(5.0, 4.25), "CURRENT", 0.75, ink, weight="bold", zorder=zorder + 0.05)
    _rrect(ax, P(10.2, 3.6), 18.6, 6.3, 0.4, "none", "#8c8574", 0.9, zorder + 0.02)
    _text(ax, P(19.5, 9.15), "VOLTAGE", 0.75, ink, weight="bold", zorder=zorder + 0.05)
    out["V_fine"], out["V_coarse"] = P(14.6, 6.4), P(23.6, 6.4)
    knob(out["V_fine"], 1.35, 90)
    knob(out["V_coarse"], 1.9, 60)
    _text(ax, P(14.6, 4.25), "FINE", 0.7, ink, weight="bold", zorder=zorder + 0.05)
    _text(ax, P(23.6, 4.25), "COARSE", 0.7, ink, weight="bold", zorder=zorder + 0.05)
    # bottom strip: POWER, rating, terminals
    ax.add_line(Line2D([x + 0.6, x + W - 0.6], [y + 3.1]*2, lw=0.6, color="#b3ab98",
                       zorder=zorder + 0.02))
    sw = P(2.6, 1.55)
    _rrect(ax, sw - (1.3, 0.75), 2.6, 1.5, 0.2, "#2c2f33", "#15171a", 0.8, zorder + 0.03)
    _rrect(ax, sw - (1.15, 0.6) + ((1.15, 0) if output else (0, 0)), 1.15, 1.2, 0.18,
           "#9aa0a6", zorder=zorder + 0.04)
    _text(ax, sw + (1.8, 0), "POWER", 0.68, ink, weight="bold", ha="left", zorder=zorder + 0.05)
    out["power"] = sw
    _text(ax, P(13.3, 1.55), "0~30V/0~5A", 0.72, ink, zorder=zorder + 0.05)
    for cx, col, key in ((20.6, "#222222", "-"), (24.0, "#f2c200", "earth"),
                         (27.4, "#d32f2f", "+")):
        c = P(cx, 1.55)
        _circle(ax, c, 1.05, col, _dark(col, 0.5), 0.9, zorder + 0.03)
        hexa = [c + 0.55*np.array([np.cos(a), np.sin(a)])
                for a in np.radians(np.arange(0, 360, 60) + 30)]
        _poly(ax, hexa, _light(col, 0.25), _dark(col, 0.5), 0.6, zorder + 0.04)
        _circle(ax, c, 0.22, "#1b1b1b", zorder=zorder + 0.05)
        out[key] = c
    return out


# --- 9 V battery and USB-C breakout -------------------------------------------

def battery_9v(ax, xy, angle=0.0, zorder=4):
    """A 9 V battery with its snap clip, 7 by 12.5 units.

    In its own frame the battery stands upright with the clip on top and its
    lower-left corner at ``xy``; ``angle`` turns it about that corner (``-90`` lays it
    on its side with the clip to the right). Returns the points where the red and
    black snap leads leave the clip, ``'+'`` and ``'-'``, and the direction
    ``'out'`` in which they leave (lead them on with ``lead``).
    """
    F = _Frame(xy, angle)
    P = lambda a, b: F(a, b)
    rot = F.angle

    def rr(x0, y0, w, h, r, fc, ec="none", lw=0.8, z=zorder, alpha=1.0):
        _poly(ax, F.pts(_round_rect_pts(x0, y0, w, h, r)), fc, ec, lw, z, alpha=alpha)

    rr(0, 0, 7, 11, 0.5, "#2b2b2b", "#111111", 0.9)
    rr(0.05, 3.2, 6.9, 4.6, 0.05, "#1f6fd1", z=zorder + 0.01)
    _text(ax, P(3.5, 5.5), "9V", 2.2, "white", weight="bold", rotation=rot,
          zorder=zorder + 0.02)
    rr(0.3, 0.3, 1.0, 10.4, 0.4, "white", z=zorder + 0.02, alpha=0.12)
    rr(0.7, 10.7, 5.6, 1.8, 0.35, "#1a1a1a", "#000000", 0.8, z=zorder + 0.03)
    _circle(ax, P(2.3, 11.6), 0.55, "#9a9a9a", "#555555", 0.6, zorder + 0.04)
    hexa = [P(*(np.array([4.7, 11.6]) + 0.6*np.array([np.cos(a), np.sin(a)])))
            for a in np.radians(np.arange(0, 360, 60))]
    _poly(ax, hexa, "#9a9a9a", "#555555", 0.6, zorder + 0.04)
    _text(ax, P(2.3, 9.7), "+", 1.1, "white", weight="bold", zorder=zorder + 0.04)
    _text(ax, P(4.7, 9.7), "\u2212", 1.1, "white", weight="bold", zorder=zorder + 0.04)
    out = F.R @ np.array([0.0, 1.0])
    return {"+": P(2.3, 12.5), "-": P(4.7, 12.5), "out": out}


def usb_breakout(ax, xy, cable_len=4.0, zorder=4):
    """A small USB-C breakout board, lower-left at ``xy`` (8 by 6 units): the USB-C
    socket on its left edge with a cable plugged in, two pins VBUS and GND on its
    right edge. Returns the pin centres ``'VBUS'`` and ``'GND'``."""
    x, y = _xy(xy)
    P = lambda a, b: np.array([x + a, y + b])
    _rrect(ax, P(0, 0), 8, 6, 0.4, "#2a2a6e", "#15153d", 0.9, zorder)
    # socket and plug
    _rrect(ax, P(-0.6, 1.7), 3.0, 2.6, 0.9, "#c9ccd0", "#7c8086", 0.8, zorder + 0.02)
    _rrect(ax, P(-0.1, 2.35), 2.0, 1.3, 0.55, "#2b2b2b", zorder=zorder + 0.03)
    _rrect(ax, P(-0.6 - cable_len*0.35, 1.8), cable_len*0.35 + 0.3, 2.4, 0.4, "#3a3a3a",
           "#1d1d1d", 0.8, zorder + 0.04)
    _line(ax, [P(-0.6 - cable_len*0.35, 3.0), P(-0.6 - cable_len, 3.0)], 1.0, "#3a3a3a",
          zorder + 0.03, edge="#1d1d1d", cap="butt")
    _text(ax, P(3.6, 5.1), "USB-C", 0.75, "white", weight="bold", zorder=zorder + 0.05)
    out = {}
    for yy, name in ((3.9, "VBUS"), (2.1, "GND")):
        c = P(7.0, yy)
        _circle(ax, c, 0.42, "#e0b84a", "#8a6d1e", 0.6, zorder + 0.04)
        _circle(ax, c, 0.18, "#5a4a1a", zorder=zorder + 0.05)
        _text(ax, c + (-1.0, 0), name, 0.72, "white", weight="bold", ha="right",
              zorder=zorder + 0.05)
        out[name] = c
    return out


# --- multimeter ----------------------------------------------------------------

_DIAL = {"off": 90, "V~": 50, "V": 15, "mV": -20, "ohm": -55, "cont": -90,
         "A": 130, "mA": 165, "uA": 200}


def multimeter(ax, xy, reading="0.00", unit="V", mode="V", width=14.0, zorder=4):
    """A handheld digital multimeter, lower-left at ``xy``, ``width`` by 1.85 ``width``.

    The display shows ``reading`` with ``unit`` (``'V'``, ``'mV'``, ``'mA'``,
    ``'A'``, ``'Ω'``; a DC symbol is added for V, mV, mA and A unless
    ``unit`` ends in ``'~'``). The rotary dial points at ``mode``: ``'V'`` (V DC),
    ``'V~'``, ``'mV'``, ``'ohm'``, ``'cont'`` (continuity), ``'mA'`` (mA DC),
    ``'A'``, ``'off'``. Jacks along the bottom: 10A, mA, COM, VΩ. Returns the jack
    centres (``'10A'``, ``'mA'``, ``'COM'``, ``'VΩ'``) for ``probe(..., jack=...)``.
    """
    s = width/14.0
    x, y = _xy(xy)
    P = lambda a, b: np.array([x + s*a, y + s*b])
    H = 26.0
    _rrect(ax, P(0, 0), 14*s, H*s, 1.6*s, "#f2b134", "#b47c12", 1.0, zorder)
    _rrect(ax, P(0.9, 0.9), 12.2*s, (H - 1.8)*s, 1.0*s, "#3c4148", "#24282d", 0.8, zorder + 0.01)
    # display
    _rrect(ax, P(1.8, 18.2), 10.4*s, 5.9*s, 0.5*s, "#232629", zorder=zorder + 0.02)
    _rrect(ax, P(2.2, 18.6), 9.6*s, 5.1*s, 0.3*s, "#c3cfae", zorder=zorder + 0.03)
    digits = reading.strip()
    nd = sum(ch != "." for ch in digits)
    hh = 2.6*s
    right = 9.7 if len(unit.replace(" DC", "").rstrip("~")) < 2 else 9.2
    _seg7(ax, P(right, 19.4) - (nd*0.78*hh - 0.2*hh, 0), digits, hh, "#1d1f1c", ghost=False,
          zorder=zorder + 0.05)
    u = unit.replace("ohm", "Ω")
    _text(ax, P(10.9, 19.9), u.rstrip("~").replace(" DC", ""), 1.1*s, "#1d1f1c",
          weight="bold", zorder=zorder + 0.05, ha="center")
    if unit.endswith("~"):
        _text(ax, P(10.9, 21.9), "~", 1.1*s, "#1d1f1c", zorder=zorder + 0.05)
    elif u in ("V", "mV", "mA", "A", "V DC", "mA DC"):
        _dc_mark(ax, P(10.9, 21.8), 1.2*s, "#1d1f1c", zorder + 0.05)
    # dial
    c = P(7, 10.6)
    labels = {"off": "OFF", "V~": "V~", "V": "V", "mV": "mV", "ohm": "Ω",
              "cont": "·)))", "A": "A", "mA": "mA", "uA": "µA"}
    for key, ang in _DIAL.items():
        a = np.radians(ang)
        r = np.array([np.cos(a), np.sin(a)])
        active = key == mode
        col = "#ffd54f" if active else "#e8e8e8"
        _line(ax, [c + 3.05*s*r, c + 3.45*s*r], 0.12*s, col, zorder + 0.03, cap="butt")
        lp = c + 4.45*s*r
        txt = labels[key]
        t = _text(ax, lp, txt, (0.95 if active else 0.85)*s, col,
                  weight="bold" if active else "normal", zorder=zorder + 0.05)
        if key in ("V", "mV", "mA", "A", "uA"):
            _dc_mark(ax, lp + (0, 0.85*s), 0.75*s, col, zorder + 0.05)
    _circle(ax, c, 2.9*s, "#2a2d31", "#16181b", 0.9, zorder + 0.03)
    a = np.radians(_DIAL[mode])
    r = np.array([np.cos(a), np.sin(a)])
    nrm = np.array([-r[1], r[0]])
    knob = [c - 2.6*s*r + 0.9*s*nrm, c + 2.6*s*r + 0.6*s*nrm, c + 2.6*s*r - 0.6*s*nrm,
            c - 2.6*s*r - 0.9*s*nrm]
    _poly(ax, knob, "#4a4f56", "#16181b", 0.8, zorder + 0.04)
    _poly(ax, [c + 2.55*s*r, c + 1.6*s*r + 0.35*s*nrm, c + 1.6*s*r - 0.35*s*nrm],
          "#ffd54f", zorder=zorder + 0.05)
    # jacks
    out = {}
    for xx, name, ring in ((2.4, "10A", "#d32f2f"), (5.4, "mA", "#d32f2f"),
                           (8.6, "COM", "#1b1b1b"), (11.6, "VΩ", "#d32f2f")):
        jc = P(xx, 3.1)
        _circle(ax, jc, 0.95*s, ring, "#111111", 0.8, zorder + 0.03)
        _circle(ax, jc, 0.55*s, "#111111", zorder=zorder + 0.04)
        _text(ax, P(xx, 5.0), name, 0.8*s, "#e8e8e8", weight="bold", zorder=zorder + 0.05)
        out[name] = jc
    out["VOhm"] = out["VΩ"]
    out["display"] = P(7, 24.1)
    out["dial"] = c
    return out


def probe(ax, tip, color="red", angle=-60.0, length=9.0, jack=None, jack_dir=(0, -1),
          slack=None, zorder=9):
    """A multimeter probe whose metal tip touches ``tip``.

    The probe lies along ``angle`` (degrees, from the tip towards the handle):
    metal tip, finger guard, handle and boot, in the lead's ``color``. With ``jack``
    (a jack centre from ``multimeter``) the cable is drawn from the back of the probe
    to a plug in that jack, leaving the jack along ``jack_dir``. Returns the point
    where the cable leaves the probe.
    """
    tip = _xy(tip)
    d = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    n = np.array([-d[1], d[0]])
    col = WIRE_COLORS.get(color, color)
    L = length/9.0
    Q = lambda s, w: tip + s*L*d + w*n

    def seg(s0, s1, w0, w1, fc, ec, z):
        _poly(ax, [Q(s0, w0), Q(s1, w1), Q(s1, -w1), Q(s0, -w0)], fc, ec, 0.6, z)

    seg(0.0, 1.3, 0.05, 0.1, "#cfd3d6", "#7a7f84", zorder + 0.01)
    seg(1.3, 2.4, 0.22, 0.34, col, _dark(col, 0.5), zorder + 0.02)
    seg(2.4, 2.8, 0.62, 0.62, col, _dark(col, 0.5), zorder + 0.03)
    seg(2.8, 7.6, 0.4, 0.42, col, _dark(col, 0.5), zorder + 0.02)
    seg(7.6, 8.8, 0.42, 0.2, _dark(col, 0.2), _dark(col, 0.5), zorder + 0.02)
    for s in (3.6, 4.2, 4.8):
        _line(ax, [Q(s, 0.36), Q(s, -0.36)], 0.07, _dark(col, 0.3), zorder + 0.03, cap="butt")
    end = Q(8.8, 0)
    if jack is not None:
        jack = _xy(jack)
        cable(ax, end, d, jack, -_unit(jack_dir), color, 0.32, slack=slack, zorder=zorder - 1)
        _circle(ax, jack, 0.85, col, _dark(col, 0.5), 0.8, zorder - 0.9)
        _circle(ax, jack, 0.4, _dark(col, 0.25), zorder=zorder - 0.85)
    return end


# --- ESP32 DevKit V1 ----------------------------------------------------------

# The 30-pin DOIT ESP32 DEVKIT V1, antenna up, USB down, each column top to bottom.
ESP32_LEFT = ["EN", "VP", "VN", "D34", "D35", "D32", "D33", "D25", "D26", "D27", "D14",
              "D12", "D13", "GND", "VIN"]
ESP32_RIGHT = ["D23", "D22", "TX0", "RX0", "D21", "D19", "D18", "D5", "TX2", "RX2", "D4",
               "D2", "D15", "GND", "3V3"]
ESP32_GPIO = {"VP": 36, "VN": 39, "TX0": 1, "RX0": 3, "TX2": 17, "RX2": 16,
              **{f"D{k}": k for k in (2, 4, 5, 12, 13, 14, 15, 18, 19, 21, 22, 23, 25,
                                      26, 27, 32, 33, 34, 35)}}
# pin classes for the pinout: (colour, legend text)
ESP32_CLASSES = {
    "power": ("#d32f2f", "power"),
    "gnd": ("#1b1b1b", "ground"),
    "safe": ("#2e9e44", "safe GPIO"),
    "strap": ("#f57c00", "strapping pin: use with care"),
    "input": ("#1e6fd0", "input only"),
    "avoid": ("#8a8a8a", "avoid: USB serial or reset"),
}


def _esp32_class(name):
    if name in ("3V3", "VIN"):
        return "power"
    if name == "GND":
        return "gnd"
    if name in ("EN", "TX0", "RX0"):
        return "avoid"
    g = ESP32_GPIO[name]
    if g in (34, 35, 36, 39):
        return "input"
    if g in (0, 2, 5, 12, 15):
        return "strap"
    return "safe"


def esp32_devkit(ax, origin, angle=0.0, labels=True, label_size=0.45, zorder=7):
    """A 30-pin ESP32 DevKit V1 (DOIT), drawn in a frame turned by ``angle`` degrees.

    In its own frame the board stands with the antenna up and the USB socket down;
    the pin columns are at x = −5 and +5 (one inch apart), the top pins at y = 0 and
    the bottom ones at y = −14, so ``origin`` is the midpoint between the two top
    pins. ``angle=90`` lays it on a breadboard with the antenna to the left: the
    right column then lies in row b and the left column in row j. ``labels`` prints
    the pin names on the board. Returns a dict pin name -> page position, with the
    two ground pins as ``'GND'`` (right column) and ``'GND2'`` (left).
    """
    F = _Frame(origin, angle)
    P = lambda a, b: F(a, b)
    R = lambda pts: F.pts(pts)
    rect = lambda x0, y0, w, h: R([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)])
    # PCB
    pcb = R(np.r_[_round_rect_pts(-5.57, -16.0, 11.14, 20.3, 0.5)])
    _poly(ax, pcb, "#20262e", "#0d1014", 1.0, zorder)
    # module: antenna part and metal can
    _poly(ax, rect(-3.4, 1.2, 6.8, 3.0), "#1a3a2a", "#0d1014", 0.6, zorder + 0.01)
    mean = [(-2.8, 1.6), (-2.8, 3.7), (-2.2, 3.7), (-2.2, 2.0), (-1.4, 2.0), (-1.4, 3.7),
            (-0.6, 3.7), (-0.6, 2.0), (0.2, 2.0), (0.2, 3.7), (1.0, 3.7), (1.0, 2.0),
            (1.8, 2.0), (1.8, 3.7), (2.6, 3.7)]
    _line(ax, R(mean), 0.18, "#c9a227", zorder + 0.02, cap="butt", join="miter")
    _poly(ax, rect(-3.4, -6.0, 6.8, 7.2), "#c9cdd2", "#80868c", 0.8, zorder + 0.02)
    _poly(ax, rect(-3.05, -5.65, 6.1, 6.5), "#d8dbdf", zorder=zorder + 0.03)
    rot = F.angle
    _text(ax, P(0, -1.2), "ESP-WROOM-32", 0.55, "#555b62", weight="bold",
          rotation=rot, zorder=zorder + 0.05)
    _text(ax, P(0, -2.4), "ESP32", 0.9, "#555b62", weight="bold", rotation=rot,
          zorder=zorder + 0.05)
    # chips: regulator and USB serial bridge
    _poly(ax, rect(-1.5, -10.2, 3.0, 3.0), "#15181c", "#000000", 0.6, zorder + 0.02)
    _poly(ax, rect(-1.0, -13.0, 2.0, 1.3), "#15181c", "#000000", 0.6, zorder + 0.02)
    _circle(ax, P(1.9, -11.4), 0.28, "#ff5252", zorder=zorder + 0.03)
    _circle(ax, P(-1.9, -11.4), 0.28, "#4fc3f7", zorder=zorder + 0.03)
    # buttons EN (left) and BOOT (right) beside the USB socket
    for bx, name in ((-2.75, "EN"), (2.75, "BOOT")):
        _poly(ax, rect(bx - 0.75, -15.25, 1.5, 1.5), "#c9cdd2", "#80868c", 0.6, zorder + 0.03)
        _circle(ax, P(bx, -14.5), 0.45, "#1b1b1b", zorder=zorder + 0.04)
        _text(ax, P(bx, -15.62), name, 0.42, "white", weight="bold", rotation=rot,
              zorder=zorder + 0.05)
    # USB socket
    _poly(ax, rect(-1.5, -16.6, 3.0, 2.2), "#c9cdd2", "#80868c", 0.8, zorder + 0.03)
    _poly(ax, rect(-1.0, -16.6, 2.0, 0.6), "#5b6066", zorder=zorder + 0.04)
    # pins
    out = {}
    for xs, names, ha in ((-5, ESP32_LEFT, "left"), (5, ESP32_RIGHT, "right")):
        for k, name in enumerate(names):
            c = P(xs, -k)
            _circle(ax, c, 0.4, "#d4af37", "#8a6d1e", 0.5, zorder + 0.03)
            _circle(ax, c, 0.18, "#3a3a3a", zorder=zorder + 0.04)
            key = name if not (name == "GND" and xs < 0) else "GND2"
            out[key] = c
            if labels:
                lx = xs + (0.55 if xs < 0 else -0.55)
                _text(ax, P(lx, -k), name, label_size, "white", ha=ha, rotation=rot,
                      zorder=zorder + 0.05)
    out["antenna"] = P(0, 2.7)
    out["usb"] = P(0, -16.0)
    out["EN_button"] = P(-2.75, -14.5)
    out["BOOT_button"] = P(2.75, -14.5)
    return out


def _round_rect_pts(x0, y0, w, h, r, n=8):
    pts = []
    for cx, cy, a0 in ((x0 + w - r, y0 + r, -90), (x0 + w - r, y0 + h - r, 0),
                       (x0 + r, y0 + h - r, 90), (x0 + r, y0 + r, 180)):
        t = np.radians(np.linspace(a0, a0 + 90, n))
        pts += list(zip(cx + r*np.cos(t), cy + r*np.sin(t)))
    return np.array(pts)


# --- annotation ---------------------------------------------------------------

def callout(ax, text, target, xy, fontsize=11, color="#222222", dot=True, ha=None,
            va=None, lw=0.8, zorder=12, **kw):
    """A friendly note at ``xy`` with a thin leader line to ``target``.

    The text sits just beyond ``xy`` on the side away from ``target`` (``ha`` and
    ``va`` override), in the serif face of the book, with the white outline.
    """
    t, p = _xy(target), _xy(xy)
    d = p - t
    if ha is None and va is None:
        if abs(d[0]) >= abs(d[1]):
            ha, va = ("left" if d[0] > 0 else "right"), "center"
        else:
            ha, va = "center", ("bottom" if d[1] > 0 else "top")
    ha = ha or "center"
    va = va or "center"
    gap = 0.35
    off = np.array([{"left": gap, "right": -gap}.get(ha, 0.0),
                    {"bottom": gap, "top": -gap}.get(va, 0.0)])
    ax.add_line(Line2D([t[0], p[0]], [t[1], p[1]], lw=lw, color=color, zorder=zorder,
                       path_effects=_halo(2.5)))
    if dot:
        ax.add_line(Line2D([t[0]], [t[1]], marker="o", ms=4.2, mfc=color, mec="white",
                           mew=0.8, zorder=zorder + 0.1))
    return ax.text(*(p + off), text, fontsize=fontsize, color=color, ha=ha, va=va,
                   zorder=zorder + 0.2, path_effects=_halo(3), linespacing=1.15, **kw)


# --- stand-alone figures -------------------------------------------------------

_FUNCTION = {
    "EN": "reset (EN button)", "VP": "ADC1", "VN": "ADC1", "D34": "ADC1", "D35": "ADC1",
    "D32": "ADC1, touch", "D33": "ADC1, touch", "D25": "DAC1, ADC2", "D26": "DAC2, ADC2",
    "D27": "ADC2, touch", "D14": "ADC2, touch", "D12": "boot: flash voltage",
    "D13": "ADC2, touch", "GND": "ground", "VIN": "5 V in (USB 5 V)",
    "D23": "SPI MOSI", "D22": "I2C SCL", "TX0": "USB serial TX", "RX0": "USB serial RX",
    "D21": "I2C SDA", "D19": "SPI MISO", "D18": "SPI SCK", "D5": "boot: SPI SS",
    "TX2": "UART2 TX", "RX2": "UART2 RX", "D4": "ADC2, touch", "D2": "on-board LED, boot",
    "D15": "boot: log output", "3V3": "3.3 V out",
}


def esp32_pinout(unit_pt=17.0, fontsize=9.5):
    """The ESP32 DevKit V1 enlarged with every pin named: GPIO number and function,
    colour-coded by class (``ESP32_CLASSES``) with a legend. Returns ``fig, ax``."""
    fig, ax = canvas((-24.5, 24.5), (-25.2, 6.6), unit_pt)
    pins = esp32_devkit(ax, (0, 0), 0, labels=True)
    for xs, names in ((-5, ESP32_LEFT), (5, ESP32_RIGHT)):
        sgn = -1 if xs < 0 else 1
        for k, name in enumerate(names):
            key = name if not (name == "GND" and xs < 0) else "GND2"
            c = pins[key]
            cls = _esp32_class(name)
            col = ESP32_CLASSES[cls][0]
            x1 = xs + sgn*1.9
            ax.add_line(Line2D([c[0] + sgn*0.4, x1], [c[1], c[1]], lw=0.8, color="#888888",
                               zorder=6))
            if name in ESP32_GPIO:
                pill = f"GPIO {ESP32_GPIO[name]}"
            else:
                pill = {"3V3": "3.3 V", "VIN": "VIN 5 V", "GND": "GND", "EN": "EN"}[name]
            ax.text(x1 + sgn*0.15, c[1], pill, fontsize=fontsize, color="white",
                    ha="left" if sgn > 0 else "right", va="center", family=SANS,
                    weight="bold", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.5", facecolor=col,
                              edgecolor="none"))
            ax.text(x1 + sgn*6.4, c[1], _FUNCTION[name], fontsize=fontsize - 0.5,
                    color="#333333", ha="left" if sgn > 0 else "right", va="center",
                    zorder=8)
    # legend: two rows of three
    items = list(ESP32_CLASSES.values())
    for i, (col, text) in enumerate(items):
        r, cidx = divmod(i, 3)
        x = -22.5 + cidx*15.5
        y = -19.6 - r*2.2
        ax.add_patch(FancyBboxPatch((x, y - 0.45), 1.6, 0.9,
                                    boxstyle="round,pad=0,rounding_size=0.35",
                                    facecolor=col, edgecolor="none", zorder=8))
        ax.text(x + 2.2, y, text, fontsize=fontsize, color="#222222", ha="left",
                va="center", zorder=8)
    return fig, ax


def led_pinout(color="red", unit_pt=26.0, fontsize=12):
    """One LED large, anode and cathode named, long and short leg and the flat edge
    marked, beside its circuit symbol with A and K. Returns ``fig, ax``."""
    fig, ax = canvas((-9.0, 17.5), (-9.6, 5.6), unit_pt)
    # the LED standing up, legs straight down: anode (long) left, cathode (short) right
    A_end, K_end = np.array([-0.5, -8.6]), np.array([0.5, -7.4])
    info = led(ax, A_end, K_end, color, lit=0, body_dir=(0, 1), lift=8.6, kink=False)
    callout(ax, "anode (+)\nlong leg", (-0.5, -4.5), (-3.2, -4.5), fontsize, ha="right")
    callout(ax, "cathode (−)\nshort leg", (0.5, -4.0), (3.2, -4.0), fontsize, ha="left")
    callout(ax, "flat edge:\ncathode side", info["flat"], (3.4, 0.3), fontsize, ha="left")
    # the symbol: A on the left, K on the right, current flows A -> K
    y, xa, xk = -2.0, 9.0, 16.0
    xm0, xm1 = 11.6, 13.4
    ax.add_line(Line2D([xa, xm0], [y, y], lw=1.6, color="black", zorder=6))
    ax.add_line(Line2D([xm1, xk], [y, y], lw=1.6, color="black", zorder=6))
    _poly(ax, [(xm0, y + 0.95), (xm0, y - 0.95), (xm1, y)], "white", "black", 1.6, 6)
    ax.add_line(Line2D([xm1, xm1], [y - 0.95, y + 0.95], lw=1.6, color="black", zorder=6))
    for dx in (0.0, 0.7):
        p0 = np.array([12.2 + dx, y + 1.1])
        ax.annotate("", xy=p0 + (0.9, 0.9), xytext=p0,
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black",
                                    mutation_scale=9, shrinkA=0, shrinkB=0), zorder=6)
    for x, s, name in ((xa, "A", "anode"), (xk, "K", "cathode")):
        ax.add_patch(Circle((x, y), 0.13, color="black", zorder=7))
        ax.text(x, y + 0.55, s, fontsize=fontsize + 2, ha="center", va="bottom",
                style="italic", zorder=7)
        ax.text(x, y - 0.55, name, fontsize=fontsize - 1, ha="center", va="top", zorder=7)
    ax.text(12.5, y - 2.4, "circuit symbol", fontsize=fontsize - 1, color="#555555",
            ha="center", va="top")
    ax.text(0, -9.4, "the LED", fontsize=fontsize - 1, color="#555555", ha="center",
            va="top")
    return fig, ax
