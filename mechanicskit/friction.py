"""Coefficients of friction for machine design, each table with its source.

Four tables, one per use:

``PAIRS``
    Material pairs, static and sliding, dry and lubricated. From Marks' Standard Handbook,
    8th ed. (1978), as reprinted in Table IV of NASA RP-1228 (Barrett 1990). Only the pairs
    with both dry coefficients, static and sliding, are kept: a pair without its static value
    cannot say whether a part slips. A lubricated
    value is the range over the mineral and vegetable oils the table lists; the boundary
    films of stearic and palmitic acid, far lower, are left out.
``BOLT_CLASSES``
    The friction classes A to E of VDI 2230 Part 1 (2015) for the thread (mu_G) and the head
    or nut bearing face (mu_K), with the surfaces and lubricants that fall in each class.
    Read from Bossard's technical sheet 57 (2025), which reproduces the VDI table.
``LININGS``
    Brake and clutch friction materials, Budynas and Nisbett, *Shigley's Mechanical
    Engineering Design*, Table 16-3. The asbestos linings of that table are left out.
``FITS_SHRINK``, ``FITS_PRESSED``
    Slip coefficients of interference fits after DIN 7190 (2001), as reproduced in the
    eAssistant manual, chapter 15.8. Shrink and oil-pressure fits in the circumferential
    and longitudinal directions; pressed fits with a longitudinal loosening (``'hold'``)
    and sliding (``'press'``) value.

A coefficient is a float, a range is a tuple ``(low, high)``, and an unknown is ``None``.
Friction scatters: a single value is one measurement under the conditions of the source,
so design with the range and check both ends.

    >>> from mechanicskit import friction
    >>> friction.pair('mild steel', 'mild steel')
    {'static_dry': 0.74, 'static_lub': None, 'sliding_dry': 0.57, 'sliding_lub': (0.09, 0.19)}
    >>> friction.bolt_class('B')
    (0.08, 0.16)
    >>> friction.classify_bolt(0.12)
    ['B']
"""
from __future__ import annotations

SOURCES = {
    'marks': "E. A. Avallone, T. Baumeister (eds.), Marks' Standard Handbook for Mechanical "
             "Engineers, 8th ed., McGraw-Hill, 1978; reprinted as Table IV in R. T. Barrett, "
             "Fastener Design Manual, NASA RP-1228, 1990, p. 16",
    'vdi2230': "VDI 2230 Part 1:2015, Systematic calculation of highly stressed bolted joints, "
               "friction classes; as reproduced in Bossard, Friction and friction "
               "coefficients, technical sheet 57, 2025",
    'shigley': "R. G. Budynas, J. K. Nisbett, Shigley's Mechanical Engineering Design, "
               "McGraw-Hill, Table 16-3",
    'din7190': "DIN 7190:2001, Interference fits, coefficients of adhesion; as reproduced in "
               "the eAssistant manual, ch. 15.8",
}

# (static dry, static lubricated, sliding dry, sliding lubricated)
_PAIRS = {
    ('hard steel', 'hard steel'): (0.78, (0.11, 0.23), 0.42, (0.029, 0.12)),
    ('mild steel', 'mild steel'): (0.74, None, 0.57, (0.09, 0.19)),
    ('cast iron', 'cast iron'): (1.10, None, 0.15, (0.064, 0.070)),
    ('brass', 'mild steel'): (0.51, None, 0.44, None),
    ('copper', 'mild steel'): (0.53, None, 0.36, 0.18),
    ('aluminium', 'mild steel'): (0.61, None, 0.47, None),
    ('aluminium', 'aluminium'): (1.05, None, 1.4, None),
    ('oak', 'oak'): (0.62, None, 0.48, (0.067, 0.164)),       # along the grain
    ('leather', 'oak'): (0.61, None, 0.52, None),
}
_KEYS = ('static_dry', 'static_lub', 'sliding_dry', 'sliding_lub')
PAIRS = {k: dict(zip(_KEYS, v)) for k, v in _PAIRS.items()}

# range of mu_G and mu_K, and the typical surfaces and lubricants of each class
BOLT_CLASSES = {
    'A': ((0.04, 0.10), "bright, black oxide, phosphated, zinc or zinc flake coated; solid "
                        "lubricants (MoS2, graphite, PTFE) in lacquers or pastes, waxes"),
    'B': ((0.08, 0.16), "as A, also Al and Mg alloys; solid lubricants, waxes, greases, oils "
                        "or as delivered; hot-dip galvanized with MoS2 or wax; stainless "
                        "steel with solid lubricant or wax"),
    'C': ((0.14, 0.24), "stainless steel with wax or paste; bright or phosphated as delivered "
                        "(lightly oiled); zinc coated without lubricant"),
    'D': ((0.20, 0.35), "stainless steel with oil; zinc coated or hot-dip galvanized without "
                        "lubricant"),
    'E': ((0.30, None), "zinc-iron or zinc-nickel coated, stainless steel, Al and Mg alloys, "
                        "all without lubricant"),
}

# friction coefficient, and the use; Table 16-3
LININGS = {
    'cermet': (0.32, 'brakes and clutches'),
    'sintered metal, dry': ((0.29, 0.33), 'clutches, caliper disc brakes'),
    'sintered metal, wet': ((0.06, 0.08), 'clutches'),
    'rigid molded non-asbestos': ((0.33, 0.63), 'clutches and brakes'),
    'woven cotton': (0.47, 'industrial clutches and brakes'),
    'resilient paper, wet': ((0.09, 0.15), 'clutches, transmission bands'),
}

# shrink and oil-pressure fits: one coefficient for sliding in either direction
FITS_SHRINK = {
    ('steel', 'steel', 'oil pressure, mineral oil'): 0.12,
    ('steel', 'steel', 'oil pressure, degreased, glycerine'): 0.18,
    ('steel', 'steel', 'shrunk'): 0.14,
    ('steel', 'steel', 'shrunk, degreased'): 0.20,
    ('steel', 'cast iron', 'oil pressure, mineral oil'): 0.10,
    ('steel', 'cast iron', 'oil pressure, degreased'): 0.16,
    ('steel', 'MgAl', 'dry'): (0.10, 0.25),
    ('steel', 'CuZn', 'dry'): (0.17, 0.25),
}
# pressed fits, hardened steel shaft (X210CrW12) in the hub material:
# (hold dry, press dry, hold lubricated, press lubricated)
FITS_PRESSED = {
    'S235': (0.10, 0.09, 0.07, 0.06),
    'E335': (0.11, 0.08, 0.08, 0.07),
    'cast steel GE300': (0.11, 0.08, 0.08, 0.07),
    'grey iron EN-GJL-250': (0.12, 0.11, 0.06, 0.05),
    'ductile iron EN-GJS-600': (0.10, 0.09, 0.06, 0.05),
    'aluminium EN AB-44000': (0.07, 0.06, 0.05, 0.04),
    'bronze CuPb10Sn': (0.07, 0.06, None, None),
}


def pair(a: str, b: str) -> dict:
    """Coefficients of the material pair ``a`` on ``b``, in either order."""
    key = (a.lower(), b.lower())
    for k in (key, key[::-1]):
        if k in PAIRS:
            return PAIRS[k]
    raise KeyError(f"no data for {a} on {b}; pairs: {sorted(PAIRS)}")


def bolt_class(letter: str) -> tuple:
    """The range ``(low, high)`` of mu_G and mu_K in the VDI 2230 friction class."""
    return BOLT_CLASSES[letter.upper()][0]


def classify_bolt(mu: float) -> list:
    """The VDI 2230 classes whose range holds ``mu``; the ranges overlap."""
    return [c for c, ((lo, hi), _) in BOLT_CLASSES.items()
            if lo <= mu and (hi is None or mu <= hi)]


def pressed_fit(hub: str, lubricated: bool = False) -> dict:
    """Slip coefficients of a pressed fit: ``'hold'`` against loosening, ``'press'`` for
    the press-in force."""
    v = FITS_PRESSED[hub]
    return dict(zip(('hold', 'press'), v[2:] if lubricated else v[:2]))


def _num(v) -> str:
    """At least two decimals, as the sources print them: 0.10, 1.10, but 0.029."""
    return f'{v:.2f}' if round(v, 2) == v else f'{v:g}'


def _fmt(v) -> str:
    """A coefficient as table text: a range with an en dash, nothing for an unknown."""
    if v is None:
        return ''
    if isinstance(v, tuple):
        lo, hi = v
        return f'≥ {_num(lo)}' if hi is None else f'{_num(lo)}–{_num(hi)}'
    return _num(v)


_NAMES = {('oak', 'oak'): 'Oak on oak, along the grain'}


def _name(m):
    return 'PTFE' if m == 'ptfe' else m


def _cap(t):
    return t[:1].upper() + t[1:]


def _rows(name):
    if name == 'pairs':
        head = ['Surfaces', r'$\mu_s$, dry', r'$\mu_s$, lubricated', r'$\mu_k$, dry',
                r'$\mu_k$, lubricated']
        rows = [[_NAMES.get((a, b), _cap(f'{_name(a)} on {_name(b)}'))]
                + [_fmt(d[k]) for k in _KEYS] for (a, b), d in PAIRS.items()]
        return head, rows, 'marks'
    if name == 'bolts':
        head = ['Class', r'$\mu_G$ and $\mu_K$', 'Typical surfaces and lubricants']
        return head, [[c, _fmt(r), t] for c, (r, t) in BOLT_CLASSES.items()], 'vdi2230'
    if name == 'linings':
        head = ['Lining', r'$\mu$', 'Use']
        return head, [[_cap(m), _fmt(v), u] for m, (v, u) in LININGS.items()], 'shigley'
    if name == 'shrink':
        head = ['Shaft', 'Hub', 'Joining', r'$\mu$']
        return head, [[s, h, j, _fmt(v)] for (s, h, j), v in FITS_SHRINK.items()], 'din7190'
    if name == 'pressed':
        head = ['Hub', 'hold, dry', 'press, dry', 'hold, lubricated', 'press, lubricated']
        return head, [[h] + [_fmt(x) for x in v] for h, v in FITS_PRESSED.items()], 'din7190'
    raise KeyError(f"no table {name!r}; tables: pairs, bolts, linings, shrink, pressed")


def table(name: str = 'pairs', source: bool = True):
    """One of the tables as Markdown, displayed as a table in Jupyter, Quarto and marimo.

    ``name`` is ``'pairs'``, ``'bolts'``, ``'linings'``, ``'shrink'`` or ``'pressed'``.
    ``source=False`` leaves out the line naming the source, for a document that cites it
    in its own caption.

        >>> print(friction.table('linings', source=False))      # doctest: +SKIP
    """
    from .markdown import Markdown
    head, rows, key = _rows(name)
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '---|'*len(head)]
    lines += ['| ' + ' | '.join(r) + ' |' for r in rows]
    if source:
        lines += ['', f'Source: {SOURCES[key]}.']
    return Markdown('\n'.join(lines))


SOURCES_SHORT = {
    'marks': "Marks' Handbook (1978)",
    'vdi2230': 'VDI 2230 Part 1 (2015)',
    'shigley': "Budynas and Nisbett, Shigley's Mechanical Engineering Design, Table 16-3",
    'din7190': 'DIN 7190 (2001)',
}


def materials() -> list:
    """Every material that appears in ``PAIRS``, in display form ('PTFE', 'Mild steel')."""
    seen = []
    for a, b in PAIRS:
        for m in (a, b):
            if _cap(_name(m)) not in seen:
                seen.append(_cap(_name(m)))
    return seen


def partners(material: str) -> list:
    """The materials that ``material`` has data against, in display form."""
    m = material.lower()
    out = [b if a == m else a for a, b in PAIRS if m in (a, b)]
    return [_cap(_name(x)) for x in dict.fromkeys(out)]
