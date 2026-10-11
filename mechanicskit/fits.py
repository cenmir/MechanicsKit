"""ISO 286 limits and fits on the hole basis: tolerance grades, fundamental deviations of
shafts, and the limits of size and the interference of a fit such as 60 H7/s6.

All deviations are in micrometres, sizes in millimetres, for basic sizes up to 400 mm.
The tolerance grades IT6 to IT11 and the fundamental deviations of the shaft letters c, d,
f, g, h, k, n, p, s and u are those of Budynas and Nisbett, *Shigley's Mechanical
Engineering Design*, Tables A-11 and A-12, which reproduce ISO 286-2. IT5, IT12, IT13 and
the letter m are added from ISO 286. Holes are always class H (the hole-basis system):
the lower deviation is zero and the upper deviation the tolerance grade.

    >>> from mechanicskit.fits import limits, interference
    >>> limits('H7/s6', 60)          # ((EI, ES), (ei, es)) in um
    ((0, 30), (53, 72))
    >>> interference('H7/s6', 60)    # smallest and largest, negative is clearance
    (23, 72)
"""
from __future__ import annotations

# upper limits of the size ranges for the tolerance grades: "over the previous, up to and
# including this one", mm
IT_TOP = [3, 6, 10, 18, 30, 50, 80, 120, 180, 250, 315, 400]
IT = {5: [4, 5, 6, 8, 9, 11, 13, 15, 18, 20, 23, 25],
      6: [6, 8, 9, 11, 13, 16, 19, 22, 25, 29, 32, 36],
      7: [10, 12, 15, 18, 21, 25, 30, 35, 40, 46, 52, 57],
      8: [14, 18, 22, 27, 33, 39, 46, 54, 63, 72, 81, 89],
      9: [25, 30, 36, 43, 52, 62, 74, 87, 100, 115, 130, 140],
      10: [40, 48, 58, 70, 84, 100, 120, 140, 160, 185, 210, 230],
      11: [60, 75, 90, 110, 130, 160, 190, 220, 250, 290, 320, 360],
      12: [100, 120, 150, 180, 210, 250, 300, 350, 400, 460, 520, 570],
      13: [140, 180, 220, 270, 330, 390, 460, 540, 630, 720, 810, 890]}

# the finer size ranges of the fundamental deviations, mm
FD_TOP = [3, 6, 10, 14, 18, 24, 30, 40, 50, 65, 80, 100, 120, 140, 160, 180, 200, 225, 250,
          280, 315, 355, 400]
# fundamental deviation of each shaft letter, um: the upper deviation es for c to h (below
# the basic size, clearance), the lower deviation ei for k to u (above it)
FD = {'c': [-60, -70, -80, -95, -95, -110, -110, -120, -130, -140, -150, -170, -180, -200,
            -210, -230, -240, -260, -280, -300, -330, -360, -400],
      'd': [-20, -30, -40, -50, -50, -65, -65, -80, -80, -100, -100, -120, -120, -145, -145,
            -145, -170, -170, -170, -190, -190, -210, -210],
      'f': [-6, -10, -13, -16, -16, -20, -20, -25, -25, -30, -30, -36, -36, -43, -43, -43,
            -50, -50, -50, -56, -56, -62, -62],
      'g': [-2, -4, -5, -6, -6, -7, -7, -9, -9, -10, -10, -12, -12, -14, -14, -14, -15, -15,
            -15, -17, -17, -18, -18],
      'h': [0]*23,
      'k': [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4],
      'm': [2, 4, 6, 7, 7, 8, 8, 9, 9, 11, 11, 13, 13, 15, 15, 15, 17, 17, 17, 20, 20, 21, 21],
      'n': [4, 8, 10, 12, 12, 15, 15, 17, 17, 20, 20, 23, 23, 27, 27, 27, 31, 31, 31, 34, 34,
            37, 37],
      'p': [6, 12, 15, 18, 18, 22, 22, 26, 26, 32, 32, 37, 37, 43, 43, 43, 50, 50, 50, 56, 56,
            62, 62],
      's': [14, 19, 23, 28, 28, 35, 35, 43, 43, 53, 59, 71, 79, 92, 100, 108, 122, 130, 140,
            158, 170, 190, 208],
      'u': [18, 23, 28, 33, 33, 41, 48, 60, 70, 87, 102, 124, 144, 170, 190, 210, 236, 258,
            284, 315, 350, 390, 435]}
CLEARANCE_LETTERS = 'cdfgh'

# the preferred hole-basis fits (Shigley Table 7-9, with H7/m6 for dowel pins added):
# designation, character, typical uses
PREFERRED_FITS = [
    ('H11/c11', 'loose running', 'hinge pins of farm and construction machinery, clevis pins, '
     'linkages that tolerate play'),
    ('H9/d9', 'free running', 'idler and loose pulleys, plain bearings at high speed or with '
     'large temperature changes'),
    ('H8/f7', 'close running', 'shafts in plain bearings of gearboxes, pumps and machine tools'),
    ('H7/g6', 'sliding', 'sliding gears and clutch members, guide pins and spigots that must '
     'locate accurately'),
    ('H7/h6', 'locational clearance', 'gears and pulleys on keyed shafts that must come off, '
     'flange spigots, pins in removable covers'),
    ('H7/k6', 'locational transition', 'pulleys and couplings on keyed shafts, tapped on with a '
     'mallet'),
    ('H7/m6', 'transition', 'hardened dowel pins (ISO 8734) in reamed holes, located parts that '
     'are seldom taken apart'),
    ('H7/n6', 'locational transition', 'bushes and gears that must not move, pressed in, '
     'armatures on motor shafts'),
    ('H7/p6', 'locational interference', 'bushings and sleeves pressed into housings, valve '
     'seats'),
    ('H7/s6', 'medium drive', 'gears, couplings and flanges fixed for good by press or shrink '
     'fit; the tightest fit for cast iron'),
    ('H7/u6', 'force fit', 'shrink fits that carry large torques by friction alone: gear rims, '
     'built-up crankshafts'),
]


def _size_range(tops, d):
    if not 0 < d <= tops[-1]:
        raise ValueError(f'basic size {d} mm is outside 0 to {tops[-1]} mm')
    return next(i for i, top in enumerate(tops) if d <= top)


def tolerance(grade, d):
    """The standard tolerance ITgrade at the basic size d, in um."""
    if grade not in IT:
        raise ValueError(f'IT{grade} is not tabulated; available: IT{min(IT)} to IT{max(IT)}')
    return IT[grade][_size_range(IT_TOP, d)]


def hole_deviations(cls, d):
    """(EI, ES) in um of a hole of class H<grade>, e.g. 'H7'."""
    if cls[0] != 'H':
        raise ValueError('only H holes (the hole-basis system) are tabulated')
    return 0, tolerance(int(cls[1:]), d)


def shaft_deviations(cls, d):
    """(ei, es) in um of a shaft of class <letter><grade>, e.g. 's6'."""
    letter, grade = cls[0], int(cls[1:])
    if letter not in FD:
        raise ValueError(f'shaft letter {letter!r} is not tabulated; available: {"".join(FD)}')
    t = tolerance(grade, d)
    dF = FD[letter][_size_range(FD_TOP, d)]
    if letter == 'k' and not 4 <= grade <= 7:          # ISO 286: k is +dF only for IT4 to IT7
        dF = 0
    if letter in CLEARANCE_LETTERS:
        return dF - t, dF
    return dF, dF + t


def limits(fit, d):
    """Deviations of hole and shaft, ((EI, ES), (ei, es)) in um, for a fit such as 'H7/s6'
    at the basic size d in mm. The limits of size are d + deviation/1000."""
    hole, shaft = fit.split('/')
    return hole_deviations(hole, d), shaft_deviations(shaft, d)


def interference(fit, d):
    """Smallest and largest interference of a fit in um, (d_min - D_max, d_max - D_min).
    A negative value is a clearance."""
    (EI, ES), (ei, es) = limits(fit, d)
    return ei - ES, es - EI


def kind(fit, d):
    """'clearance', 'transition' or 'interference'."""
    lo, hi = interference(fit, d)
    return 'clearance' if hi <= 0 else 'interference' if lo >= 0 else 'transition'


# processes and the tolerance grades they reach in normal production: a rule of thumb from
# workshop practice; a careful operator on a good machine does better, a worn machine worse
PROCESSES = [
    ('honing, lapping', 4, 5),
    ('grinding', 5, 7),
    ('reaming, fine boring', 6, 8),
    ('turning, boring', 7, 11),
    ('milling', 8, 11),
    ('drilling', 10, 13),
    ('fused-filament printing, as printed', 12, 14),
]


def processes(grade):
    """The processes of PROCESSES that reach the tolerance grade ITgrade, coarsest first:
    the cheapest way to make the part is the first one."""
    ok = [(name, lo, hi) for name, lo, hi in PROCESSES if lo <= grade]
    return [name for name, lo, hi in sorted(ok, key=lambda p: -p[1])]
