"""A worked example of a problem figure and its free body diagram, redrawn as SVG.

The problem: a uniform bar AB of length L is pinned to the wall at A and held at
an angle theta above the horizontal by a spring from B to the ceiling at C. A
force P hangs from B. The setup and the free body diagram below are drawn from
those engineering parameters, never traced from the sketch.

    bar_spring.svg        the problem as stated, with dimensions and the angle
    bar_spring_fbd.svg    the bar cut free: the pin reactions, the spring force, P and mg

    python bar_spring_figures.py [name ...]      # writes next to this script
"""
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mechanicskit import sketch as sk
from mechanicskit.sketch_extra import TAN_STOPS, dot, ground, save_svg, shaded_bar

OUT = pathlib.Path(__file__).resolve().parent
FS = 14                                     # one label size for every figure in a set

# the problem's own numbers, in drawing units: the bar is 4 units for L = 2 m
L = 4.0
TH = np.radians(30)
A = np.array([0.0, 0.0])
B = A + L*np.array([np.cos(TH), np.sin(TH)])
C = np.array([B[0], 3.4])                   # the spring hangs straight up from B


def _save(fig, name):
    save_svg(fig, OUT / name)
    plt.close(fig)


def setup():
    fig, ax = sk.canvas((5.6, 4.4))
    # fixed surfaces: walk along each so the solid lies on the right
    ground(ax, [(0, 3.6), (0, -0.9)], depth=0.3)                  # the wall at A, solid to the left
    ground(ax, [(B[0] + 0.8, C[1]), (B[0] - 0.8, C[1])], depth=0.3)   # the ceiling at C, solid above
    shaded_bar(ax, A, B, 0.22, stops=TAN_STOPS, zorder=3)
    sk.spring(ax, B, C, coils=7, width=0.3, zorder=2)
    sk.pin(ax, A, r=0.07)
    sk.pin(ax, B, r=0.07)
    sk.force(ax, B, (0, -1), 1.1, head=False)
    sk.label(ax, B + (0.3, -0.9), '$P$', fontsize=FS)
    # construction and dimensions sit outside the parts they measure
    sk.guide(ax, A, (2.0, 0))
    sk.angle(ax, A, 1.3, 0, np.degrees(TH), r'$\theta$', text_r=1.6, fontsize=FS)
    sk.dimension(ax, A, B, 0.5, '$L$', fontsize=FS, gap=0.1)
    sk.label(ax, A + (0.3, -0.35), '$A$', fontsize=FS)
    sk.label(ax, B + (0.35, 0.2), '$B$', fontsize=FS)
    sk.label(ax, C + (-0.4, -0.2), '$C$', fontsize=FS)
    sk.label(ax, (B + C)/2 + (0.45, 0), '$k$', fontsize=FS)
    ax.set_xlim(-0.6, 4.4)
    ax.set_ylim(-1.3, 3.9)
    _save(fig, 'bar_spring.svg')


def fbd():
    fig, ax = sk.canvas((5.2, 4.2))
    shaded_bar(ax, A, B, 0.22, stops=TAN_STOPS, zorder=3)
    G = (A + B)/2
    dot(ax, G, 4, zorder=4)
    # every force on the cut-free body, red, tail at its point of action
    # the unknown reactions point in the positive axis directions and end at A
    sk.force(ax, A, (1, 0), 1.0, head=True)
    sk.label(ax, A + (-0.6, -0.3), '$A_x$', fontsize=FS)
    sk.force(ax, A, (0, 1), 1.0, head=True)
    sk.label(ax, A + (0.35, -0.7), '$A_y$', fontsize=FS)
    sk.force(ax, B, (0, 1), 1.2)
    sk.label(ax, B + (0.35, 0.9), '$F_s$', fontsize=FS)
    sk.force(ax, B, (0, -1), 1.1)
    sk.label(ax, B + (0.3, -0.9), '$P$', fontsize=FS)
    sk.force(ax, G, (0, -1), 1.0)
    sk.label(ax, G + (0.35, -0.8), '$mg$', fontsize=FS)
    sk.label(ax, A + (-0.35, 0.35), '$A$', fontsize=FS)
    sk.label(ax, B + (-0.4, 0.25), '$B$', fontsize=FS)
    sk.guide(ax, A, (1.8, 0))
    sk.angle(ax, A, 0.9, 0, np.degrees(TH), r'$\theta$', text_r=1.2, fontsize=FS)
    ax.set_xlim(-1.4, 4.4)
    ax.set_ylim(-1.4, 3.8)
    _save(fig, 'bar_spring_fbd.svg')


FIGURES = {'setup': setup, 'fbd': fbd}

if __name__ == '__main__':
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
