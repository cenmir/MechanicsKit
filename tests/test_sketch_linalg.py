"""Smoke tests of the linear algebra figure parts of mechanicskit.sketch_extra."""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

from mechanicskit import sketch as sk
from mechanicskit.sketch_extra import (Proj, grid, matrix_table, plane_axes, region, rot3,
                                       save_svg)


def test_rot3_turns_x_to_y_about_z():
    assert np.allclose(rot3(2, 90) @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(rot3(0, 90) @ [0, 1, 0], [0, 0, 1])
    assert np.allclose(rot3(1, 90) @ [0, 0, 1], [1, 0, 0])


def test_matrix_table_geometry():
    fig, ax = sk.canvas((4, 3))
    m = matrix_table(ax, [[1, -2, 3], [4, 5, 6]], (0, 0), 0.6, 0.5, gap=0.05)
    assert m.left == pytest.approx(-0.05) and m.right == pytest.approx(1.85)
    assert m.top == 0 and m.bottom == pytest.approx(-1.0)
    assert np.allclose(m.cell(1, 2), (1.5, -0.75))
    c = matrix_table(ax, [[1, 2], [3, 4]], (5, 0), 0.6, centred=True, delim='|',
                     struck=(0, 1), fills={(1, 0): '#ddd'}, name='$-4$')
    assert np.allclose(c.middle, (5, 0))


def test_parts_save(tmp_path):
    fig, ax = sk.canvas((4, 3))
    grid(ax, (-1, 3), (-1, 2))
    region(ax, [(0, 0), (1, 0), (1, 1), (0, 1)], sk.GREEN, 0.2, ls='--')
    plane_axes(ax, (-1, -1), (3, 2), ticks=True)
    P = Proj.from_view((1, 0.5, 0.4))
    P.axes(ax, (0, 0, 0), (1, 2, 1.5), neg=0.5, fit=True)
    P.coordinate_lines(ax, (1, 1, 1))
    out = tmp_path/'f.svg'
    save_svg(fig, out)
    assert out.stat().st_size > 1000
