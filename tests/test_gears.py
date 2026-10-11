import numpy as np
import pytest

from mechanicskit.gears import (elasticity_factor, flank_torque, iso6336_normalised,
                                lewis_form_factor, load_limit, root_torque, zone_factor)


def test_factors_match_iso_values():
    # steel on steel: Z_E = 189.8 sqrt(MPa) (ISO 6336-2), Z_H = 2.495 at 20 degrees
    assert elasticity_factor(206e3, 0.3) == pytest.approx(189.8, abs=0.2)
    assert zone_factor(20) == pytest.approx(2.495, abs=0.002)


def test_lewis_table_and_range():
    assert lewis_form_factor(20) == pytest.approx(0.322)
    assert lewis_form_factor(25) == pytest.approx((0.337 + 0.346)/2)
    with pytest.raises(ValueError):
        lewis_form_factor(10)


def test_flank_torque_scales_as_b_d_squared_and_with_ratio():
    T1 = flank_torque(20, 2, 20, 400)
    assert flank_torque(40, 2, 20, 400) == pytest.approx(4*T1)
    assert flank_torque(20, 2, 40, 400) == pytest.approx(2*T1)
    assert flank_torque(20, 2, 20, 400, u=3) == pytest.approx(T1*0.75/0.5)


def test_flank_torque_equals_hertz_pressure():
    # invert: the Hertz pressure of the returned torque is sigma_HP
    N, m, b, E, nu = 18, 3, 30, 210e3, 0.3
    T = flank_torque(N, m, b, 500, E, nu)*1e3
    d, phi = m*N, np.radians(20)
    F = 2*T/(d*np.cos(phi))
    rho = d/2*np.sin(phi)
    p = np.sqrt(F*E/(2*np.pi*b*(1 - nu**2))*(2/rho))
    assert p == pytest.approx(500)


def test_load_limit_picks_smaller():
    T, which = load_limit(np.array([12, 100]), 1, 15, 369, 215)
    assert which[0] == 'flank'
    assert T[0] == pytest.approx(flank_torque(12, 1, 15, 369))
    assert root_torque(12, 1, 15, 215) > T[0]


def test_iso6336_normalised():
    assert iso6336_normalised(179) == pytest.approx((369.0, 2*(0.455*179 + 69)))
    with pytest.raises(ValueError):
        iso6336_normalised(250)
