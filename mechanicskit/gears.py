"""Load capacity of spur gears: the torque the tooth root and the tooth flank carry, the
two lines of a load diagram (Swedish: belastningsdiagram).

The flank limit is the Hertz contact pressure at the pitch point, written as in ISO 6336-2
with the elasticity factor Z_E and the zone factor Z_H; the root limit is Lewis's
bending stress with a stress concentration K_t for the fillet. Sizes in mm, stresses in
MPa, torques in N m. No dynamic, load-distribution or life factors: those belong to the
allowable stresses the caller passes in.

    >>> from mechanicskit.gears import flank_torque, root_torque, iso6336_normalised
    >>> s_H, s_F = iso6336_normalised(179)             # AISI 1045 cold drawn, 179 HB
    >>> round(float(flank_torque(20, 2, 20, s_H)), 2)   # 20 teeth, module 2, b = 20 mm
    4.77
"""
from __future__ import annotations

import numpy as np

# Lewis form factor Y for 20° full-depth teeth: Budynas and Nisbett, Shigley's Mechanical
# Engineering Design, 10th ed., Table 14-2 (Rack taken as 10 000 teeth)
LEWIS_Y = {12: 0.245, 13: 0.261, 14: 0.277, 15: 0.290, 16: 0.296, 17: 0.303, 18: 0.309,
           19: 0.314, 20: 0.322, 21: 0.328, 22: 0.331, 24: 0.337, 26: 0.346, 28: 0.353,
           30: 0.359, 34: 0.371, 38: 0.384, 43: 0.397, 50: 0.409, 60: 0.422, 75: 0.435,
           100: 0.447, 150: 0.460, 300: 0.472, 400: 0.480, 10000: 0.485}


def lewis_form_factor(N):
    """Lewis form factor Y of a 20° full-depth tooth with N teeth (12 or more),
    interpolated in Shigley's Table 14-2."""
    N = np.asarray(N, float)
    if np.any(N < 12):
        raise ValueError('the table starts at 12 teeth')
    return np.interp(N, list(LEWIS_Y), list(LEWIS_Y.values()))


def zone_factor(phi_deg=20.0):
    """Z_H = sqrt(2/(sin phi cos phi)), the curvature of the flanks at the pitch point."""
    phi = np.radians(phi_deg)
    return np.sqrt(2/(np.sin(phi)*np.cos(phi)))


def elasticity_factor(E1, nu1, E2=None, nu2=None):
    """Z_E = sqrt(1/(pi ((1 - nu1^2)/E1 + (1 - nu2^2)/E2))) in sqrt(MPa); the second gear
    defaults to the first material."""
    E2 = E1 if E2 is None else E2
    nu2 = nu1 if nu2 is None else nu2
    return np.sqrt(1/(np.pi*((1 - nu1**2)/E1 + (1 - nu2**2)/E2)))


def flank_torque(N, m, b, sigma_HP, E=210e3, nu=0.3, u=1.0, phi_deg=20.0, E2=None, nu2=None):
    """Pinion torque [N m] at which the Hertz pressure at the pitch point reaches sigma_HP:
    T = sigma_HP^2 b d^2 u / (2 Z_E^2 Z_H^2 (u + 1)), d = m N, u = N_gear/N_pinion."""
    d = m*np.asarray(N, float)
    Z_E, Z_H = elasticity_factor(E, nu, E2, nu2), zone_factor(phi_deg)
    return sigma_HP**2*b*d**2*u/(2*Z_E**2*Z_H**2*(u + 1))/1e3


def root_torque(N, m, b, sigma_FP, K_t=1.5):
    """Pinion torque [N m] at which the local root stress K_t F_t/(b m Y) reaches sigma_FP."""
    N = np.asarray(N, float)
    return sigma_FP*b*m**2*N*lewis_form_factor(N)/(2*K_t)/1e3


def load_limit(N, m, b, sigma_HP, sigma_FP, K_t=1.5, **material):
    """The torque the gear carries, the smaller of flank_torque and root_torque, and which
    one governs ('flank' or 'root')."""
    T_H = flank_torque(N, m, b, sigma_HP, **material)
    T_F = root_torque(N, m, b, sigma_FP, K_t)
    return np.minimum(T_H, T_F), np.where(T_H <= T_F, 'flank', 'root')


def iso6336_normalised(HB):
    """(sigma_Hlim, local root endurance 2 sigma_Flim) in MPa of normalised steel of
    hardness HB (110 to 210), normal quality ML/MQ: ISO 6336-5:2016, Table 1. sigma_Flim is
    a nominal stress of the test gear that contains a stress concentration of 2, hence the 2."""
    if not 110 <= HB <= 210:
        raise ValueError('ISO 6336-5 gives normalised steel for 110 to 210 HB')
    return HB + 190.0, 2*(0.455*HB + 69.0)
