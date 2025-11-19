import numpy as np
from Transfers.Lambert import Lambert
from Transfers.hohmann import hohmann

# leading flyby: decr heliocentric speed, turn angle pos
# trailing flyby: incr heliocentric speed, turn angle neg


v_sc1 = [v_scx, v_scy] # in Uv and Us direction

v_sc1_V = v_sc1 * np.cos(alpha1)
v_sc1_S = v_sc1 * np.sin(alpha1)


v_sc1_V = muS / h1 * (1 + ecc1 * np.cos(TA1))
v_sc1_S = -muS / h1 * ecc1 * np.sin(TA1)

V_inf = np.sqrt(np.dot(vinf1, vinf2))

# for hohmann, ph1 = 0 or 180 deg
phi1 = np.atan2(vinf1_s, vinf1_v)

# in front of +, behind -
phi2 = phi1 +- turnAngle

turnAngle = 2 * np.asin(1/ecc_hyp)

vinf2 = [vinf * np.cos(phi2), + vinf * np.sin(phi2)]

Vsc2 = Vplanet + vinf2


dV_imparted = Vsc2 - v_sc1

def flyby(flybyAlt, leadingFlyby, muPlan, muS = muS):
    muPlan = muJup
    r_jup_sun = 778.6e6 # km
    r_sun_earth = 149.6e6 # km
    rJup = 71490 # km

    vinf = hohmann(r_sun_earth, r_jup_sun)

    ecc1 = 0
    h1 = np.sqrt(muS * rE * (1 - ecc1))

    











def planetary_departure():

def planetary_arrival():