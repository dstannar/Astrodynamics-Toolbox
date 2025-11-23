import numpy as np
from MathHelpers.constants import muE
from Propagators.Propagate import Propagate

def raan_change(orbit1, orbit2):
    '''
    Input: 2 orbit like objects (TLEOrbit, KeplerianOrbit)
    Output:
        burn_loc: position vector of burn
        burn_vect: deltaV velocity vector
        burn_time: time of burn, measured since passed orbit1 TA
        dVmag: delta V magnitude (burn cost)
    '''

    print('not yet implemented')


def inc_change(orbit1, orbit2, dt=1, mu=muE):
    intersectionPos, intersectionVel, intersectionTime = best_nodal_crossing(orbit1, orbit2)
    
    # inc dV
    TA1 = orbit1.TA
    FPA1 = np.atan2(orbit1.ecc*np.sin(TA1), (1 + orbit1.ecc*np.cos(TA1)))
    
    # Vallado alg 39
    dInc = orbit2.inc - orbit1.inc
    dv_vect = 2 * intersectionVel * np.cos(FPA1) * np.sin(dInc / 2)

    burn_loc = intersectionPos
    burn_time = intersectionTime
    v_init = intersectionVel

    return burn_loc, burn_time, v_init, dv_vect
   



def nodal_crossings_array(orbit1, orbit2, dt_sample=1.0, tol=1e-8, max_refine=50):
    """
    Returns:
      [
        [orbit1, posCross1, velCross1, timeCross1],
        [orbit1, posCross2, velCross2, timeCross2],
        [orbit2, posCross1, velCross1, timeCross1],
        [orbit2, posCross2, velCross2, timeCross2],
      ]
    """
    h1_hat = orbit1.hvec / np.linalg.norm(orbit1.hvec)
    h2_hat = orbit2.hvec / np.linalg.norm(orbit2.hvec)
    if np.linalg.norm(np.cross(h1_hat, h2_hat)) < 1e-12:
        raise ValueError("Planes nearly coplanar — nodal crossings undefined.")

    rows = []

    for (orb, plane_hhat) in ((orbit1, h2_hat), (orbit2, h1_hat)):
        T = orb.period
        roots = []

        # initial sample
        t_prev = 0.0
        r_prev, v_prev = Propagate(0.0, Orbit=orb).lagrange_coeff()
        s_prev = float(np.dot(plane_hhat, r_prev))
        

        t = dt_sample
        while t <= T + 1e-12 and len(roots) < 2:
            r_cur, v_cur = Propagate(t, Orbit=orb).lagrange_coeff()
            s_cur = float(np.dot(plane_hhat, r_cur))

            if abs(s_cur) < tol:
                roots.append((t, r_cur, v_cur))

            elif s_prev * s_cur < 0.0:
                a, b, sa, sb = t - dt_sample, t, s_prev, s_cur
                r_m = v_m = None
                for _ in range(max_refine):
                    m = 0.5 * (a + b)
                    r_m, v_m = Propagate(m, Orbit=orb).lagrange_coeff()
                    sm = float(np.dot(plane_hhat, r_m))
                    if abs(sm) < tol:
                        a = b = m
                        break
                    if sa * sm <= 0.0:
                        b, sb = m, sm
                    else:
                        a, sa = m, sm
                t_root = 0.5 * (a + b)
                if r_m is None:
                    r_m, v_m = Propagate(t_root, Orbit=orb).lagrange_coeff()
                roots.append((t_root, r_m, v_m))

            t_prev, s_prev = t, s_cur
            t += dt_sample

        if len(roots) < 2:
            raise RuntimeError("Did not find two nodal crossings; adjust dt_sample/tol.")

        roots.sort(key=lambda x: x[0])
        rows.append([orb, roots[0][1], roots[0][2], roots[0][0]])
        rows.append([orb, roots[1][1], roots[1][2], roots[1][0]])

    return rows