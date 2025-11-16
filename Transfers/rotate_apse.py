import numpy as np
from MathHelpers.constants import muE

def rotate_apse(ra1, rp1, theta1, eta, e2, mu=muE):
    # get orbit geometry
    a1  = 0.5*(rp1 + ra1)
    ecc1  = (ra1 - rp1)/(ra1 + rp1)
    p1  = a1*(1.0 - ecc1**2)
    h1  = np.sqrt(mu * p1)

    # state at burn point on orbit 1
    c1  = np.cos(theta1); s1 = np.sin(theta1)
    r   = p1 / (1 + ecc1 * c1)
    vt1 = h1 / r
    vr1 = (mu / h1) * ecc1 * s1

    # anomaly on orbit 2 at same point
    theta2 = theta1 - eta
    c2 = np.cos(theta2); s2 = np.sin(theta2)

    # solve for tang burn
    B = 2 * h1
    C = h1**2 - mu * r * (1 + e2 * c2)
    disc = B**2 - 4*C # discriminant of quadratic

    x1 = (-B + np.sqrt(disc)) / (2)
    x2 = (-B - np.sqrt(disc)) / (2)

    # init loop vars
    best_dv = None
    best_dgamma = None
    # check sols. for best (and realest) delta v
    for x in (x1, x2):
        h2  = h1 + x
        dvt = x / r
        dvr = (mu / h2) * e2 * s2 - vr1
        vt2 = vt1 + dvt
        vr2 = vr1 + dvr
        dv  = np.sqrt(dvt*dvt + dvr*dvr)
        dgamma = np.arctan2(vr2, vt2) - np.arctan2(vr1, vt1)
        dgamma = (dgamma + np.pi) % (2.0*np.pi) - np.pi  # wrap to -pi, pi 

        # update best dv
        if (best_dv is None) or (dv < best_dv):
            best_dv = dv
            best_dgamma = dgamma

    return best_dv, best_dgamma
