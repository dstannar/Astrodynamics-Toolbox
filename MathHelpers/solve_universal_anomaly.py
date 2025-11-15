import numpy as np
from MathHelpers.constants import muE
from MathHelpers.stumpff import stumpffC, stumpffS

def solve_universal_anomaly(dt, r0, v0, mu=muE):
    # solver setup
    import numpy as np
    tol = 1e-10
    max_iter = 30
    ratio = 1
    iters = 0

    # init math steps
    r0mag = np.linalg.norm(r0)
    v0mag = np.linalg.norm(v0)
    vr0 = np.dot(v0, r0) / r0mag
    a = 1 / (2/r0mag - v0mag**2 / mu)
    alpha = 1/a
    chi_new = np.sqrt(mu) * np.abs(alpha) * dt

    while ratio > tol and iters < max_iter:
        z_new = alpha * chi_new**2
        Sz = stumpffS(z_new)
        Cz = stumpffC(z_new)
        fChi = r0mag*vr0 / np.sqrt(mu) * chi_new**2 * Cz + (1-alpha*r0mag)*chi_new**3*Sz+r0mag*chi_new-np.sqrt(mu)*dt
        fdotChi = r0mag*vr0 / np.sqrt(mu) * chi_new * (1-alpha*chi_new**2*Sz) + (1-alpha*r0mag)*chi_new**2*Cz+r0mag
        ratio = fChi / fdotChi
        chi_new = chi_new - ratio

    return chi_new