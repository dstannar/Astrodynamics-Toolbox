import numpy as np
from MathHelpers.constants import muE
from MathHelpers.wrap_angles import wrap_to_2pi

def phasing_maneuver(ra, rp, TA1, TA2, delta_apse=0, mu=muE):
    # orbital elements
    a1  = 0.5*(rp + ra)
    e1  = (ra - rp)/(ra + rp)
    n1  = np.sqrt(mu/a1**3)

    # radius at TA1 on orbit 1
    r11  = a1*(1 - e1**2)/(1 + e1*np.cos(TA1))

    # eccentric anomaly at TA1 and TA2
    fac = np.sqrt((1 - e1)/(1 + e1))
    E_1 = 2.0*np.arctan(fac*np.tan(TA1/2.0))
    
    # get positive
    E_1 = wrap_to_2pi(E_1)

    E_2 = 2.0*np.arctan(fac*np.tan(TA2/2.0))
    
    # get positive
    E_2 = wrap_to_2pi(E_2)
    
    # get mean anomalies with kepler
    M_1 = E_1 - e1*np.sin(E_1)
    M_2 = E_2 - e1*np.sin(E_2)
    dM  = (M_1 - M_2)

    # normalize dM into 0, 2pi
    dM = wrap_to_2pi(dM)

    T2  = dM / n1

    # phasing semimajor axis from period
    a2  = (mu**(1.0/3.0))*(T2/(2.0*np.pi))**(2.0/3.0)

    # anomaly of orbit-2 at TA1
    TA21 = TA1 - delta_apse

    # solve for ecc2 @ taB
    Bcoef = r11*np.cos(TA21)
    Ccoef = (r11 - a2)
    disc  = Bcoef**2 - 4.0*a2*Ccoef

    e2a = (-Bcoef + np.sqrt(disc))/(2.0*a2)
    e2b = (-Bcoef - np.sqrt(disc))/(2.0*a2)
    if 0 < e2a < 1:
        e2 = e2a
    elif 0 < e2b < 1:
        e2 = e2b
    else:
        raise RuntimeError('bad :(')
    
    # speeds at TA1 on both orbits
    h1  = np.sqrt(mu*a1*(1.0 - e1**2))
    h2  = np.sqrt(mu*a2*(1.0 - e2**2))
    vt1 = h1 / r11
    vr1 = (mu/h1)*e1*np.sin(TA1)
    vt2 = h2 / r11
    vr2 = (mu/h2)*e2*np.sin(TA21)

    # dv at b and total
    dv1 = np.sqrt((vt2 - vt1)**2 + (vr2 - vr1)**2)
    # symmetric maneuver
    dv2 = dv1
    dv_total = dv1 + dv2
    return dv_total