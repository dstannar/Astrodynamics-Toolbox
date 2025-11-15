import numpy as np
from MathHelpers.constants import muE

def phasing_maneuver(ra1, rp1, thetaB, thetaC, delta_apse=0, mu=muE):
    # orbit-1 elements
    a1  = 0.5*(rp1 + ra1)
    e1  = (ra1 - rp1)/(ra1 + rp1)
    n1  = np.sqrt(mu/a1**3)

    # radius at B on orbit 1
    rB  = a1*(1 - e1**2)/(1 + e1*np.cos(thetaB))

    fac = np.sqrt((1 - e1)/(1 + e1))
    E_B = 2.0*np.arctan(fac*np.tan(thetaB/2.0))
    
    while E_B < 0:
        E_B = E_B + 2*np.pi

    E_C = 2.0*np.arctan(fac*np.tan(thetaC/2.0))
    
    while E_C < 0:
        E_C = E_C + 2*np.pi
    
    # get mean anomalies
    M_B = E_B - e1*np.sin(E_B)
    M_C = E_C - e1*np.sin(E_C)
    dM  = (M_B - M_C)

    # normalize dM into 0, 2pi
    dM = dM % (2*np.pi)

    T2  = dM / n1

    # phasing semimajor axis from period
    a2  = (mu**(1.0/3.0))*(T2/(2.0*np.pi))**(2.0/3.0)

    # anomaly of orbit-2
    theta2B = thetaB - delta_apse

    # solve for ecc2 @ taB
    Bcoef = rB*np.cos(theta2B)
    Ccoef = (rB - a2)
    disc  = Bcoef**2 - 4.0*a2*Ccoef

    e2a = (-Bcoef + np.sqrt(disc))/(2.0*a2)
    e2b = (-Bcoef - np.sqrt(disc))/(2.0*a2)
    if 0 < e2a < 1:
        e2 = e2a
    elif 0 < e2b < 1:
        e2 = e2b
    else:
        raise RuntimeError('bad :(')
    
    # speeds at B on both orbits
    h1  = np.sqrt(mu*a1*(1.0 - e1**2))
    h2  = np.sqrt(mu*a2*(1.0 - e2**2))
    vt1 = h1 / rB
    vr1 = (mu/h1)*e1*np.sin(thetaB)
    vt2 = h2 / rB
    vr2 = (mu/h2)*e2*np.sin(theta2B)

    # dv at b and total
    dv1 = np.sqrt((vt2 - vt1)**2 + (vr2 - vr1)**2)
    # symmetric maneuver
    dv2 = dv1
    dv_total = dv1 + dv2
    return dv_total
