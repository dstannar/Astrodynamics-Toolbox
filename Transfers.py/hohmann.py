import numpy as np
from MathHelpers.constants import muE, rE

def hohmann(zai, zaf, zpi=None, zpf=None, m = None, Isp = None, mu=muE, g0=9.81, rB=rE):
    
    if zpi and zpf != None: # elliptical starting orbit
        rai = zai + rB
        rpi = zpi + rB
        raf = zaf + rB
        rpf = zpf + rB
    else: # circular starting orbit
        rai = zai + rB
        rpi = zai + rB
        raf = zaf + rB
        rpf = zaf + rB

    h1 = np.sqrt(2*mu) * np.sqrt(rai*rpi/(rai+rpi))
    h2 = np.sqrt(2*mu) * np.sqrt(raf*rpi/(raf+rpi))
    h3 = np.sqrt(2*mu) * np.sqrt(raf*rpf/(raf+rpf))
    print(h1, h2, h3)

    vp1 = h1/rpi
    vp2 = h2/rpi
    dvp = vp2-vp1

    va1 = h2/raf
    va2 = h3/raf
    dva = va2-va1

    dvTot = np.abs(dvp) + np.abs(dva)

    a_tran = (raf+rpi)/2
    T_tran = 2*np.pi / np.sqrt(mu) * a_tran**(3/2)
    tof = T_tran/2

    if m != None:
        dm = (1 - np.e ** (-dvTot*1e3 / (Isp*g0))) * m
        return dvp, dva, dvTot, tof, dm
    else:
        return dvp, dva, dvTot, tof
    