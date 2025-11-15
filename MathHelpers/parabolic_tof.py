import numpy as np

# vallado's parabolic time of flight 
def parabolic_tof(r1, r2):
    c = float(np.linalg.norm(r2-r1))
    r1mag = np.linalg.norm(r1)
    r2mag = np.linalg.norm(r2)
    s = 0.5*(r1mag+r2mag+c)
    sc = s - c
    if s <= 0.0 or sc <= 0.0:
        return np.inf
    return (1.0 / 3.0) * np.sqrt(2.0) * (s**1.5 - sc**1.5)