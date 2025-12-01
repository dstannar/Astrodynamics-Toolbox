import numpy as np

def time_to_perigee(orbit):
    '''
    Input
        orbit: orbit object
    Output
        t_toPer: time to next perigee (s)
    '''
    t_sincePer = orbit.period / (2*np.pi) * (orbit.EA - orbit.ecc * np.sin(orbit.EA))
    t_toPer = orbit.period - t_sincePer
    return t_toPer