"""
canonical unit helper - normalizes mu to 1

user must define mu, DU
compute:
    TU = sqrt(DU^3 / mu)
    VU = DU / TU

ensure units btwn mu and DU are consistent
"""

import numpy as np
from MathHelpers.none_check import all_set, none_of


class CanonicalUnits:
    def __init__(self, DU=None, mu=None):
        '''
        Build canonical unit system with mu_canon = 1.

        Inputs:
            DU : float
                distance unit
            mu : float
                gravitational parameter in DU^3 / s^2 (so TU comes out in seconds)
        '''
        # dummy protections
        if not all_set(DU, mu):
            raise TypeError('define DU and mu')

        self.DU = float(DU)
        self.mu = float(mu)
        self.MU = 1.0  # canonical mu

        if self.DU <= 0.0:
            raise ValueError('DU must be positive')
        if self.mu <= 0.0:
            raise ValueError('mu must be positive')

        # TU = sqrt(DU^3 / mu) s.t. mu_canon = mu * TU^2 / DU^3 = 1
        self.TU = float(np.sqrt(self.DU**3 / self.mu))

        # VU = DU / TU
        self.VU = float(self.DU / self.TU)

    # time conversions
    def t_to_canon(self, t):
        '''
        Convert time to canonical units.
        Inputs:
            t : float
                time in seconds
        Outputs:
            t_c : float
                time in TU
        '''
        return float(t) / self.TU

    def t_to_si(self, t_c):
        '''
        Convert canonical time to seconds.
        Inputs:
            t_c : float
                time in TU
        Outputs:
            t : float
                time in seconds
        '''
        return float(t_c) * self.TU

    # position conversions
    def r_to_canon(self, r):
        '''
        Convert position to canonical units.
        Inputs:
            r : array-like 3x1
                position in DU-length units (km if DU is km)
        Outputs:
            r_c : ndarray
                nondimensional position in DU
        '''
        return np.asarray(r, dtype=float) / self.DU

    def r_to_si(self, r_c):
        '''
        Convert canonical position to length units.
        Inputs:
            r_c : array-like
                position in DU
        Outputs:
            r : ndarray
                position in DU-length units (km if DU is km)
        '''
        return np.asarray(r_c, dtype=float) * self.DU

    # velocity
    def v_to_canon(self, v):
        '''
        Convert velocity to canonical units.
        Inputs:
            v : array-like (3,) or (N,3)
                velocity in (DU-length units)/s
        Outputs:
            v_c : ndarray
                velocity in DU/TU
        '''
        return np.asarray(v, dtype=float) / self.VU

    def v_to_si(self, v_c):
        '''
        Convert canonical velocity to length-units per second.
        Inputs:
            v_c : array-like
                velocity in DU/TU
        Outputs:
            v : ndarray
                velocity in (DU-length units)/s
        '''
        return np.asarray(v_c, dtype=float) * self.VU

    # state conversion (pos,vel)
    def state_to_canon(self, r, v):
        '''
        Convert (r,v) to canonical.
        Inputs:
            r : array-like
                position in length units
            v : array-like
                velocity in length-units/s
        Outputs:
            r_c, v_c : ndarray, ndarray
                canonical state
        '''
        return self.r_to_canon(r), self.v_to_canon(v)

    def state_to_si(self, r_c, v_c):
        '''
        Convert (r_c, v_c) to SI.
        Inputs:
            r_c : array-like
                position in DU
            v_c : array-like
                velocity in DU/TU
        Outputs:
            r, v : ndarray, ndarray
                position in length units, velocity in length-units/s
        '''
        return self.r_to_si(r_c), self.v_to_si(v_c)

    # summarize canonical unit system w/print for when i forgetlol
    def summary(self):
        '''
        Handy summary print.
        '''
        print('CanonicalUnits:')
        print(f'  DU = {self.DU:.12g}')
        print(f'  mu = {self.mu:.12g}')
        print(f'  MU = {self.MU:.12g}')
        print(f'  TU = {self.TU:.12g}  (s)')
        print(f'  VU = {self.VU:.12g}  (DU/s)')
