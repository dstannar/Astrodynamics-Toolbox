import numpy as np
from MathHelpers.constants import muE
from MathHelpers.none_check import all_set, none_of
from Time.conversions import dateTime_to_JDays, AbsJDay_to_J2000JDay
from datetime import datetime

class KeplerianOrbit:
    def __init__(self, r=None, v=None, hmag=None, ecc=None, 
                 ta=None, raan=None, inc=None, argp=None, mu=muE):
        '''
        Inputs:
            r0, v0 
            OR
            hmag, ecc, ta, raan, inc, argp
        '''
        self.mu = mu

        # build from COEs
        if all_set(hmag, ecc, ta, raan, inc, argp) and none_of(r, v):
            self.hmag = hmag
            self.ecc = ecc
            self.TA = ta
            self.raan = raan
            self.inc = inc
            self.argp = argp
            # get state then full COE set (including circular special case)
            self.coes_to_state()
            self.state_to_coes()

        # build from state vector
        elif all_set(r, v) and none_of(hmag, ecc, ta, raan, inc, argp):
            self.r = r
            self.v = v
            self.state_to_coes()

        else:
            raise TypeError('define EITHER all COEs or full state')

        # tag creation epoch as JD since J2000
        now = datetime.utcnow()
        _, _, JD_abs = dateTime_to_JDays(
            now.year, now.month, now.day,
            now.hour, now.minute, now.second + now.microsecond/1.0e6
        )
        self.JDsJ2000 = AbsJDay_to_J2000JDay(JD_abs)

    
    def state_to_coes(self):
        '''
        Compute classical orbital elements and useful scalars from state.
        Handles circular and equatorial special cases so we do not divide by
        zero for Nmag or ecc.
        '''
        rvec = self.r
        vvec = self.v
        mu = self.mu
        k_hat = [0, 0, 1]

        rmag = np.linalg.norm(rvec)
        vmag = np.linalg.norm(vvec)
        v_r = np.dot(rvec, vvec) / rmag

        h = np.cross(rvec, vvec)
        hmag = np.linalg.norm(h)
        inc = np.arccos(h[2] / hmag)

        Nvec = np.cross(k_hat, h)  # node line
        Nmag = np.linalg.norm(Nvec)

        eps = 1.0e-8  # small tolerance for "zero"

        # RAAN: undefined if equatorial, set to zero by convention
        if Nmag > eps:
            raan_raw = np.arccos(Nvec[0] / Nmag)
            if Nvec[1] >= 0:
                raan = raan_raw
            else:
                raan = 2.0 * np.pi - raan_raw
        else:
            raan = 0.0

        evec = 1.0 / mu * ((vmag**2 - mu / rmag) * rvec - rmag * v_r * vvec)
        ecc = np.linalg.norm(evec)

        # argument of perigee and true anomaly with special cases
        if ecc > eps and Nmag > eps:
            # general (noncircular, inclined) case
            arg_per_raw = np.arccos(np.dot(Nvec, evec) / (Nmag * ecc))
            if evec[2] >= 0.0:
                arg_per = arg_per_raw
            else:
                arg_per = 2.0 * np.pi - arg_per_raw

            ta_raw = np.arccos(np.dot(evec, rvec) / (ecc * rmag))
            if v_r >= 0.0:
                ta = ta_raw
            else:
                ta = 2.0 * np.pi - ta_raw

        elif ecc > eps and Nmag <= eps:
            # noncircular but equatorial: RAAN undefined, so use longitude
            # of periapsis measured from inertial x axis
            lon_peri = np.arctan2(evec[1], evec[0])
            if lon_peri < 0.0:
                lon_peri += 2.0 * np.pi

            arg_per = lon_peri  # RAAN = 0 by convention

            ta = np.arccos(np.dot(evec, rvec) / (ecc * rmag))
            if v_r < 0.0:
                ta = 2.0 * np.pi - ta

        else:
            # circular orbit: argument of perigee is undefined
            if Nmag > eps:
                # circular inclined: use argument of latitude u
                u_raw = np.arccos(np.dot(Nvec, rvec) / (Nmag * rmag))
                if rvec[2] >= 0.0:
                    u = u_raw
                else:
                    u = 2.0 * np.pi - u_raw
                arg_per = 0.0
                ta = u
            else:
                # circular equatorial: use true longitude from inertial x axis
                lon = np.arctan2(rvec[1], rvec[0])
                if lon < 0.0:
                    lon += 2.0 * np.pi
                arg_per = 0.0
                ta = lon

        # more fun parameters
        r_per = hmag**2 / mu * (1.0 / (1.0 + ecc * np.cos(0.0)))      # radius of perigee, km
        r_apo = hmag**2 / mu * (1.0 / (1.0 + ecc * np.cos(np.pi)))    # radius of apogee, km
        a = 0.5 * (r_per + r_apo)                                     # semi major axis, km
        To = 2.0 * np.pi / np.sqrt(mu) * a**(3.0 / 2.0)               # period, seconds
        energy = vmag**2 / 2.0 - (mu / rmag)                          # specific energy, km^2/s^2

        if ecc < eps:
            ea = ta
            ma = ta
        else:
            ea = 2.0 * np.arctan2(np.sqrt(1.0 - ecc) * np.sin(ta / 2.0),
                                  np.sqrt(1.0 + ecc) * np.cos(ta / 2.0))
            ea = ea % (2.0 * np.pi)
            ma = (ea - ecc * np.sin(ea)) % (2.0 * np.pi)

        # assign to self (match TLEOrbit outward API where applicable) :contentReference[oaicite:0]{index=0}
        self.hmag = hmag
        self.inc = inc
        self.raan = raan
        self.ecc = ecc
        self.argp = arg_per
        self.TA = ta
        self.r_per = r_per
        self.r_apo = r_apo
        self.sma = a
        self.period = To
        self.energy = energy
        self.EA = ea
        self.MA = ma
        self.hvec = h
        self.v_r = v_r
        self.v_t = hmag / rmag
        self.FPA = np.atan2(self.v_r, self.v_t)
        self.vmag = vmag
        self.rmag = rmag
        self.eccvec = evec

    def coes_to_state(self):
        hmag = self.hmag
        mu = self.mu
        ecc = self.ecc
        ta = self.TA
        raan = self.raan
        inc = self.inc
        argp = self.argp

        p = hmag**2 / mu
        rmag = p / (1.0 + ecc * np.cos(ta))

        r0 = rmag * np.array([np.cos(ta), np.sin(ta), 0.0])               # r in PQW
        v0 = (mu / hmag) * np.array([-np.sin(ta), ecc + np.cos(ta), 0.0]) # v in PQW
        
        R3_W = np.array(([np.cos(raan), np.sin(raan), 0.0],
                         [-np.sin(raan), np.cos(raan), 0.0],
                         [0.0, 0.0, 1.0]))                                # Cz(raan)
        R1_i = np.array(([1.0, 0.0, 0.0],
                         [0.0, np.cos(inc), np.sin(inc)],
                         [0.0, -np.sin(inc), np.cos(inc)]))               # Cx(inc)
        R3_w = np.array(([np.cos(argp), np.sin(argp), 0.0],
                         [-np.sin(argp), np.cos(argp), 0.0],
                         [0.0, 0.0, 1.0]))                                # Cz(argp)

        Q_peri_ECI = (R3_w @ R1_i @ R3_W)  # DCM

        Q_ECI_peri = Q_peri_ECI.T

        self.r = np.dot(Q_ECI_peri, r0)  # transform r vect
        self.v = np.dot(Q_ECI_peri, v0)  # transform v vect


    def set_state(self, rnew, vnew):
        '''
        Update orbit attributes with new state
        '''
        self.r = rnew
        self.v = vnew
        # get COEs (includes circular special cases)
        self.state_to_coes()
