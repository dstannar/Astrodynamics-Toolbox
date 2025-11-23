import numpy as np
from Orbits.fetch_tle import fetch_tle
from MathHelpers.constants import muE, JDaysInSecs
from MathHelpers.solve_kepler import solve_kepler
from Time.conversions import dateTime_to_JDays, AbsJDay_to_J2000JDay
import copy as _copy

class TLEOrbit:
    def __init__(self, NORAD_ID, mu=muE):
        self.NORAD_ID = NORAD_ID
        self.mu = mu

        # scrape TLEs and define full list of orbital attributes
        self.set_state_from_TLEs()

    def get_tles(self):
        # call TLE extractor
        self.name, L1, L2 = fetch_tle(self.NORAD_ID)

        # Line 1
        self.satnum      = int(L1[2:7])
        self.epoch_year  = int(L1[18:20])                     # two-digit year
        self.epoch_doy   = float(L1[20:32])                   # day-of-year with fraction
        self.ndot        = float(L1[33:43])                   # rev/day^2

        # nddot and bstar are in mantissa+exponent form with implied decimal
        nddot_mant       = float(L1[44:50]) * 1e-5
        nddot_exp        = int(L1[50:52])
        self.nddot       = nddot_mant * (10.0 ** nddot_exp)   # rev/day^3
        # bstar
        bstar_mant       = float(L1[53:59]) * 1e-5
        bstar_exp        = int(L1[59:61])
        self.bstar       = bstar_mant * (10.0 ** bstar_exp)
        # element set type (modern should be 0) and element set number (running counter)
        self.elset_type  = int(L1[62:63])
        self.elset_num   = int(L1[64:68])

        # Line 2
        satnum2     = int(L2[2:7])                          # should match satnum
        # check if satnum2 matches satnum
        if satnum2 != self.satnum:
            raise RuntimeError('TLE internally inconsistent')
        
        inc_deg          = float(L2[8:16])                  # inclination [deg]
        self.inc         = np.deg2rad(inc_deg)              # change to radians
        raan_deg         = float(L2[17:25])                 # RAAN [deg]
        self.raan        = np.deg2rad(raan_deg)             # to radians
        self.ecc         = float('0.' + L2[26:33].strip())  # implied decimal eccentricity
        argp_deg         = float(L2[34:42])                 # argument of perigee [deg]
        self.argp        = np.deg2rad(argp_deg)             # to radians
        MA_deg           = float(L2[43:51])                 # mean anomaly [deg]
        self.MA          = np.deg2rad(MA_deg)               # to rads
        self.n_rev_day   = float(L2[52:63])                 # mean motion [rev/day]
        self.rev_num     = int(L2[63:68])                   # rev number 

    def full_coes_from_tle(self):
        ''''
        Get hmag and ta to round out full COE set from TLEs
        '''
        n_rad = self.n_rev_day * 2 * np.pi / JDaysInSecs
        self.sma = (self.mu / n_rad**2)**(1/3)
        self.hmag = np.sqrt(self.mu * self.sma * (1 - self.ecc**2))
        EA = solve_kepler(self.MA, self.ecc)
        self.TA = 2*np.arctan2(np.sqrt(1+self.ecc)*np.sin(EA/2),
                       np.sqrt(1-self.ecc)*np.cos(EA/2)) % (2*np.pi)
        # julian days since J2000
        y2 = int(self.epoch_year) # 2-digit TLE year
        year_full = (2000 + y2) if (0 <= y2 <= 56) else (1900 + y2)  # TLE convention
        _, _, JD_jan1 = dateTime_to_JDays(year_full, 1, 1, 0, 0, 0)  # abs JD at 0h UT Jan 1
        JD_abs = JD_jan1 + (float(self.epoch_doy) - 1.0)             # add (DOY - 1), DOY may be fractional
        self.JDsJ2000 = AbsJDay_to_J2000JDay(JD_abs)                 # assign JD since J2000.0 to self

    
    def state_to_coes(self):
        rvec = self.r
        vvec = self.v
        mu = self.mu
        k_hat = [0,0,1]

        rmag = np.linalg.norm(rvec)
        vmag = np.linalg.norm(vvec)
        v_r = np.dot(rvec,vvec) / rmag

        h = np.cross(rvec,vvec)
        hmag = np.linalg.norm(h)
        inc = np.arccos(h[2]/hmag)

        Nvec = np.cross(k_hat, h) # node line
        Nmag = np.linalg.norm(Nvec)
        raan_raw = np.arccos(Nvec[0]/Nmag)

        if Nvec[1] >= 0:
            raan = raan_raw
        else:
            raan = 2*np.pi - raan_raw

        evec = 1/mu * ((vmag**2 - mu/rmag)*rvec - rmag*v_r*vvec)
        ecc = np.linalg.norm(evec)
        arg_per_raw = np.arccos(np.dot(Nvec, evec)/(Nmag*ecc))

        if evec[2] >= 0:
            arg_per = arg_per_raw
        else:
            arg_per = 2*np.pi - arg_per_raw

        ta_raw = np.arccos(np.dot(evec, rvec) / (ecc*rmag))

        if v_r >= 0:
            ta = ta_raw
        else:
            ta = 2*np.pi - ta_raw

        #more fun parameters
        r_per = hmag**2/mu * (1/(1+ecc*np.cos(0))) #radius of perigee, km
        r_apo = hmag**2/mu * (1/(1+ecc*np.cos(np.pi))) #radius of apogee, km
        a = 0.5*(r_per+r_apo) # semi major axis, km
        To = 2*np.pi / np.sqrt(mu) * a**(3/2) # period, seconds
        energy = vmag**2 / 2 - (mu/rmag) #specific energy, km^2/s^2
        ea = 2*np.arctan2(np.sqrt(1-ecc)*np.sin(ta/2),np.sqrt(1+ecc)*np.cos(ta/2)) #eccentric anomaly, radians
        ea = ea % (2*np.pi)
        ma = (ea - ecc*np.sin(ea)) % (2*np.pi)

        # assign to self
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
        rmag = p / (1 + ecc*np.cos(ta))

        r0 = rmag * np.array([np.cos(ta), np.sin(ta), 0.0])               # r in PQW
        v0 = (mu/hmag) * np.array([-np.sin(ta), ecc + np.cos(ta), 0.0])   # v in PQW
        
        R3_W = np.array(([np.cos(raan), np.sin(raan), 0], [-np.sin(raan), np.cos(raan), 0], [0,0,1])) # Cz(raan)
        R1_i = np.array(([1, 0, 0], [0, np.cos(inc), np.sin(inc)], [0,-np.sin(inc),np.cos(inc)])) # Cx(inc)
        R3_w = np.array(([np.cos(argp), np.sin(argp), 0], [-np.sin(argp), np.cos(argp), 0], [0,0,1])) # Cz(argp)

        Q_peri_ECI = (R3_w@R1_i@R3_W) # DCM

        Q_ECI_peri = Q_peri_ECI.T

        self.r = np.dot(Q_ECI_peri, r0) # transform r vect
        self.v = np.dot(Q_ECI_peri, v0) # transform r vect


    def set_state(self, rnew, vnew):
        '''
        Update orbit attributes with new state
        '''
        self.r = rnew
        self.v = vnew
        # get COEs
        self.state_to_coes()
        # None out TLEs that are no longer accurate
        self.epoch_year  = None
        self.epoch_doy   = None
        self.ndot        = None
        self.nddot       = None
        self.bstar       = None
        self.elset_type  = None
        self.elset_num   = None
        self.n_rev_day   = None
        self.rev_num     = None


    def set_state_from_TLEs(self):
        '''
        Update orbit attributes with new TLE pull
        '''
        self.get_tles()
        # round out orbital element set
        self.full_coes_from_tle()
        # get state
        self.coes_to_state()
        # get full set of helpful orbital elements
        self.state_to_coes()

    def copy(self):
        """Return a copy of this orbit object"""
        return _copy.deepcopy(self)