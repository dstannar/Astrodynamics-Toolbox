import numpy as np
from MathHelpers.constants import muE
from MathHelpers.none_check import all_set, none_of

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
        if all_set(hmag, ecc, ta, raan, inc, argp) and none_of(r, v):
            self.hmag = hmag
            self.ecc = ecc
            self.TA = ta
            self.raan = raan
            self.inc = inc
            self.argp = argp
            self.coes_to_state()
            # recall to get full set of elements
            self.state_to_coes()
        elif all_set(r, v) and none_of(hmag, ecc, ta, raan, inc, argp):
            self.r = r
            self.v = v
            self.state_to_coes()
        else:
            raise TypeError('define EITHER all COEs or full state')

    
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
        ea = np.arctan(np.sqrt((1-ecc)/(1+ecc))*np.tan(ta/2))*2 #eccentric anomaly, radians
        ma = ea - ecc*np.sin(ea)

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