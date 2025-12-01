import numpy as np
from MathHelpers.constants import muS
from MathHelpers.time_to_perigee import time_to_perigee
from Propagators.Propagate import Propagate

class Interplanetary():
    def __init__(self, mu=None):
        if mu is None: self.mu = muS

    
    def hohmann(self, orbit1, orbit2):
        '''
        Inputs
            orbit1, orbit2: orbit objects
        Outputs
            deltaV for hohmann interplanetary
        Notes:
            - orbit1 and orbit2 must be circular
            - orbit1 and orbit2 are assumed to be coplanar
        '''
        mu = self.mu
        # check coplanar
        plane_norm1 = orbit1.hvec / np.linalg.norm(orbit1.hvec)
        plane_norm2 = orbit2.hvec / np.linalg.norm(orbit2.hvec)
        if not np.isclose(np.dot(plane_norm1, plane_norm2), 1):
            raise TypeError('Orbit1 and Orbit2 must be coplanar')
        # check circular
        if not np.isclose(orbit1.ecc, 0):
            raise TypeError("Orbit1 is not circular")
        if not np.isclose(orbit2.ecc, 0):
            raise TypeError("Orbit2 is not circular")
        
        # velocity vector on orbit1 at departure
        t_toPer1 = time_to_perigee(orbit1) # time to perigee of orbit1, secs
        # propagate to perigee
        if t_toPer1 != 0:
            # rvect and vvect at perigee
            rvect_per1, vvect_per1 = Propagate(prop_time=t_toPer1, Orbit=orbit1).lagrange_coeff()
        else:
            rvect_per1 = orbit1.r
            vvect_per1 = orbit1.v
        vmag_per1 = np.linalg.norm(vvect_per1)

        # velocity vector on orbit2 at arrival
        t_toPer2 = time_to_perigee(orbit2) # time to perigee of orbit1, secs
        # propagate to perigee
        if t_toPer2 != 0:
            # rvect and vvect at perigee
            rvect_per2, vvect_per2 = Propagate(prop_time=t_toPer2, Orbit=orbit2).lagrange_coeff()
        else:
            rvect_per2 = orbit2.r
            vvect_per2 = orbit2.v
        vmag_per2 = np.linalg.norm(vvect_per2)

        # transfer ellipse speed at departure (km/s)
        vmag_depart = np.sqrt(2*mu) * np.sqrt(orbit2.r_per / (orbit1.r_per * (orbit1.r_per+orbit2.r_per)))
        if vmag_depart > vmag_per1:
            # departure velocity vect same dir as vvect_per1
            depart_dir = vvect_per1 / np.linalg.norm(vvect_per1)
        else:
            # departure velocity vect opposite dir as vvect_per1
            depart_dir = (vvect_per1 / np.linalg.norm(vvect_per1)) * -1
        vvect_depart = depart_dir * vmag_depart
        # delta V
        dVarrive_vect = vvect_depart - vvect_per1
        dVarrive_mag = vmag_depart - vmag_per1

        # transfer ellipse speed at arrival (km/s)
        vmag_arrive = np.sqrt(2*mu) * np.sqrt(orbit1.r_per / (orbit2.r_per * (orbit1.r_per+orbit2.r_per)))
        if vmag_arrive > vmag_per2:
            # departure velocity vect same dir as vvect_per1
            arrive_dir = vvect_per2 / np.linalg.norm(vvect_per2)
        else:
            # departure velocity vect opposite dir as vvect_per1
            arrive_dir = (vvect_per2 / np.linalg.norm(vvect_per2)) * -1
        vvect_arrive = arrive_dir * vmag_arrive
        # delta V
        dVdepart_vect = vvect_per2 - vvect_arrive
        dVdepart_mag = vmag_per2 - vmag_arrive
        dVtot_mag = abs(dVarrive_mag) + abs(dVdepart_mag)

        return dVarrive_vect, dVdepart_vect, dVtot_mag

