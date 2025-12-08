import numpy as np
from MathHelpers.constants import muS, PlanetData, rE, muE
from MathHelpers.time_to_perigee import time_to_perigee
from Propagators.Propagate import Propagate

class Interplanetary():
    def __init__(self, mu=muS):
        self.mu = mu

    
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
            rvect_per1, vvect_per1 = Propagate(prop_time=t_toPer1, Orbit=orbit1, mu=self.mu).lagrange_coeff()
        else:
            rvect_per1 = orbit1.r
            vvect_per1 = orbit1.v
        vmag_per1 = np.linalg.norm(vvect_per1)

        # velocity vector on orbit2 at arrival
        t_toPer2 = time_to_perigee(orbit2) # time to perigee of orbit1, secs
        # propagate to perigee
        if t_toPer2 != 0:
            # rvect and vvect at perigee
            rvect_per2, vvect_per2 = Propagate(prop_time=t_toPer2, Orbit=orbit2, mu=self.mu).lagrange_coeff()
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

    def soi(self, planet):
        '''
        Inputs
            planet: str - heliocentric planet name with first letter Capitalized
        Outputs
            soi: int - sphere of influence, km
        '''
        muStar = self.mu
        planetData = PlanetData[planet]
        muP = planetData["mu"]
        smaP = planetData["orbit_sma_km"]
        soi = smaP * (muP / muStar)**(2/5)
        return soi

    def departure(self, r_plst, r_per, parking_alt, muP = muE, rPlan=rE):
        muStar = self.mu
        r_apo = r_plst
        a_trans = (r_per + r_apo) / 2
        # vis viva
        V_trans_apo = np.sqrt(muS * (2/r_apo - (1/a_trans)))
        # circular assumption
        V_earth = np.sqrt(muStar / r_plst)
        Vinf = abs(V_earth - V_trans_apo)

        r_per_park = parking_alt + rPlan
        v_esc = np.sqrt(2*muP / r_per_park)
        v_p = np.sqrt(Vinf**2 + v_esc**2)

        v_park = np.sqrt(muP / r_per_park)
        dv = v_p - v_park
        return dv, Vinf
    
    def hohmann_gravity_assist(self, inner_planet: str, flyby_planet: str, periapsis_alt: float, sunlit=True):
        muStar = self.mu
        flyby_data = PlanetData[flyby_planet]
        inner_data = PlanetData[inner_planet]

        r1 = inner_data["orbit_sma_km"] # inner circular orbit radius
        r2 = flyby_data["orbit_sma_km"] # flyby planet's circular orbit radius
        muP = flyby_data["mu"] # planet GM
        rP = flyby_data["radius_km"]  # planet radius, km

        # hohmann transfer ellipse
        a_trans = 0.5 * (r1 + r2)
        # Vis-viva at r2 on transfer ellipse
        V1_mag = np.sqrt(muStar * (2.0 / r2 - 1.0 / a_trans))
        # Circular heliocentric speed of the flyby planet
        V_pl_mag = np.sqrt(muStar / r2)


        # define frame
        r_vec = np.array([r2, 0.0, 0.0])
        V_vec = np.array([0.0, V_pl_mag, 0.0]) # planet's heliocentric V
        V1_vec = np.array([0.0, V1_mag, 0.0]) # spacecraft heliocentric V on Hohmann

        vinf_in = V1_vec - V_vec 
        vinf_mag = np.linalg.norm(vinf_in)

        # Build unit vectors along planet velocity and radial
        u_V = V_vec / np.linalg.norm(V_vec)   # tangential
        u_S = r_vec / np.linalg.norm(r_vec)   # radial outward from Sun

        vinf_V = np.dot(vinf_in, u_V)
        vinf_S = np.dot(vinf_in, u_S)
        phi1 = np.arctan2(vinf_S, vinf_V)

        # Periapsis radius from planet center
        r_per = rP + periapsis_alt

        # Hyperbola eccentricity from v_inf and rp (Curtis Eq. (8.83))
        e_hyp = 1.0 + (r_per * vinf_mag**2) / muP

        delta = 2.0 * np.arcsin(1.0 / e_hyp)

        if sunlit:
            delta = -delta

        # Outbound angle between v_inf2 and V (Curtis Eq. (8.85))
        phi2 = phi1 + delta

        # Outgoing v_inf2 components in (u_V, u_S)
        vinf_out = (
            vinf_mag * np.cos(phi2) * u_V +
            vinf_mag * np.sin(phi2) * u_S
        )

        # outbound heliocentric velocity
        V2_vec = V_vec + vinf_out

        # dv imparted by the flyby
        dV_imparted = np.linalg.norm(V2_vec - V1_vec)

        return dV_imparted


