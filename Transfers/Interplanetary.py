import numpy as np
from MathHelpers.constants import muS, PlanetData, rE, muE, AU, rSun
from MathHelpers.time_to_perigee import time_to_perigee
from Propagators.Propagate import Propagate
from Transfers.Lambert import Lambert
from Orbits.KeplerianOrbit import KeplerianOrbit

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

    def depart_to_hohmann(self, r_plst, r_per, parking_alt, muP = muE, rPlan=rE):
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
    
    def depart_to_lambert(self, radPlan, alt_park, vinf_dep, muP):
        '''
        Assumes circular parking orbit
        '''
        rper_park = radPlan + alt_park
        v_park = np.sqrt(muP / rper_park)
        v_infmag = np.linalg.norm(vinf_dep)
        v_esc = np.sqrt(2 * muP / rper_park)
        vper_hyp = np.sqrt(v_infmag**2 + v_esc**2)
        dv_dep = abs(vper_hyp - v_park)

        return dv_dep

    def arrival_from_lambert(self, radPlan, alt_per, alt_apo, vinf_arr, muP):
        rper_park = radPlan + alt_per
        rapo_park = radPlan + alt_apo
        a_cap = 0.5*(rper_park + rapo_park)
        v_infmag = np.linalg.norm(vinf_arr)
        vesc_per = np.sqrt(2 * muP / rper_park)
        vper_hyp = np.sqrt(v_infmag**2 + vesc_per**2)
        vcap_per = np.sqrt(muP * (2/rper_park - 1/a_cap))
        dv_arr = abs(vcap_per - vper_hyp)

        return dv_arr

    def lambert_cruise(self, Orbit1, Orbit2, tof, shortWay):
        '''
        Inputs: 
            Orbit1 = orbit object AT depart position
            Orbit2 = orbit object AT arrive position
            tof = difference between Orbit1 and Orbit2 time (sec)
        Outputs:

        '''
        # init Lambert with self.mu
        interplanetaryLambert = Lambert(mu=muS)
        # solve lambert with given r1, r2, tof, shortwayFlag
        v1, v2, exitFlag = interplanetaryLambert.robust_solve(Orbit1.r, Orbit2.r, tof, shortWay=shortWay)
        # find delta Vs
        vinf_dep = v1 - Orbit1.v
        vinf_arr = v2 - Orbit2.v

        cruise = {
            "vLambert_dep": v1,
            "vLambert_arr": v2,
            "vinf_dep": vinf_dep,
            "vinf_arr": vinf_arr,
            "tof": tof,
            "shortWay": shortWay,
            "exitFlag": exitFlag,
        }

        if exitFlag == 1:
            return cruise
        else:
            return None
        
    def patched_lambert_transfer(self, Orbit1, Orbit2, tof, shortWay, muP1, muP2, radPlan1, radPlan2, depalt_park, arralt_per, arralt_apo):
        # get vinf
        cruise = self.lambert_cruise(Orbit1, Orbit2, tof, shortWay)
        if cruise is None:
            return None
        vLambert_dep = cruise["vLambert_dep"]
        vLambert_arr = cruise["vLambert_arr"]
        vinf_dep     = cruise["vinf_dep"]
        vinf_arr     = cruise["vinf_arr"]

        dv_dep = self.depart_to_lambert(radPlan1, depalt_park, vinf_dep, muP1)
        dv_arr = self.arrival_from_lambert(radPlan2, arralt_per, arralt_apo, vinf_arr, muP2)
        dv_tot = dv_dep + dv_arr

        # check for star intersection
        TransferOrbit = KeplerianOrbit(r=Orbit1.r, v=Orbit1.v + vinf_dep, mu=self.mu)
        r_closest_approach = TransferOrbit.r_per / AU

        result = {
            "dv_dep": dv_dep,
            "dv_arr": dv_arr,
            "dv_total": dv_tot,
            "transfer_orbit": TransferOrbit,
            "tof": tof,
            "vLambert_dep": vLambert_dep,
            "vLambert_arr": vLambert_arr,
            "vinf_dep": vinf_dep,
            "vinf_arr": vinf_arr,
            "shortWay": shortWay,
            "exitFlag": cruise["exitFlag"],
        }

        return result

    
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
    
    def gravity_assist_from_transfer_ellipse(self, T_trans, rtrans_apo, flyby_alt, planet, trailing):
        planetData = PlanetData[planet]
        a_trans = self.mu**(1/3) * (T_trans/(2*np.pi))**(2/3)
        rtrans_per = 2*a_trans - rtrans_apo
        v1 = np.sqrt(self.mu * (2/rtrans_apo - 1/a_trans))
        vPlanet = np.sqrt(self.mu / rtrans_apo)
        vinf = abs(v1 - vPlanet)
        
        rplanet_per = planetData["radius_km"] + flyby_alt
        ecc_flyby = 1 + rplanet_per * vinf**2 / planetData["mu"]
        turnAngle = 2*np.asin(1/ecc_flyby)
        heliocentric_dV = 2 * vinf * np.sin(turnAngle/2) # imparted deltaV
        if trailing==True:
            sign = 1
        else:
            sign = -1

        print(vinf)
        print(ecc_flyby)
        return heliocentric_dV, sign



