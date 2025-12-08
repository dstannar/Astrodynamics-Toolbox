'''
This script solves a selection of Curtis ch8 problems
Implemented to complete Cal Poly AERO 351 HW4 
Problems solved: 8.2, 8.4, 8.6, 8.7, 8.12, 8.16
'''

import numpy as np
from Transfers.Interplanetary import Interplanetary
from MathHelpers.constants import muS, muE, rE, rMars, JD_J2000, MarsData
from MathHelpers.synodic_period import synodic_period
from MathHelpers.formatting import format_time
from Time.conversions import dateTime_to_JDays, AbsJDay_to_J2000JDay, JDays_to_secs
from Transfers.Lambert import Lambert
from Orbits.planetary_elements import planetary_elements2
from Orbits.KeplerianOrbit import KeplerianOrbit
from MathHelpers.solve_kepler import solve_kepler


def ch8_2():
    '''
    Q: Find the total delta-v required for a Hohmann transfer from Mars’ orbit to Jupiter’s orbit.
    A: Use my Interplanetary class's hohmann() function
    '''
    # import circular, coplanar Mars and Jupiter orbit onjects
    from Orbits.heliocentric_planets import Mars_CC, Jupiter_CC
    # call Interplanetary.hohmann
    interplanetary = Interplanetary()
    dVarrive_vect, dVdepart_vect, dvTot_mag = interplanetary.hohmann(orbit1=Mars_CC, orbit2=Jupiter_CC)
    print("---PROBLEM 8.2---")
    print("The total required delta-v (km/s) is: ", dvTot_mag)

def ch8_4():
    '''
    Q: Calculate the synodic period of Jupiter relative to Mars.
    A: Use my synodic_period helper function which implements the synodic period eqn
    '''
    # import Mars and Jupiter periods
    from MathHelpers.constants import JupiterData, MarsData
    JupSMA = JupiterData["orbit_sma_km"]
    MarsSMA = MarsData["orbit_sma_km"]
    TSyn_JupMars = synodic_period(JupSMA, MarsSMA, mu=muS)
    TSyn, TSyn_unit = format_time(TSyn_JupMars)
    print("---PROBLEM 8.4---")
    print("The synodic period for Mars and Jupiter is: ", TSyn, TSyn_unit)

def ch8_6():
    '''
    Q: Calculate the radius of the spheres of influence of Saturn, Uranus, and Neptune.
    A: Use my soi() func in the Interplanetary class to solve
    '''
    interplanetary = Interplanetary() # with mu=muSun
    soiSaturn = interplanetary.soi("Saturn")
    soiUranus = interplanetary.soi("Uranus")
    soiNeptune = interplanetary.soi("Neptune")
    soiPluto = interplanetary.soi("Pluto")
    print("---PROBLEM 8.6---")
    print("The sphere of influence of Saturn (km) is: ", soiSaturn)
    print("The sphere of influence of Uranus (km) is: ", soiUranus)
    print("The sphere of influence of Neptune (km) is: ", soiNeptune)
    print("The sphere of influence of Pluto (km) is: ", soiPluto)

def ch8_7():
    '''
    Q: On a date when the earth was 147.4(10^6) km from the sun, a spacecraft parked in a 200-kmaltitude
    circular earth orbit was launched directly into an elliptical orbit around the sun with
    perihelion of 120(106) km and aphelion equal to the earth's distance from the sun on the launch
    date. Calculate the delta-v required and vinf of the departure hyperbola.
    A: Use my depature() function within the Interplanetary class
    '''
    r_plst = 147.4e6 # km
    alt_park = 200 # km
    r_perS = 120e6 #km
    interplanetary = Interplanetary()
    dv, v_inf = interplanetary.departure(r_plst, r_perS, alt_park)
    print("---PROBLEM 8.7---")
    print("The required delta v (km/s) is: ", dv)
    print("The hyperbolic excess speed (km/s) is: ", v_inf)

def ch8_12():
    '''
    Q: Suppose a spacecraft approaches Jupiter on a Hohmann transfer ellipse from earth. If the
    spacecraft flies by Jupiter at an altitude of 200,000 km on the sunlit side of the planet, determine
    the orbital elements of the postflyby trajectory and the delta-v imparted to the spacecraft by
    Jupiter's gravity. Assume that all the orbits lie in the same (ecliptic) plane.
    A: Use my hohmann_gravity_assist() function
    '''
    interplanetary = Interplanetary()
    dv = interplanetary.hohmann_gravity_assist("Earth", "Jupiter", 200_000)
    print("---PROBLEM 8.12---")
    print("The delta V imparted (km/s) is: ", dv)

def ch8_16():
    '''
    Q: On August 15, 2005, a spacecraft in a 190-km, 52°-inclination circular parking orbit around the
    earth departed on a mission to Mars, arriving at the red planet on March 15, 2006, whereupon
    retrorockets placed it into a highly elliptic orbit with a periapsis of 300 km and a period of 35 h.
    Determine the total delta-v required for this mission.
    A: use time and mean motions to get planet states, use lambert with given TOF and parabolic escape speeds
    '''
    # departure and arrival calendar dates
    year_dep, month_dep, day_dep = 2005, 8, 15
    year_arr, month_arr, day_arr = 2006, 3, 15

    # julian dates for 0h UT on each date
    _, _, JD_dep = dateTime_to_JDays(year_dep, month_dep, day_dep, 0, 0, 0.0)
    _, _, JD_arr = dateTime_to_JDays(year_arr, month_arr, day_arr, 0, 0, 0.0)

    # time of flight in days and seconds
    tof_days = JD_arr - JD_dep
    tof_secs = JDays_to_secs(tof_days)

    # centuries from J2000.0 for Meeus-style elements
    T_dep = (JD_dep - JD_J2000) / 36525.0
    T_arr = (JD_arr - JD_J2000) / 36525.0

    # planetary osculating elements from planetary_elements2 (Meeus)
    # coes = [a_km, ecc, inc_deg, raan_deg, w_hat_deg, L_deg]
    aE, eE, incE_deg, raanE_deg, w_hatE_deg, L_E_deg = planetary_elements2(3, T_dep)
    aM, eM, incM_deg, raanM_deg, w_hatM_deg, L_M_deg = planetary_elements2(4, T_arr)

    # angle conversions
    deg2rad = np.pi / 180.0
    incE = incE_deg * deg2rad
    raanE = raanE_deg * deg2rad
    wbarE = w_hatE_deg * deg2rad
    LE = L_E_deg * deg2rad

    incM = incM_deg * deg2rad
    raanM = raanM_deg * deg2rad
    wbarM = w_hatM_deg * deg2rad
    LM = L_M_deg * deg2rad

    # argument of perihelion = longitude of perihelion - RAAN
    argpE = wbarE - raanE
    argpM = wbarM - raanM

    # mean anomaly = mean longitude - longitude of perihelion
    ME = (LE - wbarE) % (2.0 * np.pi)
    MM = (LM - wbarM) % (2.0 * np.pi)

    # solve kepler for eccentric anomaly
    EE = solve_kepler(ME, eE)
    EM = solve_kepler(MM, eM)

    # true anomaly from eccentric anomaly
    nuE = 2.0 * np.arctan2(
        np.sqrt(1.0 + eE) * np.sin(EE / 2.0),
        np.sqrt(1.0 - eE) * np.cos(EE / 2.0),
    ) % (2.0 * np.pi)

    nuM = 2.0 * np.arctan2(
        np.sqrt(1.0 + eM) * np.sin(EM / 2.0),
        np.sqrt(1.0 - eM) * np.cos(EM / 2.0),
    ) % (2.0 * np.pi)

    # specific angular momentum for each heliocentric orbit
    hE = np.sqrt(muS * aE * (1.0 - eE**2))
    hM = np.sqrt(muS * aM * (1.0 - eM**2))

    # build heliocentric orbit objects (sun-centered)
    earth_orbit = KeplerianOrbit(
        hmag=hE, ecc=eE, ta=nuE, raan=raanE, inc=incE, argp=argpE, mu=muS
    )
    mars_orbit = KeplerianOrbit(
        hmag=hM, ecc=eM, ta=nuM, raan=raanM, inc=incM, argp=argpM, mu=muS
    )

    rE_vec = earth_orbit.r
    vE_vec = earth_orbit.v
    rM_vec = mars_orbit.r
    vM_vec = mars_orbit.v

    # lambert transfer between the planet positions
    lam = Lambert(rE_vec, rM_vec, tof_secs, mu=muS, nrev=0, shortWay=True, verbose=False)
    v1_trans, v2_trans, exit_flag = lam.robust_solve()

    # earth departure hyperbola
    vinf_E_vec = v1_trans - vE_vec
    vinf_E = np.linalg.norm(vinf_E_vec)

    r_per_park = rE + 190.0  # 190 km parking altitude
    v_esc_E = np.sqrt(2.0 * muE / r_per_park)
    v_peri_hyp_E = np.sqrt(vinf_E**2 + v_esc_E**2)
    v_circ_park = np.sqrt(muE / r_per_park)
    dv1 = v_peri_hyp_E - v_circ_park

    # mars arrival and capture
    muMars = MarsData["mu"]
    r_per_cap = rMars + 300.0  # 300 km periapsis altitude for capture orbit

    vinf_M_vec = v2_trans - vM_vec
    vinf_M = np.linalg.norm(vinf_M_vec)

    v_peri_hyp_M = np.sqrt(vinf_M**2 + 2.0 * muMars / r_per_cap)

    # final mars capture ellipse: periapsis r_per_cap and period 35 h
    T_cap = 35.0 * 3600.0
    a_cap = (muMars * T_cap**2 / (4.0 * np.pi**2)) ** (1.0 / 3.0)
    v_peri_cap = np.sqrt(muMars * (2.0 / r_per_cap - 1.0 / a_cap))

    dv2 = v_peri_hyp_M - v_peri_cap

    dv_total = dv1 + dv2  # km/s

    print("---PROBLEM 8.16---")
    print("The total mission delta V (km/s) is: ", dv_total)


if __name__ == '__main__':
    ch8_2()
    ch8_4()
    ch8_6()
    ch8_7()
    ch8_12()
    ch8_16()