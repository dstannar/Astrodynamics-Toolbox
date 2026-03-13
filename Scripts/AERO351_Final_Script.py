import numpy as np
from Time.conversions import dateTime_to_JDays, AbsJDay_to_J2000JDay, JDays_to_secs
from Transfers.Lambert import Lambert
from Orbits.planetary_elements import planetary_elements
from MathHelpers.constants import muS, EarthData, VenusData, AU, rE
from Orbits.KeplerianOrbit import KeplerianOrbit
from Transfers.Interplanetary import Interplanetary
import matplotlib.pyplot as plt
from Propagators.Propagate import Propagate
from Propagators.plot_helper import composite_trajectory


def problem1():
    '''
    Q: On 1.1.2026 sat departs 500km alt circular Earth orbit towards Venus. Is 9.1.26, 10.1.26, 11.1.26 a better arrival time?
    A: Get arrival, departure, coast arcs. Use Lambert solver to determine deltaV cost with different interplanetary flights. Check short way and long way. 
    '''
    # get dateTime in Julian Days 
    _, _, JD_1jan26 = dateTime_to_JDays(year=2026, month=1, day=1, hour=0, minute=0, second=0)
    _, _, JD_1sept26 = dateTime_to_JDays(year=2026, month=9, day=1, hour=0, minute=0, second=0)
    _, _, JD_1oct26 = dateTime_to_JDays(year=2026, month=10, day=1, hour=0, minute=0, second=0)
    _, _, JD_1nov26 = dateTime_to_JDays(year=2026, month=11, day=1, hour=0, minute=0, second=0)
    # to J2000
    JD2k_1jan26 = AbsJDay_to_J2000JDay(JD_1jan26)
    JD2k_1sept26 = AbsJDay_to_J2000JDay(JD_1sept26)
    JD2k_1oct26 = AbsJDay_to_J2000JDay(JD_1oct26)
    JD2k_1nov26 = AbsJDay_to_J2000JDay(JD_1nov26)

    # get tofs
    tof2sept = JDays_to_secs(JD_1sept26 - JD_1jan26)
    tof2oct = JDays_to_secs(JD_1oct26 - JD_1jan26)
    tof2nov = JDays_to_secs(JD_1nov26 - JD_1jan26)

    # get coes from Dr. A's planetary elements function
    hmag_Ejan, ecc_Ejan, ta_Ejan, raan_Ejan, inc_Ejan, argp_Ejan = planetary_elements(planet_id=3, T=JD2k_1jan26/36525.0) #func expects J2000 centuries
    hmag_Vsept, ecc_Vsept, ta_Vsept, raan_Vsept, inc_Vsept, argp_Vsept = planetary_elements(planet_id=2, T=JD2k_1sept26/36525.0)
    hmag_Voct, ecc_Voct, ta_Voct, raan_Voct, inc_Voct, argp_Voct = planetary_elements(planet_id=2, T=JD2k_1oct26/36525.0)
    hmag_Vnov, ecc_Vnov, ta_Vnov, raan_Vnov, inc_Vnov, argp_Vnov = planetary_elements(planet_id=2, T=JD2k_1nov26/36525.0)

    # create Orbit objects
    EarthOrbitJan = KeplerianOrbit(hmag = hmag_Ejan, ecc = ecc_Ejan, ta = ta_Ejan, raan = raan_Ejan, inc = inc_Ejan, argp = argp_Ejan, mu=muS)
    VenusOrbitSept = KeplerianOrbit(hmag = hmag_Vsept, ecc = ecc_Vsept, ta = ta_Vsept, raan = raan_Vsept, inc = inc_Vsept, argp = argp_Vsept, mu=muS)
    VenusOrbitOct = KeplerianOrbit(hmag = hmag_Voct, ecc = ecc_Voct, ta = ta_Voct, raan = raan_Voct, inc = inc_Voct, argp = argp_Voct, mu=muS)
    VenusOrbitNov = KeplerianOrbit(hmag = hmag_Vnov, ecc = ecc_Vnov, ta = ta_Vnov, raan = raan_Vnov, inc = inc_Vnov, argp = argp_Vnov, mu=muS)

    # init Interplanetary class object
    interplanetary = Interplanetary(mu=muS)

    # get Earth and Venus radii
    rE = EarthData["radius_km"]
    rV = VenusData["radius_km"]
    muE = EarthData["mu"]
    muV = VenusData["mu"]

    alt_park = 500.0    # km
    alt_per  = 2000.0   # km
    alt_apo  = 10000.0  # km

    # short way transfers
    res_sept_sw = interplanetary.patched_lambert_transfer(
        EarthOrbitJan, VenusOrbitSept, tof2sept,
        shortWay=True,
        muP1=muE, muP2=muV,
        radPlan1=rE, radPlan2=rV,
        depalt_park=alt_park,
        arralt_per=alt_per, arralt_apo=alt_apo
    )

    res_oct_sw = interplanetary.patched_lambert_transfer(
        EarthOrbitJan, VenusOrbitOct, tof2oct,
        shortWay=True,
        muP1=muE, muP2=muV,
        radPlan1=rE, radPlan2=rV,
        depalt_park=alt_park,
        arralt_per=alt_per, arralt_apo=alt_apo
    )

    res_nov_sw = interplanetary.patched_lambert_transfer(
        EarthOrbitJan, VenusOrbitNov, tof2nov,
        shortWay=True,
        muP1=muE, muP2=muV,
        radPlan1=rE, radPlan2=rV,
        depalt_park=alt_park,
        arralt_per=alt_per, arralt_apo=alt_apo
    )

    # Print dV’s from the results
    print('---PROBLEM 1 DELTA Vs SHORT WAY---')
    for res, label in zip([res_sept_sw, res_oct_sw, res_nov_sw],
                        ['Sept', 'Oct', 'Nov']):
        if res is None:
            print(f'{label}: Lambert failed')
        else:
            print(f'{label}: dV_total = {res["dv_total"]:.3f} km/s')

    # Plot shortway transfers
    plot_problem1(
        EarthOrbitJan,
        VenusOrbitSept,
        [res_sept_sw, res_oct_sw, res_nov_sw],
        labels=['Jan to Sept', 'Jan to Oct', 'Jan to Nov'],
        title='Earth to Venus Transfers, Short Way'
    )

    # long way transfers
    res_sept_lw = interplanetary.patched_lambert_transfer(
        EarthOrbitJan, VenusOrbitSept, tof2sept,
        shortWay=False,
        muP1=muE, muP2=muV,
        radPlan1=rE, radPlan2=rV,
        depalt_park=alt_park,
        arralt_per=alt_per, arralt_apo=alt_apo
    )

    res_oct_lw = interplanetary.patched_lambert_transfer(
        EarthOrbitJan, VenusOrbitOct, tof2oct,
        shortWay=False,
        muP1=muE, muP2=muV,
        radPlan1=rE, radPlan2=rV,
        depalt_park=alt_park,
        arralt_per=alt_per, arralt_apo=alt_apo
    )

    res_nov_lw = interplanetary.patched_lambert_transfer(
        EarthOrbitJan, VenusOrbitNov, tof2nov,
        shortWay=False,
        muP1=muE, muP2=muV,
        radPlan1=rE, radPlan2=rV,
        depalt_park=alt_park,
        arralt_per=alt_per, arralt_apo=alt_apo
    )

    print('---PROBLEM 1 DELTA Vs LONG WAY---')
    for res, label in zip([res_sept_lw, res_oct_lw, res_nov_lw],
                        ['Sept', 'Oct', 'Nov']):
        if res is None:
            print(f'{label}: Lambert failed')
        else:
            print(f'{label}: dV_total = {res["dv_total"]:.3f} km/s')

    plot_problem1(
        EarthOrbitJan,
        VenusOrbitSept,
        [res_sept_lw, res_oct_lw, res_nov_lw],
        labels=['Jan to Sept', 'Jan to Oct', 'Jan to Nov'],
        title='Earth to Venus Transfers Long Way'
    )

def plot_problem1(earth_orbit, venus_orbit, transfer_results, labels, title):
    figs = []
    labels_all = []

    # plot earth orbit for 1 period
    T_E = earth_orbit.period
    _, _, figE = Propagate(prop_time=T_E, Orbit=earth_orbit, mu=muS).twobody_ODE(plot=True)
    figs.append(figE)
    labels_all.append('Earth orbit')

    # plot venus orbit for 1 period
    T_V = venus_orbit.period
    _, _, figV = Propagate(prop_time=T_V, Orbit=venus_orbit, mu=muS).twobody_ODE(plot=True)
    figs.append(figV)
    labels_all.append('Venus orbit')

    # plot transfer arcs
    for res, leg_label in zip(transfer_results, labels):
        if res is None:
            continue  # Lambert failed, don't try to plot or ode45 will throw an error

        transfer_orbit = res["transfer_orbit"]
        tof = res["tof"]

        _, _, figT = Propagate(prop_time=tof, Orbit=transfer_orbit, mu=muS).twobody_ODE(plot=True)
        figs.append(figT)
        labels_all.append(leg_label)

    # put into one plot
    composite_trajectory(
        figs,
        labels=labels_all,
        title=title,
        show=True
    )


def problem2():
    '''
    Q: find deltaV imparted from s/c flyby of Earth in heliocentric frame
    A: use my function gravity_assist_from_transfer_ellipse() in the interplanetary class
    '''
    # givens
    T_trans = JDays_to_secs(284) # period in seconds
    rtrans_apo = 149_598_000 #km
    flyby_alt = 10000 #km
    # init interplanetary solver object
    interplanetarySunCentered = Interplanetary(mu=muS)
    dv_imparted, sign = interplanetarySunCentered.gravity_assist_from_transfer_ellipse(T_trans, rtrans_apo, flyby_alt, planet="Earth", trailing=True)

    print('---PROBLEM 2---')
    print('The delta V imparted from the flyby in the heliocentric frame (km/s) is: ', dv_imparted)
    print('Since the flyby is on the trailing side this deltaV is a GAIN')

def problem3():
    '''
    Q: a s/c burns its LT thruster for 2days, coasts 1day, burns 1.1 days. Given ISP, thrust, mass, initial state, find perigee post thrust
    A: use my twobody_ODE with thrust enabled
    '''
    # givens
    prop_time1 = JDays_to_secs(2) #s
    coast_time = JDays_to_secs(1) #s
    prop_time2 = JDays_to_secs(1.1) #s
    thrustN = -0.006 # kN (what my propagator expects), deorbit so burn retrograde so negative thrust
    Isp = 5000 # s
    mass_0 = 600 #kg

    # create orbit object
    rvect_eci = np.array([16378, 0, 0]) #km, given
    vvect_eci = np.array([0, 4.9333, 0]) #km/s, given
    orbit1 = KeplerianOrbit(r=rvect_eci, v=vvect_eci)

    # Do first 2 day burn with thrust, update orbit state
    r1, v1, mass_1, plot1 = Propagate(prop_time = prop_time1, Orbit = orbit1, thrust=thrustN, burnTime=prop_time1, Isp=Isp, mass=mass_0).twobody_ODE(plot=True)
    orbit1.set_state(rnew=r1, vnew=v1)

    # coast for 1 day, set state
    r2, v2, plot2 = Propagate(prop_time=coast_time, Orbit=orbit1).twobody_ODE(plot=True)
    orbit1.set_state(rnew=r2, vnew=v2)

    # burn for another 1.1 days, set state
    r3, v3, mass_2, plot3 = Propagate(prop_time=prop_time2, Orbit=orbit1, thrust=thrustN, burnTime=prop_time2, Isp=Isp, mass=mass_1).twobody_ODE(plot=True)
    orbit1.set_state(rnew=r3, vnew=v3)

    # get perigee of newest orbit object
    r_per = orbit1.r_per
    z_per = r_per - rE

    # plot for one period
    _, _, plot4 = Propagate(prop_time=orbit1.period, Orbit=orbit1).twobody_ODE(plot=True)

    # display results
    print("---PROBLEM 3---")
    print("The final mass (kg) is: ", mass_2)
    print("The perigee altitude of the spacecraft's final orbit (km) is: ", z_per)

    # plot
    composite_trajectory([plot1, plot2, plot3, plot4], ["Two Day Burn", "One Day Coast", "1.1 Day Burn", "Final Orbit"], title="Trajectory of Deorbiting Spacecraft", show=True)

def problem4():
    '''
    Q: Use universal variable propagation to plot an orbit for 35 min given initial state
    A: use my universal variable propagator @ each timestep to do just that
    '''
    # init orbit object
    rvect = np.array([-5959.72, -4338.9, 3992.93]) # given, assumed km
    vvect = np.array([4.20251, -4.4142, -0.58846]) # given, assumed km/s
    orbit1 = KeplerianOrbit(r=rvect, v=vvect)

    # use while loop to solve universal variable @ each timestep and record states
    dt = 1 # 1 second timestep, probably overkill here for basic plot but thats ok
    time = 1 # time counter, seconds. start at 1 so first pass isn't useless
    Rs = [rvect]
    Vs = [vvect]
    # loop through until 35 min
    while time <= 35*60:
        rNew, vNew = Propagate(prop_time=time, Orbit=orbit1).lagrange_coeff()
        Rs.append(rNew)
        Vs.append(vNew)
        time += dt

    Rs = np.array(Rs) # make numpy array
    # display position magntiude of spacecraft at 35 min
    rFinal = np.linalg.norm(Rs[-1,:])
    print('---PROBLEM 4---')
    print('The position (magnitude) of the s/c after 35 min (km) is: ', rFinal)

    # unpack Rs vector and plot
    x = Rs[:, 0]
    y = Rs[:, 1]
    z = Rs[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, linewidth=1)
    # Mark start and end
    ax.scatter(x[0],  y[0],  z[0],  s=50, marker='o', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], s=50, marker='x', label='End')

    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_zlabel('z [km]')
    ax.set_title('Trajectory from Universal Variable Propagation')
    ax.legend()
    ax.set_box_aspect([1, 1, 1])  # equal-ish aspect

    plt.show()



if __name__ == '__main__':
    #problem1()
    problem2()
    #problem3()
    #problem4()