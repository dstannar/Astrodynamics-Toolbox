"""
Drew Stannard-Stockton
AERO351 Group Project Orbit Debris Cleanup Mission Implementation
"""
import numpy as np
from Orbits.TLEOrbit import TLEOrbit
from Orbits.KeplerianOrbit import KeplerianOrbit
from Propagators.Propagate import Propagate
from Transfers.Lambert import Lambert
from MathHelpers.synodic_period import synodic_period
from Transfers.plane_change import nodal_crossings_array
from MathHelpers.constants import rE, muE
from Time.conversions import secs_to_JDays, JDays_to_secs
from Orbits.safe_create_orbits import create_TLEOrbits
from Propagators.plot_helper import composite_trajectory
from MathHelpers.formatting import format_time
# because we expect divide by zero warnings in Lambert solve loop as not all tofs will be physical
import warnings
warnings.filterwarnings("ignore")
# everything earth orbiting
mu = muE


def scanLambertCandidates(orbit1, orbit2, missionTimePre=0, dt=100, bestGuess=None):
    """
    Scan across time and solve Lambert transfers to get the smallest dV transfer for a variety of orbits
    Input:
        Orbit1s - orbit object to transfer from
        Orbit2s - orbit object to transfer to
        dt - time step for Lambert scan, defaults to rough scan of 100s intervals
        bestGuess - preseed best guess for finer scans, first call should be None

    Output (bestResult array):
        [0] satnum1 (NORAD ID), [1] satnum2 (NORAD ID), [2] r1 [km], [3] r2[km], [4] tof[s],
        [5] dv1[km/s], [6] dv2[km/s], [7] dvMag[km/s], [8] departTime_abs[s, since J2000],
        [9] t1_low, [10] t1_high, [11] t2_low, [12] t2_high, - these are all brackets for the solution
        [13] v_pretrans1, [14] v_pretrans2 - both km/s, pre lambert velos

    Notes:
        don't need to use abs time throughout as time is about time of propagation, just need to ensure TLE epochs are same
    """
    # init loop vars
    dvMagOld = np.inf
    bestResult = None
    # check if orbit1 and orbit2 are in same epoch, if not propagate one to catch up so we can use relative times
    if orbit1.JDsJ2000 > orbit2.JDsJ2000:
        dJdays = orbit1.JDsJ2000 - orbit2.JDsJ2000
        dSecs = JDays_to_secs(dJdays)
        # propagate for offset in seconds
        rTrue, vTrue = Propagate(prop_time=dSecs, Orbit=orbit2).lagrange_coeff()
        orbit2.set_state(rTrue, vTrue) # orbit 2 is now at same absolute time as orbit1
        # use orbit1 TLE epoch as our epoch
        epochTLE = orbit1.JDsJ2000
    # same but now if orbit1 is behind orbit2
    elif orbit1.JDsJ2000 < orbit2.JDsJ2000:
        dJdays = orbit2.JDsJ2000 - orbit1.JDsJ2000
        dSecs = JDays_to_secs(dJdays)
        # propagate for offset in seconds
        rTrue, vTrue = Propagate(prop_time=dSecs, Orbit=orbit1).lagrange_coeff()
        orbit1.set_state(rTrue, vTrue) # orbit 2 is now at same absolute time as orbit1
        # use orbit2 TLE epoch as our epoch
        epochTLE = orbit2.JDsJ2000
    elif orbit1.JDsJ2000 == orbit2.JDsJ2000:
        # use orbit1 tle epoch as our epoch but doesn't matter could be orbit 2
        epochTLE = orbit1.JDsJ2000

    # now that time is fixed:
    # create "chaser" s/c object that will execute the lambert maneuvers so we don't modify debris orbits and mess up times
    # specifically for checking earth intersection
    satOrbit = orbit1.copy() # creates exact copy of object

    # now that times are equal set search bracket either to preseeded bracket or to [0, Tsyn]     
    # get synodic period from my helper func
    Tsyn = synodic_period(orbit1.sma, orbit2.sma)

    # if not given preseeded solution bracket, start at t=0 and go until synodic period
    if bestGuess is None:
        t1_start = 0.0
        t1_end = Tsyn
        t2_start = 0.0
        t2_end = Tsyn
        # time trackers
        t1 = 0
        t2 = 0
    # if given solution bracket set high and low as given in bestGuess array
    else:
        t1_start  = float(bestGuess[9])
        t1_end = float(bestGuess[10])
        t2_start  = float(bestGuess[11])
        t2_end = float(bestGuess[12])
        # set t20 to t2_start and t10 to t1_start once
        t1 = t1_start
        t2 = t2_start

    # PER ORBIT PAIR SOLVE LOOP
    while t1 <= t1_end:
        # get state at departure
        r1New, v1New = Propagate(t1, Orbit=orbit1).lagrange_coeff()
        t2 = max(t2_start, t1)   # restart inner scan every outer step

        while t2 <= t2_end:
            tof = t2 - t1
            # dont allow negative tof
            if tof <= 0:
                t2 += dt
                continue

            # propagate orbit2 to t2
            r2New, v2New = Propagate(t2, Orbit=orbit2).lagrange_coeff()

            v1_req, v2_req, exitFlag = Lambert(r1New, r2New, tof).robust_solve()
            # if bad solution step time and continue
            if exitFlag != 1 or not np.all(np.isfinite(np.r_[v1_req, v2_req])):
                t2 += dt
                continue

            # delta Vs
            dv1 = v1_req - v1New
            dv2 = v2New - v2_req
            dvMag = np.linalg.norm(dv1) + np.linalg.norm(dv2)
            # velocities before transfer where we need to perform Lambert 
            v_pretrans1 = v1New
            v_pretrans2 = v2New

            # capture best solution if it is finite
            if np.isfinite(dvMag) and dvMag < dvMagOld:
                # Earth-intersect guard via perigee altitude
                satOrbit.set_state(r1New, v1_req)
                alt_per = satOrbit.r_per - rE
                # if lowest (perigee) altitude is higher than 100km no earth intersection, save the solution
                if np.isfinite(alt_per) and alt_per >= 100.0:
                    dvMagOld = dvMag
                    bestResult = [
                        orbit1.satnum, orbit2.satnum,
                        r1New, r2New, tof, dv1, dv2, dvMag, # positions, velocities km / km/s
                        t1 + JDays_to_secs(epochTLE), # time of departure (seconds since J2000)
                        t1 - dt, t1 + dt, # departure time bracket (relative time)
                        t2 - dt, t2 + dt, # arrival time bracket (relative time)
                        v_pretrans1, v_pretrans2 # pre transfer velocities, km/ss
                    ]

            t2 += dt # step time
        t1 += dt # step time

    return bestResult

def execute_lambert(orbit1, orbit2, missionTimePre=0, plot=False):
    '''
    Executes the lambert maneuver with the given tof, r1, r2, delta v in the bestGuess array
    the point of this function is to have the satOrbit object do the delta v burns and check for accuracy and plot
    inputs:
        orbit1, orbit2: orbit objects
        bestGuess: bestGuess array from scanLambertCandidates
        missionTimePre: preseeded mission time if the Lambert's maneuver isn't first in the sequence
        plot: bool, decides whether to plot the function 
    returns: dv_total, dv_ledger, missionTime
    '''



# Curtis ch. 6 transfers implementations: combined plane change/velo change, pure plane change, phasing maneuver
def asc_desc_node(row, lhat):
    rhat = row[1] / np.linalg.norm(row[1])
    return 1 if float(np.dot(rhat, lhat)) >= 0.0 else -1

def plane_velo_change_and_phase(orbit1, orbit2, missionTimePre=0, plot=False):
    """
    Implements a combined plane change and velocity change to go from GEO orbit to MEO orbit
    Curtis notes that combining velo change and plane change is strictly cheaper than doing both seperately, so here we are
    Inputs:
        Orbit1s: list of orbits to transfer from
        Orbit2s: list of orbits to transfer to
    Returns:

        dv_total, dv_ledger(dict), end_abs_time, (fig_list or None)

    Notes:

    """
    # start mission timer t=0 at TLE epoch
    missionTime = 0
    # check if orbit1 and orbit2 are in same epoch, if not propagate one to catch up so we can use relative times
    if orbit1.JDsJ2000 > orbit2.JDsJ2000:
        dJdays = orbit1.JDsJ2000 - orbit2.JDsJ2000
        dSecs = JDays_to_secs(dJdays)
        # propagate for offset in seconds
        rTrue, vTrue = Propagate(prop_time=dSecs, Orbit=orbit2).lagrange_coeff()
        orbit2.set_state(rTrue, vTrue) # orbit 2 is now at same absolute time as orbit1
        # use orbit1 TLE epoch as our epoch
        epochTLE = orbit1.JDsJ2000
    # same but now if orbit1 is behind orbit2
    elif orbit1.JDsJ2000 < orbit2.JDsJ2000:
        dJdays = orbit2.JDsJ2000 - orbit1.JDsJ2000
        dSecs = JDays_to_secs(dJdays)
        # propagate for offset in seconds
        rTrue, vTrue = Propagate(prop_time=dSecs, Orbit=orbit1).lagrange_coeff()
        orbit1.set_state(rTrue, vTrue) # orbit 2 is now at same absolute time as orbit1
        # use orbit2 TLE epoch as our epoch
        epochTLE = orbit2.JDsJ2000
    elif orbit1.JDsJ2000 == orbit2.JDsJ2000:
        print('were good')
        # use orbit1 tle epoch as our epoch but doesn't matter could be orbit 2
        epochTLE = orbit1.JDsJ2000
    else:
        print('wat')

    # now that epochs are matched create s/c object copying orbit1
    satOrbit = orbit1.copy()

    # get nodal crossings array for two orbits
    rows = nodal_crossings_array(orbit1, orbit2)
    # get ascending/descending nodes info
    h1hat = orbit1.hvec / np.linalg.norm(orbit1.hvec)
    h2hat = orbit2.hvec / np.linalg.norm(orbit2.hvec)
    lhat = np.cross(h1hat, h2hat)
    lhat /= np.linalg.norm(lhat) # unit vect

    # extract nodal crossings and add info abt asc/desc
    one_rows = [r for r in rows if r[0].satnum == orbit1.satnum]
    two_rows = [r for r in rows if r[0].satnum == orbit2.satnum]
    one_plus  = [r for r in one_rows if asc_desc_node(r, lhat) > 0][0]
    one_minus = [r for r in one_rows if asc_desc_node(r, lhat) < 0][0]
    two_plus  = [r for r in two_rows if asc_desc_node(r, lhat) > 0][0]
    two_minus = [r for r in two_rows if asc_desc_node(r, lhat) < 0][0]

    # check which is more efficient the one_plus & two_minus case or the one_minus & two_plus case
    dv_tot_old = np.inf# init loop var
    for i in range(2):
        if i == 0:
            _, r1, v1, _ = one_plus
            _, r2, v2, _ = two_minus
        elif i == 1:
            _, r1, v1, _ = one_minus
            _, r2, v2, _ = two_plus

        # get delta Vs for combined plane change/velo change
        r1m, r2m = np.linalg.norm(r1), np.linalg.norm(r2)
        rap_temp, rpe_temp = (r1m, r2m) if r1m >= r2m else (r2m, r1m)
        at_temp = 0.5 * (rap_temp + rpe_temp)
        vt_ap_temp = np.sqrt(mu * (2.0 / rap_temp - 1.0 / at_temp))
        vt_pe_temp = np.sqrt(mu * (2.0 / rpe_temp - 1.0 / at_temp))
        r1hat = r1 / r1m
        r2hat = r2 / r2m
        tT_1 = np.cross(h2hat, r1hat)
        tT_1 /= np.linalg.norm(tT_1) #normalize
        tT_2 = np.cross(h2hat, r2hat)
        tT_2 /= np.linalg.norm(tT_2) #normalize
        if np.dot(tT_1, v1) < 0: tT_1 = -tT_1
        if np.dot(tT_2, v2) < 0: tT_2 = -tT_2
        v_dep = (vt_ap_temp if r1m == rap_temp else vt_pe_temp) * tT_1
        v_arr = (vt_pe_temp if r2m == rpe_temp else vt_ap_temp) * tT_2
        dv1 = np.linalg.norm(v_dep - v1)
        dv2 = np.linalg.norm(v2 - v_arr)
        dv_tot = dv1 + dv2
        # collect most efficient
        if dv_tot < dv_tot_old:
            dv_tot_old = dv_tot
            rap, rpe, at, vt_ap, vt_pe = rap_temp, rpe_temp, at_temp, vt_ap_temp, vt_pe_temp
            if i == 0: #one plus two minus case
                one_row, two_row = one_plus, two_minus
            else: #one minus two plus case
                one_row, two_row = one_minus, two_plus

    _, r1_node, v1_node, t1_rel = one_row
    _, r2_node, v2_node, t2_rel = two_row
    # update mission time counter
    missionTime += t1_rel # node time

    # Burn #1 at GEO node
    r_now, v_now = Propagate(t1_rel, Orbit=satOrbit).twobody_ODE()
    satOrbit.set_state(r_now, v_now)
    r1m = np.linalg.norm(r1_node)
    r1hat_node = r1_node / r1m
    tT_1 = np.cross(h2hat, r1hat_node)
    tT_1 /= np.linalg.norm(tT_1)
    if np.dot(tT_1, v1_node) < 0: tT_1 = -tT_1
    v_depart = (vt_ap if r1m == rap else vt_pe) * tT_1
    dv1_vec = v_depart - satOrbit.v
    dv1_mag = np.linalg.norm(dv1_vec)
    # execute burn on satOrbit object
    satOrbit.set_state(satOrbit.r, satOrbit.v + dv1_vec)

    # Coast exactly half the transfer ellipse
    t_half = np.pi * np.sqrt(at ** 3 / mu)
    r_end_tr, v_end_tr = Propagate(t_half, Orbit=satOrbit).twobody_ODE()
    satOrbit.set_state(r_end_tr, v_end_tr)
    # check that we actually got to MEO node
    burn1_check = np.isclose(satOrbit.r, r2_node).all()
    print("[BURN 1 ACCURACY CHECK (T/F)]: ", burn1_check)
    # update time counter
    missionTime += t_half # coast time

    # Burn #2 at MEO node
    h_vec = orbit2.hvec
    hmag = orbit2.hmag
    h_hat = h_vec / np.linalg.norm(h_vec)
    ecc_vec = orbit2.eccvec
    e_mag = np.linalg.norm(ecc_vec)
    p_hat = ecc_vec / e_mag
    q_hat = np.cross(h_hat, p_hat)
    r_nom = np.linalg.norm(r2_node)
    rhat_node = r2_node / r_nom
    that_node = np.cross(h_hat, rhat_node)
    that_node /= np.linalg.norm(that_node)
    f_node = np.arctan2(np.dot(rhat_node, q_hat), np.dot(rhat_node, p_hat)) % (2.0 * np.pi)

    a1, e1 = orbit2.sma, orbit2.ecc
    vt_req = hmag / r_nom
    vr_req = (mu / hmag) * e1 * np.sin(f_node)
    r2_nom = rhat_node * r_nom
    v2_nom = vr_req * rhat_node + vt_req * that_node
    dv2_vec = v2_nom - satOrbit.v
    dv2_mag = np.linalg.norm(dv2_vec)
    satOrbit.set_state(satOrbit.r, satOrbit.v + dv2_vec)
    # check that our state is fully matched now and orbits are the same
    burn2_check = np.isclose(satOrbit.r, r2_node).all() and np.isclose(satOrbit.v, v2_node).all() and np.isclose(satOrbit.energy, orbit2.energy)
    print("[BURN 2 ACCURACY CHECK (T/F)]: ", burn2_check)

    # Phasing maneuver at perigee
    # step spacecraft to be at orbit2 perigee
    t_sincePerigeeSAT = satOrbit.period / (2*np.pi) * (satOrbit.EA - satOrbit.ecc * np.sin(satOrbit.EA))
    t_toPerigeeSAT = satOrbit.period - t_sincePerigeeSAT
    rp_sat, vp_sat = Propagate(prop_time=t_toPerigeeSAT, Orbit=satOrbit).twobody_ODE()
    satOrbit.set_state(rp_sat, vp_sat)
    # update mission time
    missionTime += t_toPerigeeSAT
    # place debris 2 object to the right time (missionTime)
    r2_phase, v2_phase = Propagate(prop_time=missionTime, Orbit=orbit2).twobody_ODE()
    orbit2.set_state(r2_phase, v2_phase)
    # get time until perigee for the MEO object to set period of s/c phasing orbit with kepler
    t_sincePerigee = orbit2.period / (2*np.pi) * (orbit2.EA - orbit2.ecc * np.sin(orbit2.EA))
    t_toPerigee = orbit2.period - t_sincePerigee
    Tphase = t_toPerigee
    # semi major axis of phasing orbit
    sma_phase = (np.sqrt(mu)*Tphase / (2*np.pi))**(2/3)
    # since 2*sma_phase = r_apo + r_peri & r_peri_phase = r_peri_orbit2 by defn of phase maneuver
    ra_phase = 2*sma_phase - orbit2.r_per
    h_phase = np.sqrt(2*mu) * np.sqrt(ra_phase * orbit2.r_per / (ra_phase + orbit2.r_per))
    # get velocities (all tangental, no radial). symmetric maneuver happening at perigee both times
    v3_mag = h_phase / orbit2.r_per
    v4_mag = v3_mag
    # get v vectors using unit vect of satOrbit since it is at perigee (all tangental velocity)
    vp_hat = satOrbit.v / np.linalg.norm(satOrbit.v)
    vEnter_phase = vp_hat * v3_mag # slowing down
    vExit_phase = satOrbit.v # get back into normal orbit
    # delta v vectors and mags
    dv3_vect = satOrbit.v - vEnter_phase
    dv3_mag = np.linalg.norm(dv3_vect)
    # do burn 1 into phasing orbit and set state
    satOrbit.set_state(satOrbit.r, vEnter_phase)
    # propagate for one phase period, set state, update mission timer
    r_afterPhase, v_afterPhase = Propagate(prop_time=Tphase, Orbit=satOrbit).twobody_ODE()
    missionTime += Tphase
    satOrbit.set_state(r_afterPhase, v_afterPhase)
    # check that we got to perigee at same time and position as orbit2 debris
    # propagate orbit2 with time = t_toPerigee = orbitSat.period
    r2_afterPhase, v2_afterPhase = Propagate(prop_time=Tphase, Orbit=orbit2).twobody_ODE()
    orbit2.set_state(r2_afterPhase, v2_afterPhase)
    burn3_check = np.isclose(satOrbit.r, orbit2.r).all()
    print('[BURN 3 ACCURACY CHECK (T/F)]: ', burn3_check)

    # do burn 4
    # use new state to get delta v 4 vector
    dv4_vect = satOrbit.v - vExit_phase
    dv4_mag = np.linalg.norm(dv4_vect)
    satOrbit.set_state(satOrbit.r, vExit_phase)
    # check accuracy
    burn4_check = np.isclose(satOrbit.r, orbit2.r).all() and np.isclose(satOrbit.v, orbit2.v).all() and np.isclose(dv4_mag, dv3_mag)
    print('[BURN 4 ACCURACY CHECK (T/F)]: ', burn4_check)


    dv_total = dv1_mag + dv2_mag + dv3_mag + dv4_mag
    dv_ledger = {
        "Burn #1 (combined @ GEO node)": dv1_mag,
        "Burn #2 (match nominal MEO @ node)": dv2_mag,
        "Burn #3 (enter phasing)": dv3_mag,
        "Burn #4 (exit phasing / match)": dv4_mag,
        "Total G2M": dv_total
    }
    print(dv_ledger)
    return dv_total, dv_ledger, missionTime


def hohmann_helper(orbit1, orbit2, missionTimePre = 0):
    '''
    hohmann helper which applies one hohmann burn to place the radius of the orbit to be similair to the radius of the orbit being transferred into
    the pure point of this is a cheap way to get our fourth maneuver to count
    returns: dv_total, dv_ledger, missionTime
    '''

def execute_mission(GEOTLE, MEOTLE, LEOTLE, LEO2TLE):
    '''
    executes the debris sat mission using:
    GEO to MEO: plane_velo_change_and_phase
    MEO to LEO: hohmann helper + lambert
    LEO to LEO: lambert

    keep track of mission time appropiately

    outputs: dv_total, dv_ledger, missionStartJ2000JDays, missionTimeJDays, figs of each transfer
    '''


if __name__ == "__main__":
    GEODebrisTLE = [29162, 29644, 3623, 3947, 33750, 33453, 29520, 33154, 34710, 27854]
    MEODebrisTLE = [2865, 2864, 2863, 2862, 3291, 3290, 3289, 3288, 3287, 3284]
    LEODebrisTLE = [26561, 35578, 40932, 40933, 40934, 40935]
    LEOODebrisTLE2 = [26561, 35578, 40932, 40933, 40934, 40935]
    orbit1 = TLEOrbit(29162)
    orbit2 = TLEOrbit(2865)
    orbit3 = TLEOrbit(35578)
    #plane_velo_change_and_phase(orbit1, orbit2)
    roughGuess = scanLambertCandidates(orbit2, orbit3)
    print('rough done')
    print(roughGuess)
    fineGuess = scanLambertCandidates(orbit2, orbit3, dt=50, bestGuess=roughGuess)
    print(fineGuess)
    final = scanLambertCandidates(orbit2, orbit3, dt=1, bestGuess=fineGuess)
    print(final)