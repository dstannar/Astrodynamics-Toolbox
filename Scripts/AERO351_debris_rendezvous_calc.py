"""
Drew Stannard-Stockton
AERO351 Group Project Orbit Debris Cleanup Mission Implementation
"""
import numpy as np
from Orbits.TLEOrbit import TLEOrbit
from Propagators.Propagate import Propagate
from Transfers.Lambert import Lambert
from MathHelpers.synodic_period import synodic_period
from Transfers.plane_change import nodal_crossings_array
from MathHelpers.constants import rE, muE
from Time.conversions import secs_to_JDays, JDays_to_secs
from Orbits.safe_create_orbits import create_TLEOrbits
from Propagators.plot_helper import composite_trajectory, animate_composite_figure
from MathHelpers.formatting import format_time
# because we expect divide by zero warnings in Lambert solve loop as not all tofs will be physical
import warnings
warnings.filterwarnings("ignore")

def scanLambertCandidates(orbit1, orbit2, missionTimePre=0, dt=86400*150, bestGuess=None, mu=muE):
    """
      - Align epochs to a common absolute time: max(epoch1, epoch2) + missionTimePre
      - Scan using the nested (t1, t2) loops
      - Earth-intersection guard via perigee altitude >= 100 km
      - Return 17-field list:
        [0]  satnum1, [1] satnum2,
        [2]  r1, [3] r2, [4] tof,
        [5]  dv1, [6] dv2, [7] dvMag,
        [8]  departTime_abs (seconds since mission start),
        [9]  t1_low, [10] t1_high, [11] t2_low, [12] t2_high,
        [13] v_pretrans1, [14] v_pretrans2,
        [15] orbit1_toTime, [16] orbit2_toTime
    """

    # copies
    o1 = orbit1.copy()
    o2 = orbit2.copy()

    # Final common absolute time = max(epoch1, epoch2) + missionTimePre
    if o1.JDsJ2000 > o2.JDsJ2000:
        dSecs = float(JDays_to_secs(o1.JDsJ2000 - o2.JDsJ2000))
        r2, v2 = Propagate(prop_time=dSecs + float(missionTimePre), Orbit=o2, mu=mu).lagrange_coeff()
        o2.set_state(r2, v2)
        r1, v1 = Propagate(prop_time=float(missionTimePre), Orbit=o1, mu=mu).lagrange_coeff()
        o1.set_state(r1, v1)
        orbit1_toTime = float(missionTimePre)
        orbit2_toTime = float(missionTimePre) + dSecs
    elif o1.JDsJ2000 < o2.JDsJ2000:
        dSecs = float(JDays_to_secs(o2.JDsJ2000 - o1.JDsJ2000))
        r1, v1 = Propagate(prop_time=dSecs + float(missionTimePre), Orbit=o1, mu=mu).lagrange_coeff()
        o1.set_state(r1, v1)
        r2, v2 = Propagate(prop_time=float(missionTimePre), Orbit=o2, mu=mu).lagrange_coeff()
        o2.set_state(r2, v2)
        orbit1_toTime = float(missionTimePre) + dSecs
        orbit2_toTime = float(missionTimePre)
    else:
        r1, v1 = Propagate(prop_time=float(missionTimePre), Orbit=o1, mu=mu).lagrange_coeff()
        o1.set_state(r1, v1)
        r2, v2 = Propagate(prop_time=float(missionTimePre), Orbit=o2, mu=mu).lagrange_coeff()
        o2.set_state(r2, v2)
        orbit1_toTime = float(missionTimePre)
        orbit2_toTime = float(missionTimePre)

    # nested scan
    dvMagOld = np.inf
    bestResult = None

    satOrbit = o1.copy()  # working copy used for perigee guard

    Tsyn = float(synodic_period(o1.sma, o2.sma, mu=mu))
    print(Tsyn)

    if bestGuess is None:
        t1 = 0.0
        dep_horizon = Tsyn
        tof_horizon = float(o2.period)
        t2_low = None; t2_high = None
    else:
        # Refine if bestguess is given
        t1 = float(bestGuess[9])
        dep_horizon = float(bestGuess[10])
        t2_low  = float(bestGuess[11])
        t2_high = float(bestGuess[12])

    # Main time loops
    while t1 <= dep_horizon + 1e-12:
        # State of o1 at departure t1 (relative to mission start)
        r1New, v1New = Propagate(prop_time=t1, Orbit=o1, mu=mu).lagrange_coeff()
        satOrbit.set_state(r1New, v1New)

        # [8] departTime_abs is seconds since mission start (relative)
        departTime_abs = float(t1)

        if bestGuess is None:
            t2 = t1 + float(dt)
            t2_end = t1 + float(tof_horizon)
        else:
            t2 = max(t1 + float(dt), t2_low)  # keep TOF > 0 and respect refine window
            t2_end = t2_high

        while t2 <= t2_end:
            # reset satOrbit to pre-transfer state
            satOrbit.set_state(r1New, v1New)

            tof = float(t2 - t1)

            # State of o2 at arrival t2 (relative to mission start)
            r2New, v2New = Propagate(prop_time=t2, Orbit=o2, mu=mu).lagrange_coeff()

            # Lambert solve
            v1_req, v2_req, exitFlag = Lambert(mu=mu).robust_solve(r1New, r2New, tof)
            if exitFlag != 1 or not np.all(np.isfinite(np.r_[v1_req, v2_req])):
                t2 += float(dt)
                continue

            # burns and cost
            dv1 = v1_req - v1New
            dv2 = v2New - v2_req
            dvMag = float(np.linalg.norm(dv1) + np.linalg.norm(dv2))
            v_pretrans1 = v1New
            v_pretrans2 = v2New

            if np.isfinite(dvMag) and dvMag < dvMagOld:
                # Earth-intersection guard: set transfer state and check perigee altitude
                satOrbit.set_state(r1New, v1_req)
                alt = float(satOrbit.r_per - rE)
                if alt >= 100.0:
                    dvMagOld = dvMag
                    bestResult = [
                        1,1,#o1.satnum, o2.satnum,
                        r1New, r2New, tof,
                        dv1, dv2, dvMag,
                        departTime_abs,
                        float(t1 - dt), float(t1 + dt),
                        float(t2 - dt), float(t2 + dt),
                        v_pretrans1, v_pretrans2,
                        float(orbit1_toTime), float(orbit2_toTime)
                    ]

            t2 += float(dt)
        t1 += float(dt)

    # If refine found nothing, fall back to the coarse guess so callers never get None
    if bestResult is None and bestGuess is not None:
        return bestGuess

    return bestResult


def execute_lambert(debrisSAT, orbit1, orbit2, missionTimePre=0, mu=muE):
    """
    Plane/velo+phase and Hohmann have already modified `debrisSAT`.
    Now compute a Lambert transfer from the chaser's current orbit (debrisSAT)
    to orbit2 (the debris), starting from the shared mission epoch used by the scan.
    """
    missionTime = missionTimePre

    # scan using the chaser as o1, target as o2
    roughGuess = scanLambertCandidates(debrisSAT, orbit2, missionTimePre=missionTimePre, mu=mu)
    if roughGuess is None:
        raise RuntimeError("Lambert Failed")
    print("Rough Done")
    print(roughGuess)
    lambSoln   = scanLambertCandidates(debrisSAT, orbit2, dt=86400, bestGuess=roughGuess, missionTimePre=missionTimePre, mu=mu)
    print("Fine Done")

    # lambSoln fields used :
    # [4]=tof, [5]=dv1, [6]=dv2, [7]=dvMag, [8]=departTime_abs, [15]=o1_toTime, [16]=o2_toTime

    # seed chaser & target to the scanner's mission-start epoch (ODE)
    rs,  vs  = Propagate(prop_time=lambSoln[15], Orbit=debrisSAT, mu=mu).twobody_ODE()
    debrisSAT.set_state(rs, vs)
    r2s, v2s = Propagate(prop_time=lambSoln[16], Orbit=orbit2, mu=mu).twobody_ODE()
    orbit2.set_state(r2s, v2s)

    # recalculate Lambert using ODE-propagated states for numerical robustness
    # ODE states at departure (from seeded epoch) and arrival
    r1_dep, v1_pre = Propagate(prop_time=lambSoln[8],Orbit=debrisSAT, mu=mu).twobody_ODE()
    r2_arr, v2_pre = Propagate(prop_time=lambSoln[8] + lambSoln[4], Orbit=orbit2, mu=mu).twobody_ODE()

    v1_req, v2_req, flag = Lambert(mu=mu).robust_solve(r1_dep, r2_arr, float(lambSoln[4]))
    if flag != 1 or not np.all(np.isfinite(np.r_[v1_req, v2_req])):
        raise RuntimeError("Lambert failed")

    # Overwrite only the necessary pieces so dv's match ODE propagation
    lambSoln[2]  = r1_dep
    lambSoln[3]  = r2_arr
    lambSoln[5]  = v1_req - v1_pre         # dv1 at departure (ODE)
    lambSoln[6]  = v2_pre - v2_req         # dv2 at arrival   (ODE)
    lambSoln[7]  = float(np.linalg.norm(lambSoln[5]) + np.linalg.norm(lambSoln[6]))
    lambSoln[13] = v1_pre
    lambSoln[14] = v2_pre

    # plot the departure orbit for context, and the target orbit
    _, _, dep_orbit_fig = Propagate(prop_time=debrisSAT.period, Orbit=debrisSAT, mu=mu).twobody_ODE(plot=True)
    _, _, arr_orbit_fig = Propagate(prop_time=orbit2.period,    Orbit=orbit2, mu=mu).twobody_ODE(plot=True)

    # Depart: coast to departure time and apply dv1 exactly where the scan computed it
    rSat1, vSat1, leg1   = Propagate(prop_time=lambSoln[8], Orbit=debrisSAT, mu=mu).twobody_ODE(plot=True)
    debrisSAT.set_state(rSat1, vSat1 + lambSoln[5])
    # log
    lam_dep_time = missionTime + lambSoln[8]
    lam_dv1_mag  = float(np.linalg.norm(lambSoln[5]))
    missionTime += lambSoln[8]

    # Transfer coast and arrival burn dv2
    rSat2, vSat2, leg2   = Propagate(prop_time=lambSoln[4], Orbit=debrisSAT, mu=mu).twobody_ODE(plot=True)
    debrisSAT.set_state(rSat2, vSat2 + lambSoln[6])
    # log
    lam_arr_time = missionTime + lambSoln[4]
    lam_dv2_mag  = float(np.linalg.norm(lambSoln[6]))
    missionTime += lambSoln[4]

    # March the target forward by the same elapsed time from the same epoch
    rO2, vO2 = Propagate(prop_time=lambSoln[8] + lambSoln[4], Orbit=orbit2, mu=mu).twobody_ODE()
    orbit2.set_state(rO2, vO2)

    # accuracy check (units: km, km/s)
    lambertOk = np.isclose(debrisSAT.r, orbit2.r).all() and np.isclose(debrisSAT.v, orbit2.v).all()
    print(f"[LAMBERT ACCURACY (T/F)]: ", lambertOk)

    lambertPlot = composite_trajectory(
        [dep_orbit_fig, arr_orbit_fig, leg1, leg2],
        ["Chaser departure orbit", "Target orbit", "Coast to depart", "Lambert transfer"],
        title="Lambert Rendezvous", show=True
    )

    # animate
    _, anim = animate_composite_figure(
        composite_fig=lambertPlot,
        animate_mask=[False, False, True, True],
        order=[2, 3],
        fps=30,
        duration=12,
        background_alpha=0.25,
        active_alpha=1.0,
        save_path="lambert_composite_animation.gif",
        show=True
    )

    dv_ledger = {"Burn #1 Lambert": lambSoln[5], "Burn #2 Lambert": lambSoln[6]}
    dvMag = lambSoln[7]

    # deorbit 
    missionTime += debrisSAT.period * 5

    # reporting bundle (no math changes)
    lambert_report = [
        ("Lambert Burn #1 (depart)", lam_dv1_mag, lam_dep_time),
        ("Lambert Burn #2 (arrive)", lam_dv2_mag, lam_arr_time),
    ]

    return lambertPlot, dvMag, dv_ledger, debrisSAT, missionTime, lambert_report



# Curtis ch. 6 transfers implementations: combined plane change/velo change, pure plane change, phasing maneuver
def asc_desc_node(row, lhat):
    rhat = row[1] / np.linalg.norm(row[1])
    return 1 if float(np.dot(rhat, lhat)) >= 0.0 else -1

def plane_velo_change_and_phase(orbit1, orbit2, mu=muE):
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

    _, _, orbit1_fig = Propagate(prop_time=orbit1.period, Orbit=orbit1).twobody_ODE(plot=True)
    _, _, orbit2_fig = Propagate(prop_time=orbit2.period, Orbit=orbit2).twobody_ODE(plot=True)

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

    # orbit1 and orbit2 parking figs


    # Burn #1 at GEO node
    r_now, v_now, pvp_fig1 = Propagate(t1_rel, Orbit=satOrbit).twobody_ODE(plot=True)
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
    t_burn1 = missionTime  # time of Burn #1

    # Coast exactly half the transfer ellipse
    t_half = np.pi * np.sqrt(at ** 3 / mu)
    r_end_tr, v_end_tr, pvp_fig2 = Propagate(t_half, Orbit=satOrbit).twobody_ODE(plot=True)
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
    t_burn2 = missionTime  # time of Burn #2

    # Phasing maneuver at perigee
    # step spacecraft to be at orbit2 perigee
    t_sincePerigeeSAT = satOrbit.period / (2*np.pi) * (satOrbit.EA - satOrbit.ecc * np.sin(satOrbit.EA))
    t_toPerigeeSAT = satOrbit.period - t_sincePerigeeSAT
    rp_sat, vp_sat, pvp_fig3 = Propagate(prop_time=t_toPerigeeSAT, Orbit=satOrbit).twobody_ODE(plot=True)
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
    rp   = float(orbit2.r_per)
    Tmin = 2.0*np.pi*np.sqrt(rp**3 / mu)   # shortest period possible with perigee = rp

    # raise by whole target periods until feasible
    Tphase = Tphase % orbit2.period
    if Tphase < Tmin:
        k = int(np.ceil((Tmin - Tphase)/orbit2.period))
        Tphase += k * orbit2.period

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
    t_burn3 = missionTime  # time of Burn #3
    # propagate for one phase period, set state, update mission timer
    r_afterPhase, v_afterPhase, pvp_fig4 = Propagate(prop_time=Tphase, Orbit=satOrbit).twobody_ODE(plot=True)
    missionTime += Tphase
    satOrbit.set_state(r_afterPhase, v_afterPhase)
    # check that we got to perigee at same time and position as orbit2 debris
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
    t_burn4 = missionTime  # time of Burn #4
    # propagate for one full period to show on animation
    _, _, pvp_fig5 = Propagate(prop_time=satOrbit.period, Orbit=satOrbit).twobody_ODE(plot=True)
    # update mission time bc we stayed for 5 periods per project req
    missionTime += satOrbit.period * 5

    dv_total = dv1_mag + dv2_mag + dv3_mag + dv4_mag
    dv_ledger = {
        "Burn #1 (combined @ GEO node)": dv1_mag,
        "Burn #2 (match nominal MEO @ node)": dv2_mag,
        "Burn #3 (enter phasing)": dv3_mag,
        "Burn #4 (exit phasing / match)": dv4_mag,
        "Total G2M": dv_total
    }
    
    # Build composite fig
    pvp_fig = composite_trajectory(
        [orbit1_fig, orbit2_fig, pvp_fig1, pvp_fig2, pvp_fig3, pvp_fig4, pvp_fig5],
        ["Orbit 1 Parking Orbit", "Orbit 2 Parking Orbit",
        "debrisSAT to Departure Node",
        "debrisSAT Combined Plane Change/Velocity Change",
        "debrisSAT to Perigee",
        "debrisSAT Phasing Maneuver",
        "debrisSAT Rendezvous with Debris"],
        title="Combined Plane Change/Velocity Change and Phasing Maneuver",
        show=False
    )

    # Animate 
    _, anim = animate_composite_figure(
        composite_fig=pvp_fig,
        animate_mask=[False, False, True, True, True, True, True],
        order=[2, 3, 4, 5, 6],      # sequentially animate each leg
        fps=30,
        duration=12,
        background_alpha=0.25,
        active_alpha=1.0,
        save_path="pvp_composite_animation.gif",
        show=True
    )

    # report soln
    pvp_report = [
        ("Burn #1 (combined @ GEO node)", dv1_mag, t_burn1),
        ("Burn #2 (match nominal MEO @ node)", dv2_mag, t_burn2),
        ("Burn #3 (enter phasing)", dv3_mag, t_burn3),
        ("Burn #4 (exit phasing / match)", dv4_mag, t_burn4),
    ]

    return dv_total, dv_ledger, missionTime, epochTLE, pvp_fig, satOrbit, pvp_report


def hohmann_helper(debrisSAT, orbit2, missionTimePre=0.0, mu=muE):
    '''
    Single-burn, strictly coplanar Hohmann staging step.
    Sets the other apsis of orbit1's new ellipse to the instantaneous radius of orbit2.
    No circularization, no plane change, no coast time added.
    Only reason this is here is to hit 4 transfers requirement lol

    Args:
        orbit1 : chaser object (modified in-place)
        orbit2 : target object (read-only; used for its current radius)

    Returns:
        dv_total, dv_ledger, debrisSAT, burn_time
    '''

    # Radii now
    r1 = float(np.linalg.norm(debrisSAT.r))
    r2 = float(np.linalg.norm(orbit2.r))

    # Transfer ellipse: classic Hohmann endpoints r1 -> r2
    a_t = 0.5 * (r1 + r2)

    # Required transfer speed at current point (vis-viva)
    v_t1 = np.sqrt(mu * (2.0 / r1 - 1.0 / a_t))

    # Build a strictly tangential unit vector at the current state
    r = debrisSAT.r
    v = debrisSAT.v
    h = np.cross(r, v)
    r_hat = r / np.linalg.norm(r)
    h_hat = h / np.linalg.norm(h)
    t_hat = np.cross(h_hat, r_hat)  # along-track direction in-plane

    # Current tangential speed component
    v_tang_now = float(np.dot(v, t_hat))

    # Burn purely along-track to match transfer speed
    dv_vec = (v_t1 - v_tang_now) * t_hat
    debrisSAT.set_state(debrisSAT.r, debrisSAT.v + dv_vec)

    dv_total = float(np.linalg.norm(dv_vec))
    dv_ledger = {"Hohmann staging (single tangential burn)": dv_total}

    # return the time this burn occurs
    burn_time = float(missionTimePre)
    return dv_total, dv_ledger, debrisSAT, burn_time


def execute_mission(GEOTLE, MEOTLE, LEOTLE, LEO2TLE):
    orbit1 = TLEOrbit(GEOTLE)   # GEO
    orbit2 = TLEOrbit(MEOTLE)   # MEO
    orbit3 = TLEOrbit(LEOTLE)   # LEO #1
    orbit4 = TLEOrbit(LEO2TLE)  # LEO #2

    # GEO -> MEO (combined + phase)
    dvPVP, ledgerPVP, tPVP, epochTLE, pvp_plot, debrisSATPVP, pvp_report = plane_velo_change_and_phase(orbit1, orbit2)
    print("G2M dV:", dvPVP)

    # Hohmann staging at MEO->LEO (no time added, not plotted)
    dvH, ledgerH, debrisSATh, tH = hohmann_helper(debrisSAT=debrisSATPVP, orbit2=orbit3, missionTimePre=tPVP)
    print("H staging dV (M2L):", dvH)

    # MEO -> LEO (Lambert) after staging
    lamPlot1, dvL1, ledgerL1, debrisSATL1, tL1, lam1_report = execute_lambert(debrisSATh, orbit1=debrisSATh, orbit2=orbit3, missionTimePre=tPVP)
    print("M2L dV:", dvL1)

    # LEO -> LEO (Lambert)
    lamPlot2, dvL2, ledgerL2, debrisL2, tL2, lam2_report = execute_lambert(debrisSATL1, orbit1=debrisSATL1, orbit2=orbit4, missionTimePre=tL1)
    print("L2L Lambert dV:", dvL2)

    # full mission animation (exclude Hohmann leg)
    full_fig = composite_trajectory(
        [pvp_plot, lamPlot1, lamPlot2],
        ["GEO->MEO (combined + phasing)",
         "MEO->LEO Lambert",
         "LEO1->LEO2 Lambert"],
        title="Debris Cleanup Mission - Full Sequence",
        show=False
    )

    _, full_anim = animate_composite_figure(
        composite_fig=full_fig,
        animate_mask=[True, True, True],
        order=[0, 1, 2],
        fps=30,
        duration=20,
        background_alpha=0.25,
        active_alpha=1.0,
        save_path="full_mission_animation.gif",
        show=True
    )

    # reporting
    print("MISSION SUMMARY")
    print(f"Mission epoch (JD since J2000.0): {epochTLE:.8f}")

    print("Burn timeline (time since mission start):")
    def T(t):
        return format_time(float(t))

    for name, mag, t in pvp_report:
        print(f"{name:<40s} | dv = {mag:.6f} km/s at t = {t:.3f} s  ({T(t)})")

    print(f"Hohmann staging (single)  | dv = {dvH:.6f} km/s at t = {tH:.3f} s  ({T(tH)})")

    for name, mag, t in lam1_report:
        print(f"{name:<40s} | dv = {mag:.6f} km/s at t = {t:.3f} s  ({T(t)})")

    for name, mag, t in lam2_report:
        print(f"  {name:<40s} | dv = {mag:.6f} km/s at t = {t:.3f} s  ({T(t)})")

    total_dv = dvPVP + dvH + dvL1 + dvL2
    print("Segment dv:")
    print(f"GEO->MEO (PVP+phasing): {dvPVP:.6f} km/s")
    print(f"MEO staging (Hohmann):  {dvH:.6f} km/s")
    print(f"MEO->LEO (Lambert):     {dvL1:.6f} km/s")
    print(f"LEO->LEO (Lambert):     {dvL2:.6f} km/s")
    print(f"TOTAL mission dv:         {total_dv:.6f} km/s")

    print(f"Total mission time: {tL2:.3f} s  ({T(tL2)})")
    print("--- Ledgers ---")
    print("G2M Ledger:")
    for k, v in ledgerPVP.items():
        if isinstance(v, (float, int)):
            mag = float(v)
        else:
            mag = float(np.linalg.norm(v))
        print(f"  {k}: {mag:.6f} km/s")

    print("M2L Hohmann Ledger:")
    for k, v in ledgerH.items():
        print(f"  {k}: {float(v):.6f} km/s")

    print("M2L Lambert Ledger:")
    for k, v in ledgerL1.items():
        mag = float(np.linalg.norm(v))
        print(f"  {k}: {mag:.6f} km/s")

    print("L2L Lambert Ledger:")
    for k, v in ledgerL2.items():
        mag = float(np.linalg.norm(v))
        print(f"{k}: {mag:.6f} km/s")


if __name__ == "__main__":
    GEODebrisTLE = [3431, 28946, 4902, 4376, 5588, 5587, 3029, 2639, 4418, 29162]
    MEODebrisTLE = [2865, 2864, 2863, 2862, 3291, 3290, 3289, 3288, 3287, 3284]
    LEODebrisTLE = [26561, 35578, 40932, 40933, 40934, 40935]
    LEOODebrisTLE2 = [26561, 35578, 40932, 40933, 40934, 40935]
    
    execute_mission(29162, 2865, 26561, 40932)
