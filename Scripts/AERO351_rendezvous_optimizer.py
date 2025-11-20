'''
Drew Stannard-Stockton
AERO351 Group Project Orbit Debris Cleanup Mission Implementation
'''
import numpy as np
from Orbits.TLEOrbit import TLEOrbit
from Orbits.KeplerianOrbit import KeplerianOrbit
from Propagators.Propagate import Propagate
from Transfers.Lambert import Lambert
from MathHelpers.wrap_angles import wrap_to_pi
from MathHelpers.synodic_period import synodic_period
from Transfers.plane_change import plane_change
from Transfers.phase import phasing_maneuver
from Transfers.plane_change import best_nodal_crossing
from MathHelpers.constants import rE
from Time.conversions import secs_to_JDays, JDays_to_secs, dateTime_to_JDays
from Orbits.safe_create_orbits import create_TLEOrbits
from Propagators.plot_helper import composite_trajectory
from MathHelpers.formatting import format_time
# we should expect divide by zero warnings for bad lamberts transfer, so go ahead and:
import warnings
warnings.filterwarnings("ignore")


def scanLambertCandidates(Orbit1s, Orbit2s, dt=100, bestGuess=None):
    '''
    This function scans across time and solves Lambert transfers to get the smallest dv transfer for various objects at various times
    The function is designed to be called multiple times with finer timesteps and reseeded with the last solution bracket to speed convergence
    Inputs:
        Orbit1s: list of orbit objects to try transfering from
        Orbit2s: list of orbit objects to try transfering to
        dt: time step. default 100s (rough scan)
        bestGuess: best guess from last function call when refining solution. defaults to None
    Outputs:
        bestResult array. Contains:
            orbit1.satnum: NORAD ID of object to transfer from
            orbit2.satnum: NORAD ID of object to transfer to
            r1New: position of object 1 at departure in km
            r2New: position of object 2 at arrival in km
            tof: time of flight between r1New, r2New in seconds
            dv1: first burn vector, km/s
            dv2: second burn vector, km/s
            dvMag: total cost of transfer
            departTime: time of departure in seconds from J2000.0
            t1 - dt: left hand bracket of departure time in secs since object 1 TLE epoch
            t1 + dt: right hand bracket of departure time in secs since object 1 TLE epoch
            t2 - dt: left hand bracket of arrival time in secs since object 2 TLE epoch 
            t2 + dt: right hand bracket of arrival time in secs since object 2 TLE epoch
            v_pretrans1: velocity of first object pre transfer (for state setting) km/s
            v_pretrans2: velocity of second object pre transfer (for state setting) km/s
    
    '''
    dvMagOld = np.inf
    bestResult = None

    # If refining, pull the winning IDs so we only re-check that pair
    if bestGuess is not None:
        id1_ref = int(bestGuess[0])
        id2_ref = int(bestGuess[1])

    for orbit1 in Orbit1s:
        # create satellite object to track transfers, starting at orbit1 state
        satOrbit = orbit1.copy()
        for orbit2 in Orbit2s:

            # Skip non-winning pairs during refine
            if bestGuess is not None and (orbit1.satnum != id1_ref or orbit2.satnum != id2_ref):
                continue
            # get synodic period to propagate through for
            Tsyn = synodic_period(orbit1.sma, orbit2.sma)
            # set time ranges
            if bestGuess is None:
                dep_horizon = Tsyn
                tof_horizon = orbit2.period
                t1 = 0
            else:
                # bestGuess: [orbit1.satnum, orbit2.satnum, r1New, r2New, tof, dv1, dv2, dvMag, departTime, t1 - dt, t1 + dt, t2 - dt, t2 + dt]
                t1_low  = float(bestGuess[9]) 
                t1_high = float(bestGuess[10])
                t2_low  = float(bestGuess[11]) 
                t2_high = float(bestGuess[12])
                dep_horizon = t1_high
                t1 = t1_low
                t2_start_abs = t2_low
                t2_end_abs   = t2_high
            # evil nested propagation loop using lagrange coeff propagation for speed
            while t1 <= dep_horizon:
                # set satellite state to be at orbit1 state
                r1New, v1New = Propagate(t1, Orbit=orbit1).lagrange_coeff()
                satOrbit.set_state(r1New, v1New)
                TLE_epoch_JD = orbit1.JDsJ2000 # time when TLE was scraped, Julian days since J2000
                TLE_epoch_s = JDays_to_secs(TLE_epoch_JD)
                departTime = TLE_epoch_s + t1 # departure time in seconds, from J2000

                if bestGuess is None:
                    t2 = t1 + dt
                    t2_end = t1 + tof_horizon
                else:
                    t2 = max(t1 + dt, t2_start_abs)   # keep TOF > 0 and respect refine window
                    t2_end = t2_end_abs

                while t2 <= t2_end:
                    # reset satellite state
                    satOrbit.set_state(r1New, v1New)
                    tof = t2 - t1
                    r2New, v2New = Propagate(t2, Orbit=orbit2).lagrange_coeff()
                    # solve lambert's
                    v1_req, v2_req, exitFlag = Lambert(r1New, r2New, tof).robust_solve()

                    # skip that time for bad lambert transfers
                    if exitFlag != 1 or not np.all(np.isfinite(np.r_[v1_req, v2_req])):
                        t2 += dt
                        continue

                    # calculate dv1 and dv2 burn vectors, get total dvMag cost
                    dv1 = v1_req - v1New
                    dv2 = v2New - v2_req
                    dvMag = np.linalg.norm(dv1) + np.linalg.norm(dv2)
                    v_pretrans1 = v1New
                    v_pretrans2 = v2New

                    if np.isfinite(dvMag) and dvMag < dvMagOld:
                        # check for earth intersection
                        earth_intersect = False # resent intersection flag
                        # set satOrbit state
                        satOrbit.set_state(r1New, v1_req)
                        # check if satOrbit in new state (on transfer arc) has z_per < 100km
                        alt = satOrbit.r_per - rE
                        if alt < 100:
                            earth_intersect = True

                        # if doesn't intersect earth save as new best solution
                        if earth_intersect == False:
                            dvMagOld = dvMag
                            bestResult = [orbit1.satnum, orbit2.satnum, r1New, r2New, tof, dv1, dv2, dvMag, departTime,
                                    t1 - dt, t1 + dt, t2 - dt, t2 + dt, v_pretrans1, v_pretrans2]

                    t2 += dt
                t1 += dt

    return bestResult


def get_rendezvous(G2M=True, M2L=True, L2L=True, full_soln=True):
    # debris TLEs
    GEODebrisTLE = [9416]
    MEODebrisTLE = [
        62720, 31115, 25874, 25770, 25771, 25773, 25851, 25886, 25852, 25873, 25883, 25884,
        23833, 23027, 22779, 22446, 20061, 20533, 22657, 24320, 20302, 21890, 22581, 22108,
        22700, 20724, 20185, 20452, 19802, 22275, 26690, 21930, 22014, 22231, 22877, 38858,
        25030, 23953, 20959, 38774, 20361, 20830, 21552, 41315, 28922, 26390, 32781, 26626,
        11783, 15271, 10893, 11054, 14189, 15039, 16129, 25744, 11141, 41554, 37137, 37138,
        37139, 26987, 26988, 26989, 28508, 28509, 28510, 23045, 28112, 23044, 28113, 28114,
        23043, 23512, 23511, 23513, 25594, 10684, 25593, 25595, 22056, 22057, 22058, 32394,
        21218, 21217, 21216, 11690, 15260, 13606, 15261, 33466, 33467, 33468, 14260, 29670,
        29672, 23398, 29671, 23397, 23396, 13607, 14258, 15259, 13603, 15698, 15699, 15697,
        16963, 21853, 21854, 21855, 16961, 16962, 22512, 22513, 22514, 21006, 21008, 19749,
        21007, 19750, 19165, 19163, 19164, 20620, 20619, 20621, 36113, 14259, 20024, 20025,
        37829, 26566, 26564, 26565, 19502, 19501, 19503, 23204, 23205, 23203, 23736, 23620,
        23622, 23734, 23735, 23621, 37372, 27617, 27618, 36401, 27619, 36400, 33378, 37938,
        33379, 33380, 28915, 28917, 18355, 28916, 32277, 18357, 18356, 16396, 16397, 16398,
        14592, 14977, 14978, 14591, 14590, 14979, 26483, 53106, 53110, 53111
    ]
    LEODebrisTLE = [
        26561, 40933, 40935, 40932, 40934, 35578, 37840, 25791,
        44628, 25520, 27783, 4952, 41337, 41338, 41339, 25509,
        5485, 1641, 56415, 57705
    ]
    LEOODebrisTLE2 = [26561]

    # Create debris orbit objects
    GEOOrbit = create_TLEOrbits(GEODebrisTLE)
    MEOOrbit = create_TLEOrbits(MEODebrisTLE)
    LEOOrbit = create_TLEOrbits(LEODebrisTLE)
    LEOOrbit2 = create_TLEOrbits(LEOODebrisTLE2)

    # start time tracker, mission starts at xx/xx/xxxx at xx:xx (t0 = x JD since J2000.0)
    missionEpoch = GEOOrbit[0].JDsJ2000 # start time of mission in Julian days since J2000 (time when first TLE was pulled)
    missionTime = 0 # seconds since mission epoch
    missionCost = 0 # delta V, km/s

    # Create satellite orbit object with initial state shared with GEO object
    debrisSATOrbit = KeplerianOrbit(GEOOrbit[0].r, GEOOrbit[0].v)

    print('[CHECKPOINT]: Done creating orbit objects')
    # Create graph of chosen orbit for one period via ODE propagation
    _, _, GEOPark_fig = Propagate(GEOOrbit[0].period, GEOOrbit[0]).twobody_ODE(plot=True)
    _, _, MEOPark_fig = Propagate(MEOOrbit[0].period, MEOOrbit[0]).twobody_ODE(plot=True)
    _, _, LEOPark_fig = Propagate(LEOOrbit[0].period, LEOOrbit[0]).twobody_ODE(plot=True)
    _, _, LEO2Park_fig = Propagate(LEOOrbit2[0].period, LEOOrbit2[0]).twobody_ODE(plot=True)

    # if G2M = True, evaluate GEO to MEO transfer and print results
    if G2M == True:
        print('not yet implemented')
        # Determine state of GEO object and MEO object at crossings of shared nodal line
        # Determine mission start time W.R.T J2000.0 epoch
        # determine Apse line rotation dV magnitude, rotate debrisSAT's apse line manually
        # determine combined burn (hohmann + plane change) dV, change debrisSAT's r_apo, r_per, inc, raan manually
        # determine phase burn dV, change debrisSAT's TA manually
        # Create graph of chosen orbit for one period via ODE propagation
        _, _, GEOPark_fig = Propagate(GEOOrbit[0].period, GEOOrbit[0]).twobody_ODE(plot=True)
        _, _, MEOPark_fig = Propagate(MEOOrbit[0].period, MEOOrbit[0]).twobody_ODE(plot=True)

    if M2L == True:
        # SOLVE LAMBERT
        # call scanLambertCandidates with the default rough time step to bracket solution
        roughBestM2L = scanLambertCandidates(MEOOrbit, LEOOrbit)
        print('[CHECKPOINT]: Rough M2L scan done')
        # call scanLambertCandidates with fine time step for true solution, handing rough output as initial guess
        fineBestM2L = scanLambertCandidates(MEOOrbit, LEOOrbit, dt = 1, bestGuess=roughBestM2L)
        # unpack sol'n and display to command window
        r1_M2L = fineBestM2L[2]
        r2_M2L = fineBestM2L[3]
        tof_M2L = fineBestM2L[4]
        dv1_M2L = fineBestM2L[5]
        dv2_M2L = fineBestM2L[6]
        dvMag_M2L = fineBestM2L[7]
        v_pretransM2L1 = fineBestM2L[13] 
        v_pretransM2L2 = fineBestM2L[14]
        departTime_M2L = fineBestM2L[8] # depart time in seconds w.r.t 
        departTime_M2L_JD = secs_to_JDays(departTime_M2L)
        print('[MEO TO LEO SOLUTION]')
        print('MEO Object NORAD ID: ', fineBestM2L[0])
        print('LEO Object NORAD ID: ', fineBestM2L[1])
        print('Time of Departure W.R.T J2000 epoch (Julian Days): ', departTime_M2L_JD)
        print('Position of MEO object at transfer (km): ', r1_M2L)
        print('Position of LEO object at transfer (km): ', r2_M2L)
        tof_M2L_clean, unit = format_time(tof_M2L)
        print('Transfer time of flight (',unit,'):', tof_M2L_clean)
        print('dv1 Vector (km/s): ', dv1_M2L)
        print('dv2 Vector (km/s): ', dv2_M2L)
        print('Total dv magnitude (km/s): ', dvMag_M2L)

        # EXECUTE LAMBERT TRANSFER
        # update debris sat to be at MEO orbit
        debrisSATOrbit.set_state(r1_M2L, v_pretransM2L1)
        # change debrisSAT's velocity vector, propagate and graph rendezvous with LEO object
        v_atMEO = debrisSATOrbit.v # km/s, vector
        v_trans_M2L = v_atMEO + dv1_M2L # vector addition
        # set satellite state to transfer position and give it transfer velocity (dV burn 1)
        debrisSATOrbit.set_state(rnew = r1_M2L, vnew = v_trans_M2L)
        # propagate for Lambert tof with ODE and save in progress figure to append to later
        rArrive_M2L, vArrive_M2L, M2L_transFig = Propagate(prop_time = tof_M2L, Orbit=debrisSATOrbit).twobody_ODE(plot=True)
        # check that our impulse actually got us to the position of the LEO object with np.isclose.all()
        M2L_success = np.isclose(r2_M2L, rArrive_M2L).all()
        print('[T/F ACCURACY CHECK]: ', M2L_success)
        # do another deltaV burn to stay with LEO object
        v_rendezvous_M2L = vArrive_M2L + dv2_M2L # vector addition, km/s
        debrisSATOrbit.set_state(rnew = rArrive_M2L, vnew = v_rendezvous_M2L)

        # update time and cost counter
        missionTime = missionTime + tof_M2L
        missionCost = missionCost + dvMag_M2L

        # Create graph of chosen orbit for one period via ODE propagation
        MEOIdx = MEODebrisTLE.index(fineBestM2L[0])
        LEOIdx = LEODebrisTLE.index(fineBestM2L[1])
        _, _, MEOPark_fig = Propagate(MEOOrbit[MEOIdx].period, MEOOrbit[MEOIdx]).twobody_ODE(plot=True)
        _, _, LEOPark_fig = Propagate(LEOOrbit[LEOIdx].period, LEOOrbit[LEOIdx]).twobody_ODE(plot=True)

        # PLOT RESULTS - one graph w/ MEO orbit, LEO orbit, transfer arc
        M2L_fig = composite_trajectory([MEOPark_fig, LEOPark_fig, M2L_transFig], labels = ['MEO Debris Orbit', 'LEO Debris Orbit', 'Lambert Transfer Arc'], title = 'MEO to LEO Transfer', show=True)

    if L2L == True:
        # SOLVE LAMBERT
        # call scanLambertCandidates with the default rough time step to bracket solution
        roughBestL2L = scanLambertCandidates(LEOOrbit, LEOOrbit2)
        print('[CHECKPOINT]: Rough L2L scan done')
        # call scanLambertCandidates with fine time step for true solution, handing rough output as initial guess
        fineBestL2L = scanLambertCandidates(LEOOrbit, LEOOrbit2, dt = 1, bestGuess=roughBestL2L)
        # unpack sol'n and display to command window
        r1_L2L = fineBestL2L[2]
        r2_L2L = fineBestL2L[3]
        tof_L2L = fineBestL2L[4]
        dv1_L2L = fineBestL2L[5]
        dv2_L2L = fineBestL2L[6]
        dvMag_L2L = fineBestL2L[7]
        departTime_L2L = fineBestL2L[8]
        v_pretransL2L1 = fineBestL2L[13] 
        v_pretransL2L2 = fineBestL2L[14]
        departTime_L2L_JD = secs_to_JDays(departTime_L2L)
        print('[LEO TO LEO SOLUTION]')
        print('LEO1 Object NORAD ID: ', fineBestL2L[0])
        print('LEO2 Object NORAD ID: ', fineBestL2L[1])
        print('Time of Departure W.R.T J2000 epoch (Julian Days): ', departTime_L2L_JD)
        print('Position of LEO 1 object at transfer (km): ', r1_L2L)
        print('Position of LEO 2 object at transfer (km): ', r2_L2L)
        tof_L2L_clean, unit = format_time(tof_L2L)
        print('Transfer time of flight (',unit,'):', tof_L2L_clean)
        print('dv1 Vector (km/s): ', dv1_L2L)
        print('dv2 Vector (km/s): ', dv2_L2L)
        print('Total dv magnitude (km/s): ', dvMag_L2L)

        # EXECUTE LAMBERT TRANSFER
         # update debris sat to be at MEO orbit
        debrisSATOrbit.set_state(r1_L2L, v_pretransL2L1)
        # change debrisSAT's velocity vector, propagate and graph rendezvous with LEO2 object
        v_atLEO = debrisSATOrbit.v # km/s, vector
        v_trans_L2L = v_atLEO + dv1_L2L # vector addition
        # set satellite state to transfer position and give it transfer velocity (dV burn 1)
        debrisSATOrbit.set_state(rnew = r1_L2L, vnew = v_trans_L2L)
        # propagate for Lambert tof with ODE and save in progress figure to append to later
        rArrive_L2L, vArrive_L2L, L2L_transFig = Propagate(prop_time = tof_L2L, Orbit=debrisSATOrbit).twobody_ODE(plot=True)
        # check that our impulse actually got us to the position of the LEO2 object with np.isclose.all()
        L2L_success = np.isclose(r2_L2L, rArrive_L2L).all()
        print('[T/F ACCURACY CHECK]: ', L2L_success)
        # do another deltaV burn to stay with LEO2 object
        v_rendezvous_L2L = vArrive_L2L + dv2_L2L # vector addition, km/s
        debrisSATOrbit.set_state(rnew = rArrive_L2L, vnew = v_rendezvous_L2L)

        # update mission time and cost counter
        missionTime = missionTime + tof_L2L
        missionCost = missionCost + dvMag_L2L

        # Create graph of chosen orbit for one period via ODE propagation
        LEOIdx = LEODebrisTLE.index(fineBestL2L[0])
        LEO2Idx = LEOODebrisTLE2.index(fineBestL2L[1])
        _, _, LEOPark_fig = Propagate(LEOOrbit[LEOIdx].period, LEOOrbit[LEOIdx]).twobody_ODE(plot=True)
        _, _, LEO2Park_fig = Propagate(LEOOrbit2[LEO2Idx].period, LEOOrbit2[LEO2Idx]).twobody_ODE(plot=True)

        # PLOT RESULTS - one graph w/ MEO orbit, LEO orbit, transfer arc
        L2L_fig = composite_trajectory([LEOPark_fig, LEO2Park_fig, L2L_transFig], labels = ['LEO Debris Orbit', 'LEO Debris 2 Orbit', 'Lambert Transfer Arc'], title = 'LEO to LEO Transfer', show=True)

        
    if full_soln == True:
        # export CSV with all pertinent values, make animation, show final graph with all transfers
        print('not yet implemented')




if __name__ == '__main__':
    
    get_rendezvous(G2M=False, M2L=True, L2L=False, full_soln=False)



'''
    full sets
        GEODebrisTLEs = [
        4250, 4068, 4297, 26824, 4353, 41748, 32019, 3431, 28946, 3428, 4902, 29162,
        4376, 29644, 5588, 5587, 33750, 2865, 33453, 2864, 29520, 3691, 33154, 2863,
        2862, 7318, 34710, 3029, 27854, 2639, 4418, 43241, 28884, 3430, 36358, 37804,
        8513, 32018, 27378, 2969, 2717, 37150, 28542, 29155, 40100, 23715, 9416, 28472,
        43917, 7547, 9503, 29648, 32293, 28902, 32478, 3947, 2608, 42763, 28790, 55138,
        3623, 28238, 26624, 10159, 25954, 26694, 37746, 40099, 33460, 10365, 27603, 23842,
        4881, 28903, 4478, 55683
    ]

    # sorted for debris meo objects with small inclination (approx < 7 or so)
    MEODebrisTLEs = [
        62720, 31115, 25874, 25770, 25771, 25773, 25851, 25886, 25852, 25873, 25883, 25884,
        23833, 23027, 22779, 22446, 20061, 20533, 22657, 24320, 20302, 21890, 22581, 22108,
        22700, 20724, 20185, 20452, 19802, 22275, 26690, 21930, 22014, 22231, 22877, 38858,
        25030, 23953, 20959, 38774, 20361, 20830, 21552, 41315, 28922, 26390, 32781, 26626,
        11783, 15271, 10893, 11054, 14189, 15039, 16129, 25744, 11141, 41554, 37137, 37138,
        37139, 26987, 26988, 26989, 28508, 28509, 28510, 23045, 28112, 23044, 28113, 28114,
        23043, 23512, 23511, 23513, 25594, 10684, 25593, 25595, 22056, 22057, 22058, 32394,
        21218, 21217, 21216, 11690, 15260, 13606, 15261, 33466, 33467, 33468, 14260, 29670,
        29672, 23398, 29671, 23397, 23396, 13607, 14258, 15259, 13603, 15698, 15699, 15697,
        16963, 21853, 21854, 21855, 16961, 16962, 22512, 22513, 22514, 21006, 21008, 19749,
        21007, 19750, 19165, 19163, 19164, 20620, 20619, 20621, 36113, 14259, 20024, 20025,
        37829, 26566, 26564, 26565, 19502, 19501, 19503, 23204, 23205, 23203, 23736, 23620,
        23622, 23734, 23735, 23621, 37372, 27617, 27618, 36401, 27619, 36400, 33378, 37938,
        33379, 33380, 28915, 28917, 18355, 28916, 32277, 18357, 18356, 16396, 16397, 16398,
        14592, 14977, 14978, 14591, 14590, 14979, 26483, 53106, 53110, 53111
    ]
    
LEO: 26561, 40933, 40935, 40932, 40934,


TODO
Orbital Transfers
- GEO to MEO
    - apse line rotation delta V vector
    - combined hohmann + plane change delta V vector
        - can we do this for elliptical orbits or only circular? if circular only:
            - add in transfer from circle coplanar with MEO orbit to actually rendezvous with MEO object
    - phase to get to MEO object
- MEO to LEO
    - Lambert's 
        - plot transfer & sanity check
- LEO to LEO
    - Lambert's
        - plot transfer & sanity check

Project Submission
- propagate and plot full satellite track including staying with each object for 5 periods
- what else?

'''