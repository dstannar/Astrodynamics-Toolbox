'''
Project Requirements:
- Must execute 4 seperate transfer maneuvers

'''

import numpy as np
from Orbits.TLEOrbit import TLEOrbit
from Propagators.Propagate import Propagate
from Transfers.Lambert import Lambert
from MathHelpers.wrap_angles import wrap_to_pi
from MathHelpers.synodic_period import synodic_period
from Transfers.plane_change import plane_change
from Transfers.phase import phasing_maneuver
from Transfers.plane_change import best_nodal_crossing

def create_orbits(OrbitTLEs):
    OrbitLists = []
    for orbitTLE in OrbitTLEs:
        try:
            OrbitLists.append(TLEOrbit(orbitTLE))
        except Exception as e:
            print(orbitTLE, "failed:", e)
    return OrbitLists

def smallPlaneChangeCandidates(Orbit1s, Orbit2s):
    candidate_pairs = []
    for orbit1 in Orbit1s:
        for orbit2 in Orbit2s:
            # Check for good candidates
            dRaan = wrap_to_pi(orbit2.raan - orbit1.raan)
            cosdP = np.cos(orbit1.inc)*np.cos(orbit2.inc) + np.sin(orbit1.inc)*np.sin(orbit2.inc)*np.cos(dRaan)
            dPlane = np.acos(cosdP)
            dPlaneDeg = np.rad2deg(dPlane)

            if dPlaneDeg < 10:
                pair = [orbit1, orbit2, float(dPlaneDeg)]
                candidate_pairs.append(pair)
    
    # empty list
    if not candidate_pairs:
        return [], None
   
    dplanes = [p[2] for p in candidate_pairs]
    best_idx = dplanes.index(min(dplanes))
    best_candidate = candidate_pairs[best_idx]
    orbit1Best = best_candidate[0]
    orbit2Best = best_candidate[1]
    #apseLinedV = phasing_maneuver(ra, rp, TA1, TA2, delta_apse=0, mu=muE)

    
    bestLoc, bestTime, bestdV = plane_change(orbit1Best, orbit2Best)

    return best_candidate
    #return candidate_pairs, best_candidate, bestdV, bestLoc, bestTime
'''
def manualTransfer(Orbit1s, Orbit2s):
    candidate_pairs = []
    dV_best = np.inf
    for orbit1 in Orbit1s:
        for orbit2 in Orbit2s:
            # 
    
    # empty list
    if not candidate_pairs:
        return [], None
   
    dplanes = [p[2] for p in candidate_pairs]
    best_idx = dplanes.index(min(dplanes))
    best_candidate = candidate_pairs[best_idx]
    orbit1Best = best_candidate[0]
    orbit2Best = best_candidate[1]
    #apseLinedV = phasing_maneuver(ra, rp, TA1, TA2, delta_apse=0, mu=muE)

    
    bestLoc, bestTime, bestdV = plane_change(orbit1Best, orbit2Best)

    return best_candidate

'''
def scanLambertCandidates(Orbit1s, Orbit2s, dt=100, bestGuess=None):
    dvOld = float('inf')
    best = None

    # If refining, pull the winning IDs so we only re-check that pair
    if bestGuess is not None:
        id1_ref = int(bestGuess[0])
        id2_ref = int(bestGuess[1])

    for orbit1 in Orbit1s:
        for orbit2 in Orbit2s:

            # Skip non-winning pairs during refine
            if bestGuess is not None and (orbit1.satnum != id1_ref or orbit2.satnum != id2_ref):
                continue

            Tsyn = synodic_period(orbit1.sma, orbit2.sma)

            if bestGuess is None:
                dep_horizon = Tsyn
                tof_horizon = orbit2.period
                t1 = 0
            else:
                # bestGuess: [id1, id2, r1, r2, tof, dv, t1_low, t1_high, t2_low, t2_high]
                t1_low  = float(bestGuess[6]) 
                t1_high = float(bestGuess[7])
                t2_low  = float(bestGuess[8]) 
                t2_high = float(bestGuess[9])
                dep_horizon = t1_high
                t1 = t1_low
                t2_start_abs = t2_low
                t2_end_abs   = t2_high

            while t1 <= dep_horizon:
                r1New, v1New = Propagate(orbit1, t1).lagrange_coeff()

                if bestGuess is None:
                    t2 = t1 + dt
                    t2_end = t1 + tof_horizon
                else:
                    t2 = max(t1 + dt, t2_start_abs)   # keep TOF > 0 and respect refine window
                    t2_end = t2_end_abs

                while t2 <= t2_end:
                    tof = t2 - t1
                    r2New, v2New = Propagate(orbit2, t2).lagrange_coeff()

                    v1_req, v2_req, dv = Lambert(r1New, r2New, tof).robust_solve()

                    if np.isfinite(dv) and dv < dvOld:
                        dvOld = dv
                        best = [orbit1.satnum, orbit2.satnum, r1New, r2New, tof, dv,
                                t1 - dt, t1 + dt, t2 - dt, t2 + dt]

                    t2 += dt
                t1 += dt

    return best
    


if __name__ == '__main__':
    # downloaded satcat csv from celestrak, sorted for debris geo objects with 0 < inc < 5

    GEODebrisTLEs = [
        9416
    ]

    # sorted for debris meo objects with small inclination (approx < 7 or so)
    MEODebrisTLEs = [
        32993
    ]

    LEODebrisTLEs = [
        40932
    ]

    LEOODebrisTLEs2 = [
        26561
    ]


    GEOOrbits = create_orbits(GEODebrisTLEs)
    MEOOrbits = create_orbits(MEODebrisTLEs)
    LEOOrbits = create_orbits(LEODebrisTLEs)
    LEOOrbits2 = create_orbits(LEOODebrisTLEs2)
    print('Done creating orbit objects')

    #candidate_pairs, best_candidate, bestdV, bestLoc, bestTime = smallPlaneChangeCandidates(MEOOrbits, LEOOrbits)
    #best_candidate = smallPlaneChangeCandidates(MEOOrbits, LEOOrbits)

    #roughBestML = scanLambertCandidates(MEOOrbits, LEOOrbits)
    #print('rough done')
    #fineBestML = scanLambertCandidates(MEOOrbits, LEOOrbits, dt = 1, bestGuess=roughBestML)

    #roughBestLL = scanLambertCandidates(LEOOrbits, LEOOrbits2)
    #print('rough done')
    #fineBestLL = scanLambertCandidates(LEOOrbits, LEOOrbits2, dt = 1, bestGuess=roughBestLL)

    GEOOrbit = GEOOrbits[0]
    MEOOrbit = MEOOrbits[0]
    intersectionPosG, intersectionVelG, intersectionTimeG = best_nodal_crossing(GEOOrbit, MEOOrbit)
    intersectionPosM, intersectionVelM, intersectionTimeM = best_nodal_crossing(MEOOrbit, GEOOrbit)
    GEOOrbit.set_state(intersectionPosG, intersectionVelG)
    MEOOrbit.set_state(intersectionPosM, intersectionVelM)
    print(intersectionPosG)
    print(intersectionPosM)
    print(np.linalg.norm(intersectionPosG-intersectionPosM))




    # print best candidate = [norad1, norad2, dPlaneDeg]
    #print("The GEO-MEO pair with smallest plane change is:", best_candidate)
    #print('the best lambert MEO to LEO is: ', fineBestML)
    # answer:
    # the best lambert MEO to LEO is:  [32993, 40932, array([-21289.13544757, -23924.62702742,   2142.45564878]), 
    #                                   array([3611.17144476, 5962.30291668,  492.8427704 ]), 7.0, np.float64(0.10916180846440457), 
    #                                   6499.0, 6501.0, 6506.0, 6508.0]

    #print('the best Lambert leo to leo is:', fineBestLL)
    # answer:
    # the best Lambert leo to leo is: [40932, 26561, array([-4233.84866644, -5562.18696161,  -433.75224344]), 
    #                                   array([-4546.3952247 , -5193.06454469,  -204.65148314]), 50.0, np.float64(0.41220308185891275), 
    #                                   15142.0, 15144.0, 15192.0, 15194.0]

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

'''