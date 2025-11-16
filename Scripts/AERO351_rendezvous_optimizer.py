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

            if dPlaneDeg < 2:
                pair = [orbit1.satnum, orbit2.satnum, float(dPlaneDeg)]
                candidate_pairs.append(pair)
    
    # empty list
    if not candidate_pairs:
        return [], None
   
    dplanes = [p[2] for p in candidate_pairs]
    best_idx = dplanes.index(min(dplanes))
    best_candidate = candidate_pairs[best_idx]

    return candidate_pairs, best_candidate


def scanLambertCandidates(Orbit1s, Orbit2s, dt=800, bestGuess=None):
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
                r1New, v1New, *_ = Propagate(orbit1, t1).lagrange_coeff()

                if bestGuess is None:
                    t2 = t1 + dt
                    t2_end = t1 + tof_horizon
                else:
                    t2 = max(t1 + dt, t2_start_abs)   # keep TOF > 0 and respect refine window
                    t2_end = t2_end_abs

                while t2 <= t2_end:
                    tof = t2 - t1
                    r2New, v2New, *_ = Propagate(orbit2, t2).lagrange_coeff()
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
        33752, 33751, 37206, 22654
    ]

    LEODebrisTLEs = [

    ]

    GEOOrbits = create_orbits(GEODebrisTLEs)
    MEOOrbits = create_orbits(MEODebrisTLEs)
    LEOOrbits = create_orbits(LEODebrisTLEs)
    print('Done creating orbit objects')

    #candidate_pairs, best_candidate = smallPlaneChangeCandidates(GEOOrbits, MEOOrbits)
    roughBest = scanLambertCandidates(GEOOrbits, MEOOrbits)
    fineBest = scanLambertCandidates(GEOOrbits, MEOOrbits, dt = 100, bestGuess=roughBest)

    # print best candidate = [norad1, norad2, dPlaneDeg]
    #print("The GEO-MEO pair with smallest plane change is:", best_candidate)
    print("Best Lambert cost is: ", fineBest)
    