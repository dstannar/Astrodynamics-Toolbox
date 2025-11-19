import numpy as np
from MathHelpers.constants import muE
from Propagators.Propagate import Propagate

def raan_change(orbit1, orbit2):
    '''
    Input: 2 orbit like objects (TLEOrbit, KeplerianOrbit)
    Output:
        burn_loc: position vector of burn
        burn_vect: deltaV velocity vector
        burn_time: time of burn, measured since passed orbit1 TA
        dVmag: delta V magnitude (burn cost)
    '''

    print('not yet implemented')


def inc_change(orbit1, orbit2, dt=1, mu=muE):
    intersectionPos, intersectionVel, intersectionTime = best_nodal_crossing(orbit1, orbit2)
    
    # inc dV
    TA1 = orbit1.TA
    FPA1 = np.atan2(orbit1.ecc*np.sin(TA1), (1 + orbit1.ecc*np.cos(TA1)))
    
    # Vallado alg 39
    dInc = orbit2.inc - orbit1.inc
    dv_vect = 2 * intersectionVel * np.cos(FPA1) * np.sin(dInc / 2)

    burn_loc = intersectionPos
    burn_time = intersectionTime
    v_init = intersectionVel

    return burn_loc, burn_time, v_init, dv_vect
   

def plane_change(orbit1, orbit2):
    # vallado alg 41
    intersectionPos, intersectionVel, intersectionTime = best_nodal_crossing(orbit1, orbit2)

    


    burn_loc = intersectionPos
    burn_time = intersectionTime

    return burn_loc, burn_time, dVmag

def best_nodal_crossing(orbit1, orbit2, dt=1):
    '''
    gets nodal crossing point with lowest velocity for cheapest dV burn
    
    '''
    hvec1 = orbit1.hvec
    hvec2 = orbit2.hvec

    # get shared node line
    node_vect = np.cross(hvec1, hvec2)
    norm_node = np.linalg.norm(node_vect)

    # init loop vars
    min_dist = np.inf
    intersectionVel = np.inf
    intersectionPos = None
    for i in range(int(orbit1.period / dt)):
        t = dt*i
        # initialize propagator
        prop1 = Propagate(orbit1, t)
        rNew, vNew = prop1.lagrange_coeff()

        u_node = node_vect / np.linalg.norm(node_vect)
        proj = u_node * np.dot(u_node, rNew)
        dist = np.linalg.norm(rNew - proj)
        if dist < min_dist:
            min_dist = dist
            intersectionPos = rNew
            intersectionVel = vNew
            intersectionTime = t # intersection time measured from orbit1 TA state as passed

    if intersectionPos is None:
        raise RuntimeError('No intersection of orbits')
    else:
        return intersectionPos, intersectionVel, intersectionTime