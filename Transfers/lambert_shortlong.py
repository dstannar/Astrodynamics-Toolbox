import numpy as np

def lambert_candidates_short_long(lambert_obj, r1_du, r2_du, tof_tu, nrev=0):
    '''
    Generate CANONICAL Lambert candidates for short-way vs long-way, 0-rev.
    useful for getting best dV soltns
    Inputs:
        lambert_obj : NNLambert instance
        r1_du, r2_du : 3x1 arrays
        tof_tu : float
        nrev : int

    Outputs:
        candidates : list of dict
            each dict has keys:
              name: "short" or "long"
              tof_sign: +1 or -1
              v1: 3x1 ndarray VU
              v2: 3x1 ndarray VU
              exitflag: int
    '''
    r1_du = np.asarray(r1_du, dtype=float).reshape(3)
    r2_du = np.asarray(r2_du, dtype=float).reshape(3)

    cands = []

    for name, sgn in (("short", +1), ("long", -1)):
        v1, v2, exitflag = lambert_obj.solve_traditional_lambert_canonical(
            r1_du, r2_du, float(sgn * tof_tu),
            nrev=int(nrev),
            verbose=False
        )
        if int(exitflag) == 1: # passed flag from lambert.py
            cands.append({
                "name": name,
                "tof_sign": int(sgn),
                "v1": np.asarray(v1, dtype=float).reshape(3),
                "v2": np.asarray(v2, dtype=float).reshape(3),
                "exitflag": int(exitflag),
            })

    return cands
