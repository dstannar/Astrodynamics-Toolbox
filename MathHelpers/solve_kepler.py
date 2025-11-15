import numpy as np

def solve_kepler(Me, ecc):
    tol = 1e-10
    max_iter = 30
    ratio = 1
    iters = 0

    if Me < np.pi:
        Ei = Me + ecc/2
    else:
        Ei = Me - ecc/2

    while ratio > tol and iters < max_iter:
        fEi = Ei - ecc * np.sin(Ei) - Me
        fdotEi = 1 - ecc*np.cos(Ei)

        ratio = fEi / fdotEi
        Ei = Ei - ratio

        iters +=1 

    E = Ei

    return E