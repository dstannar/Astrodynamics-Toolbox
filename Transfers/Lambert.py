import numpy as np
from MathHelpers.constants import muE
from MathHelpers.stumpff import stumpffC, stumpffS

class Lambert:
    def __init__(self, mu=muE, nrev=0, verbose=False):
        # assign to self
        self.nrev = nrev # multi rev not yet implemented
        self.mu = mu
        self.verbose = verbose

    def robust_solve(self, r0, r1, tof, shortWay=True):
        v1, v2, exitFlag = self.solve_lambertUV(r0, r1, tof, shortWay=shortWay)
        if exitFlag == 1:
            self.v1 = v1
            self.v2 = v2
            return v1, v2, exitFlag
        else:
            # eventually try more solvers, but for now:
            if self.verbose:
                print('failed to solve')
            return np.full(3, np.nan), np.full(3, np.nan), -1

    def solve_lambertUV(self, r0, r1, tof, shortWay=True):
        mu = self.mu
        r1_vect = r0
        r2_vect = r1
        r1mag = np.linalg.norm(r1_vect)
        r2mag = np.linalg.norm(r2_vect)
        badSolve = False # optimism!

        cos_dT = np.dot(r1_vect, r2_vect) / (r1mag * r2mag)
        dTheta = np.arccos(cos_dT)   
        
        rcross = np.cross(r1_vect, r2_vect)
        if rcross[2] < 0:                          
            dTheta = 2 * np.pi - dTheta     

        # transfer method setup
        tm = 1            
        if shortWay == False:
            tm = -1          

        A = tm * np.sqrt(r1mag * r2mag * (1 + cos_dT))

        # Check A for solvibility
        if A == 0:
            print("Transfer not possible, A = 0")
            # return nan, nan, nan, exitFlag = -1
            return np.full(3, np.nan), np.full(3, np.nan), -1

        # initial guess, bounds, solver tols
        z = 0                       # initial guess
        z_upper = 4 * np.pi**2      # upper z bound
        z_lower = -4 * np.pi**2     # lower z bound
        tol = 1e-8                  # solver tol.
        max_iteration = 100       # max iterations
        iteration = 0               

        # init solver values
        C = 1/2
        S = 1/6

        y = r1mag + r2mag + A * (z * S - 1) / np.sqrt(C) 
        x = np.sqrt(y / C)           
        dt_loop = x**3 * S / np.sqrt(mu) + A * np.sqrt(y) / np.sqrt(mu)
        # iterate on universal variable z with bisection method
        while abs(dt_loop - tof) > tol and iteration < max_iteration:
            if dt_loop < tof:
                z_lower = z
            else:
                z_upper = z
            z = (z_upper + z_lower) / 2
            C = stumpffC(z)
            S = stumpffS(z)

            if C == 0:
                break

            y = r1mag + r2mag + A * (z * S - 1) / np.sqrt(C)
            if y < 0:
                z = (z_upper + z_lower) / 2
            
            x = np.sqrt(y / C)
            dt_loop = x**3 * S / np.sqrt(mu) + A * np.sqrt(y) / np.sqrt(mu)
            iteration += 1

            # escape clause
            if iteration == max_iteration:
                badSolve = True
                
        # handle bisection method failure fast
        if badSolve:
            return np.full(3, np.nan), np.full(3, np.nan), -2

        # stumpff call
        C = stumpffC(z)
        S = stumpffS(z)

        # use universal variable z to get x, y
        y = r1mag + r2mag + A * (z * S - 1) / np.sqrt(C)
        x = np.sqrt(y / C)

        # lagrange coeff
        f = 1 - (x**2 / r1mag) * C
        g = tof - (x**3 / np.sqrt(mu)) * S
        fdot = np.sqrt(mu) / (r1mag * r2mag) * x * (z * S - 1)
        gdot = 1 - (x**2 / r2mag) * C

        # velo vectors from lagrange coeff
        v1_vect = (r2_vect - f * r1_vect) / g
        v2_vect = fdot * r1_vect + gdot * v1_vect
        dv = v2_vect - v1_vect
        dvMag = np.linalg.norm(dv)

        # return solution with positive exitFlag (success)
        return v1_vect, v2_vect, 1
