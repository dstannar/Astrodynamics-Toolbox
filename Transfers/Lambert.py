import numpy as np
from MathHelpers.constants import muE
from MathHelpers.stumpff import stumpffC, stumpffS

class Lambert:
    def __init__(self, r0, r1, tof, mu=muE, nrev=0, shortWay=True):
        # assign to self
        self.r0 = r0
        self.r1 = r1
        self.tof = tof
        self.nrev = nrev
        self.shortWay = shortWay
        self.mu = mu

    def robust_solve(self):
        v1, v2, dv, exitFlag = self.solve_lambertUV()
        if exitFlag == 1:
            self.v1 = v1
            self.v2 = v2
            self.dv = dv
            return v1, v2, dv
        else:
            # eventually try more solvers, but for now:
            raise RuntimeError('Failed to Solve Lambert')

    def solve_lambertUV(self):
        mu = self.mu
        r1_vect = self.r0
        r2_vect = self.r1
        shortWay = self.shortWay
        tof = self.tof
        r1 = np.linalg.norm(r1_vect)
        r2 = np.linalg.norm(r2_vect)
        badSolve = False # optimism!

        cos_dT = np.dot(r1_vect, r2_vect) / (r1 * r2)
        dTheta = np.arccos(cos_dT)   
        
        rcross = np.cross(r1_vect, r2_vect)
        if rcross[2] < 0:                          
            dTheta = 2 * np.pi - dTheta     

        # transfer method setup
        tm = 1            
        if shortWay == False:
            tm = -1          

        A = tm * np.sqrt(r1 * r2 * (1 + cos_dT))

        # Check A for solvibility
        if A == 0:
            print("Transfer not possible, A = 0")
            # return nan, nan, nan, exitFlag = -1
            return np.full(3, np.nan), np.full(3, np.nan), np.nan, -1

        # initial guess, bounds, solver tols
        z = 0                       # initial guess
        z_upper = 4 * np.pi**2      # upper z bound
        z_lower = -4 * np.pi**2     # lower z bound
        tol = 1e-8                  # solver tol.
        max_iteration = 10000       # max iterations
        iteration = 0               

        # init solver values
        C = 1/2
        S = 1/6

        y = r1 + r2 + A * (z * S - 1) / np.sqrt(C) 
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

            y = r1 + r2 + A * (z * S - 1) / np.sqrt(C)
            if y < 0:
                z = (z_upper + z_lower) / 2
                continue

            x = np.sqrt(y / C)
            dt_loop = x**3 * S / np.sqrt(mu) + A * np.sqrt(y) / np.sqrt(mu)
            iteration += 1

            # escape clause
            if iteration == max_iteration:
                badSolve = True

        # handle bisection method failure
        if badSolve:
            return np.full(3, np.nan), np.full(3, np.nan), np.nan, -2

        # stumpff call
        C = stumpffC(z)
        S = stumpffS(z)

        # use universal variable z to get x, y
        y = r1 + r2 + A * (z * S - 1) / np.sqrt(C)
        x = np.sqrt(y / C)

        # lagrange coeff
        f = 1 - (x**2 / r1) * C
        g = tof - (x**3 / np.sqrt(mu)) * S
        fdot = np.sqrt(mu) / (r1 * r2) * x * (z * S - 1)
        gdot = 1 - (x**2 / r2) * C

        # velo vectors from lagrange coeff
        v1_vect = (r2_vect - f * r1_vect) / g
        v2_vect = fdot * r1_vect + gdot * v1_vect
        dv = np.linalg.norm(v2_vect - v1_vect)

        # return solution with positive exitFlag (success)
        return v1_vect, v2_vect, dv, 1
