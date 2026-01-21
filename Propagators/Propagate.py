import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from MathHelpers.constants import muE
from MathHelpers.solve_universal_anomaly import solve_universal_anomaly
from MathHelpers.stumpff import stumpffC, stumpffS
from mpl_toolkits.mplot3d import Axes3D

class Propagate:
    def __init__(self, prop_time, Orbit=None, r=None, v=None, thrust=0, burnTime = None, Isp = None, mass=None, rtol = 1e-12, atol = 1e-12, mu=muE, verbose=False):
        self.Orbit = Orbit # orbit object
        self.r = r
        self.v = v
        self.prop_time = prop_time
        self.mu = mu
        self.thrust = thrust
        self.burnTime = burnTime
        self.Isp = Isp
        self.rtol = rtol
        self.atol = atol
        self.mass = mass
        self.verbose = verbose


    def twobody_ODE(self, plot=False):
        '''
        Propogates and plots the orbit, supports thrust
        Inputs:
            state:array(6x1) = initial state vector
            tspan:array(2x1) = time span vector, seconds
            rtol:int = solver tolerance
            atol:int = solver tolerance
        Outputs:
            Ir, v, Solution
        '''
        m0 = self.mass
        if self.r is None and self.v is None and self.Orbit is not None:
            r0 = self.Orbit.r
            v0 = self.Orbit.v
        elif self.r is not None and self.v is not None and self.Orbit is None:
            r0 = self.r
            v0 = self.v
        rtol = self.rtol
        atol = self.atol
        tspan = [0, self.prop_time]
        # solve ivp

        if m0 is None:
            state = np.concatenate((r0, v0))
        else:
            state = np.concatenate((r0, v0, [float(m0)]))

        def ivp_wrapper(t,y):
            return self.twobodymotion(t, y, self.thrust, self.burnTime, self.Isp)

        solution = solve_ivp(ivp_wrapper, tspan, state, rtol=rtol, atol=atol)
        
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            x, y, z = solution.y[0], solution.y[1], solution.y[2]
            ax.plot(x, y, z, linewidth=1.5)
            ax.scatter(x[0],  y[0],  z[0],  s=50)  # start
            ax.scatter(x[-1], y[-1], z[-1], s=50)  # end


            ax.set_xlabel('X (km)')
            ax.set_ylabel('Y (km)')
            ax.set_zlabel('Z (km)')
            ax.set_title('Orbit Propagation')
            ax.set_box_aspect([1, 1, 1]) 
            plt.close()

    
        y = solution.y

        r = y[:3, -1] # [x_f, y_f, z_f]
        v = y[3:6, -1] # [vx_f, vy_f, vz_f]

        if self.verbose == False:
            if plot == True:
                return r, v, fig
            elif plot == False:
                return r, v
        elif self.verbose == True:
            if plot == True:
                return r, v, solution, fig
            elif plot == False:
                return r, v, solution



    def lagrange_coeff(self):
        '''
        Propagates using Lagrange Coefficients
        Returns r1, v1, f, g, fdot, gdot
        doesn't support thrust
        '''
        if self.Orbit is None and self.r is not None and self.v is not None:
            r0 = self.r
            v0 = self.v
        if self.Orbit is not None and self.r is None and self.v is None:
            r0 = self.Orbit.r
            v0 = self.Orbit.v
        dt = self.prop_time
        
        mu = self.mu
        chi = solve_universal_anomaly(dt, r0, v0, mu=self.mu)
    
        r0mag = np.linalg.norm(r0)
        v0mag = np.linalg.norm(v0)
        a = 1 / (2/r0mag - v0mag**2 / mu)
        alpha = 1/a
        z = alpha*chi**2
        Cz = stumpffC(z)
        Sz = stumpffS(z)

        f = 1 - (chi**2 / r0mag) * Cz
        g = dt - (chi**3 / np.sqrt(mu)) * Sz

        r1 = f*r0 + g*v0
        r1mag = np.linalg.norm(r1)

        fdot = np.sqrt(mu)/(r1mag*r0mag) * ((z*Sz - 1)*chi)
        gdot = 1 - (chi**2 / r1mag) * Cz

        v1 = fdot*r0 + gdot*v0

        if self.verbose == True:
            return r1, v1, f, g, fdot, gdot 
        elif self.verbose == False:
            return r1, v1

    def twobodymotion(self, t, y, thrust, burnTime, Isp):
        '''
        Returns derivative of state variables
        supports thrust
        meant as an internal helper, not private bc maybe helpful at some point
        Input Args:
            time:int = scalar time value
            state:array(6,1) = state array in 3D
        Outputs:
            dstate = derivative of the state for the 2 body problem
        '''
        state = y
        time = t
        MU = self.mu

        if len(state) == 7:
            x, y, z, vx, vy, vz, m = state
        elif len(state) == 6:
            x,y,z,vx,vy,vz = state
            m = np.inf
        else:
            raise RuntimeError('Wrong Size State')
        rmag = np.linalg.norm([x, y, z])
        ddx = -MU * x / rmag**3
        ddy = -MU * y / rmag**3
        ddz = -MU * z / rmag**3

        mdot = 0
        if (thrust > 0) and (time <= burnTime):
            vmag = np.linalg.norm([vx, vy, vz])
            if (vmag > 0) and np.isfinite(m) and (m > 0):
                a_T = (thrust / m) #km/s**2
                ddx += a_T * state[3] / vmag
                ddy += a_T * state[4] / vmag
                ddz += a_T * state[5] / vmag
                g0 = 9.80665/1000                  # km/s^2
                mdot = -thrust / (Isp * g0)   # kg/s

        dstate = [vx, vy, vz, ddx, ddy, ddz]
        if len(state) == 7:
            dstate.append(mdot)
        return dstate
    
