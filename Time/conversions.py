import numpy as np
from MathHelpers.formatting import format_longitude
from MathHelpers.constants import JD_J2000, JDaysInSecs

def J2000JDay_to_AbsJDay(J2000JDay):
    '''
    Input: Julian days since J2000.0 epoch
    Output: absolute Julian days
    '''
    return J2000JDay + JD_J2000

def AbsJDay_to_J2000JDay(AbsJDay):
    '''
    Input: Absolute Julian Days
    Output: Julian days since J2000.0 epoch
    '''
    return AbsJDay - JD_J2000

def JDays_to_secs(JDay):
    '''
    Input: Julian Days since J2000
    Output: seconds since J2000
    '''
    return JDay * JDaysInSecs 
    
def secs_to_JDays(secs):
    '''
    Input: seconds since J2000
    Output: Julian Days since J2000
    '''
    return secs / JDaysInSecs  

def dateTime_to_JDays(year:float, month:float, day:float, hour:float, minute:float, second:float):
    '''
    Implements Curtis's Orbital Mechanics formula for Julian time
    Inputs:
        year, month, day : floats
        hour, minute, sec: floats
    Outputs:
        J0 : float = Julian date at 0h UT
        UT : float = Universal Time in hours
        JD : float = full Julian date
    '''
    y, m, d = year, month, day

    J0 = 367*y - np.floor(7*(y + np.floor((m+9)/12))/4) + np.floor(275*m/9) + d + 1721013.5
    UT = hour + minute/60.0 + second/3600.0
    JD = J0 + UT/24

    return J0, UT, JD

def dateTime_to_sidereal(year:float, month:float, day:float, hour:float, minute:float, second:float, longitude:float|tuple):
    '''
    Computes local sidereal time in degrees
    Inputs:
        year, month, day : floats
        hour, minute, sec: floats
        longitude: float (deg east) OR (h, m, s) tuple
    Output:
        siderealTime: float in [0, 360)
    '''
    long_deg = format_longitude(longitude)
    J0, UT, _ = dateTime_to_JDays(year, month, day, hour, minute, second)
    T0 = (J0 - 2451545.0) / 36525.0

    thetaG0 = (100.4606184
               + 36000.77004*T0
               + 0.000387933*(T0**2)
               - 2.583e-8*(T0**3))
    thetaG0 -= np.floor(thetaG0/360.0)*360.0

    thetaG = thetaG0 + 360.98564724*(UT/24.0)
    theta  = thetaG + long_deg
    theta -= np.floor(theta/360.0)*360.0

    return theta
