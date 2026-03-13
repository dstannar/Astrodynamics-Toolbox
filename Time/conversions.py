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

# MJD2000 helpers for GTOP shenanigans

def MJD2000_to_AbsJDay(MJD2000):
    '''
    Input:
        MJD2000 : float
            days since MJD2000 epoch (2000-01-01 00:00 UT), where
            MJD2000 epoch = JD 2451544.5
    Output:
        JD_abs : float
            absolute Julian Date (days)
    '''
    JD_MJD2000 = JD_J2000 - 0.5
    return float(MJD2000) + JD_MJD2000


def AbsJDay_to_MJD2000(AbsJDay):
    '''
    Input:
        AbsJDay : float
            absolute Julian Date (days)
    Output:
        MJD2000 : float
            days since MJD2000 epoch (JD 2451544.5)
    '''
    JD_MJD2000 = JD_J2000 - 0.5
    return float(AbsJDay) - JD_MJD2000


def MJD2000_to_J2000JDay(MJD2000):
    '''
    Input:
        MJD2000 : float
            days since MJD2000 epoch (JD 2451544.5)
    Output:
        J2000JDay : float
            days since J2000.0 epoch (JD 2451545.0)

    Note:
        J2000 is +0.5 day after MJD2000 epoch, so:
            J2000JDay = MJD2000 - 0.5
    '''
    return float(MJD2000) - 0.5


def J2000JDay_to_MJD2000(J2000JDay):
    '''
    Input:
        J2000JDay : float
            days since J2000.0 epoch (JD 2451545.0)
    Output:
        MJD2000 : float
            days since MJD2000 epoch (JD 2451544.5)
    '''
    return float(J2000JDay) + 0.5


def gtop_decision_vector_to_mjd2000_epochs(T_daysMJD):
    '''
    GTOP-style decision vector interpretation:
        T_daysMJD = [t0, dt1, dt2, ..., dtN]
    where:
        t0  = epoch of event 0 in MJD2000 days
        dtk = duration from event k-1 to event k (days)

    Input:
        T_daysMJD : array-like length (N+1)

    Output:
        mjd_epochs : ndarray length (N+1)
            cumulative epochs in MJD2000 days
    '''
    T = np.asarray(T_daysMJD, dtype=float).reshape(-1)
    mjd_epochs = np.zeros_like(T)
    mjd_epochs[0] = float(T[0])
    for k in range(1, len(T)):
        mjd_epochs[k] = mjd_epochs[k - 1] + float(T[k])
    return mjd_epochs
