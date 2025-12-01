'''
This script solves a selection of Curtis ch8 problems
Implemented to complete Cal Poly AERO 351 HW4 
Problems solved: 8.2, 8.4, 8.6, 8.7, 8.12, 8.16
'''

import numpy as np
from Transfers.Interplanetary import Interplanetary
from MathHelpers.constants import muS
from MathHelpers.synodic_period import synodic_period
from MathHelpers.formatting import format_time

def ch8_2():
    '''
    Q: Find the total delta-v required for a Hohmann transfer from Mars’ orbit to Jupiter’s orbit.
    A: Use my Interplanetary class's hohmann() function
    '''
    # import circular, coplanar Mars and Jupiter orbit onjects
    from Orbits.heliocentric_planets import Mars_CC, Jupiter_CC
    # call Interplanetary.hohmann
    interplanetary = Interplanetary()
    dVarrive_vect, dVdepart_vect, dvTot_mag = interplanetary.hohmann(orbit1=Mars_CC, orbit2=Jupiter_CC)
    print("---PROBLEM 8.2---")
    print("The total required delta-v (km/s) is: ", dvTot_mag)

def ch8_4():
    '''
    Q: Calculate the synodic period of Jupiter relative to Mars.
    A: Use my synodic_period helper function which implements the synodic period eqn
    '''
    # import Mars and Jupiter periods
    from MathHelpers.constants import JupiterData, MarsData
    JupSMA = JupiterData["orbit_sma_km"]
    MarsSMA = MarsData["orbit_sma_km"]
    TSyn_JupMars = synodic_period(JupSMA, MarsSMA, mu=muS)
    TSyn, TSyn_unit = format_time(TSyn_JupMars)
    print("---PROBLEM 8.4---")
    print("The synodic period for Mars and Jupiter is: ", TSyn, TSyn_unit)

if __name__ == '__main__':
    ch8_2()
    ch8_4()