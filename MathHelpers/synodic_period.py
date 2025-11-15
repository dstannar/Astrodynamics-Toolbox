import numpy as np
from MathHelpers.constants import muE

def synodic_period(sma1, sma2, mu=muE):
    
    T1=2*np.pi/np.sqrt(mu)*sma1**(3/2)
    T2=2*np.pi/np.sqrt(mu)*sma2**(3/2)

    #Synotic Period: the period of body 1 relative to body 2
    TSyn=T1*T2/np.abs(T1-T2) #[seconds]

    return float(TSyn)