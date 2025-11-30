'''
Drew Stannard-Stockton
AERO351 Group Project Orbit Debris Cleanup Mission Implementation
'''
import numpy as np
from Orbits.TLEOrbit import TLEOrbit
from Orbits.KeplerianOrbit import KeplerianOrbit
from Propagators.Propagate import Propagate
from Transfers.Lambert import Lambert
from MathHelpers.wrap_angles import wrap_to_pi
from MathHelpers.synodic_period import synodic_period
from Transfers.plane_change import plane_change
from Transfers.phase import phasing_maneuver
from Transfers.plane_change import best_nodal_crossing
from MathHelpers.constants import rE
from Time.conversions import secs_to_JDays, JDays_to_secs, dateTime_to_JDays
from Orbits.safe_create_orbits import create_TLEOrbits
from Propagators.plot_helper import composite_trajectory
from MathHelpers.formatting import format_time
# we should expect divide by zero warnings for bad lamberts transfer, so go ahead and:
import warnings
warnings.filterwarnings("ignore")

def get_cost(decision_vector) -> float:
    """
    Returns cost. Takes in genetic algorithm decision vector
    """