import numpy as np
import matplotlib.pyplot as plt
import time

from Time.conversions import (
    dateTime_to_JDays,
    AbsJDay_to_J2000JDay,
    JDays_to_secs,
)
from Orbits.planetary_elements import planetary_elements
from MathHelpers.constants import muS, AU
from Orbits.KeplerianOrbit import KeplerianOrbit
from Transfers.Lambert import Lambert
from Propagators.Propagate import Propagate
from Propagators.plot_helper import composite_trajectory, animate_composite_figure
from Scripts.AERO351_debris_rendezvous_calc import execute_lambert


_, _, JD_dep_start = dateTime_to_JDays(2028, 1, 1, 0, 0, 0)
_, _, JD_arr_start = dateTime_to_JDays(2028, 3, 1, 0, 0, 0)

Tdepart = (JD_dep_start - 2451545.0)/36525
Tarrive = (JD_arr_start - 2451545.0)/36525

h_E, e_E, ta_E, raan_E, inc_E, argp_E = planetary_elements(
    planet_id=3, T=Tdepart
)
EarthOrbit = KeplerianOrbit(
    hmag=h_E,
    ecc=e_E,
    ta=ta_E,
    raan=raan_E,
    inc=inc_E,
    argp=argp_E,
    mu=muS,
)

h_M, e_M, ta_M, raan_M, inc_M, argp_M = planetary_elements(
    planet_id=4, T=Tarrive
)
MarsOrbit = KeplerianOrbit(
    hmag=h_M,
    ecc=e_M,
    ta=ta_M,
    raan=raan_M,
    inc=inc_M,
    argp=argp_M,
    mu=muS,
)

debrisSAT = EarthOrbit.copy()

# start timer
start_time = time.time()

bestLambertTransfer = execute_lambert(debrisSAT, EarthOrbit, MarsOrbit, mu=muS)

# stop timer
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

# unpack best lambert transfer
lambertPlot, dvMag, dv_ledger, debrisSAT, missionTime, lambert_report = bestLambertTransfer

# display plot
plt.show()

# animate plot
animate_composite_figure(
    composite_fig=lambertPlot,
    animate_mask=[False, False, True, True],
    order=[2, 3],
    fps=30,
    duration=12,
    background_alpha=0.25,
    active_alpha=1.0,
    save_path="lambert_composite_animation.gif",
    show=True
)