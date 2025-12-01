import numpy as np
from Orbits.KeplerianOrbit import KeplerianOrbit
from MathHelpers.constants import PlanetData, muS

# True orbit objects, including planes and non-circularity
def circ_coplanar_planet(planet_name):
    """
    Build a KeplerianOrbit for a planet using Curtis Table A.1 data.
    Angles pass in are in degrees & converted to radians.
    """
    data = PlanetData[planet_name]

    sma = data["orbit_sma_km"]
    ecc = 0 # circular
    inc = 0 # circular

    # convert to radians
    ta = 0 #arbitrarily at perigee
    raan = 0 # circular
    argp = 0 #arbitrary choice, undefined for circ orbits

    # specific angular momentum from a and e: h = sqrt(mu a (1-e^2))
    h = np.sqrt(muS * sma * (1.0 - ecc**2))

    return KeplerianOrbit(
        hmag=h,
        ecc=ecc,
        ta=ta,
        raan=raan,
        inc=inc,
        argp=argp,
        mu=muS,        # central body is the Sun
    )

# excluding mercury and pluto because circular&coplanar is such a bad assumption
Venus_CC = circ_coplanar_planet("Venus")
Earth_CC = circ_coplanar_planet("Earth")
Mars_CC = circ_coplanar_planet("Mars")
Jupiter_CC = circ_coplanar_planet("Jupiter")
Saturn_CC = circ_coplanar_planet("Saturn")
Uranus_CC = circ_coplanar_planet("Uranus")
Neptune_CC = circ_coplanar_planet("Neptune")
