muE = 398600 # km3 / s2
muS = 132_712_440_018 #km3 / s2
rE = 6378 #km
rMars = 3396 #km
AU = 149597870.691 #km
yearInDays = 365.256 #days
JDaysInSecs = 86400 # seconds per Julian day
JD_J2000 = 2451545.0 # JD at J2000.0 (2000-01-01 12:00:00)
rSun = 696000 #km

# Planet + Moon data from Curtis Table A.1
# Units are kept exactly as in the table

MercuryData = {
    "radius_km": 2440.0,
    "mass_kg": 330.2e21,
    "sidereal_rotation": {"value": 58.65, "unit": "d"},
    "equator_inclination_deg": 0.01,
    "orbit_sma_km": 57.91e6,
    "orbit_ecc": 0.2056,
    "orbit_inclination_deg": 7.00,
    "orbit_sidereal_period": {"value": 87.97, "unit": "d"},
    "mu": 22032
}

VenusData = {
    "radius_km": 6052.0,
    "mass_kg": 4.869e24,
    "sidereal_rotation": {"value": 243.0, "unit": "d"},   # retrograde
    "equator_inclination_deg": 177.4,
    "orbit_sma_km": 108.2e6,
    "orbit_ecc": 0.0067,
    "orbit_inclination_deg": 3.39,
    "orbit_sidereal_period": {"value": 224.7, "unit": "d"},
    "mu": 324859
}

EarthData = {
    "radius_km": 6378.0,
    "mass_kg": 5.974e24,
    "sidereal_rotation": {"value": 23.9345, "unit": "h"},
    "equator_inclination_deg": 23.45,
    "orbit_sma_km": 149.6e6,
    "orbit_ecc": 0.0167,
    "orbit_inclination_deg": 0.0,
    "orbit_sidereal_period": {"value": 365.256, "unit": "d"},
    "mu": muE
}

MoonData = {
    "radius_km": 1737.0,
    "mass_kg": 73.48e21,
    "sidereal_rotation": {"value": 27.32, "unit": "d"},
    "equator_inclination_deg": 6.68,
    "orbit_sma_km": 384.4e3,   # about Earth
    "orbit_ecc": 0.0549,
    "orbit_inclination_deg": 5.145,
    "orbit_sidereal_period": {"value": 27.322, "unit": "d"},
    "mu": 4905
}

MarsData = {
    "radius_km": 3396.0,
    "mass_kg": 641.9e21,
    "sidereal_rotation": {"value": 24.62, "unit": "h"},
    "equator_inclination_deg": 25.19,
    "orbit_sma_km": 227.9e6,
    "orbit_ecc": 0.0935,
    "orbit_inclination_deg": 1.850,
    "orbit_sidereal_period": {"value": 1.881, "unit": "y"},
    "mu": 42828
}

JupiterData = {
    "radius_km": 71490.0,
    "mass_kg": 1.899e27,
    "sidereal_rotation": {"value": 9.925, "unit": "h"},
    "equator_inclination_deg": 3.13,
    "orbit_sma_km": 778.6e6,
    "orbit_ecc": 0.0489,
    "orbit_inclination_deg": 1.304,
    "orbit_sidereal_period": {"value": 11.86, "unit": "y"},
    "mu": 126686534
}

SaturnData = {
    "radius_km": 60270.0,
    "mass_kg": 5.685e26,       # 568.5×10^24 kg
    "sidereal_rotation": {"value": 10.66, "unit": "h"},
    "equator_inclination_deg": 26.73,
    "orbit_sma_km": 1.433e9,
    "orbit_ecc": 0.0565,
    "orbit_inclination_deg": 2.485,
    "orbit_sidereal_period": {"value": 29.46, "unit": "y"},
    "mu": 37931187
}

UranusData = {
    "radius_km": 25560.0,
    "mass_kg": 8.683e25,       # 86.83×10^24 kg
    "sidereal_rotation": {"value": 17.24, "unit": "h"},   # retrograde
    "equator_inclination_deg": 97.77,
    "orbit_sma_km": 2.872e9,
    "orbit_ecc": 0.0457,
    "orbit_inclination_deg": 0.772,
    "orbit_sidereal_period": {"value": 84.01, "unit": "y"},
    "mu": 5793939
}

NeptuneData = {
    "radius_km": 24764.0,
    "mass_kg": 1.024e26,       # 102.4×10^24 kg
    "sidereal_rotation": {"value": 16.11, "unit": "h"},
    "equator_inclination_deg": 28.32,
    "orbit_sma_km": 4.495e9,
    "orbit_ecc": 0.0113,
    "orbit_inclination_deg": 1.769,
    "orbit_sidereal_period": {"value": 164.8, "unit": "y"},
    "mu":6836529
}

PlutoData = {
    "radius_km": 1187.0,
    "mass_kg": 13.03e21,
    "sidereal_rotation": {"value": 6.387, "unit": "d"},   # retrograde
    "equator_inclination_deg": 122.5,
    "orbit_sma_km": 5.906e9,
    "orbit_ecc": 0.2488,
    "orbit_inclination_deg": 17.16,
    "orbit_sidereal_period": {"value": 247.9, "unit": "y"},
    "mu": 871
}

PlanetData = {
    "Mercury": MercuryData,
    "Venus": VenusData,
    "Earth": EarthData,
    "Moon": MoonData,
    "Mars": MarsData,
    "Jupiter": JupiterData,
    "Saturn": SaturnData,
    "Uranus": UranusData,
    "Neptune": NeptuneData,
    "Pluto": PlutoData,
}