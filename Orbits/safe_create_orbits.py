from Orbits.TLEOrbit import TLEOrbit
from Orbits.KeplerianOrbit import KeplerianOrbit

def create_TLEOrbits(OrbitTLEs):
    OrbitLists = []
    for orbitTLE in OrbitTLEs:
        try:
            OrbitLists.append(TLEOrbit(orbitTLE))
        except Exception as e:
            print(orbitTLE, "failed:", e)
    return OrbitLists