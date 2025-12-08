# import dependencies
import numpy as np

def format_longitude(lon):
    '''
    Convert longitude input to degrees east-positive
    Inputs:
        lon: float (degrees) OR (h, m, s) tuple/list
    Output:
        longitude in degrees
    '''
    if isinstance(lon, (int, float, np.floating)):
        return float(lon)
    h, m, s = map(float, lon)
    sign = -1.0 if h < 0 else 1.0
    return sign * (abs(h) + abs(m)/60.0 + abs(s)/3600.0)

def format_time(sec):
    '''
    converts seconds into minutes, hours, or days depending on what is most readable
    if < 1 min, returns seconds, if < 1hr returns mins, if < 1 day returns hrs, if > 1 day returns days
    '''
    if sec / 60 < 1:
        return sec, 'seconds'
    elif sec / 60 / 60 < 1:
        return sec / 60, 'minutes'
    elif sec / 60 / 60 / 24 < 1:
        return sec / 60 / 60, 'hours'
    elif sec / 60 / 60 / 24 / 356.25 < 1:
        return sec / 60 / 60 / 24, 'days'
    else:
        return sec / 60 / 60 / 24 / 365.25, 'years'