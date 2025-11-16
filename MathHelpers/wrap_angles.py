import numpy as np

def wrap_to_pi(angle):
    '''
    Input: angle (radians)
    Output: angle within -pi, pi (radians)
    '''
    return (angle + np.pi) % (2.0 * np.pi) - np.pi