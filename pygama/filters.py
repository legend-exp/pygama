import numpy as np
from scipy import signal


def rc_decay(rc1_us, freq = 100E6):
    '''
    rc1_us: decay time constant in microseconds
    freq: digitization frequency of signal you wanna process
    '''

    rc1_dig= 1E-6 * (rc1_us) * freq
    rc1_exp = np.exp(-1./rc1_dig)
    num = [1,-1]
    den = [1, -rc1_exp]

    return (num, den)

def gretina_overshoot(rc_us, pole_rel, freq = 100E6):
    zmag = np.exp(-1./freq/(rc_us*1E-6))
    pmag = zmag - 10.**pole_rel

    num = [1, -zmag]
    den = [1, -pmag]

    return (num, den)
