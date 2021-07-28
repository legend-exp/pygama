import numpy as np
from numpy.testing import assert_
import pygama.math.peak_fitting as pgf


def test_fwhm():
    pars = [ 0, 1, 0, 0, 0.1, 0, 1 ] 
    cov = [ [ 1e-16, 0, 0, 0, 0, 0, 0 ], #dmu2
            [ 0, 1e-2, 0, 0, 0, 0, 0 ], #dsig2
            [ 0, 0, 1e-16, 0, 0, 0, 0 ], #dhstep2
            [ 0, 0, 0, 1e-16, 0, 0, 0 ], #dhtail2
            [ 0, 0, 0, 0, 1e-16, 0, 0 ], #dtau2
            [ 0, 0, 0, 0, 0, 1e-16, 0 ], #dbg02
            [ 0, 0, 0, 0, 0, 0, 1e-16 ] ] #damp2
    mu, sig, hstep, htail, tau, bg0, amp = pars
    fwhm, dfwhm = pgf.radford_fwhm(sig, htail, tau, cov)
    assert_(np.absolute(fwhm - 2.3548) < 1e-4, f'got {fwhm}')
    assert_(np.absolute(dfwhm - 0.23548) < 1e-5, f'got {dfwhm}')

