#!/usr/bin/env python3
import os
import time
import timeit
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from scipy.optimize import bisect
import pandas as pd

def main():
    """
    goal: define a new operation on a 2d numpy ndarray of waveforms
    """
    # example_1()
    # example_2()
    example_3()


def run_timer(stmt, setup=None, rpt=5, num=50000, verbose=False):
    """
    similar to ipython's timeit magic for an arbitrary line of code in a script
    """
    tot = min(timeit.Timer(stmt, globals=globals()).repeat(rpt, num))
    rate = tot/num/1e-9
    print("{:.0f} ns per call  (manual timer)".format(rate))
    return rate

# ==============================================================================

def example_1():
    """
    an example of a loop operation calling a kernel function (maybe a log-likelihood fit).
    let's bisect a function f(x) and find its roots, comparing speed of
    scipy.optimize.bisect and a manual implementation using Numba
    """
    # scipy
    x0, f = bisect(python_func, a=-0.5, b=50., full_output=True)
    print(f)
    t1 = run_timer('bisect(kernel_func, a=-.5, b=50.)')

    # numba
    numba_bisect(a=-0.5, b=50.)
    t2 = run_timer('numba_bisect(a=0.5, b=50.)')

    # summary
    print("scipy: {:.1e}  numba {:.1e}".format(t1, t2))


def python_func(x):
    """
    this is just to comapre against the numba version
    """
    return x**4 - 2*x**2 - x - 3


@njit
def kernel_func(x):
    """
    arbitrary function goes here
    """
    return x**4 - 2*x**2 - x - 3


@njit
def numba_bisect(a, b, tol=1e-8, mxiter=500):
    """
    arbitrary loop operation goes here
    """
    its = 0
    fa, fb = kernel_func(a), kernel_func(b)
    if abs(fa) < tol:
        return a
    elif abs(fb) < tol:
        return b
    c = (a+b)/2.
    fc = kernel_func(c)
    while abs(fc)>tol and its<mxiter:
        its = its + 1
        if fa*fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        c = (a+b)/2.
        fc = kernel_func(c)
    return c

# ==============================================================================

def example_2():
    """
    try the fractal generation program from the numba documentation
    https://numba.pydata.org/numba-doc/dev/user/examples.html
    """
    image = np.zeros((500 * 2, 750 * 2), dtype=np.uint8)
    t_start = time.time()

    # this is a loop-based computation on a 2d numpy array
    create_fractal(image, -2.0, 1.0, -1.0, 1.0, 20)

    print("elapsed: {:.2e} sec".format(time.time()-t_start))
    plt.imshow(image)
    plt.show()


@njit
def mandel(x, y, max_iters):
    """
    given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    """
    i = 0
    c = complex(x,y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i
    return 255


@njit
def create_fractal(image, min_x, max_x, min_y, max_y, iters):
    """
    do a calculation on the input array
    """
    height = image.shape[0]
    width = image.shape[1]
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color
    return image

# ==============================================================================

def example_3():
    """
    read in some waveforms from a Tier 2 file and find the mean or something
    """
    t1_file = os.path.expandvars("~/Data/MJ60/pygama/t1_run204.h5")
    # with pd.HDFStore(t1_file, 'r') as store:
    #     print(store.keys())

    key = "/ORSIS3302DecoderForEnergy"
    chunk = pd.read_hdf(t1_file, key, where="ievt < {}".format(1000))
    chunk.reset_index(inplace=True) # required step -- fix pygama "append" bug

    # create waveform block.  todo: mask wfs of unequal lengths
    icols = []
    for idx, col in enumerate(chunk.columns):
        if isinstance(col, int):
            icols.append(col)
    wfs = chunk[icols].values
    # print(wfs.shape, type(wfs))

    # use pygama DSP functions on the wf block.
    # requires:
    # - 'waves': a dict of (nwfs, nsamp) ndarrays,
    # - 'calcs': a pd.DataFrame, and a clock freq
    rise, flat = 4, 1.8
    waves = {"waveform":wfs, "settings":{"clk":100e6}}
    calcs = pd.DataFrame()
    avg_bl(waves, calcs)
    # waves["wf_blsub"] = blsub(waves, calcs)["wf_blsub"]
    # waves["wf_trap"] = trap(waves, calcs, rise, flat)["wf_trap"]

    # print(calcs)

@njit
def avg_bl(waves, calcs, ilo=0, ihi=500, wfin="waveform", calc="bl_p0", test=False):
    """
    simple mean, vectorized baseline calculator
    """
    wfs = waves["waveform"]

    # find wf means
    avgs = np.mean(wfs[:, ilo:ihi], axis=1)

    # add the result as a new column
    calcs[calc] = avgs


if __name__=="__main__":
    main()
