"""
this is a miscellaneous / frequently used functions store.
is it good practice to keep a file like this?
"""
import sys
import numpy as np


def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at: http://billauer.co.il/peakdet.html
    Returns two arrays:
        [maxtab, mintab] = peakdet(v, delta, x)
    # PEAKDET Detect peaks in a vector
    #   [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    #   maxima and minima ("peaks") in the vector V.
    #   MAXTAB and MINTAB consist of two columns. Column 1
    #   contains indices in V, and column 2 the found values.
    #   With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    #   in MAXTAB and MINTAB are replaced with the corresponding
    #   X-values.
    #   A point is considered a maximum peak if it has the maximal
    #   value, and was preceded (to the left) by a value lower by
    #   DELTA.
    Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    This function is released to the public domain; Any use is allowed.

    TODO: can we vectorize this?
    # i think to vectorize this, i still have to a loop over
    # the timestamps, because the max/mins are set based on the previous
    # value.  we could probably store the individual max/mins in arrays,
    # and also find_max would have to be turned into an array.
    # so this can only be "partially" vectorized at best.
    # i'd have to really think about how to get rid of the loop
    # over ts's.  maybe some kind of cumsum, like the trap filter?
    """
    maxtab, mintab = [], []

    if x is None:
        x = np.arange(len(v))
    v = np.asarray(v)
    if len(v) != len(x):
        sys.exit('Peak Finder Error: Input vectors v and x must have same length')
    if not np.isscalar(delta):
        sys.exit('Peak Finder Error: Input argument delta must be a scalar')
    if delta <= 0:
        sys.exit('Peak Finder Error: Input argument delta must be positive')

    maxes, mins = [], []
    min, max = np.inf, -np.inf
    find_max = True
    for i in x:

        # for i=0, all 4 of these get set
        if v[i] > max:
            max, imax = v[i], x[i]
        if v[i] < min:
            min, imin = v[i], x[i]

        if find_max:
            # if the sample is less than the current max,
            # declare the previous one a maximum, then set this as the new "min"
            if v[i] < max - delta:
                maxes.append((imax, max))
                min, imin = v[i], x[i]
                find_max = False
        else:
            # if the sample is more than the current min,
            # declare the previous one a minimum, then set this as the new "max"
            if v[i] > min + delta:
                mins.append((imin, min))
                max, imax = v[i], x[i]
                find_max = True

    return np.array(maxes), np.array(mins)


def TDraw(tree, vars, tcut):
    """
    if you have to debase yourself and use ROOT, this is an easy
    convenience function for quickly extracting data from TTrees.
    TTree::Draw can only handle groups of 4 variables at a time,
    but here we can put in as many as we want, and
    return a list of numpy.ndarrays for each one
    """
    var_list = vars.split(":")
    np_arrs = [[] for v in var_list]

    for i in range(0, len(var_list), 4):

        tmp_list = var_list[i:i+4]
        tmp_draw = ":".join(tmp_list)
        n = tree.Draw(tmp_draw, tcut, "goff")

        for j, var in enumerate(tmp_list):
            # print(i, j, var, "getting V", j+1, "writing to np_arrs", i + j)
            tmp = getattr(tree, "GetV{}".format(j+1))()
            np_arrs[i + j] = np.array([tmp[k] for k in range(n)])

    return tuple(np_arrs)


def update_progress(progress, runNumber=None):
    """ adapted from from https://stackoverflow.com/a/15860757 """
    barLength = 20  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    if runNumber is None:
        text = "\rPROGRESS : [{}] {:0.3f}% {}".format(
            "#" * block + "-" * (barLength - block), progress * 100, status)
    else:
        text = "\rPROGRESS : [{}] {:0.3f}% {} (Run {})".format(
            "#" * block + "-" * (barLength - block), progress * 100, status,
            runNumber)

    sys.stdout.write(text)
    sys.stdout.flush()


def sizeof_fmt(num, suffix='B'):
    """ Given a file size in bytes, output a human-readable form. """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1000.0:
            return "{:.3f} {}{}".format(num, unit, suffix)
        num /= 1000.0
    return "{:.1f} {} {}".format(num, 'Y', suffix)


def get_hist(np_arr, x_lo, x_hi, xpb, nb=None, shift=True, wts=None):
    """ quick wrapper to have more control of numpy's histogram """
    if nb is None:
        nb = int((x_hi - x_lo) / xpb)
    y, x = np.histogram(np_arr, bins=nb, range=(x_lo, x_hi), weights=wts)
    y = np.insert(y, 0, 0, axis=0)
    if shift:
        x = x - xpb / 2.
    return x, y

def hist(np_arr, bins=None, xrng=None, dx=None, wts=None):
    """ quick wrapper to have more control of numpy's histogram """
    # dx overrides bins setting! Note: have to specify a range
    if dx is not None: bins = int((xrng[1]-xrng[0])/dx)
    hist, bins = np.histogram(np_arr, bins=bins, range=xrng, weights=wts)
    if wts is None: return hist, bins, hist
    var, bins = np.histogram(np_arr, bins=bins, weights=wts*wts)
    return hist, bins, var

def get_bin_centers(bins):
    return bins[:-1] + 0.5 * (bins[1] - bins[0])


def sh(cmd, sh=False):
    """ Wraps a shell command."""
    import shlex
    import subprocess as sp
    if not sh: sp.call(shlex.split(cmd))  # "safe"
    else: sp.call(cmd, shell=sh)  # "less safe"


