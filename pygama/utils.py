"""
this is a miscellaneous functions store.  i found one useful before (waveLibs)
so is it good practice to keep a file like this?
"""
import sys
import numpy as np


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


def get_bin_centers(bins):
    return bins[:-1] + 0.5 * (bins[1] - bins[0])


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


def sh(cmd, sh=False):
    """ Wraps a shell command."""
    import shlex
    import subprocess as sp
    if not sh: sp.call(shlex.split(cmd))  # "safe"
    else: sp.call(cmd, shell=sh)  # "less safe"

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
