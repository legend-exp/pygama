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
