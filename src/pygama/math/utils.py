"""
pygama utility functions.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import tqdm


def get_dataset_from_cmdline(args, run_db, cal_db):
    """
    make it easier to call this from argparse:
        arg("-ds", nargs='*', action="store", help="load runs for a DS")
        arg("-r", "--run", nargs=1, help="load a single run")
    """
    from pygama import DataSet

    if args["ds"]:
        ds_lo = int(args["ds"][0])
        try:
            ds_hi = int(args["ds"][1])
        except:
            ds_hi = None
        ds = DataSet(ds_lo, ds_hi, md=run_db, cal=cal_db, v=args["verbose"])

    if args["run"]:
        ds = DataSet(run=int(args["run"][0]), md=run_db, cal=cal_db,
                     v=args["verbose"])
    return ds


def tqdm_range(start, stop, step=1, verbose=False, text=None, bar_length=20, unit=None):
    """
    Uses tqdm.trange which wraps around the python range and also has the option
    to display a progress

    For example:
    .. code-block :: python

        for start_row in range(0, tot_n_rows, buffer_len):
            ...

    Can be converted to the following
    .. code-block :: python

        for start_row in tqdm_range(0, tot_n_rows, buffer_len, verbose):
            ...

    Parameters
    ----------
    start : int
        starting iteration value
    stop : int
        ending iteration value
    step : int
        step size in between each iteration
    verbose : int
        verbose = 0 hides progress bar verbose > 0 displays progress bar
    text : str
        text to display in front of the progress bar
    bar_length : str
        horizontal length of the bar in cursor spaces
    unit : str
        physical units to be displayed

    Returns
    -------
    iterable : tqdm.trange
        object that can be iterated over in a for loop
    """
    hide_bar = True
    if isinstance(verbose, int):
        if verbose > 0:
            hide_bar = False
    elif isinstance(verbose, bool):
        if verbose is True:
            hide_bar = False

    if text is None:
        text = "Processing"

    if unit is None:
        unit = "it"

    bar_format = f"{{l_bar}}{{bar:{bar_length}}}{{r_bar}}{{bar:{-bar_length}b}}"

    return tqdm.trange(start, stop, step,
                       disable=hide_bar, desc=text,
                       bar_format=bar_format, unit=unit, unit_scale=True)


def sizeof_fmt(num, suffix='B'):
    """
    given a file size in bytes, output a human-readable form.
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return f"{num:.3f} {unit}{suffix}"
        num /= 1024.0
    return "{:.1f} {} {}".format(num, 'Y', suffix)


def set_plot_style(style):
    """
    Choose a pygama plot style.
    Current options: 'clint', 'root'
    Or add your own [label].mpl file in the pygama directory!
    """
    path = __file__.rstrip('.utils.py')
    plt.style.use(path+'/'+style+'.mpl')


def get_par_names(func):
    """
    Return a list containing the names of the arguments of "func" other than the
    first argument. In pygamaland, those are the function's "parameters."
    """
    from scipy._lib._util import getargspec_no_self
    args, varargs, varkw, defaults = getargspec_no_self(func)
    return args[1:]


def plot_func(func, pars, range=None, npx=None, **kwargs):
    """
    plot a function.  take care of the x-axis points automatically, or user can
    specify via range and npx arguments.
    """
    if npx is None:
        npx = 100
    if range is None:
        range = plt.xlim()
    xvals = np.linspace(range[0], range[1], npx)
    plt.plot(xvals, func(xvals, *pars), **kwargs)


def get_formatted_stats(mean, sigma, ndigs=2):
    """
    convenience function for formatting mean +/- sigma to the right number of
    significant figures.
    """
    sig_pos = int(np.floor(np.log10(abs(sigma))))
    sig_fmt = '%d' % ndigs
    sig_fmt = '%#.' + sig_fmt + 'g'
    mean_pos = int(np.floor(np.log10(abs(mean))))
    mdigs = mean_pos-sig_pos+ndigs
    if mdigs < ndigs-1: mdigs = ndigs - 1
    mean_fmt = '%d' % mdigs
    mean_fmt = '%#.' + mean_fmt + 'g'
    return mean_fmt % mean, sig_fmt % sigma


def print_fit_results(pars, cov, func=None, title=None, pad=True):
    """
    convenience function for scipy.optimize.curve_fit results
    """
    if title is not None:
        print(title+":")
    if func is None:
        for i in range(len(pars)): par_names.append("p"+str(i))
    else:
        par_names = get_par_names(func)
    for i in range(len(pars)):
        mean, sigma = get_formatted_stats(pars[i], np.sqrt(cov[i][i]))
        print(par_names[i], "=", mean, "+/-", sigma)
    if pad:
        print("")


def tree_draw(tree, vars, tcut):
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

        tmp_list = var_list[i:i + 4]
        tmp_draw = ":".join(tmp_list)
        n = tree.Draw(tmp_draw, tcut, "goff")

        for j, var in enumerate(tmp_list):
            # print(i, j, var, "getting V", j+1, "writing to np_arrs", i + j)
            tmp = getattr(tree, f"GetV{j + 1}")()
            np_arrs[i + j] = np.array([tmp[k] for k in range(n)])

    return tuple(np_arrs)
