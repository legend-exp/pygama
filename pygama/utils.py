"""
pygama convenience functions.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt


class SafeDict(dict):
    """
    used in handling LEGEND file format strings.
    when a key is missing, return the string value of that key.
    """
    def __missing__(self, key):
        return '{' + key + '}'


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


def sh(cmd, sh=False):
    """
    input a shell command as you would type it on the command line.
    """
    decoders = []
    for sub in DataTaker.__subclasses__():
        for subsub in sub.__subclasses__():
            try:
                decoder = subsub(object_info) # initialize the decoder
                decoders.append(decoder)
            except Exception as e:
                print(e)
                pass
    return decoders


def update_progress(progress, run=None):
    """
    adapted from from https://stackoverflow.com/a/15860757
    """
    barLength = 20  # length of the progress bar
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

    if run is None:
        text = "\rProgress : [{}] {:0.1f}% {}".format(
            "#" * block + "-" * (barLength - block), progress * 100, status)
    else:
        text = "\rProgress : [{}] {:0.1f}% {} (Run {})".format(
            "#" * block + "-" * (barLength - block), progress * 100, status,
            run)

    sys.stdout.write(text)
    sys.stdout.flush()


def sizeof_fmt(num, suffix='B'):
    """
    given a file size in bytes, output a human-readable form.
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1000.0:
            return "{:.3f} {}{}".format(num, unit, suffix)
        num /= 1000.0
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
            tmp = getattr(tree, "GetV{}".format(j + 1))()
            np_arrs[i + j] = np.array([tmp[k] for k in range(n)])

    return tuple(np_arrs)


def peakdet(v, delta, x=None):
    """
    Converted from MATLAB script at: http://billauer.co.il/peakdet.html
    Returns two arrays: [maxtab, mintab] = peakdet(v, delta, x)
    An updated (vectorized) version is in pygama.dsp.transforms.peakdet
    """
    maxtab, mintab = [], []

    if x is None:
        x = np.arange(len(v))
    v = np.asarray(v)

    # sanity checks
    if len(v) != len(x):
        exit("Input vectors v and x must have same length")
    if not np.isscalar(delta):
        exit("Input argument delta must be a scalar")
    if delta <= 0:
        exit("Input argument delta must be positive")

    maxes, mins = [], []
    min, max = np.inf, -np.inf
    find_max = True
    for i in range(len(x)):

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


def sh(cmd, sh=False):
    """
    input a shell command as you would type it on the command line.
    """
    import shlex
    import subprocess as sp
    if not sh:
        sp.call(shlex.split(cmd))  # "safe"
    else:
        sp.call(cmd, shell=sh)  # "less safe"


def linear_fit_by_sums(x, y, var=1):
    """
    Fast computation of weighted linear least squares fit to a linear model

    Note: doesn't compute covariances. If you want covariances, just use polyfit

    Parameters
    ----------
    x : array like
        x values for the fit
    y : array like
        y values for the fit
    var : array like (optional)
        The variances for each y-value

    Returns
    -------
    (m, b) : tuple (float, float)
        The slope (m) and y-intercept (b) of the best fit (in the least-squares
        sense) of the data to y = mx + b
    """
    y = y/var
    x = x/var
    sum_wts = len(y)/var if np.isscalar(var) else np.sum(1/var) 
    sum_x = np.sum(x)
    sum_xx = np.sum(x*x)
    sum_y = np.sum(y)
    sum_yx = np.sum(y*x)
    m = (sum_wts * sum_yx - sum_y * sum_x) / (sum_wts * sum_xx - sum_x**2)
    b = (sum_y - m * sum_x) / sum_wts
    return m, b


