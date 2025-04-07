r"""
A class that creates the sum of distributions, with methods for scipy computed :func:`pdfs` and :func:`cdfs`, as well as fast :func:`get_pdfs`, and :func:`pdf_ext`

.. code-block:: python

    mu1, mu2, sigma, frac = range(4)
    moyal_add = SumDists([(moyal, [mu1, sigma]), (moyal, [mu2, sigma])], [frac], "fracs") # create two moyals that share a sigma and differ by a fraction,
    x = np.arange(-10,10)
    pars = np.array([1, 2, 2, 0.1]) # corresponds to mu1 = 1, mu2 = 2, sigma = 2, frac=0.1
    moyal_add.pdf(x, *pars)
    moyal_add.draw_pdf(x, *pars)
    moyal_add.required_args()
"""

import inspect
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats._distn_infrastructure import rv_continuous

from pygama.math.functions.pygama_continuous import PygamaContinuous


def get_dists_and_par_idxs(
    dists_and_pars_array: np.array(tuple, tuple),
) -> Tuple[np.array, np.array]:
    r"""
    Split the array of tuples passed to the :func:`SumDists` constructor into separate arrays with one containing only the
    distributions and the other containing the parameter index arrays. Also performs some sanity checks.

    Parameters
    ----------
    dists_and_pars_array
        An array of tuples, each tuple contains a distribution and the indices of the parameters
        in the :func:`SumDists` call that correspond to that distribution

    Returns
    -------
    dists, par_idxs
        Returned only if every distribution has the correct number of required arguments
    """
    # can only sum two dists at once, check:
    if len(dists_and_pars_array) != 2:
        raise ValueError("Can only sum two distributions at once.")

    dists = []
    par_idxs = []

    # Check that each tuple in the dists_and_pars_array is of length two
    for dist_and_pars in dists_and_pars_array:
        if len(dist_and_pars) != 2:
            raise ValueError(
                "Each tuple needs a distribution and a parameter index array."
            )
        else:
            dists.append(dist_and_pars[0])
            par_idxs.append(np.array(dist_and_pars[1]))

    return dists, par_idxs


def get_areas_fracs(
    params: np.array,
    area_frac_idxs: np.array,
    frac_flag: bool,
    area_flag: bool,
    one_area_flag: bool,
) -> Tuple[np.array, np.array]:
    r"""
    Grab the value(s) of either the fraction or the areas passed in the params array from the :func:`SumDists` call.
    If :func:`SumDists` is in "fracs" mode, then this grabs `f` from the params array and returns fracs = [f, 1-f] and areas of unity.
    If :func:`SumDists` is in "areas" mode, then this grabs `s, b` from the params array and returns unity fracs and areas = [s, b]
    If :func:`SumDists` is in "one_area" mode, then this grabs `s` from the params array and returns unity fracs and areas = [s, 1]

    Parameters
    ----------
    params
        An array containing the shape values from a :func:`SumDists` call
    area_frac_idxs
        An array containing the indices of either the fracs or the areas present in the params array
    frac_flag
        A boolean telling if :func:`SumDists` is in fracs mode or not
    area_flag
        A boolean telling if :func:`SumDists` is in areas mode or not
    one_area_flag
        A boolean telling if :func:`SumDists` is to apply only area to one distribution

    Returns
    -------
    fracs, areas
        Values of the fractions and the areas to post-multiply the sum of the distributions with
    """
    if frac_flag:
        fracs = np.array([params[area_frac_idxs[0]], 1 - params[area_frac_idxs[0]]])
        areas = np.array([1, 1])
    elif area_flag:
        fracs = np.array([1, 1])
        areas = np.array([*params[area_frac_idxs]])
    elif one_area_flag:
        fracs = np.array([1, 1])
        areas = np.array([*params[area_frac_idxs], 1])

    else:
        fracs = np.array([1, 1])
        areas = np.array([1, 1])

    return fracs, areas


def get_parameter_names(dists: np.array, par_idxs: np.array, par_size: int) -> np.array:
    r"""
    Returns an array of the names of the required parameters for an instance of :func:`SumDists`
    Works by calling :func:`.required_args` for each distribution present in the sum.
    If a parameter is shared between distributions its name is only added once.
    If two parameters are required and share a name, then the second parameter gets an added index at the end.

    Parameters
    ----------
    dists
        An array containing the distributions in this instance of :func:`SumDists`
    par_idxs
        An array of arrays, each array contains the indices of the parameters in the :func:`SumDists` call that correspond to that distribution
    par_size
        The size of the single parameter index array

    Returns
    -------
    param_names
        An array containing the required parameter names
    """
    param_names = np.empty(par_size + 1, dtype=object)
    overall_par_idxs = []
    for i in range(len(dists)):
        mask = ~np.isin(
            par_idxs[i], overall_par_idxs
        )  # see if indices of the required args that are not included yet
        new_idxs = par_idxs[i][
            mask
        ]  # get indices of the required args that are not included yet
        prereq_names = np.array(dists[i].required_args())[mask]
        req_names = []
        # Check for duplicate names
        for name in prereq_names:
            if name in param_names:
                if name[-1].isdigit():
                    name = name[:-1] + f"{int(name[-1])+1}"
                else:
                    name = name + "1"
            req_names.append(name)

        param_names[new_idxs] = req_names
        overall_par_idxs.extend(par_idxs[i])
    return param_names


def copy_signature(signature_to_copy, obj_to_copy_to):
    """
    Copy the signature provided in signature_to_copy into the signature for "obj_to_copy_to".
    This is necessary so that we can override the signature for the various methods attached to
    different objects.
    """

    def wrapper(*args, **kwargs):
        return obj_to_copy_to(*args, **kwargs)

    wrapper.__signature__ = signature_to_copy
    return wrapper


class SumDists(rv_continuous):
    r"""
    Initialize an rv_continuous method so that we gain access to scipy computable methods.
    Precompute the support of the sum of the distributions.

    The correct way to initialize is SumDists([(d1, [p1]), (d2, [p2])], [area_idx_1/frac_idx, area_idx_2/], frac_flag)
    Where d_i is a distribution and p_i is a parameter index array for that distribution.

    Parameter index arrays contain indices that slice a single parameter array that is passed to method calls.
    For example, if the user will eventually pass parameters=[mu, sigma, tau, frac] to function.get_pdf(x, parameters)
    and the first distribution takes (mu, sigma) as its parameters, then p1=[0,1]. If the second distribution takes (mu, sigma, tau)
    then its parameter index array would be p2=[0, 1, 2] because tau is the index 2 entry in parameters.

    Each par array can contain [x_lo, x_hi, mu, sigma, shape], and *must be placed in that order*.

    The single parameter array passed to function calls *must* follow the ordering convention [x_lo, x_hi, frac/areas, shapes_1, frac/areas2, shapes_2]
    The single parameter array that is passed to :func:`pdf_norm`, :func:`pdf_ext`, and :func:`cdf_norm` calls *must* have x_lo, x_hi
    as its first two arguments if none of the distributions require an explicit definition of their support.

    There are 4 flag options:

    1. flag = "areas", two areas passed as fit variables
    2. flag = "fracs", one fraction is passed a a fit variable, with the convention that SumDists performs f*dist_1 + (1-f)*dist_2
    3. flag = "one_area", one area is passed a a fit variable, with the convention that SumDists performs area*dist_1 + 1*dist_2
    4. flag = None, no areas or fracs passed as fit variables, both are normalized to unity.

    Notes
    -----
    dists must be unfrozen pygama distributions of the type :func:`PygamaContinuous` or :func:`sum_dist`.
    """

    def _argcheck(self, *args) -> bool:
        """
        Check that each distribution gets its own valid arguments.

        Notes
        -----
        This overloads the :func:`scipy` definition so that the methods :func:`.pdf` work.
        """
        args = np.array(args)
        cond = True
        for i, dist in enumerate(self.dists):
            cond = np.logical_and(cond, dist._argcheck(*args[self.par_idxs[i]]))
        return cond

    def __init__(
        self,
        dists_and_pars_array,
        area_frac_idxs,
        flag,
        parameter_names=None,
        components=False,
        support_required=False,
        **kwds,
    ):
        """
        Parameters
        ----------
        dists_and_pars_array
            A list of two tuples, containing [(dist_1, [mu1, sigma1, shapes1]), (dist_2, [mu2, sigma2, shapes2])] to create a sum_dist from
        area_frac_idxs
            A list of the indices at which that either the areas or fraction will be placed in the eventual method calls
        flag
            One of three strings that initialize :func:`sum_dist` in different modes. Either "fracs", "areas" or "one_area".
        parameter_names
            An optional list of strings that contain the parameters names in the order they will appear in method calls
        components
            A boolean that if true will cause methods to return components instead of the sum of the distributions
        support_required
            A boolean that if true tells :func:`sum_dist` that x_lo, x_hi will *always* be passed in method calls.
        """
        # Extract the distributions and parameter index arrays from the constructor
        dists, par_idxs = get_dists_and_par_idxs(dists_and_pars_array)

        self.dists = dists
        self.par_idxs = par_idxs
        self.area_frac_idxs = area_frac_idxs
        self.components = components
        self.support_required = support_required

        # Check that the dists are in fact distributions
        for i in range(len(dists)):
            if (not isinstance(dists[i], PygamaContinuous)) and (
                not isinstance(dists[i], SumDists)
            ):
                raise ValueError(
                    f"Distribution at index {i} has value {dists[i]},\
                and is an array and not a PygamaContinuous distribution"
                )

        # Get the parameter names for later introspection
        # First, find the length of the eventual single parameter array
        par_size = 0
        for par_idx in par_idxs:
            if np.amax(par_idx) >= par_size:
                par_size = np.amax(par_idx)
        par_size = (
            np.amax([par_size, np.amax(area_frac_idxs)])
            if len(area_frac_idxs) != 0
            else par_size
        )
        shapes = get_parameter_names(dists, par_idxs, par_size)

        # Set the internal state depending on what flag was passed
        if flag == "fracs":
            if len(area_frac_idxs) != 1:
                raise ValueError(
                    "SumDists only accepts the parameter position of one fraction."
                )
            self.frac_flag = True
            self.area_flag = False
            self.one_area_flag = False
            shapes[area_frac_idxs[0]] = "f"  # add frac name to the shapes
        elif flag == "areas":
            if len(area_frac_idxs) != 2:
                raise ValueError("SumDists needs two parameter indices of areas.")
            self.frac_flag = False
            self.area_flag = True
            self.one_area_flag = False
            shapes[area_frac_idxs] = ["s", "b"]  # add area names to the shapes
        elif flag == "one_area":
            # needed so that we can create SumDists from SumDists without having an overall area to the second dist
            # Sets the area of the second dist to 1
            if len(area_frac_idxs) != 1:
                raise ValueError("SumDists needs one parameter index of an area.")
            self.frac_flag = False
            self.area_flag = False
            self.one_area_flag = True
            shapes[area_frac_idxs] = "s"  # add area name to the shapes
        else:
            self.frac_flag = False
            self.area_flag = False
            self.one_area_flag = False

        shapes = list(shapes)
        # override the parameter names if string is passed to the constructor
        if parameter_names is not None:
            shapes = parameter_names

        # If a support is explicitly passed, flag it.
        # Record the index of x_lo and x_hi if they are required, if they aren't set them to 0, 1
        # This is so that :func:`pdf_ext`, :func:`pdf_norm` and :func:`cdf_norm` can find the x_lo, x_hi if support is required, otherwise they take x_lo, x_hi from the start of the parameter array
        if "x_lo" in shapes or "x_hi" in shapes:
            self.support_required = True
            self.x_lo_idx = shapes.index("x_lo")
            self.x_hi_idx = shapes.index("x_hi")
            extended_shapes = [
                "x",
                *shapes,
            ]  # get the parameter names for methods that require the fit range passed, like extended fits
            self.extended_shapes = extended_shapes
        else:
            self.x_lo_idx = 0
            self.x_hi_idx = 1
            extended_shapes = [
                "x",
                "x_lo",
                "x_hi",
                *shapes,
            ]  # get the parameter names for methods that require the fit range passed, like extended fits
            self.extended_shapes = extended_shapes

        # Now that we have inferred or set the shapes, store them so that :func:`required_args` can return them
        self.req_args = shapes

        # set attributes for the methods so that Iminuit can introspect parameter names
        x_shapes = [
            "x",
            *shapes,
        ]  # add the x values to the parameter names that are being passed, Iminuit assumes that this is in the first position
        self.x_shapes = x_shapes

        # Scipy requires the argument names as one string with commas in between each parameter name
        shapes = ",".join(str(x) for x in shapes)

        super().__init__(self, shapes=shapes, **kwds)

    def _pdf(self, x, *params):
        """
        Overload :func:`rv_continuous` definition of pdf in order to access other methods.
        """
        pdfs = self.dists
        params = np.array(params)[:, 0]  # scipy ravels the parameter array...

        fracs, areas = get_areas_fracs(
            params,
            self.area_frac_idxs,
            self.frac_flag,
            self.area_flag,
            self.one_area_flag,
        )

        if self.components:
            return areas[0] * fracs[0] * pdfs[0].pdf(
                x, *params[self.par_idxs[0]]
            ), areas[1] * fracs[1] * pdfs[1].pdf(x, *params[self.par_idxs[1]])

        else:
            # This is faster than list comprehension
            probs = areas[0] * fracs[0] * pdfs[0].pdf(
                x, *params[self.par_idxs[0]]
            ) + areas[1] * fracs[1] * pdfs[1].pdf(x, *params[self.par_idxs[1]])
            return probs

    def _cdf(self, x, *params):
        """
        Overload :func:`rv_continuous` definition of cdf in order to access other methods.
        """
        cdfs = self.dists
        params = np.array(params)[:, 0]

        fracs, areas = get_areas_fracs(
            params,
            self.area_frac_idxs,
            self.frac_flag,
            self.area_flag,
            self.one_area_flag,
        )

        if self.components:
            return areas[0] * fracs[0] * cdfs[0].cdf(
                x, *params[self.par_idxs[0]]
            ), areas[1] * fracs[1] * cdfs[1].cdf(x, *params[self.par_idxs[1]])

        else:
            # This is faster than list comprehension
            probs = areas[0] * fracs[0] * cdfs[0].cdf(
                x, *params[self.par_idxs[0]]
            ) + areas[1] * fracs[1] * cdfs[1].cdf(x, *params[self.par_idxs[1]])
            return probs

    def get_pdf(self, x, *params):
        """
        Returns the specified sum of all distributions' :func:`get_pdf` methods.
        """
        pdfs = self.dists
        params = np.array(params)

        fracs, areas = get_areas_fracs(
            params,
            self.area_frac_idxs,
            self.frac_flag,
            self.area_flag,
            self.one_area_flag,
        )

        if self.components:
            return areas[0] * fracs[0] * pdfs[0].get_pdf(
                x, *params[self.par_idxs[0]]
            ), areas[1] * fracs[1] * pdfs[1].get_pdf(x, *params[self.par_idxs[1]])

        else:
            # This is faster than list comprehension
            probs = areas[0] * fracs[0] * pdfs[0].get_pdf(
                x, *params[self.par_idxs[0]]
            ) + areas[1] * fracs[1] * pdfs[1].get_pdf(x, *params[self.par_idxs[1]])
            return probs

    def get_cdf(self, x, *params):
        """
        Returns the specified sum of all distributions' :func:`get_cdf` methods.
        """
        cdfs = self.dists
        params = np.array(params)

        fracs, areas = get_areas_fracs(
            params,
            self.area_frac_idxs,
            self.frac_flag,
            self.area_flag,
            self.one_area_flag,
        )

        if self.components:
            return areas[0] * fracs[0] * cdfs[0].get_cdf(
                x, *params[self.par_idxs[0]]
            ), areas[1] * fracs[1] * cdfs[1].get_cdf(x, *params[self.par_idxs[1]])

        else:
            # This is faster than list comprehension
            probs = areas[0] * fracs[0] * cdfs[0].get_cdf(
                x, *params[self.par_idxs[0]]
            ) + areas[1] * fracs[1] * cdfs[1].get_cdf(x, *params[self.par_idxs[1]])
            return probs

    def pdf_norm(self, x, *params):
        """
        Returns the specified sum of all distributions' :func:`get_pdf` methods, but normalized on a range x_lo to x_hi. Used in unbinned NLL fits.

        NOTE: This assumes that x_lo, x_hi are the first two elements in the parameter array, unless x_lo and x_hi are required to define the support
        of the :func:`SumDists` for *all* method calls. For :func:`SumDists` created from a distribution where the support needs
        to be explicitly passed, this means that :func:`pdf_norm` sets the fit range equal to the support range.
        It also means that summing distributions with two different explicit supports is not allowed, e.g. we cannot sum two step functions with different supports.
        """
        # Grab the values of x_lo and x_hi from the parameters. For functions where the support needs to be defined, this can be anywhere in the parameter array
        # Otherwise, x_lo and x_hi *must* be the first two values passed.
        x_lo = params[self.x_lo_idx]
        x_hi = params[self.x_hi_idx]
        if not self.support_required:
            params = params[
                2:
            ]  # chop off x_lo, x_hi from the shapes to actually pass to get_cdf

        norm = np.diff(self.get_cdf(np.array([x_lo, x_hi]), *params))

        if norm == 0:
            return np.full_like(x, np.inf)
        else:
            return self.get_pdf(x, *params) / norm

    def cdf_norm(self, x, *params):
        """
        Returns the specified sum of all distributions' :func:`get_cdf` methods, but normalized on a range x_lo to x_hi. Used in binned NLL fits.

        NOTE: This assumes that x_lo, x_hi are the first two elements in the parameter array, unless x_lo and x_hi are required to define the support
        of the :func:`SumDists` for *all* method calls. For :func:`SumDists` created from a distribution where the support needs
        to be explicitly passed, this means that :func:`cdf_norm` sets the fit range equal to the support range.
        It also means that summing distributions with two different explicit supports is not allowed, e.g. we cannot sum two step functions with different supports.
        """
        # Grab the values of x_lo and x_hi from the parameters. For functions where the support needs to be defined, this can be anywhere in the parameter array
        # Otherwise, x_lo and x_hi *must* be the first two values passed.
        x_lo = params[self.x_lo_idx]
        x_hi = params[self.x_hi_idx]
        if not self.support_required:
            params = params[
                2:
            ]  # chop off x_lo, x_hi from the shapes to actually pass to get_cdf

        norm = np.diff(self.get_cdf(np.array([x_lo, x_hi]), *params))

        if norm == 0:
            return np.full_like(x, np.inf)
        else:
            return (self.get_cdf(x, *params)) / norm

    def pdf_ext(self, x, *params):
        """
        Returns the specified sum of all distributions' :func:`pdf_ext` methods, normalized on a range x_lo to x_hi. Used in extended unbinned NLL fits.

        NOTE: This assumes that x_lo, x_hi are the first two elements in the parameter array, unless x_lo and x_hi are required to define the support
        of the :func:`SumDists` for *all* method calls. For :func:`SumDists` created from a distribution where the support needs
        to be explicitly passed, this means that :func:`pdf_ext` sets the fit range equal to the support range.
        It also means that summing distributions with two different explicit supports is not allowed, e.g. we cannot sum two step functions with different supports.
        """
        pdf_exts = self.dists
        params = np.array(params)

        # Grab the values of x_lo and x_hi from the parameters. For functions where the support needs to be defined, this can be anywhere in the parameter array
        # Otherwise, x_lo and x_hi *must* be the first two values passed.
        x_lo = params[self.x_lo_idx]
        x_hi = params[self.x_hi_idx]
        if not self.support_required:
            params = params[
                2:
            ]  # chop off x_lo, x_hi from the shapes to actually pass to get_cdf

        fracs, areas = get_areas_fracs(
            params,
            self.area_frac_idxs,
            self.frac_flag,
            self.area_flag,
            self.one_area_flag,
        )

        # sig = areas[0] + areas[1] # this is a hack, it performs faster but may not *always* be true
        sig = (
            areas[0]
            * fracs[0]
            * np.diff(
                pdf_exts[0].get_cdf(np.array([x_lo, x_hi]), *params[self.par_idxs[0]])
            )[0]
            + areas[1]
            * fracs[1]
            * np.diff(
                pdf_exts[1].get_cdf(np.array([x_lo, x_hi]), *params[self.par_idxs[1]])
            )[0]
        )

        if self.components:
            return (
                sig,
                areas[0] * fracs[0] * pdf_exts[0].get_pdf(x, *params[self.par_idxs[0]]),
                areas[1] * fracs[1] * pdf_exts[1].get_pdf(x, *params[self.par_idxs[1]]),
            )
        else:
            probs = areas[0] * fracs[0] * pdf_exts[0].get_pdf(
                x, *params[self.par_idxs[0]]
            ) + areas[1] * fracs[1] * pdf_exts[1].get_pdf(x, *params[self.par_idxs[1]])

            return sig, probs

    def cdf_ext(self, x, *params):
        """
        Returns the specified sum of all distributions' :func:`get_cdf` methods, used in extended binned NLL fits.
        """
        cdf_exts = self.dists
        params = np.array(params)

        fracs, areas = get_areas_fracs(
            params,
            self.area_frac_idxs,
            self.frac_flag,
            self.area_flag,
            self.one_area_flag,
        )

        if self.components:
            return areas[0] * fracs[0] * cdf_exts[0].get_cdf(
                x, *params[self.par_idxs[0]]
            ), areas[1] * fracs[1] * cdf_exts[1].get_cdf(x, *params[self.par_idxs[1]])

        else:
            # This is faster than list comprehension
            probs = areas[0] * fracs[0] * cdf_exts[0].get_cdf(
                x, *params[self.par_idxs[0]]
            ) + areas[1] * fracs[1] * cdf_exts[1].get_cdf(x, *params[self.par_idxs[1]])
            return probs

    def required_args(self) -> list:
        """
        Returns
        -------
        req_args
            A list of the required arguments for the :func:`SumDists` instance, either passed by the user, or inferred.
        """
        return self.req_args

    def draw_pdf(self, x: np.ndarray, *params) -> None:
        plt.plot(x, self.get_pdf(x, *params))

    def draw_cdf(self, x: np.ndarray, *params) -> None:
        plt.plot(x, self.get_cdf(x, *params))

    # Create some convenience functions for fitting

    def get_mu(
        self, pars: np.ndarray, cov: np.ndarray = None, errors: np.ndarray = None
    ) -> tuple:
        r"""
        Get the mu value from the output of a fit quickly

        Parameters
        ----------
        pars
            Array of fit parameters
        cov
            Array of covariances
        errors
            Array of errors

        Returns
        -------
        mu, error
            where mu is the fit value, and error is either from the covariance matrix or directly passed
        """

        req_args = np.array(self.required_args())
        mu_idx = np.where(req_args == "mu")[0][0]
        mu = pars[mu_idx]

        if errors is not None:
            return mu, errors[mu_idx]
        elif cov is not None:
            return mu, np.sqrt(cov[mu_idx][mu_idx])
        else:
            return mu

    def get_mode(
        self, pars: np.ndarray, cov: np.ndarray = None, errors: np.ndarray = None
    ) -> tuple:
        r"""
        Get the mode value from the output of a fit quickly
        Need to overload this to use hpge_peak_fwhm (to avoid a circular import) for when self is an hpge peak

        Parameters
        ----------
        pars
            Array of fit parameters
        cov
            Array of covariances
        errors
            Array of errors

        Returns
        -------
        mu, error
            where mu is the fit value, and error is either from the covariance matrix or directly passed
        """

        req_args = np.array(self.required_args())
        mu_idx = np.where(req_args == "mu")[0][0]
        mu = pars[mu_idx]

        if errors is not None:
            return mu, errors[mu_idx]
        elif cov is not None:
            return mu, np.sqrt(cov[mu_idx][mu_idx])
        else:
            return mu

    def get_fwhm(self, pars: np.ndarray, cov: np.ndarray = None) -> tuple:
        r"""
        Get the fwhm value from the output of a fit quickly
        Need to overload this to use hpge_peak_fwhm (to avoid a circular import) for when self is an hpge peak,
        and otherwise returns 2sqrt(2log(2))*sigma

        Parameters
        ----------
        pars
            Array of fit parameters
        cov
            Optional, array of covariances for calculating error on the fwhm


        Returns
        -------
        fwhm, error
            the value of the fwhm and its error
        """

        req_args = np.array(self.required_args())
        sigma_idx = np.where(req_args == "sigma")[0][0]

        if cov is None:
            return pars[sigma_idx] * 2 * np.sqrt(2 * np.log(2))
        else:
            return pars[sigma_idx] * 2 * np.sqrt(2 * np.log(2)), np.sqrt(
                cov[sigma_idx][sigma_idx]
            ) * 2 * np.sqrt(2 * np.log(2))

    def get_fwfm(self, pars: np.ndarray, cov: np.ndarray = None, frac_max=0.5) -> tuple:
        r"""
        Get the fwfm value from the output of a fit quickly
        Need to overload this to use hpge_peak_fwfm (to avoid a circular import) for when self is an hpge peak,
        and otherwise returns 2sqrt(2log(2))*sigma

        Parameters
        ----------
        pars
            Array of fit parameters
        cov
            Optional, array of covariances for calculating error on the fwhm


        Returns
        -------
        fwhm, error
            the value of the fwhm and its error
        """

        req_args = np.array(self.required_args())
        sigma_idx = np.where(req_args == "sigma")[0][0]

        if cov is None:
            return pars[sigma_idx] * 2 * np.sqrt(-2 * np.log(frac_max))
        else:
            return pars[sigma_idx] * 2 * np.sqrt(-2 * np.log(frac_max)), np.sqrt(
                cov[sigma_idx][sigma_idx]
            ) * 2 * np.sqrt(-2 * np.log(frac_max))

    def get_total_events(
        self, pars: np.ndarray, cov: np.ndarray = None, errors: np.ndarray = None
    ) -> tuple:
        r"""
        Get the total events from the output of an extended fit quickly
        The total number of events come from the sum of the area of the signal and the area of the background components

        Parameters
        ----------
        pars
            Array of fit parameters
        cov
            Array of covariances
        errors
            Array of errors

        Returns
        -------
        n_total, error
            the total number of events in a spectrum and its associated error
        """

        req_args = np.array(self.required_args())
        n_sig_idx = np.where(req_args == "n_sig")[0][0]
        n_bkg_idx = np.where(req_args == "n_bkg")[0][0]

        if errors is not None:
            return pars[n_sig_idx] + pars[n_bkg_idx], np.sqrt(
                errors[n_sig_idx] ** 2 + errors[n_bkg_idx] ** 2
            )
        elif cov is not None:
            return pars[n_sig_idx] + pars[n_bkg_idx], np.sqrt(
                cov[n_sig_idx][n_sig_idx] ** 2 + cov[n_bkg_idx][n_bkg_idx] ** 2
            )
        else:
            return pars[n_sig_idx] + pars[n_bkg_idx]

    def __getattribute__(self, attr):
        """
        Necessary to overload this so that Iminuit can use inspect.signature to get the correct parameter names
        """
        value = object.__getattribute__(self, attr)

        if attr in ["get_pdf", "get_cdf", "cdf_ext"]:
            params = [
                inspect.Parameter(param, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for param in self.x_shapes
            ]
            value = copy_signature(inspect.Signature(params), value)
        if attr in [
            "pdf_norm",
            "pdf_ext",
            "cdf_norm",
        ]:  # Set these to include the x_lo, x_hi at the correct positions since these always need those params
            params = [
                inspect.Parameter(param, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for param in self.extended_shapes
            ]
            value = copy_signature(inspect.Signature(params), value)
        return value
