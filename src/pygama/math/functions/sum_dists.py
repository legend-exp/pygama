r"""
A class that creates the sum of distributions, with methods for scipy computed :func:`pdfs` and :func:`cdfs`, as well as fast :func:`get_pdfs`, and :func:`pdf_ext` 

.. code-block:: python

    mu1, mu2, sigma = range(3)
    moyal_add = sum_dists([(moyal, [mu1, sigma]), (moyal, [mu2, sigma])], [3], "fracs") # create two moyals that share a sigma and differ by a fraction, 
    x = np.arange(-10,10)
    pars = np.array([1, 2, 2, 0.1]) # corresponds to mu1 = 1, mu2 = 2, sigma = 2, frac=0.1
    moyal_add.pdf(x, *pars)
    moyal_add.draw_pdf(x, *pars)
    moyal_add.get_req_args()


"""

import numba as nb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats._distn_infrastructure import rv_continuous, rv_frozen
from pygama.math.functions.pygama_continuous import pygama_continuous
from typing import Tuple
from iminuit.util import make_func_code

from scipy._lib._util import getfullargspec_no_self as _getfullargspec
def get_dists_and_par_idxs(dists_and_pars_array: np.array(tuple, tuple)) -> Tuple[np.array, np.array]:
    r"""
    Split the array of tuples passed to the :func:`sum_dists` constructor into separate arrays with one containing only the
    distributions and the other containing the parameter index arrays. Also performs some sanity checks.

    Parameters
    ----------
    dists_and_pars_array 
        An array of tuples, each tuple contains a distribution and the indices of the parameters 
        in the :func:`sum_dists` call that correspond to that distribution

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
            raise ValueError("Each tuple needs a distribution and a parameter index array.")
        else:
            dists.append(dist_and_pars[0])
            par_idxs.append(np.array(dist_and_pars[1]))

    return dists, par_idxs

def get_areas_fracs(params: np.array, area_frac_idxs: np.array, frac_flag: bool, area_flag: bool, one_area_flag: bool) -> Tuple[np.array, np.array]:
    r"""
    Grab the value(s) of either the fraction or the areas passed in the params array from the :func:`sum_dists` call. 
    If :func:`sum_dists` is in "fracs" mode, then this grabs `f` from the params array and returns fracs = [f, 1-f] and areas of unity. 
    If :func:`sum_dists` is in "areas" mode, then this grabs `s, b` from the params array and returns unity fracs and areas = [s, b]
    If :func:`sum_dists` is in "one_area" mode, then this grabs `s` from the params array and returns unity fracs and areas = [s, 1]

    Parameters
    ----------
    params 
        An array containing the shape values from a :func:`sum_dists` call
    area_frac_idxs
        An array containing the indices of either the fracs or the areas present in the params array
    frac_flag
        A boolean telling if :func:`sum_dists` is in fracs mode or not
    area_flag
        A boolean telling if :func:`sum_dists` is in areas mode or not
    one_area_flag 
        A boolean telling if :func:`sum_dists` is to apply only area to one distribution

    Returns
    -------
    fracs, areas
        Values of the fractions and the areas to post-multiply the sum of the distributions with
    """
    if frac_flag:
        fracs = np.array([params[area_frac_idxs[0]], 1-params[area_frac_idxs[0]]])
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

def get_parameter_names(dists: np.array, par_idxs: np.array) -> np.array:
    r"""
    Returns an array of the names of the required parameters for an instance of :func:`sum_dists`
    Works by calling :func:`.required_args` for each distribution present in the sum. 
    If a parameter is shared between distributions its name is only added once. 
    If two parameters are required and share a name, then the second parameter gets an added index at the end.

    Parameters
    ----------
    dists 
        An array containing the distributions in this instance of :func:`sum_dists`
    par_idxs
        An array of arrays, each array contains the indices of the parameters in the :func:`sum_dists` call that correspond to that distribution

    Returns
    -------
    param_names
        An array containing the required parameter names
    """
    param_names = []
    overall_par_idxs = []
    for i in range(len(dists)):
        mask = ~np.isin(par_idxs[i], overall_par_idxs) # get indices of the required args that are not included yet 
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
                

        param_names.extend(req_names)
        overall_par_idxs.extend(par_idxs[i])
    return param_names


class sum_dists(rv_continuous):
    r"""
    Initialize an rv_continuous method so that we gain access to scipy computable methods. 
    Precompute the support of the sum of the distributions. 

    The correct way to initialize is sum_dists([(d1, [p1]), (d2, [p2])], [area_idx_1/frac_idx, area_idx_2/], frac_flag)
    Where d_i is a distribution and p_i is a parameter index array for that distribution.

    Parameter index arrays contain indices that slice a single parameter array that is passed to method calls. 
    For example, if the user will eventually pass parameters=[mu, sigma, tau, frac] to function.get_pdf(x, parameters)
    and the first distribution takes (mu, sigma) as its parameters, then p1=[0,1]. If the second distribution takes (tau, mu, sigma)
    then its parameter index array would be p2=[2, 0, 1] because tau is the index 2 entry in parameters. 

    Each par array can contain [shape, mu, sigma], and *must be placed in that order*. 

    The single parameter array passed to function calls should follow the ordering convention [shapes, frac/areas]

    There are 4 flag options:

    1. flag = "areas", two areas passed as fit variables
    2. flag = "fracs", one fraction is passed a a fit variable, with the convention that sum_dists performs f*dist_1 + (1-f)*dist_2
    3. flag = "one_area", one area is passed a a fit variable, with the convention that sum_dists performs area*dist_1 + 1*dist_2
    4. flag = None, no areas or fracs passed as fit variables, both are normalized to unity.

    Notes 
    -----
    dists must be unfrozen pygama distributions of the type :func:`pygama_continuous`
    """

    def set_x_lo(self, x_lo: float) -> None:
        """
        Set the internal state of the lower bound of this distribution
        """
        self.x_lo = x_lo

    def set_x_hi(self, x_hi: float) -> None:
        """
        Set the internal state of the upper bound of this distribution
        """
        self.x_hi = x_hi


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


    def __init__(self, dists_and_pars_array, area_frac_idxs, flag, parameter_names=None):
        self.components = False # do we want users to ever ask for the components back...

        # Extract the distributions and parameter index arrays from the constructor
        dists, par_idxs = get_dists_and_par_idxs(dists_and_pars_array)

        self.dists = dists
        self.par_idxs = par_idxs
        self.area_frac_idxs = area_frac_idxs

        # Get the parameter names for later introspection
        shapes = get_parameter_names(dists, par_idxs)

        # Check that the dists are in fact distributions
        for i in range(len(dists)):
            if (not isinstance(dists[i], pygama_continuous)) and (not isinstance(dists[i], sum_dists)):
                raise ValueError(f"Distribution at index {i} has value {dists[i]},\
                and is an array and not a pygama_continuous distribution")
        
        # Set the internal state depending on what flag was passed
        if flag == "fracs":
            if len(area_frac_idxs) != 1:
                raise ValueError("sum_dists only accepts the parameter position of one fraction.")
            self.frac_flag = True
            self.area_flag = False
            self.one_area_flag = False
            shapes = np.insert(shapes, area_frac_idxs[0], "f") # add frac name to the shapes
        elif flag == "areas":
            if len(area_frac_idxs) != 2:
                raise ValueError("sum_dists needs two parameter indicies of areas.")
            self.frac_flag = False
            self.area_flag = True
            self.one_area_flag = False
            shapes = np.insert(shapes, area_frac_idxs, ["s", "b"]) # add area names to the shapes
        elif flag == "one_area":
            # needed so that we can create sum_dists from sum_dists without having an overall area to the second dist
            # Sets the area of the second dist to 1
            if len(area_frac_idxs) != 1:
                raise ValueError("sum_dists needs one parameter index of an area.")
            self.frac_flag = False
            self.area_flag = False
            self.one_area_flag = True
            shapes = np.insert(shapes, area_frac_idxs, ["s"]) # add area name to the shapes
        else:
            self.frac_flag = False
            self.area_flag = False
            self.one_area_flag = False


        # Set the support by taking the largest possible union, ignore any NoneTypes
        self.set_x_lo(np.nanmax(np.array([dists[0].x_lo, dists[1].x_lo], dtype=np.float64)))
        self.set_x_hi(np.nanmin(np.array([dists[0].x_hi, dists[1].x_hi], dtype=np.float64)))


        # override the parameter names if string is passed to the constructor
        if parameter_names != None:
            shapes = parameter_names


        self.req_args = shapes

        # set atttributes for the methods so that Iminuit can introspect parameter names
        shape_dict = {}
        for i in shapes:
            shape_dict[i] = None

        x_shapes = ["x", *shapes] # add the x values to the parameter names that are being passed, Iminuit assumes that this is in the first position

        # Need to set both the `_parameters` and `func_code` depending on what version of Iminuit is used
        self.get_pdf.__func__._parameters = shape_dict
        self.get_pdf.__func__.func_code = make_func_code(x_shapes)

        self.pdf_ext.__func__._parameters = shape_dict
        self.pdf_ext.__func__.func_code =  make_func_code(x_shapes)

        self.pdf_norm.__func__._parameters = shape_dict
        self.pdf_norm.__func__.func_code = make_func_code(x_shapes)

        self.get_cdf.__func__._parameters = shape_dict
        self.get_cdf.__func__.func_code = make_func_code(x_shapes)

        self.cdf_ext.__func__._parameters = shape_dict
        self.cdf_ext.__func__.func_code =  make_func_code(x_shapes)

        self.cdf_norm.__func__._parameters = shape_dict
        self.cdf_norm.__func__.func_code = make_func_code(x_shapes)

        # Scipy requires the argument names as one string with commas inbetween each parameter name
        shapes = ','.join(str(x) for x in shapes)


        super().__init__(self, shapes=shapes)

    def _pdf(self, x, *params):
        pdfs = self.dists 
        params = np.array(params)[:,0] # scipy ravels the parameter array...

        fracs, areas = get_areas_fracs(params, self.area_frac_idxs, self.frac_flag, self.area_flag, self.one_area_flag)

        if self.components:
            return areas[0]*fracs[0]*pdfs[0].pdf(x, *params[self.par_idxs[0]]), areas[1]*fracs[1]*pdfs[1].pdf(x, *params[self.par_idxs[1]])

        else:
            # This is faster than list comprehension
            probs = areas[0]*fracs[0]*pdfs[0].pdf(x, *params[self.par_idxs[0]]) + areas[1]*fracs[1]*pdfs[1].pdf(x, *params[self.par_idxs[1]])
            return probs

    def _cdf(self, x, *params):
        cdfs = self.dists
        params = np.array(params)[:,0] 

        fracs, areas = get_areas_fracs(params, self.area_frac_idxs, self.frac_flag, self.area_flag, self.one_area_flag)

        if self.components:
            return areas[0]*fracs[0]*cdfs[0].cdf(x, *params[self.par_idxs[0]]) , areas[1]*fracs[1]*cdfs[1].cdf(x, *params[self.par_idxs[1]])

        else:
            # This is faster than list comprehension
            probs = areas[0]*fracs[0]*cdfs[0].cdf(x, *params[self.par_idxs[0]]) + areas[1]*fracs[1]*cdfs[1].cdf(x, *params[self.par_idxs[1]])
            return probs


    def get_pdf(self, x, *params):
        pdfs = self.dists 
        params = np.array(params)

        fracs, areas = get_areas_fracs(params, self.area_frac_idxs, self.frac_flag, self.area_flag, self.one_area_flag)
        
        if self.components:
            return areas[0]*fracs[0]*pdfs[0].get_pdf(x, *params[self.par_idxs[0]]), areas[1]*fracs[1]*pdfs[1].get_pdf(x, *params[self.par_idxs[1]])

        else:
            # This is faster than list comprehension
            probs = areas[0]*fracs[0]*pdfs[0].get_pdf(x, *params[self.par_idxs[0]]) + areas[1]*fracs[1]*pdfs[1].get_pdf(x, *params[self.par_idxs[1]])
            return probs

    def get_cdf(self, x, *params):
        cdfs = self.dists
        params = np.array(params)

        fracs, areas = get_areas_fracs(params, self.area_frac_idxs, self.frac_flag, self.area_flag, self.one_area_flag)

        if self.components:
            return areas[0]*fracs[0]*cdfs[0].get_cdf(x, *params[self.par_idxs[0]]) , areas[1]*fracs[1]*cdfs[1].get_cdf(x, *params[self.par_idxs[1]])

        else:
            # This is faster than list comprehension
            probs = areas[0]*fracs[0]*cdfs[0].get_cdf(x, *params[self.par_idxs[0]]) + areas[1]*fracs[1]*cdfs[1].get_cdf(x, *params[self.par_idxs[1]])
            return probs
        

    def pdf_norm(self, x, *params):

        norm = np.diff(self.get_cdf(np.array([self.x_lo, self.x_hi]), *params))

        if norm == 0:
            return np.full_like(x, np.inf)
        else:
            return self.get_pdf(x, *params)/norm

    def cdf_norm(self, x, *params):

        norm = np.diff(self.get_cdf(np.array([self.x_lo, self.x_hi]), *params))
        
        if norm == 0:
            return np.full_like(x, np.inf)
        else:
            return (self.get_cdf(x, *params))/norm


    def pdf_ext(self, x, *params):
        #NOTE: do we want to pass x_lower and x_upper here or have the user set them before the function call and store as the state? 
        pdf_exts = self.dists 
        params = np.array(params)

        fracs, areas = get_areas_fracs(params, self.area_frac_idxs, self.frac_flag, self.area_flag, self.one_area_flag)

        # sig = areas[0] + areas[1] # this is a hack, it performs faster but may not *always* be true
        sig = areas[0]*fracs[0]*np.diff(pdf_exts[0].get_cdf(np.array([self.x_lo, self.x_hi]), *params[self.par_idxs[0]]))[0] + areas[1]*fracs[1]*np.diff(pdf_exts[1].get_cdf(np.array([self.x_lo, self.x_hi]), *params[self.par_idxs[1]]))[0]
        

        if self.components:
            return sig, areas[0]*fracs[0]*pdf_exts[0].get_pdf(x, *params[self.par_idxs[0]]) , areas[1]*fracs[1]*pdf_exts[1].get_pdf(x, *params[self.par_idxs[1]]) 
        else:
            probs = areas[0]*fracs[0]*pdf_exts[0].get_pdf(x, *params[self.par_idxs[0]]) + areas[1]*fracs[1]*pdf_exts[1].get_pdf(x, *params[self.par_idxs[1]]) 

            return sig, probs


    def cdf_ext(self, x, *params):
        cdf_exts = self.dists 
        params = np.array(params)

        fracs, areas = get_areas_fracs(params, self.area_frac_idxs, self.frac_flag, self.area_flag, self.one_area_flag)

        if self.components:
            return areas[0]*fracs[0]*cdf_exts[0].get_cdf(x, *params[self.par_idxs[0]]) , areas[1]*fracs[1]*cdf_exts[1].get_cdf(x, *params[self.par_idxs[1]])

        else:
            # This is faster than list comprehension
            probs = areas[0]*fracs[0]*cdf_exts[0].get_cdf(x, *params[self.par_idxs[0]]) + areas[1]*fracs[1]*cdf_exts[1].get_cdf(x, *params[self.par_idxs[1]])
            return probs

    def required_args(self):
        return self.req_args


    def draw_pdf(self, x: np.ndarray, *params) -> None:
        plt.plot(x, self.get_pdf(x, *params))
    
    def draw_cdf(self, x: np.ndarray, *params) -> None:
        plt.plot(x, self.get_cdf(x, *params))


    # Create some convenience functions for fitting 

    def get_mu(self, pars: np.ndarray, cov:np.ndarray = None, errors:np.ndarray = None) -> tuple:
        r"""
        Get the mu value from the output of a fit quickly 

        Parameters 
        ----------
        pars 
            Array of fit parameters
        cov 
            Array of covariances
        errors 
            Array of erros 

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
            return pars[sigma_idx]*2*np.sqrt(2*np.log(2))
        else:
            return pars[sigma_idx]*2*np.sqrt(2*np.log(2)), np.sqrt(cov[sigma_idx][sigma_idx])*2*np.sqrt(2*np.log(2))


    def get_total_events(self, pars: np.ndarray, cov: np.ndarray = None, errors: np.ndarray =None) -> tuple:
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
            Array of erros 

        Returns 
        -------
        n_total, error 
            the total number of events in a spectrum and its associated error 
        """
        
        req_args = np.array(self.required_args())
        n_sig_idx = np.where(req_args == "n_sig")[0][0]
        n_bkg_idx = np.where(req_args == "n_bkg")[0][0]


        if errors is not None:
            return pars[n_sig_idx]+pars[n_bkg_idx], np.sqrt(errors[n_sig_idx]**2 + errors[n_bkg_idx]**2)
        elif cov is not None:
            return pars[n_sig_idx]+pars[n_bkg_idx], np.sqrt(cov[n_sig_idx][n_sig_idx]**2 + cov[n_bkg_idx][n_bkg_idx]**2)
        else:
            return pars[n_sig_idx]+pars[n_bkg_idx]