r"""
A class that creates the sum of distributions, with methods for scipy computed :func:`pdfs` and :func:`cdfs`, as well as fast :func:`get_pdfs`, and :func:`pdf_ext` 

.. code-block:: python

    mu1, mu2, sigma = range(3)
    moyal_add = sum_dists(moyal, [mu1, sigma], moyal, [mu2, sigma], areas = [0.75, 0.25]) # create two moyals that share a sigma, 
    x = np.arange(-10,10)
    pars = np.array([1,2,2]) # corresponds to mu1 = 1, mu2 = 2, sigma = 2
    moyal_add.pdf(x, pars)
    moyal_add.draw_pdf(x, pars)
    moyal_add.get_req_args()


"""

import numba as nb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats._distn_infrastructure import rv_continuous, rv_frozen
from pygama.math.functions.pygama_continuous import pygama_continuous

from scipy._lib._util import getfullargspec_no_self as _getfullargspec

def get_idx(idx_array, frac_flag): 
    r"""
    Create separate arrays of parameter indices for the shape parameters, the area, the fracs, and the total area

    This is done because the user can pass a combination of areas and fracs during a method call, or can send them in
    during class initialization. So we need to know what array the actual values of the fracs and areas will be in.

    This will return None for an array index whose values are not located in the parameter array passed to a method call
    """
    
    if frac_flag is None:
        shape_par_idx = idx_array
        area_idx = None
        frac_idx = None
        total_area_idx = None
    
    elif frac_flag == "fracs":
        shape_par_idx = []
        frac_idx = []
        total_area_idx = []
        for i in range(len(idx_array)-1):
            shape_par_idx.append(np.array(idx_array[i][:-1]))
            frac_idx.append(np.array(idx_array[i][-1]))
        # The last array contains an extra element at the end, the id of the total area
        # todo: put a check in here so that we make sure that a total area was passed

        shape_par_idx.append(np.array(idx_array[-1][:-2]))
        frac_idx.append(np.array(idx_array[-1][-2]))
        total_area_idx.append(np.array(idx_array[-1][-1]))
        area_idx = None
        # put check that we for sure did get a total_area element! 
    
    elif frac_flag == "areas":
        shape_par_idx = []
        area_idx = []
        for i in range(len(idx_array)):
            shape_par_idx.append(np.array(idx_array[i][:-1]))
            area_idx.append(np.array(idx_array[i][-1]))
        frac_idx = None
        total_area_idx = None
    
    else:
        shape_par_idx = []
        area_idx = []
        frac_idx = []
        total_area_idx = []
        for i in range(len(idx_array)-1):
            shape_par_idx.append(np.array(idx_array[i][:-2]))
            area_idx.append(np.array(idx_array[i][-2]))
            frac_idx.append(np.array(idx_array[i][-1]))
        # The last array contains an extra element at the end, the id of the total area
        shape_par_idx.append(np.array(idx_array[-1][:-3]))
        area_idx.append(np.array(idx_array[-1][-3]))
        frac_idx.append(np.array(idx_array[-1][-2]))
        total_area_idx.append(np.array(idx_array[-1][-1]))
    
    return shape_par_idx, area_idx, frac_idx, total_area_idx


def _precompute_shape_par_idx(shape_par_idx, dists):
    r"""
    Check that each distribution has received the correct number of shape and location parameters

    Parameters
    ----------
    shape_par_idx 
        An array of arrays, each containing the indices of the eventual parameters the function call will take
        from a single parameter array

    dists 
        An array of pygama_continuous distributions, each must have the required_args method! 

    Returns
    -------
    shape_par_idx
        Returned only if every distribution has the correct number of required arguments 
    """ 

    if len(shape_par_idx) != len(dists):
        raise ValueError("Number of distributions does not match number of parameter arrays")
        
    for i in range(len(shape_par_idx)):
        if len(shape_par_idx[i]) != len(dists[i].required_args()):
            raise ValueError(f"distribution {dists[i]} at index {i} does not have the required number of shape\
            and location parameters, it should take f{dists[i].required_args()} as input arguments")
        else:
            pass
    return np.array(shape_par_idx, dtype = 'object')

def _precompute_fracs(fracs, dists):
    r"""
    When initializing a summed dist, check the number of fracs provided.
    If len(fracs) = len(dists)-1, then the remaining frac must be 1-sum(fracs)
    If len(fracs) = len(dists), return the fracs as given
    Otherwise, raise an error. 
    """

    if len(dists) < 2:
        raise ValueError("Cannot sum fewer than 2 distributions.")
        
    if len(fracs) == len(dists)-1 : 
        remaining_frac = 1 - np.sum(fracs)
        return fracs + [remaining_frac]
    
    if len(fracs) == len(dists):
        return fracs
    
    else: 
        raise ValueError("Not enough fractions supplied.")
        
        

# We cannot precompute the support, because the support of some dists depends on the actual value
# of the shape parameter, it is just better to compute and check the support and check it in one step in the function call

        
# Overload the scipy parse_arg_template to allow for different ranges for uniform sampling in the rvs method 
parse_arg_template = """
def _parse_args(self, %(shape_arg_str)s %(locscale_in)s):
    return (%(shape_arg_str)s), %(locscale_out)s
def _parse_args_rvs(self, %(shape_arg_str)s %(locscale_in)s, size=None, low=0.0, high=1.0):
    return self._argcheck_rvs(%(shape_arg_str)s %(locscale_out)s, size=size, low=low, high=high)
def _parse_args_stats(self, %(shape_arg_str)s %(locscale_in)s, moments='mv'):
    return (%(shape_arg_str)s), %(locscale_out)s, moments
"""
        
class sum_dists(rv_continuous):
    r"""
    Initialize and rv_continuous method so that we gain access to computable methods. 
    Also precompute the fracs of the linear combination, as well as the support of the
    sum of the distributions. 

    The correct way to initialize is sum_dists(d1, p1, d2, p2, ..., dn, pn, frac_flag)
    Where ds are distributions and ps are parameter index arrays
    There are some checks to ensure the input is correct, but it is up to the user
    to use it correctly...

    dists are the unfrozen distributions

    idx_array is an array of arrays, each array contains the indicies that each distribution should 
    pass to the actual parameters array sent to the actual function call

    Each par array can contain [shape, mu, sigma, area, frac] with area and frac being optional, and
    depend on the flag sent to the constructor

    4 options:

    1. flag = "areas", areas passed as fit variables, global fracs optional (but can be passed by kwarg, default all to 1)
    2. flag = "fracs", fracs and total area passed as fit variables, global areas optional (but can be passed by kwarg, default all to 1)
        by default, the total area is just a parameter of the last distribution...
    3. flag = "both", fracs and areas passed as fit variables
    4. flag = None, no areas or fracs passed as fit variables, all default to 1
    """
    def __init__(self, *args, **kwargs): 

        if (len(args)%2) != 0: 
            raise ValueError("Incorrect number of distributions and parameter arrays given")
            
        # Read in the args
        dists = args[::2] # the even arguments, d1, d2,
        idx_array = np.array(args[1::2], dtype=object) # the odd arguments, containing arrays of indices

        # Read in the kwargs
        self.components = kwargs.pop('components', False)
        self.areas = kwargs.pop('areas', np.ones(len(dists)))
        self.fracs = kwargs.pop('fracs', np.ones(len(dists)))
        self.frac_flag = kwargs.pop('frac_flag', None)
        self.total_area = kwargs.pop('total_area', 1)
        
        # Check that the dists are in fact distributions
        for i in range(len(dists)):
            if not isinstance(dists[i], pygama_continuous):
                raise ValueError(f"Distribution at index {i} has value {dists[i]},\
                and is an array and not a pygama_continuous distribution")
                
        # Now, create the arrays of indices corresponding to the pars, and whichever area and/or fracs are present
        shape_par_idx, area_idx, frac_idx, total_area_idx = get_idx(idx_array, self.frac_flag)
        
        fracs = self.fracs
        
        self.dists = dists
        self.fracs = _precompute_fracs(fracs, dists)
        self.shape_par_idx = _precompute_shape_par_idx(shape_par_idx, dists)
        self.area_idx = area_idx
        self.frac_idx = frac_idx
        self.total_area_idx = total_area_idx

        self.shapes = None
        
        super().__init__(self)
        
    
    # overload a couple of scipy methods to allow for an array of shape arguments instead of a tuple
    # this is needed for when scipy calls the public pdf method, before it calls the private _pdf
    def _argcheck(self, *args):
        r"""
        Default check for correct values on args and keywords.
        Returns condition array of 1's where arguments are correct and
        0's where they are not.

        Pygama needs to overload this to extend it to arrays of arguments, 
        based on how the sum_dists class is built
        """

        cond = 1
        for arg in args:
            cond = np.logical_and(cond, (np.asarray(arg) > 0))
        if isinstance(cond, np.ndarray):
            return cond.all() # overload the scipy definition, check that we're good for all argument values in the array
        else:
            return cond
        
    # overload a couple of scipy methods to allow for an array of shape arguments instead of a tuple
    # this is needed to create good random variables
    def _argcheck_rvs(self, *args, **kwargs):
        # Handle broadcasting and size validation of the rvs method.
        # Subclasses should not have to override this method.
        # The rule is that if `size` is not None, then `size` gives the
        # shape of the result (integer values of `size` are treated as
        # tuples with length 1; i.e. `size=3` is the same as `size=(3,)`.)
        #
        # `args` is expected to contain the shape parameters (if any), the
        # location and the scale in a flat tuple (e.g. if there are two
        # shape parameters `a` and `b`, `args` will be `(a, b, loc, scale)`).
        # The only keyword argument expected is 'size'.
        size = kwargs.get('size', None)
        low = kwargs.get('low', 0.0)
        high = kwargs.get('high', 1.0)

        args = list(args)
        params = args[0]
        result = []
        for element in args:
            if isinstance(element, np.ndarray):
                result.extend(element)
            else:
                result.append(element)

        args = result

        all_bcast = np.broadcast_arrays(*args)
    


        def squeeze_left(a):
            while a.ndim > 0 and a.shape[0] == 1:
                a = a[0]
            return a

        # Eliminate trivial leading dimensions.  In the convention
        # used by numpy's random variate generators, trivial leading
        # dimensions are effectively ignored.  In other words, when `size`
        # is given, trivial leading dimensions of the broadcast parameters
        # in excess of the number of dimensions  in size are ignored, e.g.
        #   >>> np.random.normal([[1, 3, 5]], [[[[0.01]]]], size=3)
        #   array([ 1.00104267,  3.00422496,  4.99799278])
        # If `size` is not given, the exact broadcast shape is preserved:
        #   >>> np.random.normal([[1, 3, 5]], [[[[0.01]]]])
        #   array([[[[ 1.00862899,  3.00061431,  4.99867122]]]])
        #
        all_bcast = [squeeze_left(a) for a in all_bcast]
        bcast_shape = all_bcast[0].shape
        bcast_ndim = all_bcast[0].ndim

        if size is None:
            size_ = bcast_shape
        else:
            size_ = tuple(np.atleast_1d(size))

        # Check compatibility of size_ with the broadcast shape of all
        # the parameters.  This check is intended to be consistent with
        # how the numpy random variate generators (e.g. np.random.normal,
        # np.random.beta) handle their arguments.   The rule is that, if size
        # is given, it determines the shape of the output.  Broadcasting
        # can't change the output size.

        # This is the standard broadcasting convention of extending the
        # shape with fewer dimensions with enough dimensions of length 1
        # so that the two shapes have the same number of dimensions.
        ndiff = bcast_ndim - len(size_)
        if ndiff < 0:
            bcast_shape = (1,)*(-ndiff) + bcast_shape
        elif ndiff > 0:
            size_ = (1,)*ndiff + size_

        # This compatibility test is not standard.  In "regular" broadcasting,
        # two shapes are compatible if for each dimension, the lengths are the
        # same or one of the lengths is 1.  Here, the length of a dimension in
        # size_ must not be less than the corresponding length in bcast_shape.
        ok = all([bcdim == 1 or bcdim == szdim
                  for (bcdim, szdim) in zip(bcast_shape, size_)])
        if not ok:
            raise ValueError("size does not match the broadcast shape of "
                             "the parameters. %s, %s, %s" % (size, size_,
                                                             bcast_shape))

        param_bcast = all_bcast[:-2]
        loc_bcast = all_bcast[-2]
        scale_bcast = all_bcast[-1]
        
        param_bcast = np.array([params])
        

        return param_bcast, loc_bcast, scale_bcast, size_, low, high
    
    # overload scipy's vectorization, we don't need to vectorize on the parameter array fed to ppf
    # In fact, we only want vectorization over input X, the params should be the same for each
    def _attach_methods(self):
        r"""
        Attaches dynamically created methods to the rv_continuous instance.
        """

        # _attach_methods is responsible for calling _attach_argparser_methods
        self._attach_argparser_methods()

        # nin correction
        self._ppfvec = np.vectorize(self._ppf_single, otypes='d', excluded = [3]) # exclude vectorization over params in position 3
        self._ppfvec.nin = self.numargs + 1
        self.vecentropy = np.vectorize(self._entropy, otypes='d', excluded = [1])
        self._cdfvec = np.vectorize(self._cdf_single, otypes='d', excluded = [1])
        self._cdfvec.nin = self.numargs + 1

        if self.moment_type == 0:
            self.generic_moment = np.vectorize(self._mom0_sc, otypes='d', excluded = [1])
        else:
            self.generic_moment = np.vectorize(self._mom1_sc, otypes='d', excluded = [1])
        # Because of the *args argument of _mom0_sc, vectorize cannot count the
        # number of arguments correctly.
        self.generic_moment.nin = self.numargs + 1

            
    # Need this to be called from the super init, because the super init needs the newly defined parse_arg_template   
    def _construct_argparser(self, meths_to_inspect, locscale_in, locscale_out):
        r"""
        Construct the parser string for the shape arguments.
        This method should be called in __init__ of a class for each
        distribution. It creates the `_parse_arg_template` attribute that is
        then used by `_attach_argparser_methods` to dynamically create and
        attach the `_parse_args`, `_parse_args_stats`, `_parse_args_rvs`
        methods to the instance.
        If self.shapes is a non-empty string, interprets it as a
        comma-separated list of shape parameters.
        Otherwise inspects the call signatures of `meths_to_inspect`
        and constructs the argument-parsing functions from these.
        In this case also sets `shapes` and `numargs`.
        """

        if self.shapes:
            # sanitize the user-supplied shapes
            if not isinstance(self.shapes, str):
                raise TypeError('shapes must be a string.')

            shapes = self.shapes.replace(',', ' ').split()

            for field in shapes:
                if keyword.iskeyword(field):
                    raise SyntaxError('keywords cannot be used as shapes.')
                if not re.match('^[_a-zA-Z][_a-zA-Z0-9]*$', field):
                    raise SyntaxError(
                        'shapes must be valid python identifiers')
        else:
            # find out the call signatures (_pdf, _cdf etc), deduce shape
            # arguments. Generic methods only have 'self, x', any further args
            # are shapes.
            shapes_list = []
            for meth in meths_to_inspect:
                shapes_args = _getfullargspec(meth)  # NB does not contain self
                args = shapes_args.args[1:]       # peel off 'x', too

                if args:
                    shapes_list.append(args)

                    # *args or **kwargs are not allowed w/automatic shapes
                    if shapes_args.varargs is not None:
                        raise TypeError(
                            '*args are not allowed w/out explicit shapes')
                    if shapes_args.varkw is not None:
                        raise TypeError(
                            '**kwds are not allowed w/out explicit shapes')
                    if shapes_args.kwonlyargs:
                        raise TypeError(
                            'kwonly args are not allowed w/out explicit shapes')
                    if shapes_args.defaults is not None:
                        raise TypeError('defaults are not allowed for shapes')

            if shapes_list:
                shapes = shapes_list[0]
                
                # make sure the signatures are consistent
                for item in shapes_list:
                    if item != shapes:
                        raise TypeError('Shape arguments are inconsistent.')
            else:
                shapes = []

        # have the arguments, construct the method from template
        shapes_str = ', '.join(shapes) + ', ' if shapes else ''  # NB: not None
        dct = dict(shape_arg_str=shapes_str,
                   locscale_in=locscale_in,
                   locscale_out=locscale_out,
                   )

        # this string is used by _attach_argparser_methods
        self._parse_arg_template = parse_arg_template % dct
        

        self.shapes = ', '.join(shapes) if shapes else None
        if not hasattr(self, 'numargs'):
            # allows more general subclassing with *args
            self.numargs = len(shapes)
    
    # Need to overload ppf to allow for scaling so that rvs will work as expected 
    def _ppf(self, q, shift, scale, *args):
        return self._ppfvec(q, shift, scale, *args)
    
    def _ppf_to_solve(self, x, q, shift, scale, *args):
        return scale*(self.get_cdf(*(np.array([x]), )+args)-shift)-q
    
    # We need to even subclass _rvs because the default is to uniform sample over [0,1]...
    def _rvs(self, *args, size=None, random_state=None, low = 0.0, high = 1.0):
        # This method must handle size being a tuple, and it must
        # properly broadcast *args and size.  size might be
        # an empty tuple, which means a scalar random variate is to be
        # generated.

        # Use basic inverse cdf algorithm for RV generation as default.
        U = random_state.uniform(low=low, high=high, size=size) # allow the user to overload the lows and highs
        
        # because sums of cdfs are not normalized, we need to shift the distribution accordingly:
        shift = self.get_cdf(np.array([-1000]), *args)
        scale = 1/(self.get_cdf(np.array([1000]), *args)-shift)
        Y = self._ppf(U, shift, scale, *args)
        return Y
   
    # need to overload this function to allow access to the lows and highs
    def rvs(self, *args, **kwds):
        r"""
        Random variates of given type.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).
        scale : array_like, optional
            Scale parameter (default=1).
        size : int or tuple of ints, optional
            Defining number of random variates (default is 1).
        low : float
            The lower bound to sample a uniform distribution
        high : float
            The upper bound to sample a uniform distribution
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional
            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance
            then that instance is used.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given `size`.
        """

        discrete = kwds.pop('discrete', None)
        rndm = kwds.pop('random_state', None)
        args, loc, scale, size, low, high = self._parse_args_rvs(*args, **kwds) # unpack optional low and high kwds
        cond = np.logical_and(self._argcheck(*args), (scale >= 0))
        ### don't need this check anymore
        
#         if not np.all(cond):
#             message = ("Domain error in arguments. The `scale` parameter must "
#                        "be positive for all distributions, and many "
#                        "distributions have restrictions on shape parameters. "
#                        f"Please see the `scipy.stats.{self.name}` "
#                        "documentation for details.")
#             raise ValueError(message)

        if np.all(scale == 0):

            return loc*ones(size, 'd')

        # extra gymnastics needed for a custom random_state
        if rndm is not None:
            random_state_saved = self._random_state
            random_state = check_random_state(rndm)
        else:
            random_state = self._random_state

        vals = self._rvs(*args, size=size, random_state=random_state, low=low, high=high)

        vals = vals * scale + loc

        # do not forget to restore the _random_state
        if rndm is not None:
            self._random_state = random_state_saved

        # Cast to int if discrete
        if discrete and not isinstance(self, rv_sample):
            if size == ():
                vals = int(vals)
            else:
                vals = vals.astype(np.int64)

        return vals


    def _pdf(self, x, params): 
        
        # params is an array containing the shape params, as well as potentially areas and fracs

        pdfs = self.dists 
        fracs = self.fracs 
        areas = self.areas 
        total_area = self.total_area
        
        shape_par_idx = self.shape_par_idx
        area_idx = self.area_idx
        frac_idx = self.frac_idx
        total_area_idx = self.total_area_idx
        

        shape_pars, cum_len, areas, fracs, total_area = self.link_pars(shape_par_idx, area_idx, frac_idx, total_area_idx, params, areas, fracs, total_area)
        
        probs = [pdfs[i].pdf(x, *shape_pars[cum_len[i]:cum_len[i+1]]) for i in range(len(pdfs))]
        
        prefactor = total_area*fracs*areas 
        
        if self.components:
            probs = (probs.T*prefactor).T
            return probs
        else:
            probs = prefactor@probs
            return probs

    
    def _cdf(self, x, params):
        
        cdfs = self.dists 
        fracs = self.fracs 
        areas = self.areas 
        total_area = self.total_area
        
        shape_par_idx = self.shape_par_idx
        area_idx = self.area_idx
        frac_idx = self.frac_idx
        total_area_idx = self.total_area_idx
        
        # right now, params COULD contain areas and/or fracs... split it off
        
        shape_pars, cum_len, areas, fracs, total_area = self.link_pars(shape_par_idx, area_idx, frac_idx, total_area_idx, params, areas, fracs, total_area)
        


        probs = [cdfs[i].cdf(x, *shape_pars[cum_len[i]:cum_len[i+1]]) for i in range(len(cdfs))]
        
        prefactor = total_area*fracs*areas 

        if self.components:
            probs = (probs.T*prefactor).T
            return probs
        else:
            probs = prefactor@probs
            return probs
    
    

    def get_pdf(self, x, params):
        pdfs = self.dists 
        fracs = self.fracs 
        areas = self.areas 
        total_area = self.total_area
        
        shape_par_idx = self.shape_par_idx
        area_idx = self.area_idx
        frac_idx = self.frac_idx
        total_area_idx = self.total_area_idx
        
        # right now, params COULD contain areas and/or fracs... split it off
        
        shape_pars, cum_len, areas, fracs, total_area = self.link_pars(shape_par_idx, area_idx, frac_idx, total_area_idx, params, areas, fracs, total_area)
        
        probs = [pdfs[i].get_pdf(x, *shape_pars[cum_len[i]:cum_len[i+1]]) for i in range(len(pdfs))]
        
        prefactor = total_area*fracs*areas 

        if self.components:
            probs = (probs.T*prefactor).T
            return probs
        else:
            probs = prefactor@probs
            return probs
    
    def get_cdf(self, x, params):
        cdfs = self.dists 
        fracs = self.fracs 
        areas = self.areas 
        total_area = self.total_area
        
        shape_par_idx = self.shape_par_idx
        area_idx = self.area_idx
        frac_idx = self.frac_idx
        total_area_idx = self.total_area_idx
        
        # right now, params COULD contain areas and/or fracs... split it off
        
        shape_pars, cum_len, areas, fracs, total_area = self.link_pars(shape_par_idx, area_idx, frac_idx, total_area_idx, params, areas, fracs, total_area)
        
        probs = [cdfs[i].get_cdf(x, *shape_pars[cum_len[i]:cum_len[i+1]]) for i in range(len(cdfs))]
        
        prefactor = total_area*fracs*areas 

        if self.components:
            probs = (probs.T*prefactor).T
            return probs
        else:
            probs = prefactor@probs
            return probs
    
    def pdf_ext(self, x, params):
        r"""
        Extended probability distribution. 

        Notes
        -----
        Params can have x_lo, x_hi as the first two parameters, and the rest of the shape parameters in the rest of the array
        If params is the same length as the required params, the x_lo and x_hi are interpreted to be the max and min of the input array x
        
        """
        pdf_exts = self.dists 
        fracs = self.fracs 
        areas = self.areas 
        total_area = self.total_area
        
        shape_par_idx = self.shape_par_idx
        area_idx = self.area_idx
        frac_idx = self.frac_idx
        total_area_idx = self.total_area_idx

        # We can't pass a separate x_lo, x_hi because then Iminuit wouldn't accept params to be an array as it needs to be... 
        # So instead, just chuck x_lo, x_hi at the start of params, and then we peel them off 
        if len(params) == len(self.get_req_args())+2:
            x_lo = params[0]
            x_hi = params[1]
            params = params[2:]
        else:
            x_lo = np.amin(x)
            x_hi = np.amax(x)

                
        shape_pars, cum_len, areas, fracs, total_area = self.link_pars(shape_par_idx, area_idx, frac_idx, total_area_idx, params, areas, fracs, total_area)
        
        prefactor = total_area*fracs*areas 


        probs = [pdf_exts[i].pdf_ext(x, 1, x_lo, x_hi, *shape_pars[cum_len[i]:cum_len[i+1]]) for i in range(len(pdf_exts))]
        probs = np.array(probs, dtype = object) # this is going to be very slow! Think of a way around this
        # Maybe try this 
        # probs = list(zip(*probs))
        # sigs = np.array(probs[0])
        # probs = np.array(probs[1])
        sigs = probs[:,0]
        probs = probs[:,1]
        
        sigs *= prefactor
        sigs = np.sum(sigs)
        
        
        if self.components:
            probs = (probs.T*prefactor).T
            return sigs, probs # returns tuple(sum of signals, frac*dist, frac*dist, ...)
        else:
            probs = prefactor@probs
            return sigs, probs # returns tuple(sum of signals, frac*dist + ... +frac*dist )

    
    def cdf_ext(self, x, params):
        cdf_exts = self.dists 
        fracs = self.fracs 
        areas = self.areas 
        total_area = self.total_area
        
        shape_par_idx = self.shape_par_idx
        area_idx = self.area_idx
        frac_idx = self.frac_idx
        total_area_idx = self.total_area_idx
        
        shape_pars, cum_len, areas, fracs, total_area = self.link_pars(shape_par_idx, area_idx, frac_idx, total_area_idx, params, areas, fracs, total_area)
        
        prefactor = total_area*fracs*areas
        
        probs = [cdf_exts[i].cdf_ext(x, 1, *shape_pars[cum_len[i]:cum_len[i+1]]) for i in range(len(cdf_exts))]
        
        if self.components:
            probs = (probs.T*prefactor).T
            return probs
        else:
            probs = prefactor@probs
            return probs
        
    def draw_pdf(self, x: np.ndarray, params, **kwargs) -> None:
        plt.plot(x, self.get_pdf(x, params))
    
    def draw_cdf(self, x: np.ndarray, params, **kwargs) -> None:
        plt.plot(x, self.get_cdf(x, params))

    def get_req_args(self):
        r""" 
        This is a default function to get the required args from the distributions in the instance of :class:`sum_dists`.
        This should be overloaded in user defined functions to ensure that :func:`get_mu`,  :func:`get_fwhm`, :func:`get_total_events`
        """
        dists = self.dists 
        args = []
        for dist in dists:
            args.append(dist.required_args())
        return args
    
    
    def link_pars(self, shape_par_idx, area_idx, frac_idx, total_area_idx, params, areas, fracs, total_area ):
        r"""
        Overload this if you want to link specific parameters together! 
        Need to overload by first calling pars, areas, fracs, total_area = super()._link_pars
        
        .. code-block:: python

        def _link_pars(self, shape_par_idx, area_idx, frac_idx, total_area_idx, params, areas, fracs, total_area):
            shape_pars, cum_len, areas, fracs, total_area = super()._link_pars(shape_par_idx, area_idx, frac_idx, total_area_idx, params, areas, fracs, total_area)
            fracs[1]= 1 - frac1
            return shape_pars, cum_len, areas, fracs, total_area
            (mu, sigma, area1, frac1, tau, area2) = range(6)

        gauss_on_exgauss = sum_dists(d1 = gauss, p1 = [mu, sigma, area1, frac1], d2 = exgauss, p2 = [tau, mu, sigma, frac1, area2])
        pars = [1,2,10,0.75, 0.1, 30]
        x = np.arange(-10,10)
        gauss_on_exgauss.pdf(x, pars)


        This computes 
        :math:`10*0.75*gauss(x, mu=1, sigma=2) + 30*(1-0.75)*exgauss(tau = 0.1, mu=1, sigma=2)`
        Which is what we want it to do...
        """
        return self._link_pars(shape_par_idx, area_idx, frac_idx, total_area_idx, params, areas, fracs, total_area) 


    def _link_pars(self, shape_par_idx, area_idx, frac_idx, total_area_idx, params, areas, fracs, total_area):
        
        # shape_par_idx can be a jagged array, so we flatten it and get the cumulative lengths of each sub-array
        shape_pars = params[np.array(np.hstack(shape_par_idx), dtype=int)]
        
        # Create the array of cumulative lengths of the original shape_par_idx jagged array
        cum_len = list([len(x) for x in shape_par_idx])
        cum_len = [0]+cum_len
        cum_len = np.cumsum(cum_len)
        
        # Depending on what frac_flag is passed, the fracs and areas might be in the param array fed to the method,
        # or they might be provided by the user during class initialization
        if (area_idx is None) and (frac_idx is None):
            pass
        
        if (frac_idx is not None) and (area_idx is None):
            fracs = params[np.array(frac_idx, dtype=int)]
            total_area = params[np.array(total_area_idx, dtype=int)] 
            
        if (area_idx is not None) and (frac_idx is None):
            areas = params[np.array(area_idx, dtype=int)]
            
        if (area_idx is not None) and (frac_idx is not None):
            areas = params[np.array(area_idx, dtype=int)] 
            fracs = params[np.array(frac_idx, dtype=int)] 
            total_area = params[np.array(total_area_idx, dtype=int)]
            
        return shape_pars, cum_len, areas, fracs, total_area



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


        req_args = np.array(self.get_req_args())
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

        req_args = np.array(self.get_req_args())
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
        
        req_args = np.array(self.get_req_args())
        n_sig_idx = np.where(req_args == "n_sig")[0][0]
        n_bkg_idx = np.where(req_args == "n_bkg")[0][0]


        if errors is not None:
            return pars[n_sig_idx]+pars[n_bkg_idx], np.sqrt(errors[n_sig_idx]**2 + errors[n_bkg_idx]**2)
        elif cov is not None:
            return pars[n_sig_idx]+pars[n_bkg_idx], np.sqrt(cov[n_sig_idx][n_sig_idx]**2 + cov[n_bkg_idx][n_bkg_idx]**2)
        else:
            return pars[n_sig_idx]+pars[n_bkg_idx]



    # overload the call method so that we can create frozen sum_dists
    def __call__(self, *args, **kwds):
        return sum_dist_frozen(self, *args, **kwds)






class sum_dist_frozen(rv_frozen):
    r"""
    Essentially, an overloading of the definition of rv_continuous_frozen so that our pygama class instantiations have 
    access to both the slower scipy methods, as well as the faster pygama methods (like get_pdf)
    """
    
    def __init__(self, dist, *args, **kwds):
        self.args = args
        self.kwds = kwds
        

        # create a new instance
        # we don't update the ctor params because they don't make sense for a sum_dist object 
        self.dist = dist.__class__()
        

        shapes, _, _ = self.dist._parse_args(*args, **kwds)
        self.a, self.b = self.dist._get_support(*shapes)
    
    def pdf(self, x):
        return self.dist.pdf(x, *self.args, **self.kwds)
    
    def logpdf(self, x):
        return self.dist.logpdf(x, *self.args, **self.kwds)
    
    def get_pdf(self, x):
        return self.dist.get_pdf(np.array(x), *self.args, **self.kwds)
    def get_cdf(self, x):
        return self.dist.get_cdf(np.array(x), *self.args, **self.kwds)
    def pdf_ext(self, x):
        return self.dist.pdf_ext(np.array(x), *self.args, **self.kwds)
    def cdf_ext(self, x):
        return self.dist.cdf_ext(np.array(x), *self.args, **self.kwds)
    
            
    def draw_pdf(self, x: np.ndarray, **kwargs) -> None:
        return self.dist.draw_pdf(x, *self.args, **kwargs)
    
    def draw_cdf(self, x: np.ndarray, **kwargs) -> None:
        return self.dist.draw_cdf(x, *self.args, **kwargs)
    
    def get_req_args(self):
        return self.dist.get_req_args()