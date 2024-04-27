"""
Subclasses of scipy.stat's rv_continuous, allows for freezing of pygama numba-fied distributions for faster functions
Example:

>>> moyal = moyal_gen(name='moyal')
>>> moyal = moyal(1, 2)
>>> moyal.get_pdf([-1,2,3]) # a direct call to the faster numbafied method
>>> moyal.pdf([-1,2,3]) # a call to the slower scipy method, allows access to other scipy methods like .rvs
>>> moyal.rvs(100) # Can access scipy methods!

NOTE: the dist.pdf method is a slow scipy method, and the dist.get_pdf method is fast
"""

import numpy as np
from scipy.stats._distn_infrastructure import rv_continuous, rv_frozen


class NumbaFrozen(rv_frozen):
    r"""
    Essentially, an overloading of the definition of rv_continuous_frozen so that our pygama class instantiations have
    access to both the slower scipy methods, as well as the faster pygama methods (like get_pdf)
    """

    def pdf(self, x: np.ndarray) -> np.ndarray:
        r"""
        Overloading of scipy's pdf function, this is a slow function
        """

        return self.dist.pdf(x, *self.args, **self.kwds)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        r"""
        Overloading of scipy's logpdf function, this is a slow function
        """

        return self.dist.logpdf(x, *self.args, **self.kwds)

    def get_pdf(self, x: np.ndarray) -> np.ndarray:
        r"""
        Direct access to numba-fied pdfs, a fast function

        Returns
        -------
        support normalized pdf
        """

        return self.dist.get_pdf(np.array(x), *self.args, **self.kwds)

    def get_cdf(self, x: np.ndarray) -> np.ndarray:
        r"""
        Direct access to numba-fied cdfs, a fast function

        Returns
        -------
        support normalized cdf
        """

        return self.dist.get_cdf(np.array(x), *self.args, **self.kwds)

    def pdf_ext(
        self, x: np.ndarray, x_lo: float, x_hi: float, area: float
    ) -> tuple[float, np.ndarray]:
        r"""
        Direct access to numba-fied extended pdfs, a fast function

        Returns
        -------
        integral, scaled support normalized pdf
        """

        return self.dist.pdf_ext(np.array(x), x_lo, x_hi, area, *self.args, **self.kwds)

    def cdf_ext(self, x: np.ndarray, area: float) -> np.ndarray:
        r"""
        Direct access to numba-fied extended pdfs, a fast function

        Returns
        -------
        scaled support normalized cdf
        """

        return self.dist.cdf_ext(np.array(x), area, *self.args, **self.kwds)

    def pdf_norm(self, x, x_lo: float, x_hi: float):
        r"""
        Direct access to numba-fied pdfs, a fast function. Normalized on a fit range

        Returns
        -------
        fit-range normalized pdf
        """
        return self.dist.pdf_norm(x, x_lo, x_hi, *self.args, **self.kwds)

    def cdf_norm(self, x, x_lo: float, x_hi: float):
        r"""
        Direct access to numba-fied cdfs, a fast function. Normalized on a fit range

        Returns
        -------
        fit-range normalized cdf
        """
        return self.dist.cdf_norm(x, x_lo, x_hi, *self.args, **self.kwds)

    def required_args(self) -> tuple:
        r"""
        Allow access to the required args of frozen distribution

        Returns
        -------
        Required shape, location, and scale arguments
        """

        return self.dist.required_args()


class PygamaContinuous(rv_continuous):
    r"""
    Subclass rv_continuous, and modify the instantiation so that we call an overloaded
    version of rv_continuous_frozen that has direct access to pygama numbafied functions
    """

    def _pdf_norm(self, x, x_lo, x_hi, *args, **kwds):
        r"""
        Normalize a pdf on a subset of its support, typically over a fit-range.

        Parameters
        ----------
        x
            The data
        x_lo
            The lower range to normalize on
        x_hi
            The upper range to normalize on
        args
            The shape and location and scale parameters of a specific distribution

        Returns
        -------
        pdf_norm
            The pdf that is normalized on a smaller range

        Notes
        -----
        If upper_range and lower_range are both :func:`np.inf`, then the function automatically takes x_lo =:func:`np.amin(x)`, x_hi=:func:`np.amax(x)`
        We also need to overload this in every subclass because we want functions to be able to introspect the shape and location/scale names
        For distributions that are only defined on a limited range, like with lower_range, upper_range, we don't need to call these, instead just call the normal pdf.
        """

        norm = np.diff(self.get_cdf(np.array([x_lo, x_hi]), *args))

        if norm == 0:
            return np.full_like(x, np.inf)
        else:
            return self.get_pdf(x, *args) / norm

    def _cdf_norm(self, x, x_lo, x_hi, *args, **kwds):
        r"""
        Derive a cdf from a pdf that is normalized on a subset of its support, typically over a fit-range.

        Parameters
        ----------
        x
            The data
        x_lo
            The lower range to normalize on
        x_hi
            The upper range to normalize on
        args
            The shape and location and scale parameters of a specific distribution

        Returns
        -------
        pdf_norm
            The pdf that is normalized on a smaller range

        Notes
        -----
        If upper_range and lower_range are both :func:`np.inf`, then the function automatically takes x_lo =:func:`np.amin(x)`, x_hi=:func:`np.amax(x)`
        We also need to overload this in every subclass because we want functions to be able to introspect the shape and location/scale names.
        For distributions that are only defined on a limited range, like with lower_range, upper_range, we don't need to call these, instead just call the normal pdf.
        """

        norm = np.diff(self.get_cdf(np.array([x_lo, x_hi]), *args))

        if norm == 0:
            return np.full_like(x, np.inf)
        else:
            return (self.get_cdf(x, *args)) / norm

    def __call__(self, *args, **kwds):
        return NumbaFrozen(self, *args, **kwds)
