r"""
Statistical distributions for the Pygama package.

Each distribution requires the definition of four vectorized numbafied functions that take an array as an input:

1. :func:`nb_dist_pdf(x, mu, sigma, shape)`
Returns the PDF normalized on the support

2. :func:`nb_dist_cdf(x, mu, sigma, shape)`
Returns the CDF derived from the PDF that is normalized on the support

3. :func:`nb_dist_scaled_pdf(x, area, mu, sigma, shape)`
Returns area*nb_dist_prd

4. :func:`nb_dist_scaled_cdf(x, area, mu, sigma, shape)`
Returns area*nb_dist_cdf

NOTE: The order of the arguments of these functions follows the ordering convention from left to right (for whichever are present): x, x_lo, x_hi, area, mu, sigma, shapes

Then these four functions are packaged into a class that subclasses our own PygamaContinuous. This distribution class then has 9 required methods.

1. :func:`_pdf(x, mu, sigma, shape)`
An overloading of the scipy rv_continuous base class' method, this is slow, but it enables access to other scipy rv_continuous methods like _rvs

2. :func:`_cdf(x, mu, sigma, shape)`
An overloading of the scipy rv_continuous base class' method, this is slow, but it enables access to other scipy rv_continuous methods like _logcdf

3. :func:`get_pdf(x, mu, sigma, shape)`
A direct call to nb_dist_pdf, it is very quick. This PDF is normalized on its support

4. :func:`get_cdf(x, mu, sigma, shape)`
A direct call to nb_dist_cdf, it is very quick. This CDF comes from the PDF that is normalized on the support

5. :func:`norm_pdf(x, x_lo, x_hi, mu, sigma, shape)`
Returns the get_pdf, normalized to unity on the fit range (normalized on [x_lo, x_hi]). Needed for unbinned fits

6. :func:`norm_cdf(x, x_lo, x_hi, mu, sigma, shape)`
Returns get_cdf, normalized on the fit range (normalized on [x_lo, x_hi]). Needed for binned fits

7. :func:`pdf_ext(x, x_lo, x_hi, area, shape, mu, sigma)`
Returns both the integral of nb_dist_scaled_pdf, as well as nb_dist_scaled_pdf itself. Needed for extended unbinned fits

8. :func:`cdf_ext(x, area, shape, mu, sigma)`
Returns a direct call to nb_dist_scaled_cdf itself. Needed for extended binned fits.

9. :func:`required_args`
A tuple of the required shape, mu, and sigma parameters

NOTE: the order of the arguments to these functions must follow the convention for whichever are present: x, x_lo, x_hi,, area, mu, sigma, shape

PygamaContinuous subclasses scipy's rv_continuous and overloads the class instantiation so that is calls numba_frozen. numba_frozen is
a subclass of rv_continuous, and acts exactly like scipy's rv_continuous_frozen, except that this class also has methods for
pygama specific get_pdf, get_cdf, pdf_ext, and cdf_ext. This subclassing is necessary so that we can instantiate frozen distributions that
have access to our required pygama class methods.

In addition, the class SumDists subclasses rv_continuous. This is so that distributions created out of the sum of distributions also have
access to scipy rv_continuous methods, such as random sampling. SumDists also has convenience functions such as draw_pdf, which plots the pdf,
and required_args which returns a list of the required arguments for the sum of the distribution.
"""
