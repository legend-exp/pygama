r"""
Contains a list of distribution functions, all implemented using Numba's
:func:`numba.jit` to take advantage of the just-in-time speed boost. 
"""

from pygama.math.functions.crystal_ball import crystal_ball, nb_crystal_ball_cdf, nb_crystal_ball_pdf
from pygama.math.functions.gauss import (
    gaussian,
    nb_gauss,
    nb_gauss_cdf,
    nb_gauss_pdf,
    nb_gauss_scaled_pdf
)
from pygama.math.functions.gauss_on_linear import gauss_on_linear
from pygama.math.functions.gauss_on_uniform import gauss_on_uniform
from pygama.math.functions.gauss_on_step import gauss_on_step
from pygama.math.functions.exgauss import (
    exgauss,
    nb_exgauss_pdf,
    nb_gauss_tail_approx,
    nb_exgauss_cdf
)
from pygama.math.functions.gauss_on_exgauss import gauss_on_exgauss
from pygama.math.functions.hpge_peak import hpge_peak
from pygama.math.functions.polynomial import nb_poly
from pygama.math.functions.step import step, nb_step_cdf, nb_step_pdf, nb_unnorm_step_pdf
from pygama.math.functions.triple_gauss_on_double_step import triple_gauss_on_double_step
from pygama.math.functions.moyal import moyal
from pygama.math.functions.uniform import uniform
from pygama.math.functions.linear import linear
from pygama.math.functions.exponential import exponential

