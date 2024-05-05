r"""
Contains a list of distribution functions, all implemented using Numba's
:func:`numba.jit` to take advantage of the just-in-time speed boost.
"""

# nopycln: file

from pygama.math.functions.crystal_ball import crystal_ball  # noqa: F401
from pygama.math.functions.crystal_ball import (  # noqa: F401
    nb_crystal_ball_cdf,
    nb_crystal_ball_pdf,
)
from pygama.math.functions.error_function import nb_erf, nb_erfc  # noqa: F401
from pygama.math.functions.exgauss import exgauss  # noqa: F401
from pygama.math.functions.exgauss import (  # noqa: F401
    nb_exgauss_cdf,
    nb_exgauss_pdf,
    nb_gauss_tail_approx,
)
from pygama.math.functions.exponential import exponential  # noqa: F401
from pygama.math.functions.gauss import (  # noqa: F401
    gaussian,
    nb_gauss,
    nb_gauss_cdf,
    nb_gauss_pdf,
    nb_gauss_scaled_pdf,
)
from pygama.math.functions.gauss_on_exgauss import gauss_on_exgauss  # noqa: F401
from pygama.math.functions.gauss_on_linear import gauss_on_linear  # noqa: F401
from pygama.math.functions.gauss_on_step import gauss_on_step  # noqa: F401
from pygama.math.functions.gauss_on_uniform import gauss_on_uniform  # noqa: F401
from pygama.math.functions.hpge_peak import hpge_peak  # noqa: F401
from pygama.math.functions.linear import linear  # noqa: F401
from pygama.math.functions.moyal import moyal  # noqa: F401
from pygama.math.functions.polynomial import nb_poly  # noqa: F401
from pygama.math.functions.step import (  # noqa: F401
    nb_step_cdf,
    nb_step_pdf,
    nb_unnorm_step_pdf,
    step,
)
from pygama.math.functions.triple_gauss_on_double_step import (  # noqa: F401
    triple_gauss_on_double_step,
)
from pygama.math.functions.uniform import uniform  # noqa: F401
