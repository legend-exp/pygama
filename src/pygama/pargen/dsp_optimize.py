from __future__ import annotations

import logging
import sys
from collections import namedtuple
from collections.abc import Callable, Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
from dspeed import build_dsp
from dspeed.units import unit_registry as ureg
from lgdo import Table
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from sklearn.utils._testing import ignore_warnings

log = logging.getLogger(__name__)


def run_one_dsp(
    tb_data: Table,
    dsp_config: Mapping,
    db_dict: Mapping | None = None,
    fom_function: Callable | None = None,
    verbosity: int = 0,
    fom_kwargs=None,
) -> float | Table:
    """Run one iteration of DSP on ``tb_data`, based on the DSP specified in ``dsp_config``.

    Optionally returns a value for optimization

    Parameters
    ----------
    tb_data : lh5 Table
        An input table of lh5 data. Typically a selection is made prior to
        sending tb_data to this function: optimization typically doesn't have to
        run over all data
    dsp_config : dict
        Specifies the DSP to be performed for this iteration (see
        build_processing_chain()) and the list of output variables to appear in
        the output table
    db_dict : dict (optional)
        DSP parameters database. See build_processing_chain for formatting info
    fom_function : function or None (optional)
        When given the output lh5 table of this DSP iteration, the
        fom_function must return a scalar figure-of-merit value upon which the
        optimization will be based. Should accept verbosity as a second argument
    verbosity : int (optional)
        verbosity for the processing chain and fom_function calls
    fom_kwargs
        any keyword arguments to pass to the fom

    Returns
    -------
    figure_of_merit : float
        If fom_function is not None, returns figure-of-merit value for the DSP iteration
    tb_out : lh5 Table
        If fom_function is None, returns the output lh5 table for the DSP iteration
    """

    tb_out = build_dsp(tb_data, dsp_config=dsp_config, database=db_dict)
    if fom_function is not None:
        if fom_kwargs is not None:
            return fom_function(tb_out, verbosity, fom_kwargs)
        return fom_function(tb_out, verbosity)
    return tb_out


ParGridDimension = namedtuple("ParGridDimension", "name parameter value_strs")


class ParGrid:
    """Parameter Grid class
    Each ParGrid entry corresponds to a dsp parameter to be varied.
    The ntuples must follow the pattern:
    ( name parameter value_strs) : ( str, str, list of str)
    where name and parameter are the same as 'db.name.parameter' in the processing chain,
    value_strs is the array of strings to set the argument to.
    """

    def __init__(self):
        self.dims = []

    def add_dimension(self, name, parameter, value_strs):
        self.dims.append(ParGridDimension(name, parameter, value_strs))

    def get_n_dimensions(self):
        return len(self.dims)

    def get_n_points_of_dim(self, i):
        return len(self.dims[i].value_strs)

    def get_shape(self):
        shape = ()
        for i in range(self.get_n_dimensions()):
            shape += (self.get_n_points_of_dim(i),)
        return shape

    def get_n_grid_points(self):
        return np.prod(self.get_shape())

    def get_par_meshgrid(self, copy=False, sparse=False):
        """return a meshgrid of parameter values
        Always uses Matrix indexing (natural for par grid) so that
        mg[i1][i2][...] corresponds to index order in self.dims
        Note copy is False by default as opposed to numpy default of True
        """
        axes = []
        for i in range(self.get_n_dimensions()):
            axes.append(self.dims[i].values_strs)
        return np.meshgrid(*axes, copy, sparse, indexing="ij")

    def get_zero_indices(self):
        return np.zeros(self.get_n_dimensions(), dtype=np.uint32)

    def iterate_indices(self, indices):
        """iterate given indices [i1, i2, ...] by one.
        For easier iteration. The convention here is arbitrary, but its the
        order the arrays would be traversed in a series of nested for loops in
        the order appearing in dims (first dimension is first for loop, etc):
        Return False when the grid runs out of indices. Otherwise returns True.
        """
        for dim in reversed(range(self.get_n_dimensions())):
            indices[dim] += 1
            if indices[dim] < self.get_n_points_of_dim(dim):
                return True
            indices[dim] = 0
        return False

    # def check_indices(self, indices):
    #    for iD in reversed(range(self.get_n_dimensions())):
    #        if indices[iD] < self.get_n_points_of_dim(iD): return True
    #        indices[iD] = 0
    #    return False

    def get_data(self, i_dim, i_par):
        name = self.dims[i_dim].name
        parameter = self.dims[i_dim].parameter
        value_str = self.dims[i_dim].value_strs[i_par]
        return name, parameter, value_str

    def print_data(self, indices):
        print_string = f"Grid point at indices {indices}:"
        for i_dim, i_par in enumerate(indices):
            name, parameter, value_str = self.get_data(i_dim, i_par)
            print_string += f"\n {name}.{parameter} = {value_str}"
        return print_string

    def set_dsp_pars(self, db_dict, indices):
        if db_dict is None:
            db_dict = {}
        for i_dim, i_par in enumerate(indices):
            name, parameter, value_str = self.get_data(i_dim, i_par)
            if name not in db_dict.keys():
                db_dict[name] = {parameter: value_str}
            else:
                db_dict[name][parameter] = value_str
        return db_dict


def run_grid(
    tb_data, dsp_config, grid, fom_function, db_dict=None, verbosity=1, **fom_kwargs
):
    """Extract a table of optimization values for a grid of DSP parameters
    The grid argument defines a list of parameters and values over which to run
    the DSP defined in dsp_config on tb_data. At each point, a scalar
    figure-of-merit is extracted.

    Returns a N-dimensional ndarray of figure-of-merit values, where the array
    axes are in the order they appear in grid.

    Parameters
    ----------
    tb_data : lh5 Table
        An input table of lh5 data. Typically a selection is made prior to
        sending tb_data to this function: optimization typically doesn't have to
        run over all data
    dsp_config : dict
        Specifies the DSP to be performed (see build_processing_chain()) and the
        list of output variables to appear in the output table for each grid point
    grid : ParGrid
        See ParGrid class for format
    fom_function : function
        When given the output lh5 table of this DSP iteration, the fom_function
        must return a scalar figure-of-merit. Should accept verbosity as a
        second keyword argument
    db_dict : dict (optional)
        DSP parameters database. See build_processing_chain for formatting info
    verbosity : int (optional)
        verbosity for the processing chain and fom_function calls
    **fom_kwargs
        Any keyword arguments for fom_function

    Returns
    -------
    grid_values : ndarray of floats
        An N-dimensional numpy ndarray whose Mth axis corresponds to the Mth row
        of the grid argument
    """

    grid_values = np.ndarray(shape=grid.get_shape(), dtype="O")
    iii = grid.get_zero_indices()
    log.info("starting grid calculations...")

    while True:
        db_dict = grid.set_dsp_pars(db_dict, iii)
        if verbosity > 1:
            log.debug(dsp_config)
        log.debug(grid.print_data(iii))
        grid_values[tuple(iii)] = run_one_dsp(
            tb_data,
            dsp_config,
            db_dict=db_dict,
            fom_function=fom_function,
            verbosity=verbosity,
            fom_kwargs=fom_kwargs,
        )
        log.debug("value:", grid_values[tuple(iii)])
        if not grid.iterate_indices(iii):
            break
    return grid_values


def run_grid_point(
    tb_data,
    dsp_config,
    grids,
    fom_function,
    iii,
    db_dict=None,
    verbosity=1,
    fom_kwargs=None,
):
    """
    Runs a single grid point for the index specified
    """
    if not isinstance(grids, list):
        grids = [grids]
    for index, grid in zip(iii, grids, strict=False):
        db_dict = grid.set_dsp_pars(db_dict, index)

    log.debug(db_dict)
    for i, grid in enumerate(grids):
        log.debug(grid.print_data(iii[i]))
    tb_out = run_one_dsp(tb_data, dsp_config, db_dict=db_dict, verbosity=verbosity)
    res = np.ndarray(shape=len(grids), dtype="O")
    if fom_function:
        for i in range(len(grids)):
            if fom_kwargs[i] is not None:
                if len(fom_function) > 1:
                    res[i] = fom_function[i](tb_out, verbosity, fom_kwargs[i])
                else:
                    res[i] = fom_function[0](tb_out, verbosity, fom_kwargs[i])
            elif len(fom_function) > 1:
                res[i] = fom_function[i](tb_out, verbosity)
            else:
                res[i] = fom_function[0](tb_out, verbosity)
        log.debug("value:", res)
        out = {"indexes": [tuple(ii) for ii in iii], "results": res}

    else:
        out = {"indexes": [tuple(ii) for ii in iii], "results": tb_out}
    return out


def get_grid_points(grid):
    """
    Generates a list of the indices of all possible grid points
    """
    out = []
    iii = [gri.get_zero_indices() for gri in grid]
    complete = np.full(len(grid), False)
    while True:
        out.append([tuple(ii) for ii in iii])

        for i, gri in enumerate(zip(iii, grid, strict=False)):
            if not gri[1].iterate_indices(gri[0]):
                log.info(f"{i} grid end")
                iii[i] = gri[1].get_zero_indices()
                complete[i] = True
        if all(complete):
            break
    return out


OptimiserDimension = namedtuple(
    "OptimiserDimension", "name parameter min_val max_val round unit"
)


class BayesianOptimizer:
    """
    Bayesian optimization using a Gaussian Process regression.

    This optimizer fits a Gaussian Process (GP) to fit kernel
    to data, takes in a series of init samples for this fit and then calculates
    the next point using the acquisition function specified.

    The GP model is implemented using
    ``sklearn.gaussian_process.GaussianProcessRegressor``.

    Parameters
    ----------
    acq_func
        Acquisition function used to select new sampling points.

        Supported values are:
            - ``"ei"`` : Expected Improvement
            - ``"ucb"`` : Upper Confidence Bound
            - ``"lcb"`` : Lower Confidence Bound
    batch_size
        Number of candidate points proposed at each optimization iteration.
    kernel
        Optional kernel used by the Gaussian Process regressor.
        If ``None``, the default kernel from ``GaussianProcessRegressor`` is used.
    sampling_rate
        Optional sampling rate associated with evaluations. If given as a string it will be converted to a ``pint.Quantity``.
    fom_value
        Column name containing the figure-of-merit values in the input data,
    fom_error
        Column name containing the uncertainty associated with the figure-of-merit values.

    Attributes
    ----------
    gauss_pr
        Gaussian Process model used.
    acq_function
        Acquisition function method corresponding to the specified `acq_func`.
    batch_size
        Batch size used in the Gaussian Process optimization.
    dims
        List of dimensions (parameters) added to the optimizer's search space.
    iters
        Counter for the number of optimization iterations performed.
    best_samples_
        DataFrame storing the best evaluated samples and their acquisition values.
    distances_
        List of distances between sampled points, used for diagnostics.
    x_init
        Initial input points provided via `add_initial_values`.
    y_init
        Observed function values corresponding to `x_init`.
    yerr_init
        Uncertainty associated with `y_init`.
    y_min
        Minimum observed value in `y_init`. Set by `get_first_point`.
    optimal_x
        Input corresponding to `y_min`. Set by `get_first_point`.
    optimal_ei
        Placeholder for expected improvement at `optimal_x`. Set by `get_first_point`.
    current_x
        Current optimization point being evaluated. Updated by `update_db_dict()`.
    current_ei
        Acquisition function value for `current_x`. Updated by `update_db_dict()`.
    current_iter
        Optimization iteration counter. Incremented each time `update_db_dict()` is called.

    Examples
    --------
    Simple 1D optimization:

    .. code-block:: python

        import numpy as np

        # Create optimizer
        optimizer = BayesianOptimizer(acq_func="ei", batch_size=1)
        optimizer.add_dimension("x", parameter="x", min_val=-5, max_val=5)

        # Add initial observations
        optimizer.add_initial_values(
            np.array([[0], [2], [-1]]),
            np.array([10.0, 5.0, 7.0]),
            np.array([0.1, 0.2, 0.1])
        )

        # Update DSP parameters database with the current optimization point
        db_dict = optimizer.update_db_dict({})

        # Simulate new evaluation and update optimizer
        optimizer.update({"x": [1.0], "y_val": 3.5, "y_val_err": 0.1})

        # Plot the result
        fig = optimizer.plot()
    """

    def __init__(
        self,
        acq_func: str,
        batch_size: int,
        kernel: Kernel | None = None,
        sampling_rate: str | pint.Quantity | None = None,
        fom_value: str = "y_val",
        fom_error: str = "y_val_err",
    ) -> None:
        np.random.seed(55)

        self.lambda_param = 0.01
        self.eta_param = 0

        self.dims = []
        self.current_iter = 0

        self.batch_size = batch_size
        self.iters = 0

        if isinstance(sampling_rate, str):
            self.sampling_rate = ureg.Quantity(sampling_rate)
        elif isinstance(sampling_rate, pint.Quantity):
            self.sampling_rate = sampling_rate
        else:
            if sampling_rate is not None:
                raise TypeError("Unknown type for sampling rate")

            self.sampling_rate = None

        self.gauss_pr = GaussianProcessRegressor(kernel=kernel)
        self.best_samples_ = pd.DataFrame(columns=["x", "y", "ei"])
        self.distances_ = []

        # other attrs
        self.x_init = None
        self.y_init = None
        self.yerr_init = None
        self.y_min = None

        # optimisation results
        self.current_x = None
        self.prev_x = None
        self.optimal_x = None
        self.optimal_ei = None

        if acq_func == "ei":
            self.acq_function = self._get_expected_improvement
        elif acq_func == "ucb":
            self.acq_function = self._get_ucb
        elif acq_func == "lcb":
            self.acq_function = self._get_lcb
        else:
            msg = f"Unknown acquisition function: {acq_func}. Supported values are 'ei', 'ucb', and 'lcb'."
            raise ValueError(msg)

        self.fom_value = fom_value
        self.fom_error = fom_error

    def add_dimension(
        self,
        name: str,
        parameter: float,
        min_val: float,
        max_val: float,
        round_to_samples: bool = False,
        unit: str | pint.Quantity | None = None,
    ) -> None:
        """
        Add a new dimension (parameter) to the optimizer's search space.

        Each dimension defines a variable that the Bayesian optimizer will vary
        to explore the objective function.

        Parameters
        ----------
        name
            Name of the dimension.
        parameter
            Initial value or reference for this parameter.
        min_val
            Minimum allowed value for this parameter.
        max_val
            Maximum allowed value for this parameter.
        round_to_samples
            If True, sampled values along this dimension will be rounded to
            multiples of `sampling_rate`. Requires `sampling_rate` to be set
            in the optimizer.
        unit
            Physical unit of the parameter (e.g., "K", "mL"). Converted to
            a `pint.Quantity` internally if provided.
        """

        if round_to_samples is True and self.sampling_rate is None:
            raise ValueError("Must provide sampling rate to round to samples")
        if unit is not None:
            unit = ureg.Quantity(unit)
        self.dims.append(
            OptimiserDimension(
                name, parameter, min_val, max_val, round_to_samples, unit
            )
        )

    def get_n_dimensions(self):
        return len(self.dims)

    def add_initial_values(self, x_init: Any, y_init: Any, yerr_init: Any) -> None:
        """
        Add initial observed data to the optimizer.

        These initial samples are used to fit the Gaussian Process model
        before starting the Bayesian optimization iterations.

        Parameters
        ----------
        x_init
            Initial input points. Shape should match the number of dimensions
            added via `add_dimension`.
        y_init
            Observed function values corresponding to `x_init`.
        yerr_init
            Uncertainty (error) associated with `y_init`.

        Notes
        -----
        The lengths of `x_init`, `y_init`, and `yerr_init` must all match.
        """
        self.x_init = x_init
        self.y_init = y_init
        self.yerr_init = yerr_init

    def _get_expected_improvement(self, x_new):
        mean_y_new, sigma_y_new = self.gauss_pr.predict(
            np.array([x_new]), return_std=True
        )

        mean_y = self.gauss_pr.predict(self.x_init)
        min_mean_y = np.min(mean_y)
        z = (mean_y_new[0] - min_mean_y - 1) / (sigma_y_new[0] + 1e-9)
        exp_imp = (mean_y_new[0] - min_mean_y - 1) * norm.cdf(z) + sigma_y_new[
            0
        ] * norm.pdf(z)
        return exp_imp

    def _get_ucb(self, x_new):
        mean_y_new, sigma_y_new = self.gauss_pr.predict(
            np.array([x_new]), return_std=True
        )
        return mean_y_new[0] + self.lambda_param * sigma_y_new[0]

    def _get_lcb(self, x_new):
        mean_y_new, sigma_y_new = self.gauss_pr.predict(
            np.array([x_new]), return_std=True
        )
        return mean_y_new[0] - self.lambda_param * sigma_y_new[0]

    def _get_next_probable_point(self):
        min_ei = float(sys.maxsize)
        x_optimal = None
        # Trial with an array of random data points
        rands = np.random.uniform(
            np.array([dim.min_val for dim in self.dims]),
            np.array([dim.max_val for dim in self.dims]),
            (self.batch_size, self.get_n_dimensions()),
        )
        for x_start in rands:
            response = minimize(
                fun=self.acq_function,
                x0=x_start,
                bounds=[(dim.min_val, dim.max_val) for dim in self.dims],
                method="L-BFGS-B",
            )
            if response.fun < min_ei:
                min_ei = response.fun
                x_optimal = []
                for y, dim in zip(response.x, self.dims, strict=False):
                    if dim.round is True and dim.unit is not None:
                        # round so samples is integer

                        x_optimal.append(
                            float(
                                round(
                                    (y * (dim.unit / self.sampling_rate)).to(
                                        "dimensionless"
                                    ),
                                    0,
                                )
                                * (self.sampling_rate / dim.unit)
                            )
                        )
                    else:
                        x_optimal.append(y)
        if x_optimal in self.x_init:
            perturb = np.random.uniform(
                -np.array([(dim.max_val - dim.min_val) / 10 for dim in self.dims]),
                np.array([(dim.max_val - dim.min_val) / 10 for dim in self.dims]),
                (1, len(self.dims)),
            )
            x_optimal += perturb
            new_x_optimal = []
            for y, dim in zip(x_optimal[0], self.dims, strict=False):
                if dim.round is True and dim.unit is not None:
                    # round so samples is integer
                    new_x_optimal.append(
                        float(
                            round(
                                (y * (dim.unit / self.sampling_rate)).to(
                                    "dimensionless"
                                ),
                                0,
                            )
                            * (self.sampling_rate / dim.unit)
                        )
                    )
                else:
                    new_x_optimal.append(y)
            x_optimal = new_x_optimal
            for i, y in enumerate(x_optimal):
                if y > self.dims[i].max_val:
                    x_optimal[i] = self.dims[i].max_val
                elif y < self.dims[i].min_val:
                    x_optimal[i] = self.dims[i].min_val
        return x_optimal, min_ei

    def _extend_prior_with_posterior_data(self, x, y, yerr):
        self.x_init = np.append(self.x_init, np.array([x]), axis=0)
        self.y_init = np.append(self.y_init, np.array(y), axis=0)
        self.yerr_init = np.append(self.yerr_init, np.array(yerr), axis=0)

    def get_first_point(self) -> tuple[Any, None]:
        """
        Get the first optimization point based on initial data.

        Selects the input point corresponding to the minimum observed value
        in the initial samples (`y_init`). This point can serve as the starting
        point for the Bayesian optimization process.

        Sets the following attributes on the optimizer:
        - `y_min` : Minimum observed value in `y_init`.
        - `optimal_x` : Input corresponding to `y_min`.
        - `optimal_ei` : Initialized to None.

        Returns
        -------
        optimal_x
            Input point corresponding to the minimum observed `y_init`.
        optimal_ei
            Placeholder for the expected improvement at `optimal_x` (initialized to None).
        """
        y_min_ind = np.nanargmin(self.y_init)

        self.y_min = self.y_init[y_min_ind]
        self.optimal_x = self.x_init[y_min_ind]
        self.optimal_ei = None

        return self.optimal_x, self.optimal_ei

    @ignore_warnings(category=ConvergenceWarning)
    def iterate_values(self):
        """Iterate the optimization process by fitting the Gaussian Process to the
        current data and selecting the next point based on the acquisition function.
        """
        nan_idxs = np.isnan(self.y_init)
        self.gauss_pr.fit(self.x_init[~nan_idxs], np.array(self.y_init)[~nan_idxs])
        x_next, ei = self._get_next_probable_point()
        return x_next, ei

    def update_db_dict(self, db_dict: Mapping) -> Mapping:
        """
        Update a DSP parameters database with the current optimization point.

        This method updates `db_dict` with the values of the current point being
        evaluated by the optimizer. If it is the first iteration, it retrieves
        the initial point; otherwise, it computes the next point using
        `iterate_values()`. Each dimension's value is stored in the database,
        with units appended if specified.

        Parameters
        ----------
        db_dict
            Dictionary representing the DSP parameters database. Keys are
            dimension names, and values are dictionaries mapping parameter
            identifiers to their corresponding values as strings (including units).

        Returns
        -------
        Mapping
            Updated DSP parameters database including the current optimization point.
        """
        if self.current_iter == 0:
            x_new, ei = self.get_first_point()

        x_new, ei = self.iterate_values()
        self.current_x = x_new
        self.current_ei = ei

        for i, val in enumerate(x_new):
            name, parameter, min_val, max_val, rounding, unit = self.dims[i]
            if unit is not None:
                value_str = f"{val}*{unit.units:~}"
                if "µ" in value_str:
                    value_str = value_str.replace("µ", "u")
            else:
                value_str = f"{val}"
            if name not in db_dict.keys():
                db_dict[name] = {parameter: value_str}
            else:
                db_dict[name][parameter] = value_str
        self.current_iter += 1
        return db_dict

    def update(self, results: Mapping[str, Any]) -> None:
        """
        Update the optimizer with new evaluation results.

        This method updates the internal state of the Bayesian optimizer with
        the latest evaluation at `current_x`. It extends the prior data used
        by the Gaussian Process, updates the current best observed point, and
        tracks distances between consecutive sampled points.

        Parameters
        ----------
        results
            Dictionary-like object containing the latest evaluation. Must
            include keys corresponding to `fom_value` and `fom_error`.
        """

        y_val = results[self.fom_value]
        y_err = results[self.fom_error]
        self._extend_prior_with_posterior_data(
            self.current_x, np.array([y_val]), np.array([y_err])
        )

        if np.isnan(y_val) | np.isnan(y_err):
            pass
        elif y_val < self.y_min:
            self.y_min = y_val
            self.optimal_x = self.current_x
            self.optimal_ei = self.current_ei
            self.optimal_results = results

        if self.current_iter == 1:
            self.prev_x = self.current_x
        else:
            self.distances_.append(
                np.linalg.norm(np.array(self.prev_x) - np.array(self.current_x))
            )
            self.prev_x = self.current_x

        new_entry = pd.DataFrame(
            {"x": self.optimal_x, "y": self.y_min, "ei": self.optimal_ei}
        )
        if (
            not new_entry.empty
            and new_entry.notnull().any().any()
            and len(new_entry) >= 1
        ):
            if self.best_samples_.empty:
                self.best_samples_ = new_entry.copy()
            else:
                self.best_samples_ = pd.concat(
                    [self.best_samples_, new_entry],
                    ignore_index=True,
                )

    def get_best_vals(self):
        """Extract the best observed values in a dictionary format suitable for updating the DSP parameters database."""

        out_dict = {}
        for i, val in enumerate(self.optimal_x):
            name, parameter, min_val, max_val, rounding, unit = self.dims[i]
            if unit is not None:
                value_str = f"{val}*{unit.units:~}"
                if "µ" in value_str:
                    value_str = value_str.replace("µ", "u")
            else:
                value_str = f"{val}"
            if name not in out_dict:
                out_dict[name] = {parameter: value_str}
            else:
                out_dict[name][parameter] = value_str
        return out_dict

    @ignore_warnings(category=ConvergenceWarning)
    def plot(self, init_samples=None) -> Figure:
        """Plot the Gaussian Process regression and acquisition function.

        For 1D and 2D optimization problems, this method visualizes the fitted
        Gaussian Process and the acquisition function across the search space.
        Initial samples, failed samples, and the optimal point are highlighted.

        Parameters
        ----------
        init_samples
            Optional list of initial sample points to highlight on the plot.

        Returns
        -------
        fig
            Matplotlib Figure object containing the plot of the Gaussian Process
            regression and acquisition function.
        """

        nan_idxs = np.isnan(self.y_init)
        fail_idxs = np.isnan(self.yerr_init)
        self.gauss_pr.fit(self.x_init[~nan_idxs], np.array(self.y_init)[~nan_idxs])
        if (len(self.dims) != 2) and (len(self.dims) != 1):
            raise Exception("Acquisition Function Plotting not implemented for dim!=2")
        if len(self.dims) == 1:
            points = np.arange(self.dims[0].min_val, self.dims[0].max_val, 0.1)

            ys, ys_err = self.gauss_pr.predict(points.reshape(-1, 1), return_std=True)

            fig = plt.figure()

            plt.scatter(np.array(self.x_init), np.array(self.y_init), label="Samples")
            plt.scatter(
                np.array(self.x_init)[fail_idxs],
                np.array(self.y_init)[fail_idxs],
                color="green",
                label="Failed samples",
            )
            plt.fill_between(points, ys - ys_err, ys + ys_err, alpha=0.1)
            if init_samples is not None:
                init_ys = np.array(
                    [
                        np.where(init_sample == self.x_init)[0][0]
                        for init_sample in init_samples
                    ]
                )
                plt.scatter(
                    np.array(init_samples)[:, 0],
                    np.array(self.y_init)[init_ys],
                    color="red",
                    label="Init Samples",
                )
            plt.scatter(self.optimal_x[0], self.y_min, color="orange", label="Optimal")

            plt.xlabel(
                f"{self.dims[0].name}-{self.dims[0].parameter}({self.dims[0].unit})"
            )
            plt.ylabel("Kernel Value")
            plt.legend()
        elif len(self.dims) == 2:
            x, y = np.mgrid[
                self.dims[0].min_val : self.dims[0].max_val : 0.1,
                self.dims[1].min_val : self.dims[1].max_val : 0.1,
            ]
            points = np.vstack((x.flatten(), y.flatten())).T
            out_grid = np.zeros(
                (
                    int((self.dims[0].max_val - self.dims[0].min_val) * 10),
                    int((self.dims[1].max_val - self.dims[1].min_val) * 10),
                )
            )

            j = 0
            for i, _ in np.ndenumerate(out_grid):
                out_grid[i] = self.gauss_pr.predict(
                    points[j].reshape(1, -1), return_std=False
                )
                j += 1

            fig = plt.figure()
            plt.imshow(
                out_grid,
                norm=LogNorm(),
                origin="lower",
                aspect="auto",
                extent=(0, out_grid.shape[1], 0, out_grid.shape[0]),
            )
            plt.scatter(
                np.array(self.x_init - self.dims[1].min_val)[:, 1] * 10,
                np.array(self.x_init - self.dims[0].min_val)[:, 0] * 10,
            )
            if init_samples is not None:
                plt.scatter(
                    (init_samples[:, 1] - self.dims[1].min_val) * 10,
                    (init_samples[:, 0] - self.dims[0].min_val) * 10,
                    color="red",
                )
            plt.scatter(
                (self.optimal_x[1] - self.dims[1].min_val) * 10,
                (self.optimal_x[0] - self.dims[0].min_val) * 10,
                color="orange",
            )
            ticks, labels = plt.xticks()
            labels = np.linspace(self.dims[1].min_val, self.dims[1].max_val, 5)
            ticks = np.linspace(0, out_grid.shape[1], 5)
            plt.xticks(ticks=ticks, labels=labels, rotation=45)
            ticks, labels = plt.yticks()
            labels = np.linspace(self.dims[0].min_val, self.dims[0].max_val, 5)
            ticks = np.linspace(0, out_grid.shape[0], 5)
            plt.yticks(ticks=ticks, labels=labels, rotation=45)
            plt.xlabel(
                f"{self.dims[1].name}-{self.dims[1].parameter}({self.dims[1].unit})"
            )
            plt.ylabel(
                f"{self.dims[0].name}-{self.dims[0].parameter}({self.dims[0].unit})"
            )
        plt.title(f"{self.dims[0].name} Kernel Prediction")
        plt.tight_layout()
        plt.close()
        return fig

    @ignore_warnings(category=ConvergenceWarning)
    def plot_acq(self, init_samples=None):
        """Plot the acquisition function across the search space."""

        nan_idxs = np.isnan(self.y_init)
        self.gauss_pr.fit(self.x_init[~nan_idxs], np.array(self.y_init)[~nan_idxs])
        if (len(self.dims) != 2) and (len(self.dims) != 1):
            raise Exception("Acquisition Function Plotting not implemented for dim!=2")
        if len(self.dims) == 1:
            points = np.arange(self.dims[0].min_val, self.dims[0].max_val, 0.1)
            ys = np.zeros_like(points)
            for i, point in enumerate(points):
                ys[i] = self.acq_function(np.array([point]).reshape(1, -1)[0])
            fig = plt.figure()
            plt.plot(points, ys)
            plt.scatter(np.array(self.x_init), np.array(self.y_init), label="Samples")
            if init_samples is not None:
                init_ys = np.array(
                    [
                        np.where(init_sample == self.x_init)[0][0]
                        for init_sample in init_samples
                    ]
                )
                plt.scatter(
                    np.array(init_samples)[:, 0],
                    np.array(self.y_init)[init_ys],
                    color="red",
                    label="Init Samples",
                )
            plt.scatter(self.optimal_x[0], self.y_min, color="orange", label="Optimal")

            plt.xlabel(
                f"{self.dims[0].name}-{self.dims[0].parameter}({self.dims[0].unit})"
            )
            plt.ylabel("Acquisition Function Value")
            plt.legend()

        elif len(self.dims) == 2:
            x, y = np.mgrid[
                self.dims[0].min_val : self.dims[0].max_val : 0.1,
                self.dims[1].min_val : self.dims[1].max_val : 0.1,
            ]
            points = np.vstack((x.flatten(), y.flatten())).T
            out_grid = np.zeros(
                (
                    int((self.dims[0].max_val - self.dims[0].min_val) * 10),
                    int((self.dims[1].max_val - self.dims[1].min_val) * 10),
                )
            )

            j = 0
            for i, _ in np.ndenumerate(out_grid):
                out_grid[i] = self.acq_function(points[j])
                j += 1

            fig = plt.figure()
            plt.imshow(
                out_grid,
                norm=LogNorm(),
                origin="lower",
                aspect="auto",
                extent=(0, out_grid.shape[1], 0, out_grid.shape[0]),
            )
            plt.scatter(
                np.array(self.x_init - self.dims[1].min_val)[:, 1] * 10,
                np.array(self.x_init - self.dims[0].min_val)[:, 0] * 10,
            )
            if init_samples is not None:
                plt.scatter(
                    (init_samples[:, 1] - self.dims[1].min_val) * 10,
                    (init_samples[:, 0] - self.dims[0].min_val) * 10,
                    color="red",
                )
            plt.scatter(
                (self.optimal_x[1] - self.dims[1].min_val) * 10,
                (self.optimal_x[0] - self.dims[0].min_val) * 10,
                color="orange",
            )
            ticks, labels = plt.xticks()
            labels = np.linspace(self.dims[1].min_val, self.dims[1].max_val, 5)
            ticks = np.linspace(0, out_grid.shape[1], 5)

            plt.xticks(ticks=ticks, labels=labels, rotation=45)
            ticks, labels = plt.yticks()

            labels = np.linspace(self.dims[0].min_val, self.dims[0].max_val, 5)
            ticks = np.linspace(0, out_grid.shape[0], 5)

            plt.yticks(ticks=ticks, labels=labels, rotation=45)
            plt.xlabel(
                f"{self.dims[1].name}-{self.dims[1].parameter}({self.dims[1].unit})"
            )
            plt.ylabel(
                f"{self.dims[0].name}-{self.dims[0].parameter}({self.dims[0].unit})"
            )
        plt.title(f"{self.dims[0].name} Acquisition Space")
        plt.tight_layout()
        plt.close()
        return fig


def run_bayesian_optimisation(
    tb_data: Table,
    dsp_config: str | Mapping,
    fom_function: list | Callable,
    optimisers: list | BayesianOptimizer,
    fom_kwargs: Mapping | None = None,
    db_dict: Mapping | None = None,
    nan_val: float | list = 10,
    n_iter: int = 10,
) -> tuple[Mapping, list]:
    """Run Bayesian optimization loop to optimize DSP parameters.

    Parameters
    ----------
    tb_data
        An input table of the data for which the DSP is performed.
    dsp_config
        Configuration for the DSP processing chain, see `run_one_dsp()`
    fom_function
        Function(s) to calculate the figure-of-merit from the DSP output table.
        If a list is given, each function will be applied to the output of each
        optimiser in order. If a single function is given, it will be applied
        to the output of all optimisers.
        This function should return a dictionary containing a keys of name `fom_value` and `fom_error` in the optimisers, which will be used to inform the optimization process.
    optimisers
        A list of `BayesianOptimizer` instances to run in parallel. Each optimizer
        will update the DSP parameters database and receive the results of each iteration to inform its next point selection.
    fom_kwargs
        Optional keyword arguments for the `fom_function`. If a list is given, each set
        of kwargs will be applied to the corresponding function in `fom_function`. If a single set of kwargs is given, it will be applied to all functions.
    db_dict
        Optional initial DSP parameters database to be updated by the optimizers. If not provided, an
        empty dictionary will be used.
    nan_val
        Value to assign to the figure-of-merit if it is NaN. Can be a single float applied to all optimizers or a list of floats corresponding to each optimizer.
    n_iter
        Number of optimization iterations to perform. Each iteration consists of each optimizer proposing a new point, running the DSP with those parameters, evaluating the figure-of-merit, and updating the optimizers with the results.

    Returns
    -------
    out_param_dict
        A dictionary containing the best parameter values found by each optimizer, formatted for updating the DSP parameters
        database.
    out_results_list
        A list of dictionaries containing the optimal results corresponding to the best parameters found by each optimizer.

    Note
    ----
    The optimisers need to be initialised with the parameters set, and initial values added.
    """

    if db_dict is None:
        db_dict = {}

    if not isinstance(optimisers, list):
        optimisers = [optimisers]
    if not isinstance(fom_kwargs, list):
        fom_kwargs = [fom_kwargs]
    if not isinstance(fom_function, list):
        fom_function = [fom_function]

    for j in range(n_iter):
        for optimiser in optimisers:
            db_dict = optimiser.update_db_dict(db_dict)

        log.info(f"Iteration number: {j + 1}")
        log.info(f"Processing with {db_dict}")

        tb_out = run_one_dsp(tb_data, dsp_config, db_dict=db_dict)

        res = np.ndarray(shape=len(optimisers), dtype="O")

        for i in range(len(optimisers)):
            if fom_kwargs[i] is not None:
                if len(fom_function) > 1:
                    res[i] = fom_function[i](tb_out, fom_kwargs[i])
                else:
                    res[i] = fom_function[0](tb_out, fom_kwargs[i])
            elif len(fom_function) > 1:
                res[i] = fom_function[i](tb_out)
            else:
                res[i] = fom_function[0](tb_out)

        log.info(f"Results of iteration {j + 1} are {res}")

        for i, optimiser in enumerate(optimisers):
            if np.isnan(res[i][optimiser.fom_value]):
                if isinstance(nan_val, list):
                    res[i][optimiser.fom_value] = nan_val[i]
                else:
                    res[i][optimiser.fom_value] = nan_val

            optimiser.update(res[i])

    out_param_dict = {}
    out_results_list = []

    for optimiser in optimisers:
        param_dict = optimiser.get_best_vals()
        out_param_dict.update(param_dict)
        results_dict = optimiser.optimal_results

        if np.isnan(results_dict[optimiser.fom_value]):
            log.error(f"Energy optimisation failed for {optimiser.dims[0][0]}")
        out_results_list.append(results_dict)

    return out_param_dict, out_results_list
