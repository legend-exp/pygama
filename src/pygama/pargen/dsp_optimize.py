import logging
import multiprocessing as mp
from collections import namedtuple
from multiprocessing import get_context
from pprint import pprint

import numpy as np

from pygama.dsp import build_processing_chain

log = logging.getLogger(__name__)


def run_one_dsp(
    tb_data, dsp_config, db_dict=None, fom_function=None, verbosity=0, fom_kwargs=None
):
    """
    run one iteration of DSP on tb_data

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

    pc, lh5_col_names, tb_out = build_processing_chain(
        tb_data, dsp_config, db_dict=db_dict
    )
    pc.execute()
    if fom_function is not None:
        if fom_kwargs is not None:
            return fom_function(tb_out, verbosity, fom_kwargs)
        else:
            return fom_function(tb_out, verbosity)
    else:
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
        the order appearin in dims (first dimension is first for loop, etc):
        Return False when the grid runs out of indices. Otherwise returns True.
        """
        for iD in reversed(range(self.get_n_dimensions())):
            indices[iD] += 1
            if indices[iD] < self.get_n_points_of_dim(iD):
                return True
            indices[iD] = 0
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
            pprint(dsp_config)
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
    for index, grid in zip(iii, grids):
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
            else:
                if len(fom_function) > 1:
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

        for i, gri in enumerate(zip(iii, grid)):
            if not gri[1].iterate_indices(gri[0]):
                log.info(f"{i} grid end")
                iii[i] = gri[1].get_zero_indices()
                complete[i] = True
        if all(complete):
            break
    return out


def run_grid_multiprocess_parallel(
    tb_data,
    dsp_config,
    grid,
    fom_function,
    db_dict=None,
    verbosity=1,
    processes=5,
    fom_kwargs=None,
):

    """
    run one iteration of DSP on tb_data with multiprocessing, can handle
    multiple grids if they are the same dimensions

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
    grid : pargrid, list of pargrids
        Grids to run optimization on
    db_dict : dict (optional)
        DSP parameters database. See build_processing_chain for formatting info
    fom_function : function or None (optional)
        When given the output lh5 table of this DSP iteration, the
        fom_function must return a scalar figure-of-merit value upon which the
        optimization will be based. Should accept verbosity as a second argument.
        If multiple grids provided can either pass one fom to have it run for each grid
        or a list of fom to run different fom on each grid.
    verbosity : int (optional)
        verbosity for the processing chain and fom_function calls
    processes : int
        DOCME
    fom_kwargs
        any keyword arguments to pass to the fom,
        if multiple grids given will need to be a list of the fom_kwargs for each grid

    Returns
    -------
    figure_of_merit : float
        If fom_function is not None, returns figure-of-merit value for the DSP iteration
    tb_out : lh5 Table
        If fom_function is None, returns the output lh5 table for the DSP iteration
    """

    if not isinstance(grid, list):
        grid = [grid]
    if not isinstance(fom_function, list) and fom_function is not None:
        fom_function = [fom_function]
    if not isinstance(fom_kwargs, list):
        fom_kwargs = [fom_kwargs for gri in grid]
    grid_values = []
    shapes = [gri.get_shape() for gri in grid]
    if fom_function is not None:
        for i in range(len(grid)):
            grid_values.append(np.ndarray(shape=shapes[i], dtype="O"))
    else:
        grid_lengths = np.array([gri.get_n_grid_points() for gri in grid])
        grid_values.append(np.ndarray(shape=shapes[np.argmax(grid_lengths)], dtype="O"))
    grid_list = get_grid_points(grid)
    pool = mp.Pool(processes=processes)
    results = [
        pool.apply_async(
            run_grid_point,
            args=(
                tb_data,
                dsp_config,
                grid,
                fom_function,
                np.asarray(gl),
                db_dict,
                verbosity,
                fom_kwargs,
            ),
        )
        for gl in grid_list
    ]

    for result in results:
        res = result.get()
        indexes = res["indexes"]
        if fom_function is not None:
            for i in range(len(grid)):
                index = indexes[i]
                if grid_values[i][index] is None:
                    grid_values[i][index] = res["results"][i]
        else:
            grid_values[0][indexes[0]] = {f"{indexes[0]}": res["results"]}

    pool.close()
    pool.join()
    return grid_values
