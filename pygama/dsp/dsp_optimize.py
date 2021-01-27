import numpy as np
from .build_processing_chain import build_processing_chain
from collections import namedtuple
from pprint import pprint


def run_one_dsp(tb_data, dsp_config, db_dict=None, fom_function=None, verbosity=0):
    """
    run one iteration of DSP on tb_data 

    Optionally returns a value for optimization

    Parameters:
    -----------
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

    Returns:
    --------
    figure_of_merit : float
        If fom_function is not None, returns figure-of-merit value for the DSP iteration
    tb_out : lh5 Table
        If fom_function is None, returns the output lh5 table for the DSP iteration
    """
    
    pc, tb_out = build_processing_chain(tb_data, dsp_config, db_dict=db_dict, verbosity=verbosity)
    pc.execute()
    if fom_function is not None: return fom_function(tb_out, verbosity)
    else: return tb_out



ParGridDimension = namedtuple('ParGridDimension', 'name i_arg value_strs companions')

class ParGrid():
    """ Parameter Grid class

    Each ParGrid entry corresponds to a dsp parameter to be varied.
    The ntuples must follow the pattern: 
    ( name, i_arg, value_strs companions) : ( str, int, list of str, list or None )
    where name is the name of the dsp routine in dsp_config whose  be
    optimized, i_arg is the index of the argument to be varied, value_strs is
    the array of strings to set the argument to, and companions is an optional
    list of ( name, i_arg, value_strs ) tuples for companion arguments that
    need to change along with this one
    """
    def __init__(self):
        self.dims = []

    def add_dimension(self, name, i_arg, value_strs, companions = None):
        self.dims.append( ParGridDimension(name, i_arg, value_strs, companions) )

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
        """ return a meshgrid of parameter values

        Always uses Matrix indexing (natural for par grid) so that
        mg[i1][i2][...] corresponds to index order in self.dims

        Note copy is False by default as opposed to numpy default of True
        """     
        axes = []
        for i in range(self.get_n_dimensions()):
            axes.append(self.dims[i].values_strs)
        return np.meshgrid(*axes, copy, sparse, indexing='ij')

    def get_zero_indices(self):
        return np.zeros(self.get_n_dimensions(), dtype=np.uint32)

    def iterate_indices(self, indices):
        """ iterate given indices [i1, i2, ...] by one.

        For easier iteration. The convention here is arbitrary, but its the
        order the arrays would be traversed in a series of nested for loops in
        the order appearin in dims (first dimension is first for loop, etc):

        Return False when the grid runs out of indices. Otherwise returns True.
        """
        for iD in reversed(range(self.get_n_dimensions())):
            indices[iD] += 1
            if indices[iD] < self.get_n_points_of_dim(iD): return True
            indices[iD] = 0
        return False

    def get_data(self, i_dim, i_par):
        name = self.dims[i_dim].name
        i_arg = self.dims[i_dim].i_arg
        value_str = self.dims[i_dim].value_strs[i_par]
        companions = self.dims[i_dim].companions
        return name, i_arg, value_str, companions

    def print_data(self, indices):
        print(f"Grid point at indices {indices}:")
        for i_dim, i_par in enumerate(indices):
            name, i_arg, value_str, _ = self.get_data(i_dim, i_par)
            print(f"{name}[{i_arg}] = {value_str}")

    def set_dsp_pars(self, dsp_config, indices):
        for i_dim, i_par in enumerate(indices):
            name, i_arg, value_str, companions = self.get_data(i_dim, i_par)
            dsp_config['processors'][name]['args'][i_arg] = value_str
            if companions is None: continue
            for ( c_name, c_i_arg, c_value_str ) in companions:
               dsp_config['processors'][c_name]['args'][c_i_arg] = c_value_str[i_par]


def run_grid(tb_data, dsp_config, grid, fom_function, db_dict=None, verbosity=0):
    """Extract a table of optimization values for a grid of DSP parameters 

    The grid argument defines a list of parameters and values over which to run
    the DSP defined in dsp_config on tb_data. At each point, a scalar
    figure-of-merit is extracted

    Returns a N-dimensional ndarray of figure-of-merit values, where the array
    axes are in the order they appear in grid.

    Parameters:
    -----------
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

    Returns:
    --------
    grid_values : ndarray of floats
        An N-dimensional numpy ndarray whose Mth axis corresponds to the Mth row
        of the grid argument
    """

    grid_values = np.ndarray(shape=grid.get_shape())
    iii = grid.get_zero_indices()
    if verbosity > 0: print("starting grid calculations...")
    while True:    
        grid.set_dsp_pars(dsp_config, iii)
        if verbosity > 1: pprint(dsp_config)
        if verbosity > 0: grid.print_data(iii)
        grid_values[tuple(iii)] = run_one_dsp(tb_data,
                                              dsp_config,
                                              db_dict=db_dict,
                                              fom_function=fom_function,
                                              verbosity=verbosity)
        if verbosity > 0: print("value:", grid_values[tuple(iii)])
        if not grid.iterate_indices(iii): break
    return grid_values
        
