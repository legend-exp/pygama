from .table import Table
from .array import Array
from .arrayofequalsizedarrays import ArrayOfEqualSizedArrays
from .vectorofvectors import VectorOfVectors


class WaveformTable(Table):
    """
    An lgdo for storing blocks of (1D) time-series data.

    A WaveformTable is an lgdo Table with the 3 columns t0, dt, and values:
    * t0[i] is a time offset (relative to a user-defined global reference) for
      the sample in values[i][0]. Implemented as an lgdo Array with optional
      attribute "units"
    * dt[i] is the sampling period for the waveform at values[i]. Implemented as
      an lgdo Array with optional attribute "units"
    * values[i] is the i'th waveform in the table. Internally, the waveforms
      "values" may be either an lgdo ArrayOfEqualSizedArrays<1,1> or as an lgdo
      VectorOfVectors that supports waveforms of unequal length. Can optionally
      be given a "units" attribute

    Note that on-disk and in-memory versions could be different e.g. if a
    compression routine is used.
    """

    def __init__(self, size=None, t0=0, t0_units=None, dt=1, dt_units=None, 
                 values=None, values_units=None, wf_len=None, dtype=None, attrs={}):
        """
        Parameters
        ----------
        size : int or None (optional)
            sets the number of rows in the table. If None, the size will be
            determined from the first among t0, dt, or values to return a valid
            length. If not None, t0, dt, and values will be resized as necessary
            to match size. If size is None and t0, dt, and values are all
            non-array-like, a default size of 1024 is used.
        t0 : float, array-like, or lgdo.Array (optional)
            t0 values to be used (or broadcast) to the t0 column. 0 by default.
        t0_units : str or None
            units for the t0 values. If not none and t0 is an lgdo Array,
            overrides what's in t0.
        dt : float, array-like, or lgdo.Array (optional)
            dt values to be used (or broadcast) to the t0 column
        dt_units : str or None
            units for the dt values. If not none and dt is an lgdo Array,
            overrides what's in dt.
        values : 2D ndarray, lgdo VectorOfVectors, lgdo ArrayOfEqualSizedArrays, or None (optional)
            The waveform data to be stored in the table. If None (the default) a
            block of data is prepared based on the wf_len and dtype arguments.
        values_units : str or None
            units for the waveform values. If not none and values is an lgdo
            Array, overrides what's in values
        wf_len : int or None (optional)
            The length of the waveforms in each entry of a table. If None (the
            default), unequal lengths are assumed and VectorOfVectors is used
            for the values column. Ignored if values is a 2D ndarray, in which
            case values.shape[1] is used
        dtype : numpy dtype or None (optional)
            The numpy dtype of the waveform data. If values is not None, this
            argument is ignored. If both values and dtype are None, np.float64
            is used.
        attrs : dict (optional)
            A set of user attributes to be carried along with this lgdo
        """

        if size is None:
            if hasattr(t0, '__len__'): size = len(t0)
            elif hasattr(dt, '__len__'): size = len(dt)
            elif hasattr(values, '__len__'): size = len(values)
            if size is None: size = 1024

        if not isinstance(t0, Array):
            shape = (size,)
            t0_dtype = t0.dtype if hasattr(t0, 'dtype') else np.float32
            nda = t0 if isinstance(t0, np.ndarray) else np.full(shape, t0, dtype=t0_dtype)
            if nda.shape != shape: nda.resize(shape)
            t0 = Array(nda)
        if t0_units is not None: t0.attrs['units'] = f'{t0_units}'
            
        if not isinstance(dt, Array):
            shape = (size,)
            dt_dtype = dt.dtype if hasattr(dt, 'dtype') else np.float32
            nda = dt if isinstance(dt, np.ndarray) else np.full(shape, dt, dtype=dt_dtype)
            if nda.shape != shape: nda.resize(shape)
            dt = Array(nda)
        if dt_units is not None: dt.attrs['units'] = f'{dt_units}'

        if not isinstance(values, ArrayOfEqualSizedArrays) and not isinstance(values, VectorOfVectors):
            if isinstance(values, np.ndarray): wf_len = values.shape[1]
            if wf_len is None: # VectorOfVectors
                shape_guess = (size, 100)
                if dtype is None: dtype = np.float64
                values = VectorOfVectors(shape_guess=shape_guess, dtype=dtype)
            else: # ArrayOfEqualSizedArrays
                shape = (size, wf_len)
                if dtype is None:
                    dtype = values.dtype if hasattr(values, 'dtype') else np.float64
                nda = values if isinstance(values, np.ndarray) else np.zeros(shape, dtype=dtype)
                if nda.shape != shape: nda.resize(shape)
                values = ArrayOfEqualSizedArrays(dims=(1,1), nda)
        if values_units is not None: values.attrs['units'] = f'{values_units}'

        col_dict = {}
        col_dict['t0'] = t0
        col_dict['dt'] = dt
        col_dict['values'] = values
        super().__init__(size=size, col_dict=col_dict, attrs=attrs)
