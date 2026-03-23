"""
Utility functions for parameter fitting, unit conversion, and LH5 data loading
used across the pargen calibration and optimisation modules.
"""

from __future__ import annotations

import logging
from types import FunctionType

import numpy as np
import pandas as pd
from iminuit import Minuit, cost
from lgdo import lh5

log = logging.getLogger(__name__)


def convert_to_minuit(pars, func) -> Minuit:
    """
    Create an :class:`iminuit.Minuit` instance from a parameter set and a PDF.

    Parameters
    ----------
    pars
        Initial parameter values.  Either a dict mapping parameter names to
        values, or a sequence of values that will be passed positionally.
    func
        Callable whose signature defines the parameters.  If the object
        exposes a ``pdf_ext`` attribute it is used as the cost function (so
        that extended PDFs are handled correctly); otherwise the callable
        itself is used.

    Returns
    -------
    m
        Configured Minuit object ready for minimisation.
    """
    try:
        c = cost.UnbinnedNLL(np.array([0]), func.pdf_ext)
    except AttributeError:
        c = cost.UnbinnedNLL(np.array([0]), func)
    return Minuit(c, **pars) if isinstance(pars, dict) else Minuit(c, *pars)


def return_nans(input) -> tuple:
    """
    Return a NaN-filled result tuple with the same structure as a successful fit.

    Useful for propagating fit failures without raising exceptions.

    Parameters
    ----------
    input
        Either a plain callable (whose positional arguments after the first
        define the parameter list) or an object with a ``required_args()``
        method (e.g. a pygama distribution).

    Returns
    -------
    values
        Parameter values, all set to NaN.
    errors
        Parameter uncertainties, all set to NaN.
    covariance
        Square covariance matrix filled with NaN.
    """
    if isinstance(input, FunctionType):
        args = input.__code__.co_varnames[: input.__code__.co_argcount][1:]
        m = convert_to_minuit(np.full(len(args), np.nan), input)
        return m.values, m.errors, np.full((len(m.values), len(m.values)), np.nan)  # noqa: PD011
    args = input.required_args()
    m = convert_to_minuit(np.full(len(args), np.nan), input)
    return m.values, m.errors, np.full((len(m.values), len(m.values)), np.nan)  # noqa: PD011


def load_data(
    files: str | list | dict,
    lh5_path: str,
    cal_dict: dict,
    params: set,
    cal_energy_param: str = "cuspEmax_ctc_cal",
    threshold=None,
    return_selection_mask=False,
) -> pd.DataFrame | tuple(pd.DataFrame, np.array):
    """
    Load parameters from LH5 files and apply calibration expressions.

    Reads *params* from *files*, evaluates all expressions in *cal_dict*
    to produce calibrated columns, and optionally applies a lower energy
    threshold.  When *files* is a dict keyed by run timestamp, the
    function recurses over each timestamp and concatenates the results.

    Parameters
    ----------
    files
        A single file path, a list of file paths, or a dict mapping run
        timestamps to lists of file paths.
    lh5_path
        Path to the LH5 table within each file.
    cal_dict
        Calibration expressions in hit-dict format:
        ``{outname: {"expression": ..., "parameters": {...}}}``.  When
        *files* is a timestamp dict, this may also be keyed by timestamp.
    params
        Set of output column names to include in the returned DataFrame.
    cal_energy_param
        Name of the calibrated energy column used to apply the threshold.
    threshold
        Minimum energy value; events below this are dropped.  ``None``
        keeps all events.
    return_selection_mask
        If ``True``, also return the boolean threshold mask.

    Returns
    -------
    df
        DataFrame containing the requested *params* (plus an optional
        ``run_timestamp`` column when *files* is a dict).
    masks
        Boolean threshold mask of the same length as *df*.  Only returned
        when *return_selection_mask* is ``True``.
    """

    params = set(params)
    if isinstance(files, str):
        files = [files]

    if isinstance(files, dict):
        # Go through each tstamp and recursively load_data on file lists
        data_df = []
        masks = []
        for tstamp, tfiles in files.items():
            file_df = load_data(
                tfiles,
                lh5_path,
                cal_dict.get(tstamp, cal_dict),
                params,
                cal_energy_param,
                threshold,
                return_selection_mask,
            )

            if return_selection_mask:
                file_df[0]["run_timestamp"] = np.full(
                    len(file_df[0]), tstamp, dtype=object
                )
                data_df.append(file_df[0])
                masks.append(file_df[1])
            else:
                file_df["run_timestamp"] = np.full(len(file_df), tstamp, dtype=object)
                data_df.append(file_df)

        data_df = pd.concat(data_df)
        if return_selection_mask:
            masks = np.concatenate(masks)

    elif isinstance(files, list):
        # Get set of available fields between input table and cal_dict
        file_keys = lh5.ls(
            files[0], lh5_path if lh5_path[-1] == "/" else lh5_path + "/"
        )
        file_keys = {key.split("/")[-1] for key in file_keys}

        # Get set of keys in calibration expressions that show up in file
        cal_keys = {
            name
            for info in cal_dict.values()
            for name in compile(info["expression"], "0vbb is real!", "eval").co_names
        } & file_keys

        # Get set of fields to read from files
        fields = cal_keys | (file_keys & params)

        lh5_it = lh5.iterator.LH5Iterator(
            files, lh5_path, field_mask=fields, buffer_len=100000
        )
        df_fields = params & (fields | set(cal_dict))
        if df_fields != params:
            log.debug(
                "load_data(): params not found in data files or cal_dict: %s",
                params - df_fields,
            )
        data_df = pd.DataFrame(columns=list(df_fields))

        for table in lh5_it:
            # Evaluate all provided expressions and add to table
            for outname, info in cal_dict.items():
                table[outname] = table.eval(
                    info["expression"], info.get("parameters", None)
                )
            entry = lh5_it.current_global_entries[0]
            n_rows = len(table)

            # Copy params in table into dataframe
            for par in data_df:
                # First set of entries: allocate enough memory for all entries
                if entry == 0:
                    data_df[par] = np.resize(table[par], len(lh5_it))
                else:
                    data_df.loc[entry : entry + n_rows - 1, par] = table[par][:n_rows]

        # Evaluate threshold mask and drop events below threshold
        if threshold is not None:
            masks = data_df[cal_energy_param] > threshold
            data_df = data_df.drop(np.where(~masks)[0])
        else:
            masks = np.ones(len(data_df), dtype=bool)

    log.debug("data loaded")
    if return_selection_mask:
        return data_df, masks
    return data_df
