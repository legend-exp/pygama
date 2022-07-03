from __future__ import annotations

import re

import pygama.evt.tcm as ptcm
import pygama.lgdo as lgdo


def build_tcm(input_tables:list, coin_col:str, hash_func:str=r'\d+',
              coin_window:float=0, window_ref:str='last',
              out_file:str=None, out_name:str='tcm', wo_mode:str='write_safe'):
    """
    Given a list of input tables, create an output table containing an entry
    list of coincidences among the inputs. Uses tcm.generate_tcm_cols().  For
    use with the data loader.

    Parameters
    ----------
    input_tables : list of tuples
        each entry is (filename, table_name_pattern). All tables matching
        [table_name_pattern] in [filename] will be added to the list of input
        tables.
    coin_col : str
        the name of the column in each tables used to build coincidences. All
        tables must contain a column with this name
    hash_func : str (re) or None
        function to map table names to ints for use in the tcm
        str hash_func is a regexp pattern that acts on each table_name. The
        default hash_func pull the first int out of the table name
        setting to None will use a table's index in input_tables
        Later can add list or dict or a function(str) --> int
    coin_window : float
        The clustering window width (see generate_tcm_cols)
    window_ref : str
        Configuration for the clustering window (see generate_tcm_cols)
    out_file : str or None
        Name (including path) for the output file. If None, no file will be
        written; the tcm will just be returned in memory
    out_name : str
        Name for the tcm table in the output file
    wo_mode : str
        mode to send to LH5Store.write_object(). Typically 'w', 'o', or 'of'

    Returns
    -------
    tcm : lgdo.Table
        The tcm!
    """

    store = lgdo.LH5Store()
    coin_data = []
    array_ids = []
    all_tables = []
    for filename, pattern in input_tables:
        tables = lgdo.ls(filename, lh5_group=pattern)
        for table in tables:
            all_tables.append(table)
            array_id = len(array_ids)
            if hash_func is not None:
                if isinstance(hash_func, str):
                    array_id = int(re.search(hash_func, table).group())
                else: raise NotImplementedError(f"hash_func of type {type(hash_func).__name__}")
            else: array_id = len(all_tables)-1
            table = table + '/' + coin_col
            coin_data.append(store.read_object(table, filename)[0].nda)
            array_ids.append(array_id)

    tcm_cols = ptcm.generate_tcm_cols(coin_data, coin_window=coin_window,
                                      window_ref=window_ref, array_ids=array_ids)

    for key in tcm_cols: tcm_cols[key] = lgdo.Array(nda=tcm_cols[key])
    tcm = lgdo.Table(col_dict=tcm_cols, attrs={ 'tables':str(all_tables), 'hash_func':str(hash_func) })

    if out_file is not None:
        store.write_object(tcm, out_name, out_file, wo_mode=wo_mode)

    return tcm
