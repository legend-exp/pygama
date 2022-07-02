from __future__ import annotations

import re

import pygama.evt.tcm as ptcm
import pygama.lgdo as lgdo


def build_tcm(input_tables:list, coin_col:str, hash_func:str|list|dict=r'\d+',
              coin_window:float=0, window_ref:str='last',
              out_file:str=None, out_name:str='tcm', overwrite:bool=True):
    """
    Given a list of input tables, create an output table containing an entry
    list of coincidences among the inputs. Uses tcm.generate_coincidence_map_cols().
    For use with the data loader.

    Parameters
    ----------
    input_tables : list of tuples
        each entry is (filename, group_name_pattern)
    coin_col : str
    hash_func : str (re) or list or dict or a function(str) --> int or None
        hash function for table_name --> int to use as an array_id in the tcm
        set to None to use index in input_tables list as array_id
    coin_window : float
    window_ref : str
    out_file : str
    out_name : str
    overwrite : bool

    Returns
    -------
    tcm : lgdo.Table or None
        If out_file is None, return the lgdo Table
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
            table = table + '/' + coin_col
            coin_data.append(store.read_object(table, filename)[0].nda)
            array_ids.append(array_id)

    tcm_cols = ptcm.generate_tcm_cols(coin_data, coin_window=coin_window,
                                      window_ref=window_ref, array_ids=array_ids)

    for key in tcm_cols: tcm_cols[key] = lgdo.Array(nda=tcm_cols[key])
    tcm = lgdo.Table(col_dict=tcm_cols, attrs={ 'tables':str(all_tables), 'hash_func':str(hash_func) })

    if out_file is None: return tcm
    if overwrite: store.write_object(tcm, out_name, out_file, wo_mode='o', verbosity=1)
    else: store.write_object(tcm, out_name, out_file)
