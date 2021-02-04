"""
utility functions for metadata lookup
"""
import tinydb as db
from tinydb.storages import MemoryStorage
import pprint as pp


def write_pretty(db_dict, f_db):
    """
    write a TinyDB database to a special pretty-printed JSON file.
    if i cared less about the format, i would just call TinyDB(index=2)
    """
    pretty_db = {}
    for tb_name, tb_vals in db_dict.items(): # convert pprint integer keys to str
        pretty_db[tb_name] = {str(tb_idx):tb_row for tb_idx, tb_row in tb_vals.items()}
    pretty_str = pp.pformat(pretty_db, indent=2, width=120) #sort_dicts is Py>3.8
    pretty_str = pretty_str.replace("'", "\"")
    with open(f_db, 'w') as f:
        f.write(pretty_str)
