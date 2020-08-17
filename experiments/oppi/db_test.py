#!/usr/bin/env python3
import os, time
import json
import tinydb as db
from tinydb.storages import MemoryStorage

import pandas as pd
import pprint as pp
from pprint import pprint
from datetime import datetime

def main():
    """
    maybe I should make a class that handles creation of calib metadata?
    """
    # test functions
    # write_db()
    # view_db()
    # modify_db()
    
    # set up empty file for ecal_db to write to
    setup_oppi_ecaldb()
    
    
def write_db():
    """
    reference:
    https://tinydb.readthedocs.io/en/stable/getting-started.html#basic-usage
    
    pretty-printing files:
    https://tinydb.readthedocs.io/en/stable/usage.html#storage-middleware
    
    main API:
    https://tinydb.readthedocs.io/en/stable/api.html
    """
    f_db = 'db_test.json'
    if os.path.exists(f_db):
        os.remove(f_db)
    
    # create the database in-memory
    cal_db = db.TinyDB(storage=MemoryStorage)
    query = db.Query()
    
    # example: create a record in the "_default" table
    # cal_db.insert({'created_gmt':datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S")})
    
    # create a table with metadata (provenance) about this calibration file
    file_info = {
        "system" : "oppi",
        "created_gmt" : datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S"), 
        "input_table" : "/ORSIS3302DecoderForEnergy/raw"
        }
    cal_db.table('_file_info').insert(file_info)
    # tb_info.insert(file_info)

    # add some tables with calibration constants
    table = cal_db.table('energy_cal')
    for i in range(5):
        row = {'_run':f'{i:0>4d}', 'a':i+2, 'b':i/2, 'c':i*3, 
               'tscov':int(time.time() + i*60)}
        table.insert(row) 
        
    table = cal_db.table('trapEmax_cal')
    for i in range(3):
        row = {'_run':f'{i:0>4d}', 'a':i+2, 'b':i/2, 'c':i*3}
        table.insert(row) 
        
    # pretty-print the JSON database to file
    raw_db = cal_db.storage.read()
    write_pretty(raw_db, f_db)
    
    # show the file as-is on disk
    with open(f_db) as f:
        print(f.read()) 
    
    
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

    
def view_db():
    """
    """
    f_db = 'db_test.json'
    cal_db = db.TinyDB(f_db)
    query = db.Query()
    
    # search records -- match any existing key at any level
    # from tinydb import where # idk, doesn't work
    # print(cal_db.search(where('_run') == "0004"))
    
    # convert tables to pandas DataFrame
    tb = cal_db.table("energy_cal").all()
    df_cal = pd.DataFrame(tb)
    print(df_cal)
    
    
def modify_db():
    """
    need to be able to load an existing db file and make changes to it,
    and preserve the pretty printing
    """
    f_db = 'db_test.json'
    
    # load the db this way so the pretty formatting isn't changed
    cal_db = db.TinyDB(storage=MemoryStorage)
    with open(f_db) as f:
        raw_db = json.load(f)
        cal_db.storage.write(raw_db)
    query = db.Query()
    
    # example 1
    # update file_info with a new field
    tb_info = cal_db.table('_file_info')
    now = datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S")
    # tb_info.insert({'updated_gmt':now})
    tb_info.upsert({'updated_gmt':now}, query['updated_gmt'])
    
    # example 2
    # overwrite part of a calibration table using upsert
    tb_ecal = cal_db.table('energy_cal')
    for i in range(3):
        row = {'_run':f'{i:0>4d}', 'a':i+20, 'b':i/10, 'c':i*42,
               'tscov':int(time.time() + i*60)}
        tb_ecal.upsert(row, query['_run'] == f'{i:0>4d}') 
        # tb_ecal.upsert(row, query.idx == i) 

    # show in-memory state
    # pprint(cal_db.storage.read())
    
    # write to file
    write_pretty(cal_db.storage.read(), f_db)
    
    # show the file as-is on disk
    with open(f_db) as f:
        print(f.read()) 


if __name__=="__main__":
    main()