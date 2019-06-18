import os
import tinydb as db
import pandas as pd
from pprint import pprint
from collections import Counter

def main():
    """
    using the TinyDB module,
    show an example of filling a fake database file,
    then give a couple of examples on accessing it.
    
    important methods to cover:
    query (i.e. search), insert, update, table
    
    read the docs:
    https://tinydb.readthedocs.io/en/latest/api.html
    """
    fdb = "testDB.json"

    create_db(fdb)
    parse_db(fdb)
    update_db(fdb)
    
    
def create_db(fdb):
    """
    overwrite any existing testDB file,
    and create a small DB with a few different tables.
    NOTE: tables should be the primary structure we use, but we can also
    insert single documents as needed.
    """
    # access the DB file.  this routine creates a new one every time
    if os.path.exists(fdb):
        os.remove(fdb)

    testDB = db.TinyDB(fdb)
    query = db.Query()

    # make a test record that will be added to the "_default" table.
    test_record = {"p1cal":{"ds":20, "cal":0.4056}}
    
    # use 'insert' the first time
    testDB.insert(test_record) 
    
    # use 'update' on subsequent times
    test_record = {"p1cal":{"ds":20, "cal1":0.4056, "cal2":123}}
    testDB.update(test_record)
    
    # search the DB for this key (returns a list)
    # access is through the first key, which is why it's nested
    rec_found = testDB.search(db.where("p1cal"))
    if len(rec_found) > 1:
        print("warning, multiple entries present")
    else:
        rec_found = rec_found[0]
        
    # ** Make a new table to hold a set of records. This is encouraged ** 
    
    # creates new if it doesn't exist, or appends if it does exist.
    # give it a key that is memorable with no spaces, commas, or periods.
    table = testDB.table("cal_pass1") 
    
    # be careful, this will add duplicates if you call it multiple times
    table.insert({"ds":3, "p1cal":0.40670})
    table.insert({"ds":4, "p1cal":0.51781})
    docid = table.insert({"ds":4, "p1cal":0.51781}) # <- avoid this mistake!
    table.remove(doc_ids=[docid]) 
    
    
def parse_db(fdb):
    """
    access our test database and show a few examples of parsing.
    """
    testDB = db.TinyDB(fdb)
    query = db.Query()
    
    print("Method 1: print all tables and values in the file")
    for tb in testDB.tables():
        print("\nTable:", tb)
        print(testDB.table(tb).all())
        
    print("\nMethod 2: explore keys and num. entries only")
    for tb in testDB.tables():
        contents = testDB.table(tb).all()
        schema = Counter(frozenset(doc.keys()) for doc in contents)
        print(f"table: {tb}  ({sum(schema.values())} documents)")
        for fields, count in schema.items():
            print(f"  documents: {count}\n  keys:")
            print('\n'.join(f'    {field}' for field in fields))

    print("\nMethod 3: search for a particular document")
    results = testDB.search(query.p1cal.ds == 20)
    print(results)
    
    print("\nMethod 4: dump table to pandas DataFrame")
    table = testDB.table("cal_pass1")
    vals = table.all()
    df = pd.DataFrame(vals) # <<---- omg awesome
    print(df)
    
    
def update_db(fdb):
    """
    appending to tinyDB tables is tricky, you can get duplicate values.
    show a couple of examples of handling this.
    """
    print("\n Updating table example")
    
    testDB = db.TinyDB(fdb)
    query = db.Query()
    
    table = testDB.table("cal_pass1")
    
    # create some duplicated entries
    table.insert({"ds":4, "p1cal":0.2345})
    table.insert({"ds":4, "p1cal":0.2345})
    
    new_1 = {"ds":4, "p1cal":0.3456}
    table.upsert(new_1, query.cal_pass1.ds == 4)
    
    new_2 = {"ds":5, "p1cal":0.6789}
    table.upsert(new_2, query.cal_pass1.ds == 5)
    
    new_3 = {"ds":6, "p1cal":0.6789, "p2cal":0.5678}
    table.upsert(new_3, query.cal_pass1.ds == 6)

    # check again
    print(pd.DataFrame(table.all()))
    
    # use a query to find the docids of the duplicated entries
    results = table.search(query.ds == 4)

    # remove duplicates and only keep the last (most recent) one
    ids = []
    for r in results:
        vals = dict(r)
        print(type(r), r.doc_id, vals)
        ids.append(r.doc_id)
    ids.pop()
    table.remove(doc_ids=ids)

    # check the duplicates are gone
    print(pd.DataFrame(table.all()))
    
    
if __name__=="__main__":
    main()