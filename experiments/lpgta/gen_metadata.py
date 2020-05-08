import tinydb

f_md = 'LPGTA_r0019_phy_geds_calib.json'

calDB = ds.calDB
query = db.Query()
table = calDB.table("cal_pass3")

for dset in ds.ds_list:
    row = {"ds":dset, "lin":linfit, "slope":xF[0], "offset":xF[1]}
    table.upsert(row, query.ds == dset)
