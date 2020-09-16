import pandas as pd

df = pd.read_hdf('../LPGTA_fileDB.h5')

dates = df.groupby('YYYYmmdd').groups.keys()

for date in dates: print(date)
