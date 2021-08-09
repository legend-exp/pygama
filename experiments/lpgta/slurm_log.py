# import python modules
import datetime
import numpy as np
import os
import sys

# check the arguments
if len(sys.argv) < 2:
    print(f'usage: {sys.argv[0]} [log files]')
    sys.exit()

# the duration per file
durations = []

# loop through the files
for file in sys.argv[1:]:
    # read in the file
    with open(file, 'r') as f:
        # extract the lines from the file
        lines = f.readlines()
        
        # extract the timestamps
        ts_beg = datetime.datetime.strptime(lines[ 0], '%a %b %d %H:%M:%S %Z %Y\n')
        ts_end = datetime.datetime.strptime(lines[-1], '%a %b %d %H:%M:%S %Z %Y\n')

        # include the duration
        durations.append((ts_end - ts_beg).total_seconds())
    
# status printout
print(f'Total processing time is {sum(durations)        / 3600:.2f} hr.' )
print(f'Longest job-node time is {max(durations)        /   60:.2f} min.')
print(f'Average job-node time is {np.average(durations) /   60:.2f} min.')
