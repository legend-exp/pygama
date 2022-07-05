import sys, os.path
from parse import parse
from datetime import datetime

if len(sys.argv) < 2:
    print(f'usage: {sys.argv[0]} [log_file]')
    sys.exit()

size_sum_GB = 0
last_line = None
start_datetime = None
with open(sys.argv[1]) as f:
    lines = f.readlines()
    for line in lines:
        if last_line is None:
            start_datetime = datetime.strptime(line[:-1], '%a %b %d %H:%M:%S %Z %Y')
        last_line = line
        parse_data = parse('Opened file {}', line)
        if parse_data is not None:
            filename = parse_data[0]
            size_sum_GB += os.path.getsize(filename) / (1024**3)

stop_datetime = datetime.strptime(last_line[:-1], '%a %b %d %H:%M:%S %Z %Y')
duration = stop_datetime - start_datetime
rate_MBps = size_sum_GB/duration.total_seconds()*1024

print(f'{sys.argv[1]}: {size_sum_GB:g} GB processed in {duration.total_seconds()/3600:g} hours = {rate_MBps:g} MB/s')
