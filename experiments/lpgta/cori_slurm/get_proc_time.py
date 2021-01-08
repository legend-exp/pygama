import sys, os.path
from datetime import datetime, timedelta

if len(sys.argv) < 2:
    print(f'usage: {sys.argv[0]} [log_files]')
    sys.exit()


last_line = None
start_datetime = None
duration = timedelta(0)
for log_file in sys.argv[1:]:
    with open(log_file) as f:
        lines = f.readlines()
        for line in lines:
            if last_line is None:
                start_datetime = datetime.strptime(line[:-1], '%a %b %d %H:%M:%S %Z %Y')
            last_line = line
    stop_datetime = datetime.strptime(last_line[:-1], '%a %b %d %H:%M:%S %Z %Y')
    duration += stop_datetime - start_datetime

print(f'Total duration: {duration.total_seconds()/3600:g} hours')
