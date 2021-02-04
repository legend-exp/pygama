import json
import sys
import os
import argparse
import glob

# Examples:
#   python detectordbparser.py -h
#   python detectordbparser.py
#   python detectordbparser.py -c test.db
#   python detectordbparser.py -rc test.db
#   python detectordbparser.py -s test.db
#   python detectordbparser.py -s test.db -ch all
#   python detectordbparser.py -s test.db -ch all -k cc4ch
#   python detectordbparser.py -s test.db -ch all -k cc4ch -id 1

parser = argparse.ArgumentParser()
parser.add_argument('-ch',help='channel to be parsed (e.g. all or 1)')
parser.add_argument('-k' ,help='key to be parsed (e.g. pzs)')
parser.add_argument('-bk',help='parse specific block (e.g. ge-string-1)')
parser.add_argument('-id',help='print detector ID')
parser.add_argument('-s', help='show channels map (e.g. channel-map-run0001.db)')
args = parser.parse_args()

showch = "all" # By default show all channels
if args.ch is not None:
  showch = args.ch

print('')
print(' FlashCam channels map parser ')
print('')

###############################
# CHECK THE DETECTOR DATABASE #
###############################

path = './detectors/'
detdb=[]
nfailed=0

for file in glob.glob(path+'*.json'):
  try:
    with open(file) as json_file:
      #print("    Detector file %s opened" % (file))
      data = json.load(json_file)
      detdb.append(data['name'])
  except IOError:
    print('Error: ',file,' does not appear to exist.')

########################
# SHOW THE CHANNEL MAP #
########################

if(args.s is not None):
  filename =  './' + args.s
  
  try:
    with open(filename) as json_file:
      print("    Channel map %s opened" % filename)
      data = json.load(json_file)
    
    if args.bk is not None:
      for entry in data:
        if data[entry]["block"] == args.bk: print(entry,data[entry])
      sys.exit()
    if "all" in showch:
      for entry in data:
        if args.k is None: print(entry,data[entry])
        elif args.k == "all":
          print(entry,end=" ")
          for key in data[entry]: print(data[entry][key],end=" ")
          print("")
        else:
          if args.id is not None: print(entry,data[entry]['detid'],data[entry][args.k])
          else: print(entry,data[entry][args.k])
    else:
      print(args.ch,data[args.ch])
    
    print('')
    print('Check channel-map consistency with detector database:')
    print('')
    
    for entry in data:
      if (data[entry]['detid'] == 'none'): continue
      elif data[entry]['detid'] not in detdb:
        print("Detector '%s' not found in the detector database" % data[entry]['detid'])
        nfailed=nfailed+1
        
    if nfailed > 0:
      print("%d channels not found in the detector database." % nfailed)
      print("Check in", detdb)
      
  except IOError:
    print('Error: ',filename,' does not appear to exist.')

  print('')
  sys.exit()
