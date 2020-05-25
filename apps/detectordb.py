import json
import sys
import os
import argparse

# Examples:
#   python detectordbparser.py -h
#   python detectordbparser.py
#   python detectordbparser.py -c test
#   python detectordbparser.py -d I01239A
#   python detectordbparser.py -k dep_voltage_in_V
#   python detectordbparser.py -d I01239A -k dep_voltage_in_V

parser = argparse.ArgumentParser()
parser.add_argument('-d', help='specific detector name (e.g. V001239A)')
parser.add_argument('-k', help='key to be parsed (e.g. mass_in_g)')
parser.add_argument('-p', help='path to detector json files (e.g. /path/to/detector_database). Default is ./detectors')
parser.add_argument('-c', help='create new detector [ged-spm]')
args = parser.parse_args()

if len(sys.argv) == 1:
  parser.print_help()
  sys.exit()

print('')
print(' Detector database parser ')
print('')

detid  = args.d
key    = args.k
path   = args.p
create = args.c

#First letter of detid must be in this list
types=['B','C','V','P','S']
if detid is not None:
  t = [e for e in types if e in detid][0]

if(path is None): path = './detectors'

if(create is not None):
  if(detid is None):
    print("You must specify a detector ID (-d <detid>) to create a new detector")
  
  if(t == ''):
    print("Detector ID %s must start with %s" %(detid,types))
    sys.exit()
  else: print("Found detector type",t)
  
  filename = path + '/' + detid + '.json'
  if(os.path.isfile(filename)):
    print(path+'/'+detid+'.json already exist. Exit.')
    print('')
    sys.exit()

  if(t != 'S'):
    data = {
      'name' : detid,
      'type' : '',
      'production':{
        'manufacturer': '',
        'order': '',
        'serialno': '',
        'crystal': '',
        'slice': '',
        'enrichment': '',
        'reprocessing': '',
        'dep_voltage_in_V': '',
        'rec_voltage_in_V': '',
        'impcc':{
          'value_in_1e9cc':[0,0],
          'dist_from_contact_in_mm':[0,0]
        },
        'delivered':''
      },
      'geometry':{
        'mass_in_g':'',
        'height_in_mm':'',
        'diameter_in_mm':'',
        'well':{
          'depth_in_mm'    : '',
          'radius_in_mm'   : ''
        },
        'bulletization':'none',
        'groove':{
          'radius_in_mm' : '',
          'depth_in_mm'  : '',
          'width_in_mm'  : ''
        },
        'contact':{
          'radius_in_mm' : '',
          'depth_in_mm' : ''
        },
        'taper':{
          'inner':{
            'radius_in_mm'  : 'none',
            'height_in_mm'  : 'none'
          },
          'outer':{
            'radius_in_mm'  : 'none',
            'height_in_mm'  : 'none'
          }
        },
        'dl_thickness_in_mm':''
      },
      'characterization':{
        'manufacturer':{
          'dep_voltage_in_V': '',
          'op_voltage_in_V': '',
          '57co_fep_res_in_keV': '',
          '60co_fep_res_in_keV': '',
          'dl_thickness_in_mm': ''
        },
        'l200_site':{
          'data':'',
          'daq':'',
          'elog':'https://elog.legend-exp.org/site/?Det.+ID=name',
          'res':{
            'cofep_in_keV':'',
            'tlfep_in_keV':''
          },
          'sf':{
            'tldep_in_pc'  : '',
            'qbb_in_pc'    : '',
            'tlsep_in_pc'  : '',
            'tlfep_in_pc'  : ''
          }
        }
      }
    }
  else:
    data = {
      'name' : detid,
      'type' : create,
      'production':{
        'manufacturer': '',
        'order': '',
        'serialno': '',
        'date':''
      },
      'characterization':{
        'bdv_in_V': [0,0],
        'dcr_in_Hz': [0,0],
        'aps': [0,0],
        'xtalk': [0,0],
        'gain': [0,0],
        'other': ''
      },
      'other':{
      }
    }

  with open(filename, 'w') as outfile:
    json.dump(data, outfile, indent=2)

  print(filename + ' successfully created.')
  print('')
  sys.exit()

dets = []

if(detid is None):
  try:
    files = os.listdir(path)
    for file in files:
      det, ext = os.path.splitext(file)
      if(ext=='.json'): dets.append(det)
  except IOError:
    print('Error: ',path,' does not appear to exist in:')
    print(dets)
    sys.exit()
else:
  dets.append(detid)

if(len(dets) == 0):
  print('No detector database found in ',path)
  sys.exit()

for det in dets:
  filename = path + '/' + det + '.json'
  try:
    with open(filename) as json_file:
      data = json.load(json_file)
      value = 'None'
      if(key is None):
        for entry in data:
          if(isinstance(data[entry],dict)):
            print(entry)
            for subentry in data[entry]:
              if(isinstance(data[entry][subentry],dict)):
                print('\t',subentry)
                for subsubentry in data[entry][subentry]:
                  if(isinstance(data[entry][subentry][subsubentry],dict)):
                    print('\t \t',subsubentry)
                    for subsubsubentry in data[entry][subentry][subsubentry]:
                      print('\t \t \t',data[entry][subentry][subsubentry])
                  else:
                    print('\t \t \t',subsubentry,'->',data[entry][subentry][subsubentry])
              else:
                print('\t',subentry,'->',data[entry][subentry])
          else:
            print(entry,'->',data[entry])
        continue
      for entry in data:
        if(key==entry): value = data[entry]
        elif(isinstance(data[entry],dict)):
          for subentry in data[entry]:
            if(key==subentry): value = data[entry][subentry]
            elif(isinstance(data[entry][subentry],dict)):
              for subsubentry in data[entry][subentry]:
                if(key==subsubentry): value = data[entry][subentry][subsubentry]
                elif(isinstance(data[entry][subentry][subsubentry],dict)):
                  for subsubsubentry in data[entry][subentry][subsubentry]:
                    if(key==subsubsubentry): value = data[entry][subentry][subsubentry][subsubsubentry]
      if(value != 'None'): print(data["name"],value)
  except IOError:
    print('Error: ',filename,' does not appear to exist in')
    print(os.listdir(path))

print('')
