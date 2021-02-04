import pandas as pd
import sys
import json
import os
import numpy as np
import scipy as sp
import scipy.optimize as opt
from scipy.signal import medfilt, find_peaks
from pygama import DataSet
import pygama.analysis.histograms as pgh
import pygama.utils as pgu
import pygama.analysis.peak_fitting as pga
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.style.use('style.mplstyle')

# this code uses the values from the results of fit_peaks.py to make a second-pass calibration.

if(len(sys.argv) != 2):
    print('Usage: non_linear_calibration.py [run number]')
    sys.exit()

with open("runDB.json") as f:
    runDB = json.load(f)
meta_dir = os.path.expandvars(runDB["meta_dir"])

def func(x, a, b, c):
    return a + b*x + c*(x**2)

df =  pd.read_hdf("{}/Spectrum_{}.hdf5".format(meta_dir,sys.argv[1]), key="df")

E_rough = [2607.42, 1761.51, 1459.07, 610.06, 353.56]
E_real = [2614.51, 1764.49, 1460.82, 609.32, 351.93]
errors = [0.06, 0.07, 0.01, 0.05, 0.01]

popt, pcov = opt.curve_fit(func, E_rough, E_real, sigma = errors)

df['e_cal'] = popt[0] + popt[1]*df['e_cal'] + popt[2]*(df['e_cal']**2)

print('a = '+str(popt[0])+' +/- '+str(pcov[0][0]))
print('b = '+str(popt[1])+' +/- '+str(pcov[1][1]))
print('c = '+str(popt[2])+' +/- '+str(pcov[2][2]))

df.to_hdf('{}/Spectrum_{}_2.hdf5'.format(meta_dir,sys.argv[1]), key='df', mode='w')

pks_lit_all = [238.6, 351.9, 511.0, 583.2, 609.3, 911.2, 969, 1120.3, 1460.8, 1764.5, 2614.5]
plt.axvline(x=238.6, ymin=0, ymax=30, color='red', linestyle='--', lw=1, zorder=1)
plt.axvline(x=351.9, ymin=0, ymax=30, color='aqua', linestyle='--', lw=1, zorder=1)
plt.axvline(x=511.0, ymin=0, ymax=30, color='darkgreen', linestyle='--', lw=1, zorder=1)
plt.axvline(x=583.2, ymin=0, ymax=30, color='darkorange', linestyle='--', lw=1, zorder=1)
plt.axvline(x=609.3, ymin=0, ymax=30, color='gray', linestyle='--', lw=1, zorder=1)
plt.axvline(x=911.2, ymin=0, ymax=30, color='brown', linestyle='--', lw=1, zorder=1)
plt.axvline(x=969.0, ymin=0, ymax=30, color='saddlebrown', linestyle='--', lw=1, zorder=1)
plt.axvline(x=1120.3, ymin=0, ymax=30, color='navy', linestyle='--', lw=1, zorder=1)
plt.axvline(x=1460.8, ymin=0, ymax=30, color='olive', linestyle='--', lw=1, zorder=1)
plt.axvline(x=1764.5, ymin=0, ymax=30, color='indigo', linestyle='--', lw=1, zorder=1)
plt.axvline(x=2614.5, ymin=0, ymax=30, color='limegreen', linestyle='--', lw=1, zorder=1)
n = np.array(df['e_cal'])
plt.hist(n, np.arange(0,9500,0.5), histtype='step', color = 'black', zorder=2, label='{} entries'.format(len(n)))
plt.xlim(0,4000)
plt.ylim(0,plt.ylim()[1])
plt.xlabel('Energy (keV)', ha='right', x=1.0)
plt.ylabel('Counts', ha='right', y=1.0)
E_cal = 'calibrated energy spectrum'
E_1 = 'E=238.6 keV (212Pb peak)'
E_2 = 'E=351.93 keV (214Pb peak)'
E_3 = 'E=511.0 keV (beta+ peak)'
E_4 = 'E=583.19 keV (208Tl peak)'
E_5 = 'E=609.32 keV (214Bi peak)*'
E_6 = 'E=911.2 keV (228Ac peak)'
E_7 = 'E=969 keV (228Ac peak)'
E_8 = 'E=1120.3 keV (214Bi peak)'
E_9 = 'E=1460.82 keV (40K peak)*'
E_10 = 'E=1764.49 keV (214Bi peak)'
E_11 = 'E=2614.51 keV (208Tl peak)'
colors = ['black', 'red', 'aqua', 'darkgreen', 'darkorange', 'gray', 'brown', 'saddlebrown', 'navy', 'olive', 'indigo', 'limegreen']
lines = [Line2D([0], [0], color=c) for c in colors]
labels = [E_cal, E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8, E_9, E_10, E_11]
#plt.title('Energy Spectrum')
plt.legend(lines, labels, frameon=True, loc='upper right', fontsize='x-small')
plt.tight_layout()
#plt.semilogy()
plt.show()

