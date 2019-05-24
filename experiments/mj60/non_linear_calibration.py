import pandas as pd
import sys
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

def func(x, a, b, c):
    return a + b*x + c*(x**2)

df =  pd.read_hdf("Spectrum_203.hdf5", key="df")

E_rough = [510.22, 582.89, 910.82, 968.44, 1460.96, 2613.78]
E_real = [511.0, 583.2, 911.2, 969.0, 1460.8, 2614.5]
errors = [0.20, 0.20, 0.27, 0.28, 0.59, 0.25]

popt, pcov = opt.curve_fit(func, E_rough, E_real, sigma = errors)

df['e_real'] = popt[0] + popt[1]*df['e_cal'] + popt[2]*(df['e_cal']**2)

df.to_hdf('Spectrum_203_Calibrated.hdf5', key='df', mode='w')

pks_lit_all = [238.6, 338.3, 463.0, 511,0, 583.2, 727.3, 794.9, 860.6, 911.2, 969, 1460.8, 1592.5, 2103.5, 2614.5]
plt.axvline(x=238.6, ymin=0, ymax=30, color='red', linestyle='--', lw=1, zorder=1)
plt.axvline(x=338.3, ymin=0, ymax=30, color='aqua', linestyle='--', lw=1, zorder=1)
plt.axvline(x=463.0, ymin=0, ymax=30, color='teal', linestyle='--', lw=1, zorder=1)
plt.axvline(x=511.0, ymin=0, ymax=30, color='darkgreen', linestyle='--', lw=1, zorder=1)
plt.axvline(x=583.2, ymin=0, ymax=30, color='darkorange', linestyle='--', lw=1, zorder=1)
plt.axvline(x=727.3, ymin=0, ymax=30, color='gray', linestyle='--', lw=1, zorder=1)
plt.axvline(x=794.9, ymin=0, ymax=30, color='brown', linestyle='--', lw=1, zorder=1)
plt.axvline(x=860.6, ymin=0, ymax=30, color='purple', linestyle='--', lw=1, zorder=1)
plt.axvline(x=911.2, ymin=0, ymax=30, color='fuchsia', linestyle='--', lw=1, zorder=1)
plt.axvline(x=969.0, ymin=0, ymax=30, color='saddlebrown', linestyle='--', lw=1, zorder=1)
plt.axvline(x=1460.8, ymin=0, ymax=30, color='navy', linestyle='--', lw=1, zorder=1)
plt.axvline(x=1592.5, ymin=0, ymax=30, color='limegreen', linestyle='--', lw=1, zorder=1)
plt.axvline(x=2103.5, ymin=0, ymax=30, color='olive', linestyle='--', lw=1, zorder=1)
plt.axvline(x=2614.5, ymin=0, ymax=30, color='indigo', linestyle='--', lw=1, zorder=1)
n = np.array(df['e_real'])
plt.hist(n, np.arange(0,9500,0.5), histtype='step', color = 'black', zorder=2, label='{} entries'.format(len(n)))
plt.xlim(0,4000)
plt.ylim(0,plt.ylim()[1])
plt.xlabel('Energy (keV)', ha='right', x=1.0)
plt.ylabel('Counts', ha='right', y=1.0)
E_cal = ('calibrated energy spectrum, run 203')
E_1 = 'E=238.6 keV (212Pb peak)*'
E_2 = 'E=338.3 keV (228Ac peak)'
E_3 = 'E=463.0 keV (228Ac peak)'
E_4 = 'E=511.0 keV (beta+ peak)'
E_5 = 'E=583.2 keV (208Tl peak)*'
E_6 = 'E=727.3 keV (212Bi peak)'
E_7 = 'E=794.9 keV (228Ac peak)'
E_8 = 'E=860.6 keV (208Tl peak)'
E_9 = 'E=911.2 keV (228Ac peak)'
E_10 = 'E=969 keV (228Ac peak)'
E_11 = 'E=1460.8 keV (40K peak)'
E_12 = 'E=1592.5 keV (208Tl DE)'
E_13 = 'E=2103.5 keV (208Tl SE)'
E_14 = 'E=2614.5 keV (208Tl peak)'
colors = ['black', 'red', 'aqua', 'teal', 'darkgreen', 'darkorange', 'gray', 'brown', 'purple', 'fuchsia', 'saddlebrown', 'navy', 'limegreen', 'olive', 'indigo']
lines = [Line2D([0], [0], color=c) for c in colors]
labels = [E_cal, E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8, E_9, E_10, E_11, E_12, E_13, E_14]
plt.legend(lines, labels, frameon=True, loc='upper right', fontsize='x-small')
plt.tight_layout()
#plt.semilogy()
plt.show()

