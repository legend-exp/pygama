import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from pygama import DataSet

with open("runDB.json") as f:
    runDB = json.load(f)
tier_dir = runDB["tier_dir"]

ds0 = DataSet(runlist=[554], md='./runDB.json', tier_dir=tier_dir)
t2df_0 = ds0.get_t2df()
ds1 = DataSet(runlist=[555], md='./runDB.json', tier_dir=tier_dir)
t2df_1 = ds1.get_t2df()
ds2 = DataSet(runlist=[556], md='./runDB.json', tier_dir=tier_dir)
t2df_2 = ds2.get_t2df()

e_0 = t2df_0["energy"]
e_1 = t2df_1["energy"]
e_2 = t2df_2["energy"]

e_full = [0, 3.3e6]
e_pks = [1.2e6, 2.6e6]
e_K = [1.3e6, 1.36e6]
e_T = [2.35e6, 2.42e6]

h_0,edg_0 = np.histogram(e_0, bins=5000, range=e_full)
x_0 = (edg_0[:-1] + edg_0[1:])/2

# h_0_K,edg_0_K = np.histogram(e_0, bin=500, range=e_K)
# x_0_K = (edg_0_K[:-1] + edg_0_K[1:])/2
#
# h_0_T,edg_0_T = np.histogram(e_0, bin=750, range=e_T)
# x_0_T = (edg_0_T[:-1] + edg_0_T[1:])/2

h_1,edg_1 = np.histogram(e_1, bins=5000, range=e_full)
x_1 = (edg_1[:-1] + edg_1[1:])/2

# h_1_K,edg_1_K = np.histogram(e_1, bin=500, range=e_K)
# x_1_K = (edg_1_K[:-1] + edg_1_K[1:])/2
#
# h_1_T,edg_1_T = np.histogram(e_1, bin=750, range=e_T)
# x_1_T = (edg_1_T[:-1] + edg_1_T[1:])/2

h_2,edg_2 = np.histogram(e_2, bins=5000, range=e_full)
x_2 = (edg_2[:-1] + edg_2[1:])/2

# h_2_K,edg_2_K = np.histogram(e_2, bin=500, range=e_K)
# x_2_K = (edg_2_K[:-1] + edg_2_K[1:])/2
#
# h_2_T,edg_2_T = np.histogram(e_2, bin=750, range=e_T)
# x_2_T = (edg_2_T[:-1] + edg_2_T[1:])/2

plt.semilogy(x_0, h_0, c='0.5', ls='steps', label='Preamp 0')
plt.semilogy(x_1, h_1, c='k', ls='steps', label='Preamp 1')
plt.semilogy(x_2, h_2, c='c', ls='steps', label='Preamp 2')

def gauss(x, *p):
    A, mu, sigma, B, C=p
    return A*np.exp(-(x-mu)**2/(2*(sigma**2))) + B*np.exp(C*x)

# p0K = [...]
# popt0K,pcov0K = curve_fit(gauss, x_0_K, h_0_K, p0=p0K)
# p0T = [...]
# popt0T,pcov0T = curve_fit(gauss, x_0_T, h_0_T, p0=p0T)
# p1K = [...]
# popt1K,pcov1K = curve_fit(gauss, x_1_K, h_1_K, p0=p1K)
# p1T = [...]
# popt1T,pcov1T = curve_fit(gauss, x_1_T, h_1_T, p0=p1T)
# p2K = [...]
# popt2K,pcov2K = curve_fit(gauss, x_2_K, h_2_K, p0=p2K)
# p2T = [...]
# popt2T,pcov2T = curve_fit(gauss, x_2_T, h_2_T, p0=p2T)
#
# h_0_fit_K = gauss(x_0_K, *popt0K)
# h_0_fit_T = gauss(x_0_T, *popt0T)
# h_1_fit_K = gauss(x_1_K, *popt1K)
# h_1_fit_T = gauss(x_1_T, *popt1T)
# h_2_fit_K = gauss(x_2_K, *popt2K)
# h_2_fit_T = gauss(x_2_T, *popt2T)
#
# plt.semilogy(x_0_K, h_0_fit_K, c='xkcd:bright red', label='K, Preamp 0')
# plt.semilogy(x_0_T, h_0_fit_T, c='xkcd:bright orange', label='T, Preamp 0')
# plt.semilogy(x_1_K, h_1_fit_K, c='xkcd:bright yellow', label='K, Preamp 1')
# plt.semilogy(x_1_T, h_1_fit_T, c='xkcd:electric green', label='T, Preamp 1')
# plt.semilogy(x_2_K, h_2_fit_K, c='xkcd:true blue', label='K, Preamp 2')
# plt.semilogy(x_2_T, h_2_fit_T, c='xkcd:electric purple', label='T, Preamp 2')

plt.xlabel("Energy (Uncal)")
plt.ylabel("Counts")
plt.legend(loc=1)
plt.tight_layout()

plt.show()
