import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pygama import DataSet
from pygama.utils import set_plot_style
from pygama.analysis.histograms import *

table_1 = pd.read_csv('./table_1.txt')
table_2 = pd.read_csv('./table_2.txt')
table_3 = pd.read_csv('./biasingIIItable.txt')
table_4 = pd.read_csv('./biasingIVtable.txt')
table_5 = pd.read_csv('./biasingVtable.txt')

# Assuming a 1 pF feedback capacitance, the value of C_eff
# will be in units of pF, and I will neglect the factor of 1
# as that does not change the magnitude of the calculation

# C_eff_1 = (table_1['V1_out'][0:19] / 1000) / table_1['V_HV'][0:19]
# C_eff_2 = table_2['V1_out'] / table_2['V_HV']
# C_eff_3 = (table_3['V_out'] / 1000) / table_3['V_HV']
# C_eff_4 = (table_4['V_out'] / 1000) / table_4['V_HV']
# C_eff_5 = (table_5['V_out'] / 1000) / table_5['V_HV']

C_eff_1 = table_1['V1_out'][0:19] / 50
C_eff_2 = (table_2['V1_out'] * 1000) / 200
C_eff_3 = table_3['V_out'] / 100
C_eff_4 = table_4['V_out'] / 100
C_eff_5 = table_5['V_out'] / 100

voltage_1 = table_1['V_HV'][0:19]
voltage_2 = table_2['V_HV']
voltage_3 = table_3['V_HV']
voltage_4 = table_4['V_HV']
voltage_5 = table_5['V_HV']

plt.plot(voltage_1, C_eff_1, c='xkcd:grapefruit', marker='.', label='10/15/2019, Set 1: 50mVpp pulser amplitude')
plt.plot(voltage_2, C_eff_2, c='xkcd:vivid blue', marker='.', label='10/15/2019, Set 2: 200 mVpp pulser amplitude')
plt.plot(voltage_3, C_eff_3, c='xkcd:mint', marker='.', label='9/30/2018, 100mVpp pulser amplitude')
plt.plot(voltage_4, C_eff_4, c='xkcd:neon pink', marker='.', label='11/26/2018, 100 mVpp pulser amplitude')
plt.plot(voltage_5, C_eff_5, c='xkcd:baby pink', marker='.', label='12/10/2018, 100 mVpp pulser amplitude')
plt.xlabel('Input Voltage (V)')
plt.ylabel('Effective Capacitance (pF)')
plt.tight_layout()
plt.legend(fontsize='8')

plt.show()

# print(C_eff_2)
