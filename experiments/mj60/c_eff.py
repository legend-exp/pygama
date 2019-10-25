import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pygama import DataSet
from pygama.utils import set_plot_style
from pygama.analysis.histograms import *

table_1 = pd.read_csv('./table_1.txt')
table_2 = pd.read_csv('./table_2.txt')

# Assuming a 1 pF feedback capacitance, the value of C_eff
# will be in units of pF, and I will neglect the factor of 1
# as that does not change the magnitude of the calculation

C_eff_1 = (table_1['V1_out'][0:19] / 1000) / table_1['V_HV'][0:19]
C_eff_2 = table_2['V1_out'] / table_2['V_HV']

voltage_1 = table_1['V_HV'][0:19]
voltage_2 = table_2['V_HV']

plt.plot(voltage_1, C_eff_1, c='xkcd:grapefruit', label='Set 1: 50mVpp pulser amplitude')
plt.plot(voltage_2, C_eff_2, c='xkcd:vivid blue', label='Set 2: 200 mVpp pulser amplitude')
plt.xlabel('Input Voltage (V)')
plt.ylabel('Effective Capacitance (pF)')
plt.legend()

plt.show()
