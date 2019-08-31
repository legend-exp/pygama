#!/usr/bin/env python3
import os, time, json
import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from scipy.integrate import quad
import tinydb as db
import argparse

from matplotlib.lines import Line2D


c_f = 1e-12
data = np.genfromtxt("./V_HV.txt") #test1
data1 = np.genfromtxt('./V_HV2.txt') #test4
data2 = np.genfromtxt('./V_HV3.txt') #test5

v_hv = np.asarray(data[1:,0])
v_out = np.asarray(data[1:,2]) * 1e-3
v_hv1 = np.asarray(data1[1:,0])
v_out1 = np.asarray(data1[1:,2]) * 1e-3
v_hv2 = np.asarray(data2[1:,0])
v_out2 = np.asarray(data2[1:,2]) * 1e-3

v_in = 100e-3
c_eff = c_f * v_out / v_in
c_eff1 = c_f * v_out1 / v_in
c_eff2 = c_f * v_out2 / v_in

plt.scatter(v_hv, c_eff, label='9/30/18')
plt.scatter(v_hv1, c_eff1, label='11/26/18')
plt.scatter(v_hv2, c_eff2, label='12/10/18')
plt.title('C_eff of MJ60')
plt.ylim(0,1.1e-11)
plt.legend()
plt.show()
exit()
