import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygama import DataSet
from pygama.analysis.histograms import *

ds = DataSet(runlist=[204], md='./runDB.json', tier_dir=tier_dir)

t2df = ds.get_t2df()


print(t2df.columns)
