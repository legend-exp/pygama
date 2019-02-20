#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

def main():

    fList = glob.glob("./frontenddata10sep2018/*")
    # print(fList)

    # -- load the dataframe --

    # https://chrisalbon.com/python/data_wrangling/pandas_dataframe_importing_csv/
    testFile = "0@DT5725 #2-14-1146_Data_run.csv"

    # get the number of samples, to manually specify the adc sample indexes
    # this is a dumb hack to get the caen file to work as a dataframe
    with open(testFile) as f:
        raw = f.readlines()
    wf = raw[1].split(";")
    wf = [int(v) for v in wf[4:]]
    ns = len(wf) # get the number of samples

    # df = pd.read_csv(testFile, delimiter=';')

    # print(df)

    # exit()

    # df = pd.read_csv(testFile, skiprows=1, delimiter=';', nrows=2, names=None)
    # df = pd.read_csv(testFile, delimiter=';', nrows=1)

    df = pd.read_csv(testFile, sep=';', header=None, skiprows=1)
    df.columns = ["TIMETAG","ENERGY","ENERGYSHORT","FLAGS"] + [i for i in range(ns)]

    print(df.shape)
    exit()

    # print(df)

    # exit()

    # -- print some interesting stuff --
    # print(df)
    # print(df.to_string()) # print the raw data
    # print(list(df)) # print the column names, method 1
    # print(list(df.columns.values))  # method 2
    # print(len(df)) # number of rows
    # print(df.keys())
    # ddf = df.to_dict() # can convert to a dict, but let's not do that

    # print(type(df["ENERGY"]))
    # print(df["ENERGY"].values)

    # measure the baseline variance
    # print(type(df[0]))
    # print(np.sqrt(df[0].var()))

    # print(df.loc["ENERGY"])

    # nparr[0,4:] similar to this,
    # wf = df.iloc[0,4:].max # make a Series from row 0, columns 4 -> end
    # wf.plot()

    # df.iloc[0,4:].plot() # plot the wf directly from the df

    for i in range(50): # plot 50 wf's on top of each other
        df.iloc[i, 4:].plot()

    # wf_np = wf.values # convert to a numpy array
    # ts = np.arange(len(wf_np))
    # plt.plot(ts, wf_np, "-")

    plt.show()


if __name__=="__main__":
    main()