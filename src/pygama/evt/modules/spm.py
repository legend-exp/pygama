import numpy as np
import pygama.lgdo.lh5_store as store

"""
Module for special event level routines for SiPMs

functions must take as the first 3 args in order:
- path to the hit file
- path to the dsp file
- list of channels processed
additional parameters are free to the user and need to be defined in the JSON
"""
#get LAr energy per event over all channels
def get_energy(f_hit,f_dsp,chs,lim,trgr,tdefault,tmin,tmax):
    trig = np.where(np.isnan(trgr),tdefault,trgr)  
    tmi = trig - tmin
    tma = trig + tmax
    sum = np.zeros(len(trig))
    for ch in chs:
        df =store.load_nda(f_hit, ["energy_in_pe","is_valid_hit",'trigger_pos'],ch+"/hit/")
        mask =  (df["trigger_pos"]<tma[:,None]/16) & (df["trigger_pos"]>tmi[:,None]/16) & (df["energy_in_pe"] > lim)
        pes=df["energy_in_pe"]
        pes= np.where(np.isnan(pes), 0, pes)
        pes= np.where(mask,pes,0)
        chsum= np.nansum(pes, axis=1)
        sum = sum + chsum
    return sum

#get LAr majority per event over all channels
def get_majority(f_hit,f_dsp,chs,lim,trgr,tdefault,tmin,tmax):  
    trig = np.where(np.isnan(trgr),tdefault,trgr)
    tmi = trig - tmin
    tma = trig + tmax
    maj = np.zeros(len(trig))
    for ch in chs:
        df =store.load_nda(f_hit, ["energy_in_pe","is_valid_hit",'trigger_pos'],ch+"/hit/")
        mask =  (df["trigger_pos"]<tma[:,None]/16) & (df["trigger_pos"]>tmi[:,None]/16) & (df["energy_in_pe"] > lim)
        pes=df["energy_in_pe"]
        pes= np.where(np.isnan(pes), 0, pes)
        pes= np.where(mask,pes,0)
        chsum= np.nansum(pes, axis=1)
        chmaj = np.where(chsum>lim,1,0)
        maj = maj + chmaj
    return maj