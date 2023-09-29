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

def get_etc(f_hit,f_dsp,chs,lim,trgr,tdefault,tmin,tmax,swin,trail):
        predf = store.load_nda(f_hit, ["energy_in_pe",'timestamp'],chs[0]+"/hit/")
    
        peshape = (predf["energy_in_pe"]).shape
        # 1D = channel, 2D = event num, 3D = array per event
        pes=np.zeros([len(chs),peshape[0],peshape[1]])
        times = np.zeros([len(chs),peshape[0],peshape[1]])

        tge = np.where(np.isnan(trgr),tdefault,trgr)
        tmi = tge - tmin
        tma = tge + tmax
        for i in range(len(chs)):
            df =store.load_nda(f_hit, ["energy_in_pe",'trigger_pos','timestamp'],chs[i]+"/hit/")
            mask =  (df["trigger_pos"]<tma[:,None]/16) & (df["trigger_pos"]>tmi[:,None]/16) & (df["energy_in_pe"] > lim)
            pe=df["energy_in_pe"]
            time = df["trigger_pos"]*16
        
            pe= np.where(mask,pe,np.nan)
            time= np.where(mask,time,np.nan)

            pes[i] = pe
            times[i] = time
        
        outi = None
        if trail >0:
            t1d = np.nanmin(times,axis=(0,2))
            if trail == 2: t1d[t1d>tge] = tge[t1d>tge]
            tt = t1d[:,None]
            outi = np.where(np.nansum(np.where((times >= tt),pes,0),axis=(0,2)) > 0,
                            np.nansum(np.where((times >= tt) & (times < tt+swin),pes,0),axis=(0,2))/np.nansum(np.where((times >= tt),pes,0),axis=(0,2)),
                            np.nansum(np.where((times >= tt),pes,0),axis=(0,2)))
            return outi
            
        else:
            outi = np.where(np.nansum(pes,axis=(0,2)) > 0,
                            np.nansum(np.where((times >= tge[:,None]) & (times <= tge[:,None]+swin),pes,0),axis=(0,2))/np.nansum(np.where((times >= tge[:,None]),pes,0),axis=(0,2)),
                            np.nansum(pes,axis=(0,2)))
            return outi