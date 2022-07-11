from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import math
#import similaritymeasures
#from scipy.spatial import distance
#from matplotlib.widgets import Slider, Button, TextBox
#import seaborn as sns;
import pandas as pd


#RISING AND QUENCHING
def irl1(t,ifi,tq,ti,td):
    return ifi*(1-((tq-ti)/(td-ti))*math.exp(-t/ti)+ ((tq-td)/(td-ti))*math.exp(-t/td))


def ivi(nf,vov,rq,rd,rl):
    i=nf*(vov/(rq+rd+nf*rl))
    return i

#rising of avalanch tau
def ti(a1,a2):
    return (2*a2)/(a1+math.sqrt(a1*a1-4*a2))

#quenching of avalanch tau
def td(a1,a2):
    return (2*a2)/(a1-math.sqrt(a1*a1-4*a2))

#alpha 1
def a1(rd,cd,rq,cq,nf,rl,ceq):
    oben= rd*cd*(rq+nf*rl)+rq*cq*(rd+nf*rl)+ceq*rl*(rq+rd)
    unten=rq+rd+nf*rl
    return oben/unten

#alpha 2
def a2(rd,cd,rq,cq,nf,rl,ceq):
    a = (rd*rq*rl)/(rq+rd+nf*rl)
    b = ceq*(cd+cq)+nf*cd*cq
    return a*b

#################################
#DIODE current
def ird(t,ifi,td,td2,ti,ti2):
    a = (((td2-ti)*(ti2-ti))/(ti*(td-ti)))*math.exp(-1*t/ti)
    b = (((td2-td)*(ti2-td))/(td*(td-ti)))*math.exp(-1*t/td)
    return ifi*(1+a-b)

#rising of diode tau
def ti2(a1d,a2d):
    return (2*a2d)/(a1d+math.sqrt(a1d*a1d-4*a2d))

#fall of diode tau
def td2(a1d,a2d):
    return (2*a2d)/(a1d-math.sqrt(a1d*a1d-4*a2d))

#alpha 1 diode = rd->inf alpha 1
def a1d(rq,cq,cd,nf,rl,ceq):
    return rq*(cd+cq)+rl*(ceq+nf*cd)

#alpha 2 diode = rd->inf alpha 2
def a2d(rq,cq,cd,nf,rl,ceq):
    return rq*rl*(ceq*(cd+cq)+nf*cd*cq)

#################################
#RECHARGING OPERATION
def irl2(t,ho2,tq2,ti2,td2,T):
    a = ho2/(td2-ti2)
    b = ((tq2-ti2)/ti2)*math.exp(-1*(t-T)/ti2)
    c = ((tq2-td2)/td2)*math.exp(-1*(t-T)/td2)
    return a*(b-c)

#tq2
def tq2(tq,qcd,ts,qcq,tss,qceq,ho2):
    return (tq*qcd+ts*qcq+tss*qceq)/ho2

#ho2
def ho2(qceq,qcd):
    return qceq+qcd

#determine end of avalanche time (T)
def avT(ti,ti2,td,td2,ifi,nf,ith):
    a= ((td2-ti)*(ti2-ti))/(ti*(td-ti))
    b = ifi/(nf*ith-ifi)
    return ti*math.log(a*b)


def getPulse(t,N,rq,rd,rl,cq,cd,cm,nf,vov,ith,scale):

    yfit = [0]*len(t)
    i =0
    for time in t:
        if time<0:
            yfit[i]=0
        else:
            ceq = (N-nf)*((cd*cq)/(cd+cq))+N*cm
            alpha1 = a1(rd,cd,rq,cq,nf,rl,ceq)
            alpha1d = a1d(rq,cq,cd,nf,rl,ceq)
            alpha2 = a2(rd,cd,rq,cq,nf,rl,ceq)
            alpha2d = a2d(rq,cq,cd,nf,rl,ceq)

            taui = ti(alpha1,alpha2)
            taui2 =ti2(alpha1d,alpha2d)
            taud = td(alpha1,alpha2)
            taud2 = td2(alpha1d,alpha2d)
            tauq = rq*cq

            ifired = ivi(nf,vov,rq,rd,rl)

            bigT = avT(taui,taui2,taud,taud2,ifired,nf,ith)

            vcd = vov - rd*ird(bigT,ifired,taud,taud2,taui,taui2)
            vcq = rl*irl1(bigT,ifired,tauq,taui,taud)+rd*rd*ird(bigT,ifired,taud,taud2,taui,taui2) - vov
            vceq = rl*irl1(bigT,ifired,tauq,taui,taud)

            qcd = cd*nf*vcd
            qcq = cq*nf*vcq
            qceq = ceq*vceq

            transferH2 = ho2(qceq,qcd)

            taus = rq*cd
            tauss = rq*(cd+cq)

            tauq2 = tq2(tauq,qcd,taus,qcq,tauss,qceq,transferH2)

            #diode = ird(t,ifired,taud,taud2,taui,taui2)

            if(bigT > time):
                yfit[i]= scale*irl1(time,ifired,tauq,taui,taud) #rising and quenching
            else:
                yfit[i]= scale*irl2(time,transferH2,tauq2,taui2,taud2,bigT) #recharging
        i=i+1
    return yfit

def getPulsedata(npfile, doPlot=False):
    with open(npfile, 'rb') as f:
        t = np.load(f)*1e-6
        y = np.load(f)
    if(doPlot):
        plt.plot(t[0:1700], y[0:1700])
        plt.show()
    return t , y

def fitfunc(t,rq,cq,cd,scale,nf):
    N=8100
    #rq=1e6
    #nf=1
    rl=50
    rd=300
    cm=10.3e-15
    vov=3
    ith=100e-6
    return getPulse(t,N,rq,rd,rl,cq,cd,cm,nf,vov,ith,scale)