import numpy as np
import numba as nb
import h5py
import time
import sys

#compression function
@nb.jit(nopython=True)
def compression(sig_in, sig_out, sig_len_in):
    mask = [0,1,3,7,15,31,63,127,255,511,1023,2047,4095,8191,16383,32767,65535]
    j = iso = bp = 0
    sig_out[iso]=sig_len_in 
    db = np.zeros(2, dtype=np.ushort)
    dd = np.frombuffer(db, dtype=np.uintc)

    iso += 1
    while (j < sig_len_in):
        max1 = min1 = sig_in[j]
        max2 = -16000
        min2 = 16000
        nb1 = nb2 = 2
        nw = 1
        i = j+1
        while ((i < sig_len_in) and (i<j+48)):
                if max1 < sig_in[i]:
                    max1 = sig_in[i]
                if min1 > sig_in[i]:
                    min1 = sig_in[i]
                ds = (sig_in[i] - sig_in[i-1])
                if max2 < ds:
                    max2 = ds
                if min2 > ds:
                    min2 = ds
                nw += 1
                i += 1
        if (max1-min1 <= max2-min2):
            nb2 = 99
            while (max1 - min1) > mask[nb1]:
                nb1 += 1
            while (i < sig_len_in) and (i < j+128):
                if (max1 < sig_in[i]):
                    max1 = sig_in[i]
                dd1 = max1 - min1
                if (min1 > sig_in[i]):
                    dd1 = max1 - sig_in[i]
                if (dd1 > mask[nb1]):
                    break
                if (min1 > sig_in[i]):
                    min1 = sig_in[i]
                nw +=1
                i += 1
        else:
            nb1 = 99
            while (max2 - min2 > mask[nb2]):
                nb2 += 1
            while ((i < sig_len_in) and (i < j+128)):
                ds = sig_in[i] - sig_in[i-1]
                if (max2 < ds):
                    max2 = ds
                dd2 = max2 - min2
                if (min2 > ds):
                    dd2 = max2 - ds 
                if (dd2 > mask[nb2]):
                    break
                if (min2 > ds):
                    min2 = ds
                nw +=1
                i+=1

        if (bp > 0): 
            iso += 1
        sig_out[iso] = nw 
        iso += 1
        bp = 0 
        if (nb1 <= nb2):
            sig_out[iso] = nb1
            iso += 1
            sig_out[iso] = min1 
            iso += 1
 
            i=iso
            while (i <= (iso + nw*nb1/16)):
                sig_out[i] =0
                i+=1

            i=j
            while (i < j + nw):
                dd[0] = (sig_in[i] - min1)
                dd[0] = dd[0] << (32 - bp - nb1)  
                sig_out[iso] |= db[1]
                bp += nb1
                if (bp > 15):
                    iso += 1
                    sig_out[iso] = db[0]
                    bp -=16
                i += 1

        else:
            sig_out[iso] = (nb2 + 32)
            iso += 1
            sig_out[iso] = (sig_in[j])
            iso += 1
            sig_out[iso] = (min2)
            iso += 1

            i = iso
            while (i <= iso + nw*nb2/16):
                sig_out[i] = (0)
                i += 1
            
            i = j+1
            while (i < j + nw):
                dd[0] = sig_in[i] - sig_in[i-1] - min2
                dd[0] = dd[0] <<(32-bp-nb2)
                sig_out[iso] |= db[1]
                bp += nb2
                if bp > 15:
                    iso += 1
                    sig_out[iso] = db[0]
                    bp -= 16
                i += 1
        j += nw

    if (bp > 0):
        iso += 1

    return iso


#decompression function
@nb.jit(nopython = True)
def decompression(sig_in, sig_out, sig_len_in):
    mask= [0, 1,3,7,15, 31,63,127,255, 511,1023,2047,4095, 8191,16383,32767,65535]

    j = isi = iso = bp = 0
    siglen = np.ushort(sig_in[isi])
    isi += 1
    db = np.zeros(2,dtype=np.ushort)
    dd = np.frombuffer(db, dtype=np.uintc)
    while ((isi < sig_len_in) and (iso < siglen)):
        if bp > 0:
            isi += 1
        bp = 0
        nw = sig_in[isi]
        isi += 1
        nb = sig_in[isi]
        isi += 1

        if (nb < 32):
            min_val = sig_in[isi]
            isi += 1
            db[0] = sig_in[isi]
            i = 0
            while ((i < nw) and (iso < siglen)):
                if (bp+nb)>15:
                    bp -= 16
                    db[1] = sig_in[isi]
                    isi += 1
                    db[0] = sig_in[isi]
                    dd[0] = dd[0] << (bp+nb)
                else:
                    dd[0] = dd[0] << nb
                sig_out[iso] = (db[1] & mask[nb]) + min_val
                iso += 1
                bp += nb
                i += 1

        else:
            nb -= 32
            sig_out[iso] = np.short(sig_in[isi])
            iso += 1
            isi += 1
            min_val = np.short(sig_in[isi])
            isi += 1
            db[0] = sig_in[isi]

            i = 1
            while ((i<nw) and (iso < siglen)):
                if (bp+nb)>15:
                    bp -= 16
                    db[1] = sig_in[isi]
                    isi += 1
                    db[0] = sig_in[isi]
                    dd[0] = dd[0] << (bp+nb)
                else:
                    dd[0] = dd[0] <<nb
                sig_out[iso] = (db[1] & mask[nb]) + min_val + sig_out[iso-1]
                iso += 1
                bp += nb
                i += 1
        j += nw

    if (siglen != iso):
        raise Exception("iso != siglen")
    return siglen


#turn ndarray to vecotr of vectors
@nb.jit(nopython = True)
def nda_to_vect(ndarray, flattened_data, cumulative_length):
    length = 0
    cumulative_length[0] = length
    i = 0
    for sig_in in ndarray:
        sig_len_in = len(sig_in)
        sig_out = np.empty(sig_len_in, dtype=np.ushort)
        iso = compression(sig_in, sig_out, sig_len_in)
        flattened_data[length : length+iso] = sig_out[:iso]
        i += 1
        length += iso
        cumulative_length[i] = length

    return length


#turn vector of vectors to ndarray
@nb.jit(nopython=True)
def vect_to_nda(flattened_data, cumulative_length, nda):
    i = 0
    for idx1, idx2 in zip(cumulative_length[:-1], cumulative_length[1:]):
        sig_in = flattened_data[idx1:idx2]
        sig_len_in = len(sig_in)
        iso = decompression(sig_in, nda[i,:], sig_len_in)
        i += 1

    return None 


#generate an empty ndarray with appropriate shape for converting vector of vectors to ndarray
def empty(flattened_data, cumulative_length):
    idx1, idx2 = cumulative_length[0], cumulative_length[1]
    sig_in = flattened_data[idx1:idx2]
    sig_len_in = len(sig_in)
    sig_out = np.empty(sig_in[0], dtype = np.ushort)
    r = cumulative_length.size-1
    c = decompression(sig_in, sig_out, sig_len_in)
    empty_ndarray  = np.empty(r*c, dtype = np.ushort).reshape(r,c)

    return empty_ndarray


