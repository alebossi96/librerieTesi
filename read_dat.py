"""
Module to read DAT files.
"""
import struct
import numpy as np
HEADLEN=764
SUBLEN=204
NUMCHAN=4096

def read_dat_list(filename, length):
    """
    Read dat files.
    """
    with open(filename,'rb') as data:
        data.read(HEADLEN)
        tot = []
        for _ in range(length):
            data.read(SUBLEN)
            res=[]
            for _ in range(NUMCHAN):
                value=struct.unpack('H',data.read(2))
                res.append(value[0])
            tot.append(res)
    return np.array(tot)

def read_dat(filename, loop):
    length = 1
    for i in range(len(loop)):
        length *= loop[i]
    tot = read_dat_list(filename, length)
    loop.append(NUMCHAN)
    return np.reshape(tot, loop)
