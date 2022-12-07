"""
Module to read DAT files.
"""
import struct
import numpy as np
HEADLEN=764
SUBLEN=204


def read_dat_list(filename, length, length_ch, type_data):
    """
    Read dat files.
    """
    with open(filename,'rb') as data:
        data.read(HEADLEN)
        tot = []
        for _ in range(length):
            data.read(SUBLEN)
            res=[]
            for _ in range(length_ch):
                value=struct.unpack(type_data[0],data.read(type_data[1]))
                res.append(value[0])
            tot.append(res)
    return np.array(tot)

def read_dat(filename, loop, length_ch=4096,  type_data = ['H',2]):
    length = 1
    for i in range(len(loop)):
        length *= loop[i]
    tot = read_dat_list(filename, length, length_ch, type_data)
    loop.append(length_ch)
    return np.reshape(tot, loop)
