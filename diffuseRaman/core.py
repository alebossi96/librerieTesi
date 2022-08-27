from typing import List, Tuple
from scipy.signal import peak_widths
import numpy as np
import pandas as pd
def time_to_idx(array,time):
    i = 0
    while array[i]<=time:
        i+=1
    return i
def wavenumber_to_idx(array,wn):
    i = 0
    while array[i]<=wn:
        i+=1
    return i
       
def fwhm(axis: np.array,data: np.array) -> float:
    pos0 = np.where(data == np.amax(data))[0][0]
    fwhm_idx = int(peak_widths(data, [pos0])[0]/2)
    fwhm =  (axis[pos0+fwhm_idx] - axis[pos0-fwhm_idx]) #sarà metà da una parte e metà altra anche se questo è decisamente sbagliato
    return fwhm
    
def conv_matrix(signal, number_of_points):
    M_out = np.zeros((number_of_points,number_of_points))
    for i in range(number_of_points):
        M_out[i][i:] = signal[:(number_of_points-i)]
    return M_out.T
def get_data_FIT(filename, source, output, delim = "\t"):
    df = pd.read_csv(filename, delimiter = delim)
    for name, param in source:
        df = df[df[name] == param]
    out = []
    for name in output:
        out.append(df[name])
    out = tuple(out)
    return out
