import scipy.io
import numpy as np
from  librerieTesi.diffuseRaman import raw_data
import matplotlib.pyplot as plt
class ReadDataMichele(raw_data.RawData):
    def __init__(self, folder, var_name, background_time = None, num_store = 1):
        if num_store == 2:
            TotMatrix1 = scipy.io.loadmat(folder+'/'+var_name+'1.mat')[var_name+'1']
            TotMatrix2 = scipy.io.loadmat(folder+'/'+var_name+'2.mat')[var_name+'2']
            shape = TotMatrix1.shape
            TM = np.zeros((4096,shape[1],256));
            TM[:,:,0:128] = TotMatrix1
            print("1", np.sum(TotMatrix1))
            TM[:,:,128:256] = TotMatrix2
            print("TM", np.sum(TM))
        elif num_store == 1:
            TM = scipy.io.loadmat(folder+'/'+var_name+'.mat')[var_name]
        self.tot = np.sum(TM, axis = 1).T
        shape = self.tot.shape
        self.n_basis = shape[0]
        self.n_meas = self.n_basis
        self.n_points = shape[1]
        self.time =  scipy.io.loadmat(folder+'/'+'raw_times.mat')['time'][0, 0:4096]*1e-9
        
        self.compress = True
        self.use_michele_data = True
        if background_time is not None:
            self.remove_bkg(background_time[0], background_time[1])
