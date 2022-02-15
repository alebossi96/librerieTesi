import sys
import copy
from typing import Tuple
import numpy as np
from scipy.linalg import hadamard
import librerieTesi.hadamardOrdering as hadamardOrdering
import librerieTesi.timeLasso as tLs
from sklearn import linear_model
from librerieTesi.diffuseRaman import raw_data as rd
from librerieTesi.diffuseRaman import core
import tesi_stefan.core.SignalsAndSystems as sis
RAST = "rast"
HAD = "had"
class Reconstruction:
    """
    This class performs the reconstruction from the raw_data (from the .sdt files).
    It transforms it into the usual wavelength/wavenumber axis.
    """
    def __init__(self,
            data: rd.RawData,
            rast_had: str,
            lambda_0: int,
            method: str="lsmr",
            alpha: float=0,
            remove_first_line: bool = True,
            ref_cal_wn : Tuple[int, int] = None,
            ref_cal_wl : Tuple[int, int] = None,
            ref_bas : int = None,
            cake_cutting = False,
            normalize = False,
            filename_bkg = None,
            n_banks_bkg= None,
            ):
        self.data = data
        self.rast_had = rast_had
        self.lambda_0 = lambda_0
        self.method = method
        self.ref_cal_wl = ref_cal_wl
        if rast_had == RAST:
            self.remove_first_line = False
        else:
            self.remove_first_line = remove_first_line
        self.n_basis = self.data.n_basis
        self.cake_cutting = cake_cutting
        self.normalize = normalize
        if ref_bas is None:
            ref_bas = self.n_basis
        if ref_cal_wn is not None:
            ref_cal_wl = ref_cal_wn
            ref_cal_wl[1] = 1/(1/ lambda_0 - ref_cal_wn[1] / 1e7)
        if self.normalize:
            self.normalize_with_efficiency(filename_bkg,n_banks_bkg)
        assert not(rast_had != RAST and rast_had != HAD), "It must be RAST or HAD"
        assert not (rast_had == RAST and not self.data.compress),"It cannot be raster and not compressed!"
        assert not(rast_had == RAST and self.remove_first_line), "It cannot be raster and remove first line!"
        #TODO da fare assert per normalize!
        if method == "lsmr":
            print("lsmr")
            self.reg = linear_model.LinearRegression()
        elif method == "Lasso":
            print("Lasso")
            self.reg = linear_model.Lasso(alpha=alpha)
        elif method == "Ridge":
            print("Ridge")
            self.reg = linear_model.Ridge(alpha=alpha)
        elif method == "tLs":
            print("tLs")
            self.reg = tLs.timeLasso(alpha=alpha)
        self.execute()
        if remove_first_line and ref_cal_wl is not None:
            ref_cal_wl[0] += 1
        self.axis(ref_cal_wl, ref_bas)
        self.start_range = 0
        self.stop_range = len(self.wavelength())
    def M(self) -> np.array:
        """
        Generates the reconstruction matrix
        """
        dim = len(self.data)
        if self.cake_cutting:
            H = 0.5*(hadamardOrdering.cake_cutting(dim) + np.ones((dim,dim)))
        else:
            H = 0.5*(hadamard(dim) + np.ones((dim,dim)))
        if self.data.compress:
            return H
        matrix = np.zeros((len(self.data), self.n_basis))
        for i in range(len(self.data)):
            matrix[i,:]= H[i%self.n_basis,:]
        return matrix
    def execute(self) -> None:
        """
        It execute the reconstruction.
        The reconstructed data are stored into self.recons.
        """
        if self.rast_had == HAD:
            self.reconstruct_hadamard()
        elif self.rast_had == RAST:
            self.recons = self.data.tot
        else:
            sys.exit("controlla se hai scritto RAST o HAD")
    def reconstruct_hadamard(self) -> None:
        """
        Reconstruct from the hadamard basis into the usual wavelength-wavenumber base.
        """
        self.reg.fit(self.M() ,self.data.tot)
        recons1 = self.reg.coef_
        self.recons= recons1
    def axis(self, ref_cal : Tuple[int, int] = None, ref_bas : int = 32) -> None:
        """
        From the calibration data it generates the wavenumber - wavelength axis.
        """
        self.calibration_wl(ref_cal, ref_bas)
        self.calibration_wn()
    def calibration_wl(self, ref_cal_wl : Tuple[int, int] = None, ref_bas : int = 32) -> None:
        """
        Calibration of wavenuber axis.
        """
        #ref is in a basis 32
        p = np.array([  1.51491966* 64/self.n_basis, 809.28844682])
        if ref_cal_wl is not None:
            p[1] = ref_cal_wl[1] - ref_cal_wl[0]*p[0]*self.n_basis/ref_bas
        self.wl=np.zeros(self.n_basis)
        for i in range(self.n_basis):
            self.wl[i]=i*p[0]+p[1]
    def calibration_wn(self) -> None:
        """
        From the wavelength it is converted into wavenumber.
        """
        self.wn=np.zeros(self.n_basis)
        for i in range(self.n_basis):
            self.wn[i]=(1/self.lambda_0-1/self.wl[i])*1e7
    def wl_with_time(self, time: int) -> np.array:
        """
        Returns the time-domain wavelength distribution.
        """
        idx = np.where(self.time() == time)
        return self.recons[:,idx]
    def wavenumber(self) -> np.array:
        """
        Returns the wavenuber axis.
        """
        if not self.normalize:
            if self.remove_first_line:
                return self.wn[1:]
            return self.wn
        return self.wn[self.start_range:self.stop_range]
    def wavelength(self) -> np.array:
        """
        Returns the wavelength axis.
        """
        if not self.normalize:
            if self.remove_first_line:
                return self.wl[1:]
            return self.wl
        return self.wl[self.start_range:self.stop_range]
    def time(self, conv_to_ns: bool = False) -> np.array:
        """
        Returns the time axis.
        """
        if conv_to_ns:
            conv = 1e9
        else:
            conv = 1
        return self.data.time * conv
    def reconstruction(self) -> np.array:
        """
        Returns the data.
        """
        if self.rast_had == RAST:
            return self.recons
        recons = self.recons.T
        if not self.normalize:
            if self.remove_first_line:
                return recons[1:,:]
            return recons
        norm_spect = recons[ self.start_range:self.stop_range,:]#TODO/self.bkg_spectr[self.start_range:self.stop_range, np.newaxis]
        return norm_spect
    def get_list_meas(self):
        rec = self.reconstruction()
        output = []
        for i in range(self.data.n_points):
            output.append(rec[:,i])
        output = np.array(output)
        return output
    def normalize_rec(self, const = 1.):
        self.recons/=self.data.tot_counts()
        self.recons*= const
    def find_maximum_idx(self):
        """
        Find the position of the maximum of the spectrograph. Useful during calibration.
        """
        return np.where(self.spectrograph() == np.amax(self.spectrograph()))[0][0]
        #TODO useful for calibration but I will need to implement it
    def t_gates(self, init, fin):
        """
        Gating of the measurement.
        """
        #TODO dovrei fare nuovo oggetto?
        init = int(init)
        fin = int(fin)
        return np.sum(self.reconstruction()[:,init:fin],axis = 1)
    def spectrograph(self):
        """
        returns the sum on time of each wavenumber
        """
        return np.sum(self.reconstruction(), axis = 1)
    def normalize_with_efficiency(self, filename_bkg,n_banks):
        """
        From the light background of the sun we find the relative efficiency of the spectrometer at each wavelength.
        This fuction re-normalize the measurement.
        """
        data = rd.RawData(filename = filename_bkg,
                n_banks = n_banks,#Banks[i],
                n_basis = self.n_basis,#B_measurements[i],
                compress = True,
                )
        rec = Reconstruction(data = data,
            rast_had = HAD,
            lambda_0 = self.lambda_0,
            method ="lsmr",
            remove_first_line = True,
            ref_cal_wl = self.ref_cal_wl,
            ref_bas = self.n_basis)
        self.bkg_spectr = rec.spectrograph()/np.max(rec.spectrograph())
        limit = 0.2
        self.start_range = np.argmax(self.bkg_spectr > limit)
        self.stop_range = np.argmin(self.bkg_spectr[(self.start_range+10):] > limit) +self.start_range+10
    def cut_spectra(self, idx_start, idx_stop):
        """
        It selects the spectra into the desired range
        """
        """
        idx_start = core.wavenumber_to_idx(self.wavenumber(),wn_start)
        idx_stop = core.wavenumber_to_idx(self.wavenumber(),wn_stop)
        """
        new = copy.deepcopy(self)
        new.wl = self.wavelength()[idx_start:idx_stop]
        new.wn = self.wavenumber()[idx_start:idx_stop]
        new.recons = self.reconstruction()[idx_start:idx_stop]
        new.remove_first_line = False
        if self.normalize:
            new.bkg_spectr = self.bkg_spectr[idx_start:idx_stop]
        return new
    def get_signal(self,idx_start, n_points):
        """
        Returns the Signal class for the reconstruction
        """
        idx_stop = idx_start+ n_points
        real_signal = sis.Signal(n_points =n_points, length = len(self.time()))
        real_signal.t = self.time()
        real_signal.l= self.wavenumber()[idx_start:idx_stop]
        real_signal.s = self.reconstruction()[idx_start:idx_stop]
        return real_signal
    def tot_counts(self):
        """
        Return the total counts measured and reconstructed
        """
        recons = np.sum(self.reconstruction())
        meas = self.data.tot_counts()
        return (meas,recons)
    def __len__(self):
        if self.remove_first_line:
            return self.n_basis -1
        return self.n_basis
    def __sub__(self, b):
        new = copy.copy(self)
        return new.recons-b.recons
    def __sum__(self):
        return sum(self.spectrograph())
