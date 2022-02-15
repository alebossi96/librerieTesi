import  sdtfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from scipy.sparse.linalg import lsmr
from scipy.linalg import hadamard
import librerieTesi.hadamardOrdering as hadamardOrdering
import librerieTesi.timeLasso as tLs
from sklearn import linear_model
from scipy.signal import peak_widths
import sys
import copy
from typing import List, Tuple
RAST = "rast"
#assert False, "Oh no! This assertion failed!"
HAD = "had"

"""
USARE scipy.optimize
class ridgeTime:
    def __init__(self,alpha):
        self.alpha = alpha
    def fit(self, matrix, data):
        self.recons = np.zeros((nBasis,self.nPoints ))
        lmd = 10;
        prev = np.zeros(self.recons[:,0].shape)
        for i in range(self.nPoints ):
            M = np.matmul(self.matrix.T,self.matrix)
            M += lmd *np.eye(M.shape[0])
            b = np.matmul(self.matrix.T, dataNp[:,i]) +lmd* prev
            self.recons[:,i] = lsmr(M,b)[0]
            prev = self.recons[:,i]
    
"""
#TODO fare velocemente il calcolo della risoluzione
#TODO importare modifiche fatte nel lab computer
#TODO data type f_tT_
#TODO mypy
#TODO dovrei anche selezionare solo un certo tGates
#TODO fare calibrazione direttamente con i wavenumber!
class f_tT_:
    
    def __init__(
            self,
            fileName: str,
            nBanks: int,
            nBasis: int,
            nMeas: int = -1,
            compress: bool = True,
            tot_banks = None,
            nPoints = 4096,
            background_idx = None):
        self.nPoints = nPoints
        self.nBanks =nBanks
        self.nBasis = nBasis
        if nMeas == -1: # se non specificato è uguale a nBasis, (caso compressioni)
            self.nMeas = nBasis
        else:
            self.nMeas = nMeas
        (self.time,self.data) = self.readSdt(fileName, tot_banks);
        
        self.compress = compress
        self.accumulate()
        if background_idx:
            print("remove")
            self.remove_off(background_idx[0], background_idx[1])
    def readSdt(self, fileName: str, tot_banks = None ) -> Tuple[np.array,List]:
        data = []
        for i in range(self.nBanks):
            if tot_banks:
                zeros = int(np.log10(tot_banks+1))
            else:
                zeros = int(np.log10(self.nBanks+1))
            fn = fileName
            if self.nBanks == 1:
                sdt = sdtfile.SdtFile(fn+".sdt")
            elif i==0:       
                 sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros)+str(i)+".sdt")
            elif i<10:       
                 sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros)+str(i)+".sdt")
            elif i<100:       
                 sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros-1)+str(i)+".sdt")
            elif i<1000:       
                 sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros-2)+str(i)+".sdt")
            elif i<10000:       
                 sdt = sdtfile.SdtFile(fn+"_c"+'0'*(zeros-3)+str(i)+".sdt")
            time = sdt.times[0][:]
            batch = sdt.data[0].shape[0];
            for j in range(batch):
                data.append(sdt.data[0][ j,:])
        return (time,data)
    def accumulate(self) -> None:
        self.tot = np.zeros((self.dim(),self.nPoints ))
        
        i = 0
        for el in self.data:
            if self.compress:
                idx = i%self.nBasis        
                self.tot[idx,:]+= el
            else:
                self.tot[i,:]= el
            i+=1
        
    def getData(self) -> np.array:
        return self.tot
    #magari posso passare :
    def getIsCompress(self) -> bool:
        return self.compress
    def getnBasis(self) -> int:
        return self.nBasis
    def getnMeas(self) -> int:
        return self.nMeas
    def dim(self) -> int:
        if self.compress:
            return self.nBasis
        return len(self.data)#ho copiato tutto, cambio solo tot
    def getTime(self) -> np.array:
        return self.time
    def __call__(self,i0: int, iF: int):#TODO era copy
        new = copy.copy(self)
        new.nMeas = iF-i0
        new.data = self.data[i0:iF]
        new.tot =self.tot[i0:iF,:]
        return new
    def __add__(self,b):
        return self.tot+b.tot
    def __sub__(self,b):
        return self.tot-b.tot
    def tGateBin(self, tGates):
        #mi ritorna il nuovo tGates e counts
        time = np.zeros((tGates,))
        tot = np.zeros((self.dim(),tGates))
        step = self.nPoints/tGates
        #TODO sbagliato!
        for i in range(tGates):
            time[i] = self.time[int(i*step)]
            tot[:,i] = np.sum(self.tot[:,int(i*step):int((i+1)*step)], axis = 1)
        self.time = time
        self.tot = tot
        self.nPoints = tGates
    def tot_counts(self):
        return sum(sum(self.tot))
    def tot_time_domain(self):
        return np.sum(self.tot, axis = 0)
    def remove_off(self, idx_start, idx_stop):
        dim = idx_stop-idx_start
        off = np.sum(self.tot[:,idx_start:idx_stop], axis = 1)/dim
        self.tot =self.tot - off[:,np.newaxis]
        print(self.tot.shape)
class reconstructTDDR:
    def __init__(self,
            data: f_tT_, 
            rastOrHad: str, 
            lambda_0: int ,
            method: str="lsmr", 
            Alpha: float=0, 
            removeFirstLine: bool = True,
            ref_cal_wl : Tuple[int, int] = None,
            ref_bas : int = 32,
            cake_cutting = False,
            normalize = False,
            filename_bkg = None, 
            n_banks_bkg= None,
            nPoints_bkg = None):
        
        self.data = data
        self.rastOrHad = rastOrHad
        self.lambda_0 = lambda_0
        self.method = method
        self.ref_cal_wl = ref_cal_wl
        if rastOrHad == RAST:
            self.removeFirstLine = False
        else:
            self.removeFirstLine = removeFirstLine
        self.nBasis = self.data.getnBasis()
        self.cake_cutting = cake_cutting
        self.normalize = normalize
        if self.normalize:
            self.normalize_with_backgroound(filename_bkg,n_banks_bkg, nPoints_bkg)
        assert not(rastOrHad != RAST and rastOrHad != HAD), "It must be RAST or HAD"

        assert not (rastOrHad == RAST and not self.data.getIsCompress()),"It cannot be raster and not compressed!"
        assert not(rastOrHad == RAST and self.removeFirstLine), "It cannot be raster and remove first line!"
        #TODO da fare assert per normalize!
        if method == "lsmr":
            print("lsmr")
            self.reg = linear_model.LinearRegression()
        elif method == "Lasso":
            print("Lasso")
            self.reg = linear_model.Lasso(alpha=Alpha)
        elif method == "Ridge":
            print("Ridge")
            self.reg = linear_model.Ridge(alpha=Alpha)
        elif method == "tLs":
            print("tLs")
            self.reg = tLs.timeLasso(alpha=Alpha)
        
        self.sz = self.data.dim()
        self.execute()    
        if removeFirstLine:
            ref_cal_wl[0] += 1
        self.axis(ref_cal_wl, ref_bas)


        self.start_range = 0
        self.stop_range = len(self.wavelength())
    def M(self) -> np.array:
        dim = self.data.getnBasis()
        if self.cake_cutting: 
            H = 0.5*(hadamardOrdering.cake_cutting(dim) + np.ones((dim,dim)))
        else:
            H = 0.5*(hadamard(dim) + np.ones((dim,dim)))
        if self.data.getIsCompress():
            return H
        matrix = np.zeros((self.sz, self.nBasis))
        
        for i in range(self.sz):
            matrix[i,:]= H[i%self.nBasis,:]
        return matrix
    def execute(self) -> None:

        if self.rastOrHad == HAD:
            self.reconstructHadamard()
        elif self.rastOrHad == RAST:
            self.recons = self.data.getData()
        else:
            sys.exit("controlla se hai scritto RAST o HAD")
    def reconstructHadamard(self) -> None:
        """
        self.reg.fit(self.M() ,self.data.getData())  
        self.reg.fit(self.M() ,self.data.getData())   
        self.recons = self.reg.coef_     
        print(self.M().shape)
        print(self.data.getData().shape)
        

        """
        self.reg.fit(self.M() ,self.data.getData())   
        recons1 = self.reg.coef_
        """    
        recons2 = np.zeros((self.nBasis,4096)) 
        for i in range(4096):
            recons2[:,i] = lsmr(self.M(),self.data.getData()[:,i])[0] #Perchè qui c'è la prima linea a caso e in reg..fit no?
        """
        self.recons= recons1#self.recons.T

    def axis(self, refCal : Tuple[int, int] = None, ref_bas : int = 32) -> None:
        self.calibrationToWl(refCal, ref_bas)
        self.calibrateToWn()
    def calibrationToWl(self, ref_cal_wl : Tuple[int, int] = None, ref_bas : int = 32) -> None: 
        #ref is in a basis 32
        p = np.array([  1.51491966* 64/self.nBasis, 809.28844682])
        if ref_cal_wl != None:
            p[1] = ref_cal_wl[1] - ref_cal_wl[0]*p[0]*self.nBasis/ref_bas          
        self.wl=np.zeros(self.nBasis)
        for i in range(self.nBasis):
            self.wl[i]=i*p[0]+p[1]
    def calibrateToWn(self) -> None:
        self.wn=np.zeros(self.nBasis)
        for i in range(self.nBasis):
            self.wn[i]=(1/self.lambda_0-1/self.wl[i])*1e7
            
    def wlAtTimeGate(self, tGate: int) -> np.array:
        idx = np.where(self.time == tGate)
        return self.recons[:,idx]
    def line(self, idx: int) -> np.array:
        return self.recons[idx,:]
    def wavenumber(self) -> np.array:
        if not self.normalize:
            if self.removeFirstLine:
                return self.wn[1:]
            return self.wn
        return self.wn[self.start_range:self.stop_range]
    def wavelength(self) -> np.array:
        if not self.normalize:
            if self.removeFirstLine:
                return self.wl[1:]
            return self.wl
        return self.wl[self.start_range:self.stop_range]
    def time(self, inNs: bool = True) -> np.array:
        if inNs:
            conv = 1e9
        else:
            conv = 1
        return self.data.getTime() * conv
    def reconstruction(self) -> np.array:
        if self.rastOrHad == RAST:
            return self.recons
        recons = self.recons.T
        if not self.normalize:
            if self.removeFirstLine:
                return recons[1:,:]
            return recons
        norm_spect = recons[ self.start_range:self.stop_range,:]#TODO/self.bkg_spectr[self.start_range:self.stop_range, np.newaxis]
        return norm_spect
    def find_maximum_idx(self):
        return np.where(self.spectrograph() == np.amax(self.spectrograph()))[0][0]
        #TODO useful for calibration but I will need to implement it
    def fwhm(self,axis: np.array,data: np.array) -> float:
        pos0 = np.where(data == np.amax(data))[0][0]
        fwhm_idx = int(peak_widths(data, [pos0])[0])
        
        fwhm =  (axis[pos0+fwhm_idx] - axis[pos0-fwhm_idx])
        return fwhm
    def tGates(self, init, fin):
        #TODO dovrei fare nuovo oggetto?
        init = int(init)
        fin = int(fin)
        return np.sum(self.reconstruction()[:,init:fin],axis = 1)
    def spectrograph(self):
        return np.sum(self.reconstruction(), axis = 1)
    def toF(self):
        return np.sum(self.reconstruction(), axis = 0)
    def background(self, initial_pos, final_pos):
        return np.sum(self.reconstruction()[:,initial_pos:final_pos],axis = 1)/(final_pos-initial_pos)
    def reconstruction_remove_background(self,initial_pos, final_pos):
        return self.reconstruction() - self.background(initial_pos, final_pos)[:, np.newaxis]
    def __len__(self):
        if self.removeFirstLine:
            return self.nBasis -1
        return self.nBasis
    def __sub__(self, b):
        new = copy.copy(self)
        return new.recons-b.recons
    def __sum__(self):
        return sum(self.spectrograph())
    def normalize_with_backgroound(self, filename_bkg,n_banks, nPoints_bkg):
        data = f_tT_(fileName = filename_bkg,
                nBanks = n_banks ,#Banks[i], 
                nBasis = self.nBasis,#B_measurements[i],
                compress = True,
                nPoints = nPoints_bkg
                )
        rec = reconstructTDDR(data = data,
            rastOrHad = HAD,
            lambda_0 = self.lambda_0,
            method ="lsmr",
            removeFirstLine = True,
            ref_cal_wl = self.ref_cal_wl,
            ref_bas = self.nBasis)
        self.bkg_spectr = rec.spectrograph()/np.max(rec.spectrograph())
        limit = 0.2
        self.start_range = np.argmax(self.bkg_spectr > limit)
        self.stop_range = np.argmin(self.bkg_spectr[(self.start_range+10):] > limit) +self.start_range+10
