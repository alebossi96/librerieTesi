#TODO potrebbe essere automatica lo start dell'impulso
from librerieTesi.reconstructTDDRV2 import RAST, HAD, f_tT_, reconstructTDDR
from librerieTesi.latex import *
import matplotlib.pyplot as plt
from librerieTesi.exportSpectra import ExportSpectra
from scipy.signal import peak_widths
class Data_analisys:
    def __init__(self, fn, nBanks,  n_basis,rastOrHad, lambda_0, ref_cal_wn = None, ref_bas = 32,nPoints = 4096, tot_banks = None,normalize = True, filename_bkg = None, n_banks_bkg = None, nPoints_bkg = None, tGates_bin =  None, save_as = None, background_idx = None):
        self.fn = fn
        if tot_banks == None:
            tot_banks = nBanks
        self.data = f_tT_(fileName = fn,
                    nBanks = nBanks ,#Banks[i], 
                    nBasis = n_basis,#B_measurements[i],
                    compress = True,
                    tot_banks = tot_banks,
                    nPoints = nPoints,
                    background_idx = background_idx )
        if not tGates_bin == None:
            self.data.tGateBin(tGates_bin)         
        self.ltx = latex("test")
        self.ltx.changeExt("png")
        tot = self.data.tot_counts()
        if save_as:
            self.save_as = save_as
        else:
            self.save_as = fn
        print("tot counts measured", tot)
        if ref_cal_wn == None:
            ref_cal_wl = None
        else:
            ref_cal_wl = ref_cal_wn
            ref_cal_wl[1] = 1/(1/ lambda_0 - ref_cal_wn[1] / 1e7)
        self.rec = reconstructTDDR(data = self.data,
                rastOrHad = rastOrHad,
                lambda_0 = lambda_0,
                method ="lsmr",
                Alpha = 0.7,
                removeFirstLine = True,
                ref_cal_wl = ref_cal_wl,
                ref_bas = ref_bas,
                normalize = normalize,
                filename_bkg = filename_bkg,
                n_banks_bkg = n_banks_bkg,
                nPoints_bkg = nPoints_bkg)
        if normalize:        
            self.rec.normalize_with_backgroound(filename_bkg,n_banks_bkg, nPoints_bkg)
    def chose_time_startStop(self):
        self.plotTime(show = True, show_time = False)
    def calibration(self, zoom_start = 0,zoom_stop = None,vertical_lines = []):
        if zoom_stop == None:
            zoom_stop = len(self.rec.wavenumber())
        self.plot_spectrograph(vertical_lines = vertical_lines , show = True, show_wavenumber = False, zoom = (zoom_start,zoom_stop))
    def plotTime(self, show,save, show_time):
        if show_time:
            x = self.rec.time()
        else:
            x = np.arange(0,self.data.nPoints)
        self.ltx.plotData( x = x ,
                 y= self.data.tot_time_domain(),#,hist,
                 x_label = "time",
                 x_um="$ns$",
                 plot_title = "",
                 IsLog = True,
                 saveas = self.save_as+"time",
                 show = show,
                 save = save
                  )        
    def plot2D(self):
        self.ltx.plotData2D( x= self.rec.time(), #sarebbe un filo da automatizzare, è ovvio che metto queste cose
                        y = self.rec.wavenumber(),
                        data = self.rec.reconstruction(), #rec.reconstruction(),
                        x_label="time",
                        x_um = "ns",
                        y_label = "wavenumber",
                        y_um= "$cm^{-1}$",
                        plot_title="measurement",
                        save = True,
                        saveas = self.save_as)
    def plot_spectrograph(self,vertical_lines, show, save = True, show_wavenumber = True):
        x = self.rec.wavenumber()
        if not show_wavenumber:
            x = np.arange(self.rec.start_range, len(x)+self.rec.start_range)
        self.ltx.plotData( x = x,
                         y= self.rec.spectrograph(),
                         x_label = "wavenumber",
                         x_um="$cm^{-1}$",
                         plot_title = "Spectrograph. Integration on all time gates",
                         IsLog = False,
                         saveas = self.save_as+"spectra",
                         vertical_lines = vertical_lines,
                         show = show,
                         save = save )

    def export_csv_spectrograph(self):                 
        exp = ExportSpectra(self.rec.wavenumber(), [self.rec.spectrograph()])
        exp.title("wavenumber", ["integration on all time-gates"]) 
        exp.exportCsv(self.save_as+"_sum", first_line = True)  
    def bin_rec_tGates_V2(self, dim, t0, tf):    
        recGate = []
        times = []
        
        def find_idx(t):
            i = 0
            
            while(self.rec.time()[i]< t):
                i+=1
            return i
            
        iS = find_idx(t0)
        fS = find_idx(tf)
        for i in range(0,dim):
            times.append(self.rec.time()[int(i*(fS-iS)/dim)])
            data = self.rec.tGates(iS+ i*(fS-iS)/dim,iS+ (i+1)*(fS-iS)/dim)
            recGate.append(data)
        return (times, recGate)
    def bin_rec_tGates(self, dim, iS, fS):    
        recGate = []
        times = []
        for i in range(0,dim):
            times.append(self.rec.time()[int(i*(fS-iS)/dim)])
            data = self.rec.tGates(iS+ i*(fS-iS)/dim,iS+ (i+1)*(fS-iS)/dim)
            recGate.append(data)
        return (times, recGate)
    def export_csv_spectrograph_gated(self, times, recGate):               
        exp = ExportSpectra(self.rec.wavenumber(), recGate)
        exp.title("wavenumber", times) 
        exp.exportCsv(self.save_as+"_gates", first_line = True)   
    def subplot(self,times, recGate, show,vertical_lines = [], save = True, show_wavenumber = True, saveName = "tGAtesI"):      
        """
        self.ltx.multipleLineSubPlotsSP( x = self.rec.wavenumber(),
                            data = recGate,
                            title ="",
                            xLabels ="wavenumber",
                            x_um="$cm^{-1}$",
                            saveas = self.save_as+"tGAtes",
                            show = show )
        """
        x = self.rec.wavenumber()
        if not show_wavenumber:
            x = np.arange(0, len(x))
        self.ltx.multipleLineSubPlots( x = x,
                        y = times,
                        data = recGate,
                        title ="",
                        y_um = "ns",
                        xLabels ="wavenumber",
                        x_um="$cm^{-1}$",
                        vertical_lines = vertical_lines,
                        saveas = self.save_as+saveName,
                        show = show,
                        save = save )
    def subplot_multipleLine(self,times, recGates, legend, show,vertical_lines = [], save = True, show_wavenumber = True, saveName = "tGAtesI"):      
        x = self.rec.wavenumber()
        if not show_wavenumber:
            x = np.arange(0, len(recGates[0][0]))
        self.ltx.multipleLineSubPlots_multipleline( x = x,
                        y = times,
                        data = recGates,
                        legend = legend,
                        title ="",
                        y_um = "ns",
                        xLabels ="wavenumber",
                        x_um="$cm^{-1}$",
                        vertical_lines = vertical_lines,
                        saveas = self.save_as+saveName,
                        show = show,
                        save = save )
    def __len__(self):
        return len(self.rec.wavenumber())
    def time_fwhm(self):
        #pos0 = np.where(data == np.amax(self.data.tot_time_domain()))[0][0]
        pos0 = np.argmax(self.data.tot_time_domain())
        print(pos0)
        fwhm_idx = int(peak_widths(self.data.tot_time_domain(), [pos0])[0])
        
        fwhm =  (self.data.time[pos0+fwhm_idx] - self.data.time[pos0-fwhm_idx])
        return fwhm
    
