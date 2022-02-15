import itertools
import matplotlib.pyplot as plt
import numpy as np
class Plot:
    def __init__(self, ext: str = "png", folder = ''):
        self.ext =ext
        self.x = []
        self.y = []
        self.legend = []
        self.num_lines = 0
        self.vertical_lines = []
        self.folder = folder+"/"
    def change_ext(self,ext):
        """
        Changes the output file extention
        """
        self.ext = ext
    def add_vertical_line(self,line, color = 'b'):
        self.vertical_lines.append((line, color))
    def add_line(self, x, y,label = ''):
        self.x.append(x)
        self.y.append(y)
        self.num_lines += 1
        if label is not None:
            self.legend.append(label)
        else:
            self.legend.append('')
    def close(self, vertical_lines = False):
        self.x = []
        self.y = []
        self.legend = []
        if vertical_lines:
            self.vertical_lines = []
    def plot_data(self,
             xlabel: str = '',
             x_um: str = '',
             title: str = '',
             saveas: str = '',
             is_log: bool = False,
             show = True,
             save = True,
             show_legend = False,
             show_vertical_line = True,
             normalize = False
             ) -> str:
        for (x,y,legend) in zip(self.x,self.y,self.legend):
            y = np.array(y)
            if normalize:
                const = np.max(y)
            else:
                const = 1
            plt.plot(x,y/const, label = legend)
        if show_legend:
            plt.legend()
        if is_log:
            plt.yscale('log')
        plt.xlabel(xlabel+"["+x_um+"]")
        plt.ylabel("counts [a.u.]")
        plt.title(title)
        if show_vertical_line:
            for el in self.vertical_lines:
                plt.axvline(el[0], color = el[1])
        if save:
            plt.savefig(self.folder+saveas+"." +self.ext)
        if show:
            plt.show()
        plt.close()
        return saveas
    def multiple_line_subplots(self, y, y_um ='', y_digit = 0,
                                    xlabel = '', x_um = '',
                                    title = '', saveas = '',
                                    is_setLabel = True, show = True, 
                                    save = False, show_legend = False,
                                     show_vertical_line = True,
                                     normalize = False, fix_height = False):
        sz = len(y)
        #TODO se è solo 1 cosa devo fare??
        fig, axs = plt.subplots(sz)
        fig.suptitle(title)
        if fix_height:
            max_val = np.max(self.y)*1.1
        for (x,data,legend) in zip(self.x,self.y,self.legend):
            for i in range(sz):
                if normalize:
                    norm = max(data[i])
                else:
                    norm = 1
                axs[i].plot(x, data[i]/norm, label = legend)
                axs[i].tick_params(labelbottom=False)
                if fix_height:
                    axs[i].axis(ymax = max_val)#ylim(top = max_val)
                if is_setLabel:
                    print(round(y[i]))
                    axs[i].set_ylabel(str(round(y[i], y_digit))+ " "+ y_um)
                if show_vertical_line:
                    for el in self.vertical_lines:
                        axs[i].axvline(el[0], color = el[1])
                if show_legend:
                    plt.legend()
        axs[sz-1].tick_params(labelbottom=True)
        plt.xlabel(xlabel + "["+ x_um+"]")
        plt.title(title)
        if save:
            plt.savefig(self.folder + saveas+"."+ self.ext)
        if show:
            plt.show()
        plt.close()
        return saveas
    #eccezioni
    def plot_data2D(self,
             x: np.array,
             y: np.array,
             data: np.array,
             xlabel: str,
             x_um: str,
             y_label: str,
             y_um: str,
             plot_title: str,
             saveas: str,
             show = False,
             save = True ) -> str:
        plt.rcParams['pcolor.shading'] ='nearest'
        plt.pcolormesh(x, y,data)
        plt.xlabel( xlabel+" ["+x_um+"]")
        plt.ylabel(y_label+ "["+y_um+"]")
        plt.title(plot_title)
        plt.colorbar()
        #TODO consigliabile non usare eps. perchè pesante in 2D!
        if save:
            plt.savefig(self.folder + saveas+"."+ self.ext)
        if show:
            plt.show()
        plt.close()
        return saveas
    def multiple_line_subplots_2(self, x, data, title, xlabel, x_um, saveas, show = False, save = True):
        sz = len(data)
        for i in range(sz):
            mx = max(data[i])
            off = i*1
            plt.plot(x, data[i]/mx+off)
        plt.yticks([])
        plt.xlabel(xlabel + "["+ x_um+"]")
        plt.title(title)
        if save:
            plt.savefig(self.folder + saveas+"."+ self.ext)
        if show:
            plt.show()
        plt.close()
        return saveas
