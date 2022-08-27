import math
import pandas as pd
import numpy as np
#import matplotlib
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import openpyxl
class Multiplot:
    def __init__(self, FileInstructions, argv = None):
        "read the ecxel file and save the data into the attributes of the class"
        scenario = pd.read_excel(FileInstructions+".xlsx",sheet_name="scenario")
        parameters = pd.read_excel(FileInstructions+".xlsx",sheet_name="parameters")
        variables = pd.read_excel(FileInstructions+".xlsx",sheet_name="variables")
        #GET INPUT FROM PARAMETERS FILE
        field = parameters["Field"]
        value = parameters["Value"]
        if argv is None:
            PATH_IP = value[list(field).index("Input_data_dir")]
            FileData = value[list(field).index("Input_data")]
            PATH_OP = value[list(field).index("Save_into")]
            pdf_name = value[list(field).index("Pdf_name")]
        else:
            PATH_IP = argv[1]
            FileData = argv[2]
            PATH_OP = argv[1]
            pdf_name = argv[2]
        self.width = value[list(field).index("Width")]
        self.hieght = value[list(field).index("Hieght")]
        #GET INPUT FROM SCENARIO FILE
        self.index = scenario["Section"]
        self.page = scenario["Page"].fillna(" ")
        self.title = scenario["Title"]
        try:
            self.range_i_x = scenario["Range_init_x"]#catch no range field
        except KeyError:
            self.range_i_x = None
        try:
            self.range_f_x = scenario["Range_fin_x"]
        except KeyError:
            self.range_f_x = None
            
        self.num_selection = 0
        for el in scenario.columns:
            if el == "Select" +str(self.num_selection+1):
                self.num_selection+=1
        """
        Select and the others are lists(they could be an objects) because we want to have n possible selection
        """
        self.select = []
        self.value = []
        self.range_select_min = []
        self.range_select_max = []
        for i in range(self.num_selection):
            for el in scenario.columns:
                if "Select"+str(i+1) == el:
                    self.select.append(scenario["Select"+str(i+1)])
                elif "Value"+str(i+1) == el:
                    self.value.append( scenario["Value"+str(i+1)])
                elif "Range_min"+str(i+1) == el:
                    self.range_select_min.append(scenario["Range_min"+str(i+1)])
                elif "Range_max"+str(i+1) == el:
                    self.range_select_max.append(scenario["Range_max"+str(i+1)])
                
            if len(self.select)> len(self.value):
                self.value.append(['']*len(self.index))
            if len(self.select)> len(self.range_select_min):
                self.range_select_min.append(['']*len(self.index))
            if len(self.select)> len(self.range_select_max):
                self.range_select_max.append(['']*len(self.index))
                
        self.Rows = scenario["Rows"]
        self.Columns = scenario["Columns"]
        self.row_title = scenario["Row_title"]
        self.col_title = scenario["Col_title"]
        self.reference = scenario["Reference"]
        self.multigraph = scenario["MultiGraph"]
        try:
            #TODO da estendere a select!
            """
            show once every n step
            """
            self.subsample = scenario["subsample"]
        except KeyError:
            self.subsample = 1
        self.x_axis = scenario["X-axis"]
        #self.y_axis = scenario["Y-axis"]
        self.num_subplots = 1
        self.y_subplots = []
        self.y_subplots.append(scenario["Y-axis"])
        for el in scenario.columns:
            """
            I could want to have multiple graph to show on a single step.
            """
            if el == "Y-axis" +str(self.num_subplots+1):
                self.num_subplots+=1
        for i in range(self.num_subplots):
            for el in scenario.columns:
                if "Y-axis"+str(i+1) == el:
                    self.y_subplots.append(scenario["Y-axis"+str(i+1)])
        self.y_scale = scenario["Scale"]
        self.y_lower = scenario["Y-lower"]
        self.y_upper = scenario["Y-upper"]
        try:
            self.vertical_lines = scenario["Vertical_lines"]
        except KeyError:
            self.vertical_lines = None
        #GET INPUT FROM VARIABLES FILE
        var_old_name = variables["Var_old_name"]
        var_new_name = variables["Var_new_name"]
        self.units = {}
        for i,u in zip(var_new_name,variables["Units"]):
            self.units[i]=u
        #read data
        self.Indata = self.read_data(filename = PATH_IP+FileData+".txt")
        for old,new in zip(var_old_name,var_new_name):
            self.Indata = self.Indata.rename(columns={old:new})
        self.pdf = PdfPages(PATH_OP+pdf_name+".pdf")
        self.plot()   
        
        
    def read_data(self, filename = None, data_numpy = None, data_titles = None):
        """
        If we override self.read_data into the function that we want we can use this program to plot from any source
        """
        if filename is not None:
            #TODO mettere caso excel
            return pd.read_csv(filename,delimiter="\t")
            #TODO ricorda che deve essere giusto il delimiter
        if data_numpy is not None and data_titles is not None:
            return pd.DataFrame(data_numpy, columns = data_titles)
        raise TypeError("you must enter some data to be parsed")
    def select_data(self, data_in, select, value = '', range_select_min = '', range_select_max = ''):
        """
        Selects the data in a given range.
        """
        if pd.isnull(select):
            return data_in
        if not math.isnan(value):
            return data_in[data_in[select]==value]
        if not math.isnan(range_select_min) and not math.isnan(range_select_max):
            return data_in[(data_in[select]>range_select_min) & (data_in[select]<range_select_max)]
        if not math.isnan(range_select_min):
            return data_in[data_in[select]>range_select_min]
        if not math.isnan(range_select_max):
            return data_in[data_in[select]<range_select_max]
                
        raise TypeError("you must enter a value or range to be selected")
    def plot(self):
        """
        It retrives the data, using pivot tables with the specifiers given in the init it generates the plot.
        """
        for idx in self.index:
            
            if self.page[idx] == " ":
                pages = [" "]
                skip_pages = True
            else:
                pages = self.remove_duplicate_order_list(self.Indata[self.page[idx]])
                skip_pages = False
            for page in pages:
                cont = 0
                for i in range(self.num_selection):
                    if i == 0:
                        data1 = self.Indata
                    if self.num_selection<0:
                        break
                    data1 = self.select_data(data_in = data1, select = self.select[i][idx],
                                                range_select_min = self.range_select_min[i][idx], range_select_max = self.range_select_max[i][idx], 
                                                value = self.value[i][idx] )
                    self.data = data1
                if pd.isnull(self.Rows[idx]):
                    row=0
                    col=0
                    nrows=int(math.sqrt(len(set(self.data[self.Columns[idx]]))))#fix subplots array size
                    ncols=math.ceil((len(set(self.data[self.Columns[idx]])))/nrows)#fix subplots array size
                    fig, outer = self.gen_page(nrows, ncols, idx, page)
                    for c in self.remove_duplicate_order_list(self.data[self.Columns[idx]]):
                        if skip_pages:
                            self.plot_single(data_pivot =self.data[(self.data[self.Columns[idx]]==c)],
                                          outer = outer[cont], fig = fig, c = c, idx = idx, col = col, row = row)
                        else:
                            print(self.data[(self.data[self.Columns[idx]]==c)&
                                         (self.data[self.page[idx]]==page)])
                            kjkjk=1
                            self.plot_single(data_pivot =self.data[(self.data[self.Columns[idx]]==c)&
                                         (self.data[self.page[idx]]==page)],
                                          outer = outer[cont], fig = fig, c = c, idx = idx, col = col, row = row)
                        col=col+1
                        cont+=1
                        if col==ncols:
                            row=row+1
                            col=0
                #CODE FOR PLOTTING BY ROWS AND COLUMNS: INDEPENDENTLY    
                else:
                    fig,outer = self.gen_page(nrows = len(set(self.data[self.Rows[idx]])),
                                                     ncols = len(set(self.data[self.Columns[idx]])),
                                                                 idx = idx,
                                                                 page = page)
                    for row,r in enumerate(self.remove_duplicate_order_list(self.data[self.Rows[idx]])):
                        for col,c in enumerate(self.remove_duplicate_order_list(self.data[self.Columns[idx]])):
                        
                            if skip_pages:
                                self.plot_single(data_pivot =self.data[(self.data[self.Columns[idx]]==c)&
                                             (self.data[self.Rows[idx]]==r)],
                                          outer = outer[cont], fig = fig, c = c, r = r,idx = idx, col= col, row = row )
                            else:
                                self.plot_single(data_pivot =self.data[(self.data[self.Columns[idx]]==c)&
                                             (self.data[self.Rows[idx]]==r)&
                                             (self.data[self.page[idx]]==page)],
                                          outer = outer[cont], fig = fig, c = c, r = r,idx = idx, col= col, row = row )
                            cont+=1
                plt.show()
                self.pdf.savefig(fig)
                plt.clf()
        self.pdf.close()
    def gen_page(self,nrows, ncols, idx, page):
        """
        Generates the plot structure.
        """
        fig = plt.figure(figsize=(self.width,self.hieght))
        if len(self.y_subplots)>1:
            space = 0.1
        else:
            space = 0
        outer = gridspec.GridSpec(nrows,ncols, wspace=space, hspace=space)
        #fig,axs = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(self.width,self.hieght),squeeze=False)
        #plt.subplots_adjust(hspace=0.01,wspace=0.01,left=0.09,bottom=0.09)
        fig.supxlabel(self.x_axis[idx]+" ("+self.units[self.x_axis[idx]]+")",fontsize=12,fontweight="bold")
        #fig.supylabel(self.y_axis[idx]+" ("+self.units[self.y_axis[idx]]+")",fontsize=12,fontweight="bold")
        fig.suptitle("%s"%self.title[idx]+"  %s"%self.page[idx]+"=%s"%page,fontsize=12,fontweight = 'bold')
        return (fig, outer)
    """
    def get_index_range(self,index, idx):
        range_low = 1
        range_high = 1
        if self.range_i_x is not None:
            range_low = (index>self.range_i_x[idx])
        if self.range_f_x is not None:
            range_high = (index<self.range_f_x[idx])
        return range_low & range_high
    """
    def plot_inner(self, fig, inner, pivot, label, idx):
        """
        Plots the data.
        """
        axs = plt.Subplot(fig, inner)
        axs.plot(pivot.index,pivot.values,marker='.',label=label)
        print(label)
        self.plot_reference(axs)
        fig.add_subplot(axs) 
        if self.vertical_lines is not None and self.vertical_lines[idx] is not None:  
            lines = self.vertical_lines[idx].split(",")
            for line in lines:
                axs.axvline(float(line))
        axs.grid(axis='both',which='major')
        axs.tick_params(axis="both",which="major",direction="in",labelsize=4)
        axs.legend(loc=0,fontsize=7)
        if pd.isnull(self.y_lower[idx]):#Y-limits
            pass
        else:
            axs.set_ylim([self.y_lower[idx],self.y_upper[idx]])
            if self.range_i_x is None or self.range_i_x[idx] is None:
                min_x = min(self.data[self.x_axis[idx]]) 
            else:
                min_x = self.range_i_x[idx]
            if self.range_f_x is None or self.range_f_x[idx] is None:
                max_x = max(self.data[self.x_axis[idx]])
            else:
                max_x = self.range_f_x[idx]
            axs.set_xlim(left = min_x, right = max_x)
                
    def plot_single(self, data_pivot, outer, fig, c, idx, col, row, r = None):
        """
        Selects the data to be plot in a single quadrant.
        """
        inner = gridspec.GridSpecFromSubplotSpec(len(self.y_subplots), 1,
                    subplot_spec=outer, wspace=0, hspace=0)
        for i, y_subplot in enumerate(self.y_subplots):
            if pd.isnull(self.multigraph[idx]):#reshape dataframe for multiplot
                pivot = pd.pivot_table(data_pivot,
                                       index = self.x_axis[idx],
                                       values = y_subplot[idx])
                if r is None:
                    self.plot_inner(fig = fig, inner = inner[i], pivot = pivot, label = "%s"%c, idx = idx)
                else:
                    self.plot_inner(fig = fig, inner = inner[i], pivot = pivot, label = "%s"%r+", %s"%c, idx = idx)
            else:
                #np.arange( self.subsample[idx]
                pivot = pd.pivot_table(data_pivot,
                                       columns = self.multigraph[idx],
                                       index = self.x_axis[idx],
                                       values = y_subplot[idx])
                idx_multi = np.arange(0,len(pivot),self.subsample[idx] )
                #pivot = pivot.sample(self.subsample[idx],axis = 1).sort_values('time', axis = 1)
                self.plot_inner(fig = fig, inner = inner[i], pivot = pivot[idx_multi], label = pivot.columns, idx = idx)
            
            #TODO: have labells everywhere

            #TODO add here self.reference that will be called in a new class
            if self.row_title[idx] == True and col==0:
                self.set_ylabel(fig = fig, text = "%s"%self.Columns[idx]+" \n= " +"%s"%c, row = row, outer = outer)
                #outer.set_ylabel("%s"%self.Columns[idx]+" \n= " +"%s"%c)
            
            if self.col_title[idx] == True and row==0:
                self.set_xlabel(fig = fig, text = "%s"%self.Columns[idx]+" \n= " +"%s"%c, col = col, outer = outer)
                #outer.set_title("%s"%self.Columns[idx]+" \n= "+"%s"%c)
            #self.add_subplot_border(axs, width = 2, color = 'b')
            
            
    def set_ylabel(self, fig, text, row, outer):
        nrows = outer._gridspec._nrows
        step = (900 - (110+93))/900
        step/= nrows
        off = 93/900
        pos = (nrows - row)*step  - step/2  + off
        fig.text(0.07, pos, text, va='center', rotation='vertical')
    def set_xlabel(self, fig, text, col, outer):
        ncols = outer._gridspec.ncols
        step = (1407 - (150+140))/1407
        step/= ncols
        off = 140/900
        pos = ( col)*step    + off
        fig.text(pos, 0.9, text, va='center')
    def add_subplot_border(self,ax, width=1, color=None ):

        fig = ax.get_figure()

        # Convert bottom-left and top-right to display coordinates
        x0, y0 = ax.transAxes.transform((0, 0))
        x1, y1 = ax.transAxes.transform((1, 1))

        # Convert back to Axes coordinates
        x0, y0 = ax.transAxes.inverted().transform((x0, y0))
        x1, y1 = ax.transAxes.inverted().transform((x1, y1))

        rect = plt.Rectangle(
            (x0, y0), x1-x0, y1-y0,
            color=color,
            transform=ax.transAxes,
            zorder=-1,
            lw=2*width+1,
            fill=None,
        )
        fig.patches.append(rect)  
    def remove_duplicate_order_list(self, el):
        #TODO use unique?
        return sorted(list(set(el)))
    def plot_reference(self, axs):
        pass
     
        

