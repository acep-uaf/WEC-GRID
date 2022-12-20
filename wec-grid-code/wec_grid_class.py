# Wec_grid_class 
import os,sys
from pickle import HIGHEST_PROTOCOL

# Path stuff
sys.path.append(r"C:\Program Files\PTI\PSSE35\35.3\PSSPY37")
sys.path.append(r"C:\Program Files\PTI\PSSE35\35.3\PSSBIN")
sys.path.append(r"C:\Program Files\PTI\PSSE35\35.3\PSSLIB")
sys.path.append(r"C:\Program Files\PTI\PSSE35\35.3\EXAMPLE")
os.environ['PATH'] = (r"C:\Program Files\PTI\PSSE35\35.3\PSSPY37;" 
  + r"C:\Program Files\PTI\PSSE35\35.3\PSSBIN;" 
  + r"C:\Program Files\PTI\PSSE35\35.3\EXAMPLE;" + os.environ['PATH'])

# imports all the GOOOOOOD stuff we need for our program / automation stuff 

import pandas as pd
import psse35
psse35.set_minor(3)
import numpy as np
import matplotlib.pyplot as plt
import psspy
psspy.psseinit(50)
import matlab.engine


class Wec_grid:
    def __init__(self, case, solver, wec_bus):
        # initalized variables and files
        self.case_file = case
        self.lst_param = ['BASE', 'PU', 'ANGLED', 'P', 'Q']
        self.dataframe = pd.DataFrame()
        self.wecBus_num = wec_bus
        self.history = {}
        self.solver = solver
        self._i = psspy.getdefaultint()
        self._f = psspy.getdefaultreal()
        self._s = psspy.getdefaultchar()

        # initialization functions
        psspy.read(1, case)
        self.run_powerflow(self.solver)

        # program variables
        self.swingBus = self.dataframe.loc[self.dataframe['Bus'] == 'BUS 1']
        self.swingBus.insert(0, 'time', None)
        self.wecBus = self.dataframe.loc[self.dataframe['Bus'] == 'BUS {}'.format(str(self.wecBus_num))]
        self.wecBus.insert(0, 'time', None)
        self.history['Start'] = self.dataframe

    def clear(self):
        # initalized variables and files
        self.lst_param = ['BASE', 'PU', 'ANGLED', 'P', 'Q']
        self.dataframe = pd.DataFrame()
        self.history = {}

        # initialization functions
        psspy.read(1, self.case_file)
        self.run_powerflow(self.solver)

        # program variables
        self.swingBus = self.dataframe.loc[self.dataframe['Bus'] == 'BUS 1']
        self.swingBus.insert(0, 'time', None)
        self.wecBus = self.dataframe.loc[self.dataframe['Bus'] == 'BUS {}'.format(str(self.wecBus_num))]
        self.wecBus.insert(0, 'time', None)
        self.history['Start'] = self.dataframe

    # def run_wec_sim(self):
    #     eng = matlab.engine.start_matlab()
    #     eng.cd(".\input_files\W2G_RM3")
    #     path = r'C:\Users\alex_barajas\Desktop\WEC-Sim\source'  # Update to match your WEC-SIM source location
    #     eng.addpath(eng.genpath(path), nargout=0)
    #     eng.w2gSim(nargout=0)
    #     eng.WECsim_to_PSSe_dataFormatter(nargout=0)

    def run_powerflow(self,solver):
        if solver == 'fnsl':
            psspy.fnsl()
            self.get_values()
        elif solver == 'GS':
            psspy.solv()
            self.get_values()
        elif solver == 'DC':
            psspy.dclf_2(1, 1, [1,0,1,2,1,1],[0,0,0], '1')
            self.get_values()
        else:
            print("error in run_pf")

    def get_values(self):
        """
        Descrtiption: 
        input: List of Parameters such as BASE, PU, KV, ANGLED 
        output: Dataframe of the selected parameters for each bus
        """
        lst = self.lst_param
        temp_dict = {}
        for i in range(len(lst)):
            #print('here {}'.format(i))
            if lst[i] != "P" and lst[i] != "Q":
                ierr, bus_voltages = psspy.abusreal(-1, string=lst[i])
                bus_add = {}
                for j in range(len(bus_voltages[0])):
                    bus_add['BUS {}'.format(j+1)] = bus_voltages[0][j]
                    temp_dict[lst[i]] = bus_voltages[0]
                temp_dict[lst[i]] = bus_add
                

        self.dataframe = pd.DataFrame.from_dict(temp_dict) 
        self.dataframe = self.dataframe.reset_index()
        self.dataframe = self.dataframe.rename(columns={'index':"Bus"})
        self.dataframe['Type'] = psspy.abusint(-1, string="TYPE")[1][0]
        self.dataframe.insert(0, "BUS_ID", range(1, 1 + len(self.dataframe)))
        self.addGeninfo()
        self.addLoadinfo()
        
        if "P" in lst:
            self.dataframe['P'] = 0 # initalize P column
            self.get_p_or_q('P')

        if "Q" in lst:
            self.dataframe['Q'] = 0 # initalize Q column
            self.get_p_or_q('Q')
               
    def get_p_or_q(self, letter):
        """
        Descrtiption: retre P (activate) Q (reactive) Voltage (in PU) and Voltage Angle for each Bus in the current loaded case
        input:
        output:
        """
        gen = self.dataframe['{} Gen'.format(letter)]
        load = self.dataframe['{} Load'.format(letter)]
        temp = []
        for i in range(len(gen)):
            if (not np.isnan(gen[i])) and (not np.isnan(load[i])):
                temp.append(gen[i] - load[i])
            elif np.isnan(gen[i]) and np.isnan(load[i]):
                temp.append(None)
            else:
                if np.isnan(gen[i]):
                    temp.append(-load[i])
                else:
                    temp.append(gen[i])
        self.dataframe['{}'.format(letter)] = temp
            
    def busNum(self):
        """
        Descrtiption: Returns the number of Buses in the currently loaded case
        input: None 
        output: Number of Buses
        """
        psspy.bsys(0,0,[0.0,0.0],1,[1],0,[],0,[],0,[])
        ierr,all_bus = psspy.abusint(0,1,['number'])
        return all_bus[0]
    
    def dc_injection(self, ibus, p, pf_solver, time):
        """
        Descrtiption:
        input:
        output:
        """
        ierr = psspy.machine_chng_3(ibus, "1", [], [p])
        if ierr > 0:
            print("Failed | machine_chng_3 code = {}".format(ierr))
        #self.run_powerflow('DC')
        psspy.dclf()
        self.get_values()

        temp = pd.DataFrame(self.dataframe.loc[self.dataframe['Bus'] == 'BUS 1'])
        temp.insert(0, 'time', time)
        self.swingBus = self.swingBus.append(temp)


        temp = pd.DataFrame(self.dataframe.loc[self.dataframe['Bus'] == 'BUS {}'.format(str(self.wecBus_num))])
        temp.insert(0,'time', time)
        self.wecBus = self.wecBus.append(temp)

        self.history[time] = self.dataframe
        
    def ac_injection(self, ibus, p, v, pf_solver, time):
        """
        Descrtiption:
        input:
        output:
        """
        ierr = psspy.machine_chng_3(ibus, "1", [], [p])
        if ierr > 0:
            print("Failed | machine_chng_3 code = {}".format(ierr))
            
        ierr = psspy.bus_chng_4(ibus, 0, [],[self._f, v])
        if ierr > 0:
            print("Failed | bus_chng_4 code = {}".format(ierr))
            
        self.run_powerflow(pf_solver)
        temp = pd.DataFrame(self.dataframe.loc[self.dataframe['Bus'] == 'BUS 1'])
        temp.insert(0, 'time', time)
        self.swingBus = self.swingBus.append(temp)
        temp = pd.DataFrame(self.dataframe.loc[self.dataframe['Bus'] == 'BUS {}'.format(str(self.wecBus_num))])
        temp.insert(0,'time', time)
        self.wecBus = self.wecBus.append(temp)

        self.history[time] = self.dataframe

    def plotSwingBus(self):
        """
        Descrtiption:
        input:
        output:
        """
        # fig, axs = plt.subplots(2)
        # plt.plot(self.swingBus.time, self.swingBus[letter], marker="o", markersize=5, markerfacecolor="green")
        # plt.xlabel("Time (sec)")
        # plt.ylabel("{} in MW".format(letter))
        # plt.title("Swing bus")
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle("Swing bus")
        ax1.plot(self.swingBus.time, self.swingBus["P"], marker="o", markersize=5, markerfacecolor="green")
        ax2.plot(self.swingBus.time, self.swingBus["Q"], marker="o", markersize=5, markerfacecolor="green")
        ax1.set(xlabel="Time(sec)", ylabel="P(MW)")
        ax2.set(xlabel="Time(sec)", ylabel="Q(MW)")
        plt.show()
    
    def plotWecBus(self, mode="Gen"):
        """
        Descrtiption:
        input:
        output:
        """
        if mode == "Gen":
            fig, (ax1, ax2) = plt.subplots(2)
            fig.suptitle("Swing bus")
            ax1.plot(self.wecBus.time, self.wecBus["P"], marker="o", markersize=5, markerfacecolor="green")
            ax2.plot(self.wecBus.time, self.wecBus["Q"], marker="o", markersize=5, markerfacecolor="green")
            ax1.set(xlabel="Time(sec)", ylabel="P (MW)")
            ax2.set(xlabel="Time(sec)", ylabel="Q (sMW)")
            plt.show()
        else: 
            fig, (ax1, ax2) = plt.subplots(2)
            fig.suptitle("Swing bus")
            ax1.plot(self.wecBus.time, self.wecBus["P Gen"], marker="o", markersize=5, markerfacecolor="green")
            ax2.plot(self.wecBus.time, self.wecBus["Q Gen "], marker="o", markersize=5, markerfacecolor="green")
            ax1.set(xlabel="Time(sec)", ylabel="P Gen (MW)")
            ax2.set(xlabel="Time(sec)", ylabel="Q Gen (MW)")
          
    def addGeninfo(self):
        """
        Descrtiption:
        input:
        output:
        """
        buses = []
        for string in psspy.amachchar(-1,1,'NAME')[1][0]:
            buses.append(self.findBusNum(string))

        temp = psspy.amachcplx(-1,1,'PQGEN')
        pointer = 0 
        pointer_1 = 0
        p_gen_df_list = []
        q_gen_df_list = []

        for index, row in self.dataframe.iterrows():
            if row['BUS_ID'] == buses[pointer]:
                p_gen_df_list.append(temp[1][0][pointer].real)
                q_gen_df_list.append(temp[1][0][pointer].imag)
                pointer +=1
                if pointer >= len(buses):
                    pointer = 0
            else:
                p_gen_df_list.append(None)
                q_gen_df_list.append(None)
        self.dataframe['P Gen'] = p_gen_df_list
        self.dataframe['Q Gen'] = q_gen_df_list

    def addLoadinfo(self):
        """
        Descrtiption:
        input:
        output:
        """
        buses = []
        for string in psspy.aloadchar(-1,1,'NAME')[1][0]:
            buses.append(self.findBusNum(string))

        temp = psspy.aloadcplx(-1, 1, "MVAACT")
        pointer = 0 
        pointer_1 = 0
        p_load_df_list = []
        q_load_df_list = []

        for index, row in self.dataframe.iterrows():
            if row['BUS_ID'] == buses[pointer]:

                p_load_df_list.append(temp[1][0][pointer].real)
                q_load_df_list.append(temp[1][0][pointer].imag)
                pointer +=1
            else:
                p_load_df_list.append(None)
                q_load_df_list.append(None)
        self.dataframe['P Load'] = p_load_df_list
        self.dataframe['Q Load'] = q_load_df_list
        
    def findBusNum(self,string_bus_name):
        """
        Descrtiption:
        input:
        output:
        """
        temp_string = ''
        for i in string_bus_name:
            if i.isdigit():
                temp_string += i
        return int(temp_string)