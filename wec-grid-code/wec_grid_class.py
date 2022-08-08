# Wec_grid_class 
import os,sys

# Path stuff
sys.path.append(r"C:\Program Files\PTI\PSSE35\35.3\PSSPY37")
sys.path.append(r"C:\Program Files\PTI\PSSE35\35.3\PSSBIN")
sys.path.append(r"C:\Program Files\PTI\PSSE35\35.3\PSSLIB")
sys.path.append(r"C:\Program Files\PTI\PSSE35\35.3\EXAMPLE")
os.environ['PATH'] = (r"C:\Program Files\PTI\PSSE35\35.3\PSSPY37;" 
  + r"C:\Program Files\PTI\PSSE35\35.3\PSSBIN;" 
  + r"C:\Program Files\PTI\PSSE35\35.3\EXAMPLE;" + os.environ['PATH'])

# imports all the GOOOOOOD stuff we need for our program / automation stuff 
import glob
from pathlib import Path
import pandas as pd
import psse35
psse35.set_minor(3)

import psspy
psspy.psseinit(50)


class Wec_grid:
    def __init__(self, case):
        self.case_file = case
        self.dataframe = pd.DataFrame()
        psspy.read(1, case)
        self.lst = []

    def run_powerflow(self):
        psspy.fnsl()

    def get_values(self, lst):
        """
        Descrtiption: 
        input: List of Parameters such as BASE, PU, KV, ANGLED 
        output: Dataframe of the selected parameters for each bus
        """
        self.lst = lst
        temp_dict = {}
        for i in range(len(lst)):
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
        ierr, from_to = psspy.aflowint(sid=-1, string=["FROMNUMBER", "TONUMBER"])
        ierr, p_q = psspy.aflowreal(sid=-1, string=[letter])
        from_to_p_q = from_to + p_q
        branches = zip(*from_to_p_q)
        for ibus in self.busNum():
            temp = 0
            for i in range(len(from_to_p_q[0])):
                if ibus == from_to_p_q[0][i]:
                    temp += from_to_p_q[2][i]    
            self.dataframe.loc[self.dataframe['Bus'] == 'BUS {}'.format(ibus), letter] = temp

    def busNum(self):
        """
        Descrtiption: Returns the number of Buses in the currently loaded case
        input: None 
        output: Number of Buses
        """
        psspy.bsys(0,0,[0.0,0.0],1,[1],0,[],0,[],0,[])
        ierr,all_bus = psspy.abusint(0,1,['number'])
        return all_bus[0]
    
    def dc_injection(self, ibus, p):
        ierr = psspy.machine_chng_3(ibus, "1", [], [p])
        if ierr > 0:
            print("Failed | machine_chng_3 code = {}".format(ierr))
        self.run_powerflow()
        self.get_values(self.lst)

    # def get_pq(bus_df):
    #     """
    #     Descrtiption: retre P (activate) Q (reactive) Voltage (in PU) and Voltage Angle for each Bus in the current loaded case
    #     input:
    #     output:
    #     """
    #     ierr, from_to = psspy.aflowint(sid=-1, string=["FROMNUMBER", "TONUMBER"])
    #     ierr, p_q = psspy.aflowreal(sid=-1, string=["P","Q"])
    #     from_to_p_q = from_to + p_q
    #     branches = zip(*from_to_p_q)

    #     for ibus in busNum():
    #         p = 0
    #         q = 0
    #         for i in range(len(from_to_p_q[0])):
    #             if ibus == from_to_p_q[0][i]:
    #                 p += from_to_p_q[2][i]
    #                 q += from_to_p_q[3][i]
    #         bus_df.loc[bus_df['Bus'] == 'BUS {}'.format(ibus), 'P'] = p
    #         bus_df.loc[bus_df['Bus'] == 'BUS {}'.format(ibus), 'Q'] = q
    #     return bus_df

    # def bus_voltages_to_list(list_bus):
    #     """
    #     Descrtiption:
    #     input:
    #     output:
    #     """
    #     temp_dict = {}
    #     for i in range(len(list_bus)):
    #         temp_dict[str("BUS {}".format(i+1))] =  list_bus[i]
    #     return temp_dict