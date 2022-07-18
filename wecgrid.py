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

def format_voltage_bus(temp_dict):
    """
    Descrtiption: 
    input:
    output: Dataframe of 
    """
    df = pd.DataFrame.from_dict(temp_dict, orient ='index') 
    df = df.reset_index()
    df = df.rename(columns={0:"Voltages", 'index':"Bus"})
    return df

def bus_voltages_to_list(list_bus):
    """
    Descrtiption:
    input:
    output:
    """
    temp_dict = {}
    for i in range(len(list_bus)):
        temp_dict[str("BUS {}".format(i+1))] =  list_bus[i]
    return temp_dict

def get_values(lst):
    """
    Descrtiption: 
    input: List of Parameters such as BASE, PU, KV, ANGLED 
    output: Dataframe of the selected parameters for each bus
    """
    temp_dict = {}
    for i in range(len(lst)):
        if lst[i] != "P" and lst[i] != "Q":
            ierr, bus_voltages = psspy.abusreal(-1, string=lst[i])
            bus_add = {}
            for j in range(len(bus_voltages[0])):
                bus_add['BUS {}'.format(j+1)] = bus_voltages[0][j]
                temp_dict[lst[i]] = bus_voltages[0]
            temp_dict[lst[i]] = bus_add

    df = pd.DataFrame.from_dict(temp_dict) 
    df = df.reset_index()
    df = df.rename(columns={'index':"Bus"})
    df['Type'] = psspy.abusint(-1, string="TYPE")[1][0]
    if "P" in lst:
        df['P'] = 0
        df = get_p_or_q(df, 'P')
    if "Q" in lst:
        df['Q'] = 0
        df = get_p_or_q(df, 'Q')
    return df

def get_pq(bus_df):
    """
    Descrtiption: retre P (activate) Q (reactive) Voltage (in PU) and Voltage Angle for each Bus in the current loaded case
    input:
    output:
    """
    ierr, from_to = psspy.aflowint(sid=-1, string=["FROMNUMBER", "TONUMBER"])
    ierr, p_q = psspy.aflowreal(sid=-1, string=["P","Q"])
    from_to_p_q = from_to + p_q
    branches = zip(*from_to_p_q)

    for ibus in busNum():
        p = 0
        q = 0
        for i in range(len(from_to_p_q[0])):
            if ibus == from_to_p_q[0][i]:
                p += from_to_p_q[2][i]
                q += from_to_p_q[3][i]
        bus_df.loc[bus_df['Bus'] == 'BUS {}'.format(ibus), 'P'] = p
        bus_df.loc[bus_df['Bus'] == 'BUS {}'.format(ibus), 'Q'] = q
    return bus_df


def get_p_or_q(bus_df, letter):
    """
    Descrtiption: retre P (activate) Q (reactive) Voltage (in PU) and Voltage Angle for each Bus in the current loaded case
    input:
    output:
    """
    ierr, from_to = psspy.aflowint(sid=-1, string=["FROMNUMBER", "TONUMBER"])
    ierr, p_q = psspy.aflowreal(sid=-1, string=[letter])
    from_to_p_q = from_to + p_q
    branches = zip(*from_to_p_q)
    for ibus in busNum():
        temp = 0
        for i in range(len(from_to_p_q[0])):
            if ibus == from_to_p_q[0][i]:
                temp += from_to_p_q[2][i]    
        bus_df.loc[bus_df['Bus'] == 'BUS {}'.format(ibus), letter] = temp
    return bus_df

def busNum():
    """
    Descrtiption: Returns the number of Buses in the currently loaded case
    input: None
    output: Number of Buses
    """
    psspy.bsys(0,0,[0.0,0.0],1,[1],0,[],0,[],0,[])
    ierr,all_bus = psspy.abusint(0,1,['number'])
    return all_bus[0]