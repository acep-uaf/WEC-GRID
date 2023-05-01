# DC/AC power flow

import os,sys
sys.path.append('../wec-grid-code') #
import wec_grid_class as wg
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import time


def main():
  # Read in case file
  case24 = r"../input_files/RTS96DYN/IEEE_24_bus.RAW"
  solver = input("Solver: ")

  # initalize WEC GRID Object
  pf = wg.Wec_grid(case24, solver, [2, 13, 21])

  # Run first Power Flow and get bus values
  print(pf.dataframe)

  #os.remove("../input_files/WEC-SIM.db")
  start = time.time()
  for i in range(1, 4):
    pf.run_WEC_Sim(wec_id=i, sim_length=24*60*60, Tsample=300, waveHeight=2.5, wavePeriod=8,
                   waveSeed=np.random.randint(999999999))
  end = time.time()
  print(end - start)

  # # Read in Wec set point values
  # con = sqlite3.connect("../input_files/r2g_database.db")
  # injection = pd.read_sql_query("SELECT * from WEC_output", con)
  # print(injection)
  #
  # for i in range(len(injecti
  #   print("Time: {}".format(injection.iloc[i].time))
  #   print("P setpoint: {}".format(injection.pg.iloc[i]))
  #   if pf.solver == 'AC':
  #       pf.ac_injection(pf.wecBus_num, injection.pg.iloc[i], injection.vs.iloc[i],
  #                       pf.solver, injection.iloc[i].time)
  #   if pf.solver == 'DC':
  #       pf.dc_injection(pf.wecBus_num, injection.pg.iloc[i], pf.solver, injection.iloc[i].time)
  #   print("========================")
  # exit()
main()
