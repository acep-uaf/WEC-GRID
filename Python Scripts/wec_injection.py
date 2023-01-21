# DC/AC power flow

import sys
sys.path.append('../wec-grid-code') # 
import wec_grid_class as wg
import sqlite3
import pandas as pd


def main():
  # Read in case file
  case14 = r"../input_files/case14.raw"
  solver = input("Solver: ")

  # initalize WEC GRID Object
  pf = wg.Wec_grid(case14,solver,3)

  # Run first Power Flow and get bus values
  print(pf.dataframe)

  sim = input("Run WEC Simulator?")
  if sim == "yes":
    pf.run_WEC_Sim()

  # Read in Wec set point values
  con = sqlite3.connect("../input_files/r2g_database.db")
  injection = pd.read_sql_query("SELECT * from WEC_output", con)
  print(injection)

  for i in range(len(injection)):
    print("Time: {}".format(injection.iloc[i].time))
    print("P setpoint: {}".format(injection.pg.iloc[i]))
    if pf.solver == 'AC':
        pf.ac_injection(pf.wecBus_num, injection.pg.iloc[i], injection.vs.iloc[i],
                        pf.solver, injection.iloc[i].time)
    if pf.solver == 'DC':
        pf.dc_injection(pf.wecBus_num, injection.pg.iloc[i], pf.solver, injection.iloc[i].time)
    print("========================")
  exit()
main()
