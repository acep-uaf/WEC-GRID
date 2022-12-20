# AC power flow

import sys
sys.path.append('../wec-grid-code') #
import wec_grid_class as wg
import pandas as pd
import sqlite3


def main():
  # Read in case file
  case14 = r"../input_files/case14.raw"

  # initalize WEC GRID Object
  pf = wg.Wec_grid(case14,"fnsl",3)

  # Run first Power Flow and get bus values
  print(pf.dataframe)

  # Read in Wec set point values
  con = sqlite3.connect("../input_files/r2g_database.db")
  injection = pd.read_sql_query("SELECT * from WEC_output", con)
  print(injection)

  for i in range(len(injection)):
    print("Time: {}".format(injection.iloc[i].time))
    print("P setpoint: {}".format(injection.pg.iloc[i]))

    pf.ac_injection(pf.wecBus_num, injection.pg.iloc[i], injection.vs.iloc[i],pf.solver, injection.iloc[i].time)
    print("========================")
    exit()
main()
