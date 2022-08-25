import sys
sys.path.append('../wec-grid-code') #
import wec_grid_class as wg
import pandas as pd
from uniplot import plot
import sqlite3


def main():
  # Read in case file 
  case14 = r"C:\Users\barajale\Desktop\research_code\case14.raw"

  # initalize WEC GRID Object
  pf = wg.Wec_grid(case14)

  # Run first Power Flow and get bus values
  pf.run_powerflow
  pf.get_values(['BASE', 'PU', 'ANGLED', 'P', 'Q'])
  print(pf.dataframe)

  # connect to Database and open in df
  con = sqlite3.connect('../input_files/WEC_database.db')
  cur = con.cursor()
  wec_gen_values = pd.read_sql_query("SELECT * from WecOutput", con)
  wec_gen_values

  #for i in range(len(wec_gen_values)):
  time = [] 
  history_swing = []
  history_injection = []
  history = []

  wec_bus = 3
  for i in range(len(wec_gen_values)):
      print("Time: {}".format(wec_gen_values.iloc[i].Time))
      print("P setpoint: {}".format(wec_gen_values.iloc[i].P))
      pf.dc_injection(wec_bus, wec_gen_values.iloc[i].P)
      history_swing.append(pf.dataframe.P[0])
      history_injection.append(pf.dataframe.P[wec_bus - 1])
      time.append(wec_gen_values.iloc[i].Time)
      history.append(pf.dataframe)
      print("========================")

  plot(xs=time, ys=history_swing, lines=True, title="Swing BUS Chart", y_unit=" P", x_unit=" time")
  plot(xs=time, ys=history_injection, lines=True, title="PV BUS 3 Chart",y_unit=" P", x_unit=" time")

main()