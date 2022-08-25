import sys
sys.path.append('../wec-grid-code') # 
import pandas as pd
from uniplot import plot


def main():
  # Read in case file 
  case14 = r"C:\Users\barajale\Desktop\research_code\case14.raw"

  # initalize WEC GRID Object
  pf = wg.Wec_grid(case14)

  # Run first Power Flow and get bus values
  pf.run_powerflow
  pf.get_values(['BASE', 'PU', 'ANGLED', 'P', 'Q'])
  print(pf.dataframe)

  # Read in Wec set point values
  wec_gen_values = pd.read_csv("./input_files/WECgen_data.csv")
  print(pf.dataframe.P[2])

  #for i in range(len(wec_gen_values)):
  time = [] 
  history_swing = []
  history_injection = []

  for i in range(len(wec_gen_values)):
    # choose the next time step set point and run the solver 
    print("Time: {}".format(wec_gen_values.iloc[i].time))
    print("P setpoint: {}".format(wec_gen_values.iloc[i].pg))
    pf.dc_injection(3, wec_gen_values.iloc[i].pg)
    #print(pf.dataframe)

    history_swing.append(pf.dataframe.P[0])
    history_injection.append(pf.dataframe.P[2])
    time.append(wec_gen_values.iloc[i].time)
    print("========================")

  plot(xs=time, ys=history_swing, lines=True, title="Swing BUS Chart", y_unit=" P", x_unit=" time")
  plot(xs=time, ys=history_injection, lines=True, title="PV BUS 3 Chart",y_unit=" P", x_unit=" time")

main()