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
    pf = wg.Wec_grid(case14,'fnsl', 3)
    print(pf.dataframe)

    # # Run first Power Flow and get bus values
    # pf.run_powerflow
    # pf.get_values(['BASE', 'PU', 'ANGLED', 'P', 'Q'])
    # print(pf.dataframe)

    # # connect to Database and open in df
    # con = sqlite3.connect('../input_files/WEC_database.db')
    # cur = con.cursor()
    # injection = pd.read_sql_query("SELECT * from WecOutput", con)
    # injection

    # history_swing = pd.DataFrame(pf.dataframe.loc[pf.dataframe['Bus'] == 'BUS 1'])
    # history_swing['time'] = None
    # history_injection = pd.DataFrame(pf.dataframe.loc[pf.dataframe['Bus'] == 'BUS 3'])
    # history_injection['time'] = None
    # history_df = [] 
    # test = []
    # wec_bus = 3

    # for i in range(len(injection)):
    #     print("Time: {}".format(injection.iloc[i].Time))
    #     print("P setpoint: {}".format(injection.P.iloc[i]))
        
    #     pf.dc_injection(wec_bus, injection.P.iloc[i], 'fnsl')
    #     pf.dataframe

    #     temp = pd.DataFrame(pf.dataframe.loc[pf.dataframe['Bus'] == 'BUS 1'])
    #     temp['time'] = injection.iloc[i].Time
    #     history_swing = history_swing.append(temp)
        
    #     temp = pd.DataFrame(pf.dataframe.loc[pf.dataframe['Bus'] == 'BUS 3'])
    #     temp['time'] = injection.iloc[i].Time
    #     history_injection = history_injection.append(temp)
            
    
        
    # #     history_swing.append(pf.dataframe.loc[pf.dataframe['Bus'] == 'BUS 1'])
    # #     test.append(pf.dataframe.loc[pf.dataframe['Bus'] == 'BUS 1'])
    # #     history_injection.append(pf.dataframe.P[wec_bus - 1])
    # #     time.append(wec_gen_values.iloc[i].Time)
    # #     history.append(pf.dataframe)
    # #     print(pf.dataframe)
    #     print("========================")
    
    # print(history_injection)
    # exit()

main()

