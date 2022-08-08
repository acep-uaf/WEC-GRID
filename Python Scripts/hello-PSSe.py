import wec_grid_class as wg
import pandas as pd
from uniplot import plot


def main():
    # Read in case file 
    case = r"C:\Users\barajale\Desktop\research_code\case14.raw" # Path to your raw file

    pf_obj = wg.Wec_grid(case)

    pf_obj.run_powerflow

main()