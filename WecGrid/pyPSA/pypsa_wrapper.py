"""
pyPSA Class Module file
"""

# Standard Libraries
import os
import sys
from datetime import datetime, timezone, timedelta

# 3rd Party Libraries
import pypsa
import pandas as pd
import cmath
import matlab.engine
import pypower.api as pypower

# Local Libraries (updated with relative imports)
from ..utilities.util import read_paths  # Relative import for utilities/util.py

# Initialize the PATHS dictionary
PATHS = read_paths()
CURR_DIR = os.path.dirname(os.path.abspath(__file__))

class pyPSAWrapper:
    def __init__(self, case, WecGridCore):
        self.case_file = case
        self.pypsa_history = {}
        self.pypsa_object_history = {}
        self.dataframe = pd.DataFrame()
        self.WecGridCore = WecGridCore  # Reference to the parent 

    def initialize(self, solver):
        """
        Description: Initializes a pyPSA case, uses the topology passed at original initialization
        input:
            solver: the solver you want to use supported by PSSe, "fnsl" is a good default (str)
        output: None
        notes: only works with .raw files, needs to use matpower->pypower->pyPSA conversion process
        """

        eng = matlab.engine.start_matlab()
        eng.workspace["case_path"] = self.case_file
        eng.eval("mpc = psse2mpc(case_path)", nargout=0)
        eng.eval("savecase('here.mat',mpc,1.0)", nargout=0)

        # Load the MATPOWER case file from a .mat file
        ppc = pypower.loadcase("./here.mat") # TODO: this is hardcode, should be passing the mat file, tbh I forgot what this is for. 

        # Convert Pandapower network to PyPSA network
        pypsa_network = pypsa.Network()
        pypsa_network.import_from_pypower_ppc(ppc, overwrite_zero_s_nom=True)
        pypsa_network.set_snapshots([datetime.now().strftime("%m/%d/%Y %H:%M:%S")])

        pypsa_network.pf()

        self.dataframe = pypsa_network.buses
        self.pypsa_object = pypsa_network
        self.pypsa_history[-1] = self.dataframe
        self.pypsa_object_history[-1] = self.pypsa_object
        print("pyPSA initialized")

    def ac_injection(self, p, v, time):
        """
        Description: WEC AC injection for pypsa powerflow solver
        input:
            p - a vector of active power values in order of bus num
            v - a vector of voltage mag PU values in order of bus num
            time: (Int)
        output: no output but pypsa_dataframe is updated and so is pypsa_history
        """
        for idx, bus in enumerate(self.wecBus_nums):

            self.pypsa_object.generators.loc[
                self.pypsa_object.generators.bus == str(bus), "v_set_pu"
            ] = v[idx]
            self.pypsa_object.generators.loc[
                self.pypsa_object.generators.bus == str(bus), "p_set"
            ] = p[idx]

        self.run_powerflow()
        self.pypsa_history[time] = self.pypsa_dataframe

    def run_powerflow(self):
        """
        Description: This function runs the powerflow for pyPSA model
        Input: None
        output: None
        """
        # this can be updated, it's a bit sloppy
        temp = self.pypsa_object.copy()
        temp.pf()
        self.pypsa_object = temp.copy()
        self.pypsa_dataframe = self.pypsa_object.buses
        
    # TODO: build out the viz function for pyPSA
