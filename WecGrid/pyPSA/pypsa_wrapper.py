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
from ..viz.pypsa_viz import PyPSAVisualizer

# Initialize the PATHS dictionary
PATHS = read_paths()
CURR_DIR = os.path.dirname(os.path.abspath(__file__))


class pyPSAWrapper:
    def __init__(self, case, WecGridCore):
        self.case_file = case
        self.pypsa_history = {}
        self.pypsa_object_history = {}
        self.dataframe = pd.DataFrame()
        self.flow_data = {}
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
        ppc = pypower.loadcase(
            "./here.mat"
        )  # TODO: this is hardcode, should be passing the mat file, tbh I forgot what this is for.

        # Convert Pandapower network to PyPSA network
        #pypsa_network = pypsa.Network()
        self.pypsa_object = pypsa.Network()
        self.pypsa_object.import_from_pypower_ppc(ppc, overwrite_zero_s_nom=True)
        self.pypsa_object.set_snapshots([datetime.now().strftime("%m/%d/%Y %H:%M:%S")])

        self.run_powerflow()

        self.dataframe = self.pypsa_object.df("Bus").copy() 
        self.format_df()
        #self.pypsa_object = pypsa_network
        self.pypsa_history[-1] = self.dataframe
        self.pypsa_object_history[-1] = self.pypsa_object
        #self.format_df()
        self.store_p_flow(t=-1)
        print("pyPSA initialized")

        # Define a function to classify the generator types
    def classify_generator(self, generator):
        if generator == "G0":
            return 3  # Type 3
        elif isinstance(generator, str) and generator.startswith("G"):
            return 2  # Type 2 (e.g., G1, G2, ...)
        elif pd.isna(generator):
            return 1  # Type 1 (NaN)
        elif generator == "WEC":
            return 4  # Type 4 (WEC)
        else:
            return 0  # Default to Type 3 for unknown cases
        
    def format_df(self):
        self.dataframe["type"] = self.dataframe["generator"].apply(self.classify_generator)
        self.dataframe.reset_index(inplace=True)
        self.dataframe.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in self.dataframe.columns]
        # TODO: update WEC locations
        
        
    def ac_injection(self, start, end, p=None, v=None, time=None):
        """
        Description: WEC AC injection for pypsa powerflow solver
        input:
            p - a vector of active power values in order of bus num
            v - a vector of voltage mag PU values in order of bus num
            time: (Int)
        output: no output but pypsa_dataframe is updated and so is pypsa_history
        """
        # TODO: There has to be a better way to do this.
        if time is None:
            time = self.WecGridCore.wecObj_list[0].dataframe.time.to_list()
        num_wecs = len(self.WecGridCore.wecObj_list)
        num_cecs = len(self.WecGridCore.cecObj_list)

        for t in time:
            if t >= start and t <= end:
                if num_wecs > 0:
                    for idx, wec_obj in enumerate(self.WecGridCore.wecObj_list):
                        bus = wec_obj.bus_location
                        pg = wec_obj.dataframe.loc[wec_obj.dataframe.time == t].pg
                        vs = wec_obj.dataframe.loc[wec_obj.dataframe.time == t].vs

                        self.pypsa_object.generators.loc[
                            self.pypsa_object.generators.bus == str(bus), "v_set"
                        ] = vs
                        self.pypsa_object.generators.loc[
                            self.pypsa_object.generators.bus == str(bus), "p_set"
                        ] = pg
                if num_cecs > 0:
                    for idx, cec_obj in enumerate(self.WecGridCore.cecObj_list):
                        bus = cec_obj.bus_location
                        pg = cec_obj.dataframe.loc[cec_obj.dataframe.time == t].pg
                        vs = wec_obj.dataframe.loc[cec_obj.dataframe.time == t].vs

                        self.pypsa_object.generators.loc[
                            self.pypsa_object.generators.bus == str(bus), "v_set"
                        ] = vs
                        self.pypsa_object.generators.loc[
                            self.pypsa_object.generators.bus == str(bus), "p_set"
                        ] = pg

                self.run_powerflow()
                self.store_p_flow(t)
                self.format_df()
                self.pypsa_history[t] = self.dataframe

    def run_powerflow(self):
        """
        Description: This function runs the powerflow for pyPSA model
        Input: None
        output: None
        """
        # this can be updated, it's a bit sloppy
        # temp = self.pypsa_object.copy()
        # temp.pf()
        # self.pypsa_object = temp.copy()
        self.pypsa_object.pf()
        #self.pypsa_dataframe = self.pypsa_object.buses
        self.dataframe = self.pypsa_object.df("Bus").copy() 

    

    def viz(self, dataframe=None):
        """ """
        visualizer = PyPSAVisualizer(pypsa_obj=self)  # need to pass this object itself?
        return visualizer.viz()


    def get_flow_data(self, t=None):
        """
        Retrieves power flow data for all branches in the PyPSA network.

        Inputs:
        - t (float): timestamp for which to retrieve the power flow data (optional)

        Outputs:
        - flow_data (dict): dictionary containing power flow data for all branches
        """
        if t is None:
            flow_data = {}

            try:
                # Iterate over all lines in the network
                for line in self.pypsa_object.lines.index:
                    source = self.pypsa_object.lines.loc[line, "bus0"]
                    target = self.pypsa_object.lines.loc[line, "bus1"]
                    
                    # Assuming power flow data is stored/calculated in network.lines as 'p0'
                    p_flow = self.pypsa_object.lines_t.p0.loc[:, line].iloc[-1]

                    # Create a dictionary for this branch's flow data
                    edge_data = {
                        "source": source if p_flow >= 0 else target,
                        "target": target if p_flow >= 0 else source,
                        "p_flow": p_flow,
                    }

                    # Use a tuple (source, target) as a unique identifier for each edge
                    edge_identifier = (edge_data["source"], edge_data["target"])
                    flow_data[edge_identifier] = edge_data["p_flow"]

            except Exception as e:
                print(f"Error fetching data: {e}")

            return flow_data

        else:
            return self.flow_data.get(t, {})
        
        
    def store_p_flow(self, t):
        """
        Stores the p_flow values of a PyPSA network in a dictionary for an internal timestamp.

        Parameters:
        - t (float): Internal timestamp at which the p_flow values are to be retrieved and stored.
        """
        # Create an empty dictionary for this particular timestamp
        p_flow_dict = {}

        try:
            # Iterate over all lines in the network
            for line in self.pypsa_object.lines.index:
                # Get source and target buses
                source = self.pypsa_object.lines.loc[line, "bus0"]
                target = self.pypsa_object.lines.loc[line, "bus1"]
                #print("{} -> {}".format(source, target))

                # Get the last available power flow value for the line
                # (assuming time-series data exists in lines_t.p0)
                try:
                    p_flow = self.pypsa_object.lines_t.p0.loc[:, line].iloc[-1]
                except KeyError:
                    print(f"Line {line} not found in lines_t.p0.")
                    continue
                except IndexError:
                    print(f"No power flow data available for line {line}.")
                    continue

                # # Determine the direction of flow
                # source = source if p_flow >= 0 else target
                # target = target if p_flow >= 0 else source

                # Store the power flow in the dictionary
                p_flow_dict[(source, target)] = p_flow

            # Store the power flow data for this internal timestamp in the flow_data dictionary
            self.flow_data[t] = p_flow_dict

        except Exception as e:
            print(f"Error storing power flow data for timestamp {t}: {e}")