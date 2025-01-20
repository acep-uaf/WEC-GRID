"""
WEC-GRID source code
Author: Alexander Barajas-Ritchie
Email: barajale@oreogonstate.edu
"""

# Standard Libraries
import os
import sys
import re
import time
import json
from datetime import datetime, timezone, timedelta

# Third-party Libraries
import pandas as pd
import numpy as np
import sqlite3
import pypsa
import pypower.api as pypower
import matlab.engine
import cmath
import matplotlib.pyplot as plt

# local libraries
from WEC_GRID.cec import cec_class
from WEC_GRID.wec import wec_class
from WEC_GRID.utilities.util import dbQuery
from WEC_GRID.utilities.util import read_paths
from WEC_GRID.database_handler.connection_class import DB_PATH

from WEC_GRID.pyPSA import pyPSAWrapper
from WEC_GRID.PSSe import PSSeWrapper
from WEC_GRID.viz import PSSEVisualizer


# Initialize the PATHS dictionary
PATHS = read_paths()
CURR_DIR = os.path.dirname(os.path.abspath(__file__))



class WecGrid:
    """
    The WecGrid class represents the WEC-Grid system.
    It provides methods to initialize solvers (PSSE, PyPSA) and manage system data.
    """

    def __init__(self, case):
        """
        Initializes the WecGrid system with a case file.

        Args:
            case (str): The file path to a raw or sav file.
        """
        self.case_file = case
        self.case_file_name = os.path.basename(case)
        self.psseObj = None
        self.pypsaObj = None
        #self.dbObj = None
        self.wecObj_list = [] # list of the WEC Model objects for the simulations
        self.cecObj_list = [] # list of the WEC Model objects for the simulations
        # these list could probably be combined? ^^^

    def initialize_psse(self, solver_args=None):
        """
        Initializes the PSSE solver.

        Args:
            solver_args (dict): Optional arguments for the PSSE initialization.
        """
        solver_args = solver_args or {}  # Use empty dict if no args are provided
        self.psseObj = PSSeWrapper(self.case_file, self)
        self.psseObj.initalize(solver_args)
        print(f"PSSE initialized with case file: {self.case_file_name}.")

    def initialize_pypsa(self, solver_args=None):
        """
        Initializes the PyPSA solver.

        Args:
            solver_args (dict): Optional arguments for the PyPSA initialization.
        """
        solver_args = solver_args or {}  # Use empty dict if no args are provided
        self.pypsaObj = pyPSAWrapper(self.case_file)
        self.pypsaObj.initalize(solver_args)
        print(f"PyPSA initialized with case file: {self.case_file_name}.")

    def get_solver_data(self):
        """
        Returns a summary of solver states (useful for debugging or logging).

        Returns:
            dict: A dictionary summarizing solver initialization status.
        """
        return {
            "psse_initialized": self.psseObj is not None,
            "pypsa_initialized": self.pypsaObj is not None,
        }
        
    def create_wec(self, ID, model, bus_location, run_sim=True):
        self.wecObj_list.append(wec_class.WEC(ID, model, bus_location, run_sim))

        self.psseObj.dataframe.loc[
            self.psseObj.dataframe["BUS_ID"] == bus_location, "Type"
        ] = 4 # This updated the Grid Model for the grid to know that the bus now has a WEC/CEC on it.
        self.psseObj.wecObj_list = self.wecObj_list
        # TODO: need to update pyPSA obj too
        
    def create_cec(self, ID, model, bus_location, run_sim=True):
        self.cecObj_list.append(cec_class.CEC(ID, model, bus_location, run_sim))
        self.psseObj.dataframe.loc[
            self.psseObj.dataframe["BUS_ID"] == bus_location, "Type"
        ] = 4 # This updated the Grid Model for the grid to know that the bus now has a WEC/CEC on it.
        # TODO: need to update pyPSA obj too




# class Wec_grid:
#     psspy = None # don't need this? 

#     # def __init__(self, case):
#     #     """
#     #     Description: welcome to WEC-Grid, this is the initialization function.
#     #     input:
#     #         case: a file path to a raw or sav file (str)
#     #     output: None
#     #     """
#     #     # initalized variables and files
#     #     self.case_file = case
#     #     self.softwares = [] # Grid solvers in use
#     #     self.wec_list = []  # list
#     #     self.cec_list = []  # list
#     #     self.wec_data = {}
#     #     self.psse_dataframe = pd.DataFrame() # this should be in the wrapper
#     #     self.pypsa_dataframe = pd.DataFrame() # this should be in the wrapper
#     #     self.load_profiles = pd.DataFrame()
#     #     # self.migrid_file_names = self.Migrid_file()
#     #     # self.wec_data = {}
#     #     # self.electrical_distance_dict = {}
#     #     self.z_history = {} # z history? 
#     #     self.flow_data = {} # again what is this for?
#     #     # probably should have some db stuff in here

#     # def initalize_psse(self, solver): # init_psse 
#     #     """
#     #     Description: Initializes a PSSe case, uses the topology passed at original initialization
#     #     input:
#     #         solver: the solver you want to use supported by PSSe, "fnsl" is a good default (str)
#     #     output: None
#     #     """
#     #     os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#     #     psse_path = PATHS["psse"]
#     #     sys.path.extend(
#     #         [
#     #             os.path.join(psse_path, subdir)
#     #             for subdir in ["PSSPY37", "PSSBIN", "PSSLIB", "EXAMPLE"]
#     #         ]
#     #     )
#     #     os.environ["PATH"] = (
#     #         os.path.join(psse_path, "PSSPY37")
#     #         + ";"
#     #         + os.path.join(psse_path, "PSSBIN")
#     #         + ";"
#     #         + os.path.join(psse_path, "EXAMPLE")
#     #         + ";"
#     #     )

#     #     # Initialize PSSe
#     #     self.psspy = pypower.runpsse(self.case_file, solver) # need to investiage this more


#     def __init__(self, case, solvers=None):
#         """
#         Initializes a WEC-Grid system with dynamic solver management.

#         Args:
#             case (str): A file path to a raw or sav file.
#             solvers (dict): A dictionary of solvers to initialize, with keys as solver names and values as their classes or factories.

#         Example:
#             solvers = {
#                 "psse": PsseSolverClass,
#                 "pypsa": PypsaSolverClass,
#             }
#         """
#         # Initialized variables and files
#         self.case_file = case
#         self.softwares = solvers.keys() if solvers else []
#         self.wec_list = []
#         self.cec_list = []
#         self.wec_data = {}
#         self.total_buses = 0
#         self.load_profiles = {}
#         self.z_history = {}
#         self.flow_data = {}

#         # Initialize solvers dynamically
#         self.solvers = {}
#         if solvers:
#             for solver_name, solver_class in solvers.items():
#                 self.solvers[solver_name] = solver_class(case)
                
#     def get_solver(self, name):
#         """
#         Retrieve a solver instance by name.

#         Args:
#             name (str): The name of the solver.

#         Returns:
#             Solver instance if it exists, otherwise None.
#         """
#         return self.solvers.get(name)

#     # def __init__(self, case):
#     #     """
#     #         The `Wec_grid` class represents a WEC-Grid system. It contains methods for initializing a PSSe case,
#     #         adding WECs and CECs to the system, and running simulations.

#     #         Attributes:
#     #             case_file (str): A file path to a raw or sav file.
#     #             softwares (list): A list of software packages used in the simulation.
#     #             wec_list (list): A list of `wec_class` objects representing the WECs in the system.
#     #             cec_list (list): A list of `cec_class` objects representing the CECs in the system.
#     #             wec_data (dict): A dictionary containing data for each WEC in the system.
#     #             psse_dataframe (pandas.DataFrame): A DataFrame containing data from the PSSe simulation.
#     #             pypsa_dataframe (pandas.DataFrame): A DataFrame containing data from the PyPSA simulation.
#     #             load_profiles (pandas.DataFrame): A DataFrame containing load profile data.
#     #             z_history (dict): A dictionary containing the history of the impedance seen by each WEC.
#     #             flow_data (dict): A dictionary containing flow data for each WEC.
#     #         """
#     #     """
#     #     Description: welcome to WEC-Grid, this is the initialization function.
#     #     input:
#     #         case: a file path to a raw or sav file (str)
#     #     output: None
#     #     """
#     #     # initalized variables and files
#     #     self.case_file = case
#     #     self.softwares = []
#     #     self.wec_list = []  # list
#     #     self.cec_list = []  # list
#     #     self.wec_data = {}
#     #     self.total_buses = 0
#     #     self.load_profiles = {}
#     #     # self.migrid_file_names = self.Migrid_file()
#     #     # self.electrical_distance_dict = {}
#     #     self.z_history = {}
#     #     self.flow_data = {}
#     #     self.psse = None # I'd like these to be made dynamically? 
#     #     self.pypsa = None
        
#         #         # initalized variables and files
#         # self.case_file = case
#         # self.softwares = [] # Grid solvers in use
#         # self.wec_list = []  # list
#         # self.cec_list = []  # list
#         # self.wec_data = {}
#         # self.psse_dataframe = pd.DataFrame() # this should be in the wrapper
#         # self.pypsa_dataframe = pd.DataFrame() # this should be in the wrapper
#         # self.load_profiles = pd.DataFrame()
#         # # self.migrid_file_names = self.Migrid_file()
#         # # self.wec_data = {}
#         # # self.electrical_distance_dict = {}
#         # self.z_history = {} # z history? 
#         # self.flow_data = {} # again what is this for?
#         # # probably should have some db stuff in here


#     def run_simulation(self, start, end, solver) -> bool:
#         # Initial empty time_stamps list
#         time_stamps = []

#         # Check and get time_stamps from wec_list or cec_list
#         if len(self.wec_list) != 0:
#             time_stamps = self.wec_list[0].dataframe.time.to_list()
#         elif len(self.cec_list) != 0:
#             time_stamps = self.cec_list[0].dataframe.time.to_list()

#         # Filter time_stamps based on start and end parameters
#         time_stamps = [ts for ts in time_stamps if start <= ts <= end]

#         # Run the simulation for each time stamp within the range
#         for time_stamp in time_stamps:
#             gen = {}
#             load = {}
#             # print("Simulating time stamp: {}".format(time_stamp))
#             if len(self.wec_list) != 0 or len(self.cec_list) != 0:
#                 gen = self.gen_values(time_stamp)
#             if len(self.load_profiles) != 0:
#                 load = self.load_values(time_stamp)
#             self.psse.steady_state(gen, load, time_stamp, solver)
#             # self.pypsa.run_simulation(gen, load, time_stamp, solver)
#             print("-----------------------------------")
#         return True

#     def gen_values(self, t):
#         gen = {}

#         for idx, wec_obj in enumerate(self.wec_list):
#             df = wec_obj.dataframe
#             if t in df.time.values:
#                 pg = df.loc[df.time == t, "pg"].iloc[0]
#                 vs = df.loc[df.time == t, "vs"].iloc[0]
#                 qt = df.loc[df.time == t, "qt"].iloc[0]
#                 qb = df.loc[df.time == t, "qb"].iloc[0]
#                 gen[wec_obj.bus_location] = [pg, vs, qt, qb]

#         for idx, cec_obj in enumerate(self.cec_list):
#             df = cec_obj.dataframe
#             if t in df.time.values:
#                 pg = df.loc[df.time == t, "pg"].iloc[0]
#                 vs = df.loc[df.time == t, "vs"].iloc[0]
#                 gen[cec_obj.bus_location] = [pg, vs]

#         # adjust other generators values here?

#         return gen

#     def load_values(self, t):
#         return self.load_profiles[t].to_dict(orient="index")

#     def initalize_psse(self, solver):
#         self.psse = PSSeWrapper(self.case_file)
#         self.psse.initalize(solver)
#         self.total_buses = len(self.psse.dataframe)

#     def initalize_pypsa(self, solver):
#         self.pypsa = pyPSAWrapper(self.case_file)
#         self.pypsa.initalize(solver)
#         self.total_buses = len(self.pypsa.dataframe)

#     def add_wec(self, wec):
#         """
#         Description: Adds a WEC to the WEC list and adjusts the activate power of the machine data in PSSe
#         input:
#             wec: WEC object to be added to the WEC list (wec_class.WEC)
#         output: None
#         """
#         self.wec_list.append(wec)
#         for w in self.wec_list:
#             ierr = Wec_grid.psspy.machine_data_2(
#                 w.bus_location,
#                 "1",
#                 realar3=w.Qmax,
#                 realar4=w.Qmin,
#                 realar5=w.Pmax,
#                 realar6=w.Pmin,
#             )  # adjust activate power
#             if ierr > 0:
#                 raise Exception("Error adding WEC")

#     def create_wec(self, ID, model, bus_location, run_sim=True):
#         self.wec_list.append(wec_class.WEC(ID, model, bus_location, run_sim))

#         self.psse.dataframe.loc[
#             self.psse.dataframe["BUS_ID"] == bus_location, "Type"
#         ] = 4

#     def create_cec(self, ID, model, bus_location, run_sim=True):
#         self.cec_list.append(cec_class.CEC(ID, model, bus_location, run_sim))
#         self.psse.dataframe.loc[
#             self.psse.dataframe["BUS_ID"] == bus_location, "Type"
#         ] = 4

#     def run_WEC_Sim(self, wec_id, sim_config):
#         for wec in self.wec_list:
#             if wec.ID == wec_id:
#                 wec.WEC_Sim(sim_config)
#                 return True

#     def run_CEC_Sim(self, cec_id, sim_config):
#         for cec in self.cec_list:
#             if cec.ID == cec_id:
#                 cec.CEC_Sim(sim_config)

#     def generate_load_profiles(self, peak_values, curve_type, noise_type, noise_level, step_size, sim_length):
#         """
#         Generate a dictionary of load profiles with timestamps as keys and DataFrames as values.
#         """
#         ierr = 1  # default there is an error

#         if solver == "fnsl":
#             ierr = Wec_grid.psspy.fnsl()
#         elif solver == "GS":
#             ierr = Wec_grid.psspy.solv()
#         elif solver == "DC":
#             ierr = Wec_grid.psspy.dclf_2(1, 1, [1, 0, 1, 2, 1, 1], [0, 0, 0], "1")
#         else:
#             print("not a valid solver")
#             return 0

#         if ierr < 1:  # no error in solving
#             ierr = self._psse_get_values()
#         else:
#             print("Error while solving")
#             return 0

#         if ierr == 1:  # no error while grabbing values
#             return 1
#         else:
#             print("Error while grabbing values")
#             return 0

#         # Time normalization to a 24-hour cycle
#         seconds_in_day = 24 * 60 * 60
#         time_data = np.arange(0, sim_length, step_size)
#         normalized_time = (time_data % seconds_in_day) / seconds_in_day

#     def _psse_get_values(self):
#         """
#         Description: This function grabs all the important values we want in our dataframe
#         input: None, uses lst_param tho
#         output: Dataframe of the selected parameters for each bus.
#         """
#         lst = self.lst_param
#         temp_dict = {}
#         for bus_parameter in lst:
#             if bus_parameter != "P" and bus_parameter != "Q":
#                 # grabs the bus parameter values for the specified parameter - list
#                 ierr, bus_parameter_values = Wec_grid.psspy.abusreal(
#                     -1, string=bus_parameter
#                 )
#                 if ierr != 0:
#                     print("error in get_values function")
#                     return 0
#                 bus_add = {}
#                 for bus_index, value in enumerate(
#                     bus_parameter_values[0]
#                 ):  # loops over those values to create bus num & value pairs
#                     bus_add["BUS {}".format(bus_index + 1)] = value
#                 temp_dict[bus_parameter] = bus_add

#         self.psse_dataframe = pd.DataFrame.from_dict(temp_dict)
#         self.psse_dataframe = self.psse_dataframe.reset_index()
#         self.psse_dataframe = self.psse_dataframe.rename(columns={"index": "Bus"})
#         # gets the bus type (3 = swing)
#         self.psse_dataframe["Type"] = Wec_grid.psspy.abusint(-1, string="TYPE")[1][0]
#         self.psse_dataframe.insert(0, "BUS_ID", range(1, 1 + len(self.psse_dataframe)))
#         self._psse_addGeninfo()
#         self._psse_addLoadinfo()

#         if "P" in lst:
#             self._psse_get_p_or_q("P")
#         if "Q" in lst:
#             self._psse_get_p_or_q("Q")

#         # Check if column exists, if not then initialize
#         if "ΔP" not in self.psse_dataframe.columns:
#             self.psse_dataframe["ΔP"] = 0.0  # default value
#         if "ΔQ" not in self.psse_dataframe.columns:
#             self.psse_dataframe["ΔQ"] = 0.0  # default value
#         if "M_Angle" not in self.psse_dataframe.columns:
#             self.psse_dataframe["M_Angle"] = 0.0  # default value
#         if "M_Mag" not in self.psse_dataframe.columns:
#             self.psse_dataframe["M_Mag"] = 0.0  # default value

#         # Your loop remains unchanged
#         for index, row in self.psse_dataframe.iterrows():
#             mismatch = Wec_grid.psspy.busmsm(row["BUS_ID"])[1]
#             real = mismatch.real
#             imag = mismatch.imag
#             angle = abs(mismatch)
#             mag = cmath.phase(mismatch)
#             self.psse_dataframe.at[index, "ΔP"] = mismatch.real  # should be near zero
#             self.psse_dataframe.at[index, "ΔQ"] = mismatch.imag
#             self.psse_dataframe.at[index, "M_Angle"] = abs(mismatch)
#             self.psse_dataframe.at[index, "M_Mag"] = cmath.phase(mismatch)
#         return 1

#     def _psse_get_p_or_q(self, letter):
#         """
#         Description: retrieves P (activate) or Q (reactive) Voltage (in PU) and Voltage Angle for each Bus in the current loaded case
#         input:
#             letter: either P or Q as a string
#         output: None
#         """
#         gen_values = self.psse_dataframe["{} Gen".format(letter)]  #
#         load_values = self.psse_dataframe["{} Load".format(letter)]
#         letter_list = [None] * len(self.psse_dataframe)

#         for i in range(len(letter_list)):
#             gen = gen_values[i]
#             load = load_values[i]
#             if (not pd.isnull(gen)) and (not pd.isnull(load)):
#                 letter_list[i] = gen - load
#             else:
#                 if not pd.isnull(gen):
#                     letter_list[i] = gen
#                 if not pd.isnull(load):
#                     letter_list[i] = 0 - load  # gen is
#         self.psse_dataframe["{}".format(letter)] = letter_list

#     def _psse_busNum(self):
#         """
#         Description: Returns the number of Buses in the currently loaded case
#         input: None
#         output: Number of Buses
#         """
#         Wec_grid.psspy.bsys(0, 0, [0.0, 0.0], 1, [1], 0, [], 0, [], 0, [])
#         ierr, all_bus = Wec_grid.psspy.abusint(0, 1, ["number"])
#         return all_bus[0]

#     def _psse_dc_injection(self, ibus, p, pf_solver, time):
#         """
#         Description: preforms the DC injection of the wec buses
#         input:
#             p: a list of active power set point in order(list)
#             pf_solver: supported PSSe solver (Str)
#             time: (Int)
#         output: None
#         """
#         ierr = Wec_grid.psspy.machine_chng_3(ibus, "1", [], [p])
#         if ierr > 0:
#             print("Failed | machine_chng_3 code = {}".format(ierr))
#         # psspy.dclf_2(status4=2)
#         ierr = Wec_grid.psspy.dclf_2(1, 1, [1, 0, 1, 2, 0, 1], [0, 0, 1], "1")
#         if ierr > 0:
#             raise Exception("Error in DC injection")
#         self._psse_get_values()
#         self.psse_history[time] = self.psse_dataframe

#     def _pypsa_ac_injection(self, p, v, time):
#         """
#         Description: WEC AC injection for pypsa powerflow solver
#         input:
#             p - a vector of active power values in order of bus num
#             v - a vector of voltage mag PU values in order of bus num
#             time: (Int)
#         output: no output but pypsa_dataframe is updated and so is pypsa_history
#         """
#         for idx, bus in enumerate(self.wecBus_nums):

#             self.pypsa_object.generators.loc[
#                 self.pypsa_object.generators.bus == str(bus), "v_set_pu"
#             ] = v[idx]
#             self.pypsa_object.generators.loc[
#                 self.pypsa_object.generators.bus == str(bus), "p_set"
#             ] = p[idx]

#         self._pypsa_run_powerflow()
#         self.pypsa_history[time] = self.pypsa_dataframe

#     # def _generate_load_curve(self, peak_value, curve_type, noise_type, noise_level):
#     #     """
#     #     Generate a load curve.

#     #     Parameters:
#     #     peak_value: float
#     #         The peak value of the curve.
#     #     curve_type: str
#     #         The type of curve ('normal', 'summer', 'winter').
#     #     noise_type: str
#     #         The type of noise ('uniform', 'gaussian').
#     #     noise_level: float
#     #         The level of noise to add.
#     #     """
#     #     time_data = self.wec_list[0].dataframe.time.to_list()
#     #     num_timesteps = len(time_data)
#     #     midpoint_index = num_timesteps // 2
#     #     midpoint_time = time_data[midpoint_index]
#     #     time_range = time_data[-1] - time_data[0]

#     #     if peak_value == 0:
#     #         return np.zeros(num_timesteps)

#     #     std_dev = time_range * 0.15

#     #     if curve_type == "normal":
#     #         curve = np.exp(
#     #             -((np.array(time_data) - midpoint_time) ** 2) / (2 * std_dev**2)
#     #         )
#     #     elif curve_type == "summer":
#     #         # Example summer curve
#     #         curve = np.sin(np.array(time_data) / time_range * np.pi) ** 2
#     #     elif curve_type == "winter":
#     #         # Example winter curve
#     #         curve = np.cos(np.array(time_data) / time_range * np.pi) ** 2

#     #         # Normalize and add noise
#     #         curve = curve / curve.max() * peak_value
#     #         if noise_type == "uniform":
#     #             curve += np.random.uniform(-noise_level, noise_level, 1)
#     #         elif noise_type == "gaussian":
#     #             curve += np.random.normal(0, noise_level, 1)
#     #         curve = np.maximum(curve, 0)  # Ensure no negative values

#     #     curve_data.append(curve[0])

#     #     df.loc[bus_id] = curve_data

#     def _generate_all_load_profiles(self,peak_values,curve_type="normal",noise_type=None, noise_level=, hours=12,resolution=5):
#         """
#         Generate load profiles for all buses.

#         Parameters:
#         - peak_values (dict): Dictionary with bus IDs as keys and peak values as values.
#         - curve_type (str): Type of load curve to generate ('normal', 'summer', 'winter').
#         - noise_type (str): Type of noise to add ('uniform', 'gaussian').
#         - noise_level (float): Level of noise to add.
#         - hours (int): Duration of the simulation in hours.
#         - resolution (int): Time resolution in minutes.
#         """
#         time_data = self.wec_list[0].dataframe.time.to_list()
#         self.load_profiles = pd.DataFrame(
#             {
#             f"bus {bus_id}": self._generate_load_curve(
#                 peak, curve_type, noise_type, noise_level
#             )
#             for bus_id, peak in peak_values.items()
#             }
#         )
#         self.load_profiles["time"] = time_data

#     def plot_bus_load_profiles(self, bus_id):
#         """
#         Plot the load profiles (P and Q values) for a specific bus.

#         Parameters:
#         bus_id: int
#             The ID of the bus to plot.
#         """

#         # Extracting time series and P, Q values for the specified bus
#         times = list(self.load_profiles.keys())
#         P_values = [self.load_profiles[time].loc[bus_id, "P"] for time in times]
#         Q_values = [self.load_profiles[time].loc[bus_id, "Q"] for time in times]

#         # Plotting Active Power (P)
#         plt.figure(figsize=(10, 5))
#         plt.plot(times, P_values, label="P - Active Power", marker="o")
#         plt.title(f"Active Power Load Profile for Bus {bus_id}")
#         plt.xlabel("Time")
#         plt.ylabel("Active Power (P)")
#         plt.legend()
#         plt.grid(True)
#         plt.show()

#         # Call PSS/E API to update load
#         ierr = Wec_grid.psspy.load_data_6(ibus, _id, intgar, realar, lodtyp)

#         return ierr

#     def _psse_ac_injection(self, start, end, p=None, v=None, time=None):
#         """
#         Description: WEC AC injection for PSSe powerflow solver
#         input:
#             p - a vector of active power values in order of bus num
#             v - a vector of voltage mag PU values in order of bus num
#             pf_solver - Power flow solving algorithm  (Default-"fnsl")
#             time: (Int)
#         output:
#             no output but psse_dataframe is updated and so is psse_history
#         """
#         time = self.wec_list[0].dataframe.time.to_list()
#         for t in time:
#             # print("time: {}".format(t))
#             if t >= start and t <= end:
#                 for idx, wec_obj in enumerate(self.wec_list):
#                     bus = wec_obj.bus_location
#                     pg = wec_obj.dataframe.loc[
#                         wec_obj.dataframe.time == t
#                     ].pg  # adjust activate power
#                     ierr = Wec_grid.psspy.machine_data_2(
#                         bus, "1", realar1=pg
#                     )  # adjust activate power
#                     if ierr > 0:
#                         raise Exception("Error in AC injection")
#                     vs = wec_obj.dataframe.loc[wec_obj.dataframe.time == t].vs
#                     ierr = Wec_grid.psspy.bus_chng_4(
#                         bus, 0, realar2=vs
#                     )  # adjsut voltage mag PU
#                     if ierr > 0:
#                         raise Exception("Error in AC injection")

#                     # self._psse_run_powerflow(self.solver)
#                     self._psse_update_load(bus, t)
#                     # print("=======")
#                 if t in self.cec_list[0].dataframe["time"].values:
#                     for idx, cec_obj in enumerate(self.cec_list):
#                         bus = cec_obj.bus_location
#                         pg = cec_obj.dataframe.loc[
#                             cec_obj.dataframe.time == t
#                         ].pg  # adjust activate power
#                         ierr = Wec_grid.psspy.machine_data_2(
#                             bus, "1", realar1=pg
#                         )  # adjust activate power
#                         if ierr > 0:
#                             raise Exception("Error in AC injection")
#                         vs = wec_obj.dataframe.loc[wec_obj.dataframe.time == t].vs
#                         ierr = Wec_grid.psspy.bus_chng_4(
#                             bus, 0, realar2=vs
#                         )  # adjsut voltage mag PU
#                         if ierr > 0:
#                             raise Exception("Error in AC injection")

#                         # self._psse_run_powerflow(self.solver)
#                         self._psse_update_load(bus, t)
#                     #     #print("=======")

#                 self._psse_run_powerflow(self.solver)
#                 self.update_type()
#                 self.psse_history[t] = self.psse_dataframe
#                 self.z_values(time=t)
#                 self.store_p_flow(t)
#             if t > end:
#                 break
#         return

#     def _psse_bus_history(self, bus_num):
#         """
#         Description: this function grab all the data associated with a bus through the simulation
#         input:
#             bus_num: bus number (Int)
#         output:
#             bus_dataframe: a pandas dateframe of the history
#         """
#         # maybe I should add a filering parameter?

#         bus_dataframe = pd.DataFrame()
#         for time, df in self.psse_history.items():
#             temp = pd.DataFrame(df.loc[df["BUS_ID"] == bus_num])
#             temp.insert(0, "time", time)
#             bus_dataframe = bus_dataframe.append(temp)
#         return bus_dataframe

#     def _psse_plot_bus(self, bus_num, time, arg_1="P", arg_2="Q"):
#         """
#         Description: This function plots the activate and reactive power for a given bus
#         input:
#             bus_num: the bus number we wanna viz (Int)
#             time: a list with start and end time (list of Ints)
#         output:
#             matplotlib chart
#         """
#         visualizer = PSSEVisualizer(
#             psse_dataframe=self.psse_dataframe,
#             psse_history=self.psse_history,
#             load_profiles=self.load_profiles,
#             flow_data=self.get_flow_data(),
#         )
#         visualizer._psse_plot_bus(bus_num, time, arg_1, arg_2)

#     def plot_load_curve(self, bus_id):
#         """
#         Description: This function plots the load curve for a given bus
#         input:
#             bus_id: the bus number we want to visualize (Int)
#         output:
#             matplotlib chart
#         """
#         # Check if the bus_id exists in load_profiles
#         viz = PSSEVisualizer(
#             psse_dataframe=self.psse_dataframe,
#             psse_history=self.psse_history,
#             load_profiles=self.load_profiles,
#             flow_data=self.flow_data,
#         )
#         viz.plot_load_curve(bus_id)

#     def _psse_addGeninfo(self):
#         """
#         Description: This function grabs the generator values from the PSSe system and updates the psse_dataframe with the generator data.
#         Input: None
#         Output: None but updates psse_dataframe with generator data
#         """
#         machine_bus_nums = Wec_grid.psspy.amachint(-1, 4, "NUMBER")[1][
#             0
#         ]  # get the bus numbers of the machines - list
#         # grabs the complex values for the machine
#         ierr, machine_bus_values = Wec_grid.psspy.amachcplx(-1, 1, "PQGEN")
#         if ierr != 0:
#             raise Exception("Error in grabbing PGGEN values in addgen function")
#         p_gen_df_list = [None] * len(self.psse_dataframe)
#         q_gen_df_list = [None] * len(self.psse_dataframe)
#         # iterate over the machine values
#         for list_index, value in enumerate(machine_bus_values[0]):
#             p_gen_df_list[
#                 machine_bus_nums[list_index] - 1
#             ] = value.real  # -1 is for the offset
#             q_gen_df_list[machine_bus_nums[list_index] - 1] = value.imag

#         self.psse_dataframe["P Gen"] = p_gen_df_list
#         self.psse_dataframe["Q Gen"] = q_gen_df_list

#     def _psse_addLoadinfo(self):
#         """
#         Description: this function grabs the load values from the PSSe system
#         input: None
#         output: None but updates psse_dataframe with load data
#         """
#         load_bus_nums = Wec_grid.psspy.aloadint(-1, 4, "NUMBER")[1][
#             0
#         ]  # get the bus numbers of buses with loads - list
#         ierr, load_bus_values = Wec_grid.psspy.aloadcplx(-1, 1, "MVAACT")  # load values
#         if ierr != 0:
#             raise Exception("Error in grabbing PGGEN values in addgen function")
#         p_load_df_list = [None] * len(self.psse_dataframe)
#         q_load_df_list = [None] * len(self.psse_dataframe)

#         # iterate over the machine values
#         for list_index, value in enumerate(load_bus_values[0]):
#             p_load_df_list[
#                 load_bus_nums[list_index] - 1
#             ] = value.real  # -1 is for the offset
#             q_load_df_list[load_bus_nums[list_index] - 1] = value.imag
#         self.psse_dataframe["P Load"] = p_load_df_list
#         self.psse_dataframe["Q Load"] = q_load_df_list

#     def z_values(self, time):
#         """
#         Retrieve the impedance values for each branch in the power grid at a given time.

#         Parameters:
#         - time (float): The time at which to retrieve the impedance values.

#         Returns:
#         - None. The impedance values are stored in the `z_history` attribute of the object.
#         """
#         # Retrieve FROMNUMBER and TONUMBER for all branches
#         ierr, (from_numbers, to_numbers) = Wec_grid.psspy.abrnint(
#             sid=-1, flag=3, string=["FROMNUMBER", "TONUMBER"]
#         )
#         assert ierr == 0, "Error retrieving branch data"

#         # Create a dictionary to store the impedance values for each branch
#         impedances = {}

#         for from_bus, to_bus in zip(from_numbers, to_numbers):
#             ickt = "1"  # Assuming a default circuit identifier; might need adjustment for your system

#             ierr, cmpval = Wec_grid.psspy.brndt2(from_bus, to_bus, ickt, "RX")
#             if ierr == 0:
#                 impedances[
#                     (from_bus, to_bus)
#                 ] = cmpval  # Store the complex impedance value directly
#             else:
#                 print(f"Error fetching impedance data for branch {from_bus}-{to_bus}")

#         # The impedances dictionary contains impedance for each branch
#         self.z_history[time] = impedances

#     def store_p_flow(self, t):
#         """
#         Function to store the p_flow values of a grid network in a dictionary.

#         Parameters:
#         - t (float): Time at which the p_flow values are to be retrieved.
#         """
#         # Create an empty dictionary for this particular time
#         p_flow_dict = {}

#         try:
#             ierr, (fromnumber, tonumber) = Wec_grid.psspy.abrnint(
#                 sid=-1, flag=3, string=["FROMNUMBER", "TONUMBER"]
#             )

#             for index in range(len(fromnumber)):
#                 ierr, p_flow = Wec_grid.psspy.brnmsc(
#                     int(fromnumber[index]), int(tonumber[index]), "1", "P"
#                 )

#                 source = str(fromnumber[index]) if p_flow >= 0 else str(tonumber[index])
#                 target = str(tonumber[index]) if p_flow >= 0 else str(fromnumber[index])

#                 p_flow_dict[(source, target)] = p_flow

#             # Store the p_flow data for this time in the flow_data dictionary
#             self.flow_data[t] = p_flow_dict

#         except Exception as e:
#             print(f"Error fetching data: {e}")

#     def _psse_viz(self, dataframe=None):
#         """
#         Description: Generates a visualization of the PSSE data using the PSSEVisualizer class.

#         Parameters:
#         - dataframe (pandas.DataFrame): Optional parameter to pass a custom PSSE dataframe.

#         Returns:
#         - matplotlib.figure.Figure: A matplotlib figure object containing the visualization.
#         """
#         visualizer = PSSEVisualizer(
#             # psse_dataframe=self.psse.dataframe,
#             # psse_history=self.psse.history,
#             # #load_profiles=self.load_profiles, # need to add a if exist statement
#             # flow_data=self.psse.flow_data(),
#             psse_obj = self.psse
#         )
#         return visualizer.viz()

#     def _psse_adjust_gen(self, bus_num, p=None, v=None, q=None):
#         """
#         Description: Given a generator bus number. Adjust the values based on the parameters passed.
#         input:
#         output:
#         """

#         if p is not None:
#             ierr = Wec_grid.psspy.machine_data_2(
#                 bus_num, "1", realar1=p
#             )  # adjust activate power
#             if ierr > 0:
#                 raise Exception("Error in AC injection")

#         if v is not None:
#             ierr = Wec_grid.psspy.bus_chng_4(
#                 bus_num, 0, realar2=v
#             )  # adjsut voltage mag PU
#             if ierr > 0:
#                 raise Exception("Error in AC injection")

#         if q is not None:
#             ierr = Wec_grid.psspy.machine_data_2(
#                 bus_num, "1", realar2=q
#             )  # adjust Reactivate power
#             if ierr > 0:
#                 raise Exception("Error in AC injection")

#     def clear_database(self):
#         """
#         Clears all the tables from the database.
#         """

#         # Fetch all table names from the database
#         tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
#         tables = dbQuery(tables_query)

#         # Drop each table from the database
#         for table_name in tables:
#             drop_query = f"DROP TABLE IF EXISTS {table_name[0]}"
#             dbQuery(drop_query)

#     def migrid_warm_start(self):
#         """
#         Description: Adjusts the active power of regular generators to match the values in the migrid_data dictionary.
#         input:
#         output:
#         """
#         generator_buses = self.psse_dataframe[
#             self.psse_dataframe["Type"] == 2
#         ].BUS_ID.to_list()
#         regular_gens = [x for x in generator_buses if x not in self.wecBus_nums]
#         print(regular_gens)
#         pointer = 0
#         for key, value in self.migrid_data.items():
#             if key[:3] == "gen":
#                 self._psse_adjust_gen(
#                     bus_num=regular_gens[pointer], p=value.iloc[0].gen_value
#                 )
#                 pointer += 1

#     def get_flow_data(self, t=None):
#         """
#         Description:
#         This method retrieves the power flow data for all branches in the power system at a given timestamp.
#         If no timestamp is provided, the method fetches the data from PSS/E and returns it.
#         If a timestamp is provided, the method retrieves the corresponding data from the dictionary and returns it.

#         Inputs:
#         - t (float): timestamp for which to retrieve the power flow data (optional)

#         Outputs:
#         - flow_data (dict): dictionary containing the power flow data for all branches in the power system
#         """
#         # If t is not provided, fetch data from PSS/E
#         if t is None:
#             flow_data = {}

#             try:
#                 ierr, (fromnumber, tonumber) = self.psspy.abrnint(
#                     sid=-1, flag=3, string=["FROMNUMBER", "TONUMBER"]
#                 )

#                 for index in range(len(fromnumber)):
#                     ierr, p_flow = self.psspy.brnmsc(
#                         int(fromnumber[index]), int(tonumber[index]), "1", "P"
#                     )

#                     edge_data = {
#                         "source": str(fromnumber[index])
#                         if p_flow >= 0
#                         else str(tonumber[index]),
#                         "target": str(tonumber[index])
#                         if p_flow >= 0
#                         else str(fromnumber[index]),
#                         "p_flow": p_flow,
#                     }

#                     # Use a tuple (source, target) as a unique identifier for each edge
#                     edge_identifier = (edge_data["source"], edge_data["target"])
#                     flow_data[edge_identifier] = edge_data["p_flow"]
#             except Exception as e:
#                 print(f"Error fetching data: {e}")

#             # Assign the fetched data to the current timestamp and return it
#             # self.flow_data[time.time()] = flow_data
#             return flow_data

#         # If t is provided, retrieve the corresponding data from the dictionary
#         else:
#             return self.flow_data.get(t, {})
