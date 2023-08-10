"""
WEC-GRID source code
Author: Alexander Barajas-Ritchie
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
import netCDF4
import pypsa
import pypower.api as pypower
import matlab.engine
import networkx as nx
import seaborn as sns
import sqlite3
import matplotlib.pyplot as plt
import ipycytoscape
import ipywidgets as widgets
from IPython.display import display
import cmath


def dbQuery(query, parameters=(), return_type="cursor"):
    """
    Execute a given SQL query and return the response.

    Args:
        query (str): SQL query to be executed.
        parameters (tuple, optional): Parameters for the SQL query. Defaults to ().
        return_type (str, optional): Return type can be 'cursor' or 'df'. Defaults to 'cursor'.

    Returns:
        sqlite3.Cursor or pd.DataFrame: Depending on return_type, returns a cursor object or a dataframe.
    """

    with SQLiteConnection(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(query, parameters)

        if return_type == "df":
            return pd.read_sql_query(query, conn, params=parameters)
        else:
            return cursor.fetchall()


def read_paths():
    path_config_file = os.path.join(CURR_DIR, "path_config.txt")
    path_names = ["psse", "wec_sim", "wec_model", "wec_grid_class", "wec_grid_folder"]

    with open(path_config_file, "r") as fp:
        return dict(zip(path_names, map(str.strip, fp.readlines())))


# Global Constants
CURR_DIR = os.path.dirname(__file__)
DB_NAME = "WEC-GRID.db"
DB_PATH = os.path.join(CURR_DIR, DB_NAME)
PATHS = read_paths()


class SQLiteConnection:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_name)
        return self.conn  # return connection object

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        self.conn.close()


class CEC:
    """
    This class represents a CEC (Current Energy Converter).

    Attributes:
        ID (int): The ID of the WEC.
        bus_location (str): The location of the bus.
        model (str): The model of the WEC.
        dataframe (DataFrame): The pandas DataFrame holding WEC data.
        Pmax (float): The maximum P value, defaults to 9999.
        Pmin (float): The minimum P value, defaults to -9999.
        Qmax (float): The maximum Q value, defaults to 9999.
        Qmin (float): The minimum Q value, defaults to -9999.
    """

    def __init__(
        self, ID, model, bus_location, Pmax=9999, Pmin=-9999, Qmax=9999, Qmin=-9999
    ):
        """
        The constructor for the WEC class.

        Args:
            ID (int): The ID of the CEC.
            model (str): The model of the WEC.
            bus_location (str): The location of the bus.
            Pmax (float, optional): The maximum P value. Defaults to 9999.
            Pmin (float, optional): The minimum P value. Defaults to -9999.
            Qmax (float, optional): The maximum Q value. Defaults to 9999.
            Qmin (float, optional): The minimum Q value. Defaults to -9999.
        """
        self.ID = ID
        self.bus_location = bus_location
        self.model = model
        self.dataframe = pd.DataFrame()
        self.Pmax = Pmax
        self.Pmin = Pmin
        self.Qmax = Qmax
        self.Qmin = Qmin

        # Try to load data from the database on initialization
        if not self.pull_cec_data():
            print("Data for CEC {} not found in the database.".format(self.ID))

    def pull_cec_data(self):
        """
        Pulls WEC data from the database. If wec_num is provided, pulls data for that specific wec.

        Args:
            wec_num (int, optional): The number of the specific wec to pull data for.

        Returns:
            bool: True if the data pull was successful, False otherwise.
        """

        # Check if the database file exists
        if not os.path.exists(DB_PATH):
            print(
                "Database does not exist. Creating new database here {}".format(DB_PATH)
            )
            # You can create the database file here if needed
            # For now, just exit the function
            return False

        # Check if the table exists
        table_check_query = "SELECT name FROM sqlite_master WHERE type='table' AND name='CEC_output_{}'".format(
            self.ID
        )

        table_check_result = dbQuery(table_check_query)

        if not table_check_result or table_check_result[0][0] != "CEC_output_{}".format(
            self.ID
        ):
            return False

        data_query = "SELECT * from CEC_output_{}".format(self.ID)
        self.dataframe = dbQuery(data_query, return_type="df")

        return True

    def CEC_Sim(self, config):

        # Using the context manager for the SQLite connection.
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()

            table_name = f"CEC_output_{self.ID}"
            drop_table_query = f"DROP TABLE IF EXISTS {table_name};"

            cursor.execute(drop_table_query)
            conn.commit()

        eng = matlab.engine.start_matlab()
        eng.cd(os.path.join(PATHS["wec_model"], self.model))
        # path = wec_sim_path  # Update to match your WEC-SIM source location
        eng.addpath(eng.genpath(PATHS["wec_sim"]), nargout=0)
        print(f"Running {self.model}")

        # Variables required to run w2gSim
        eng.workspace["cecId"] = self.ID
        eng.workspace["simLength"] = config["sim_length"]  # Uncomment if needed

        eng.eval("m2g_out = c2gSim(cecId, simLength);", nargout=0)

        # eng.eval("NewEnergy_20_ohms_100hz;", nargout=0)
        # eng.eval("r2g_ne5kW_init;", nargout=0)
        # eng.eval("sim('R2G_ss_NE5kW_R2019a.slx', [], simset('SrcWorkspace', 'current'));", nargout=0)
        # eng.eval(f"m2g_out.cecId = {self.ID};", nargout=0)
        # eng.eval("c2gSim(cecId,simLength);", nargout=0)  # Uncomment if needed
        eng.workspace["DB_PATH"] = DB_PATH
        eng.eval("CECsim_to_PSSe_dataFormatter", nargout=0)
        print("Sim Completed")
        print("==========")

        # Using dbQuery to fetch the results and put them into the dataframe.
        data_query = f"SELECT * from CEC_output_{self.ID}"
        self.dataframe = dbQuery(data_query, return_type="df")


class WEC:
    """
    This class represents a WEC (Wave Energy Converter).

    Attributes:
        ID (int): The ID of the WEC.
        bus_location (str): The location of the bus.
        model (str): The model of the WEC.
        dataframe (DataFrame): The pandas DataFrame holding WEC data.
        Pmax (float): The maximum P value, defaults to 9999.
        Pmin (float): The minimum P value, defaults to -9999.
        Qmax (float): The maximum Q value, defaults to 9999.
        Qmin (float): The minimum Q value, defaults to -9999.
    """

    def __init__(
        self, ID, model, bus_location, Pmax=9999, Pmin=-9999, Qmax=9999, Qmin=-9999
    ):
        self.ID = ID
        self.bus_location = bus_location
        self.model = model
        self.dataframe = pd.DataFrame()
        self.Pmax = Pmax
        self.Pmin = Pmin
        self.Qmax = Qmax
        self.Qmin = Qmin

        if not self.pull_wec_data():
            print(f"Data for WEC {self.ID} not found in the database.")

    def pull_wec_data(self):
        """
        Pulls WEC data from the database. If wec_num is provided, pulls data for that specific wec.

        Args:
            wec_num (int, optional): The number of the specific wec to pull data for.

        Returns:
            bool: True if the data pull was successful, False otherwise.
        """

        table_check_query = "SELECT name FROM sqlite_master WHERE type='table' AND name='WEC_output_{}'".format(
            self.ID
        )
        table_check_result = dbQuery(table_check_query)

        if (
            not table_check_result
            or table_check_result[0][0] != f"WEC_output_{self.ID}"
        ):
            return False

        data_query = f"SELECT * from WEC_output_{self.ID}"
        self.dataframe = dbQuery(data_query, return_type="df")
        return True

    def WEC_Sim(self, config):
        """
        Description: This function runs the WEC-SIM simulation for the model in the input folder.
        input:
            wec_id = Id number for your WEC (INT)
            sim_length = simulation length in seconds (INT)
            Tsample = The sample resolution in seconds (INT)
            waveHeight = wave height of the sim (FLOAT) // 2.5 is the default
            wavePeriod = wave period of the sim (FLOAT) // 8 is the default
            waveSeed = seed number for the simulation // np.random.randint(99999999999)

        output: output is the the SQL database, you can query the data with "SELECT * from WEC_output_{wec_id}"
        """

        table_name = f"WEC_output_{self.ID}"
        drop_table_query = f"DROP TABLE IF EXISTS {table_name};"
        dbQuery(drop_table_query)

        eng = matlab.engine.start_matlab()
        eng.cd(os.path.join(PATHS["wec_model"], self.model))
        eng.addpath(eng.genpath(PATHS["wec_sim"]), nargout=0)
        print(f"Running {self.model}")

        eng.workspace["wecId"] = self.ID
        for key, value in config.items():
            eng.workspace[key] = value

        eng.workspace["DB_PATH"] = DB_PATH  # move to front end?
        eng.eval(
            "m2g_out = w2gSim_LUPA(wecId,simLength,Tsample,waveHeight,wavePeriod,waveSeed);",
            nargout=0,
        )
        eng.eval("WECsim_to_PSSe_dataFormatter", nargout=0)
        print("Sim Completed")
        print("==========")

        data_query = f"SELECT * from WEC_output_{self.ID}"
        self.dataframe = dbQuery(data_query, return_type="df")


class Wec_grid:

    psspy = None

    def __init__(self, case):
        """
        Description: welcome to WEC-Grid, this is the initialization function.
        input:
            case: a file path to a raw or sav file (str)
        output: None
        """
        # initalized variables and files
        self.case_file = case
        self.softwares = []
        self.wec_list = []  # list
        self.cec_list = []  # list
        self.wec_data = {}
        self.psse_dataframe = pd.DataFrame()
        self.pypsa_dataframe = pd.DataFrame()
        self.load_profiles = pd.DataFrame()
        # self.migrid_file_names = self.Migrid_file()
        # self.wec_data = {}

    def initalize_psse(self, solver):
        """
        Description: Initializes a PSSe case, uses the topology passed at original initialization
        input:
            solver: the solver you want to use supported by PSSe, "fnsl" is a good default (str)
        output: None
        """
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        psse_path = PATHS["psse"]
        sys.path.extend(
            [
                os.path.join(psse_path, subdir)
                for subdir in ["PSSPY37", "PSSBIN", "PSSLIB", "EXAMPLE"]
            ]
        )
        os.environ["PATH"] = (
            os.path.join(psse_path, "PSSPY37")
            + ";"
            + os.path.join(psse_path, "PSSBIN")
            + ";"
            + os.path.join(psse_path, "EXAMPLE")
            + ";"
            + os.environ["PATH"]
        )

        import psse35

        psse35.set_minor(3)
        import psspy

        Wec_grid.psspy = psspy

        Wec_grid.psspy.report_output(
            islct=2, filarg="NUL", options=[0]
        )  # Discards output

        Wec_grid.psspy.psseinit(50)
        self.softwares.append("psse")
        self.psse_history = {}
        self.lst_param = ["BASE", "PU", "ANGLED", "P", "Q"]
        self.solver = solver
        self.dynamic_case_file = ""

        # self.psse_dataframe = pd.DataFrame()
        self._i = Wec_grid.psspy.getdefaultint()
        self._f = Wec_grid.psspy.getdefaultreal()
        self._s = Wec_grid.psspy.getdefaultchar()

        if self.case_file.endswith(".sav"):
            Wec_grid.psspy.case(self.case_file)
        elif self.case_file.endswith(".raw"):
            Wec_grid.psspy.read(1, self.case_file)
        elif self.case_file.endswith(".RAW"):
            Wec_grid.psspy.read(1, self.case_file)
        self._psse_run_powerflow(self.solver)

        self.psse_history[-1] = self.psse_dataframe

    def initalize_pypsa(self):
        """
        Description: Initializes a pyPSA case, uses the topology passed at original initialization
        input:
            solver: the solver you want to use supported by PSSe, "fnsl" is a good default (str)
        output: None
        notes: only works with .raw files, needs to use matpower->pypower->pyPSA conversion process
        """
        self.pypsa_history = {}
        self.softwares.append("pypsa")
        self.pypsa_object_history = {}

        eng = matlab.engine.start_matlab()
        eng.workspace["case_path"] = self.case_file
        eng.eval("mpc = psse2mpc(case_path)", nargout=0)
        eng.eval("savecase('here.mat',mpc,1.0)", nargout=0)

        # Load the MATPOWER case file from a .mat file
        ppc = pypower.loadcase("./here.mat")

        # Convert Pandapower network to PyPSA network
        pypsa_network = pypsa.Network()
        pypsa_network.import_from_pypower_ppc(ppc, overwrite_zero_s_nom=True)
        pypsa_network.set_snapshots([datetime.now().strftime("%m/%d/%Y %H:%M:%S")])

        pypsa_network.pf()

        self.pypsa_dataframe = pypsa_network.buses
        self.pypsa_object = pypsa_network
        self.pypsa_history[-1] = self.pypsa_dataframe
        self.pypsa_object_history[-1] = self.pypsa_object

    def add_wec(self, wec):
        self.wec_list.append(wec)
        for w in self.wec_list:
            ierr = Wec_grid.psspy.machine_data_2(
                w.bus_location,
                "1",
                realar3=w.Qmax,
                realar4=w.Qmin,
                realar5=w.Pmax,
                realar6=w.Pmin,
            )  # adjust activate power
            if ierr > 0:
                raise Exception("Error adding WEC")

    def create_wec(self, ID, model, bus_location, run_sim=True):
        self.wec_list.append(WEC(ID, model, bus_location, run_sim))
        self.psse_dataframe.loc[
            self.psse_dataframe["BUS_ID"] == bus_location, "Type"
        ] = 4

    def create_cec(self, ID, model, bus_location, run_sim=True):
        self.cec_list.append(CEC(ID, model, bus_location, run_sim))
        self.psse_dataframe.loc[
            self.psse_dataframe["BUS_ID"] == bus_location, "Type"
        ] = 4

    def _psse_clear(self):
        """
        Description: This function clears all the data and resets the variables of the PSSe valuables
        input: None
        output: None
        """
        # initalized variables and files
        self.lst_param = ["PU", "P", "Q"]
        self.psse_dataframe = pd.DataFrame()
        self.history = {}
        # initialization functions
        Wec_grid.psspy.read(1, self.case_file)
        self.run_powerflow(self.solver)
        # program variables
        # self.psse_history['Start'] = self.psse_dataframe

    def _psse_run_dynamics(self, dyr_file=""):
        """
        Description: This function is a wrapper around the dynamic modeling process in PSSe, this function checks for .dyr file and
        then proceeds to run a simple simulations.
        input: you can add dyr file path or the function will just ask you via CL
        output: None
        """
        # check if dynamic file is loaded
        if self.dynamic_case_file == "":
            self.c = input("Dynamic File location")

        # Convert loads (3 step process):
        Wec_grid.psspy.conl(-1, 1, 1)

        Wec_grid.psspy.conl(
            sid=-1, all=1, apiopt=2, status=[0, 0], loadin=[100, 0, 0, 100]
        )

        Wec_grid.psspy.conl(-1, 1, 3)

        # Convert generators:
        Wec_grid.psspy.cong()

        # Solve for dynamics
        Wec_grid.psspy.ordr()
        Wec_grid.psspy.fact()
        Wec_grid.psspy.tysl()
        # Save converted case
        case_root = os.path.splitext(self.case_file)[0]
        Wec_grid.psspy.save(case_root + ".sav")

        Wec_grid.psspy.dyre_new(dyrefile=self.dynamic_case_file)

        # Add channels by subsystem
        #   BUS VOLTAGE
        Wec_grid.psspy.chsb(sid=0, all=1, status=[-1, -1, -1, 1, 13, 0])
        #   MACHINE SPEED
        Wec_grid.psspy.chsb(sid=0, all=1, status=[-1, -1, -1, 1, 7, 0])

        # Add channels individually
        #   BRANCH MVA
        # psspy.branch_mva_channel([-1,-1,-1,3001,3002],'1')

        path = os.path.abspath(os.path.dirname(self.case_file)) + "\\test.snp"
        # Save snapshot
        Wec_grid.psspy.snap(sfile=path)

        # Initialize
        Wec_grid.psspy.strt(outfile=path)

        # Run to 3 cycles
        time = 3.0 / 60.0
        Wec_grid.psspy.run(tpause=time)

    def run_WEC_Sim(self, wec_id, sim_config):
        for wec in self.wec_list:
            if wec.ID == wec_id:
                wec.WEC_Sim(sim_config)
                return True

    def run_CEC_Sim(self, cec_id, sim_config):
        for cec in self.cec_list:
            if cec.ID == cec_id:
                cec.CEC_Sim(sim_config)

    def _psse_run_powerflow(self, solver):
        """
        Description: This function runs the powerflow for PSSe for the given solver passed for the case in memory
        input:
             solver: the solver you want to use supported by PSSe, "fnsl" is a good default (str)
        output: None
        """
        if solver == "fnsl":
            Wec_grid.psspy.fnsl()
        elif solver == "GS":
            Wec_grid.psspy.solv()
        elif solver == "DC":
            Wec_grid.psspy.dclf_2(1, 1, [1, 0, 1, 2, 1, 1], [0, 0, 0], "1")
        else:
            print("error in run_pf")
        self._psse_get_values()

    def _pypsa_run_powerflow(self):
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

    def _psse_get_values(self):
        """
        Description: This function grabs all the important values we want in our dataframe
        input: None, uses lst_param tho
        output: Dataframe of the selected parameters for each bus.
        """
        lst = self.lst_param
        temp_dict = {}
        for bus_parameter in lst:
            if bus_parameter != "P" and bus_parameter != "Q":
                # grabs the bus parameter values for the specified parameter - list
                ierr, bus_parameter_values = Wec_grid.psspy.abusreal(
                    -1, string=bus_parameter
                )
                if ierr != 0:
                    print("error in get_values function")
                bus_add = {}
                for bus_index, value in enumerate(
                    bus_parameter_values[0]
                ):  # loops over those values to create bus num & value pairs
                    bus_add["BUS {}".format(bus_index + 1)] = value
                temp_dict[bus_parameter] = bus_add

        self.psse_dataframe = pd.DataFrame.from_dict(temp_dict)
        self.psse_dataframe = self.psse_dataframe.reset_index()
        self.psse_dataframe = self.psse_dataframe.rename(columns={"index": "Bus"})
        # gets the bus type (3 = swing)
        self.psse_dataframe["Type"] = Wec_grid.psspy.abusint(-1, string="TYPE")[1][0]
        self.psse_dataframe.insert(0, "BUS_ID", range(1, 1 + len(self.psse_dataframe)))
        self._psse_addGeninfo()
        self._psse_addLoadinfo()

        if "P" in lst:
            self._psse_get_p_or_q("P")
        if "Q" in lst:
            self._psse_get_p_or_q("Q")

        # Check if column exists, if not then initialize
        if "ΔP" not in self.psse_dataframe.columns:
            self.psse_dataframe["ΔP"] = 0.0  # default value
        if "ΔQ" not in self.psse_dataframe.columns:
            self.psse_dataframe["ΔQ"] = 0.0  # default value
        if "M_Angle" not in self.psse_dataframe.columns:
            self.psse_dataframe["M_Angle"] = 0.0  # default value
        if "M_Mag" not in self.psse_dataframe.columns:
            self.psse_dataframe["M_Mag"] = 0.0  # default value

        # Your loop remains unchanged
        for index, row in self.psse_dataframe.iterrows():
            mismatch = Wec_grid.psspy.busmsm(row["BUS_ID"])[1]
            real = mismatch.real
            imag = mismatch.imag
            angle = abs(mismatch)
            mag = cmath.phase(mismatch)
            self.psse_dataframe.at[index, "ΔP"] = mismatch.real
            self.psse_dataframe.at[index, "ΔQ"] = mismatch.imag
            self.psse_dataframe.at[index, "M_Angle"] = abs(mismatch)
            self.psse_dataframe.at[index, "M_Mag"] = cmath.phase(mismatch)

    def _psse_get_p_or_q(self, letter):
        """
        Description: retrieves P (activate) or Q (reactive) Voltage (in PU) and Voltage Angle for each Bus in the current loaded case
        input:
            letter: either P or Q as a string
        output: None
        """
        gen_values = self.psse_dataframe["{} Gen".format(letter)]  #
        load_values = self.psse_dataframe["{} Load".format(letter)]
        letter_list = [None] * len(self.psse_dataframe)

        for i in range(len(letter_list)):
            gen = gen_values[i]
            load = load_values[i]
            if (not pd.isnull(gen)) and (not pd.isnull(load)):
                letter_list[i] = gen - load
            else:
                if not pd.isnull(gen):
                    letter_list[i] = gen
                if not pd.isnull(load):
                    letter_list[i] = 0 - load  # gen is
        self.psse_dataframe["{}".format(letter)] = letter_list

    def _psse_busNum(self):
        """
        Description: Returns the number of Buses in the currently loaded case
        input: None
        output: Number of Buses
        """
        Wec_grid.psspy.bsys(0, 0, [0.0, 0.0], 1, [1], 0, [], 0, [], 0, [])
        ierr, all_bus = Wec_grid.psspy.abusint(0, 1, ["number"])
        return all_bus[0]

    def _psse_dc_injection(self, ibus, p, pf_solver, time):
        """
        Description: preforms the DC injection of the wec buses
        input:
            p: a list of active power set point in order(list)
            pf_solver: supported PSSe solver (Str)
            time: (Int)
        output: None
        """
        ierr = Wec_grid.psspy.machine_chng_3(ibus, "1", [], [p])
        if ierr > 0:
            print("Failed | machine_chng_3 code = {}".format(ierr))
        # psspy.dclf_2(status4=2)
        ierr = Wec_grid.psspy.dclf_2(1, 1, [1, 0, 1, 2, 0, 1], [0, 0, 1], "1")
        if ierr > 0:
            raise Exception("Error in DC injection")
        self._psse_get_values()
        self.psse_history[time] = self.psse_dataframe

    def _pypsa_ac_injection(self, p, v, time):
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

        self._pypsa_run_powerflow()
        self.pypsa_history[time] = self.pypsa_dataframe

    def _generate_load_curve(self, peak_value):
        """Generate a load curve based on normal distribution."""
        time_data = self.wec_list[0].dataframe.time.to_list()
        num_timesteps = len(time_data)
        midpoint_index = num_timesteps // 2
        midpoint_time = time_data[midpoint_index]
        time_range = time_data[-1] - time_data[0]

        if peak_value == 0:
            return np.zeros(num_timesteps)

        std_dev = time_range * 0.15
        curve = np.exp(
            -((np.array(time_data) - midpoint_time) ** 2) / (2 * std_dev**2)
        )
        return curve / curve.max() * peak_value

    def _generate_all_load_profiles(self, peak_values, hours=12, resolution=5):
        """Generate load profiles for all buses."""
        time_data = self.wec_list[0].dataframe.time.to_list()
        self.load_profiles = pd.DataFrame(
            {
                f"bus {bus_id}": self._generate_load_curve(peak)
                for bus_id, peak in peak_values.items()
            }
        )
        self.load_profiles["time"] = time_data

        # Rearrange to make time the first column
        self.load_profiles = self.load_profiles[
            ["time"] + [col for col in self.load_profiles if col != "time"]
        ]

    def _psse_update_load(self, ibus, time_step):
        """
        Update the load at the given bus using the load_profiles DataFrame for the given time_step.

        Parameters:
        - ibus (int): The bus number to update.
        - time_step (int): The time step index in the load_profiles DataFrame.
        - load_profiles_df (pd.DataFrame): The load profile DataFrame. Columns should be bus numbers and rows are the time steps.

        Returns:
        - int: Error code from PSS/E API call.
        """

        load_value = self.load_profiles.loc[
            self.load_profiles["time"] == time_step, f"bus {ibus}"
        ].values[0]

        # Default values from the documentation
        _id = "1"  # Identifier; default is '1' for most loads
        # intgar = [1, 0, 0, 0, 1, 0, 0]  # default values based on documentation
        realar = [
            load_value,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]  # Update active power, keep others default
        lodtyp = "CONSTP"  # Load type description; 'CONSTP' for constant power
        intgar = [1, 1, 1, 1, 1, 0, 0]

        # Call PSS/E API to update load
        ierr = Wec_grid.psspy.load_data_6(ibus, _id, intgar, realar, lodtyp)

        return ierr

    def _psse_ac_injection(self, start, end, p=None, v=None, time=None):
        """
        Description: WEC AC injection for PSSe powerflow solver
        input:
            p - a vector of active power values in order of bus num
            v - a vector of voltage mag PU values in order of bus num
            pf_solver - Power flow solving algorithm  (Default-"fnsl")
            time: (Int)
        output:
            no output but psse_dataframe is updated and so is psse_history
        """
        time = self.wec_list[0].dataframe.time.to_list()
        for t in time:
            print("time: {}".format(t))
            if t >= start and t <= end:
                for idx, wec_obj in enumerate(self.wec_list):

                    bus = wec_obj.bus_location
                    print("bus {}".format(bus))
                    pg = wec_obj.dataframe.loc[
                        wec_obj.dataframe.time == t
                    ].pg  # adjust activate power
                    ierr = Wec_grid.psspy.machine_data_2(
                        bus, "1", realar1=pg
                    )  # adjust activate power
                    if ierr > 0:
                        raise Exception("Error in AC injection")
                    vs = wec_obj.dataframe.loc[wec_obj.dataframe.time == t].vs
                    ierr = Wec_grid.psspy.bus_chng_4(
                        bus, 0, realar2=vs
                    )  # adjsut voltage mag PU
                    if ierr > 0:
                        raise Exception("Error in AC injection")

                    # self._psse_run_powerflow(self.solver)
                    self._psse_update_load(bus, t)
                    print("=======")

                self._psse_run_powerflow(self.solver)
                self.psse_history[t] = self.psse_dataframe
            if t > end:
                break
        return

    def _psse_bus_history(self, bus_num):
        """
        Description: this function grab all the data associated with a bus through the simulation
        input:
            bus_num: bus number (Int)
        output:
            bus_dataframe: a pandas dateframe of the history
        """
        # maybe I should add a filering parameter?

        bus_dataframe = pd.DataFrame()
        for time, df in self.psse_history.items():
            temp = pd.DataFrame(df.loc[df["BUS_ID"] == bus_num])
            temp.insert(0, "time", time)
            bus_dataframe = bus_dataframe.append(temp)
        return bus_dataframe

    def _psse_plot_bus(self, bus_num, time, arg_1="P", arg_2="Q"):
        """
        Description: This function plots the activate and reactive power for a given bus
        input:
            bus_num: the bus number we wanna viz (Int)
            time: a list with start and end time (list of Ints)
        output:
            matplotlib chart
        """
        ylabel = ""
        # if bus_num in self.wecBus_nums:
        #     ylabel = "kW"
        # sns.set_theme()
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle("Bus {}".format(bus_num))
        bus_df = self._psse_bus_history(bus_num)
        bus_df = bus_df.loc[(bus_df["time"] >= time[0]) & (bus_df["time"] <= time[1])]
        ax1.plot(
            bus_df.time,
            bus_df[arg_1],
            marker="o",
            markersize=5,
            markerfacecolor="green",
        )
        ax2.plot(
            bus_df.time,
            bus_df[arg_2],
            marker="o",
            markersize=5,
            markerfacecolor="green",
        )
        ax1.set(xlabel="Time(sec)", ylabel="{} - {}".format(arg_1, ylabel))
        ax2.set(xlabel="Time(sec)", ylabel="{} - {}".format(arg_2, ylabel))
        plt.show()
        return [bus_df[arg_1], bus_df[arg_2]]

    def plot_load_curve(self, bus_id):
        """Plot the load curve for a given bus."""
        # Check if the bus_id exists in load_profiles
        bus_col_name = f"bus {bus_id}"
        if bus_col_name not in self.load_profiles.columns:
            print(f"No load profile available for bus {bus_id}.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(
            self.load_profiles["time"],
            self.load_profiles[bus_col_name],
            label=f"Bus {bus_id} Load Curve",
            color="blue",
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Load (MW or MVAR)")
        plt.title(f"Load Curve for Bus {bus_id}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _psse_addGeninfo(self):
        """
        Description:
        input:
        output:
        """
        machine_bus_nums = Wec_grid.psspy.amachint(-1, 4, "NUMBER")[1][
            0
        ]  # get the bus numbers of the machines - list
        # grabs the complex values for the machine
        ierr, machine_bus_values = Wec_grid.psspy.amachcplx(-1, 1, "PQGEN")
        if ierr != 0:
            raise Exception("Error in grabbing PGGEN values in addgen function")
        p_gen_df_list = [None] * len(self.psse_dataframe)
        q_gen_df_list = [None] * len(self.psse_dataframe)
        # iterate over the machine values
        for list_index, value in enumerate(machine_bus_values[0]):
            p_gen_df_list[
                machine_bus_nums[list_index] - 1
            ] = value.real  # -1 is for the offset
            q_gen_df_list[machine_bus_nums[list_index] - 1] = value.imag

        self.psse_dataframe["P Gen"] = p_gen_df_list
        self.psse_dataframe["Q Gen"] = q_gen_df_list

    def _psse_addLoadinfo(self):
        """
        Description: this function grabs the load values from the PSSe system
        input: None
        output: None but updates psse_dataframe with load data
        """
        load_bus_nums = Wec_grid.psspy.aloadint(-1, 4, "NUMBER")[1][
            0
        ]  # get the bus numbers of buses with loads - list
        ierr, load_bus_values = Wec_grid.psspy.aloadcplx(-1, 1, "MVAACT")  # load values
        if ierr != 0:
            raise Exception("Error in grabbing PGGEN values in addgen function")
        p_load_df_list = [None] * len(self.psse_dataframe)
        q_load_df_list = [None] * len(self.psse_dataframe)

        # iterate over the machine values
        for list_index, value in enumerate(load_bus_values[0]):
            p_load_df_list[
                load_bus_nums[list_index] - 1
            ] = value.real  # -1 is for the offset
            q_load_df_list[load_bus_nums[list_index] - 1] = value.imag
        self.psse_dataframe["P Load"] = p_load_df_list
        self.psse_dataframe["Q Load"] = q_load_df_list

    def viz(self, dataframe=None):
        from ipywidgets import Dropdown
        import ipywidgets as widgets

        # Map Bus Types to colors and labels
        color_map = {1: "grey", 2: "lightgreen", 3: "red", 4: "lightblue"}
        color_map_1 = {1: "grey", 2: "Green", 3: "Red", 4: "Blue"}
        label_map = {1: "PQ Bus", 2: "PV Bus", 3: "Swing Bus", 4: "WEC Bus"}

        if dataframe is None:
            dataframe = self.psse_dataframe

        dataframe_copy = dataframe.copy()

        dataframe_copy.fillna(0, inplace=True)

        dataframe_copy.loc[:, ["P", "Q", "ANGLED"]] = dataframe_copy.loc[
            :, ["P", "Q", "ANGLED"]
        ].clip(-1.0e100, 1.0e100)

        G_cyto = ipycytoscape.CytoscapeWidget()

        def node_click(node):
            P = node["data"]["P"]
            Q = node["data"]["Q"]
            angle = node["data"]["angle"]

            node_info.value = f"Bus: {node['data']['id']} | Type: {label_map[node['data']['type']]} | P {format(P, '.3g')} | Q {format(Q, '.3g')} | Angle {format(angle, '.3g')}"

        G_cyto.on("node", "click", node_click)

        G_cyto.max_zoom = 1.1
        G_cyto.min_zoom = 0.5

        for index, row in dataframe_copy.iterrows():
            node_data = {
                "id": str(row["BUS_ID"]),
                "label": f"BUS: {row['BUS_ID']}",
                "type": row["Type"],
                "classes": color_map[row["Type"]],
                "P": row["P"],
                "Q": row["Q"],
                "angle": row["ANGLED"],
            }
            node = ipycytoscape.Node(data=node_data)
            G_cyto.graph.add_node(node)

        ierr, (fromnumber, tonumber) = Wec_grid.psspy.abrnint(
            sid=-1, flag=3, string=["FROMNUMBER", "TONUMBER"]
        )
        for index in range(len(fromnumber)):
            edge_data = {
                "source": str(fromnumber[index]),
                "target": str(tonumber[index]),
            }
            edge = ipycytoscape.Edge(data=edge_data)
            G_cyto.graph.add_edge(edge)

        G_cyto.set_style(
            [
                {
                    "selector": "node",
                    "css": {
                        "background-color": "data(classes)",
                        "label": "data(label)",
                        "text-wrap": "wrap",
                    },
                },
                {"selector": "node.hide", "style": {"display": "none"}},
                {
                    "selector": "edge",
                    "style": {
                        "width": 4,
                        "line-color": "#9dbaea",
                        "target-arrow-shape": "none",
                    },
                },
            ]
        )

        items = []

        for bus_type, color in color_map_1.items():
            color_box = widgets.Box(
                layout=widgets.Layout(width="20px", height="20px", background=color)
            )

            description = {
                1: " - Grey",
                2: " - Green",
                3: " - Red",
                4: " - Blue",
            }[bus_type]

            label = widgets.Label(value=f"{label_map[bus_type]}{description}")

            items.append(widgets.HBox([color_box, label]))

        legend_title = widgets.Label(
            value="Legend:", layout=widgets.Layout(margin="0 0 5px 0")
        )

        legend = widgets.VBox(
            [legend_title] + items, layout=widgets.Layout(border="solid", padding="5px")
        )

        node_info = widgets.Label()

        bus_type_dropdown = Dropdown(
            options=[("All", 0)] + [(label_map[i], i) for i in range(1, 5)],
            value=0,
            description="Bus Type:",
        )

        def filter_nodes(bus_type):
            if bus_type == 0:
                for node in G_cyto.graph.nodes:
                    node.classes = color_map[int(node.data["type"])]
            else:
                for node in G_cyto.graph.nodes:
                    if int(node.data["type"]) == bus_type:
                        node.classes = color_map[bus_type]
                    else:
                        node.classes = "hide"

        filter_nodes(bus_type_dropdown.value)

        bus_type_dropdown.observe(
            lambda change: filter_nodes(change.new), names="value"
        )

        display(widgets.VBox([bus_type_dropdown, G_cyto, legend, node_info]))

    def MiGrid_to_db(self):
        # broken 6/26/23
        """
        Description:
        input:
        output:
        """

        # Connect to the database
        conn = sqlite3.connect("WEC-SIM.db")
        c = conn.cursor()

        # directory_path = '../input_files/Run0/OutputData/'  # need to update this
        file_patterns = [
            r"gen\d+PSet\d+Run\d+\.nc",  # Pattern for gen files
            r"wtg\d+PAvailSet\d+Run\d+",  # Pattern for wtg files
            r"wtg\d+PSet\d+Run\d+",  # Pattern for other wtg files
        ]

        matching_files = []

        for file_pattern in file_patterns:
            pattern = re.compile(file_pattern)
            matching_files += [
                f for f in os.listdir(directory_path) if pattern.match(f)
            ]

        # Drop all tables
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = c.fetchall()
        for table_name in tables:
            if (table_name[0] + ".nc") in matching_files:
                c.execute(f"DROP TABLE IF EXISTS {table_name[0]}")

        # Commit the changes to the database
        conn.commit()

        for file in matching_files:
            count = 0
            nc = netCDF4.Dataset(directory_path + file)

            # Create a new table for the file
            # Use the filename as the table name
            tablename = os.path.splitext(file)[0]
            c.execute(f"CREATE TABLE {tablename} (time float64, gen_value float64)")

            # need to adjust to only grab 5 min resolution
            i = 0
            sum = 0
            steps = 0
            while count <= 72:  # 72 entires of 5 min soooo 6 hours?
                time_value_raw = nc.variables["time"][i].item()
                time_value = datetime.fromtimestamp(
                    time_value_raw
                )  # .strftime('%Y-%m-%d %H:%M:%S')
                gen_value = nc.variables["value"][i].item()

                sum += gen_value
                steps += 1

                if time_value.minute % 5 == 0 and time_value.second == 0:
                    sma = sum / steps
                    c.execute(
                        f"INSERT INTO {tablename} (time , gen_value) VALUES (?, ?)",
                        (time_value, sma),
                    )
                    # print(str(time_value))
                    count += 1
                    sum = 0
                    steps = 0
                i += 1
        conn.commit()
        conn.close()

    # def pull_MiGrid(self):

    #     # Connect to the database
    #     c = sqlite3.connect('../input_files/WEC-SIM.db')

    #     for file in self.migrid_file_names:
    #         base = os.path.splitext(file)[0]
    #         migrid_data = pd.read_sql_query("SELECT * from {}".format(base), c)
    #         self.migrid_data[base] = migrid_data
    #     c.close()

    def Migrid_file(self, file_patterns=None):

        directory_path = "../input_files/Run0/OutputData/"  # need to update this
        matching_files = []
        if file_patterns is None:

            file_patterns = [
                r"gen\d+PSet\d+Run\d+\.nc",  # Pattern for gen files
                r"wtg\d+PAvailSet\d+Run\d+",  # Pattern for wtg files
                r"wtg\d+PSet\d+Run\d+",  # Pattern for other wtg files
            ]

            for file_pattern in file_patterns:
                pattern = re.compile(file_pattern)
                matching_files += [
                    f for f in os.listdir(directory_path) if pattern.match(f)
                ]

        else:
            for file_pattern in file_patterns:
                pattern = re.compile(file_pattern)
                matching_files += [
                    f for f in os.listdir(directory_path) if pattern.match(f)
                ]

        return matching_files

    def compare_v(self):
        """
        Description:
        input:
        output:
        """
        v_mag = pd.concat(
            [
                self.psse_dataframe[["PU"]],
                self.pypsa_dataframe[["v_mag_pu_set"]]
                .reset_index()
                .drop(columns=["Bus"]),
            ],
            axis=1,
        ).rename(
            columns={"PU": "PSSe voltage mag", "v_mag_pu_set": "pyPSA voltage mag"}
        )
        v_mag["abs diff"] = (
            v_mag["PSSe voltage mag"] - v_mag["pyPSA voltage mag"]
        ).abs()
        return v_mag

    def compare_p(self):
        """
        Description:
        input:
        output:
        """
        p_load = pd.concat(
            [
                self.psse_dataframe[["P Load"]],
                self.pypsa_dataframe[["Pd"]].reset_index().drop(columns=["Bus"]),
            ],
            axis=1,
        ).rename(columns={"P Load": "PSSe P-Load", "Pd": "pyPSA P-Load"})
        p_load["abs diff"] = (p_load["PSSe P-Load"] - p_load["pyPSA P-Load"]).abs()
        return p_load

    def compare_q(self):
        """
        Description:
        input:
        output:
        """
        q_load = pd.concat(
            [
                self.psse_dataframe[["Q Load"]],
                self.pypsa_dataframe[["Qd"]].reset_index().drop(columns=["Bus"]),
            ],
            axis=1,
        ).rename(columns={"Q Load": "PSSe Q-Load", "Qd": "pyPSA Q-Load"})
        q_load["abs diff"] = (q_load["PSSe Q-Load"] - q_load["pyPSA Q-Load"]).abs()
        return q_load

    def _psse_adjust_gen(self, bus_num, p=None, v=None, q=None):
        """
        Description: Given a generator bus number. Adjust the values based on the parameters passed.
        input:
        output:
        """

        if p is not None:
            ierr = Wec_grid.psspy.machine_data_2(
                bus_num, "1", realar1=p
            )  # adjust activate power
            if ierr > 0:
                raise Exception("Error in AC injection")

        if v is not None:
            ierr = Wec_grid.psspy.bus_chng_4(
                bus_num, 0, realar2=v
            )  # adjsut voltage mag PU
            if ierr > 0:
                raise Exception("Error in AC injection")

        if q is not None:
            ierr = Wec_grid.psspy.machine_data_2(
                bus_num, "1", realar2=q
            )  # adjust Reactivate power
            if ierr > 0:
                raise Exception("Error in AC injection")

    # def pull_wec_data(self, wec_num=None):
    #     """
    #     Description: Pulls WEC data from the database. If wec_num is provided, pulls data for that specific wec.
    #     input: wec_num (optional) - the number of the specific wec to pull data for
    #     output: None
    #     """
    #     # Connect to the database
    #     conn = sqlite3.connect(str(current_dir) + "\\WEC-SIM.db") # need to update with dynamic location

    #     #temp = os.getcwd()

    #     # If wec_num is provided, use it to create a single-element list
    #     wec_numbers = [wec_num] if wec_num is not None else self.wec_list

    #     for num in wec_numbers:
    #         wec = pd.read_sql_query("SELECT * from WEC_output_{}".format(num), conn)
    #         self.wec_data[num] = wec

    #     conn.close()

    def clear_database(self):
        """
        Clears all the tables from the database.
        """

        # Fetch all table names from the database
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = dbQuery(tables_query)

        # Drop each table from the database
        for table_name in tables:
            drop_query = f"DROP TABLE IF EXISTS {table_name[0]}"
            dbQuery(drop_query)

    def migrid_warm_start(self):
        """
        Description:
        input:
        output:
        """
        generator_buses = self.psse_dataframe[
            self.psse_dataframe["Type"] == 2
        ].BUS_ID.to_list()
        regular_gens = [x for x in generator_buses if x not in self.wecBus_nums]
        print(regular_gens)
        pointer = 0
        for key, value in self.migrid_data.items():
            if key[:3] == "gen":
                self._psse_adjust_gen(
                    bus_num=regular_gens[pointer], p=value.iloc[0].gen_value
                )
                pointer += 1

    def compare(self):
        """
        Description:
        input:
        output:
        """
        py = self.pypsa_dataframe.copy()
        py = py.reset_index(level=0, drop=True)
        py_final = py.rename(
            columns={"Pd": "P Load", "Qd": "Q Load", "v_mag_pu_set": "PU"}
        )[["PU", "P Load", "Q Load"]].copy()
        py_final = py_final.fillna(0)
        ps = self.psse_dataframe.copy()
        ps_final = ps[["PU", "P Load", "Q Load"]]
        ps_final = ps_final.fillna(0)
        return ps_final.compare(py_final, keep_equal=True, keep_shape=True).rename(
            columns={"self": "pyPSA", "other": "PSSe"}
        )
