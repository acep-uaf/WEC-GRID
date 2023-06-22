# Wec_grid_class
import os
import sys
import pandas as pd
import numpy as np
import re
import datetime
import netCDF4
import pypsa
import pypower.api as pypower
import matlab.engine
import networkx as nx
import seaborn as sns
import sqlite3
import matplotlib.pyplot as plt
import datetime
from datetime import datetime, timezone
from datetime import datetime, timedelta

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

paths = []
current_dir = os.path.dirname(__file__)
with open('{}\\path_config.txt'.format(current_dir), 'r') as fp:
    while 1:
        line = fp.readline()
        if len(line) == 0:  # end of file break
            break
        temp = line.split('\n')
        paths.append(r'{}'.format(temp[0])) #will allow spaces in path strings

psse_path = paths[0]
wec_sim_path = paths[1]
wec_model_path = paths[2]
wec_grid_class_path = paths[3]
wec_grid_folder = paths[4]

# Path stuff

sys.path.append(psse_path + "\\PSSPY37")
sys.path.append(psse_path + "\\PSSBIN")
sys.path.append(psse_path + "\\PSSLIB")
sys.path.append(psse_path + "\\EXAMPLE")
sys.path.append(psse_path + "\\PSSPY37")
os.environ['PATH'] = (psse_path + "\\PSSPY37;" + psse_path + "\\PSSBIN;" + psse_path + "\\EXAMPLE;" + os.environ['PATH'])


import psse35
psse35.set_minor(3)
import psspy



class WEC:
    def __init__(self, ID, model, bus_location, Pmax=9999, Pmin=-9999, Qmax=9999, Qmin=-9999):
        self.ID = ID
        self.bus_location = bus_location
        self.model = model
        self.dataframe = pd.DataFrame()
        self.Pmax = Pmax
        self.Pmin = Pmin
        self.Qmax = Qmax
        self.Qmin = Qmin



class Wec_grid:
    # def __init__(self, case, solver, wec_bus, software = "PSSe"):

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
        self.wec_list = [] # list 
        self.wec_data = {}
        self.psse_dataframe = pd.DataFrame()
        self.pypsa_dataframe = pd.DataFrame()
        self.migrid_file_names = self.Migrid_file()
        #self.wec_data = {}
        self.migrid_data = {}


    def initalize_psse(self, solver):
        """
        Description: Initializes a PSSe case, uses the topology passed at original initialization
        input: 
            solver: the solver you want to use supported by PSSe, "fnsl" is a good default (str)
        output: None
        """
        psspy.psseinit(50)
        self.softwares.append("psse")
        self.psse_history = {}
        self.lst_param = ['BASE', 'PU', 'ANGLED', 'P', 'Q']
        self.solver = solver
        self.dynamic_case_file = ""

        #self.psse_dataframe = pd.DataFrame()
        self._i = psspy.getdefaultint()
        self._f = psspy.getdefaultreal()
        self._s = psspy.getdefaultchar()

        if self.case_file.endswith('.sav'):
            psspy.case(self.case_file)
        elif self.case_file.endswith('.raw'):
            psspy.read(1, self.case_file)
        elif self.case_file.endswith('.RAW'):
            psspy.read(1, self.case_file)
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
        eng.workspace['case_path'] = self.case_file 
        eng.eval("mpc = psse2mpc(case_path)", nargout=0)
        eng.eval("savecase('here.mat',mpc,1.0)", nargout=0)

        # Load the MATPOWER case file from a .mat file
        ppc = pypower.loadcase("./here.mat")

        # Convert Pandapower network to PyPSA network
        pypsa_network = pypsa.Network()
        pypsa_network.import_from_pypower_ppc(ppc, overwrite_zero_s_nom=True)
        pypsa_network.set_snapshots(
            [datetime.now().strftime('%m/%d/%Y %H:%M:%S')])
        

        pypsa_network.pf()

        self.pypsa_dataframe = pypsa_network.buses
        self.pypsa_object = pypsa_network
        self.pypsa_history[-1] = self.pypsa_dataframe
        self.pypsa_object_history[-1] = self.pypsa_object


    def add_wec(self, wec):
        self.wec_list.append(wec)
        for w in self.wec_list:
            ierr = psspy.machine_data_2(w.bus_location, '1', realar3=w.Qmax, realar4=w.Qmin, realar5=w.Pmax, realar6=w.Pmin) # adjust activate power 
            if ierr > 0:
                raise Exception('Error adding WEC')


    def _psse_clear(self):
        """
        Description: This function clears all the data and resets the variables of the PSSe valuables
        input: None
        output: None
        """
        # initalized variables and files
        self.lst_param = ['PU', 'P', 'Q']
        self.psse_dataframe = pd.DataFrame()
        self.history = {}
        # initialization functions
        psspy.read(1, self.case_file)
        self.run_powerflow(self.solver)
        # program variables
        #self.psse_history['Start'] = self.psse_dataframe

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
        psspy.conl(-1, 1, 1)

        psspy.conl(sid=-1,
                   all=1,
                   apiopt=2,
                   status=[0, 0],
                   loadin=[100, 0, 0, 100]
                   )

        psspy.conl(-1, 1, 3)

        # Convert generators:
        psspy.cong()

        # Solve for dynamics
        psspy.ordr()
        psspy.fact()
        psspy.tysl()
        # Save converted case
        case_root = os.path.splitext(self.case_file)[0]
        psspy.save(case_root + ".sav")

        psspy.dyre_new(dyrefile=self.dynamic_case_file)

        # Add channels by subsystem
        #   BUS VOLTAGE
        psspy.chsb(sid=0, all=1, status=[-1, -1, -1, 1, 13, 0])
        #   MACHINE SPEED
        psspy.chsb(sid=0, all=1, status=[-1, -1, -1, 1, 7, 0])

        # Add channels individually
        #   BRANCH MVA
        # psspy.branch_mva_channel([-1,-1,-1,3001,3002],'1')

        path = os.path.abspath(os.path.dirname(self.case_file)) + "\\test.snp"
        # Save snapshot
        psspy.snap(sfile=path)

        # Initialize
        psspy.strt(outfile=path)

        # Run to 3 cycles
        time = 3.0 / 60.0
        psspy.run(tpause=time)

    def run_WEC_Sim(self, wec_id, sim_length, Tsample, waveHeight, wavePeriod, waveSeed):
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
        eng = matlab.engine.start_matlab()
        print("Matlab Engine estbalished")
        eng.cd(wec_model_path)
        path = wec_sim_path  # Update to match your WEC-SIM source location
        eng.addpath(eng.genpath(path), nargout=0)
        print("calling W2G")

        # Variables required to run w2gSim
        eng.workspace['wecId'] = wec_id
        eng.workspace['simLength'] = sim_length
        eng.workspace['Tsample'] = Tsample
        eng.workspace['waveHeight'] = waveHeight
        eng.workspace['wavePeriod'] = wavePeriod
        eng.workspace['waveSeed'] = waveSeed
        eng.eval(
            "m2g_out = w2gSim(wecId,simLength,Tsample,waveHeight,wavePeriod,waveSeed);", nargout=0)
        print("displaying simulation plots")
        # display(Image(filename="..\input_files\W2G_RM3\sim_figures\Pgen_Pgrid_Qgrid.jpg"))
        # display(Image(filename="..\input_files\W2G_RM3\sim_figures\Pgen_Pgrid_comp.jpg"))
        # display(Image(filename="..\input_files\W2G_RM3\sim_figures\DClink_voltage.jpg"))
        print("calling PSSe formatting")
        conn = sqlite3.connect('../input_files/WEC-SIM.db')
        eng.eval("WECsim_to_PSSe_dataFormatter", nargout=0)
        print("sim complete")

    def _psse_run_powerflow(self, solver):
        """
        Description: This function runs the powerflow for PSSe for the given solver passed for the case in memory
        input: 
             solver: the solver you want to use supported by PSSe, "fnsl" is a good default (str)
        output: None
        """
        if solver == 'fnsl':
            psspy.fnsl()
        elif solver == 'GS':
            psspy.solv()
        elif solver == 'DC':
            psspy.dclf_2(1, 1, [1, 0, 1, 2, 1, 1], [0, 0, 0], '1')
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
                ierr, bus_parameter_values = psspy.abusreal(
                    -1, string=bus_parameter)
                if ierr != 0:
                    print("error in get_values function")
                bus_add = {}
                for bus_index, value in enumerate(
                        bus_parameter_values[0]):  # loops over those values to create bus num & value pairs
                    bus_add['BUS {}'.format(bus_index + 1)] = value
                temp_dict[bus_parameter] = bus_add

        self.psse_dataframe = pd.DataFrame.from_dict(temp_dict)
        self.psse_dataframe = self.psse_dataframe.reset_index()
        self.psse_dataframe = self.psse_dataframe.rename(
            columns={'index': "Bus"})
        # gets the bus type (3 = swing)
        self.psse_dataframe['Type'] = psspy.abusint(-1, string="TYPE")[1][0]
        self.psse_dataframe.insert(
            0, "BUS_ID", range(1, 1 + len(self.psse_dataframe)))
        self._psse_addGeninfo()
        self._psse_addLoadinfo()

        if "P" in lst:
            self._psse_get_p_or_q('P')
        if "Q" in lst:
            self._psse_get_p_or_q('Q')

    def _psse_get_p_or_q(self, letter):
        """
        Description: retrieves P (activate) or Q (reactive) Voltage (in PU) and Voltage Angle for each Bus in the current loaded case
        input: 
            letter: either P or Q as a string
        output: None
        """
        gen_values = self.psse_dataframe['{} Gen'.format(letter)]  #
        load_values = self.psse_dataframe['{} Load'.format(letter)]
        letter_list = [None] * len(self.psse_dataframe)

        for i in range(len(letter_list)):
            gen = gen_values[i]
            load = load_values[i]
            if (not pd.isnull(gen)) and (not pd.isnull(load)):
                letter_list[i] = gen - load
            else:
                if (not pd.isnull(gen)):
                    letter_list[i] = gen
                if (not pd.isnull(load)):
                    letter_list[i] = 0 - load  # gen is
        self.psse_dataframe['{}'.format(letter)] = letter_list

    def _psse_busNum(self):
        """
        Description: Returns the number of Buses in the currently loaded case
        input: None 
        output: Number of Buses
        """
        psspy.bsys(0, 0, [0.0, 0.0], 1, [1], 0, [], 0, [], 0, [])
        ierr, all_bus = psspy.abusint(0, 1, ['number'])
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
        ierr = psspy.machine_chng_3(ibus, "1", [], [p])
        if ierr > 0:
            print("Failed | machine_chng_3 code = {}".format(ierr))
        # psspy.dclf_2(status4=2)
        ierr = psspy.dclf_2(1, 1, [1, 0, 1, 2, 0, 1], [0, 0, 1], '1')
        if ierr > 0:
            raise Exception('Error in DC injection')
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

            self.pypsa_object.generators.loc[self.pypsa_object.generators.bus == str(
                bus), "v_set_pu"] = v[idx]
            self.pypsa_object.generators.loc[self.pypsa_object.generators.bus == str(
                bus), "p_set"] = p[idx]

        self._pypsa_run_powerflow()
        self.pypsa_history[time] = self.pypsa_dataframe

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
        time = self.wec_data[self.wec_list[0]].time.tolist()
        for t in time:
            if t >= start and t <= end:
                for idx, bus in enumerate(self.wec_list):
                    ierr = psspy.machine_data_2(bus, '1', realar1=self.wec_data[bus].loc[self.wec_data[bus].time == t].pg) # adjust activate power 
                    if ierr > 0:
                        raise Exception('Error in AC injection')

                    ierr = psspy.bus_chng_4(bus, 0, realar2=self.wec_data[bus].loc[self.wec_data[bus].time == t].vs) # adjsut voltage mag PU
                    if ierr > 0:
                        raise Exception('Error in AC injection')

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
            temp.insert(0, 'time', time)
            bus_dataframe = bus_dataframe.append(temp)
        return bus_dataframe

    def _psse_plot_bus(self, bus_num, time):
        """
        Description: This function plots the activate and reactive power for a given bus
        input:
            bus_num: the bus number we wanna viz (Int)
            time: a list with start and end time (list of Ints)
        output:
            matplotlib chart
        """
        ylabel = "MW"
        # if bus_num in self.wecBus_nums:
        #     ylabel = "kW"
        # sns.set_theme()
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle("Bus {}".format(bus_num))
        bus_df = self._psse_bus_history(bus_num)
        bus_df = bus_df.loc[(bus_df["time"] >= time[0]) &
                            (bus_df["time"] <= time[1])]
        ax1.plot(bus_df.time, bus_df["P"], marker="o",
                 markersize=5, markerfacecolor="green")
        ax2.plot(bus_df.time, bus_df["Q"], marker="o",
                 markersize=5, markerfacecolor="green")
        ax1.set(xlabel="Time(sec)", ylabel="P - {}".format(ylabel))
        ax2.set(xlabel="Time(sec)", ylabel="Q - {}".format(ylabel))
        plt.show()

    def _psse_addGeninfo(self):
        """
        Description:
        input:
        output:
        """
        machine_bus_nums = psspy.amachint(-1, 4, "NUMBER")[
            1][0]  # get the bus numbers of the machines - list
        # grabs the complex values for the machine
        ierr, machine_bus_values = psspy.amachcplx(-1, 1, 'PQGEN')
        if ierr != 0:
            raise Exception(
                'Error in grabbing PGGEN values in addgen function')
        p_gen_df_list = [None] * len(self.psse_dataframe)
        q_gen_df_list = [None] * len(self.psse_dataframe)
        # iterate over the machine values
        for list_index, value in enumerate(machine_bus_values[0]):
            p_gen_df_list[machine_bus_nums[list_index] -
                          1] = value.real  # -1 is for the offset
            q_gen_df_list[machine_bus_nums[list_index] - 1] = value.imag

        self.psse_dataframe['P Gen'] = p_gen_df_list
        self.psse_dataframe['Q Gen'] = q_gen_df_list

    def _psse_addLoadinfo(self):
        """
        Description: this function grabs the load values from the PSSe system
        input: None
        output: None but updates psse_dataframe with load data
        """
        load_bus_nums = psspy.aloadint(-1, 4, "NUMBER")[
            1][0]  # get the bus numbers of buses with loads - list
        ierr, load_bus_values = psspy.aloadcplx(-1, 1, "MVAACT")  # load values
        if ierr != 0:
            raise Exception(
                'Error in grabbing PGGEN values in addgen function')
        p_load_df_list = [None] * len(self.psse_dataframe)
        q_load_df_list = [None] * len(self.psse_dataframe)

        # iterate over the machine values
        for list_index, value in enumerate(load_bus_values[0]):
            p_load_df_list[load_bus_nums[list_index] -
                           1] = value.real  # -1 is for the offset
            q_load_df_list[load_bus_nums[list_index] - 1] = value.imag
        self.psse_dataframe['P Load'] = p_load_df_list
        self.psse_dataframe['Q Load'] = q_load_df_list

    def _psse_viz(self):
        """
        Description:
        input: 
        output:
        """
        # https://github.com/jtrain/psse_sld_viewer

        
        ierr, (fromnumber, tonumber) = psspy.abrnint(
            sid=-1, flag=3, string=["FROMNUMBER", "TONUMBER"])
        ierr, (og_weights) = psspy.abrncplx(sid=-1, flag=3,
                                            string=["RX"])  # Branch impedance, in pu
        weights = map(abs, og_weights[0])
        G = nx.DiGraph()
        G.add_weighted_edges_from(zip(fromnumber, tonumber, weights))
        plt.figure(1, figsize=(8, 8))
        ax = plt.gca()
        file_name = os.path.basename(self.case_file)
        file = os.path.splitext(file_name)
        ax.set_title('{} | Directed Graph of Branch impedance'.format(file[0]))
        pos = nx.spring_layout(G, 1000)
        nx.draw(G, pos, node_size=1000, node_color="#32a852",
                font_size=14, edge_color="#444444", with_labels=True, ax=ax)
        _ = ax.axis('off')

    def MiGrid_to_db(self):
        """
        Description:
        input: 
        output:
        """

        # Connect to the database
        conn = sqlite3.connect('../input_files/WEC-SIM.db')
        c = conn.cursor()

        directory_path = '../input_files/Run0/OutputData/'  # need to update this
        file_patterns = [
            r"gen\d+PSet\d+Run\d+\.nc",  # Pattern for gen files
            r"wtg\d+PAvailSet\d+Run\d+",  # Pattern for wtg files
            r"wtg\d+PSet\d+Run\d+"  # Pattern for other wtg files
        ]

        matching_files = []

        for file_pattern in file_patterns:
            pattern = re.compile(file_pattern)
            matching_files += [f for f in os.listdir(
                directory_path) if pattern.match(f)]

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
            c.execute(
                f'CREATE TABLE {tablename} (time float64, gen_value float64)')

            # need to adjust to only grab 5 min resolution
            i = 0
            sum = 0
            steps = 0
            while (count <= 72):  # 72 entires of 5 min soooo 6 hours?
                time_value_raw = nc.variables["time"][i].item()
                time_value = datetime.fromtimestamp(
                    time_value_raw)  # .strftime('%Y-%m-%d %H:%M:%S')
                gen_value = nc.variables["value"][i].item()

                sum += gen_value
                steps += 1

                if time_value.minute % 5 == 0 and time_value.second == 0:
                    sma = sum / steps 
                    c.execute(
                        f'INSERT INTO {tablename} (time , gen_value) VALUES (?, ?)', (time_value, sma))
                    # print(str(time_value))
                    count += 1
                    sum = 0
                    steps = 0
                i += 1
        conn.commit()
        conn.close()

    def pull_MiGrid(self):

        # Connect to the database
        c = sqlite3.connect('../input_files/WEC-SIM.db')
        
        for file in self.migrid_file_names:
            base = os.path.splitext(file)[0]
            migrid_data = pd.read_sql_query("SELECT * from {}".format(base), c)
            self.migrid_data[base] = migrid_data
        c.close()
        
    def Migrid_file(self, file_patterns=None):

        directory_path = '../input_files/Run0/OutputData/'  # need to update this
        matching_files = []
        if file_patterns is None:

            file_patterns = [
                r"gen\d+PSet\d+Run\d+\.nc",  # Pattern for gen files
                r"wtg\d+PAvailSet\d+Run\d+",  # Pattern for wtg files
                r"wtg\d+PSet\d+Run\d+"  # Pattern for other wtg files
            ]

            for file_pattern in file_patterns:
                pattern = re.compile(file_pattern)
                matching_files += [f for f in os.listdir(
                    directory_path) if pattern.match(f)]
                

        else:
            for file_pattern in file_patterns:
                pattern = re.compile(file_pattern)
                matching_files += [f for f in os.listdir(
                    directory_path) if pattern.match(f)]
                
        return matching_files

    def compare_v(self):
        """
        Description:
        input: 
        output:
        """
        v_mag = pd.concat([self.psse_dataframe[["PU"]], self.pypsa_dataframe[["v_mag_pu_set"]].reset_index().drop(
            columns=['Bus'])], axis=1).rename(columns={"PU": "PSSe voltage mag", "v_mag_pu_set": "pyPSA voltage mag"})
        v_mag["abs diff"] = (v_mag["PSSe voltage mag"] -
                             v_mag["pyPSA voltage mag"]).abs()
        return v_mag

    def compare_p(self):
        """
        Description:
        input: 
        output:
        """
        p_load = pd.concat([self.psse_dataframe[["P Load"]], self.pypsa_dataframe[["Pd"]].reset_index(
        ).drop(columns=['Bus'])], axis=1).rename(columns={"P Load": "PSSe P-Load", "Pd": "pyPSA P-Load"})
        p_load["abs diff"] = (p_load["PSSe P-Load"] -
                              p_load["pyPSA P-Load"]).abs()
        return p_load

    def compare_q(self):
        """
        Description:
        input: 
        output:
        """
        q_load = pd.concat([self.psse_dataframe[["Q Load"]], self.pypsa_dataframe[["Qd"]].reset_index(
        ).drop(columns=['Bus'])], axis=1).rename(columns={"Q Load": "PSSe Q-Load", "Qd": "pyPSA Q-Load"})
        q_load["abs diff"] = (q_load["PSSe Q-Load"] -
                              q_load["pyPSA Q-Load"]).abs()
        return q_load

    def _psse_adjust_gen(self, bus_num, p=None, v=None, q=None):
        """
        Description: Given a generator bus number. Adjust the values based on the parameters passed.
        input: 
        output:
        """

        if p is not None:
            ierr = psspy.machine_data_2(bus_num, '1', realar1=p) # adjust activate power 
            if ierr > 0:
                raise Exception('Error in AC injection')

        if v is not None:
            ierr = psspy.bus_chng_4(bus_num, 0, realar2=v) # adjsut voltage mag PU
            if ierr > 0:
                raise Exception('Error in AC injection')
        
        if q is not None:
            ierr = psspy.machine_data_2(bus_num, '1', realar2=q) # adjust Reactivate power 
            if ierr > 0:
                raise Exception('Error in AC injection')

    def pull_wec_data(self):
        """
        Description:
        input: 
        output:
        """
        # Connect to the database
        conn = sqlite3.connect('../input_files/WEC-SIM.db') # need to update with dynamic location
        #c = conn.cursor()
        for wec_num in self.wec_list:
            wec = pd.read_sql_query("SELECT * from WEC_output_{}".format(wec_num), conn)
            self.wec_data[wec_num] = wec
            #elf.wec_data[i] = wec
        conn.close()

    def clear_database(self):
        """
        Description:
        input: 
        output:
        """
        # Connect to the database
        conn = sqlite3.connect('../input_files/WEC-SIM.db') # need to update with dynamic location
        c = conn.cursor()
        # Drop all tables
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = c.fetchall()
        for table_name in tables:
            c.execute(f"DROP TABLE IF EXISTS {table_name[0]}")
        # Commit the changes to the database
        conn.commit()
        # Close the database connection
        conn.close()

    def migrid_warm_start(self):
        """
        Description:
        input: 
        output:
        """
        generator_buses = self.psse_dataframe[self.psse_dataframe["Type"] == 2].BUS_ID.to_list()
        regular_gens = [x for x in generator_buses if x not in self.wecBus_nums]
        print(regular_gens)
        pointer = 0
        for key, value in self.migrid_data.items():
            if key[:3] == 'gen':
                self._psse_adjust_gen(bus_num=regular_gens[pointer], p=value.iloc[0].gen_value)
                pointer+=1

    def compare(self):
        """
        Description:
        input: 
        output:
        """
        py = self.pypsa_dataframe.copy()
        py = py.reset_index(level=0, drop=True)
        py_final = py.rename(columns={'Pd':"P Load", 'Qd':"Q Load", 'v_mag_pu_set':'PU'})[["PU","P Load", "Q Load"]].copy()
        py_final = py_final.fillna(0)
        ps = self.psse_dataframe.copy()
        ps_final = ps[["PU", "P Load", "Q Load"]]
        ps_final = ps_final.fillna(0)
        return ps_final.compare(py_final, keep_equal=True, keep_shape=True).rename(columns={'self': 'pyPSA', 'other': 'PSSe'})
