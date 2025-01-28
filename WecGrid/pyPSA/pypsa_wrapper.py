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
from pandas.tseries.offsets import DateOffset

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
        self.timestamp_start = datetime.now()

    def get_snapshots(self):
        
        return self.pypsa_object.snapshots

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
        self.pypsa_object.set_snapshots([self.timestamp_start])  # Set the initial snapshot
        
        

        #self.run_powerflow()
        self.pypsa_object.pf()

        self.dataframe = self.pypsa_object.df("Bus").copy() 
        self.format_df()
        #self.pypsa_object = pypsa_network
        #self.pypsa_history[self.timestamp_start] = self.dataframe # TODO: need to replace with snapshot 
        #self.pypsa_object_history[]] = self.pypsa_object
        #self.format_df()
        self.store_p_flow() # not storing init p flow i guess?
        print("pyPSA initialized")
        
    
    def add_wec(self, model, ID, from_bus, to_bus):
        
        name = "{}-{}".format(model, ID)
        
        # TODO if bus doesn't exist, add it
        
        # Step 1: Add a new bus for the WEC system
        if str(to_bus) not in self.pypsa_object.buses.index:
            from_bus_voltage = self.pypsa_object.buses.loc[str(from_bus), "v_nom"]
            self.pypsa_object.add(
                "Bus",
                str(to_bus),
                v_nom=from_bus_voltage,  # Match from_bus voltage
                control='PV',  # WEC operates as a PV generator
                carrier="AC",  # Standard AC grid connection
                v_mag_pu_set=1.0,  # Set voltage at 1.0 p.u.
                v_mag_pu_min=0.95,  # Min voltage in p.u.
                v_mag_pu_max=1.05,  # Max voltage in p.u.
                generator=name,
                p = 0.0,
                q = 0.0,
                v_ang = 0.0
            )
            print(f"Bus {to_bus} added successfully.")

        # self.pypsa_object.add(
        #     "Load",
        #     name=f"{name}_load",
        #     bus=str(to_bus),
        #     p_set=0.0,  # Active power demand (MW)
        #     q_set=0.0,  # Reactive power demand (MVar)
        # )
        
        self.pypsa_object.add(
            "Generator",
            name=name,
            bus=str(to_bus),
            control="PV",  # WEC operates as a PV generator
            p_nom_max=1.2,  # Maximum active power output (MW)
            p_set=0.0,
            p_nom_extendable=False,  # Fixed capacity, not expandable
            p_min_pu=0.0,  # No negative generation
        )

        
        self.pypsa_object.add("Line", "Line-{}".format(name), bus0=str(from_bus), bus1=str(to_bus), length=10, r=0.05, x=0.15)
        # TODO: need to fix this hardcoded line length, r, x values
        
        
        
        self.pypsa_object.pf()
        self.dataframe = self.pypsa_object.df("Bus").copy()
        self.format_df()
        self.store_p_flow()

        # Define a function to classify the generator types
    def classify_generator(self, control):
        if control == "Slack":
            return 3  # Type 3
        
        
        elif control == "PV":
            return 2
        
        elif control == "PQ":
            return 1
        
        elif control == "WEC":
            return 4
        else:
            return 0
        
    def format_df(self):
        # self.dataframe["type"] = self.dataframe["generator"].apply(self.classify_generator)
        # self.dataframe.reset_index(inplace=True)
        # self.dataframe.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in self.dataframe.columns]
        # # TODO: update WEC locations
        
        snapshots = self.get_snapshots() 
                # Specify the bus variables to collect
        variables = ['v_mag_pu_set', 'p', 'q', 'v_mag_pu', 'v_ang']

        # Loop over each snapshot
        for snapshot in snapshots:
            # Initialize a DataFrame to store data for this snapshot
            snapshot_data = pd.DataFrame()

            # Loop over the variables and collect the data
            for var in variables:
                if var in self.pypsa_object.buses_t:
                    # Add the variable data as a column in the DataFrame
                    snapshot_data[var] = self.pypsa_object.buses_t[var].loc[snapshot]
            
            # combine snapshot_data to have control, generator
            bus_static_data = self.pypsa_object.df("Bus")[["control", "generator"]]
            merged_data = pd.merge(snapshot_data, bus_static_data.copy(), on="Bus", how="left")
            
            merged_data["type"] = merged_data["control"].apply(self.classify_generator)
            
            # Add the snapshot data to the history dictionary
            
            # update WEC bus type
            
            for wec in self.WecGridCore.wecObj_list:
                name = "{}-{}".format(wec.model, wec.ID)
                merged_data.loc[merged_data["generator"] == str(name), "type"] = 4
                
    
            self.pypsa_history[snapshot] = merged_data.reset_index()
            
            # update WEC bus type
            
            
            
        
        
        
    def current_df(self):
        
        return self.pypsa_history[self.get_snapshots()[-1]]
        
    def ac_injection(self, snapshots=None):
        """
        Description: Simplified WEC AC injection for the PyPSA powerflow solver.
        
        Assumes one WEC generator per bus and runs power flow for all snapshots.
        
        Input:
            - snapshots (optional): A list of snapshot timestamps (e.g., DatetimeIndex).
            If not provided, the function will create snapshots based on the length
            of the `pg` data from the first WEC object.
        
        Output:
            - Power flow results stored in self.pypsa_history for all snapshots.
        """
        # Initialize empty dictionaries for generator active power (p_set)
        p_set_data = {}
        #q_set_data = {}

        # Determine snapshots if not provided
        if snapshots is None:
            # Use the initial timestamp and create a range of snapshots
            num_snapshots = len(self.WecGridCore.wecObj_list[0].dataframe["pg"])
            # snapshots = pd.date_range(start=self. = pd.date_range(start=timestamp_start, periods=num_snapshots, freq="5T"), periods=num_snapshots, freq="5T")
            
            snapshots = pd.date_range(
                        start=self.timestamp_start + DateOffset(minutes=5),  # Add 5 minutes
                        periods=num_snapshots,
                        freq="5T"  # 5-minute intervals
                    )

        # Iterate over all WEC objects to gather data
        for idx, wec_obj in enumerate(self.WecGridCore.wecObj_list):
            bus = wec_obj.bus_location

            # Ensure the generator exists for the bus
            generator_row = self.pypsa_object.generators.loc[
                self.pypsa_object.generators.bus == str(bus)
            ]
            
            
            generator_name = generator_row.index[0]  # Get the generator name (index)
            
            generator_name = "{}-{}".format(wec_obj.model, wec_obj.ID)

            # Fetch pg (active power) data and assume it's aligned with snapshots
            pg_data = wec_obj.dataframe["pg"] # Scale pg as needed
            #vs_data = wec_obj.dataframe["vs"] # Scale pg as needed

            # Store active power values for this generator
            p_set_data[generator_name] = pg_data.to_list()
            #q_set_data[generator_name] = [0.0] * len(pg_data)  # Assume no reactive power for now
            #vs_set_data[generator_name] = vs_data.to_list()

        # Set snapshots in the PyPSA network
        self.pypsa_object.set_snapshots(snapshots)


        # Create a DataFrame for p_set (active power) for all generators
        p_set_df = pd.DataFrame(p_set_data, index=snapshots)
        #v_set_df = pd.DataFrame(v_set_data, index=snapshots)
        #q_set_df =  pd.DataFrame(q_set_data, index=snapshots)
        
        

        # Assign active power values to the generators in the PyPSA network
        self.pypsa_object.generators_t.p_set = p_set_df
        #self.pypsa_object.generators_t.q_set = q_set_df
        
        # TODO: should try using p instead of p_set? 
        
        # should maybe set p_max_pu to 0.1 ? and mayeb q_set
        

        # Run power flow for all snapshots
        self.pypsa_object.pf()
        
        self.format_df() # also saves in history

        #snapshots = self.pypsa_object.snapshots

        # # Specify the bus variables to collect
        # variables = ['v_mag_pu_set', 'p', 'q', 'v_mag_pu', 'v_ang']

        # # Loop over each snapshot
        # for snapshot in snapshots:
        #     # Initialize a DataFrame to store data for this snapshot
        #     snapshot_data = pd.DataFrame()

        #     # Loop over the variables and collect the data
        #     for var in variables:
        #         if var in self.pypsa_object.buses_t:
        #             # Add the variable data as a column in the DataFrame
        #             snapshot_data[var] = self.pypsa_object.buses_t[var].loc[snapshot]

        #     # Add the snapshot data to the history dictionary
        #     self.pypsa_history[snapshot] = snapshot_data.copy().reset_index() 
            
        self.store_p_flow()

        #print(f"Power flow completed for all snapshots: {snapshots}")
        
    # def ac_injection(self, start, end, p=None, v=None, time=None):
    #     """
    #     Description: WEC AC injection for pypsa powerflow solver
    #     input:
    #         p - a vector of active power values in order of bus num
    #         v - a vector of voltage mag PU values in order of bus num
    #         time: (Int)
    #     output: no output but pypsa_dataframe is updated and so is pypsa_history
    #     """
    #     # Set default time if not provided
    #     if time is None:
    #         time = self.WecGridCore.wecObj_list[0].dataframe.time.to_list()
        
    #     # Create a time index if necessary
    #     #time_index = pd.to_datetime(time, unit="s")  # Convert `time` to datetime

    #     # Set snapshots in the PyPSA object
    #     #self.pypsa_object.set_snapshots(time_index)

    #     # Loop through all time snapshots and update p_set for each generator
    #     for t in time:
    #         if t >= start and t <= end:
    #             # Loop through WEC objects for generator updates
    #             for idx, wec_obj in enumerate(self.WecGridCore.wecObj_list):
    #                 bus = wec_obj.bus_location
    #                 generator_row = self.pypsa_object.generators.loc[
    #                     self.pypsa_object.generators.bus == str(bus)
    #                 ]

    #                 # Access the generator name (index)
    #                 generator_name = generator_row.index[0]

    #                 # Get pg and vs values for the current time
    #                 pg = float(wec_obj.dataframe.loc[wec_obj.dataframe.time == t].pg) * 50
    #                 vs = float(wec_obj.dataframe.loc[wec_obj.dataframe.time == t].vs)

    #                 print(f"Injecting at bus {bus}, time {t}: P = {pg}, V = {vs}")

    #                 # Update p_set for this snapshot
    #                 self.pypsa_object.generators.p_set.loc[generator_name] = pg

    #                 # Optionally update control or voltage setpoint if required
    #                 self.pypsa_object.generators.at[generator_name, "control"] = "PV"

    #             # Run the power flow for the current snapshot
    #             self.run_powerflow()

    #             # Store power flow results
    #             self.store_p_flow(t)

    #             # Format the dataframe for the current state
    #             self.format_df()

    #             # Store the formatted dataframe in the history
    #             self.pypsa_history[t] = self.dataframe

    #             print("Power flow completed for time:", t)
    #             print("===============")
    # def ac_injection(self, start, end, p=None, v=None, time=None):
    #     """
    #     Description: WEC AC injection for pypsa powerflow solver
    #     input:
    #         p - a vector of active power values in order of bus num
    #         v - a vector of voltage mag PU values in order of bus num
    #         time: (Int)
    #     output: no output but pypsa_dataframe is updated and so is pypsa_history
    #     """
    #     # TODO: There has to be a better way to do this.
    #     if time is None:
    #         time = self.WecGridCore.wecObj_list[0].dataframe.time.to_list()
    #     num_wecs = len(self.WecGridCore.wecObj_list)
    #     num_cecs = len(self.WecGridCore.cecObj_list)

    #     for t in time:
    #         if t >= start and t <= end:
    #             if num_wecs > 0:
    #                 for idx, wec_obj in enumerate(self.WecGridCore.wecObj_list):
                        
                        
                        
    #                     bus = wec_obj.bus_location
                        
    #                     generator_row = self.pypsa_object.generators.loc[self.pypsa_object.generators.bus == str(bus)]

    #                     # Access the generator name (index)
    #                     generator_name = generator_row.index[0]
                        
    #                     pg = float(wec_obj.dataframe.loc[wec_obj.dataframe.time == t].pg)
    #                     vs = float(wec_obj.dataframe.loc[wec_obj.dataframe.time == t].vs)
                        
    #                     print("Power injection at bus {} at time {}: P = {}, V = {}".format(bus, t, pg, vs))
                        
    #                     self.pypsa_object.generators.at[generator_name, "p_set"] = pg
    #                     self.pypsa_object.generators.at[generator_name, "control"] = "PV"
    #                     # self.pypsa_object.generators.loc[
    #                     #     self.pypsa_object.generators.bus == str(bus), "v_set"
    #                     # ] = vs
                        
    #                     # self.pypsa_object.generators_t.p[generator_name] = p
                        
                        
    #                     # self.pypsa_object.generators.loc[
    #                     #     self.pypsa_object.generators.bus == str(bus), "p_set"
    #                     # ] = pg
    #             if num_cecs > 0:
    #                 for idx, cec_obj in enumerate(self.WecGridCore.cecObj_list):
    #                     bus = cec_obj.bus_location
    #                     pg = cec_obj.dataframe.loc[cec_obj.dataframe.time == t].pg
    #                     vs = wec_obj.dataframe.loc[cec_obj.dataframe.time == t].vs

    #                     self.pypsa_object.generators.loc[
    #                         self.pypsa_object.generators.bus == str(bus), "v_set"
    #                     ] = vs
    #                     self.pypsa_object.generators.loc[
    #                         self.pypsa_object.generators.bus == str(bus), "p_set"
    #                     ] = pg

    #             self.run_powerflow()
    #             self.store_p_flow(t)
    #             self.format_df()
    #             self.pypsa_history[t] = self.dataframe
    #             print("===============")

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


    # def get_flow_data(self, t=None):
    #     """
    #     Retrieves power flow data for all branches in the PyPSA network.

    #     Inputs:
    #     - t (float): timestamp for which to retrieve the power flow data (optional)

    #     Outputs:
    #     - flow_data (dict): dictionary containing power flow data for all branches
    #     """
    #     if t is None:
    #         flow_data = {}

    #         try:
    #             # Iterate over all lines in the network
    #             for line in self.pypsa_object.lines.index:
    #                 source = self.pypsa_object.lines.loc[line, "bus0"]
    #                 target = self.pypsa_object.lines.loc[line, "bus1"]
                    
    #                 # Assuming power flow data is stored/calculated in network.lines as 'p0'
    #                 p_flow = self.pypsa_object.lines_t.p0.loc[:, line].iloc[-1]

    #                 # Create a dictionary for this branch's flow data
    #                 edge_data = {
    #                     "source": source if p_flow >= 0 else target,
    #                     "target": target if p_flow >= 0 else source,
    #                     "p_flow": p_flow,
    #                 }

    #                 # Use a tuple (source, target) as a unique identifier for each edge
    #                 edge_identifier = (edge_data["source"], edge_data["target"])
    #                 flow_data[edge_identifier] = edge_data["p_flow"]

    #         except Exception as e:
    #             print(f"Error fetching data: {e}")

    #         return flow_data

    #     else:
    #         return self.flow_data.get(t, {})
        
        
    def store_p_flow(self):
        """
        Loops over all snapshots in the pypsa_object and stores the p0 power flow values
        for each snapshot in a dictionary.

        Output:
            Updates self.flow_data with active power flow values (p0) for each snapshot.
        """
        # Ensure flow_data dictionary exists
        if not hasattr(self, "flow_data"):
            self.flow_data = {}

        # Extract all valid snapshots
        #snapshots = [key for key in self.pypsa_history.keys() if key != -1]
        
        snapshots = self.get_snapshots()
        
        

        # Loop over snapshots
        for t in snapshots:
            # Create an empty dictionary for this particular timestamp
            p_flow_dict = {}

            try:
                # Iterate over all lines in the network
                for line in self.pypsa_object.lines.index:
                    # Get source and target buses
                    source = self.pypsa_object.lines.loc[line, "bus0"]
                    target = self.pypsa_object.lines.loc[line, "bus1"]

                    # Get the power flow value for p0 at the current snapshot
                    try:
                        p_flow = self.pypsa_object.lines_t["p0"].loc[t, line]
                    except KeyError:
                        print(f"Line {line} not found in lines_t.p0 for snapshot {t}.")
                        continue
                    except IndexError:
                        print(f"No power flow data available for line {line} at snapshot {t}.")
                        continue

                    # Store the power flow in the dictionary
                    p_flow_dict[(source, target)] = p_flow

                for transformer in self.pypsa_object.transformers.index:
                    
                    source = self.pypsa_object.transformers.loc[transformer, "bus0"]
                    target = self.pypsa_object.transformers.loc[transformer, "bus1"]
                    
                    try:
                        p_flow = self.pypsa_object.transformers_t["p0"].loc[t, transformer]
                    except KeyError:
                        print(f"Transformer {transformer} not found in t.p0 for snapshot {t}.")
                        continue
                    except IndexError:
                        print(f"No power flow data available for transformer {transformer} at snapshot {t}.")
                        continue
                    p_flow_dict[(source, target)] = p_flow
                # Store the power flow data for this snapshot in the flow_data dictionary
                self.flow_data[t] = p_flow_dict

            except Exception as e:
                print(f"Error storing power flow data for snapshot {t}: {e}")
        
            
    def plot_bus(self, bus_num, arg_1="p", arg_2='q'):
        """
        Description: This function plots the activate and reactive power for a given bus
        input:
            bus_num: the bus number we wanna viz (Int)
            time: a list with start and end time (list of Ints)
        output:
            matplotlib chart
        """
        visualizer = PyPSAVisualizer(pypsa_obj=self)
        visualizer.plot_bus(bus_num, arg_1, arg_2)
        
    def bus_history(self, bus_num):
        """
        Description: this function grab all the data associated with a bus through the simulation
        input:
            bus_num: bus number (int)
        output:
            bus_dataframe: a pandas dateframe of the history
        """
        # maybe I should add a filering parameter?

        bus_dataframe = pd.DataFrame()
        for time, df in self.pypsa_history.items():
            temp = pd.DataFrame(df.loc[df["Bus"] == str(bus_num)])
            temp.insert(0, "time", time)
            bus_dataframe = bus_dataframe.append(temp)
        return bus_dataframe