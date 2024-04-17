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


# Initialize the PATHS dictionary
PATHS = read_paths()
CURR_DIR = os.path.dirname(os.path.abspath(__file__))


class Wec_grid:
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
        self.total_buses = 0
        self.load_profiles = {}
        # self.migrid_file_names = self.Migrid_file()
        # self.electrical_distance_dict = {}
        self.z_history = {}
        self.flow_data = {}
        self.psse = None
        self.pypsa = None

    def run_simulation(self, start, end, solver) -> bool:
        # Initial empty time_stamps list
        time_stamps = []

        # Check and get time_stamps from wec_list or cec_list
        if len(self.wec_list) != 0:
            time_stamps = self.wec_list[0].dataframe.time.to_list()
        elif len(self.cec_list) != 0:
            time_stamps = self.cec_list[0].dataframe.time.to_list()

        # Filter time_stamps based on start and end parameters
        time_stamps = [ts for ts in time_stamps if start <= ts <= end]

        # Run the simulation for each time stamp within the range
        for time_stamp in time_stamps:
            gen = {}
            load = {}
            # print("Simulating time stamp: {}".format(time_stamp))
            if len(self.wec_list) != 0 or len(self.cec_list) != 0:
                gen = self.gen_values(time_stamp)
            if len(self.load_profiles) != 0:
                load = self.load_values(time_stamp)
            self.psse.steady_state(gen, load, time_stamp, solver)
            # self.pypsa.run_simulation(gen, load, time_stamp, solver)
            print("-----------------------------------")
        return True

    def gen_values(self, t):
        gen = {}

        for idx, wec_obj in enumerate(self.wec_list):
            df = wec_obj.dataframe
            if t in df.time.values:
                pg = df.loc[df.time == t, "pg"].iloc[0]
                vs = df.loc[df.time == t, "vs"].iloc[0]
                qt = df.loc[df.time == t, "qt"].iloc[0]
                qb = df.loc[df.time == t, "qb"].iloc[0]
                gen[wec_obj.bus_location] = [pg, vs, qt, qb]

        for idx, cec_obj in enumerate(self.cec_list):
            df = cec_obj.dataframe
            if t in df.time.values:
                pg = df.loc[df.time == t, "pg"].iloc[0]
                vs = df.loc[df.time == t, "vs"].iloc[0]
                gen[cec_obj.bus_location] = [pg, vs]

        # adjust other generators values here?

        return gen

    def load_values(self, t):
        return self.load_profiles[t].to_dict(orient="index")

    def initalize_psse(self, solver):
        self.psse = PSSeWrapper(self.case_file)
        self.psse.initalize(solver)
        self.total_buses = len(self.psse.dataframe)

    def initalize_pypsa(self, solver):
        self.pypsa = pyPSAWrapper(self.case_file)
        self.pypsa.initalize(solver)
        self.total_buses = len(self.pypsa.dataframe)

    def add_wec(self, wec):
        """
        Description: Adds a WEC to the WEC list and adjusts the activate power of the machine data in PSSe
        input:
            wec: WEC object to be added to the WEC list (wec_class.WEC)
        output: None
        """
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
        self.wec_list.append(wec_class.WEC(ID, model, bus_location, run_sim))

        self.psse.dataframe.loc[
            self.psse.dataframe["BUS_ID"] == bus_location, "Type"
        ] = 4

    def create_cec(self, ID, model, bus_location, run_sim=True):
        self.cec_list.append(cec_class.CEC(ID, model, bus_location, run_sim))
        self.psse.dataframe.loc[
            self.psse.dataframe["BUS_ID"] == bus_location, "Type"
        ] = 4

    def run_WEC_Sim(self, wec_id, sim_config):
        for wec in self.wec_list:
            if wec.ID == wec_id:
                wec.WEC_Sim(sim_config)
                return True

    def run_CEC_Sim(self, cec_id, sim_config):
        for cec in self.cec_list:
            if cec.ID == cec_id:
                cec.CEC_Sim(sim_config)

    def generate_load_profiles(
        self, peak_values, curve_type, noise_type, noise_level, step_size, sim_length
    ):
        """
        Generate a dictionary of load profiles with timestamps as keys and DataFrames as values.
        """
        load_profiles = {}

        # Time normalization to a 24-hour cycle
        seconds_in_day = 24 * 60 * 60
        time_data = np.arange(0, sim_length, step_size)
        normalized_time = (time_data % seconds_in_day) / seconds_in_day

        # Standard deviation for the normal curve
        std_dev = 0.15

        for i, time_step in enumerate(time_data):
            df = pd.DataFrame(columns=["P", "Q"], index=peak_values.keys())
            for bus_id, peaks in peak_values.items():
                curve_data = []
                for idx, peak_value in enumerate(peaks):
                    if peak_value == 0:
                        curve = np.zeros(1)
                    else:
                        # Adjusting curve based on the type
                        if curve_type == "normal":
                            curve = np.exp(
                                -((normalized_time[i] - 0.5) ** 2) / (2 * std_dev**2)
                            )
                        elif curve_type == "summer":
                            curve = np.sin(normalized_time[i] * np.pi) ** 2
                        elif curve_type == "winter":
                            curve = np.cos(normalized_time[i] * np.pi) ** 2
                        else:
                            curve = np.ones(1)  # Default curve

                        # Normalize and add noise
                        curve = curve / curve.max() * peak_value
                        if noise_type == "uniform":
                            curve += np.random.uniform(-noise_level, noise_level, 1)
                        elif noise_type == "gaussian":
                            curve += np.random.normal(0, noise_level, 1)
                        curve = np.maximum(curve, 0)  # Ensure no negative values

                    curve_data.append(curve[0])

                df.loc[bus_id] = curve_data

            load_profiles[time_step] = df

        self.load_profiles = load_profiles

    def plot_bus_load_profiles(self, bus_id):
        """
        Plot the load profiles (P and Q values) for a specific bus.

        Parameters:
        bus_id: int
            The ID of the bus to plot.
        """

        # Extracting time series and P, Q values for the specified bus
        times = list(self.load_profiles.keys())
        P_values = [self.load_profiles[time].loc[bus_id, "P"] for time in times]
        Q_values = [self.load_profiles[time].loc[bus_id, "Q"] for time in times]

        # Plotting Active Power (P)
        plt.figure(figsize=(10, 5))
        plt.plot(times, P_values, label="P - Active Power", marker="o")
        plt.title(f"Active Power Load Profile for Bus {bus_id}")
        plt.xlabel("Time")
        plt.ylabel("Active Power (P)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plotting Reactive Power (Q)
        plt.figure(figsize=(10, 5))
        plt.plot(times, Q_values, label="Q - Reactive Power", marker="x")
        plt.title(f"Reactive Power Load Profile for Bus {bus_id}")
        plt.xlabel("Time")
        plt.ylabel("Reactive Power (Q)")
        plt.legend()
        plt.grid(True)
        plt.show()

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
        Description: Adjusts the active power of regular generators to match the values in the migrid_data dictionary.
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

    # def compare_v(self):
    #     """
    #     Description:
    #     input:
    #     output:
    #     """
    #     v_mag = pd.concat(
    #         [
    #             self.psse_dataframe[["PU"]],
    #             self.pypsa_dataframe[["v_mag_pu_set"]]
    #             .reset_index()
    #             .drop(columns=["Bus"]),
    #         ],
    #         axis=1,
    #     ).rename(
    #         columns={"PU": "PSSe voltage mag", "v_mag_pu_set": "pyPSA voltage mag"}
    #     )
    #     v_mag["abs diff"] = (
    #         v_mag["PSSe voltage mag"] - v_mag["pyPSA voltage mag"]
    #     ).abs()
    #     return v_mag

    # def compare_p(self):
    #     """
    #     Description:
    #     input:
    #     output:
    #     """
    #     p_load = pd.concat(
    #         [
    #             self.psse_dataframe[["P Load"]],
    #             self.pypsa_dataframe[["Pd"]].reset_index().drop(columns=["Bus"]),
    #         ],
    #         axis=1,
    #     ).rename(columns={"P Load": "PSSe P-Load", "Pd": "pyPSA P-Load"})
    #     p_load["abs diff"] = (p_load["PSSe P-Load"] - p_load["pyPSA P-Load"]).abs()
    #     return p_load

    # def compare_q(self):
    #     """
    #     Description:
    #     input:
    #     output:
    #     """
    #     q_load = pd.concat(
    #         [
    #             self.psse_dataframe[["Q Load"]],
    #             self.pypsa_dataframe[["Qd"]].reset_index().drop(columns=["Bus"]),
    #         ],
    #         axis=1,
    #     ).rename(columns={"Q Load": "PSSe Q-Load", "Qd": "pyPSA Q-Load"})
    #     q_load["abs diff"] = (q_load["PSSe Q-Load"] - q_load["pyPSA Q-Load"]).abs()
    #     return q_load

    # def compare(self):
    #     """
    #     Description:
    #     input:
    #     output:
    #     """
    #     py = self.pypsa_dataframe.copy()
    #     py = py.reset_index(level=0, drop=True)
    #     py_final = py.rename(
    #         columns={"Pd": "P Load", "Qd": "Q Load", "v_mag_pu_set": "PU"}
    #     )[["PU", "P Load", "Q Load"]].copy()
    #     py_final = py_final.fillna(0)
    #     ps = self.psse_dataframe.copy()
    #     ps_final = ps[["PU", "P Load", "Q Load"]]
    #     ps_final = ps_final.fillna(0)
    #     return ps_final.compare(py_final, keep_equal=True, keep_shape=True).rename(
    #         columns={"self": "pyPSA", "other": "PSSe"}
    #     )
