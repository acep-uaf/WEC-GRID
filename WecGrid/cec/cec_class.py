"""
CEC Class Module file
"""

import os

import pandas as pd
import matlab.engine
import sqlite3

from ..utilities.util import dbQuery, read_paths
from ..database_handler.connection_class import DB_PATH, SQLiteConnection

PATHS = read_paths()


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

    def clean_df(self):

        # Define the intervals
        intervals = [(150, 450), (451, 750), (751, 1050)]
        averages = []

        for start, end in intervals:
            # Get the average pg value for each interval
            avg = self.dataframe.loc[
                (self.dataframe["time"] >= start) & (self.dataframe["time"] <= end),
                "pg",
            ].mean()
            averages.append(avg)

        # Combine intervals and averages for the result
        result = pd.DataFrame({"time": [end for _, end in intervals], "pg": averages})

        result["vs"] = 1.1  # this shouldn't be hardcoded

        self.dataframe = result

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

        self.clean_df()

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

        self.clean_df()
