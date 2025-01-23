"""
WEC-GRID source code
Author: Alexander Barajas-Ritchie
Email: barajale@oreogonstate.edu

core.py
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
from WecGrid.cec import cec_class
from WecGrid.wec import wec_class
from WecGrid.utilities.util import dbQuery, read_paths
from WecGrid.database_handler.connection_class import DB_PATH
from WecGrid.pyPSA import pyPSAWrapper
from WecGrid.PSSe import PSSeWrapper
from WecGrid.viz import PSSEVisualizer


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
        self.case_file = case  # TODO: need to verify file exist
        self.case_file_name = os.path.basename(case)
        self.psseObj = None
        self.pypsaObj = None
        # self.dbObj = None
        self.wecObj_list = []  # list of the WEC Model objects for the simulations
        self.cecObj_list = []  # list of the WEC Model objects for the simulations
        # these list could probably be combined? ^^^

    def initialize_psse(self, solver_args=None):
        """
        Initializes the PSSE solver.

        Args:
            solver_args (dict): Optional arguments for the PSSE initialization.
        """
        solver_args = solver_args or {}  # Use empty dict if no args are provided
        self.psseObj = PSSeWrapper(self.case_file, self)
        self.psseObj.initialize(solver_args)
        print(
            f"PSSE initialized with case file: {self.case_file_name}."
        )  # TODO: this shoould be a check not a print

    def initialize_pypsa(self, solver_args=None):
        """
        Initializes the PyPSA solver.

        Args:
            solver_args (dict): Optional arguments for the PyPSA initialization.
        """
        solver_args = solver_args or {}  # Use empty dict if no args are provided
        self.pypsaObj = pyPSAWrapper(self.case_file, self)
        self.pypsaObj.initialize(solver_args)
        print(
            f"PyPSA initialized with case file: {self.case_file_name}."
        )  # TODO: this shoould be a check not a print

    def create_wec(self, ID, model, bus_location, run_sim=True):
        self.wecObj_list.append(wec_class.WEC(ID, model, bus_location, run_sim))

        self.psseObj.dataframe.loc[
            self.psseObj.dataframe["BUS_ID"] == bus_location, "Type"
        ] = 4  # This updated the Grid Model for the grid to know that the bus now has a WEC/CEC on it.
        self.psseObj.wecObj_list = self.wecObj_list
        # TODO: need to update pyPSA obj too

    def create_cec(self, ID, model, bus_location, run_sim=True):
        self.cecObj_list.append(cec_class.CEC(ID, model, bus_location, run_sim))
        self.psseObj.dataframe.loc[
            self.psseObj.dataframe["BUS_ID"] == bus_location, "Type"
        ] = 4  # This updated the Grid Model for the grid to know that the bus now has a WEC/CEC on it.
        # TODO: need to update pyPSA obj too
