import os
import sqlite3
import json
import pandas as pd


from WEC_GRID.database_handler.connection_class import DB_PATH

from WEC_GRID.database_handler.connection_class import SQLiteConnection

CURR_DIR = os.path.dirname(os.path.abspath(__file__))


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
