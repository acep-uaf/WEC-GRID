import os
import sqlite3
from contextlib import contextmanager


CURR_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = CURR_DIR + "/WEC-GRID.db"


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
