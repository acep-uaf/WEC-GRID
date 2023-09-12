import unittest
from unittest.mock import patch, Mock

from WEC_GRID.wec_grid import Wec_grid
from WEC_GRID.utilities.util import read_paths

class TestWecGrid(unittest.TestCase):

    @patch.object(Wec_grid, 'initalize_psse', return_value=None)
    def test_initialize_psse(self, mock_initialize_psse):

        # Read the paths using the read_paths function
        actual_PATHS = read_paths()

        # Create a mock case file for initialization
        mock_case = "WEC_GRID/models/grid_models/IEEE_24_bus.RAW"

        # Create an instance of your class
        grid_instance = Wec_grid(mock_case)
            
        grid_instance.initalize_psse(solver="fnsl")

        # Add your assertions here to verify the behavior
        # Example:
        # self.assertTrue(some_condition)

if __name__ == "__main__":
    unittest.main()
