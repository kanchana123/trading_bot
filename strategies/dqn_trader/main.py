import os
import sys

# Get the project root directory (one level up from the current directory)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Add the project root to the Python path
sys.path.insert(0, project_root)

from base_models.angel_api import AngelAPI
from data_generator import DataGenerator

if __name__ == "__main__":
    # Initialize the AngelAPI
    api = AngelAPI()

    # Set instrument_symbol, and filepath
    instrument_symbol = "ICICIBANK-EQ"
    filepath = f'{os.path.abspath(".")}/base_models/token_symbol_list.txt'

    # Initialize the DataGenerator
    data_generator = DataGenerator(api, instrument_symbol, filepath)

    # Generate data and charts for 365 days
    data_generator.generate_data_and_charts(days=50)
