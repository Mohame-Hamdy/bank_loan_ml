
# Loads CSV into pandas DataFrame

import pandas as pd
from .config import DATA_PATH

def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    return df
