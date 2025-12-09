
# Loads CSV into pandas DataFrame

import pandas as pd
from .config import DATA_PATH
from .config import DROP_COLUMNS

def load_dataset(DATA_PATH):
    """
    Load CSV and drop unwanted columns if present.
    Returns pandas DataFrame.
    """
    df = pd.read_csv(DATA_PATH)
    for c in DROP_COLUMNS:
        if c in df.columns:
            df = df.drop(columns=[c])
    return df
