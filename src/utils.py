import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from pandas.api.types import CategoricalDtype


# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

# Mute warnings
warnings.filterwarnings('ignore')

def load_data():
    # Read data
    data_dir = Path("../data/")
    df_train = pd.read_csv(data_dir / "train.csv", index_col="Id")
    df_test = pd.read_csv(data_dir / "test.csv", index_col="Id")
    print(f"Train dataset has {df_train.shape[0]} rows and {df_train.shape[1]} columns.")
    print(f"Test dataset has {df_test.shape[0]} rows and {df_test.shape[1]} columns.")
    
    #all the data
    df = pd.concat([df_train, df_test])
    return df_train, df_test, df