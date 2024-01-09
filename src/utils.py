import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from pandas.api.types import CategoricalDtype

from sklearn.model_selection import KFold, StratifiedKFold

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


def plot_cross_val(n_splits: int, splitter_func, df: pd.DataFrame, target_column: str, title_text: str) -> None:
    """Function to plot the cross-validation of various sklearn splitter objects."""
    split = 1
    plot_data = []
    
    if splitter_func == StratifiedKFold:
        y = df[target_column]
        X = df.drop(columns=[target_column])
        for train_index, valid_index in splitter_func(n_splits=n_splits).split(X, y):
            plot_data.append([train_index, 'Train', f'{split}'])
            plot_data.append([valid_index, 'Test', f'{split}'])
            split += 1
    else:
        for train_index, valid_index in splitter_func(n_splits=n_splits).split(df):
            plot_data.append([train_index, 'Train', f'{split}'])
            plot_data.append([valid_index, 'Test', f'{split}'])
            split += 1

    plot_df = pd.DataFrame(plot_data, columns=['Index', 'Dataset', 'Split']).explode('Index')

    plt.figure(figsize=(10, 6))
    handles = []
    labels = []
    
    train_handles = plt.scatter([], [], label='Train', color='blue', alpha=0.7)
    test_handles = plt.scatter([], [], label='Test', color='goldenrod', alpha=0.7)
    
    for split_num, group in plot_df.groupby('Split'):
        train_indices = group['Index'][group['Dataset'] == 'Train']
        test_indices = group['Index'][group['Dataset'] == 'Test']

        plt.scatter(train_indices, [split_num] * len(train_indices), color='blue', alpha=0.7)
        plt.scatter(test_indices, [split_num] * len(test_indices), color='goldenrod', alpha=0.7)
    
    handles.extend([train_handles, test_handles])
    labels.extend(['Train', 'Test'])

    plt.xlabel('Index')
    plt.ylabel('Split')
    plt.title(title_text)
    plt.legend(handles, labels)
    plt.grid(True)
    plt.show()