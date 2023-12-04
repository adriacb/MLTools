import os

import pickle
import uuid
import time

import umap

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom
from rdkit.Chem import PandasTools
from rdkit.Chem import Draw 
#from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs

from typing import Tuple, Union, List

from sklearn.utils import resample

import numpy as np  # numpy for math
import pandas as pd      # for dataframes and csv files
import matplotlib.pyplot as plt  # for plotting
from matplotlib import animation  # animate 3D plots
from mpl_toolkits.mplot3d import Axes3D  # 3D plots

# Scikit learn
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn import preprocessing

from tensorflow.keras import models

# TensorFlow and Keras
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow import keras
from keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization


from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, Descriptors3D, Draw, rdMolDescriptors, Draw, PandasTools

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def plot_accuracy():
    pass



def resample(df: pd.DataFrame, label: str, over: bool=True, under: bool=False) -> pd.DataFrame:
    """
    Description:
    -----------
        Resample a dataframe to balance the classes.
        It checks if there is a class imbalance and if so, it resamples the dataframe.
        If there is no imbalance, it returns the original dataframe.
        If over is True, uses SMOTE to oversample the dataframe.
        If under is True, uses RandomUnderSampler to undersample the dataframe.
        It returns the resampled dataframe.
    Parameters:
    -----------
        df: pd.DataFrame
            The dataframe to be resampled.
        label: str
            The label of the class to be resampled.
        over: bool
            If True, it resamples the dataframe to have more samples of the class.
        under: bool
            If True, it resamples the dataframe to have less samples of the class.
    Returns:
    --------
        pd.DataFrame
            The resampled dataframe.
    """

    assert over or under, "You must choose to oversample or undersample the dataframe."
    assert label in df.columns, "The label {} is not in the dataframe columns.".format(label)

    if over:
        sm = SMOTE(random_state=42, n_jobs=-1)
        df_resampled = pd.DataFrame(sm.fit_resample(df.values, df[label].values))
        df_resampled.columns = df.columns
        return df_resampled
    
    elif under:
        rus = RandomUnderSampler(random_state=42, n_jobs=-1)
        df_resampled = pd.DataFrame(rus.fit_resample(df.values, df[label].values))
        df_resampled.columns = df.columns
        return df_resampled
    
    else:
        return df





def prepareInput(df: pd.DataFrame):
    """
    Description:
    -----------
        Creates the input for the neural network.
        It returns the input.
    Parameters:
    -----------
        df: pd.DataFrame
            The dataframe to be used.
    Returns:
    --------
        np.array
            The input for the neural network.
    """


    inputs = {}
    for name, column in df.items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32
        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

    # deal with numeric features
    numeric_inputs = {name: input for name, input in inputs.items() if input.dtype == tf.float32}

    x = tf.keras.layers.Concatenate()(list(numeric_inputs.values()))
    norm = preprocessing.Normalization()
    #print(np.array(df[numeric_inputs.keys()]).astype('float32'))

    norm.adapt(np.array(df[numeric_inputs.keys()]).astype('float32'))
    all_numeric_inputs = norm(x)
    preprocessed_inputs = [all_numeric_inputs]

    # deal with categorical features
    preprocessed_inputs_cat = keras.layers.Concatenate()(preprocessed_inputs)
    preprocessing_layer = tf.keras.Model(inputs, preprocessed_inputs_cat, name="ProcessData")

    items_features_dict = {name: np.array(value) for name, value in df.items()}
    items_features_fitted = preprocessing_layer(items_features_dict)

    return items_features_fitted


