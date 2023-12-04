import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

########## Feature Engineering ############

## Data Imputation

def impute(df: pd.DataFrame, cols: list, type: str) -> pd.DataFrame:
    """
    Impute mean/median/mode values in the specified columns in the fiven DatFrame.

    Parameters:
        df (DataFrame): The DataFrame to be imputed.
        cols (list): A list of column names to be imputed.
        type (str): The type of imputation to be performed. Can be 'mean', 'median', or 'mode'.

    Returns:
        DataFrame: The DataFrame with the specified columns imputed.
    """
    for each in cols:
        if type == 'mean':
            df[each] = df[each].fillna(df[each].mean())
        elif type == 'median':
            df[each] = df[each].fillna(df[each].median())
        elif type == 'mode':
            df[each] = df[each].fillna(df[each].mode()[0])
    return df
    

## Categorical Features
def one_hot_encode(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    One-hot encodes the specified columns in the given DataFrame.

    Parameters:
        df (DataFrame): The DataFrame to be one-hot encoded.
        cols (list): A list of column names to be one-hot encoded.

    Returns:
        DataFrame: The DataFrame with the specified columns one-hot encoded.
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df

## Outlier Detection
def z_score_outliers(df: pd.DataFrame, cols: list, threshold: float) -> pd.DataFrame:
    """
    Detects outliers in the specified columns in the given DataFrame using the Z-score method.
    For each column, it creates a new one with the outliers marked as True.
    
    Parameters:
        df (DataFrame): The DataFrame to be checked for outliers.
        cols (list): A list of column names to be checked for outliers.
        threshold (float): The threshold value for the Z-score method.

    Returns:
        DataFrame: The DataFrame with the outliers marked as True.
    """
    for each in cols:
        z = np.abs((df[each] - df[each].mean()) / df[each].std())
        df[f"{each}_outlier"] = np.where(z > threshold, True, False)
    return df

def modified_z_score_outliers(df: pd.DataFrame, cols: list, threshold: float) -> pd.DataFrame:
    """
    Detects outliers in the specified columns in the given DataFrame using the modified Z-score method.
    For each column, it creates a new one with the outliers marked as True.
    
    Parameters:
        df (DataFrame): The DataFrame to be checked for outliers.
        cols (list): A list of column names to be checked for outliers.
        threshold (float): The threshold value for the modified Z-score method.

    Returns:
        DataFrame: The DataFrame with the outliers marked as True.
    """
    for each in cols:
        median = np.median(df[each])
        median_absolute_deviation = median_abs_deviation(df[each])
        median_z_score = 0.6745 * (df[each] - median) / median_absolute_deviation
        df[f"{each}_outlier"] = np.where(np.abs(median_z_score) > threshold, True, False)
    return df

def tukeys_fences(df: pd.DataFrame, cols: list, threshold: float) -> pd.DataFrame:
    """
    Detects outliers in the specified columns in the given DataFrame using the Tukey's fences method.
    For each column, it creates a new one with the outliers marked as True.
    
    Parameters:
        df (DataFrame): The DataFrame to be checked for outliers.
        cols (list): A list of column names to be checked for outliers.
        threshold (float): The threshold value for the Tukey's fences method.

    Returns:
        DataFrame: The DataFrame with the outliers marked as True.
    """
    for each in cols:
        q1 = df[each].quantile(0.25)
        q3 = df[each].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        df[f"{each}_outlier"] = np.where((df[each] < lower_bound) | (df[each] > upper_bound), True, False)
    return df

def k_nearest_neighbors(df: pd.DataFrame, cols: list, k: int, threshold: float) -> pd.DataFrame:
    """
    Detects outliers in the specified columns in the given DataFrame using the K-nearest neighbors method.
    For each column, it creates a new one with the outliers marked as True.
    
    Parameters:
        df (DataFrame): The DataFrame to be checked for outliers.
        cols (list): A list of column names to be checked for outliers.
        k (int): The number of nearest neighbors to consider.

    Returns:
        DataFrame: The DataFrame with the outliers marked as True.
    """
    for each in cols:
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(df[each].values.reshape(-1, 1))
        distances, _ = neigh.kneighbors(df[each].values.reshape(-1, 1))
        normalized_distances = distances / distances.sum(axis=1, keepdims=True)
        df[f"{each}_outlier"] = normalized_distances[:, -1] > threshold
    return df

def local_outlier_factor(df: pd.DataFrame, cols: list, contamination=0.1) -> pd.DataFrame:
    """
    Detects outliers in the specified columns in the given DataFrame using the Local Outlier Factor method.
    For each column, it creates a new one with the outliers marked as True.

    Parameters:
        df (DataFrame): The DataFrame to be checked for outliers.
        cols (list): A list of column names to be checked for outliers.
        contamination (float): The contamination parameter for the Local Outlier Factor method.

    Returns:
        DataFrame: The DataFrame with the outliers marked as True.
    """
    for col in cols:
        lof = LocalOutlierFactor(contamination=contamination)
        outliers = lof.fit_predict(df[col].values.reshape(-1, 1))
        df[f"{col}_outlier"] = outliers == -1
    return df

def capping_outliers_by_quantiles(df: pd.DataFrame, cols: list, upper_threshold: float, lower_threshold: float) -> pd.DataFrame:
    """
    Capping outliers in the specified columns in the given DataFrame using the quantiles method.
    For each column, it creates a new one with the outliers marked as True.

    Parameters:
        df (DataFrame): The DataFrame to be checked for outliers.
        cols (list): A list of column names to be checked for outliers.
        upper_threshold (float): The upper threshold value for the quantiles method.
        lower_threshold (float): The lower threshold value for the quantiles method.

    Returns:
        DataFrame: The DataFrame with the outliers marked as True.
    """
    for each in cols:
        upper_lim = df[each].quantile(upper_threshold)
        lower_lim = df[each].quantile(lower_threshold)
        df[f"{each}_outlier"] = np.where((df[each] > upper_lim) | (df[each] < lower_lim), True, False)
    return df


## Feature Scaling

# Involves transforming the numerical features of a dataset to a common scale.

def z_score_scaling(df: pd.DataFrame, cols: list, drop_original: bool = True) -> pd.DataFrame:
    """
    Standardizes the specified columns in the given DataFrame using the Z-score method.

    z = (x - mean) / std
    
    Thus, each feature will be scaled to have a mean of 0 and a standard deviation of 1.

    Parameters:
        df (DataFrame): The DataFrame to be standardized.
        cols (list): A list of column names to be standardized.
        drop_original (bool, optional): Whether to drop the original columns. Defaults to True.

    Returns:
        DataFrame: The DataFrame with the specified columns standardized.
    """
    for each in cols:
        scaler = StandardScaler()
        df[f"{each}_scaled"] = scaler.fit_transform(df[each].values.reshape(-1, 1))
        if drop_original:
            df.drop(each, axis=1, inplace=True)
    return df

def min_max_scaling(df: pd.DataFrame, cols: list, feature_range: tuple = (0, 1), drop_original: bool = True) -> pd.DataFrame:
    """
    Scales the specified columns in the given DataFrame using the Min-Max method.

    x = (x - min) / (max - min)
    
    Thus, each feature will be scaled to have a minimum value of 0 and a maximum value of 1.

    Parameters:
        df (DataFrame): The DataFrame to be standardized.
        cols (list): A list of column names to be standardized.
        feature_range (tuple, optional): The range of the feature values. Defaults to (0, 1).
        drop_original (bool, optional): Whether to drop the original columns. Defaults to True.

    Returns:
        DataFrame: The DataFrame with the specified columns standardized.
    """
    for each in cols:
        scaler = MinMaxScaler(feature_range=feature_range)
        df[f"{each}_scaled"] = scaler.fit_transform(df[each].values.reshape(-1, 1))
        if drop_original:
            df.drop(each, axis=1, inplace=True)        
    return df

def max_abs_scaling(df: pd.DataFrame, cols: list, drop_original: bool = True) -> pd.DataFrame:
    """
    Scales the specified columns in the given DataFrame using the Max-Abs method.

    x = x / max
    
    Thus, each feature will be scaled to have a maximum value of 1.

    Parameters:
        df (DataFrame): The DataFrame to be standardized.
        cols (list): A list of column names to be standardized.
        drop_original (bool, optional): Whether to drop the original columns. Defaults to True.

    Returns:
        DataFrame: The DataFrame with the specified columns standardized.
    """
    for each in cols:
        scaler = MaxAbsScaler()
        df[f"{each}_scaled"] = scaler.fit_transform(df[each].values.reshape(-1, 1))
        if drop_original:
            df.drop(each, axis=1, inplace=True)
    return df