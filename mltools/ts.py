import pandas as pd
from pandas.tseries.offsets import Day, Week, MonthEnd, YearEnd

from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt

from typing import List, Union, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

def create_ts_features(df: pd.DataFrame, column:str) -> pd.DataFrame:
    df['year'] = df[column].dt.year
    df['month'] = df[column].dt.month
    df['day'] = df[column].dt.day
    df['day_of_week'] = df[column].dt.dayofweek
    df['is_weekend'] = df[column].dt.dayofweek >= 5
    df['quarter'] = df[column].dt.quarter
    df['week_of_year'] = df[column].dt.isocalendar().week
    return df

def check_missing_dates(
        df:pd.DataFrame, datetime_col:str="Date", freq:str="D"
                        ) -> List[Union[str, pd.Timestamp]]:
    """
    Check missing dates in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the datetime column.
    datetime_col : str, optional
        The column name of the datetime column, by default "Date".
    freq : str, optional
        The frequency of the datetime column, by default "D".

    Returns
    -------
    List[Union[str, pd.Timestamp]]
        List of missing dates.
    """

    freq_dict = {
        "D": Day(),
        "W": Week(),
        "M": MonthEnd(),
        "Y": YearEnd()
    }

    df[datetime_col] = pd.to_datetime(df[datetime_col])

    df = df.sort_values(by=datetime_col)

    full_date_range = pd.date_range(start=df[datetime_col].min(), end=df[datetime_col].max(), freq=freq_dict.get(freq))

    missing_dates = full_date_range.difference(df[datetime_col])
    
    return list(missing_dates)

def fill_missing_dates(df: pd.DataFrame, datetime_col: str = "Date", freq: str = "D") -> pd.DataFrame:
    """
    Fill missing dates in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the datetime column.
    datetime_col : str, optional
        The column name of the datetime column, by default "Date".
    freq : str, optional
        The frequency of the datetime column, by default "D".

    Returns
    -------
    pd.DataFrame
        DataFrame with filled missing dates.
    """
    # Define the frequency dictionary
    freq_dict = {
        "D": Day(),
        "W": Week(),
        "M": MonthEnd(),
        "Y": YearEnd()
    }

    # Check if the datetime column is in the index
    if df.index.name == datetime_col:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq_dict.get(freq))
        df = df.reindex(full_date_range).reset_index()
        df = df.rename(columns={"index": datetime_col})
    else:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.sort_values(by=datetime_col)
        full_date_range = pd.date_range(start=df[datetime_col].min(), end=df[datetime_col].max(), freq=freq_dict.get(freq))
        df = df.set_index(datetime_col).reindex(full_date_range).reset_index()
        df = df.rename(columns={"index": datetime_col})

    return df

def check_duplicates(df:pd.DataFrame, datetime_col:str="Date") -> pd.DataFrame:
    """
    Check for duplicates in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the datetime column.
    datetime_col : str, optional
        The column name of the datetime column, by default "Date".

    Returns
    -------
    pd.DataFrame
        DataFrame containing duplicates.
    """

    df[datetime_col] = pd.to_datetime(df[datetime_col])

    duplicates = df[df.duplicated(subset=[datetime_col], keep=False)]

    return duplicates

def plot_time_series(df: pd.DataFrame, datetime_col: str = "Date", y_cols: Union[str, List[str]] = None, subplots: bool = False) -> None:
    """
    Plot time series data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the datetime column and other columns to plot.
    datetime_col : str, optional
        The column name of the datetime column, by default "Date".
    y_cols : Union[str, List[str]], optional
        The column name(s) of the y-axis, by default None (plots all numeric columns except datetime_col).
    subplots : bool, optional
        Whether to create subplots for each time series, by default False (all in one plot).
    """

    df[datetime_col] = pd.to_datetime(df[datetime_col])

    if y_cols is None:
        y_cols = df.columns.difference([datetime_col])
    elif isinstance(y_cols, str):
        y_cols = [y_cols]

    if subplots:
        num_plots = len(y_cols)
        df.set_index('Date').plot(subplots=True, figsize=(12, 6))
        plt.show()
        
    else:
        plt.figure(figsize=(12, 6))

        for col in y_cols:
            plt.plot(df[datetime_col], df[col], label=col)

        plt.title("Time Series")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


def fill_nan_values(df: pd.DataFrame, method: str = "ffill", window:int=3, min_periods:int=1, center:bool=True) -> pd.DataFrame:
    """
    Fill NaN values in a DataFrame. (inspired in https://medium.com/@datasciencewizards/preprocessing-and-data-exploration-for-time-series-handling-missing-values-e5c507f6c71c)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing NaN values.
    method : str, optional
        The method to use for filling NaN values, by default "ffill".
    window : int, optional
        The rolling window size for rolling mean, by default 3.
    min_periods : int, optional
        The minimum number of periods for rolling mean, by default 1.

    Returns
    -------
    pd.DataFrame
        DataFrame with filled NaN values.
    """

    if method not in ["ffill", "bfill", "rolling_mean", "interpolate"]:
        raise ValueError("Invalid method. Choose 'ffill', 'bfill', 'rolling_mean', or 'interpolate'.")

    if method == "ffill":
        df = df.ffill()
    elif method == "bfill":
        df = df.bfill()
    elif method == "rolling_mean":
        df = df.rolling(window=window, min_periods=min_periods, center=center).mean()
    elif method == "interpolate":
        df = df.interpolate(method="linear")
    else:
        df = df.fillna(method)

    return df

def train_test_split(df: pd.DataFrame, datetime_col: str = None, test_size: float = 0.2, split_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train and test sets based on a specified datetime column and test_size proportion.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the time series data.
    datetime_col (str): The name of the datetime column in the DataFrame.
    test_size (float): The proportion of the data to include in the test set (default is 0.2, which means 20%).
    split_date (str): The date used for splitting the data into train and test sets. If None, split based on test_size.

    Returns:
    pd.DataFrame, pd.DataFrame: Two DataFrames for train and test sets.
    """
    if datetime_col is None:
        df.index = pd.to_datetime(df.index)
        datetime_col = df.index.name

    # Sort DataFrame by the datetime column
    df = df.sort_index()
    
    if split_date:
        # Split based on the specified date
        split_index = df[df[datetime_col] >= split_date].index[0]
    else:
        # Split based on test_size proportion
        split_index = int(len(df) * (1 - test_size))
    
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    
    return train, test

def plot_train_test_splits(train: pd.DataFrame, test: pd.DataFrame, columns: Union[str, List[str]]) -> None:
    """
    Plot train and test splits for specified columns of a time series dataset.

    Parameters:
    train (pd.DataFrame): DataFrame containing the training set.
    test (pd.DataFrame): DataFrame containing the testing set.
    columns_to_plot (list or str): List of column names or a single column name to plot.

    Returns:
    None
    """
    if isinstance(columns, str):
        columns = [columns]

    num_cols = len(columns)
    fig, axs = plt.subplots(num_cols, 1, figsize=(14, 7*num_cols))

    for i, col in enumerate(columns):
        axs[i].plot(train.index, train[col], label=f'Train Set ({col})', marker='o')
        axs[i].plot(test.index, test[col], label=f'Test Set ({col})', marker='o', linestyle='--')
        axs[i].set_title(f'Train-Test Split: {col}')
        axs[i].set_xlabel('Date')
        axs[i].set_ylabel('Value')
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()

def adfuller_test(series:pd.Series, sig:float=0.05, name:str='') -> None:
    """
    Perform ADF test to check if the series is stationary or not.

    Parameters
    ----------
    series : pd.Series
        The series to check.
    sig : float
        Significance level.
    name : str
        Name of the series.
    
    Returns
    -------
    None
    """
    res = adfuller(series, autolag='AIC')    
    p_value = round(res[1], 3) 
    if p_value <= sig:
        print(f" {name} : P-Value = {p_value} => Stationary. ")
    else:
        print(f" {name} : P-Value = {p_value} => Non-stationary.")

def seasonal_decompose_plot(series:pd.Series, freq:int=12) -> None:
    """
    Decompose the time series into trend, seasonal, and residual components.

    Parameters
    ----------
    series : pd.Series
        The time series data.
    freq : int
        The frequency of the time series.

    Returns
    -------
    None
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    decomposition = seasonal_decompose(series, model='additive', period=freq)
    decomposition.plot()
    plt.show()

def plot_acf_pacf(series:pd.Series, lags:int=40) -> None:
    """
    Plot the ACF and PACF for the time series data.

    Parameters
    ----------
    series : pd.Series
        The time series data.
    lags : int
        The number of lags to include in the plot.

    Returns
    -------
    None
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    plot_acf(series, lags=lags, ax=ax[0])
    plot_pacf(series, lags=lags, ax=ax[1])

    plt.show()


def stat_anomaly_detection(df:pd.DataFrame, datetime_col:str=None, value_col:str="Value", method="z-score", threshold:float=3.5) -> pd.DataFrame:
    """
    Detect anomalies in a time series using statistical methods.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data.
    datetime_col : str, optional
        The name of the datetime column in the DataFrame.
    value_col : str, optional
        The name of the value column in the DataFrame.
    method : str, optional
        The method to use for anomaly detection, by default "z-score".
    threshold : float, optional
        The threshold value for detecting anomalies, by default 3.5.

    Returns
    -------
    pd.DataFrame
        DataFrame with anomaly detection results.
    """
    if datetime_col is None:
        df.index = pd.to_datetime(df.index)
        datetime_col = df.index.name

    if method not in ["z-score", "iqr", "MAD"]:
        raise ValueError("Invalid method. Choose 'z-score', 'iqr', or 'MAD'.")
    
    if method == "z-score":
        df['zscore'] = (df[value_col] - df[value_col].mean()) / df[value_col].std()
        anomalies = df[(df['zscore'] > threshold) | (df['zscore'] < -threshold)]
    
    elif method == "iqr":
        q1 = df[value_col].quantile(0.25)
        q3 = df[value_col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        anomalies = df[(df[value_col] < lower_bound) | (df[value_col] > upper_bound)]
    
    elif method == "MAD":
        median = df[value_col].median()
        mad = df[value_col].mad()
        anomalies = df[abs(df[value_col] - median) / mad > threshold]
    
    return anomalies

def ml_anomaly_detection(df: pd.DataFrame, datetime_col: str=None, value_col: str="Value", method: str="isolation_forest", threshold: float=0.05) -> tuple:
    """
    Detect anomalies in a time series using machine learning methods.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series data.
    datetime_col : str, optional
        The name of the datetime column in the DataFrame.
    value_col : str, optional
        The name of the value column in the DataFrame.
    method : str, optional
        The method to use for anomaly detection, by default "isolation_forest".
    threshold : float, optional
        The threshold value for detecting anomalies, by default 0.05.

    Returns
    -------
    tuple
        Tuple containing anomaly detection results DataFrame and anomaly detection model.
    """
    if datetime_col is None:
        df.index = pd.to_datetime(df.index)
        datetime_col = df.index.name

    if method not in ["isolation_forest", "lof", "svm"]:
        raise ValueError("Invalid method. Choose 'isolation_forest', 'lof', or 'svm'.")
    
    # Initialize model variable
    model = None

    if method == "isolation_forest":
        model = IsolationForest(contamination=threshold)
    
    elif method == "lof":
        model = LocalOutlierFactor(contamination=threshold)
    
    elif method == "svm":
        model = OneClassSVM(nu=threshold)
    
    if model is None:
        raise ValueError("Failed to initialize anomaly detection model.")

    # Fit the model and predict anomalies
    df[f'anomaly.{method}'] = model.fit_predict(df[value_col].values.reshape(-1, 1))

    # Filter anomalies
    anomalies = df[df[f'anomaly.{method}'] == -1]

    # Return anomalies DataFrame and model
    return anomalies

def plot_anomalies(df: pd.DataFrame, anomalies: pd.DataFrame,
                   datetime_col: str=None, value_col: str=None, anomaly_col: str=None, threshold: float=0.05) -> None:
    """
    Plot anomalies in a time series DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the time series data.
    - anomalies (pd.DataFrame): DataFrame containing the anomalies detected.
    - datetime_col (str): Name of the datetime column in df.
    - value_col (str): Name of the column containing the values to plot.
    - anomaly_col (str): Name of the column indicating anomalies (binary).
    - threshold (float): Threshold value above which points are considered anomalies.
    """
    
    if datetime_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            datetime_col = df.index.name
        else:
            raise ValueError("datetime_col must be specified or the DataFrame index must be a DatetimeIndex.")
    
    if value_col is None:
        raise ValueError("value_col must be specified.")
    
    if anomaly_col is None:
        raise ValueError("anomaly_col must be specified.")
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Plotting normal points
    plt.plot(df.index, df[value_col], label='Data')
    
    # Highlighting anomalies
    plt.scatter(anomalies.index, anomalies[value_col], color='red', label='Anomalies')
    
    plt.title('Time Series with Anomalies')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()