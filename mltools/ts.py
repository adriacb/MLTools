import pandas as pd

def create_ts_features(df: pd.DataFrame, column:str) -> pd.DataFrame:
    df['year'] = df[column].dt.year
    df['month'] = df[column].dt.month
    df['day'] = df[column].dt.day
    df['day_of_week'] = df[column].dt.dayofweek
    df['is_weekend'] = df[column].dt.dayofweek >= 5
    df['quarter'] = df[column].dt.quarter
    df['week_of_year'] = df[column].dt.isocalendar().week
    return df