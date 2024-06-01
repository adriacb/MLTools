import pandas as pd

def create_new_features(df: pd.DataFrame, column:str) -> pd.DataFrame:
    df['year'] = df.column.year
    df['month'] = df.column.month
    df['day'] = df.column.day
    df['day_of_week'] = df.column.dayofweek
    df['is_weekend'] = df.column.dayofweek >= 5
    df['quarter'] = df.column.quarter
    df['week_of_year'] = df.column.isocalendar().week
    return df