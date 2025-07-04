import pandas as pd
from sklearn.preprocessing import LabelEncoder

def add_temporal_features(df, date_col='appointment_day'):
    df[date_col] = pd.to_datetime(df[date_col])
    df['day_of_week'] = df[date_col].dt.day_name()
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    df['month'] = df[date_col].dt.month
    df['season'] = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    return df

def encode_categorical(df, cat_cols):
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders

def create_features(df):
    df = add_temporal_features(df)
    categorical_cols = ['gender', 'appointment_type', 'weather', 'day_of_week', 'season']
    df, encoders = encode_categorical(df, categorical_cols)
    return df, encoders
