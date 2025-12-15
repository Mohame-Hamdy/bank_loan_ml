
# Cleans dataset, handles missing values, splits features/target


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .config import CLASS_TARGET, REG_TARGET, TEST_SIZE, RANDOM_STATE

def fill_numeric_mean(df):
    """
    Return a copy of df where numeric columns' missing values are replaced with column mean.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].mean())
    return df

def prepare_classification(df):
    """
    Prepare features and target for bank loan approval task.
    Returns: X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    if CLASS_TARGET not in df.columns:
        raise KeyError(f"{CLASS_TARGET} not found in dataframe")
    df = fill_numeric_mean(df)
    X = df.drop(columns=[CLASS_TARGET])
    y = df[CLASS_TARGET].astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X.columns)

def prepare_regression(df):
    """
    Prepare features and target for regression task .
    Returns: X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    if REG_TARGET not in df.columns:
        raise KeyError(f"{REG_TARGET} not found in dataframe")
    df = fill_numeric_mean(df)
    X = df.drop(columns=[REG_TARGET])
    y = df[REG_TARGET].astype(float).values
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X.columns)