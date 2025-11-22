
# Cleans dataset, handles missing values, splits features/target

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .config import TARGET, TEST_SIZE, RANDOM_STATE

def preprocess_data(df):

    # 1. Drop ID column (useless)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # 2. Ensure target exists
    if TARGET not in df.columns:
        raise ValueError(f"Target '{TARGET}' not found in dataset.")

    # 3. Missing values → replace with mean
    df = df.fillna(df.mean(numeric_only=True))

    # 4. Split into X and y
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # 6. Scale features (improves KNN performance)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
