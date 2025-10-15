import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

DATAFILE = "syntheticnetworkdata.csv"
TARGETCOL = "label"
TESTSIZERATIO = 0.25
RANDOMSTATE = 42

def load_and_prepare_data():
    """Loads CSV, processes features (X1, X2), scales and splits them."""

    possible_paths = [f"content/{DATAFILE}", DATAFILE, f"./{DATAFILE}"]
    file_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if not file_path:
        print(f"File {DATAFILE} not found. Please upload it first.")
        return None, None, None, None, None

    df = pd.read_csv(file_path)
    print(f"Loaded data from {file_path}")
    print(f"Total rows: {len(df)}, Columns: {list(df.columns)}")

    # Feature derivation
    if not all(col in df.columns for col in ["amplifiertargetdb", "amplifiergaindb", "spanlosstargetdb", "spanlossdb"]):
        raise ValueError("Required columns for feature derivation are missing.")

    df["X1"] = df["amplifiertargetdb"] - df["amplifiergaindb"]
    df["X2"] = df["spanlosstargetdb"] - df["spanlossdb"]

    X = df[["X1", "X2"]].values
    Y = df[TARGETCOL].values

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features scaled successfully.")

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=TESTSIZERATIO, random_state=RANDOMSTATE, stratify=Y
    )

    print(f"Samples: {len(X)}, Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, Y_train, Y_test, scaler
