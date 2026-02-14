"""
Data Preparation Module
=======================

Handles:
- Dataset loading via ucimlrepo (default path)
- Uploaded CSV preparation (user-supplied test data path)
- Multi-level target engineering (5 performance classes)
- Categorical feature encoding (OneHotEncoder)
- Numeric feature scaling (StandardScaler)
- Data leakage prevention (G1, G2 excluded; fit on train only)
"""

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# ------------------------------------------------------------
# Helper: Map final grade (G3) to 5-class performance label
# ------------------------------------------------------------
def map_grade_to_class(g3):
    """
    Converts a numeric G3 grade into one of five performance levels:
        0 — Very Poor   (0–9)
        1 — Poor        (10–11)
        2 — Satisfactory(12–13)
        3 — Good        (14–15)
        4 — Excellent   (16–20)
    """
    if g3 <= 9:
        return 0
    elif g3 <= 11:
        return 1
    elif g3 <= 13:
        return 2
    elif g3 <= 15:
        return 3
    else:
        return 4


# ------------------------------------------------------------
# Internal helper: build and fit the preprocessor
# ------------------------------------------------------------
def _build_preprocessor(X_train):
    """
    Identifies numeric and categorical columns from X_train and
    returns a fitted ColumnTransformer that scales numerics with
    StandardScaler and encodes categoricals with OneHotEncoder.

    Fitting is done exclusively on the training split to prevent
    data leakage into the test set.
    """
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


# ------------------------------------------------------------
# Default path: load from UCI repository
# ------------------------------------------------------------
def load_and_prepare_data():
    """
    Fetches the Student Performance dataset (UCI ID 320),
    engineers the target variable, removes leakage columns,
    preprocesses features, and returns stratified train/test splits.

    Returns
    -------
    X_train_processed : np.ndarray
    X_test_processed  : np.ndarray
    y_train           : pd.Series
    y_test            : pd.Series
    """
    # Fetch dataset programmatically
    student_performance = fetch_ucirepo(id=320)

    X = student_performance.data.features
    y = student_performance.data.targets

    # Combine for joint processing
    df = pd.concat([X, y], axis=1)

    # Engineer multi-class target from G3
    df["performance_level"] = df["G3"].apply(map_grade_to_class)

    # Drop all grade columns to prevent data leakage
    df.drop(columns=["G1", "G2", "G3"], inplace=True)

    # Separate features and target
    X = df.drop(columns=["performance_level"])
    y = df["performance_level"]

    # Stratified train-test split (preserves class proportions)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Build and fit preprocessor on training data only
    preprocessor, _, _ = _build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test


# ------------------------------------------------------------
# Uploaded CSV path: user supplies test data
# ------------------------------------------------------------
def prepare_uploaded_data(test_df):
    """
    Accepts a user-uploaded DataFrame (test data only).
    Trains the preprocessor on the full UCI dataset and applies
    it to the uploaded test set — ensuring no leakage and that
    feature columns align correctly.

    The uploaded CSV must contain:
      - The same feature columns as the UCI Student dataset
      - A 'performance_level' column with integer values 0–4
        OR a 'G3' column which will be converted automatically.

    Returns
    -------
    X_train_processed : np.ndarray  (UCI training data)
    X_test_processed  : np.ndarray  (user-uploaded test data)
    y_train           : pd.Series
    y_test            : pd.Series
    """
    # ── Load full UCI dataset for training ──
    student_performance = fetch_ucirepo(id=320)
    X_full = student_performance.data.features
    y_full = student_performance.data.targets

    df_full = pd.concat([X_full, y_full], axis=1)
    df_full["performance_level"] = df_full["G3"].apply(map_grade_to_class)
    df_full.drop(columns=["G1", "G2", "G3"], inplace=True)

    X_train_raw = df_full.drop(columns=["performance_level"])
    y_train = df_full["performance_level"]

    # ── Prepare uploaded test data ──
    test_df = test_df.copy()

    # Auto-convert G3 to performance_level if column is present
    if "G3" in test_df.columns:
        test_df["performance_level"] = test_df["G3"].apply(map_grade_to_class)

    # Drop grade columns that would cause leakage
    for col in ["G1", "G2", "G3"]:
        if col in test_df.columns:
            test_df.drop(columns=[col], inplace=True)

    if "performance_level" not in test_df.columns:
        raise ValueError(
            "Uploaded CSV must contain a 'performance_level' column (0–4) "
            "or a 'G3' column that can be converted automatically."
        )

    X_test_raw = test_df.drop(columns=["performance_level"])
    y_test = test_df["performance_level"].astype(int)

    # ── Align columns: keep only columns that exist in training data ──
    shared_cols = [c for c in X_train_raw.columns if c in X_test_raw.columns]
    X_train_raw = X_train_raw[shared_cols]
    X_test_raw = X_test_raw[shared_cols]

    # ── Build preprocessor fitted on training data ──
    preprocessor, _, _ = _build_preprocessor(X_train_raw)
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)

    return X_train_processed, X_test_processed, y_train, y_test
