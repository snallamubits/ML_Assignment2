"""
Model Training and Evaluation Module
=====================================

Implements six classification models:
    1. Logistic Regression
    2. Decision Tree Classifier
    3. K-Nearest Neighbor (KNN)
    4. Naive Bayes (Gaussian)
    5. Random Forest (Ensemble)
    6. XGBoost (Ensemble)

Evaluates each model using:
    Accuracy, AUC (OvR macro), Precision (macro),
    Recall (macro), F1 (macro), MCC
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


# ------------------------------------------------------------
# Model factory — returns unfitted model instances
# ------------------------------------------------------------
def build_models():
    """
    Returns an ordered dictionary of unfitted model instances.

    All six models use the same hyperparameter defaults to ensure
    a fair, comparable baseline evaluation on the same dataset.

    Gaussian Naive Bayes is selected because the feature set
    contains continuous numerical values after preprocessing,
    making it more appropriate than Multinomial NB (which expects
    discrete count-based inputs).
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),
        # Classifies each sample based on the 5 nearest training neighbours.
        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),
        # Gaussian NB selected: features are continuous after StandardScaler.
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            objective="multi:softprob",
            num_class=5,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0
        ),
    }


# ------------------------------------------------------------
# Single model evaluation
# ------------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    """
    Generates predictions from a fitted model and computes
    six evaluation metrics.

    Parameters
    ----------
    model   : fitted sklearn-compatible estimator
    X_test  : array-like, shape (n_samples, n_features)
    y_test  : array-like, shape (n_samples,)

    Returns
    -------
    dict with keys: Accuracy, AUC, Precision, Recall, F1, MCC
    """
    y_pred = model.predict(X_test)

    # AUC requires probability estimates; gracefully handle models
    # that do not expose predict_proba.
    try:
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(
            y_test, y_prob,
            multi_class="ovr",
            average="macro"
        )
    except Exception:
        auc = np.nan

    return {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "AUC":       auc,
        "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall":    recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1":        f1_score(y_test, y_pred, average="macro", zero_division=0),
        "MCC":       matthews_corrcoef(y_test, y_pred),
    }


# ------------------------------------------------------------
# Convenience wrapper — train all models and return results df
# ------------------------------------------------------------
def run_all_models(X_train, X_test, y_train, y_test):
    """
    Trains all six models on X_train / y_train and evaluates
    each on X_test / y_test.

    Returns
    -------
    pd.DataFrame with columns: Model, Accuracy, AUC,
                                Precision, Recall, F1, MCC
    """
    models = build_models()
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        metrics["Model"] = name
        results.append(metrics)

    return pd.DataFrame(results)
