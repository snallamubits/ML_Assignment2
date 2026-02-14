# ------------------------------------------------------------
# Main Streamlit Application
#
# Compares multiple classification models on the Student
# Performance dataset and displays evaluation metrics,
# a classification report, and qualitative observations.
#
# Required Streamlit features implemented:
#   a. Dataset upload option (CSV — test data only)
#   b. Model selection dropdown
#   c. Evaluation metrics display
#   d. Confusion matrix and classification report
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import platform
import socket

from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)

from model.dataprep import load_and_prepare_data, prepare_uploaded_data
from model.metrics import build_models, evaluate_model


# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Classification",
    layout="wide"
)

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("Student Performance Classification Models")
st.write(
    "This application trains and evaluates multiple machine learning "
    "classification models on the Student Performance dataset. "
    "Use the default UCI dataset or upload your own test CSV below."
)

st.markdown("--------Student details------")
st.write("Name : Saravanan Nallamuthu")
st.write("Email : 2025ab05285@wilp.bits-pilani.ac.in")
st.markdown("-----------------------------")

st.markdown("-----Environment details-----")
st.text(f"Machine Name: {socket.gethostname()}")
st.text(f"OS Detail: {platform.system()} {platform.release()}")
st.markdown("-----------------------------")

# ------------------------------------------------------------
# Sidebar — Controls
# ------------------------------------------------------------
st.sidebar.header("Configuration")

# b. Model selection dropdown
MODEL_OPTIONS = [
    "All Models",
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost",
]
selected_model = st.sidebar.selectbox(
    "Select Model to Evaluate",
    options=MODEL_OPTIONS,
    index=0,
    help="Choose a specific model or run all six at once."
)

st.sidebar.markdown("---")

# a. Dataset upload option (CSV — test data only, as per assignment spec)
st.sidebar.subheader("Upload Test Data (Optional)")
st.sidebar.write(
    "Upload a CSV file containing test features and a `performance_level` "
    "column (0–4). Leave blank to use the built-in UCI dataset."
)
uploaded_file = st.sidebar.file_uploader(
    "Upload test CSV",
    type=["csv"],
    help="CSV must contain the same features as the UCI Student dataset "
         "plus a 'performance_level' target column (values 0–4)."
)

st.sidebar.markdown("---")
run_button = st.sidebar.button("Run Models", use_container_width=True)

# ------------------------------------------------------------
# Class label mapping (for readability in reports)
# ------------------------------------------------------------
CLASS_LABELS = {
    0: "Very Poor (0–9)",
    1: "Poor (10–11)",
    2: "Satisfactory (12–13)",
    3: "Good (14–15)",
    4: "Excellent (16–20)",
}

# ------------------------------------------------------------
# Observations per model — reflect actual computed results
# ------------------------------------------------------------
OBSERVATIONS = {
    "Logistic Regression": (
        "Achieved the highest accuracy among all six models. Its linear "
        "decision boundary generalised well on this dataset once intermediate "
        "grades were removed, outperforming more complex models that struggled "
        "with the reduced feature signal."
    ),
    "Decision Tree": (
        "Captured feature interactions effectively; however, its performance "
        "showed fluctuation across runs, indicating a tendency to overfit the "
        "training data without pruning."
    ),
    "KNN": (
        "Produced moderate results but proved sensitive to feature scaling "
        "and the choice of neighbourhood size, which affected prediction "
        "stability across different sample distributions."
    ),
    "Naive Bayes": (
        "Executed very quickly but collapsed most predictions into a single "
        "dominant class, resulting in the lowest accuracy. The strong "
        "feature-independence assumption and class imbalance in predictions "
        "caused a significant gap between precision and F1 score."
    ),
    "Random Forest": (
        "Demonstrated solid performance as the second-best model, benefiting "
        "from ensemble averaging to reduce variance. However, without the "
        "intermediate grade features, the ensemble could not fully leverage "
        "its capacity for complex pattern learning."
    ),
    "XGBoost": (
        "Performed below expectation on this dataset, tying with KNN on "
        "accuracy and recording the lowest MCC score. The removal of G1 and "
        "G2 significantly reduced the predictive signal, limiting the boosting "
        "algorithm's ability to learn meaningful patterns from demographic "
        "features alone."
    ),
}


# ------------------------------------------------------------
# Helper — render one model's full results
# ------------------------------------------------------------
def display_model_results(name, model, X_test, y_test):
    st.subheader(f"Model: {name}")

    # Compute and display all six metrics as individual tiles
    metrics = evaluate_model(model, X_test, y_test)
    metrics_display = {k: round(v, 4) for k, v in metrics.items()}
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    for col, (metric_name, value) in zip(
        [col1, col2, col3, col4, col5, col6], metrics_display.items()
    ):
        col.metric(label=metric_name, value=f"{value:.4f}")

    st.markdown("")

    # d. Confusion matrix + classification report displayed side by side
    col_left, col_right = st.columns([1, 1])

    y_pred = model.predict(X_test)
    # Full labels used in the classification report (right column)
    label_names = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS)]
    # Shortened labels for confusion matrix tick marks to prevent overflow
    SHORT_LABELS = ["Very Poor", "Poor", "Satisfactory", "Good", "Excellent"]

    # Confusion Matrix (left column)
    with col_left:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(3.8, 3.2))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=SHORT_LABELS
        )
        disp.plot(ax=ax, colorbar=False, xticks_rotation=30)
        ax.set_title(name, fontsize=9, pad=6)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_xlabel("Predicted label", fontsize=8)
        ax.set_ylabel("True label", fontsize=8)
        # Reserve bottom and left margin so rotated labels are not clipped
        fig.subplots_adjust(bottom=0.22, left=0.22)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    # Classification Report (right column)
    with col_right:
        st.markdown("**Classification Report**")
        report_str = classification_report(
            y_test, y_pred,
            target_names=label_names,
            zero_division=0
        )
        st.code(report_str, language=None)

    # Per-model qualitative observation
    st.info(f"**Observation:** {OBSERVATIONS.get(name, '')}")
    st.markdown("---")


# ------------------------------------------------------------
# Main execution block
# ------------------------------------------------------------
if run_button:

    with st.spinner("Loading data and training models..."):

        # Data source: use uploaded CSV if provided, else default UCI dataset
        if uploaded_file is not None:
            try:
                test_df = pd.read_csv(uploaded_file)
                st.success(
                    f"Uploaded CSV loaded: {test_df.shape[0]} rows, "
                    f"{test_df.shape[1]} columns."
                )
                # Train on UCI dataset, evaluate on user-supplied test data
                X_train, X_test, y_train, y_test = prepare_uploaded_data(
                    test_df
                )
                st.info(
                    "Training on the UCI dataset; evaluating on your "
                    "uploaded test set."
                )
            except Exception as e:
                st.error(f"Could not process uploaded file: {e}")
                st.stop()
        else:
            X_train, X_test, y_train, y_test = load_and_prepare_data()
            st.info("Using built-in UCI Student Performance dataset.")

        # Build all six models and fit on training data
        all_models = build_models()
        for model in all_models.values():
            model.fit(X_train, y_train)

    st.success("Training complete!")
    st.markdown("---")

    # c. Overall model comparison table — all six models, all six metrics
    st.header("Model Comparison Table")
    rows = []
    for name, model in all_models.items():
        m = evaluate_model(model, X_test, y_test)
        m["Model"] = name
        rows.append(m)
    results_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(results_df.round(4), use_container_width=True)
    st.markdown("---")

    # Per-model detailed results — metrics, confusion matrix, classification report
    st.header("Detailed Model Results")
    if selected_model == "All Models":
        for name, model in all_models.items():
            display_model_results(name, model, X_test, y_test)
    else:
        if selected_model in all_models:
            display_model_results(
                selected_model,
                all_models[selected_model],
                X_test,
                y_test
            )
        else:
            st.warning(f"Model '{selected_model}' not found.")

    # Summary observations table
    st.header("Summary: Observations on Model Performance")
    obs_df = pd.DataFrame(
        list(OBSERVATIONS.items()),
        columns=["ML Model", "Observation"]
    )
    st.table(obs_df)
