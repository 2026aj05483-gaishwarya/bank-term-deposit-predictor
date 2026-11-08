"""Streamlit app: score an uploaded test CSV with a saved classifier."""

from pathlib import Path

import io

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors",
    "naive_bayes": "Naive Bayes",
    "svm": "Support Vector Machine",
}

st.set_page_config(page_title="Term deposit predictor", layout="wide")
st.title("Bank term deposit predictor")
st.write("BITS ID: 2026 AJ05483")
st.caption(
    "Upload a test CSV that includes the original input columns and the target y. "
    "The saved pipeline encodes and scales the rows, then scores them without retraining."
)

uploaded = st.file_uploader("Test CSV", type=["csv"])
model_paths = {path.stem: path for path in sorted(MODEL_DIR.glob("*.joblib"))}

if not model_paths:
    st.error(f"No saved models found in {MODEL_DIR}.")
    st.stop()

choice = st.selectbox(
    "Model",
    options=list(model_paths),
    format_func=lambda key: MODEL_LABELS.get(key, key),
)

if uploaded is None:
    st.info("Upload test_data.csv, or any CSV with the same columns and a y column of yes or no.")
    st.stop()

raw = uploaded.getvalue().decode("utf-8", errors="replace")
separator = ";" if raw.splitlines()[0].count(";") > raw.splitlines()[0].count(",") else ","
frame = pd.read_csv(io.StringIO(raw), sep=separator)

if "y" not in frame.columns:
    st.error("The CSV must include the target column y.")
    st.stop()

y_true = (frame["y"].astype(str).str.lower() == "yes").astype(int)
features = frame.drop(columns=["y"])

pipeline = joblib.load(model_paths[choice])
y_pred = pipeline.predict(features)
y_score = pipeline.predict_proba(features)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC": roc_auc_score(y_true, y_score),
    "Precision": precision_score(y_true, y_pred, zero_division=0),
    "Recall": recall_score(y_true, y_pred, zero_division=0),
    "F1": f1_score(y_true, y_pred, zero_division=0),
    "MCC": matthews_corrcoef(y_true, y_pred),
}

st.subheader(MODEL_LABELS.get(choice, choice))
st.dataframe(
    pd.DataFrame(metrics, index=["Score"]).T.round(4),
    use_container_width=True,
)

labels = ["no", "yes"]
matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
report = classification_report(
    y_true, y_pred, labels=[0, 1], target_names=labels, zero_division=0
)

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Classification report")
    st.code(report, language="text")
