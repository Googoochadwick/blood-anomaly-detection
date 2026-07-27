from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

FEATURE_COLUMNS = [
    "circularity",
    "cell_area_px",
    "lobularity_score",
    "granularity_score",
    "nucleus_area_pct",
]


@st.cache_resource
def get_model():
    return joblib.load(" final_pro/pages/bloodXGBmin.joblib")


def predict_anomaly(model, values):
    input_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0].max()

    label = "Anomaly" if prediction == 1 else "Normal"
    return label, probability


st.set_page_config(page_title="Blood Anomaly Predictor", layout="centered")
st.title("Blood Cell Anomaly Prediction")
st.markdown('''This app predicts the anomaly label for a blood cell sample using 5 input features with :rainbow[XGBoost].

---''')

try:
    model = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.sidebar.header("Enter Blood Cell Features")
circularity = st.sidebar.number_input("Circularity", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
cell_area_px = st.sidebar.number_input("Cell area (px)", min_value=0, max_value=10000, value=300, step=1)
lobularity_score = st.sidebar.number_input("Lobularity score", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
granularity_score = st.sidebar.number_input("Granularity score", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
nucleus_area_pct = st.sidebar.number_input("Nucleus area (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

if st.sidebar.button("Predict"):
    values = {
        "circularity": circularity,
        "cell_area_px": cell_area_px,
        "lobularity_score": lobularity_score,
        "granularity_score": granularity_score,
        "nucleus_area_pct": nucleus_area_pct,
    }
    prediction, probability = predict_anomaly(model, values)

    if prediction == "Anomaly":
        st.error("Prediction: Anomaly detected (anomaly_label = 1)")
    else:
        st.success("Prediction: Normal (anomaly_label = 0)")

    st.info(f"Model confidence: {probability * 100:.2f}%")

st.markdown('''
---

# Conclusions

here are the metrics of all the models i used

''')


data = {
    "Model": ["Decision Tree (full)", "Decision Tree (minimal)", "XGBoost (full)", "XGBoost (minimal)"],
    "Params": [
        "criterion=gini, min_samples_split=5",
        "criterion=gini, min_samples_split=5",
        "n_estimators=100, learning_rate=0.1, max_depth=3",
        "n_estimators=100, learning_rate=0.1, max_depth=3",
    ],
    "Accuracy": [0.93, 0.92, 0.93, 0.96],
    "Precision (0)": [0.95, 0.94, 0.93, 0.95],
    "Recall (0)": [0.95, 0.93, 0.97, 0.99],
    "F1 (0)": [0.95, 0.94, 0.95, 0.97],
    "Precision (1)": [0.90, 0.86, 0.93, 0.98],
    "Recall (1)": [0.89, 0.88, 0.84, 0.89],
    "F1 (1)": [0.89, 0.87, 0.88, 0.93],
    "Macro F1": [0.92, 0.91, 0.92, 0.95],
    "Weighted F1": [0.93, 0.92, 0.93, 0.96],
}

df = pd.DataFrame(data)

st.subheader("Model comparison")
st.dataframe(
    df.style.format({
        "Accuracy": "{:.2f}",
        "Precision (0)": "{:.2f}",
        "Recall (0)": "{:.2f}",
        "F1 (0)": "{:.2f}",
        "Precision (1)": "{:.2f}",
        "Recall (1)": "{:.2f}",
        "F1 (1)": "{:.2f}",
        "Macro F1": "{:.2f}",
        "Weighted F1": "{:.2f}",
    }).background_gradient(cmap="Greens"),
    use_container_width=True
)

st.markdown('''I would like to point out that even though minimal models performed well, the
inputs were selected based on feature importance, in the real world, modification of input fields should
always be discussed with medical professionals and approved by a medical administration for compliance.''')