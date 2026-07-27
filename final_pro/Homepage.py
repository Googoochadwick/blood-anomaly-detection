from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
PAGES_DIR = BASE_DIR / "pages"
DATA_PATH = next(
    (path for path in (BASE_DIR / "blood_cell_anomaly_detection.csv", BASE_DIR.parent / "blood_cell_anomaly_detection.csv") if path.exists()),
    BASE_DIR.parent / "blood_cell_anomaly_detection.csv",
)

st.set_page_config(
    page_title="Blood Anomoly Detection",
    page_icon='🩸',
)
st.title("Blood Anomoly Detection")
st.markdown("---")
st.markdown('''# Project Description
The blood is the transport system of the body, if affected, it can be dangerous and sometimes even lethal.
Catching these anomolies before they create complications is of utmost importance in medical field.
AI can identify patterns in a large dataset, and if tuned properly, can deliver conclusions with the same accuracy as doctors, reducing the time required to conclude the presence of an anomoly.''')

st.markdown('''# Project Contents
The project consists of 4 trained models, 2 on :rainbow[Decision Tree] and 2 on :rainbow[XGboost algorithem].

We will also look into how we choose a :green[baseline model] and how we can spot :red[bad model behavior] using :rainbow[seaborn metrics].''')
st.image(str(PAGES_DIR / "smsmin.png"), caption="Seaborn correlation heatmapped")

st.markdown('''# Database Description
The dataset that was used for this is Blood Cell Anomaly Detection 2025 from Kaggle''')
st.image(str(PAGES_DIR / "kaggle.png"), caption="https://www.kaggle.com/datasets/alitaqishah/blood-cell-anomaly-detection-2025")
df = pd.read_csv(DATA_PATH)

st.code("""
import pandas as pd
df = pd.read_csv("blood_cell_anomaly_detection.csv")
df
""")
st.dataframe(df)