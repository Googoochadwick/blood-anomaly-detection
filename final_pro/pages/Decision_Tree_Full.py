from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

df = pd.read_csv("final_pro/blood_cell_anomaly_detection.csv")

@st.cache_resource
def get_model():
    return joblib.load("final_pro/pages/bloodfull.joblib")

def predict_anomaly(model, values):
    input_df = pd.DataFrame([values], columns=[
        "cell_diameter_um",
        "nucleus_area_pct",
        "chromatin_density",
        "cytoplasm_ratio",
        "circularity",
        "eccentricity",
        "granularity_score",
        "lobularity_score",
        "membrane_smoothness",
        "cell_area_px",
        "perimeter_px",
        "mean_r",
        "mean_g",
        "mean_b",
        "stain_intensity",
        "patient_age_group",
        "patient_sex",
        "wbc_count_per_ul",
        "rbc_count_millions_per_ul",
        "hemoglobin_g_dl",
        "hematocrit_pct",
        "platelet_count_per_ul",
        "mcv_fl",
        "mchc_g_dl",
    ])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0].max()

    label = "Anomaly" if prediction == 1 else "Normal"
    return label, probability

st.set_page_config(page_title="Blood Anomaly Predictor", layout="centered")
st.title("Blood Cell Anomaly Prediction App")
st.markdown('''This app predicts the anomaly label for a blood cell sample using 23 input features with a :rainbow[Decision Tree].

---''')

try:
    model = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.sidebar.header("Enter Blood Cell Features")
cell_diameter_um = st.sidebar.number_input("Cell diameter (µm)", min_value=1.0, max_value=25.0, value=10.0, step=0.1)
nucleus_area_pct = st.sidebar.number_input("Nucleus area (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
chromatin_density = st.sidebar.number_input("Chromatin density", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
cytoplasm_ratio = st.sidebar.number_input("Cytoplasm ratio", min_value=0.05, max_value=1.0, value=0.5, step=0.01)
circularity = st.sidebar.number_input("Circularity", min_value=0.1, max_value=1.0, value=0.5, step=0.01)
eccentricity = st.sidebar.number_input("Eccentricity", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
granularity_score = st.sidebar.number_input("Granularity score", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
lobularity_score = st.sidebar.number_input("Lobularity score", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
membrane_smoothness = st.sidebar.number_input("Membrane smoothness", min_value=0.29, max_value=1.0, value=0.7, step=0.01)
cell_area_px = st.sidebar.number_input("Cell area (px)", min_value=10, max_value=2000, value=300, step=1)
perimeter_px = st.sidebar.number_input("Perimeter (px)", min_value=8, max_value=500, value=100, step=1)
mean_r = st.sidebar.number_input("Mean R", min_value=0, max_value=255, value=150, step=1)
mean_g = st.sidebar.number_input("Mean G", min_value=0, max_value=255, value=120, step=1)
mean_b = st.sidebar.number_input("Mean B", min_value=0, max_value=255, value=130, step=1)
stain_intensity = st.sidebar.number_input("Stain intensity", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
patient_age_group = st.sidebar.selectbox("Patient age group", ["Adult", "Elderly", "Pediatric"])
patient_sex = st.sidebar.selectbox("Patient sex", ["F", "M"])
wbc_count_per_ul = st.sidebar.number_input("WBC count (/µL)", min_value=1000, max_value=20000, value=6000, step=1)
rbc_count_millions_per_ul = st.sidebar.number_input("RBC count (million/µL)", min_value=2.11, max_value=7.0, value=4.5, step=0.01)
hemoglobin_g_dl = st.sidebar.number_input("Hemoglobin (g/dL)", min_value=5.4, max_value=20.0, value=14.0, step=0.1)
hematocrit_pct = st.sidebar.number_input("Hematocrit (%)", min_value=20.3, max_value=60.0, value=40.0, step=0.1)
platelet_count_per_ul = st.sidebar.number_input("Platelet count (/µL)", min_value=20000, max_value=543084, value=250000, step=1)
mcv_fl = st.sidebar.number_input("MCV (fL)", min_value=60.0, max_value=120.0, value=90.0, step=0.1)
mchc_g_dl = st.sidebar.number_input("MCHC (g/dL)", min_value=28.0, max_value=38.0, value=33.0, step=0.1)

if st.sidebar.button("Predict"):
    values = {
        "cell_diameter_um": cell_diameter_um,
        "nucleus_area_pct": nucleus_area_pct,
        "chromatin_density": chromatin_density,
        "cytoplasm_ratio": cytoplasm_ratio,
        "circularity": circularity,
        "eccentricity": eccentricity,
        "granularity_score": granularity_score,
        "lobularity_score": lobularity_score,
        "membrane_smoothness": membrane_smoothness,
        "cell_area_px": cell_area_px,
        "perimeter_px": perimeter_px,
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "stain_intensity": stain_intensity,
        "patient_age_group": patient_age_group,
        "patient_sex": patient_sex,
        "wbc_count_per_ul": wbc_count_per_ul,
        "rbc_count_millions_per_ul": rbc_count_millions_per_ul,
        "hemoglobin_g_dl": hemoglobin_g_dl,
        "hematocrit_pct": hematocrit_pct,
        "platelet_count_per_ul": platelet_count_per_ul,
        "mcv_fl": mcv_fl,
        "mchc_g_dl": mchc_g_dl,
    }
    prediction, probability = predict_anomaly(model, values)

    if prediction == "Anomaly":
        st.error("Prediction: Anomaly detected (anomaly_label = 1)")
    else:
        st.success("Prediction: Normal (anomaly_label = 0)")

    st.info(f"Model confidence: {probability * 100:.2f}%")
st.dataframe((df.drop(df.columns[[0,28,29,30,31,32,33,34,35]], axis=1)).head(5))
st.markdown('---') 
st.markdown('''# Immediate concerns
Once we press predict, we can see with the default prediction values, while it predicted correctly, the confidence is 100%, this usually isnt a big
problem, however, changes to the input do not change the confidence score.
    
The other problem we can immediatly see are the no of input fields present, a wopping 23! ''')

st.markdown('''# How did we get here?
Before I choose the model to train this on, I first had to clean the data.
''')
st.code("""
import pandas as pd
df = pd.read_csv("final_pro/blood_cell_anomaly_detection.csv")
df
""")
df = pd.read_csv("final_pro/blood_cell_anomaly_detection.csv")
st.dataframe((df.drop(df.columns[[0,28,29,30,31,32,33,34,35]], axis=1)).sample(5))
st.markdown('''The dataset has multiple columes that are of no use to the model. These consist of columes that
compare this dataset findings to the findings of the cytodiffuse research paper''')
st.code("""
df = df.drop(df.columns[[0,28,29,30,31,32,33,34,35]], axis=1)
df
""")
st.dataframe((df.drop(df.columns[[0,28,29,30,31,32,33,34,35]], axis=1)).sample(5))
st.markdown('''The above are all the values that are of use to the model.''')

st.markdown('''# Model selection
We wish to know if a cell is showcasing an anomaly or not using the anomaly_lable column in our data.
This makes our problem a binary classification problem.

As a baseline, I used Decision Tree as it is one than can easily fit both categorical and numerical data.

# Model training

The model training process usually requires the usual spliting and encoding as one would expect.
My main goal was to focus on hyperparameters to finetune the model.


''')
st.code("""
params=[
{"criterion":'gini', "max_depth":5, "min_samples_split":5},
{"criterion":'gini', "max_depth":10, "min_samples_split":5},
{"criterion": 'entropy', "max_depth":5, "min_samples_split":5},
{"criterion":'entropy', "max_depth":8, "min_samples_split":5},
{"criterion":'gini', "min_samples_split":5},
{"criterion":'entropy', "min_samples_split":5}]
""")
st.markdown('''The above are the parameters that were tested to be used for the model.

    Params	                                Accuracy	
    gini, no max_depth, min_split=5	      0.93	
    entropy, no max_depth, min_split=5      0.93	
    entropy, depth=8, min_split=5	      0.92	
    gini, depth=10, min_split=5             0.92	
    entropy, depth=5, min_split=5           0.89	
    
based on the above results, I chose to use the first one.''')
with st.expander("Spoiler"):
  st.write("I would later on realise that just checking accuracy is a bad idea.")
st.markdown('''# Next steps
My immediate intent was to:
    
+ Reduce the number of input fields
+ Confirm whether the model is overfitting
+ Choose a better model

Check Decision Tree Minimal''')