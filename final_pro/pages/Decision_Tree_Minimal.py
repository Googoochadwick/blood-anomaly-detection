from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

df = pd.read_csv("final_pro/blood_cell_anomaly_detection.csv")

FEATURE_COLUMNS = [
    "circularity",
    "cell_area_px",
    "lobularity_score",
    "granularity_score",
    "nucleus_area_pct",
]


@st.cache_resource
def get_model():
    return joblib.load("final_pro/pages/bloodminimal.joblib")


def predict_anomaly(model, values):
    input_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0].max()

    label = "Anomaly" if prediction == 1 else "Normal"
    return label, probability


st.set_page_config(page_title="Blood Anomaly Predictor", layout="centered")
st.title("Blood Cell Anomaly Prediction")
st.markdown('''This app predicts the anomaly label for a blood cell sample using 5 input features with a :rainbow[Decision tree].

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
st.dataframe(df[['circularity',                   
'cell_area_px',                 
'lobularity_score',              
'granularity_score',             
'nucleus_area_pct',
'anomaly_label', ]].head(5))
st.markdown('---') 
st.markdown('''# Solving for the number of fields 

23 fields might be fine in the medical field, however, I was sure we can find the most important ones.
To do so I first made a correlation matrix to check the correlation between the values anddddd.....
# Interesting......''')

st.image("final_pro/pages/smsmax.png", caption="Seaborn correlation heatmapped")

st.markdown('''This is a very :rainbow[interesting] heatmap

Just to make sure we are on the same page, a heatmap shows how each value in a dataset affects
another value.

+ :blue[Positive] means if the value of one changes, so does the value for the other in the same manner.
+ Neutral  means if the value changes, no change occures.
+ :red[Negative] means if the value changes, so does the value for the other in the opposit manner. 

As we can see in this chart, the values after stain intensity have no correlation at all. 
This is not enough to remove the fields, I needed more data to remove them.

---
# :rainbow[Lets call a friend] (model)''')
st.image("final_pro/pages/caf.png", caption="Call a friend from who wants to be a millionare")

st.markdown('''In the show who wants to be a millionare, contestants had an option to call a friend to ask them what the
answer to their question could be, it can easily be assumed that their 
friends or family had a higher chance of knowing the answer as they could
search it up.

So how about we call our friend! The trained model can tell us the importance of the features used
and how much they affect out model output.''')

st.image("final_pro/pages/featimp.png", caption="Feature Importance")

st.markdown('''Looking at the chart above, you can see why I choose the values on the left.
The values after nucleas area drop below < 0.1 with chromatin density being 0.05''')

st.markdown('''# Next steps
lets look at whats left:
    
+ :red[~Reduce the number of input fields~]
    + We have succesfully reduced the no of input fields using feature importance.
+ :red[~Confirm whether the model is overfitting~]
    + We can clearly see the confidence score being 100% at all times we later confirm using other metrics.
+ :green[Choose a better model]

Ok one step to go, lets select a better model''')