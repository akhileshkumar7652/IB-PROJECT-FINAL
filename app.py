import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= PAGE =================
st.set_page_config(
    page_title="AI Bioprocess Scale-Up Predictor",
    layout="centered"
)

st.title("🧪 AI Bioprocess Scale-Up Predictor")

st.write("Predict Max FPU/ml from fermentation conditions")

# ================= LOAD MODEL =================
try:
    model = pickle.load(open("model.pkl", "rb"))
except:
    st.error("model.pkl not found. Run model.py first.")
    st.stop()

# ================= INPUTS =================
st.subheader("Enter Fermentation Parameters")

avg_temp = st.slider(
    "Average Temperature (°C)",
    20.0,
    40.0,
    28.0
)

avg_ph = st.slider(
    "Average pH",
    3.0,
    8.0,
    5.0
)

min_do = st.slider(
    "Minimum DO (%)",
    0.0,
    100.0,
    20.0
)

inoculum_size = st.slider(
    "Inoculum size (mL)",
    10,
    1000,
    100
)

max_rpm = st.slider(
    "Max RPM",
    50,
    1000,
    300
)

culture_duration = st.slider(
    "Culture Duration (days)",
    1,
    30,
    7
)

lactose_feed = st.slider(
    "Total Lactose Feed (mL)",
    0,
    5000,
    500
)

contamination = st.selectbox(
    "Contamination Present?",
    ["No", "Yes"]
)

media_volume = st.slider(
    "Total Media Volume (mL)",
    1000,
    10000,
    4000
)

# ================= CONVERT =================
contamination_value = 1 if contamination == "Yes" else 0

# ================= CREATE INPUT =================
feature_columns = [
    "Avg_temp",
    "Avg_pH",
    "Min_DO %",
    "Inoculum size (mL)",
    "Max_RPM",
    "Total culture duration (days)",
    "Total feed of Lactose (mL)",
    "Contamination (Y/N)",
    "Total media volume (mL)"
]

input_values = [[
    avg_temp,
    avg_ph,
    min_do,
    inoculum_size,
    max_rpm,
    culture_duration,
    lactose_feed,
    contamination_value,
    media_volume
]]

input_data = pd.DataFrame(
    input_values,
    columns=feature_columns
)

# ================= PREDICT =================
prediction = model.predict(input_data)[0]

# ================= OUTPUT =================
st.subheader("Prediction")

st.success(f"Predicted Max FPU/ml: {prediction:.2f}")

# ================= INTERPRETATION =================
st.subheader("🧬 Interpretation")

if prediction < 2:
    st.warning("Low enzyme production predicted")

elif prediction < 5:
    st.info("Moderate enzyme production predicted")

else:
    st.success("High enzyme production predicted")

# ================= GRAPH =================
st.subheader("Lactose Feed vs Predicted FPU")

feed_range = np.linspace(0, 5000, 20)

predictions = []

for feed in feed_range:

    temp_values = [[
        avg_temp,
        avg_ph,
        min_do,
        inoculum_size,
        max_rpm,
        culture_duration,
        feed,
        contamination_value,
        media_volume
    ]]

    temp_input = pd.DataFrame(
        temp_values,
        columns=feature_columns
    )

    pred = model.predict(temp_input)[0]

    predictions.append(pred)

# ================= PLOT =================
fig, ax = plt.subplots()

ax.plot(feed_range, predictions)

ax.set_xlabel("Total Lactose Feed (mL)")

ax.set_ylabel("Predicted Max FPU/ml")

ax.set_title("Effect of Lactose Feed on Enzyme Production")

st.pyplot(fig)