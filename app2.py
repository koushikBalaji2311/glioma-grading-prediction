import streamlit as st
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('my_model.keras')

st.title("Cancer Grade Prediction")

# Gender input with caption
gender = st.selectbox("Gender (0: Female, 1: Male)", options=[0, 1])
st.caption("Select the gender of the patient.")

# Age at Diagnosis input with caption
age = st.number_input("Age at Diagnosis (years)", min_value=0.0)
st.caption("Enter the age of the patient at the time of diagnosis.")

# Race input with caption
race = st.selectbox("Race (0: White, 1: Black, 2: Asian, 3: Other)", options=[0, 1, 2, 3])
st.caption("Select the race category that applies to the patient.")

# IDH1 Mutation Status input with caption
idh1 = st.selectbox("IDH1 Mutation Status (0: No, 1: Yes)", options=[0, 1])
st.caption("Specify whether the patient has an IDH1 mutation.")

# TP53 Mutation Status input with caption
tp53 = st.selectbox("TP53 Mutation Status (0: No, 1: Yes)", options=[0, 1])
st.caption("Specify whether the patient has a TP53 mutation.")

# ATRX Mutation Status input with caption
atrx = st.selectbox("ATRX Mutation Status (0: No, 1: Yes)", options=[0, 1])
st.caption("Specify whether the patient has an ATRX mutation.")

# PTEN Mutation Status input with caption
pten = st.selectbox("PTEN Mutation Status (0: No, 1: Yes)", options=[0, 1])
st.caption("Specify whether the patient has a PTEN mutation.")

# EGFR Mutation Status input with caption
egfr = st.selectbox("EGFR Mutation Status (0: No, 1: Yes)", options=[0, 1])
st.caption("Specify whether the patient has an EGFR mutation.")

# CIC Mutation Status input with caption
cic = st.selectbox("CIC Mutation Status (0: No, 1: Yes)", options=[0, 1])
st.caption("Specify whether the patient has a CIC mutation.")

# MUC16 Mutation Status input with caption
muc16 = st.selectbox("MUC16 Mutation Status (0: No, 1: Yes)", options=[0, 1])
st.caption("Specify whether the patient has a MUC16 mutation.")

# PIK3CA Mutation Status input with caption
pik3ca = st.selectbox("PIK3CA Mutation Status (0: No, 1: Yes)", options=[0, 1])
st.caption("Specify whether the patient has a PIK3CA mutation.")

# NF1 Mutation Status input with caption
nf1 = st.selectbox("NF1 Mutation Status (0: No, 1: Yes)", options=[0, 1])
st.caption("Specify whether the patient has an NF1 mutation.")

# PIK3R1 Mutation Status input with caption
pik3r1 = st.selectbox("PIK3R1 Mutation Status (0: No, 1: Yes)", options=[0, 1])
st.caption("Specify whether the patient has a PIK3R1 mutation.")

# FUBP1 Mutation Status input with caption
fubp1 = st.selectbox("FUBP1 Mutation Status (0: No, 1: Yes)", options=[0, 1])
st.caption("Specify whether the patient has a FUBP1 mutation.")

# Prepare the input data for prediction
input_data = np.array([[gender, age, race, idh1, tp53, atrx, pten, egfr, cic, muc16, pik3ca, nf1, pik3r1, fubp1]])

# Make a prediction
if st.button("Predict Grade"):
    prediction = model.predict(input_data)
    predicted_grade = "High" if prediction[0][0] > 0.5 else "Low"
    st.write(f"The predicted cancer grade is: {predicted_grade}")
