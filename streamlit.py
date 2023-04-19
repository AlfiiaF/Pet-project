import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.title('Kidney Stone Prediction based on Urine Analysis')

st.image('Kidney.jpeg')

st.markdown("""
      This application can predict the presence of kidney stones based on your urinalysis results.
      You need to enter the following parameters:
      1. specific gravity, the density of the urine relative to water;
      2. pH balance of urine;
      3. urine osmolality (mOsm);
      4. conductivity (mMho milliMho);
      5. urea concentration in millimoles per litre;
      6. calcium concentration (CALC) in millimolesllitre.
      ***
""")
st.markdown("The data for this application is obtained from 'Physical Characteristics of Urines With and Without Crystals',a chapter from Springer Series in Statistics.")

st.sidebar.header("Specify input parameters")

GRAVITY = st.sidebar.slider('GRAVITY', 1005, 1040, 1017)
PH = st.sidebar.slider('PH', 4.76, 7.94, 5.96)
OSMO = st.sidebar.slider('OSMO', 187, 1236, 645)
COND = st.sidebar.slider('COND', 5.1, 38.0, 21.34)
UREA = st.sidebar.slider('UREA', 10, 620, 277)
CALC = st.sidebar.slider('CALC', 0.17, 14.34, 4.12)

data = {'gravity': GRAVITY/1000,
      'ph': PH,
      'osmo': OSMO,
      'cond': COND, 
      'urea': UREA,
      'calc': CALC}

features = pd.DataFrame(data, index=[0])

st.header('Specified input parameters')
st.write(features)
st.write('---')

model = joblib.load("model.pkl")
prediction = model.predict(features)

st.header('Prediction of Kidney Stone')

if prediction == 0:
  st.write("Congratulations! You don't have kidney stones.")
else:
  st.write("You have kidney stones. Please, visit a doctor.")

st.write('---')

