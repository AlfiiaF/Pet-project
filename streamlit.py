import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.title('Kidney Stone Prediction based on Urine Analysis')

st.image('Kidney.jpeg')

st.markdown("""
      The six physical characteristics of the urine are: (1) specific gravity, the density of the urine relative to water; (2) pH, the negative logarithm of the hydrogen ion; (3) osmolarity (mOsm), a unit used in biology and medicine but not in
      physical chemistry. Osmolarity is proportional to the concentration of
      molecules in solution; (4) conductivity (mMho milliMho). One Mho is one
      reciprocal Ohm. Conductivity is proportional to the concentration of charged
      ions in solution; (5) urea concentration in millimoles per litre; and (6) calcium
      concentration (CALC) in millimolesllitre.
      The data is obtained from 'Physical Characteristics of Urines With and Without Crystals',a chapter from Springer Series in Statistics.
      ***
""")

st.sidebar.header("Specify input parameters")

uploaded_files = st.file_uploader("train.csv", accept_multiple_files=True)

for file in uploaded_files:

    bytes_data = file.read()

    st.write("File uploaded:", file.name)

    st.write(bytes_data)

df= pd.read_csv(file)

st.write(df)

GRAVITY = st.sidebar.slider('GRAVITY', float(X.gravity.min()), float(X.gravity.max()), float(X.gravity.mean()))
PH = st.sidebar.slider('PH', float(X.ph.min()), float(X.ph.max()), float(X.ph.mean()))
OSMO = st.sidebar.slider('OSMO', float(X.osmo.min()), float(X.osmo.max()), float(X.osmo.mean()))
COND = st.sidebar.slider('COND', float(X.cond.min()), float(X.cond.max()), float(X.cond.mean()))
UREA = st.sidebar.slider('UREA', float(X.urea.min()), float(X.urea.max()), float(X.urea.mean()))
CALC = st.sidebar.slider('CALC', float(X.calc.min()), float(X.calc.max()), float(X.calc.mean()))

