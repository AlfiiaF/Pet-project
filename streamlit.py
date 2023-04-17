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

GRAVITY = st.sidebar.slider('GRAVITY', 1.005, 1.04, 1.017)
PH = st.sidebar.slider('PH', 4.76, 7.94, 5.96)


