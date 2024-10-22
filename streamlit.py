import joblib
import streamlit as st
import pandas as pd
import base64
import pickle
from PIL import Image
from urllib.request import urlopen
import shap

image = Image.open('Kidney.jpeg')

st.title('Kidney Stone Prediction based on Urine Analysis')

st.image(image, use_column_width=True)

st.markdown("""
The six physical characteristics of the urine are: (1) specific gravity...
***
""")

st.sidebar.header("Specify input parameters")

# Предполагается, что у вас есть набор данных X для слайдеров
# Здесь я буду использовать случайные значения для демонстрации
# Вам нужно загрузить ваш набор данных, чтобы использовать реальные значения
import numpy as np
X = pd.DataFrame({
    'gravity': np.random.rand(10),
    'ph': np.random.rand(10),
    'osmo': np.random.rand(10),
    'cond': np.random.rand(10),
    'urea': np.random.rand(10),
    'calc': np.random.rand(10)
})

def user_input_features():
    GRAVITY = st.sidebar.slider('GRAVITY', float(X.gravity.min()), float(X.gravity.max()), float(X.gravity.mean()))
    PH = st.sidebar.slider('PH', float(X.ph.min()), float(X.ph.max()), float(X.ph.mean()))
    OSMO = st.sidebar.slider('OSMO', float(X.osmo.min()), float(X.osmo.max()), float(X.osmo.mean()))
    COND = st.sidebar.slider('COND', float(X.cond.min()), float(X.cond.max()), float(X.cond.mean()))
    UREA = st.sidebar.slider('UREA', float(X.urea.min()), float(X.urea.max()), float(X.urea.mean()))
    CALC = st.sidebar.slider('CALC', float(X.calc.min()), float(X.calc.max()), float(X.calc.mean()))

    data = {'gravity': GRAVITY,
            'ph': PH,
            'osmo': OSMO,
            'cond': COND,
            'urea': UREA,
            'calc': CALC}

    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.header('Specified input parameters')
st.write(df)
st.write('---')

model = joblib.load("model.pkl")

prediction = model.predict(df)

st.header('Prediction of Kidney Stone')
if prediction == 0:
    st.write("You don't have kidney stones.")
else:
    st.write("You have kidney stones. See a doctor.")

st.write('---')

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X)
st.pyplot(fig)
st.write('---')


