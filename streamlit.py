import joblib
import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import shap

# Загрузка изображения
image = Image.open('Kidney.jpeg')

st.title('Kidney Stone Prediction based on Urine Analysis')

st.image(image, use_column_width=True)

# Описание
st.markdown("""
      The six physical characteristics of the urine are: (1) specific gravity...
      ***
""")

st.sidebar.header("Specify input parameters")

# Функция для получения пользовательского ввода
def user_input_features(X):
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

# Загрузка модели и данных
model = joblib.load("model.pkl")
X = pd.read_csv('train.csv') # Предполагается, что у вас есть файл данных 'data.csv'

df = user_input_features(X)

st.header('Specified input parameters')
st.write(df)
st.write('---')

# Предсказание
prediction = model.predict(df)

st.header('Prediction of Kidney Stone')
if prediction == 0:
  st.write("You don't have kidney stones.")
else:
  st.write("You have kidney stones. See a doctor.")

st.write('---')

# Важность признаков
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
st.pyplot(fig)
st.write('---')
