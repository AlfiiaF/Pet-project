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


@st.cache_data
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

# add new features 
#df = add_features(df)


st.header('Specified input parameters')
st.write(df)
st.write('---')

model = joblib.load("model.pkl")

prediction = model.predict(df)


st.header('Prediction of Kidney Stone')
st.write(prediction)
if prediction == 0:
  st.write("You don't have kidney stones")
else:
  st.write("You have kidney stones. See a doctor")

st.write('---')


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')
