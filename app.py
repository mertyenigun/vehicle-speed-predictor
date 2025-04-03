import streamlit as st
import joblib
import numpy as np


st.set_page_config(page_title="Araç Hızı Tahmini", page_icon="🚗", layout="centered")

st.markdown("## 🚗 Araç Hızı Tahmini")
st.markdown("Motor deviri (RPM) ve Gaz Pedalı (%) bilgisine göre aracın hızını tahmin eder.")


@st.cache_resource
def load_model():
    return joblib.load("hiz_tahmin_modeli_grid.pkl")  

model = load_model()


rpm = st.slider("Motor Devir Sayısı (RPM)", min_value=500, max_value=4000, step=100, value=1500)
tPos = st.slider("Gaz Pedalı Pozisyonu (%)", min_value=0, max_value=100, step=1, value=50)

if st.button("Tahmini Hızı Göster"):
    tahmin_input = np.array([[rpm, tPos]])
    tahmini_hiz = model.predict(tahmin_input)[0]
    
    st.success(f"🚀 Tahmini Hız: {tahmini_hiz:.2f} km/h")
