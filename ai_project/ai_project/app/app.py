import streamlit as st
import joblib
import numpy as np
from ai_project.config import AppConfig

st.set_page_config(page_title="AI Project Demo", page_icon="🤖", layout="centered")

cfg = AppConfig()
st.title(cfg.title)
st.caption("Starter template — swap this with your NLP/CV/tabular model.")

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

model = None
try:
    model = load_model(cfg.model_path)
    st.success(f"Loaded model from {cfg.model_path}")
except Exception as e:
    st.warning("Train a model first (see README). Using a fake predictor.")
    class Fake:
        def predict(self, X): return [0]
    model = Fake()

sepal_len = st.number_input("Sepal length", value=5.1)
sepal_wid = st.number_input("Sepal width", value=3.5)
petal_len = st.number_input("Petal length", value=1.4)
petal_wid = st.number_input("Petal width", value=0.2)

if st.button("Predict"):
    x = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
    pred = model.predict(x)[0]
    st.write("### Prediction:", pred)