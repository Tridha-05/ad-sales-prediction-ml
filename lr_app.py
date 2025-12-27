import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("linear_regression_model.pkl", "rb"))

st.title("Advertisement Sales Prediction")

# User inputs
tv = st.text_input("Enter TV Advertising Spend")
radio = st.text_input("Enter Radio Advertising Spend")
newsp = st.text_input("Enter Newspaper Advertising Spend")

if st.button("Predict"):
    try:
        tv = float(tv)
        radio = float(radio)
        newsp = float(newsp)

        features = np.array([[tv, radio, newsp]])
        prediction = model.predict(features)

        st.success(f"Predicted Sales: {prediction[0]:.2f}")

    except ValueError:
        st.error("Please enter valid numeric values")

