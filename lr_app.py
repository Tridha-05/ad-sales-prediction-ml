import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Ad Sales Prediction",
    layout="centered"
)

# Load trained model
model = pickle.load(open("linear_regression_model.pkl", "rb"))

st.title("Advertisement Sales Prediction")
st.write("Predict product sales based on advertising spend across TV, Radio, and Newspaper.")
st.info("Note: Advertising spend values are in **thousands** (e.g., 100 = 100,000 units).")


# User inputs
st.subheader("Enter Advertising Spend")

tv = st.number_input("TV Advertising Spend",min_value=0.0,max_value=300.0,step=0.1)
radio = st.number_input("Radio Advertising Spend",min_value=0.0,max_value=50.0,step=0.1)
newsp = st.number_input("Newspaper Advertising Spend",min_value=0.0,max_value=120.0,step=0.1)


if st.button("Predict"):
    try:
        tv = float(tv)
        radio = float(radio)
        newsp = float(newsp)

        features = np.array([[tv, radio, newsp]])
        prediction = model.predict(features)

        st.markdown("---")
        st.success(f"Predicted Sales: {prediction[0]:.2f}")

    except ValueError:
        st.error("Please enter valid numeric values")

