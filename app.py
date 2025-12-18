import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("house_price_model.pkl")

st.title("🏠 House Price Prediction")

# Inputs
location = st.text_input("Location", "Whitefield")
total_sqft = st.number_input("Total Sqft", 300, 10000, 1200)
bath = st.number_input("Bathrooms", 1, 10, 2)
bhk = st.number_input("BHK", 1, 10, 2)

if st.button("Find Price"):
    data = pd.DataFrame({
        "location": [location],
        "total_sqft": [total_sqft],
        "bath": [bath],
        "BHK": [bhk]
    })

    data = pd.get_dummies(data, columns=["location"], drop_first=True)
    data = data.reindex(columns=model.feature_names_in_, fill_value=0)

    price = model.predict(data)[0]

    if price >= 100:
        st.success(f"Estimated Price: ₹{price/100:.2f} Crore")
    else:
        st.success(f"Estimated Price: ₹{price:.0f} Lakh")
model = joblib.load("house_price_model.pkl")

