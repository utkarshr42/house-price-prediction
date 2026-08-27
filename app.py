import streamlit as st
import pandas as pd
import joblib

# Load trained model, features, and scaler
model = joblib.load("house_price_model.pkl")
features = joblib.load("house_price_features.pkl")
scaler = joblib.load("house_price_scaler.pkl")

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠"
)


st.sidebar.title("🏠 House Price Predictor")

st.sidebar.write(
    "This machine learning application "
    "estimates house prices based on property details."
)

st.sidebar.info(
    "Enter the house information and click "
    "'Predict House Price' to get an estimated price."
)

st.title("🏠 House Price Predictor")

st.write(
    "Enter the house details below to estimate "
    "the property's price."
)

st.divider()


st.subheader("House Information")

area_sqft = st.number_input(
    "Area (sq ft)",
    min_value=500,
    max_value=4000,
    value=1500
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=5,
    value=3
)

bathrooms = st.number_input(
    "Number of Bathrooms",
    min_value=1,
    max_value=4,
    value=2
)

age_years = st.number_input(
    "Age of House (Years)",
    min_value=0,
    max_value=40,
    value=5
)

location = st.selectbox(
    "Location",
    ["Urban", "Suburban", "Rural"]
)

parking = st.selectbox(
    "Parking",
    ["Yes", "No"]
)

furnished = st.selectbox(
    "Furnished Status",
    ["Furnished", "Semi-Furnished", "Unfurnished"]
)

st.divider()

if st.button(
    "🏠 Predict House Price",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "Area_sqft": [area_sqft],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "Age_years": [age_years],
        "Location": [location],
        "Parking": [parking],
        "Furnished": [furnished]
    })

    # One-hot encode categorical variables
    input_encoded = pd.get_dummies(
        input_data,
        columns=[
            "Location",
            "Parking",
            "Furnished"
        ],
        drop_first=True
    )

    # Match the training feature columns
    input_encoded = input_encoded.reindex(
        columns=features,
        fill_value=0
    )

    # Scale the input using the saved scaler
    input_scaled = scaler.transform(input_encoded)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Display prediction
    st.subheader("💰 Estimated Price")

    st.metric(
        label="Predicted House Price",
        value=f"₹{prediction:,.0f}"
    )

    st.divider()

    st.caption(
        "House Price Prediction | Data Science & AI/ML Portfolio Project"
    )
