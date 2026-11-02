
import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide"
)

# Load saved models
classifier = joblib.load(
    "/content/real_estate_models/classification_model.pkl"
)

regressor = joblib.load(
    "/content/real_estate_models/regression_model.pkl"
)

with open(
    "/content/real_estate_models/metadata.json"
) as f:

    metadata = json.load(f)

features = metadata["features"]
CURRENT_YEAR = metadata["year_reference"]

# Application heading
st.title("🏡 Real Estate Investment Advisor")

st.write(
    "Evaluate property investments using Machine Learning."
)

st.divider()

st.sidebar.header("Enter Property Details")

# User inputs
area = st.sidebar.number_input(
    "Property Area (sq. ft.)",
    min_value=100,
    value=2500
)

bedrooms = st.sidebar.number_input(
    "Bedrooms",
    min_value=1,
    value=3
)

bathrooms = st.sidebar.number_input(
    "Bathrooms",
    min_value=1,
    value=2
)

floors = st.sidebar.number_input(
    "Floors",
    min_value=1,
    value=2
)

year_built = st.sidebar.number_input(
    "Year Built",
    min_value=1900,
    max_value=CURRENT_YEAR,
    value=2018
)

location_score = st.sidebar.slider(
    "Location Score",
    0, 100, 80
)

school_score = st.sidebar.slider(
    "School Score",
    0, 100, 75
)

current_price = st.sidebar.number_input(
    "Current Property Price",
    min_value=10000,
    value=650000
)

# Prediction button
if st.sidebar.button(
    "Analyze Investment",
    type="primary"
):

    # Feature engineering
    property_age = max(
        0,
        CURRENT_YEAR - year_built
    )

    price_per_sqft = current_price / area

    input_data = pd.DataFrame([{

        "Area_sqft": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "Year_Built": year_built,
        "Location_Score": location_score,
        "School_Score": school_score,
        "Current_Price": current_price,
        "Property_Age": property_age,
        "Price_Per_Sqft": price_per_sqft

    }])

    input_data = input_data[features]

    # Classification
    investment_prediction = classifier.predict(
        input_data
    )[0]

    investment_probability = classifier.predict_proba(
        input_data
    )[0][1]

    # Regression
    future_price = regressor.predict(
        input_data
    )[0]

    # ROI calculation
    appreciation = future_price - current_price

    roi = (
        appreciation / current_price
    ) * 100

    # Display results
    st.subheader("Investment Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Current Property Price",
            f"{current_price:,.2f}"
        )

    with col2:

        st.metric(
            "Estimated 5-Year Price",
            f"{future_price:,.2f}"
        )

    with col3:

        st.metric(
            "Estimated ROI",
            f"{roi:.2f}%"
        )

    st.divider()

    st.subheader("Investment Classification")

    if investment_prediction == 1:

        st.success("GOOD INVESTMENT")

    else:

        st.warning("NOT A GOOD INVESTMENT")

    st.write(
        "Good Investment Model Probability:",
        f"{investment_probability * 100:.2f}%"
    )

    st.progress(
        float(investment_probability)
    )

    st.subheader("Five-Year Price Forecast")

    st.write(
        "Expected Property Appreciation:",
        f"{appreciation:,.2f}"
    )

    st.write(
        "Estimated Five-Year ROI:",
        f"{roi:.2f}%"
    )

    st.caption(
        "This is an educational machine learning prediction. "
        "Actual property prices may differ. "
        "The model does not include rental yield, taxes, "
        "maintenance, financing costs or transaction costs."
    )

else:

    st.info(
        "Enter property details in the sidebar "
        "and click Analyze Investment."
    )
