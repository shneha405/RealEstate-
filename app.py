
import streamlit as st
import pandas as pd
import joblib
import json
import os

# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide"
)

# --------------------------------
# 2. LOAD MACHINE LEARNING MODELS
# --------------------------------

MODEL_DIR = "real_estate_models"

try:
    classifier = joblib.load(
        os.path.join(
            MODEL_DIR,
            "classification_model.pkl"
        )
    )

    regressor = joblib.load(
        os.path.join(
            MODEL_DIR,
            "regression_model.pkl"
        )
    )

    with open(
        os.path.join(MODEL_DIR, "metadata.json"),
        "r"
    ) as f:
        metadata = json.load(f)

    features = metadata["features"]
    CURRENT_YEAR = metadata["year_reference"]

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# --------------------------------
# 3. APPLICATION HEADER
# --------------------------------

st.title("🏡 Real Estate Investment Advisor")

st.subheader(
    "Machine Learning Based Property Investment Analysis"
)

st.write(
    "Analyze property investment potential, "
    "predict future property prices and estimate ROI."
)

st.divider()


# --------------------------------
# 4. SIDEBAR - USER INPUTS
# --------------------------------

st.sidebar.header("🏠 Enter Property Details")

area = st.sidebar.number_input(
    "Property Area (sq. ft.)",
    min_value=100,
    value=2500
)

bedrooms = st.sidebar.number_input(
    "Number of Bedrooms",
    min_value=1,
    value=3
)

bathrooms = st.sidebar.number_input(
    "Number of Bathrooms",
    min_value=1,
    value=2
)

floors = st.sidebar.number_input(
    "Number of Floors",
    min_value=1,
    value=2
)

year_built = st.sidebar.number_input(
    "Year Built",
    min_value=1900,
    max_value=int(CURRENT_YEAR),
    value=min(2018, int(CURRENT_YEAR))
)

location_score = st.sidebar.slider(
    "Location Score",
    min_value=0,
    max_value=100,
    value=80
)

school_score = st.sidebar.slider(
    "School Score",
    min_value=0,
    max_value=100,
    value=75
)

current_price = st.sidebar.number_input(
    "Current Property Price (₹)",
    min_value=10000,
    value=650000
)


# --------------------------------
# 5. INVESTMENT ANALYSIS
# --------------------------------

if st.sidebar.button(
    "Analyze Investment",
    type="primary",
    use_container_width=True
):

    try:

        # Feature Engineering

        property_age = max(
            0,
            int(CURRENT_YEAR) - year_built
        )

        price_per_sqft = current_price / area

        # Prepare Input Data

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

        # Match training feature order

        input_data = input_data[features]

        # --------------------------------
        # 6. CLASSIFICATION PREDICTION
        # --------------------------------

        investment_prediction = classifier.predict(
            input_data
        )[0]

        probabilities = classifier.predict_proba(
            input_data
        )[0]

        # Confirm the class label before reading probability

        classes = list(classifier.classes_)

        if 1 not in classes:
            raise ValueError(
                "Good Investment class (1) not found "
                "in classifier classes."
            )

        good_index = classes.index(1)

        investment_probability = probabilities[good_index]

        # --------------------------------
        # 7. REGRESSION PREDICTION
        # --------------------------------

        future_price = float(
            regressor.predict(input_data)[0]
        )

        # --------------------------------
        # 8. ROI CALCULATION
        # --------------------------------

        appreciation = future_price - current_price

        roi = (
            appreciation / current_price
        ) * 100

        # --------------------------------
        # 9. DISPLAY RESULTS
        # --------------------------------

        st.success("Investment Analysis Completed!")

        st.header("📊 Investment Dashboard")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Current Property Price",
                f"₹{current_price:,.0f}"
            )

        with col2:

            st.metric(
                "Estimated 5-Year Price",
                f"₹{future_price:,.0f}",
                delta=f"₹{appreciation:,.0f}"
            )

        with col3:

            st.metric(
                "Estimated 5-Year ROI",
                f"{roi:.2f}%"
            )

        st.divider()

        # --------------------------------
        # 10. INVESTMENT CLASSIFICATION
        # --------------------------------

        st.header("🏆 Investment Classification")

        if investment_prediction == 1:

            st.success("GOOD INVESTMENT")

        else:

            st.warning("NOT A GOOD INVESTMENT")

        st.write(
            "Good Investment Probability:"
        )

        st.progress(
            float(investment_probability)
        )

        st.write(
            f"{investment_probability * 100:.2f}%"
        )

        st.divider()

        # --------------------------------
        # 11. PRICE FORECAST
        # --------------------------------

        st.header("📈 Five-Year Price Forecast")

        forecast_data = pd.DataFrame({

            "Category": [
                "Current Price",
                "Predicted Future Price"
            ],

            "Price (₹)": [
                current_price,
                future_price
            ]

        })

        st.bar_chart(
            forecast_data.set_index("Category")
        )

        st.write(
            f"Expected Appreciation: ₹{appreciation:,.2f}"
        )

        st.write(
            f"Estimated ROI: {roi:.2f}%"
        )

        st.divider()

        # --------------------------------
        # 12. PROPERTY SUMMARY
        # --------------------------------

        st.header("🏠 Property Summary")

        summary = pd.DataFrame({

            "Parameter": [
                "Property Area",
                "Bedrooms",
                "Bathrooms",
                "Floors",
                "Year Built",
                "Property Age",
                "Location Score",
                "School Score",
                "Price Per Sq. Ft."
            ],

            "Value": [
                f"{area} sq. ft.",
                bedrooms,
                bathrooms,
                floors,
                year_built,
                property_age,
                location_score,
                school_score,
                f"₹{price_per_sqft:,.2f}"
            ]

        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------
        # 13. DOWNLOAD REPORT
        # --------------------------------

        report = pd.DataFrame({

            "Metric": [
                "Current Price",
                "Predicted Five-Year Price",
                "Expected Appreciation",
                "Estimated ROI",
                "Investment Classification",
                "Good Investment Probability"
            ],

            "Value": [
                current_price,
                future_price,
                appreciation,
                f"{roi:.2f}%",
                str(investment_prediction),
                f"{investment_probability * 100:.2f}%"
            ]

        })

        csv = report.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Investment Report",
            data=csv,
            file_name="real_estate_investment_report.csv",
            mime="text/csv"
        )

        # --------------------------------
        # 14. DISCLAIMER
        # --------------------------------

        st.caption(
            "Disclaimer: This application provides educational "
            "machine learning predictions. Actual property prices "
            "may vary. ROI excludes rental income, taxes, "
            "maintenance, financing and transaction costs."
        )

    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )

else:

    st.info(
        "Enter property details in the sidebar "
        "and click Analyze Investment to generate predictions."
    )
