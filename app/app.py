import os
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT_DIR = Path(__file__).resolve().parents[1]

def prepare_features(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "manufacture_year" in df.columns:
        if "Age of car" not in df.columns:
            df["Age of car"] = 2025 - df["manufacture_year"]

        if "Car_Age" not in df.columns:
            df["Car_Age"] = 2025 - df["manufacture_year"]

    return df

st.set_page_config(
    page_title="Indian Pre-Owned Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(ROOT_DIR / "models" / "best_model.pkl")

@st.cache_data
def load_data():
    return prepare_features(
        pd.read_csv(ROOT_DIR / "data" / "Cap_Training_Data_2025.csv")
    )

model = load_model()
train = load_data()
required_columns = list(model.feature_names_in_)

makers = sorted(train["Maker"].dropna().unique())
models = sorted(train["model"].dropna().unique())
locations = sorted(train["Location"].dropna().unique())
owners = sorted(train["Owner Type"].dropna().unique())
body_types = sorted(train["body_type"].dropna().unique())
transmissions = sorted(train["transmission"].dropna().unique())
fuel_types = sorted(train["fuel_type"].dropna().unique())

st.title("🚗 Indian Pre-Owned Car Price Prediction")
st.write("Estimate the market value of a used car using Machine Learning.")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a Page",
    [
        "Single Prediction",
        "Batch Prediction",
        "Dashboard"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("### Models Used")
st.sidebar.write("""
- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
""")

if page == "Single Prediction":

    st.header("Single Car Price Prediction")

    col1, col2 = st.columns(2)

    with col1:

        maker = st.selectbox(
            "Maker",
            makers
        )

        model_name = st.selectbox(
            "Model",
            models
        )

        location = st.selectbox(
            "Location",
            locations
        )

        distance = st.number_input(
            "Distance (KM)",
            min_value=0,
            value=50000,
            step=1000
        )

        owner = st.selectbox(
            "Owner Type",
            owners
        )

        manufacture_year = st.number_input(
            "Manufacture Year",
            min_value=1990,
            max_value=2025,
            value=2020
        )

    with col2:

        engine_displacement = st.number_input(
            "Engine Displacement (CC)",
            min_value=500,
            max_value=7000,
            value=1200
        )

        engine_power = st.number_input(
            "Engine Power",
            min_value=20,
            max_value=1000,
            value=100
        )

        body_type = st.selectbox(
            "Body Type",
            body_types
        )

        rating = st.slider(
            "Vroom Audit Rating",
            1,
            5,
            3
        )

        transmission = st.selectbox(
            "Transmission",
            transmissions
        )

        door_count = st.selectbox(
            "Door Count",
            [2,3,4,5,6]
        )

        seat_count = st.selectbox(
            "Seat Count",
            [2,4,5,6,7,8]
        )

        fuel_type = st.selectbox(
            "Fuel Type",
            fuel_types
        )

    age = 2025 - manufacture_year

    if st.button(
        "Predict Price",
        use_container_width=True
    ):

        input_df = pd.DataFrame({

            "Maker":[maker],
            "model":[model_name],
            "Location":[location],
            "Distance":[distance],
            "Owner Type":[owner],
            "manufacture_year":[manufacture_year],
            "Age of car":[age],
            "Car_Age":[age],
            "engine_displacement":[engine_displacement],
            "engine_power":[engine_power],
            "body_type":[body_type],
            "Vroom Audit Rating":[rating],
            "transmission":[transmission],
            "door_count":[door_count],
            "seat_count":[seat_count],
            "fuel_type":[fuel_type]

        })

        prediction = model.predict(input_df)[0]

        st.markdown("---")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Estimated Price",
                f"₹ {prediction:,.0f}"
            )

        with c2:

            lower = prediction * 0.95
            upper = prediction * 1.05

            st.metric(
                "Fair Price Range",
                f"₹ {lower:,.0f} - ₹ {upper:,.0f}"
            )

        st.success("Prediction Completed Successfully!")

        st.dataframe(input_df)

elif page == "Batch Prediction":

    st.header("Batch Prediction")

    st.write(
        "Upload a CSV file to predict prices for multiple cars."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            batch_df = prepare_features(pd.read_csv(uploaded_file))

            st.subheader("Uploaded Data")

            st.dataframe(batch_df.head())

            predict_df = batch_df.copy()

            ids = None

            if "ID" in predict_df.columns:
                ids = predict_df["ID"]
                predict_df = predict_df.drop(columns=["ID"])

            missing_columns = [
                column
                for column in required_columns
                if column not in predict_df.columns
            ]

            if missing_columns:
                st.error(
                    "Uploaded CSV is missing the columns required for prediction."
                )
                st.write("Please upload a CSV like `data/Cap_Test_Data_2025.csv`.")
                st.write("Missing columns:")
                st.dataframe(
                    pd.DataFrame({"Column": missing_columns}),
                    use_container_width=True
                )
                st.stop()

            predict_df = predict_df[required_columns]

            predictions = model.predict(predict_df)

            if ids is not None:

                output = pd.DataFrame({
                    "ID": ids,
                    "Price": predictions
                })

            else:

                output = batch_df.copy()
                output["Predicted Price"] = predictions

            st.subheader("Prediction Result")

            st.dataframe(output.head(20))

            csv = output.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction CSV",
                data=csv,
                file_name="submission.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.success("Prediction completed successfully.")

        except Exception as e:

            st.error("Unable to process the uploaded file.")

            st.exception(e)

elif page == "Dashboard":

    st.header("Dashboard")

    st.write("Explore the training dataset.")

    st.markdown("## Dataset Summary")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Rows", train.shape[0])

    with c2:
        st.metric("Columns", train.shape[1])

    with c3:
        st.metric(
            "Average Price",
            f"₹ {train['Price'].mean():,.0f}"
        )

    with c4:
        st.metric(
            "Maximum Price",
            f"₹ {train['Price'].max():,.0f}"
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            train,
            x="Price",
            nbins=40,
            title="Price Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.scatter(
            train,
            x="Distance",
            y="Price",
            title="Distance vs Price"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:

        age_column = None

        if "Age of car" in train.columns:
            age_column = "Age of car"

        elif "Age" in train.columns:
            age_column = "Age"

        if age_column is not None:

            fig = px.scatter(
                train,
                x=age_column,
                y="Price",
                title="Age vs Price"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col4:

        avg_price = (
            train.groupby("Maker")["Price"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            avg_price,
            x="Maker",
            y="Price",
            title="Top 10 Makers by Average Price"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    col5, col6 = st.columns(2)

    with col5:

        fuel_price = (
            train.groupby("fuel_type")["Price"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            fuel_price,
            x="fuel_type",
            y="Price",
            color="fuel_type",
            title="Average Price by Fuel Type"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col6:

        location_count = (
            train["Location"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        location_count.columns = [
            "Location",
            "Count"
        ]

        fig = px.bar(
            location_count,
            x="Location",
            y="Count",
            color="Count",
            title="Top 10 Locations"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(train.head(20), use_container_width=True)

    st.success("Dashboard Loaded Successfully")
